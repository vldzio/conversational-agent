"""VoiceFlow order logic for the web demo.

Mirrors ``model.py``: an LLM extracts ``{product_name, order_qty}`` from the
caller's sentence (``parse_query``), then plain Python checks the inventory
(``check_inventory``) - the LLM never decides stock or confirms orders.
Differences from the local script: inventory lives in memory per demo session
(the shared CSV is never modified) and an LLM call is used to turn the result
into one friendly spoken sentence, as ``proompt`` in model.py intended.
"""
from __future__ import annotations

import csv
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import llm

CSV_PATH = Path(__file__).resolve().parents[1] / "inventory.csv"
MAX_SESSIONS = 500
SESSION_TTL_S = 3600

PARSE_SCHEMA = {
    "type": "object",
    "properties": {"product_name": {"type": ["string", "null"]}, "order_qty": {"type": ["integer", "null"]}},
    "required": ["product_name", "order_qty"],
}


def load_rows(path: Path = CSV_PATH) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["Quantity Sold"] = int(r["Quantity Sold"])
        r["Quantity In Hand"] = int(r["Quantity In Hand"])
    return rows


@dataclass
class Session:
    rows: list[dict[str, Any]] = field(default_factory=load_rows)
    touched: float = field(default_factory=time.time)


_sessions: dict[str, Session] = {}
_lock = threading.Lock()


def get_session(session_id: str | None) -> tuple[str, Session]:
    with _lock:
        now = time.time()
        for k in [k for k, v in _sessions.items() if now - v.touched > SESSION_TTL_S]:
            _sessions.pop(k, None)
        if len(_sessions) >= MAX_SESSIONS and (not session_id or session_id not in _sessions):
            oldest = min(_sessions, key=lambda k: _sessions[k].touched)
            _sessions.pop(oldest, None)
        sid = session_id if session_id and session_id in _sessions else uuid.uuid4().hex
        sess = _sessions.setdefault(sid, Session())
        sess.touched = now
        return sid, sess


def product_names(rows: list[dict[str, Any]]) -> list[str]:
    return [r["Product Name"].lower() for r in rows]


def parse_query(query: str, names: list[str]) -> tuple[str | None, int | None]:
    system = (
        "Extract the product and quantity from a customer's spoken request, choosing the product ONLY from the given list. "
        "If the request does not clearly name a product from the list, or gives no quantity, use null. "
        "A spoken number word (e.g. 'two') must be returned as an integer."
    )
    user = f"Product list: {', '.join(names)}\n\nCustomer: {query}\n\nReturn JSON {{\"product_name\": string|null, \"order_qty\": integer|null}}."
    out = llm.chat_json(system, user, schema=PARSE_SCHEMA, temperature=0.0, max_tokens=120, metadata={"step": "parse_query"})
    name, qty = out.get("product_name"), out.get("order_qty")
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        qty = None
    return (name.lower().strip() if isinstance(name, str) and name.strip() else None), qty


def check_inventory(rows: list[dict[str, Any]], product_name: str, order_qty: int) -> dict[str, Any]:
    """Same decision rules as model.check_inventory, returning a structured result."""
    key = product_name.lower()
    row = next((r for r in rows if r["Product Name"].lower() == key), None)
    if row is None:
        return {"status": "unknown_product", "message": f"Sorry, we do not have {key} in our inventory."}
    available = row["Quantity In Hand"]
    if available <= 0:
        return {"status": "out_of_stock", "message": f"{key} is out of stock. Would you like to check similar products?"}
    if available < order_qty:
        return {
            "status": "insufficient",
            "message": f"We only have {available} of {key} in stock. Would you like to adjust your order?",
            "available": available,
        }
    row["Quantity In Hand"] -= order_qty
    row["Quantity Sold"] += order_qty
    return {"status": "confirmed", "message": f"Order confirmed! Your {order_qty}x {key} will be processed.", "remaining": row["Quantity In Hand"]}


def spoken_reply(result_message: str) -> str:
    """One natural spoken sentence conveying exactly the result (model.py's `proompt` idea)."""
    system = (
        "You are a friendly phone sales agent. Rewrite the line as ONE short, natural spoken sentence. "
        "Keep every number and product name exactly as given. Do not add facts, offers or prices."
    )
    return llm.text(
        [{"role": "system", "content": system}, {"role": "user", "content": result_message}],
        temperature=0.3, max_tokens=80, metadata={"step": "spoken_reply"},
    ).split("\n")[0].strip() or result_message


def handle_order(session_id: str | None, utterance: str) -> dict[str, Any]:
    sid, sess = get_session(session_id)
    names = product_names(sess.rows)
    product, qty = parse_query(utterance, names)
    if not product or not qty:
        result = {"status": "not_understood", "message": "Sorry, I couldn't understand your order. Could you please specify the product and quantity?"}
    elif product not in names:
        result = {"status": "unknown_product", "message": f"Sorry, we do not have {product} in our inventory."}
    else:
        result = check_inventory(sess.rows, product, qty)
    try:
        spoken = spoken_reply(result["message"])
    except llm.LLMUnavailable:
        spoken = result["message"]  # the order result is authoritative; the rewrite is cosmetic
    return {
        "sessionId": sid,
        "parsed": {"product": product, "quantity": qty},
        "result": result,
        "spoken": spoken,
        "inventory": [{"id": r["Product ID"], "name": r["Product Name"], "inHand": r["Quantity In Hand"], "sold": r["Quantity Sold"]} for r in sess.rows],
    }


def inventory_view(session_id: str | None) -> dict[str, Any]:
    sid, sess = get_session(session_id)
    return {
        "sessionId": sid,
        "inventory": [{"id": r["Product ID"], "name": r["Product Name"], "inHand": r["Quantity In Hand"], "sold": r["Quantity Sold"]} for r in sess.rows],
    }
