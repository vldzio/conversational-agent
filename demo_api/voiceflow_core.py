"""VoiceFlow order logic for the web demo.

Mirrors ``model.py``: an LLM extracts ``{product_name, order_qty}`` from the
caller's sentence (``parse_query``), then plain Python checks the inventory
(``check_inventory``) - the LLM never decides stock or confirms orders.
Differences from the local script: inventory lives in memory per demo session
(the shared CSV is never modified) and an LLM call is used to turn the result
into one friendly spoken sentence, as ``proompt`` in model.py intended.

Conversation: an incomplete or ambiguous order is not a dead end. The agent asks
the one question that is missing (which product / how many) and the session
remembers what it is waiting for, so "three" or "the bluetooth one" completes the
order on the next turn. All of this is plain Python: the LLM only reads the
sentence, it never decides what is asked, in stock, or confirmed.

Latency: a turn is produced as a stream of events (``turn_events``) so the
caller sees the decision before the spoken rewrite exists, and a turn can
be *prepared* while the customer is still talking (``prepare``): parse, a
dry-run decision and the rewrite are done ahead of time and reused only if
the real decision then gives exactly the same result.
"""
from __future__ import annotations

import csv
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import llm

CSV_PATH = Path(__file__).resolve().parents[1] / "inventory.csv"
MAX_SESSIONS = 500
SESSION_TTL_S = 3600
PENDING_TTL_S = 180  # how long the agent remembers what it asked
PREPARED_TTL_S = 30
PREPARED_MAX = 8
PREPARE_WAIT_S = 2.0  # how long a turn waits for an in-flight preparation of the same sentence
MAX_CHOICES = 4  # more matches than this is "too vague", not "pick one"

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {"type": ["string", "null"]},
        "order_qty": {"type": ["integer", "null"]},
        "candidates": {"type": ["array", "null"], "items": {"type": "string"}},
    },
    "required": ["product_name", "order_qty"],
}

GREETING = "Hello! Thanks for calling. What would you like to order today?"
MAX_SPOKEN = 30  # lines remembered per session as "allowed to be voiced"

_CANCEL = re.compile(r"\bcancel\w*", re.IGNORECASE)
CANCEL_MESSAGE = "Sorry, I cannot process order cancellations, please reach out to the support team"

_STOP = {"to", "of", "a", "an", "the", "and", "for", "c", "in", "with"}


def load_rows(path: Path = CSV_PATH) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["Quantity Sold"] = int(r["Quantity Sold"])
        r["Quantity In Hand"] = int(r["Quantity In Hand"])
    return rows


@dataclass
class Pending:
    """What the agent asked last, so the next sentence can answer it."""

    product: str | None = None  # waiting for a quantity of this product
    qty: int | None = None  # waiting for a product to go with this quantity
    candidates: tuple[str, ...] = ()  # waiting for a choice between these
    at: float = field(default_factory=time.time)

    def key(self) -> tuple:
        return (self.product, self.qty, self.candidates)


@dataclass
class Parsed:
    product: str | None
    qty: int | None
    candidates: tuple[str, ...]


@dataclass
class Prepared:
    parsed: Parsed
    ctx: tuple  # the pending context it was parsed under; a different context discards it
    result: dict[str, Any]
    spoken: str
    at: float


@dataclass
class Session:
    rows: list[dict[str, Any]] = field(default_factory=load_rows)
    touched: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)  # guards rows / pending_ask / prepared / pending
    pending_ask: Pending | None = None
    spoken: dict[str, None] = field(default_factory=lambda: {GREETING: None})  # lines the agent said (ordered set)
    prepared: dict[str, Prepared] = field(default_factory=dict)
    pending: dict[str, threading.Event] = field(default_factory=dict)  # preparations in flight


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


def inventory_list(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": r["Product ID"], "name": r["Product Name"], "inHand": r["Quantity In Hand"], "sold": r["Quantity Sold"]} for r in rows]


def is_cancellation(utterance: str) -> bool:
    """Cancelling is not something this agent can do. Detected in plain code, before any LLM call."""
    return bool(_CANCEL.search(utterance))


def cancellation_result() -> dict[str, Any]:
    email = os.environ.get("SUPPORT_EMAIL", "").strip()
    out: dict[str, Any] = {"status": "cancellation_unsupported", "message": CANCEL_MESSAGE + (f" at {email}." if email else ".")}
    if email:
        out["supportEmail"] = email
    return out


def normalise(utterance: str) -> str:
    """Key for matching a prepared sentence with the final one (case, punctuation and spacing ignored)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", utterance.lower())).strip()


# ---------------------------------------------------------------- understanding (LLM reads, Python validates)

def _tokens(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def match_products(text: str, names: list[str]) -> list[str]:
    """Products whose name shares the most words with ``text`` ("speakers" -> bluetooth speaker)."""
    q = _tokens(text)
    scored = [(len(q & _tokens(n)), n) for n in names]
    best = max((s for s, _ in scored), default=0)
    return [n for s, n in scored if s == best and s > 0]


def _ctx_text(p: Pending | None) -> str:
    if not p:
        return ""
    if p.candidates:
        return f"\nThe agent just asked which of these the customer wants: {', '.join(p.candidates)}."
    if p.product:
        return f"\nThe agent just asked how many {p.product} the customer wants."
    if p.qty:
        return f"\nThe agent just asked which product the customer wants {p.qty} of."
    return ""


def parse_query(query: str, names: list[str], pending: Pending | None = None) -> Parsed:
    """Read a sentence into (product, quantity, ambiguous candidates). Every value is validated here."""
    system = (
        "Extract the product and quantity from a customer's spoken request. Choose the product ONLY from the given list. "
        "If exactly one product fits (even if said loosely, like 'speakers' for 'bluetooth speaker'), return its exact list name. "
        "If several products fit, set product_name to null and list them in candidates. "
        "Use null for anything not stated. A spoken number word (e.g. 'two') must be an integer."
    )
    user = (
        f"Product list: {', '.join(names)}{_ctx_text(pending)}\n\nCustomer: {query}\n\n"
        'Return JSON {"product_name": string|null, "order_qty": integer|null, "candidates": string[]}.'
    )
    out = llm.chat_json(system, user, schema=PARSE_SCHEMA, schema_in_prompt=False, temperature=0.0, max_tokens=70, metadata={"step": "parse_query"})
    name, qty = out.get("product_name"), out.get("order_qty")
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        qty = None
    product = name.lower().strip() if isinstance(name, str) and name.strip() else None
    cands = [c.lower().strip() for c in (out.get("candidates") or []) if isinstance(c, str)]
    cands = tuple(dict.fromkeys(c for c in cands if c in names))

    if product not in names:  # nothing usable from the model: fall back to word matching, in Python
        found = match_products(product, names) if product else []
        if not found:
            found = match_products(query, names)
        if len(found) == 1:
            product, cands = found[0], ()
        elif 1 < len(found) <= MAX_CHOICES and (not cands or len(found) < len(cands)):
            product, cands = None, tuple(found)  # the sentence's own words narrow it down more than the model did
    if product in names:
        cands = ()
    elif len(cands) == 1:
        product, cands = cands[0], ()
    elif len(cands) > MAX_CHOICES:
        cands = ()
    return Parsed(product, qty, cands)


def merge_pending(parsed: Parsed, pending: Pending | None) -> Parsed:
    """Complete a sentence with what the agent asked about last turn."""
    if not pending:
        return parsed
    product, qty, cands = parsed.product, parsed.qty, parsed.candidates
    if product is None and not cands:
        product = pending.product
    if qty is None:
        qty = pending.qty
    return Parsed(product, qty, cands)


# ---------------------------------------------------------------- the decision (plain Python, no LLM)

def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _examples(rows: list[dict[str, Any]], n: int = 3) -> str:
    top = sorted(rows, key=lambda r: -r["Quantity Sold"])[:n]
    return _join([re.sub(r"\s*\(.*?\)", "", r["Product Name"]).lower() for r in top])


def check_inventory(rows: list[dict[str, Any]], product_name: str, order_qty: int, *, commit: bool = True) -> dict[str, Any]:
    """Same decision rules as model.check_inventory, returning a structured result.

    ``commit=False`` is a dry run: the same decision, but the stock is left untouched.
    """
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
    if commit:
        row["Quantity In Hand"] -= order_qty
        row["Quantity Sold"] += order_qty
    return {"status": "confirmed", "message": f"Order confirmed! Your {order_qty}x {key} will be processed.", "remaining": available - order_qty}


def decide(rows: list[dict[str, Any]], p: Parsed, *, commit: bool) -> dict[str, Any]:
    """The deterministic outcome for a parsed request. Nothing here involves the LLM.

    Beyond the order results of ``check_inventory`` it can ask for what is missing:
    ``ambiguous_product``, ``need_quantity``, ``need_product``.
    """
    if p.candidates and not p.product:
        return {"status": "ambiguous_product", "message": f"We have {_join(list(p.candidates))}. Which one would you like?"}
    if p.product and not p.qty:
        if p.product not in product_names(rows):
            return {"status": "unknown_product", "message": f"Sorry, we do not have {p.product} in our inventory. We have things like {_examples(rows)}."}
        return {"status": "need_quantity", "message": f"How many {p.product} would you like?"}
    if p.qty and not p.product:
        return {"status": "need_product", "message": f"Which product would you like {p.qty} of? We have things like {_examples(rows)}."}
    if not p.product or not p.qty:
        return {"status": "not_understood", "message": f"Sorry, I didn't catch what you'd like. We have things like {_examples(rows)}. What would you like, and how many?"}
    if p.product not in product_names(rows):
        return {"status": "unknown_product", "message": f"Sorry, we do not have {p.product} in our inventory. We have things like {_examples(rows)}."}
    return check_inventory(rows, p.product, p.qty, commit=commit)


def next_pending(p: Parsed, result: dict[str, Any]) -> Pending | None:
    """What the agent should remember after this decision (nothing once an order is settled)."""
    s = result["status"]
    if s == "need_quantity":
        return Pending(product=p.product)
    if s == "need_product":
        return Pending(qty=p.qty)
    if s == "ambiguous_product":
        return Pending(qty=p.qty, candidates=p.candidates)
    return None


# ---------------------------------------------------------------- the spoken line (LLM rewrites, Python verifies)

_THINKING = re.compile(r"^\s*(we need|we should|we must|here'?s|here is|let me|let's|the user|the line|okay[,.]|so[,.]|first[,.]|rewrite)", re.IGNORECASE)


def faithful(rewrite: str, message: str, names: list[str] | None = None) -> bool:
    """A rewrite is only spoken if it is one short, finished sentence (not model "thinking") that keeps
    every number and every product named in the authoritative message."""
    if not rewrite or len(rewrite) > 280 or _THINKING.match(rewrite) or rewrite.rstrip()[-1] not in ".!?":
        return False
    flat, low = rewrite.replace(",", ""), rewrite.lower().replace(",", "")
    if not all(re.search(rf"(?<!\d){n}(?!\d)", flat) for n in re.findall(r"\d+", message.replace(",", ""))):
        return False
    msg_low = message.lower()
    for name in names or []:
        if name in msg_low and not all(t in low for t in re.findall(r"[a-z0-9]{3,}", name.replace(",", ""))):
            return False
    return True


def spoken_reply(result_message: str, names: list[str] | None = None) -> str:
    """One natural spoken line conveying exactly the result (model.py's `proompt` idea)."""
    system = (
        "You are a friendly phone sales agent. Rewrite the line as one or two short, natural spoken sentences. "
        "Keep every number and product name exactly as given. If the line asks the customer a question, keep it as a question. "
        "Do not add facts, offers or prices."
    )
    rewrite = llm.text(
        [{"role": "system", "content": system}, {"role": "user", "content": result_message}],
        temperature=0.0, max_tokens=80, metadata={"step": "spoken_reply"},
    ).split("\n")[0].strip()
    return rewrite if faithful(rewrite, result_message, names) else result_message


def _safe_spoken(message: str, names: list[str]) -> str:
    try:
        return spoken_reply(message, names)
    except llm.LLMUnavailable:
        return message  # the decision is authoritative; the rewrite is cosmetic


# ---------------------------------------------------------------- preparation and turns

def _live_pending(sess: Session) -> Pending | None:
    p = sess.pending_ask
    return p if p and time.time() - p.at <= PENDING_TTL_S else None


def remember_spoken(sess: Session, text: str) -> None:
    with sess.lock:
        sess.spoken.pop(text, None)
        sess.spoken[text] = None
        while len(sess.spoken) > MAX_SPOKEN:
            sess.spoken.pop(next(iter(sess.spoken)))


def may_speak(session_id: str | None, text: str) -> bool:
    """The voice endpoint only speaks lines this session's agent produced, never arbitrary text."""
    sess = _sessions.get(session_id or "")
    if not sess or not text:
        return False
    with sess.lock:
        return text in sess.spoken


def has_prepared(session_id: str | None, utterance: str) -> bool:
    """True if this sentence was (or is being) prepared for the session; used to price the turn."""
    if not session_id or session_id not in _sessions:
        return False
    sess, key = _sessions[session_id], normalise(utterance)
    with sess.lock:
        p = sess.prepared.get(key)
        return key in sess.pending or bool(p and time.time() - p.at <= PREPARED_TTL_S)


def _understand(sess: Session, utterance: str) -> tuple[Parsed, tuple]:
    with sess.lock:
        ask = _live_pending(sess)
        names = product_names(sess.rows)
    ctx = ask.key() if ask else ()
    return merge_pending(parse_query(utterance, names, ask), ask), ctx


def prepare(session_id: str | None, utterance: str) -> dict[str, Any]:
    """Do the slow work for a sentence ahead of time (nothing is committed). See ``turn_events``."""
    sid, sess = get_session(session_id)
    key = normalise(utterance)
    with sess.lock:
        if key in sess.pending or key in sess.prepared:
            return {"sessionId": sid, "prepared": True}
        done = sess.pending[key] = threading.Event()
    try:
        parsed, ctx = _understand(sess, utterance)
        with sess.lock:
            result = decide(sess.rows, parsed, commit=False)
            names = product_names(sess.rows)
        spoken = _safe_spoken(result["message"], names)
        with sess.lock:
            now = time.time()
            for k in [k for k, p in sess.prepared.items() if now - p.at > PREPARED_TTL_S]:
                sess.prepared.pop(k, None)
            while len(sess.prepared) >= PREPARED_MAX:
                sess.prepared.pop(min(sess.prepared, key=lambda k: sess.prepared[k].at))
            sess.prepared[key] = Prepared(parsed, ctx, result, spoken, now)
        remember_spoken(sess, spoken)  # so the page may ask for its audio ahead of time
        return {"sessionId": sid, "prepared": True, "spoken": spoken}
    finally:
        with sess.lock:
            sess.pending.pop(key, None)
        done.set()


def _take_prepared(sess: Session, key: str) -> Prepared | None:
    with sess.lock:
        waiting = sess.pending.get(key)
    if waiting:
        waiting.wait(PREPARE_WAIT_S)
    with sess.lock:
        p = sess.prepared.pop(key, None)
        ask = _live_pending(sess)
    if not p or time.time() - p.at > PREPARED_TTL_S:
        return None
    return p if p.ctx == (ask.key() if ask else ()) else None  # parsed under a different conversation state


def turn_events(session_id: str | None, utterance: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """One turn as ``(event, data)`` pairs: ``parsed`` -> ``result`` -> ``say`` -> ``done``.

    ``result`` (the deterministic decision + inventory) is emitted before the spoken rewrite is
    requested. If the sentence was prepared, its rewrite is reused only when the real decision gives
    exactly the dry-run result; otherwise the rewrite is generated from the real result.
    """
    t0 = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t0) * 1000)  # noqa: E731
    timings: dict[str, Any] = {}
    sid, sess = get_session(session_id)
    if is_cancellation(utterance):  # fixed reply, no LLM: the same words every time
        with sess.lock:
            sess.pending_ask = None
            inventory = inventory_list(sess.rows)
        result = cancellation_result()
        yield "parsed", {"sessionId": sid, "parsed": {"product": None, "quantity": None}}
        yield "result", {"result": result, "inventory": inventory}
        remember_spoken(sess, result["message"])
        yield "say", {"spoken": result["message"]}
        yield "done", {"timings": {"result": ms(), "say": ms(), "prepared": False}}
        return
    prepared = _take_prepared(sess, normalise(utterance))
    parsed = prepared.parsed if prepared else _understand(sess, utterance)[0]
    timings["parse"] = ms()
    yield "parsed", {"sessionId": sid, "parsed": {"product": parsed.product, "quantity": parsed.qty}}

    with sess.lock:
        result = decide(sess.rows, parsed, commit=True)
        sess.pending_ask = next_pending(parsed, result)
        inventory = inventory_list(sess.rows)
        names = product_names(sess.rows)
    timings["result"] = ms()
    yield "result", {"result": result, "inventory": inventory}

    reuse = bool(prepared and prepared.result == result)
    spoken = prepared.spoken if reuse else _safe_spoken(result["message"], names)
    timings["say"] = ms()
    timings["prepared"] = reuse
    remember_spoken(sess, spoken)
    yield "say", {"spoken": spoken}
    yield "done", {"timings": timings}


def handle_order(session_id: str | None, utterance: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _, data in turn_events(session_id, utterance):
        out.update(data)
    return out


def inventory_view(session_id: str | None) -> dict[str, Any]:
    sid, sess = get_session(session_id)
    with sess.lock:
        return {"sessionId": sid, "inventory": inventory_list(sess.rows), "greeting": GREETING}
