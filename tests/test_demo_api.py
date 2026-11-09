"""VoiceFlow demo tests. The LLM is stubbed at the llm.chat_json / llm.text seam; inventory logic is real."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from demo_api import guard, llm, voiceflow_core as core  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(guard, "UPSTASH_URL", "")
    guard._mem.clear()
    core._sessions.clear()
    from demo_api.app import app

    return TestClient(app)


def stub(monkeypatch, parsed, spoken="Sure thing."):
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: parsed)
    monkeypatch.setattr(llm, "text", lambda *a, **k: spoken)


def order(client, text, sid=None):
    return client.post("/order", json={"utterance": text, "sessionId": sid})


def test_confirmed_order_updates_session_inventory(client, monkeypatch):
    stub(monkeypatch, {"product_name": "Wireless Charger", "order_qty": 2})
    r = order(client, "two wireless chargers").json()
    assert r["result"]["status"] == "confirmed"
    row = next(i for i in r["inventory"] if i["name"] == "Wireless Charger")
    assert row["inHand"] == 18 and row["sold"] == 42  # 20 -> 18, 40 -> 42
    assert r["spoken"] == "Sure thing."


def test_sessions_are_isolated_and_csv_untouched(client, monkeypatch):
    before = (ROOT / "inventory.csv").read_text(encoding="utf-8")
    stub(monkeypatch, {"product_name": "Wireless Charger", "order_qty": 5})
    a = order(client, "x").json()
    b = order(client, "x").json()  # new session
    assert a["sessionId"] != b["sessionId"]
    assert next(i for i in b["inventory"] if i["name"] == "Wireless Charger")["inHand"] == 15
    assert (ROOT / "inventory.csv").read_text(encoding="utf-8") == before


def test_insufficient_stock_does_not_change_inventory(client, monkeypatch):
    stub(monkeypatch, {"product_name": "VR Headset", "order_qty": 50})
    r = order(client, "fifty vr headsets").json()
    assert r["result"]["status"] == "insufficient" and r["result"]["available"] == 9
    assert next(i for i in r["inventory"] if i["name"] == "VR Headset")["inHand"] == 9


def test_unknown_product_and_unparsed(client, monkeypatch):
    stub(monkeypatch, {"product_name": "flux capacitor", "order_qty": 1})
    assert order(client, "a flux capacitor").json()["result"]["status"] == "unknown_product"
    stub(monkeypatch, {"product_name": None, "order_qty": None})
    assert order(client, "hello there").json()["result"]["status"] == "not_understood"


def test_bad_quantities_are_not_trusted(client, monkeypatch):
    for qty in (0, -3, True, "2", 2.5):
        stub(monkeypatch, {"product_name": "Wireless Charger", "order_qty": qty})
        assert order(client, "x").json()["result"]["status"] == "not_understood", qty


def test_spoken_rewrite_failure_falls_back_to_authoritative_message(client, monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"product_name": "Selfie Stick", "order_qty": 1})

    def boom(*a, **k):
        raise llm.LLMUnavailable("busy")

    monkeypatch.setattr(llm, "text", boom)
    r = order(client, "a selfie stick").json()
    assert r["spoken"] == r["result"]["message"] and r["result"]["status"] == "confirmed"


def test_empty_utterance_rejected_and_limits_apply(client, monkeypatch):
    assert client.post("/order", json={"utterance": "  "}).status_code == 400
    monkeypatch.setattr(guard, "PER_IP_DAILY", 2)
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    assert order(client, "x").status_code == 200
    assert order(client, "x").status_code == 429


def test_inventory_endpoint(client):
    r = client.post("/inventory", json={}).json()
    assert len(r["inventory"]) == 20 and r["sessionId"]
