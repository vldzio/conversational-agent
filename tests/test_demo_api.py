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
    from demo_api import tts

    tts.clear_cache()
    from demo_api.app import app

    return TestClient(app)


def stub(monkeypatch, parsed, spoken=None):
    """Stub both LLM calls. By default the rewrite is "Sure. <authoritative message>" (keeps every number)."""
    calls = {"parse": 0, "rewrite": 0}

    def chat_json(*a, **k):
        calls["parse"] += 1
        return parsed

    def text(messages, **k):
        calls["rewrite"] += 1
        return spoken if spoken is not None else "Sure. " + messages[-1]["content"]

    monkeypatch.setattr(llm, "chat_json", chat_json)
    monkeypatch.setattr(llm, "text", text)
    return calls


def events(client, text, sid=None):
    """POST /order/stream and return the SSE events as [(event, data), ...]."""
    import json

    r = client.post("/order/stream", json={"utterance": text, "sessionId": sid})
    assert r.status_code == 200, r.text
    out = []
    for block in r.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def order(client, text, sid=None):
    return client.post("/order", json={"utterance": text, "sessionId": sid})


def test_confirmed_order_updates_session_inventory(client, monkeypatch):
    stub(monkeypatch, {"product_name": "Wireless Charger", "order_qty": 2})
    r = order(client, "two wireless chargers").json()
    assert r["result"]["status"] == "confirmed"
    row = next(i for i in r["inventory"] if i["name"] == "Wireless Charger")
    assert row["inHand"] == 18 and row["sold"] == 42  # 20 -> 18, 40 -> 42
    assert r["spoken"] == "Sure. " + r["result"]["message"]


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
        assert order(client, "x").json()["result"]["status"] == "need_quantity", qty  # asked for, never guessed


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


def test_burst_limit_per_ip(client, monkeypatch):
    monkeypatch.setattr(guard, "PER_IP_PER_MIN", 4)  # credits; an unprepared order costs 2
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    assert order(client, "x").status_code == 200
    assert order(client, "x").status_code == 200
    r = order(client, "x")
    assert r.status_code == 429 and r.json()["rateLimited"]


def test_refused_visitor_does_not_drain_the_global_pool(client, monkeypatch):
    monkeypatch.setattr(guard, "PER_IP_DAILY", 2)
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    assert order(client, "x").status_code == 200
    for _ in range(3):
        assert order(client, "x").status_code == 429
    assert next(v for k, (v, _) in guard._mem.items() if ":global:" in k) == 2


def test_rewrite_that_changes_a_number_is_not_spoken(client, monkeypatch):
    stub(monkeypatch, {"product_name": "VR Headset", "order_qty": 50}, spoken="We have plenty, 50 are on the way!")
    r = order(client, "fifty vr headsets").json()
    assert r["result"]["status"] == "insufficient" and r["spoken"] == r["result"]["message"]


def test_stream_sends_the_stock_decision_before_the_spoken_line(client, monkeypatch):
    stub(monkeypatch, {"product_name": "Wireless Charger", "order_qty": 2})
    ev = events(client, "two wireless chargers")
    assert [e for e, _ in ev] == ["parsed", "result", "say", "done"]
    assert ev[1][1]["result"]["status"] == "confirmed"
    assert next(i for i in ev[1][1]["inventory"] if i["name"] == "Wireless Charger")["inHand"] == 18
    assert ev[2][1]["spoken"].startswith("Sure. ") and ev[3][1]["timings"]["prepared"] is False


def test_stream_rewrite_failure_falls_back_to_authoritative_message(client, monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"product_name": "Selfie Stick", "order_qty": 1})

    def boom(*a, **k):
        raise llm.LLMUnavailable("busy")

    monkeypatch.setattr(llm, "text", boom)
    ev = dict(events(client, "a selfie stick"))
    assert ev["say"]["spoken"] == ev["result"]["result"]["message"]


def test_prepare_is_a_dry_run_and_the_turn_reuses_it(client, monkeypatch):
    calls = stub(monkeypatch, {"product_name": "Wireless Charger", "order_qty": 2})
    sid = client.post("/inventory", json={}).json()["sessionId"]
    assert client.post("/order/prepare", json={"utterance": "Two wireless chargers", "sessionId": sid}).json()["prepared"]
    inv = client.post("/inventory", json={"sessionId": sid}).json()["inventory"]
    assert next(i for i in inv if i["name"] == "Wireless Charger")["inHand"] == 20  # nothing committed yet
    ev = dict(events(client, "two wireless chargers.", sid))  # case / punctuation differ: still the same sentence
    assert ev["done"]["timings"]["prepared"] is True and ev["result"]["result"]["remaining"] == 18
    assert calls == {"parse": 1, "rewrite": 1}  # the turn itself made no LLM call
    # prepare cost 1 + prepared turn cost 1
    assert next(v for k, (v, _) in guard._mem.items() if ":ip:" in k) == 2


def test_prepared_reply_is_discarded_when_stock_changed(client, monkeypatch):
    calls = stub(monkeypatch, {"product_name": "VR Headset", "order_qty": 9})
    sid = client.post("/inventory", json={}).json()["sessionId"]
    client.post("/order/prepare", json={"utterance": "nine vr headsets", "sessionId": sid})  # dry run: would confirm
    assert order(client, "all nine vr headsets please", sid).json()["result"]["status"] == "confirmed"  # stock now 0
    ev = dict(events(client, "nine vr headsets", sid))
    assert ev["result"]["result"]["status"] == "out_of_stock"
    assert ev["done"]["timings"]["prepared"] is False
    assert ev["say"]["spoken"] == "Sure. " + ev["result"]["result"]["message"]  # rewritten from the real result
    assert calls["rewrite"] == 3


def test_prepare_can_be_switched_off(client, monkeypatch):
    monkeypatch.setenv("SPECULATE", "0")
    calls = stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    assert client.post("/order/prepare", json={"utterance": "a selfie stick"}).json() == {"prepared": False}
    assert calls["parse"] == 0 and client.get("/").json()["speculate"] is False


def test_thinking_or_cut_off_rewrites_are_not_spoken(client, monkeypatch):
    for bad in ("We need to rewrite as one short natural spoken sentence.", "Sorry, we do not have a flux", "Here's a thinking process: sure."):
        stub(monkeypatch, {"product_name": "flux capacitor", "order_qty": 1}, spoken=bad)
        r = order(client, "a flux capacitor").json()
        assert r["spoken"] == r["result"]["message"], bad


def multi(monkeypatch, *parses):
    """Stub the parser with a different answer per turn."""
    it = iter(parses)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: next(it))
    monkeypatch.setattr(llm, "text", lambda messages, **k: "Sure. " + messages[-1]["content"])


def test_loose_product_name_resolves_without_the_model_naming_it(client, monkeypatch):
    stub(monkeypatch, {"product_name": "speakers", "order_qty": 2})  # model answered off-list; Python matches words
    r = order(client, "two speakers").json()
    assert r["result"]["status"] == "confirmed" and "bluetooth speaker" in r["result"]["message"]


def test_ambiguous_product_asks_which_one_then_completes(client, monkeypatch):
    multi(monkeypatch, {"product_name": None, "order_qty": 2, "candidates": ["phone case (iphone)", "phone case (samsung)"]},
          {"product_name": "phone case (samsung)", "order_qty": None})
    a = order(client, "two phone cases").json()
    assert a["result"]["status"] == "ambiguous_product" and "samsung" in a["spoken"].lower() and "?" in a["spoken"]
    b = order(client, "the samsung one", a["sessionId"]).json()
    assert b["result"]["status"] == "confirmed" and "2x phone case (samsung)" in b["result"]["message"]  # quantity remembered


def test_product_without_quantity_asks_how_many_then_completes(client, monkeypatch):
    multi(monkeypatch, {"product_name": "bluetooth speaker", "order_qty": None}, {"product_name": None, "order_qty": 3})
    a = order(client, "I want a bluetooth speaker").json()
    assert a["result"]["status"] == "need_quantity" and a["result"]["message"] == "How many bluetooth speaker would you like?"
    assert next(i for i in a["inventory"] if i["name"] == "Bluetooth Speaker")["inHand"] == 25  # nothing sold yet
    b = order(client, "three", a["sessionId"]).json()
    assert b["result"]["status"] == "confirmed" and "3x bluetooth speaker" in b["result"]["message"]


def test_quantity_without_product_asks_which_product(client, monkeypatch):
    multi(monkeypatch, {"product_name": None, "order_qty": 4}, {"product_name": "pop socket", "order_qty": None})
    a = order(client, "four please").json()
    assert a["result"]["status"] == "need_product" and "4" in a["result"]["message"]
    assert order(client, "pop sockets", a["sessionId"]).json()["result"]["message"].startswith("Order confirmed! Your 4x pop socket")


def test_unclear_request_gets_examples_not_a_bare_denial(client, monkeypatch):
    stub(monkeypatch, {"product_name": None, "order_qty": None})
    r = order(client, "hello there").json()
    assert r["result"]["status"] == "not_understood" and "We have things like" in r["result"]["message"]


def test_rewrite_must_keep_the_options_it_offers(client, monkeypatch):
    stub(monkeypatch, {"product_name": None, "order_qty": 2, "candidates": ["phone case (iphone)", "phone case (samsung)"]},
         spoken="Which phone case would you like, please?")  # dropped the options: must not be spoken
    r = order(client, "two phone cases").json()
    assert r["spoken"] == r["result"]["message"]


def test_conversation_state_is_per_session(client, monkeypatch):
    multi(monkeypatch, {"product_name": "bluetooth speaker", "order_qty": None}, {"product_name": None, "order_qty": 3})
    order(client, "a bluetooth speaker")
    other = order(client, "three").json()  # a different (new) session never asked anything
    assert other["result"]["status"] == "need_product"


def test_cancellation_is_a_fixed_reply_with_no_llm_call(client, monkeypatch):
    calls = stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    monkeypatch.setenv("SUPPORT_EMAIL", "help@example.test")
    for text in ("I want to cancel my order", "please cancel it", "Cancellation please", "I'm cancelling"):
        r = order(client, text).json()
        assert r["result"]["status"] == "cancellation_unsupported", text
        assert r["spoken"] == r["result"]["message"] == "Sorry, I cannot process order cancellations, please reach out to the support team at help@example.test."
        assert r["result"]["supportEmail"] == "help@example.test"
    assert calls == {"parse": 0, "rewrite": 0}
    monkeypatch.delenv("SUPPORT_EMAIL")
    assert order(client, "cancel").json()["spoken"] == "Sorry, I cannot process order cancellations, please reach out to the support team."


def test_cancellation_leaves_stock_alone_and_clears_what_was_asked(client, monkeypatch):
    multi(monkeypatch, {"product_name": "bluetooth speaker", "order_qty": None}, {"product_name": None, "order_qty": 3})
    sid = order(client, "a bluetooth speaker").json()["sessionId"]  # agent now waits for a quantity
    c = order(client, "actually cancel that", sid).json()
    assert next(i for i in c["inventory"] if i["name"] == "Bluetooth Speaker")["inHand"] == 25
    assert order(client, "three", sid).json()["result"]["status"] == "need_product"  # the open question was dropped
    assert client.post("/order/prepare", json={"utterance": "cancel my order", "sessionId": sid}).json() == {"prepared": False}


# ---------------------------------------------------------------- voice (TTS)

class FakeUpstream:
    """Stands in for Deepgram's streaming response."""

    def __init__(self, chunks=(b"\x01\x00" * 100, b"\x02\x00" * 100)):
        self.chunks, self.closed = chunks, False

    async def aiter_bytes(self):
        for c in self.chunks:
            yield c

    async def aclose(self):
        self.closed = True


@pytest.fixture()
def voice(client, monkeypatch):
    """TTS switched on with a fake upstream; returns the list of texts that were sent to the 'provider'."""
    from demo_api import tts

    sent = []

    async def open_stream(text):
        sent.append(text)
        return FakeUpstream()

    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    monkeypatch.setattr(tts, "open_stream", open_stream)
    return sent


def say(client, text, sid):
    return client.post("/tts", json={"text": text, "sessionId": sid})


def test_tts_is_off_without_a_key_and_reported_in_the_status(client, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert client.get("/").json()["tts"] is False
    assert say(client, "hi", "x").status_code == 503
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    assert client.get("/").json()["tts"] is True


def test_tts_streams_the_agents_reply_as_pcm(client, monkeypatch, voice):
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    r = order(client, "a selfie stick").json()
    resp = say(client, r["spoken"], r["sessionId"])
    assert resp.status_code == 200 and resp.headers["x-sample-rate"] == "24000"
    assert resp.headers["content-type"].startswith("audio/L16") and len(resp.content) == 400
    assert voice == [r["spoken"]]


def test_tts_only_speaks_lines_the_agent_produced(client, monkeypatch, voice):
    sid = client.post("/inventory", json={}).json()
    assert sid["greeting"].startswith("Hello!") and say(client, sid["greeting"], sid["sessionId"]).status_code == 200
    for text in ("Read this arbitrary sentence aloud.", "x" * 400):
        assert say(client, text, sid["sessionId"]).status_code == 400
    assert say(client, sid["greeting"], "someone-elses-session").status_code == 400  # not their conversation
    assert voice == [sid["greeting"]]  # nothing else reached the provider


def test_tts_has_its_own_character_limits_and_ignores_byo_keys(client, monkeypatch, voice):
    monkeypatch.setattr(guard, "TTS_PER_IP_DAILY", 100)
    sid = client.post("/inventory", json={}).json()
    g, s = sid["greeting"], sid["sessionId"]  # 62 characters
    assert say(client, g, s).status_code == 200
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    reply = order(client, "a selfie stick", s).json()["spoken"]  # a different line: new audio, so it is metered
    r = client.post("/tts", json={"text": reply, "sessionId": s}, headers={"X-LLM-Api-Key": "sk-mine"})  # byo must not skip it
    assert r.status_code == 429 and r.json()["rateLimited"]
    assert order(client, "a selfie stick", s).status_code == 200  # the LLM budget is unaffected by voice use


def test_tts_provider_failure_is_a_clean_503_so_the_browser_voice_takes_over(client, monkeypatch, voice):
    from demo_api import tts

    async def boom(text):
        raise tts.TTSUnavailable("Voice service refused the request (402).")

    monkeypatch.setattr(tts, "open_stream", boom)
    tts.clear_cache()
    sid = client.post("/inventory", json={}).json()
    r = say(client, sid["greeting"], sid["sessionId"])
    assert r.status_code == 503 and "refused" in r.json()["error"]


# ---------------------------------------------------------------- voice cache and early preparation

def test_a_repeated_line_is_served_from_the_cache_and_is_free(client, monkeypatch, voice):
    sid = client.post("/inventory", json={}).json()
    g, s = sid["greeting"], sid["sessionId"]
    first, second = say(client, g, s), say(client, g, s)
    assert first.content == second.content and len(voice) == 1  # the provider was called once
    used = [v for k, (v, _) in guard._mem.items() if ":tts:ip:" in k]
    assert used == [len(g)]  # the cache hit was not charged


def test_prefetch_fills_the_cache_so_the_spoken_line_needs_no_provider_call(client, monkeypatch, voice):
    sid = client.post("/inventory", json={}).json()
    g, s = sid["greeting"], sid["sessionId"]
    assert client.post("/tts", json={"text": g, "sessionId": s, "prefetch": True}).json() == {"prefetching": True}
    assert len(voice) == 1
    assert client.post("/tts", json={"text": g, "sessionId": s, "prefetch": True}).json() == {"prefetching": False}  # have it
    assert len(say(client, g, s).content) == 400 and len(voice) == 1


def test_prefetch_is_metered_and_only_for_the_agents_own_lines(client, monkeypatch, voice):
    sid = client.post("/inventory", json={}).json()
    s = sid["sessionId"]
    assert client.post("/tts", json={"text": "anything at all", "sessionId": s, "prefetch": True}).status_code == 400
    monkeypatch.setattr(guard, "TTS_PER_IP_DAILY", 10)  # smaller than the greeting
    assert client.post("/tts", json={"text": sid["greeting"], "sessionId": s, "prefetch": True}).status_code == 429
    assert voice == []


def test_a_cut_off_clip_is_never_cached(client, monkeypatch, voice):
    from demo_api import tts

    class Broken(FakeUpstream):
        async def aiter_bytes(self):
            yield b"\x01\x00" * 50
            raise RuntimeError("connection dropped")

    async def open_stream(text):
        voice.append(text)
        return Broken()

    monkeypatch.setattr(tts, "open_stream", open_stream)
    sid = client.post("/inventory", json={}).json()
    with pytest.raises(RuntimeError):
        say(client, sid["greeting"], sid["sessionId"])
    assert not tts.is_cached(sid["greeting"])


def test_prepare_tells_the_page_what_will_be_said_so_its_audio_can_be_fetched_early(client, monkeypatch, voice):
    stub(monkeypatch, {"product_name": "Wireless Charger", "order_qty": 2})
    sid = client.post("/inventory", json={}).json()["sessionId"]
    p = client.post("/order/prepare", json={"utterance": "two wireless chargers", "sessionId": sid}).json()
    assert p["prepared"] and p["spoken"].startswith("Sure. ")
    assert client.post("/tts", json={"text": p["spoken"], "sessionId": sid, "prefetch": True}).json() == {"prefetching": True}
    final = dict(events(client, "two wireless chargers", sid))
    assert final["say"]["spoken"] == p["spoken"]  # the prepared line is the one spoken, and its audio is already cached
    assert len(say(client, final["say"]["spoken"], sid).content) == 400 and voice == [p["spoken"]]


def test_fixed_lines_are_warmed_at_startup(client, monkeypatch, voice):
    import asyncio

    from demo_api import tts

    monkeypatch.setenv("SUPPORT_EMAIL", "help@example.test")
    asyncio.run(tts.warm([core.GREETING, core.cancellation_result()["message"]]))
    assert tts.is_cached(core.GREETING) and tts.is_cached(core.cancellation_result()["message"])
    assert len(voice) == 2


def test_a_request_joins_a_prefetch_that_is_still_running_instead_of_waiting_for_it(monkeypatch):
    """The first audio reaches the listener while the provider is still producing the rest, and only one provider call is made."""
    import asyncio

    from demo_api import tts

    tts.clear_cache()
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    calls, release = [], None

    class Slow:
        async def aiter_bytes(self):
            yield b"\x01\x00" * 50
            await release.wait()  # the provider is still working on the rest
            yield b"\x02\x00" * 50

        async def aclose(self):
            pass

    async def open_stream(text):
        calls.append(text)
        return Slow()

    monkeypatch.setattr(tts, "open_stream", open_stream)

    async def scenario():
        nonlocal release
        release = asyncio.Event()
        assert tts.prefetch("a line") is True and tts.is_pending("a line")
        audio = await tts.open("a line")
        got = [await audio.__anext__()]  # arrives although the clip is not finished
        assert not tts.is_cached("a line") and len(got[0]) == 100
        release.set()
        got += [c async for c in audio]
        return b"".join(got)

    pcm = asyncio.run(scenario())
    assert len(pcm) == 200 and calls == ["a line"]
    assert tts.is_cached("a line")


def test_a_failed_prefetch_falls_back_to_a_live_request(monkeypatch):
    import asyncio

    from demo_api import tts

    tts.clear_cache()
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    attempts = []

    class Fine:
        async def aiter_bytes(self):
            yield b"\x01\x00" * 10

        async def aclose(self):
            pass

    async def open_stream(text):
        attempts.append(1)
        if len(attempts) == 1:
            raise tts.TTSUnavailable("boom")  # the prefetch fails...
        return Fine()  # ...the real request succeeds

    monkeypatch.setattr(tts, "open_stream", open_stream)

    async def scenario():
        tts.prefetch("x")
        audio = await tts.open("x")
        return [c async for c in audio]

    assert asyncio.run(scenario()) == [b"\x01\x00" * 10] and len(attempts) == 2


# ---------------------------------------------------------------- who is the visitor (rate-limit identity)

def test_visitor_is_taken_from_the_cloudflare_header_not_from_forwarded_for(client):
    h = {"CF-Connecting-IP": "203.0.113.7", "X-Forwarded-For": "6.6.6.6, 172.70.1.1"}
    r = client.get("/whoami", headers=h).json()
    assert r["ip"] == "203.0.113.7" and r["source"] == "cf-connecting-ip"


def test_without_the_header_it_falls_back_to_the_last_forwarded_entry_then_the_socket(client):
    assert client.get("/whoami", headers={"X-Forwarded-For": "6.6.6.6, 198.51.100.9"}).json()["ip"] == "198.51.100.9"
    assert client.get("/whoami").json()["source"] == "socket"


def test_the_header_can_be_changed_or_switched_off(client, monkeypatch):
    monkeypatch.setenv("CLIENT_IP_HEADER", "True-Client-IP")
    assert client.get("/whoami", headers={"True-Client-IP": "192.0.2.5", "CF-Connecting-IP": "1.1.1.1"}).json()["ip"] == "192.0.2.5"
    monkeypatch.setenv("CLIENT_IP_HEADER", "")
    assert client.get("/whoami", headers={"CF-Connecting-IP": "1.1.1.1", "X-Forwarded-For": "9.9.9.9"}).json()["ip"] == "9.9.9.9"


def test_one_visitor_cannot_dodge_the_limit_by_changing_forwarded_for(client, monkeypatch):
    monkeypatch.setattr(guard, "PER_IP_DAILY", 2)
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    codes = [client.post("/order", json={"utterance": "x"}, headers={"CF-Connecting-IP": "203.0.113.7", "X-Forwarded-For": f"10.0.0.{i}"}).status_code
             for i in range(3)]
    assert codes == [200, 429, 429]
    other = client.post("/order", json={"utterance": "x"}, headers={"CF-Connecting-IP": "203.0.113.8"})
    assert other.status_code == 200  # a different visitor has their own allowance


# ---------------------------------------------------------------- visible allowance and refunds

def ip_credits_used():
    return sum(v for k, (v, _) in guard._mem.items() if ":ip:" in k and ":tts:" not in k)


def test_responses_say_how_much_of_the_free_allowance_is_left(client, monkeypatch):
    monkeypatch.setattr(guard, "PER_IP_DAILY", 10)
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    assert order(client, "x").json()["freeRemaining"] == 8  # 10 - one turn (2 credits)
    assert order(client, "x").json()["freeRemaining"] == 6
    ev = dict(events(client, "x"))
    assert ev["done"]["freeRemaining"] == 4 and ev["done"]["timings"]["limiter"] >= 0


def test_a_visitors_own_key_has_no_allowance_to_show(client, monkeypatch):
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    r = client.post("/order", json={"utterance": "x"}, headers={"X-LLM-Api-Key": "sk-mine"}).json()
    assert "freeRemaining" not in r and ip_credits_used() == 0


def test_a_failed_model_call_gives_the_credits_back(client, monkeypatch):
    monkeypatch.setattr(guard, "PER_IP_DAILY", 2)  # room for exactly one turn

    def down(*a, **k):
        raise llm.LLMUnavailable("pool busy")

    monkeypatch.setattr(llm, "chat_json", down)
    assert order(client, "x").status_code == 503 and ip_credits_used() == 0
    assert next(v for k, (v, _) in guard._mem.items() if ":global:" in k and ":tts:" not in k) == 0
    stub(monkeypatch, {"product_name": "Selfie Stick", "order_qty": 1})
    assert order(client, "x").status_code == 200  # still allowed: the failed attempt was free


def test_a_failed_streamed_turn_is_refunded_but_a_completed_stock_decision_is_not(client, monkeypatch):
    monkeypatch.setattr(guard, "PER_IP_DAILY", 100)

    def down(*a, **k):
        raise llm.LLMUnavailable("pool busy")

    monkeypatch.setattr(llm, "chat_json", down)
    r = client.post("/order/stream", json={"utterance": "x"})
    assert "event: error" in r.text and ip_credits_used() == 0  # failed before any stock decision
    # the stock decision was made and sent, only the spoken rewrite failed: the turn happened, so it is paid for
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"product_name": "Selfie Stick", "order_qty": 1})
    monkeypatch.setattr(llm, "text", lambda *a, **k: (_ for _ in ()).throw(llm.LLMUnavailable("busy")))
    dict(events(client, "a selfie stick"))
    assert ip_credits_used() == 2
