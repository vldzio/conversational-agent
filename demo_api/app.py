"""VoiceFlow demo API.

* ``POST /order``         one turn as a single JSON response (LLM parse -> inventory check -> spoken reply)
* ``POST /order/stream``  the same turn as Server-Sent Events, so the stock decision arrives before the spoken line
* ``POST /order/prepare`` do the slow work for a sentence while the customer is still talking (no stock change)
* ``POST /inventory``     the session's inventory

Credits: a turn costs 2, or 1 when it was prepared; a preparation costs 1.
"""
from __future__ import annotations

import os

os.environ.setdefault("DEMO_PROJECT", "voiceflow")
os.environ.setdefault("LANGSMITH_PROJECT", "voiceflow-demo")

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from fastapi.responses import StreamingResponse  # noqa: E402

from . import tts, voiceflow_core as core  # noqa: E402
from .base import admit_voice, create_app, guarded, guarded_stream  # noqa: E402


def speculate() -> bool:
    return os.environ.get("SPECULATE", "1").lower() not in ("0", "false", "no")


app = create_app("VoiceFlow demo API", info=lambda: {"speculate": speculate(), "tts": tts.enabled()})


async def _body(req: Request) -> dict:
    try:
        data = await req.json()
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _turn(body: dict) -> tuple[str, str | None]:
    utterance = str(body.get("utterance") or "").strip()[:300]
    session_id = body.get("sessionId") if isinstance(body.get("sessionId"), str) else None
    return utterance, session_id


EMPTY = {"error": "Say or type what you would like to order."}


@app.on_event("startup")
async def warm_voice() -> None:
    """Pre-synthesise the fixed lines in the background so the greeting and the cancellation reply start instantly."""
    import asyncio

    if tts.enabled():
        asyncio.get_running_loop().create_task(tts.warm([core.GREETING, core.cancellation_result()["message"]]))


@app.post("/inventory")
async def inventory(req: Request):
    body = await _body(req)
    return core.inventory_view(body.get("sessionId"))


@app.post("/order")
async def order(req: Request):
    utterance, session_id = _turn(await _body(req))
    if not utterance:
        return JSONResponse(EMPTY, status_code=400)
    cost = 1 if core.has_prepared(session_id, utterance) or core.is_cancellation(utterance) else 2
    return await guarded(req, cost, lambda: core.handle_order(session_id, utterance), name="voiceflow.order")


@app.post("/order/stream")
async def order_stream(req: Request):
    utterance, session_id = _turn(await _body(req))
    if not utterance:
        return JSONResponse(EMPTY, status_code=400)
    cost = 1 if core.has_prepared(session_id, utterance) or core.is_cancellation(utterance) else 2
    return await guarded_stream(req, cost, lambda: core.turn_events(session_id, utterance), name="voiceflow.order")


@app.post("/tts")
async def speak(req: Request):
    """Stream the agent's voice (raw 16-bit mono PCM) for a line this session's agent produced."""
    body = await _body(req)
    text = str(body.get("text") or "").strip()
    session_id = body.get("sessionId") if isinstance(body.get("sessionId"), str) else None
    if not tts.enabled():
        return JSONResponse({"error": "Voice is not configured."}, status_code=503)
    if len(text) > tts.MAX_CHARS or not core.may_speak(session_id, text):
        return JSONResponse({"error": "Only the agent's own replies can be voiced."}, status_code=400)
    prefetch = body.get("prefetch") is True
    if tts.is_cached(text) or tts.is_pending(text):
        if prefetch:
            return JSONResponse({"prefetching": False})  # already have it: nothing to fetch, nothing to charge
    else:  # new audio costs provider quota: apply the voice limits (a cache hit costs nothing)
        refusal = await admit_voice(req, len(text))
        if refusal:
            return refusal
        if prefetch:
            return JSONResponse({"prefetching": tts.prefetch(text)})
    try:
        audio = await tts.open(text)
    except tts.TTSUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    return StreamingResponse(
        audio, media_type="audio/L16",
        headers={"X-Sample-Rate": str(tts.SAMPLE_RATE), "Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/order/prepare")
async def order_prepare(req: Request):
    utterance, session_id = _turn(await _body(req))
    if not utterance or not speculate() or core.is_cancellation(utterance):  # nothing to prepare
        return JSONResponse({"prepared": False})
    if core.has_prepared(session_id, utterance):
        return JSONResponse({"sessionId": session_id, "prepared": True})  # already paid for
    return await guarded(req, 1, lambda: core.prepare(session_id, utterance), name="voiceflow.prepare")
