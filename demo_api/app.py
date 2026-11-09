"""VoiceFlow demo API: POST /order (LLM parse -> inventory check -> spoken reply), GET/POST /inventory."""
from __future__ import annotations

import os

os.environ.setdefault("DEMO_PROJECT", "voiceflow")
os.environ.setdefault("LANGSMITH_PROJECT", "voiceflow-demo")

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from . import voiceflow_core as core  # noqa: E402
from .base import create_app, guarded  # noqa: E402

app = create_app("VoiceFlow demo API")


async def _body(req: Request) -> dict:
    try:
        data = await req.json()
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


@app.post("/inventory")
async def inventory(req: Request):
    body = await _body(req)
    return core.inventory_view(body.get("sessionId"))


@app.post("/order")
async def order(req: Request):
    body = await _body(req)
    utterance = str(body.get("utterance") or "").strip()[:300]
    if not utterance:
        return JSONResponse({"error": "Say or type what you would like to order."}, status_code=400)
    session_id = body.get("sessionId") if isinstance(body.get("sessionId"), str) else None
    return await guarded(req, 2, lambda: core.handle_order(session_id, utterance), name="voiceflow.order")
