"""Shared FastAPI scaffolding for every demo API: CORS, gateway bootstrap,
admission control (per-IP/global limits or BYO key) and LangSmith-traced handlers."""
from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from typing import Any, Callable, Iterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import gateway, guard, llm


def create_app(title: str, *, on_startup: Callable[[], None] | None = None, info: Callable[[], dict[str, Any]] | None = None) -> FastAPI:
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")

    app = FastAPI(title=title, docs_url="/docs")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Server-Timing", "X-Sample-Rate"],
        max_age=86400,  # cache the CORS preflight so it is not paid on every turn
    )

    @app.on_event("startup")
    def _startup() -> None:
        if on_startup:
            on_startup()
        threading.Thread(target=_connect_and_warm, daemon=True, name="gateway-connect").start()

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": title,
            "llmReady": bool(os.environ.get("LLM_BASE_URL")),
            "tracing": llm.tracing_enabled(),
            "langsmithProject": os.environ.get("LANGSMITH_PROJECT"),
            **(info() if info else {}),
        }

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _connect_and_warm() -> None:
    """Connect to the gateway, then pay the one-off costs (imports, client, first connection to the
    provider) with a one-token call so the first visitor's turn is not the slow one."""
    if not gateway.connect():
        return
    try:
        import jsonschema  # noqa: F401

        llm.text([{"role": "user", "content": "Reply with: ok"}], temperature=0.0, max_tokens=1)
    except Exception:  # noqa: BLE001 - warming up is best effort
        pass


def client_ip(req: Request) -> str:
    # The platform proxy appends the real client IP as the LAST entry; earlier entries are client-supplied
    # and spoofable, so never use them for rate limiting.
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[-1].strip() if fwd else (req.client.host if req.client else "unknown")


def _flush_traces() -> None:
    """Send pending LangSmith traces from a background thread, never on the request path."""
    if not llm.tracing_enabled():
        return

    def flush() -> None:
        try:
            from langsmith import Client

            Client().flush()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=flush, daemon=True, name="langsmith-flush").start()


def _traced(fn: Callable[[], Any], name: str, byo: bool) -> Callable[[], Any]:
    if llm.tracing_enabled():
        try:
            from langsmith import traceable

            return traceable(name=name, run_type="chain", metadata={"byo_key": byo, "project": guard.PROJECT})(fn)
        except Exception:  # noqa: BLE001
            pass
    return fn


async def _admit(req: Request, cost: int, scope: str = "llm") -> tuple[tuple[str, str] | None, JSONResponse | None, int]:
    """Validate a BYO key and apply the limits. Returns (byo, refusal, limiter_ms)."""
    import anyio

    byo = None
    if scope == "llm":
        try:
            byo = guard.validate_byo(req.headers.get("x-llm-api-key"), req.headers.get("x-llm-base-url"))
        except ValueError as exc:
            return None, JSONResponse({"error": str(exc)}, status_code=400), 0
    t0 = time.perf_counter()
    decision = await anyio.to_thread.run_sync(guard.check, client_ip(req), cost, bool(byo), scope)
    limiter_ms = round((time.perf_counter() - t0) * 1000)
    if not decision.allowed:
        return byo, JSONResponse({"error": decision.message, "rateLimited": True}, status_code=429), limiter_ms
    return byo, None, limiter_ms


async def guarded(req: Request, cost: int, fn: Callable[[], Any], *, name: str) -> JSONResponse:
    """Admit the request (limits / BYO key), run ``fn`` in a worker thread under a LangSmith trace.

    ``fn`` must be a sync callable returning something JSON-serialisable.
    """
    import anyio

    byo, refusal, limiter_ms = await _admit(req, cost)
    if refusal:
        return refusal

    def run() -> Any:
        with llm.byo(byo[0] if byo else None, byo[1] if byo else None):
            return _traced(fn, name, bool(byo))()

    ctx = contextvars.copy_context()
    t0 = time.perf_counter()
    try:
        result = await anyio.to_thread.run_sync(lambda: ctx.run(run))
        total = round((time.perf_counter() - t0) * 1000)
        return JSONResponse(result, headers={"Server-Timing": f"limiter;dur={limiter_ms}, handler;dur={total}"})
    except llm.LLMUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except ValueError as exc:  # bad input detected by handlers
        return JSONResponse({"error": str(exc)}, status_code=400)
    finally:
        _flush_traces()


async def admit_voice(req: Request, chars: int) -> JSONResponse | None:
    """Apply the voice limits (characters). Returns a 429 response, or None when admitted."""
    _, refusal, _ = await _admit(req, chars, "tts")
    return refusal


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


async def guarded_stream(
    req: Request, cost: int, fn: Callable[[], Iterator[tuple[str, dict[str, Any]]]], *, name: str
) -> StreamingResponse | JSONResponse:
    """Like ``guarded`` but ``fn`` returns an iterator of ``(event, data)`` sent as Server-Sent Events,
    so the client can act on early events (the stock decision) before later ones (the spoken line) exist.
    Failures after the stream has started are sent as an ``error`` event.
    """
    import anyio

    byo, refusal, limiter_ms = await _admit(req, cost)
    if refusal:
        return refusal

    # Each step of the iterator runs in a worker thread but inside this one context, so the BYO key
    # (a context variable) stays visible for the whole turn and never leaks into another request.
    ctx = contextvars.copy_context()
    ctx.run(llm._byo_key.set, byo[0] if byo else None)
    ctx.run(llm._byo_url.set, byo[1] if byo else None)
    end = object()

    def collect() -> list[tuple[str, dict[str, Any]]]:
        return list(fn())

    async def body():
        try:
            if llm.tracing_enabled():
                # a traced run needs a plain call, so the events are produced first and then sent
                events = await anyio.to_thread.run_sync(lambda: ctx.run(_traced(collect, name, bool(byo))))
                for event, data in events:
                    yield _sse(event, data)
            else:
                it = ctx.run(fn)
                while True:
                    item = await anyio.to_thread.run_sync(lambda: ctx.run(next, it, end))
                    if item is end:
                        break
                    event, data = item
                    if event == "done":
                        data = {**data, "timings": {**data.get("timings", {}), "limiter": limiter_ms}}
                    yield _sse(event, data)
        except llm.LLMUnavailable as exc:
            yield _sse("error", {"error": str(exc)})
        except ValueError as exc:
            yield _sse("error", {"error": str(exc)})
        finally:
            _flush_traces()

    return StreamingResponse(
        body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
