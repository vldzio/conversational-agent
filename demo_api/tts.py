"""Text-to-speech for the demo, streamed from Deepgram Aura-2.

The browser never sees the Deepgram key: it asks ``POST /tts`` for a line the agent
already said, and this module relays the raw 16-bit PCM audio as it is produced, so
playback can start on the first chunk instead of after the whole sentence.
If ``DEEPGRAM_API_KEY`` is not set (or Deepgram fails) the UI falls back to the
browser's own voice.

Latency: finished audio is kept in a small in-memory cache, so a repeated line costs
nothing and starts at once. ``prefetch`` fills the cache ahead of time (used while the
caller is still talking, and at start-up for the fixed lines such as the greeting).
"""
from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from typing import AsyncIterator, Iterable

import httpx

URL = os.environ.get("TTS_URL", "https://api.deepgram.com/v1/speak")  # overridable for local testing
SAMPLE_RATE = 24000
MAX_CHARS = 300
CACHE_MAX_BYTES = int(os.environ.get("TTS_CACHE_BYTES", str(16 * 1024 * 1024)))  # 0 turns the cache off
CACHE_MAX_ENTRIES = 200
SLICE = 16384  # cached audio is replayed in slices so the player can start on the first one
WAIT_FOR_PREFETCH_S = 8.0


class TTSUnavailable(RuntimeError):
    """Deepgram is not configured, refused the request, or could not be reached."""


def enabled() -> bool:
    return bool(os.environ.get("DEEPGRAM_API_KEY"))


def _model() -> str:
    return os.environ.get("TTS_MODEL", "aura-2-thalia-en")


_client: httpx.AsyncClient | None = None
_cache: "OrderedDict[tuple[str, str], bytes]" = OrderedDict()
_cache_bytes = 0
_inflight: "dict[tuple[str, str], _Fill]" = {}


class _Fill:
    """A line being synthesised in the background. Chunks are shared, so a request that arrives while it is
    still running joins the audio already coming in instead of waiting for the whole clip."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.done = False
        self.cond = asyncio.Condition()
        self.task: asyncio.Task | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:  # one pooled connection set, so a turn does not pay a TLS handshake
        _client = httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5))
    return _client


def _key(text: str) -> tuple[str, str]:
    return _model(), text


def clear_cache() -> None:
    global _cache_bytes
    _cache.clear()
    _inflight.clear()
    _cache_bytes = 0


def _store(key: tuple[str, str], pcm: bytes) -> None:
    global _cache_bytes
    if CACHE_MAX_BYTES <= 0 or not pcm or len(pcm) > CACHE_MAX_BYTES:
        return
    if key in _cache:
        _cache_bytes -= len(_cache.pop(key))
    _cache[key] = pcm
    _cache_bytes += len(pcm)
    while _cache and (_cache_bytes > CACHE_MAX_BYTES or len(_cache) > CACHE_MAX_ENTRIES):
        _cache_bytes -= len(_cache.popitem(last=False)[1])  # drop the least recently used line


def is_cached(text: str) -> bool:
    return _key(text) in _cache


def is_pending(text: str) -> bool:
    fill = _inflight.get(_key(text))
    return bool(fill and not fill.done)


async def open_stream(text: str) -> httpx.Response:
    """Start synthesis and return the open response once Deepgram has accepted it (status checked)."""
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise TTSUnavailable("Voice is not configured.")
    params = {"model": _model(), "encoding": "linear16", "sample_rate": str(SAMPLE_RATE), "container": "none"}
    try:
        req = _http().build_request("POST", URL, params=params, headers={"Authorization": f"Token {key}"}, json={"text": text[:MAX_CHARS]})
        resp = await _http().send(req, stream=True)
    except httpx.HTTPError as exc:
        raise TTSUnavailable(f"Voice service unreachable ({type(exc).__name__}).") from exc
    if resp.status_code != 200:
        await resp.aclose()
        raise TTSUnavailable(f"Voice service refused the request ({resp.status_code}).")
    return resp


async def _fill(text: str, fill: _Fill) -> None:
    """Synthesise a line, sharing chunks with anyone following it, and keep the finished clip.
    Failures are silent: speaking the line later just takes the normal path."""
    key = _key(text)
    complete = False
    try:
        resp = await open_stream(text)
        try:
            async for chunk in resp.aiter_bytes():
                async with fill.cond:
                    fill.parts.append(chunk)
                    fill.cond.notify_all()
            complete = True
        finally:
            await resp.aclose()
    except Exception:  # noqa: BLE001
        pass
    finally:
        if complete:
            _store(key, b"".join(fill.parts))  # a cut-off clip is never kept
        fill.done = True
        _inflight.pop(key, None)
        async with fill.cond:
            fill.cond.notify_all()


def _start(text: str) -> _Fill:
    fill = _Fill()
    _inflight[_key(text)] = fill
    fill.task = asyncio.get_running_loop().create_task(_fill(text, fill))
    return fill


def prefetch(text: str) -> bool:
    """Start filling the cache for ``text`` in the background. Returns False if there is nothing to do."""
    if CACHE_MAX_BYTES <= 0 or _key(text) in _cache or is_pending(text):
        return False
    _start(text)
    return True


async def warm(lines: Iterable[str]) -> None:
    """Pre-synthesise fixed lines (greeting, cancellation reply) so they play instantly."""
    for line in lines:
        if enabled() and not is_cached(line) and not is_pending(line):
            fill = _start(line)
            await fill.task


async def _follow(fill: _Fill) -> AsyncIterator[bytes]:
    i = 0
    while True:
        async with fill.cond:
            await fill.cond.wait_for(lambda: i < len(fill.parts) or fill.done)
            if i >= len(fill.parts):
                return
            chunk = fill.parts[i]
        i += 1
        yield chunk


async def _replay(pcm: bytes) -> AsyncIterator[bytes]:
    for i in range(0, len(pcm), SLICE):
        yield pcm[i : i + SLICE]


async def _relay(resp: httpx.Response, key: tuple[str, str]) -> AsyncIterator[bytes]:
    parts: list[bytes] = []
    complete = False
    try:
        async for chunk in resp.aiter_bytes():
            parts.append(chunk)
            yield chunk
        complete = True
    finally:
        await resp.aclose()
        if complete:
            _store(key, b"".join(parts))  # a cut-off clip is never kept


async def open(text: str) -> AsyncIterator[bytes]:  # noqa: A001
    """Audio for ``text`` as an async iterator of PCM chunks.

    Served from the cache when possible, joined to a prefetch already under way, or streamed live.
    Raises ``TTSUnavailable`` before returning if it cannot start.
    """
    key = _key(text)
    fill = _inflight.get(key)
    if fill and not fill.done and key not in _cache:
        async with fill.cond:
            try:
                await asyncio.wait_for(fill.cond.wait_for(lambda: bool(fill.parts) or fill.done), WAIT_FOR_PREFETCH_S)
            except asyncio.TimeoutError:
                pass
        if fill.parts:
            return _follow(fill)
    if key in _cache:
        _cache.move_to_end(key)
        return _replay(_cache[key])
    return _relay(await open_stream(text), key)
