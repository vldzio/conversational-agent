"""Abuse protection for the public demo.

* Per-visitor (IP) limits in credits: a per-minute burst limit and a daily budget,
  plus a global daily cap on the shared free LLM pool. Counters live in Upstash
  Redis (REST) so they survive restarts and are shared across instances; without
  Upstash they fall back to in-process counters (local dev, reset on restart).
  All counters for a request are updated in ONE round trip over a kept-alive
  connection, so the limiter adds little latency.
* Bring-your-own-key: a visitor may send ``X-LLM-Api-Key`` (and optionally
  ``X-LLM-Base-Url`` from an allow-list). Such requests spend the visitor's own
  quota, so they bypass the shared limits. The key is only held in memory for
  the duration of the request and is never logged.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

PROJECT = os.environ.get("DEMO_PROJECT", "demo")
PER_IP_PER_MIN = int(os.environ.get("RL_PER_IP_PER_MIN", "20"))  # credits per visitor per minute (0 = off)
PER_IP_DAILY = int(os.environ.get("RL_PER_IP_DAILY", "40"))  # credits per visitor per UTC day
GLOBAL_DAILY = int(os.environ.get("RL_GLOBAL_DAILY", "1000"))  # credits for everyone per UTC day
# Voice (text-to-speech) is metered separately, in characters, because it spends a different free quota.
TTS_PER_IP_PER_MIN = int(os.environ.get("TTS_PER_IP_PER_MIN", "800"))
TTS_PER_IP_DAILY = int(os.environ.get("TTS_PER_IP_DAILY", "8000"))
TTS_GLOBAL_DAILY = int(os.environ.get("TTS_GLOBAL_DAILY", "150000"))
UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

ALLOWED_BYO_HOSTS = {
    "api.openai.com",
    "api.groq.com",
    "openrouter.ai",
    "api.mistral.ai",
    "api.cerebras.ai",
    "generativelanguage.googleapis.com",
    *[h.strip() for h in os.environ.get("LLM_BYO_ALLOWED_HOSTS", "").split(",") if h.strip()],
}
DEFAULT_BYO_BASE_URL = "https://api.openai.com/v1"

_mem: dict[str, tuple[int, float]] = {}
_mem_lock = threading.Lock()
_http: httpx.Client | None = None


@dataclass
class Decision:
    allowed: bool
    message: str = ""
    remaining: int | None = None
    byo: bool = False


def _day() -> str:
    return time.strftime("%Y%m%d", time.gmtime())


def _seconds_to_midnight_utc() -> int:
    return int(86400 - (time.time() % 86400)) + 1


def _client() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(timeout=5, headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"})
    return _http


def _incr_many(items: list[tuple[str, int, int]]) -> list[int]:
    """Add ``amount`` to each ``(key, amount, ttl)`` counter and return the new totals, in one round trip."""
    if UPSTASH_URL and UPSTASH_TOKEN:
        cmds: list[list] = []
        for key, amount, ttl in items:
            cmds += [["INCRBY", key, amount], ["EXPIRE", key, ttl]]
        r = _client().post(f"{UPSTASH_URL}/pipeline", json=cmds)
        r.raise_for_status()
        out = r.json()
        return [int(out[i * 2]["result"]) for i in range(len(items))]
    totals = []
    with _mem_lock:
        now = time.time()
        for key, amount, ttl in items:
            count, exp = _mem.get(key, (0, now + ttl))
            if now > exp:
                count, exp = 0, now + ttl
            count += amount
            _mem[key] = (count, exp)
            totals.append(count)
        if len(_mem) > 5000:  # drop expired keys so the local fallback cannot grow without bound
            for k in [k for k, (_, e) in _mem.items() if now > e]:
                _mem.pop(k, None)
    return totals


def validate_byo(api_key: str | None, base_url: str | None) -> tuple[str, str] | None:
    """Return (key, base_url) for a usable BYO key; raise ValueError if the URL is not allowed."""
    if not api_key:
        return None
    url = (base_url or DEFAULT_BYO_BASE_URL).strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "") not in ALLOWED_BYO_HOSTS:
        raise ValueError("Base URL not allowed. Use one of: " + ", ".join(sorted(ALLOWED_BYO_HOSTS)))
    return api_key.strip(), url


def _counters(ip: str, scope: str) -> tuple[tuple[int, int, int], list[tuple[str, int]]]:
    """((per-minute, per-day, global) limits, [(key, ttl)] for minute / visitor-day / global) for a scope."""
    voice = scope == "tts"
    limits = (TTS_PER_IP_PER_MIN, TTS_PER_IP_DAILY, TTS_GLOBAL_DAILY) if voice else (PER_IP_PER_MIN, PER_IP_DAILY, GLOBAL_DAILY)
    pre = f"rl:{PROJECT}:tts" if voice else f"rl:{PROJECT}"
    day, ttl = _day(), _seconds_to_midnight_utc()
    return limits, [(f"{pre}:min:{int(time.time() // 60)}:{ip}", 90), (f"{pre}:ip:{day}:{ip}", ttl), (f"{pre}:global:{day}", ttl)]


def refund(ip: str, cost: int, scope: str = "llm") -> None:
    """Give credits back when the work they paid for failed (e.g. the model pool was unavailable)."""
    if cost <= 0:
        return
    try:
        _incr_many([(key, -cost, ttl) for key, ttl in _counters(ip, scope)[1]])
    except Exception:  # noqa: BLE001 - a failed refund must never break the response
        pass


def check(ip: str, cost: int, byo: bool, scope: str = "llm") -> Decision:
    """Admit a request. Scope "llm" is metered in credits, "tts" (voice) in characters; each has its own counters."""
    if byo or cost <= 0:
        return Decision(True, byo=byo)
    voice = scope == "tts"
    (per_min, per_day, per_all), counters = _counters(ip, scope)
    try:
        ttl = counters[2][1]
        global_key = counters[2][0]
        minute_used, ip_used, global_used = _incr_many([(key, cost, t) for key, t in counters])
        denied = None
        if per_min > 0 and minute_used > per_min:
            denied = "Too many requests too quickly. Wait a minute and try again."
        elif ip_used > per_day:
            denied = (
                "Daily voice limit reached for this visitor; replies will use the browser voice instead. Come back tomorrow (UTC)."
                if voice else
                f"Daily free-demo limit reached ({PER_IP_DAILY} credits per visitor). "
                "Add your own OpenAI-compatible API key in the header to keep going, or come back tomorrow (UTC)."
            )
        if denied:
            # a refused visitor spent nothing from the shared pool: give the credits back to the global counter
            _incr_many([(global_key, -cost, ttl)])
            return Decision(False, denied, 0)
        if global_used > per_all:
            return Decision(
                False,
                "The shared voice quota for today is used up; replies will use the browser voice instead."
                if voice else
                "The shared free LLM quota for today is used up. Add your own API key to continue, or try again after 00:00 UTC.",
                0,
            )
        return Decision(True, remaining=max(per_day - ip_used, 0))
    except Exception as exc:  # fail closed: a broken limiter must never burn the shared quota
        return Decision(False, f"Rate limiter unavailable ({type(exc).__name__}); try again shortly.")
