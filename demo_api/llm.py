"""Traced, OpenAI-compatible LLM helper used by every demo API.

* Talks to the private FreeLLMAPI gateway (LLM_BASE_URL / LLM_API_KEY, set by
  ``gateway.connect``) or, for bring-your-own-key requests, to the visitor's
  provider (see ``guard.validate_byo``).
* Calls go to a pinned fast model first (``LLM_MODELS``, comma-separated, tried in
  order) because the gateway's ``auto`` route can pick slow or reasoning models.
* Every call is traced to LangSmith (project from ``LANGSMITH_PROJECT``) when
  ``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY`` are set.
* ``chat_json`` enforces JSON output portably (free models differ in
  ``response_format`` support): lenient parse + optional ``jsonschema``
  validation + one corrective retry.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_byo_key: ContextVar[str | None] = ContextVar("byo_key", default=None)
_byo_url: ContextVar[str | None] = ContextVar("byo_url", default=None)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# Tried in order. "name@effort" sets reasoning_effort for a reasoning model (less thinking = faster).
DEFAULT_MODELS = "qwen3.8-27b,gpt-oss-120b@low,gpt-oss-20b@low,auto"

_clients: dict[tuple[str, str], Any] = {}  # gateway clients only: reused so connections stay warm
_no_response_format: set[str] = set()  # models that rejected response_format (skip the failed try next time)


class LLMUnavailable(RuntimeError):
    """The model pool could not answer (rate-limited, no key, upstream error)."""


@contextmanager
def byo(api_key: str | None, base_url: str | None) -> Iterator[None]:
    t1, t2 = _byo_key.set(api_key), _byo_url.set(base_url)
    try:
        yield
    finally:
        _byo_key.reset(t1)
        _byo_url.reset(t2)


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))


def client():
    from openai import OpenAI

    is_byo = bool(_byo_key.get())
    base_url = _byo_url.get() or os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise LLMUnavailable("The LLM gateway is not ready yet. Try again in a few seconds.")
    api_key = _byo_key.get() or os.environ.get("LLM_API_KEY") or "not-needed"
    if not is_byo and (base_url, api_key) in _clients:
        return _clients[(base_url, api_key)]
    c = OpenAI(base_url=base_url, api_key=api_key, timeout=float(os.environ.get("LLM_TIMEOUT_S", "20")), max_retries=0)
    if tracing_enabled():
        try:
            from langsmith.wrappers import wrap_openai

            c = wrap_openai(c)
        except Exception:  # noqa: BLE001 - tracing must never break a request
            pass
    if not is_byo:  # a visitor's own key is never kept beyond the request
        _clients[(base_url, api_key)] = c
    return c


def model_ids() -> list[str]:
    """Models to try in order: the first is the fast path, the rest are fallbacks."""
    if _byo_key.get():
        return [os.environ.get("LLM_BYO_MODEL_ID") or os.environ.get("LLM_MODEL_ID", "auto")]
    raw = os.environ.get("LLM_MODELS") or DEFAULT_MODELS
    return [m.strip() for m in raw.split(",") if m.strip()] or ["auto"]


def chat(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    json_mode: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """One chat completion; returns the SDK response object."""
    from openai import APIStatusError, APIConnectionError, APITimeoutError

    base: dict[str, Any] = dict(messages=messages, temperature=temperature)
    if tools:
        base["tools"] = tools
        if tool_choice:
            base["tool_choice"] = tool_choice
    if metadata:
        base["extra_body"] = {"metadata": metadata}
    c = client()
    last: Exception | None = None
    for i, model in enumerate(model_ids()):
        # reasoning models (marked "@effort") and unknown fallbacks need room to think before the answer,
        # otherwise the reply is cut off mid-thought and the gateway puts the model on cooldown
        model, _, effort = model.partition("@")
        kwargs = dict(base, model=model, max_tokens=max_tokens * 4 + 300 if (effort or i > 0) else max_tokens)
        if effort:
            kwargs["reasoning_effort"] = effort
        try:
            if json_mode and model not in _no_response_format:
                try:
                    return c.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
                except APIStatusError as exc:
                    if exc.status_code >= 500 or exc.status_code in (429, 413):
                        raise
                    _no_response_format.add(model)  # this model rejects response_format: ask plainly from now on
            return c.chat.completions.create(**kwargs)
        except (APIStatusError, APIConnectionError, APITimeoutError) as exc:
            last = exc
    raise LLMUnavailable(
        "The free LLM pool could not answer right now (rate-limited or busy). "
        "Try again in a minute, or add your own API key via 'LLM key'."
    ) from last


def text(messages: list[dict[str, Any]], **kw: Any) -> str:
    resp = chat(messages, **kw)
    choice = resp.choices[0]
    if getattr(choice, "finish_reason", None) == "length":  # cut off (often mid-thought): never use it
        raise LLMUnavailable("The model's answer was cut off. Please try again.")
    return (choice.message.content or "").strip()


def parse_json(raw: str) -> Any:
    s = _FENCE.sub("", raw.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            a, b = s.find(open_c), s.rfind(close_c)
            if a != -1 and b > a:
                try:
                    return json.loads(s[a : b + 1])
                except json.JSONDecodeError:
                    continue
        raise


def chat_json(
    system: str,
    user: str,
    *,
    schema: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    metadata: dict[str, Any] | None = None,
    schema_in_prompt: bool = True,
) -> Any:
    """Ask for a JSON object; validate against ``schema`` if given; retry once with feedback.

    ``schema_in_prompt=False`` keeps the prompt short; the schema is still enforced here.
    """
    sys_prompt = system + "\n\nRespond with ONLY valid JSON (no prose, no markdown fences)."
    if schema and schema_in_prompt:
        sys_prompt += f" It must conform to this JSON Schema:\n{json.dumps(schema)}"
    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
    last: Exception | None = None
    for _ in range(2):
        raw = text(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True, metadata=metadata)
        try:
            out = parse_json(raw)
            if schema:
                import jsonschema

                jsonschema.validate(out, schema)
            return out
        except Exception as exc:  # noqa: BLE001
            last = exc
            messages += [
                {"role": "assistant", "content": raw[:4000]},
                {"role": "user", "content": f"That was not valid ({str(exc)[:300]}). Reply again with ONLY the corrected JSON."},
            ]
    raise LLMUnavailable(f"The model did not return valid JSON ({last}). Please try again.")
