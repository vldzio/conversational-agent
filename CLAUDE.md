# VoiceFlow (conversational-agent) - project context

## What this is
Voice sales-assistant proof of concept: speech in -> LLM extracts product + quantity -> deterministic inventory check -> spoken reply. The original local app uses Google Cloud STT/TTS and a Hugging Face model (`model.py`, `streamlit_ui.py`). The `demo-ready` branch adds a web demo:

```text
Browser (Vercel, static `demo/`, Web Speech API for the mic, streamed Deepgram voice via `/tts` with browser-voice fallback) -> Render Docker service `demo_api/` (limits, BYO key, LangSmith)
                                       -> private FreeLLMAPI gateway (127.0.0.1:3001, same container) -> free-tier LLMs
```

## Layout
- `demo_api/voiceflow_core.py` - `parse_query` (LLM, schema-validated), `check_inventory` (same rules as `model.check_inventory`, plain Python; `commit=False` = dry run), `spoken_reply` (LLM rewrite; spoken only if it keeps every number, else the authoritative message), `turn_events` (one turn as `parsed` -> `result` -> `say` -> `done`), `prepare` (does a turn's LLM work early; its reply is reused only if the real stock check equals the dry run), per-session in-memory inventory copy (`inventory.csv` is never written).
- `demo_api/app.py` - `POST /order` (JSON), `POST /order/stream` (SSE, what the UI uses), `POST /order/prepare`, `POST /inventory`. Shared plumbing: `llm.py`, `base.py`, `guard.py`, `gateway.py`.
- `demo/index.html` (+ `byo.js`, `base.css`, `config.js`, `write-config.mjs`) - call-simulator UI. `deploy/render/` - Render recipe (`render.yaml`, Dockerfile).
- Original files fixed on this branch: `model.py` (token from env), `tts_for_model.py` (no side effect on import), `requirements.txt`. Original files keep CRLF line endings; preserve them when editing.

## Run and test
- `python -m venv .venv && .venv\Scripts\pip install -r demo_api/requirements.txt pytest && .venv\Scripts\pytest tests` (47 tests; LLM stubbed at `llm.chat_json`/`llm.text`).
- Local UI: set `API_URL` in `demo/config.js` to a local `uvicorn demo_api.app:app --port 8150`, serve `demo/` with `python -m http.server`. Restore `API_URL` to `""` before committing.

## Rules
- The LLM never decides stock, what to ask, or confirms orders; only `decide`/`check_inventory` do. Incomplete orders get a deterministic follow-up (`ambiguous_product`, `need_quantity`, `need_product`, `not_understood`) and the session remembers what was asked (`Pending`, 3 min) so the next sentence can answer it. Loose names ("speakers") are resolved in Python by word match (`match_products`), not trusted from the model. Quantities must be positive ints (bools/strings/floats are rejected). Never hard-code LLM text.
- Voice: `POST /tts` (`demo_api/tts.py`) relays Deepgram Aura-2 raw PCM (key `DEEPGRAM_API_KEY`, server-side only) and only speaks lines the session's agent produced (`core.may_speak`; greeting comes from `/inventory`). Metered in characters, separately from LLM credits (`TTS_*` limits); a visitor's own LLM key does not bypass it. Audio is cached in memory (`tts.open`/`prefetch`/`warm`): a repeated line is instant and free, the greeting and cancellation reply are warmed at startup, and while the caller is still talking the page asks `/tts` with `prefetch: true` for the audio of the line `/order/prepare` returned (a request that arrives mid-synthesis joins the live stream). The rewrite runs at temperature 0 so the same result gives the same sentence and hits the cache. UI `speak()` in `demo/index.html` streams and schedules PCM chunks with WebAudio and falls back to the browser voice on any failure or when `/` reports `tts: false`.
- Cancellation requests (`cancel*`) get a fixed reply from `core.cancellation_result` (no LLM, `status: cancellation_unsupported`) pointing to `SUPPORT_EMAIL`; the agent never cancels or reverses orders.
- Latency: LLM calls use the pinned chain in `LLM_MODELS` (default `qwen3.8-27b,gpt-oss-120b@low,gpt-oss-20b@low,auto`; `@low` = reasoning effort). Never make `auto` the first choice: it can route to slow reasoning models. The fast model's free tier is about 30 requests/min and 250/day; after that calls fall back to slower models.
- Responses carry `freeRemaining` (credits left today for that visitor; absent with a BYO key) and the page shows "about N free turns left". A turn that fails before any stock decision (model pool down) is refunded (`guard.refund`).
- Credits = LLM calls: a turn costs 2, a prepared turn 1, a prepare 1. Limits: `RL_PER_IP_PER_MIN` (20), `RL_PER_IP_DAILY` (40), `RL_GLOBAL_DAILY` (1000). `SPECULATE=0` turns early preparation off.
- Client IP for limits = `CLIENT_IP_HEADER` (default `cf-connecting-ip`, set by Cloudflare in front of Render), else the LAST `X-Forwarded-For` entry, else the socket (`base.client_ip_source`; `GET /whoami` shows which). Never use the first `X-Forwarded-For` entry: clients can forge it. Never log API keys. Voice input needs a Chromium/Safari browser; typing always works.

## Deploy facts
- Render service `voiceflow-demo-api` (blueprint `deploy/render/render.yaml`), LangSmith project `voiceflow-demo`, Vercel dir `demo/`, env var `DEMO_API_URL`.
- Render env vars: `ENCRYPTION_KEY` (64 hex), `FREEAPI_CONFIG_JSON`, `LANGSMITH_API_KEY`, `UPSTASH_REDIS_REST_URL/TOKEN`; optional `CORS_ORIGINS`, `RL_*`, `LLM_MODELS`, `SPECULATE`. The image clones branch `demo-ready`.
- Steps: create the Render service from `deploy/render/render.yaml`, set the env vars, `vercel login`, then `vercel deploy --prod --yes --build-env DEMO_API_URL=<render url>` from the repo root.

## Status
Verified: 47 tests pass; live turns through a local Docker gateway (typed turn about 0.4 s to the spoken line on the fast model, prepared turn about 10 ms). NOT verified: Render Docker build, speech recognition and early preparation in a real mic session, LangSmith tracing of streamed turns, live deploy.
