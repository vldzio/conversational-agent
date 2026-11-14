# VoiceFlow (conversational-agent) - project context

## What this is
Voice sales-assistant proof of concept: speech in -> LLM extracts product + quantity -> deterministic inventory check -> spoken reply. The original local app uses Google Cloud STT/TTS and a Hugging Face model (`model.py`, `streamlit_ui.py`). The `demo-ready` branch adds a web demo:

```text
Browser (Vercel, static `demo/`, Web Speech API for mic + voice) -> Hugging Face Space `demo_api/` (limits, BYO key, LangSmith)
                                       -> private FreeLLMAPI gateway (127.0.0.1:3001, same container) -> free-tier LLMs
```

## Layout
- `demo_api/voiceflow_core.py` - `parse_query` (LLM, schema-validated), `check_inventory` (same rules as `model.check_inventory`, plain Python), `spoken_reply` (LLM rewrite, falls back to the authoritative message), per-session in-memory inventory copy (`inventory.csv` is never written).
- `demo_api/app.py` - `POST /order`, `POST /inventory`. Shared plumbing: `llm.py`, `base.py`, `guard.py`, `gateway.py`.
- `demo/index.html` (+ `byo.js`, `base.css`, `config.js`, `write-config.mjs`) - call-simulator UI. `deploy/space/` - HF Space recipe.
- Original files fixed on this branch: `model.py` (token from env), `tts_for_model.py` (no side effect on import), `requirements.txt`. Original files keep CRLF line endings; preserve them when editing.

## Run and test
- `python -m venv .venv && .venv\Scripts\pip install -r demo_api/requirements.txt pytest && .venv\Scripts\pytest tests` (8 tests; LLM stubbed at `llm.chat_json`/`llm.text`).
- Local UI: set `API_URL` in `demo/config.js` to a local `uvicorn demo_api.app:app --port 8150`, serve `demo/` with `python -m http.server`. Restore `API_URL` to `""` before committing.

## Rules
- The LLM never decides stock or confirms orders; only `check_inventory` does. Quantities must be positive ints (bools/strings/floats are rejected). Never hard-code LLM text.
- Client IP for limits = LAST `X-Forwarded-For` entry. Never log API keys. Voice input needs a Chromium/Safari browser; typing always works.

## Deploy facts
- Space `voiceflow-demo-api`, LangSmith project `voiceflow-demo`, Vercel dir `demo/`, env var `DEMO_API_URL`.
- Space secrets: `ENCRYPTION_KEY` (64 hex), `FREEAPI_CONFIG_JSON`, `LANGSMITH_API_KEY`, `UPSTASH_REDIS_REST_URL/TOKEN`; optional `CORS_ORIGINS`, `RL_PER_IP_DAILY`, `RL_GLOBAL_DAILY`. The Space clones branch `demo-ready`.
- Steps: `hf auth login`, `vercel login`, create the Space from `deploy/space/`, set secrets, then `vercel deploy --prod --yes --build-env DEMO_API_URL=<space url>` from the repo root.

## Status
Verified: 8 tests pass; UI checked in a browser (inventory loads, errors surface cleanly). NOT verified: Space Docker build, real provider calls, speech recognition in a real mic session, live deploy.
