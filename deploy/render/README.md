# VoiceFlow demo API on Render

One Docker web service = private [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) gateway (127.0.0.1:3001) + the public `demo_api` (rate limits, BYO key, LangSmith). Use `render.yaml` as a Blueprint or create a Docker web service with root dir `deploy/render`.

| Env var | Value |
| --- | --- |
| `ENCRYPTION_KEY` | 64 hex chars, required by the gateway |
| `FREEAPI_CONFIG_JSON` | gateway config with provider keys (see `freellmapi.config.example.json`) |
| `LANGSMITH_API_KEY` | enables tracing to project `voiceflow-demo` |
| `UPSTASH_REDIS_REST_URL` / `_TOKEN` | durable counters; without them limits live in memory and reset on restart/spin-down |
| `DEEPGRAM_API_KEY` | enables the streamed Aura-2 voice (`TTS_MODEL`, default `aura-2-thalia-en`); without it, or when it fails or runs out, the browser voice is used |
| `TTS_CACHE_BYTES` | in-memory audio cache size, default 16 MB (`0` = off); cached lines cost no voice quota |
| `TTS_PER_IP_PER_MIN` / `TTS_PER_IP_DAILY` / `TTS_GLOBAL_DAILY` | voice limits in characters (about 60 per reply): defaults 800/min and 8000/day per visitor, 150000/day overall |
| `SUPPORT_EMAIL` | address shown when a caller asks to cancel an order (cancellations are never processed by the agent) |
| `CORS_ORIGINS` | the frontend origin |
| `CLIENT_IP_HEADER` | header that carries the visitor IP, default `cf-connecting-ip` (Render is behind Cloudflare and only appends to `X-Forwarded-For`). `GET /whoami` shows how a request is identified |
| `RL_PER_IP_PER_MIN` / `RL_PER_IP_DAILY` / `RL_GLOBAL_DAILY` | credits (about one LLM call each): defaults 20/min and 40/day per visitor, 1000/day overall (UTC) |
| `LLM_MODELS` | models tried in order, fastest first; `name@low` sets reasoning effort. Default `qwen3.8-27b,gpt-oss-120b@low,gpt-oss-20b@low,auto` |
| `SPECULATE` | `1` (default) prepares a voice turn while the caller is still talking; `0` turns it off to save credits |

Render's free tier spins down when idle; the first request after that waits for the gateway to boot.
