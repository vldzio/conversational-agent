# VoiceFlow — voice-driven sales assistant

**A voice ordering assistant that separates language understanding from business logic.** The caller speaks an order, a language model extracts the product and quantity, plain Python checks and updates the inventory, and the agent answers out loud. The model never decides what is in stock.

**Live demo:** [conversational-agent-rouge.vercel.app](https://conversational-agent-rouge.vercel.app) (simulated call, typing works too)

```text
Caller's speech
      ↓ speech-to-text
Transcript
      ↓ LLM: extract {product, quantity} (schema-validated)
Inventory check in Python → in stock / not enough / not found / follow-up question
      ↓ reply sentence
Text-to-speech → caller hears the answer
```

VoiceFlow runs in two forms:

| | Local app (`streamlit_ui.py`) | Web demo (`demo/` + `demo_api/`) |
| --- | --- | --- |
| Speech in | Microphone + Google Cloud Speech-to-Text | Browser Web Speech API, or typing |
| Understanding | Hugging Face endpoint via LangChain (`model.py`) | LLM via OpenAI-compatible gateway, schema-validated |
| Inventory | `inventory.csv` | Per-visitor copy of `inventory.csv` |
| Speech out | Google Cloud Text-to-Speech (WaveNet) | Streamed Deepgram Aura-2 voice, browser voice as fallback |
| UI | Streamlit "Simulate Call" | Static call-simulator page (Vercel) + FastAPI backend (Render) |

## Features

- Understands requests such as “I want two wireless earbuds” and resolves loose product names (“speakers”) against the catalogue in code.
- Handles incomplete orders with a follow-up question (which product? how many?) and remembers what it asked.
- Replies with stock confirmations, shortfalls, or alternatives; confirmed orders update the inventory.
- Web demo streams each turn (parsed → result → spoken reply), prepares likely replies while the caller is still talking, and caches synthesized audio so repeated lines play instantly.
- Every LLM call in the web demo is traced to LangSmith; visitors get a daily allowance or can bring their own key.

## Run the local app

Requires Python 3.10+, a microphone and speakers, a Google Cloud service account with Speech-to-Text and Text-to-Speech enabled, and a Hugging Face token. PyAudio may also need the PortAudio library for your OS.

```bash
git clone https://github.com/vldzio/conversational-agent.git
cd conversational-agent
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your credentials (see `.env.example`):

```bash
export HUGGINGFACEHUB_API_TOKEN="your_hugging_face_token"
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
```

On Windows PowerShell use `$env:HUGGINGFACEHUB_API_TOKEN="..."` and `$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"`.

```bash
streamlit run streamlit_ui.py
```

Click **Simulate Call** and speak your order. Audio is recorded and played on the machine running Streamlit, and confirmed orders update `inventory.csv`.

## Run the web demo locally

```bash
pip install -r demo_api/requirements.txt pytest
pytest tests                                   # demo API tests, LLM stubbed
uvicorn demo_api.app:app --port 8150
```

Set `API_URL` in `demo/config.js` to `http://localhost:8150`, then serve `demo/` with `python -m http.server`. The API needs an OpenAI-compatible endpoint (`LLM_BASE_URL`, `LLM_API_KEY`); `DEEPGRAM_API_KEY` enables the streamed voice. Deployment recipes are in `deploy/render/` and `vercel.json`.

## Project structure

| Path | Purpose |
| --- | --- |
| `streamlit_ui.py` | Local simulated-call UI: audio → agent → speech |
| `audiotranscriber.py` | Microphone capture and Google Speech-to-Text |
| `model.py` | Hugging Face endpoint, order parsing, inventory action |
| `tts_for_model.py` | Google Text-to-Speech synthesis |
| `tester.py` | Command-line audio → agent → speech test |
| `inventory.csv` | Sample products and stock |
| `demo_api/` | FastAPI backend: order parsing, inventory rules, streaming turns, TTS relay, rate limits |
| `demo/` | Browser call simulator |
| `deploy/render/` | Render deployment blueprint |
| `tests/` | Web demo API tests |
