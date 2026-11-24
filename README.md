# VoiceFlow — conversational inventory demo

**A local, voice-driven sales assistant.** VoiceFlow records a short microphone utterance, transcribes it with Google Cloud Speech-to-Text, uses a Hugging Face-hosted language model to extract the requested product and quantity, checks a CSV inventory, and speaks the answer back with Google Cloud Text-to-Speech. A Streamlit interface ties the interaction together as a **simulated sales call**.

The design splits **language understanding** from **deterministic action**: the LLM interprets the customer's words, while plain Python performs the stock check and inventory update.

```text
Local microphone (5-second recording)
      ↓ Google Cloud Speech-to-Text
Transcript
      ↓ Hugging Face Mixtral endpoint: product + quantity JSON
CSV inventory check → stock/alternative/confirmation response
      ↓ Google Cloud Text-to-Speech
MP3 playback via pygame
```

## Capabilities

- Captures short spoken requests and transcribes them to English text.
- Parses requests such as “I want two wireless earbuds” into product and quantity.
- Checks `inventory.csv` and replies with an out-of-stock, insufficient-stock, or order-confirmation message. Confirmed orders update the quantity-on-hand and quantity-sold columns.
- Synthesizes the reply as an MP3 using a Google Cloud WaveNet voice and plays it back.
- Provides a Streamlit **Simulate Call** UI for running the whole conversation in one click.

## Getting started

**Requirements:** Python 3.10+, a microphone and speakers, a Google Cloud project with Speech-to-Text and Text-to-Speech enabled, a service-account JSON key, and a Hugging Face access token. PyAudio needs the PortAudio library on your system (`brew install portaudio` on macOS, `sudo apt install portaudio19-dev` on Debian/Ubuntu; the Windows wheel includes it).

### 1. Install

```bash
git clone https://github.com/vldzio/conversational-agent.git
cd conversational-agent
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure credentials

VoiceFlow reads two environment variables (see `.env.example`):

| Variable | Value |
| --- | --- |
| `HUGGINGFACEHUB_API_TOKEN` | Your Hugging Face access token |
| `GOOGLE_APPLICATION_CREDENTIALS` | Absolute path to your Google Cloud service-account JSON key |

macOS / Linux:

```bash
export HUGGINGFACEHUB_API_TOKEN="hf_..."
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
```

Windows PowerShell:

```powershell
$env:HUGGINGFACEHUB_API_TOKEN = "hf_..."
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"
```

### 3. Run

```bash
streamlit run streamlit_ui.py
```

Enter a phone number, click **Simulate Call**, and speak your order when prompted (for example, “I'd like to order two wireless chargers”). For a terminal-only run of the same pipeline, use `python tester.py`.

**Notes:** audio is recorded and played on the machine running Streamlit. Confirmed orders write back to `inventory.csv`, so keep a copy if you want to reset stock levels between runs.

## Project structure

| File | Responsibility |
| --- | --- |
| `streamlit_ui.py` | Simulated call UI; orchestrates audio → agent → speech |
| `audiotranscriber.py` | PyAudio capture and Google Speech-to-Text transcription |
| `model.py` | Hugging Face endpoint, request parsing, CSV-backed inventory action |
| `tts_for_model.py` | Google Cloud Text-to-Speech synthesis to `output.mp3` |
| `inventory.csv` | Example products and stock counts |
| `tester.py` | Command-line audio → agent → speech run |

## Roadmap

- Browser-side audio capture
- Telephony integration for real calls
- Multi-turn conversations and order adjustments
