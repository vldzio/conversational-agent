# VoiceFlow — conversational inventory demo

**A local, voice-driven sales-assistant proof of concept.** VoiceFlow records a short microphone utterance, transcribes it with Google Cloud Speech-to-Text, uses a Hugging Face-hosted language model to extract the requested product and quantity, checks a CSV inventory, and speaks the resulting answer with Google Cloud Text-to-Speech. A Streamlit interface ties the interaction together as a **simulated call**.

The interesting split is between **language understanding** and **deterministic action**: the LLM interprets the customer's words, while Python performs the stock check and inventory update.

```text
Local microphone (5-second recording)
      ↓ Google Cloud Speech-to-Text
Transcript
      ↓ Hugging Face Mixtral endpoint: product + quantity JSON
CSV inventory check → stock/alternative/confirmation response
      ↓ Google Cloud Text-to-Speech
MP3 playback via pygame on the machine running Streamlit
```

## Capabilities

- Captures short spoken requests and transcribes them to English text.
- Parses requests such as “I want two wireless earbuds” into product and quantity.
- Checks `inventory.csv`; returns out-of-stock, insufficient-stock, or confirmation messages. Confirmed requests update the CSV's quantity-on-hand and quantity-sold fields.
- Synthesizes the response as an MP3 using a Google Cloud WaveNet voice and plays it locally.
- Offers a Streamlit **Simulate Call** UI; the phone-number field is a simulation cue, not a real telephone connection.

## Running the current code locally

**Requirements:** Python 3.10+, a local microphone and audio output device, Google Cloud Speech-to-Text and Text-to-Speech access, a service-account JSON key, and access to the configured Hugging Face inference model. API use may incur charges.

**A small configuration cleanup is required before the checked-in code can run with your credentials:**

1. In `model.py`, replace the literal `"your-access-token"` assignment with `os.environ["HUGGINGFACEHUB_API_TOKEN"]`. The current line overwrites any token you set in the shell.
2. In `streamlit_ui.py`, replace the hardcoded service-account JSON filename passed to `get_speech_client(...)` with `os.environ["GOOGLE_APPLICATION_CREDENTIALS"]`. Make the same change in `audiotranscriber.py` if you run that file directly.
3. In `tts_for_model.py`, remove the line that overwrites `GOOGLE_APPLICATION_CREDENTIALS` with a placeholder. Move the example `text_to_speech(...)` call at the bottom under `if __name__ == "__main__":` (or remove it), so importing the module does not call Google TTS before the UI starts.
4. The repository's existing `requirements.txt` contains standard-library modules (`wave`, `threading`), omits `pygame` and `langchain-huggingface`, and includes package names that may not resolve. Correct it before using `pip install -r requirements.txt`. For the imports used by this code, a starting point is:

   ```bash
   python -m pip install streamlit pyaudio pygame pandas langchain langchain-huggingface huggingface-hub google-cloud-speech google-cloud-texttospeech google-auth
   ```

   PyAudio may also require the native PortAudio development library on your OS. Dependency versions are not pinned here; if current provider libraries change APIs, pin a tested environment.

After those edits, run from the repository root:

```bash
git clone https://github.com/vldzio/conversational-agent.git
cd conversational-agent
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Install the corrected dependencies as noted above.
export HUGGINGFACEHUB_API_TOKEN="your_hugging_face_token"
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
streamlit run streamlit_ui.py
```

On Windows PowerShell, set environment variables using `$env:HUGGINGFACEHUB_API_TOKEN="..."` and `$env:GOOGLE_APPLICATION_CREDENTIALS="C:\\path\\to\\service-account.json"` instead of `export`.

**Important:** the microphone recording and `pygame` playback happen on the computer running the Python server. Opening the page from another computer does **not** turn that browser into a phone or stream its microphone to the server. Keep service-account JSON and API tokens out of Git. Save a copy of `inventory.csv` before testing because confirmed requests modify it.

## How the code is organized

| File | Responsibility |
| --- | --- |
| `streamlit_ui.py` | Simulated call UI, orchestrates audio → agent → speech |
| `audiotranscriber.py` | Local PyAudio capture and Google STT transcription |
| `model.py` | Hugging Face endpoint, request parsing, CSV-backed inventory action |
| `tts_for_model.py` | Google Cloud TTS synthesis to `output.mp3` |
| `inventory.csv` | Example products and mutable stock counts |
| `tester.py` | Local command-line audio → agent → speech test |

## Scope and next steps

VoiceFlow shows how a voice interface can connect model-assisted request parsing with an explicit inventory operation. The checked-in implementation uses a **single LLM parsing call and ordinary Python functions**, not a LangGraph/ReAct agent; it simulates calls rather than placing them through a telephony provider. Useful next steps are browser-side audio capture, a real telephony integration, stronger validation before decrementing stock, credential configuration, and reproducible dependency pins.
