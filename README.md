
# Phone Call Conversational AI

## Overview
This project is a proof-of-concept (PoC) conversational AI system that simulates phone calls. It integrates:
- A **conversational agent** built using LangChain and Hugging Face models.
- **Speech-to-Text (STT)** using Google Cloud Speech-to-Text for transcribing user input.
- **Text-to-Speech (TTS)** using Google Cloud Text-to-Speech for generating responses.
- **A Streamlit-based interface** that simulates phone call interactions.

The application allows a user to enter a phone number, simulate a call, input speech, and receive AI-generated responses played back as audio.

---

## Conversational Agent
The conversational AI agent is built using:
- **LangChain**: Provides structured agent reasoning and tool usage.
- **Hugging Face Inference API**: Uses the `Mixtral-8x7B-Instruct-v0.1` model to generate responses based on user input.
- **LangGraph with ReAct paradigm**: Allows dynamic decision-making in conversations.

The agent processes user queries by:
1. Receiving transcribed text from the STT system.
2. Checking inventory data from a CSV-based database.
3. Confirming or denying orders based on stock availability.
4. Suggesting alternatives for unavailable items.
5. Responding with relevant information.
6. Passing the response to the TTS system for audio playback.

---

## System Logic
### 1. Speech-to-Text (STT)
- Uses Google Cloud Speech-to-Text.
- Records user speech using `pyaudio`.
- Converts the recorded speech into text and passes it to the AI agent.

### 2. AI Response Generation
- The transcribed text is processed by the LangChain-powered agent.
- The agent retrieves information and formulates a response.

### 3. Text-to-Speech (TTS)
- Uses Google Cloud Text-to-Speech.
- Converts the AI response into speech.
- Saves the generated speech as an `MP3` file.

### 4. Streamlit UI
- Provides a simple interface to:
  - Enter a phone number.
  - Simulate a call.
  - Record user speech.
  - Play AI-generated responses.

---

## Setup and Installation
### Prerequisites
- Python 3.10+
- Google Cloud account with STT and TTS enabled
- Hugging Face account with API access

### Installation
1. Clone the repository:
   ```sh
   git clone https://https://github.com/vunderscore/conversational_call_agent.git
   cd conversational_call_agent
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Set up Google Cloud credentials:
   - Obtain a service account JSON key from Google Cloud.
   - Save it in the project directory.
   - Set the environment variable:
     ```sh
     export GOOGLE_APPLICATION_CREDENTIALS="your-key.json"
     ```

4. Configure Hugging Face API access:
   - Set up your Hugging Face token:
     ```sh
     huggingface-cli login
     ```

---

## Running the Project
To start the Streamlit interface:
```sh
streamlit run streamlit_ui.py
```
This will launch a web interface where users can enter a number, simulate a call, and interact with the AI.

---

## File Structure
```
├── agent.py                 # Conversational AI agent logic
├── audiotranscriber.py       # Handles STT using Google Cloud
├── tts_for_model.py         # Handles TTS using Google Cloud
├── streamlit_ui.py          # Streamlit interface for interaction
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
```

---

## Future Improvements
- Implement real phone calling features with Twilio or other VoIP providers.
- Improve real-time audio streaming instead of saving files.
- Enhance agent capabilities for more complex sales interactions.

