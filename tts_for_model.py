import os
from google.cloud import texttospeech
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-json-file-path"




def text_to_speech(text, filename="output.mp3"):
    # Ensure the file is saved in the current working directory
    output_path = os.path.join(os.getcwd(), filename)

    # 🔥 Fix: Delete existing file to avoid permission issues


    if os.path.exists(output_path):
        try:
            os.remove(output_path)  # Remove old file
        except PermissionError:
            time.sleep(1)  # Wait a moment and retry
            os.remove(output_path)  # Try again

    # Initialize client
    client = texttospeech.TextToSpeechClient()

    # Configure text input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Use a more natural-sounding WaveNet voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",   
        name="en-US-Wavenet-D",  # Change this for different voices
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Configure audio settings
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.1,  # Adjust speed (1.0 is normal)
        pitch=-4           # Adjust pitch (-20.0 to 20.0)
    )

    # Generate speech
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # Save the file in the current directory
    with open(output_path, "wb") as out:
        out.write(response.audio_content)

    print(f"✅ Audio saved at: {output_path}")

# Example usage
text_to_speech("Hello! This is a test of Google Cloud Text-to-Speech with a more natural voice.")
