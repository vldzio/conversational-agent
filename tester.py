
import audiotranscriber
from model import agent
from tts_for_model import text_to_speech

def main():
    # Initialize Google Speech Client
    client = audiotranscriber.get_speech_client('astral-trees-452808-p1-203f79031f31.json')

    # Initialize and record audio
    recorder = audiotranscriber.AudioRecorder(
        max_record_seconds=5,  # Record for 5 seconds
        output_filename="test_output.wav"
    )
    audio_file = recorder.record_audio()

    # Transcribe the recorded audio
    transcript = audiotranscriber.transcribe_audio(client, audio_file)
    response = agent(transcript)
    print(response)
    text_to_speech(response)

if __name__ == "__main__":
    main()