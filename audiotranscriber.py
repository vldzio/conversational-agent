import os
import pyaudio
import wave
import threading
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
import time

def get_speech_client(json_path='your-json-file-path'):
    """
    Get Speech Client using local JSON key
    """
    credentials = service_account.Credentials.from_service_account_file(
        json_path
    )
    return speech.SpeechClient(credentials=credentials)

class AudioRecorder:
    def __init__(self, 
                 chunk=1024, 
                 format=pyaudio.paInt16, 
                 channels=1, 
                 rate=16000, 
                 max_record_seconds=10,
                 output_filename="output.wav"):
        """
        Initialize audio recorder
        """
        self.CHUNK = chunk
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        self.MAX_RECORD_SECONDS = max_record_seconds
        self.WAVE_OUTPUT_FILENAME = output_filename
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

    def record_audio(self):
        """
        Record audio from microphone and save to file
        """
        self.frames = []
        self.is_recording = True

        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        print("* Recording audio...")
        
        for _ in range(0, int(self.RATE / self.CHUNK * self.MAX_RECORD_SECONDS)):
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            self.frames.append(data)
        
        self.stream.stop_stream()
        self.stream.close()
        self.is_recording = False

        print("* Done recording")

        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        return self.WAVE_OUTPUT_FILENAME

def transcribe_audio(client, audio_file):
    """
    Transcribe audio file using Google Speech-to-Text
    """
    with open(audio_file, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    
    response = client.recognize(config=config, audio=audio)
    
    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)
    
    return " ".join(transcripts)

def main():
    json_path = 'astral-trees-452808-p1-203f79031f31.json'
    client = get_speech_client(json_path)
    
    recorder = AudioRecorder(
        max_record_seconds=10,  
        output_filename="output.wav"  
    )
    audio_file = recorder.record_audio()
    
    transcript = transcribe_audio(client, audio_file)
    
    print("\nTranscript:")
    print(transcript)

if __name__ == "__main__":
    main()
