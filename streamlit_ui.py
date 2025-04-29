import streamlit as st
import os
import audiotranscriber
from model import agent
from tts_for_model import text_to_speech
import pygame

def play_audio(file_path):
    """Play the generated audio file."""
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

def main():
    st.title("Fake Sales Call Simulator")
    
    phone_number = st.text_input("Enter phone number:", "")
    
    if st.button("Simulate Call"):
        if phone_number:
            st.write("Calling...")
            
            # Initialize STT Client
            client = audiotranscriber.get_speech_client('astral-trees-452808-p1-203f79031f31.json')
            
            # Record audio
            st.write("Listening for input...")
            recorder = audiotranscriber.AudioRecorder(max_record_seconds=5, output_filename="test_output.wav")
            audio_file = recorder.record_audio()
            
            # Transcribe audio
            transcript = audiotranscriber.transcribe_audio(client, audio_file)
            st.write(f"User said: {transcript}")
            
            # Get agent response
            response = agent(transcript)
            st.write(f"Agent Response: {response}")
            
            # Convert text to speech
            text_to_speech(response)
            
            # Play response
            st.write("Playing response...")
            play_audio("output.mp3")
        else:
            st.warning("Please enter a valid phone number.")

main()
