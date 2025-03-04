#!/usr/bin/env python3
"""
Test script for Kokoro TTS adapter

This script tests the Kokoro TTS adapter for speech-mcp.
It attempts to initialize the adapter and speak some text.
"""

import os
import sys
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the speech_mcp package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print("=== Kokoro TTS Adapter Test ===")
    
    try:
        # Try to import the Kokoro adapter
        print("Importing Kokoro adapter...")
        from speech_mcp.tts_adapters.kokoro_adapter import KokoroTTS
        
        # Initialize the adapter
        print("Initializing Kokoro TTS adapter...")
        tts = KokoroTTS()
        
        # Check if Kokoro is available
        if tts.kokoro_available:
            print("Kokoro TTS is available!")
            print(f"Using voice: {tts.voice}")
            print(f"Using language code: {tts.lang_code}")
            print(f"Using speed: {tts.speed}")
            
            # Get available voices
            voices = tts.get_available_voices()
            print(f"Available voices: {', '.join(voices)}")
            
            # Speak some text
            print("\nSpeaking test text...")
            test_text = "Hello! This is a test of the Kokoro text-to-speech system. I hope you can hear me clearly."
            tts.speak(test_text)
            
            print("\nTest completed successfully!")
        else:
            print("Kokoro TTS is not available. Using fallback.")
            
            # Check if pyttsx3 fallback is available
            if tts.pyttsx3_engine is not None:
                print("pyttsx3 fallback is available.")
                
                # Speak some text using the fallback
                print("\nSpeaking test text using pyttsx3 fallback...")
                test_text = "Hello! This is a test of the pyttsx3 fallback text-to-speech system."
                tts.speak(test_text)
                
                print("\nTest completed successfully with fallback!")
            else:
                print("No text-to-speech engines are available. Test failed.")
                return 1
    except ImportError as e:
        print(f"Error importing Kokoro adapter: {e}")
        print("Make sure you have installed the speech-mcp package.")
        return 1
    except Exception as e:
        print(f"Error testing Kokoro adapter: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())