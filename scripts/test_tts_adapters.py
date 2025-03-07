#!/usr/bin/env python3
"""
Test script for TTS adapters

This script tests the TTS adapters for speech-mcp.
It attempts to initialize the adapters and speak some text.
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

def test_tts_adapter(adapter_name):
    """Test a specific TTS adapter"""
    print(f"\n=== {adapter_name} TTS Adapter Test ===")
    
    try:
        # Try to import the adapter
        print(f"Importing {adapter_name} adapter...")
        
        if adapter_name == "Kokoro":
            from speech_mcp.tts_adapters import KokoroTTS
            tts = KokoroTTS()
        elif adapter_name == "Pyttsx3":
            from speech_mcp.tts_adapters import Pyttsx3TTS
            tts = Pyttsx3TTS()
        else:
            print(f"Unknown adapter: {adapter_name}")
            return False
        
        # Check if adapter is initialized
        if tts.is_initialized:
            print(f"{adapter_name} TTS is initialized!")
            print(f"Using voice: {tts.voice}")
            print(f"Using language code: {tts.lang_code}")
            print(f"Using speed: {tts.speed}")
            
            # Get available voices
            voices = tts.get_available_voices()
            print(f"Available voices: {len(voices)}")
            print(f"Voice examples: {', '.join(voices[:5])}" + ("..." if len(voices) > 5 else ""))
            
            # Speak some text
            print(f"\nSpeaking test text using {adapter_name}...")
            test_text = f"Hello! This is a test of the {adapter_name} text-to-speech system. I hope you can hear me clearly."
            result = tts.speak(test_text)
            
            if result:
                print(f"\n{adapter_name} test completed successfully!")
                return True
            else:
                print(f"\n{adapter_name} speak method returned False. Test failed.")
                return False
        else:
            print(f"{adapter_name} TTS is not initialized. Test failed.")
            return False
    except ImportError as e:
        print(f"Error importing {adapter_name} adapter: {e}")
        print("Make sure you have installed the speech-mcp package with the required dependencies.")
        return False
    except Exception as e:
        print(f"Error testing {adapter_name} adapter: {e}")
        return False

def main():
    print("=== TTS Adapters Test ===")
    
    # Test Kokoro adapter first (primary)
    kokoro_success = test_tts_adapter("Kokoro")
    
    # Test Pyttsx3 adapter if Kokoro failed
    if not kokoro_success:
        print("\nKokoro test failed or not available. Testing Pyttsx3 adapter...")
        pyttsx3_success = test_tts_adapter("Pyttsx3")
        
        if pyttsx3_success:
            print("\nPyttsx3 adapter test successful!")
            return 0
        else:
            print("\nBoth Kokoro and Pyttsx3 adapters failed. No TTS available.")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())