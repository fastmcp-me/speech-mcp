"""
Test script for the speech recognition module.

This script tests the functionality of the speech_recognition.py module.
"""

import os
import sys
import logging
import time

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the speech recognition module
from speech_mcp.speech_recognition import SpeechRecognizer, transcribe_audio, initialize_speech_recognition

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_speech_recognizer_initialization():
    """Test initializing the SpeechRecognizer class"""
    logger.info("Testing SpeechRecognizer initialization")
    
    # Create a SpeechRecognizer instance
    recognizer = SpeechRecognizer(model_name="base", device="cpu", compute_type="int8")
    
    # Check if initialization was successful
    assert recognizer.is_initialized, "SpeechRecognizer initialization failed"
    logger.info("SpeechRecognizer initialization successful")
    
    # Check the current model
    current_model = recognizer.get_current_model()
    assert current_model["name"] == "base", f"Expected model 'base', got {current_model['name']}"
    assert current_model["engine"] == "faster-whisper", f"Expected engine 'faster-whisper', got {current_model['engine']}"
    logger.info(f"Current model: {current_model}")
    
    # Check available models
    available_models = recognizer.get_available_models()
    assert len(available_models) > 0, "No available models found"
    logger.info(f"Found {len(available_models)} available models")
    
    return recognizer

def test_global_functions():
    """Test the global functions in the speech_recognition module"""
    logger.info("Testing global functions")
    
    # Test initialize_speech_recognition
    result = initialize_speech_recognition(model_name="base", device="cpu", compute_type="int8")
    assert result, "initialize_speech_recognition failed"
    logger.info("initialize_speech_recognition successful")
    
    # Test transcribe_audio with a non-existent file (should return empty string)
    transcription = transcribe_audio("non_existent_file.wav")
    assert transcription == "", "Expected empty transcription for non-existent file"
    logger.info("transcribe_audio with non-existent file test passed")

def main():
    """Main test function"""
    logger.info("Starting speech recognition tests")
    
    try:
        # Test SpeechRecognizer initialization
        recognizer = test_speech_recognizer_initialization()
        
        # Test global functions
        test_global_functions()
        
        logger.info("All tests passed!")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())