"""
Test script for the audio processor module.

This script demonstrates how to use the AudioProcessor class directly.
"""

import sys
import time
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to allow importing speech_mcp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the AudioProcessor
from speech_mcp.audio_processor import AudioProcessor

def on_audio_level(level):
    """Callback function for audio level updates"""
    # Print a simple visualization of the audio level
    bars = int(level * 50)
    print(f"\r[{'#' * bars}{' ' * (50 - bars)}] {level:.2f}", end="")

def main():
    """Main test function"""
    print("Testing AudioProcessor...")
    
    # Create an AudioProcessor with our callback
    processor = AudioProcessor(on_audio_level=on_audio_level)
    
    print("\nAvailable audio devices:")
    devices = processor.get_available_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} ({device['channels']} channels, {device['sample_rate']} Hz)")
    
    print("\nRecording audio for 5 seconds or until silence is detected...")
    print("Speak now!")
    
    # Record audio
    audio_file = processor.record_audio()
    
    if audio_file:
        print(f"\nAudio saved to: {audio_file}")
        
        # Play back the recorded audio
        print("Playing back recorded audio...")
        processor.play_audio_file(audio_file)
        
        # Clean up the audio file
        try:
            os.unlink(audio_file)
            print(f"Deleted temporary audio file: {audio_file}")
        except Exception as e:
            print(f"Error deleting audio file: {e}")
    else:
        print("Failed to record audio")
    
    # Clean up resources
    processor.cleanup()
    print("Test complete")

if __name__ == "__main__":
    main()