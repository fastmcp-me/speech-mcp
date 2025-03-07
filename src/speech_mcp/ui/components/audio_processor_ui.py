"""
Audio processor UI wrapper for the Speech UI.

This module provides a PyQt wrapper around the AudioProcessor for speech recognition.
"""

import os
import time
import threading
from PyQt5.QtCore import QObject, pyqtSignal

# Import centralized constants
from speech_mcp.constants import TRANSCRIPTION_FILE

# Import shared audio processor and speech recognition
from speech_mcp.audio_processor import AudioProcessor
from speech_mcp.speech_recognition import SpeechRecognizer

class AudioProcessorUI(QObject):
    """
    UI wrapper for AudioProcessor that handles speech recognition.
    """
    audio_level_updated = pyqtSignal(float)
    transcription_ready = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_listening = False
        self.speech_recognizer = None
        
        # Create the shared AudioProcessor with a callback for audio levels
        self.audio_processor = AudioProcessor(on_audio_level=self._on_audio_level)
        
        # Initialize speech recognition in a background thread
        threading.Thread(target=self._initialize_speech_recognition, daemon=True).start()
    
    def _on_audio_level(self, level):
        """Callback for audio level updates from the AudioProcessor"""
        self.audio_level_updated.emit(level)
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition in a background thread"""
        try:
            # Create a speech recognizer instance
            self.speech_recognizer = SpeechRecognizer(model_name="base", device="cpu", compute_type="int8")
        except Exception:
            pass
    
    def start_listening(self):
        """Start listening for audio input."""
        if self.is_listening:
            return
            
        self.is_listening = True
        
        # Start the shared audio processor
        if not self.audio_processor.start_listening():
            self.is_listening = False
            return
        
        # Start a thread to detect silence and stop recording
        threading.Thread(target=self._listen_and_process, daemon=True).start()
    
    def _listen_and_process(self):
        """Thread function that waits for audio processor to finish and then processes the recording"""
        try:
            # Wait for the audio processor to finish recording
            while self.audio_processor.is_listening:
                time.sleep(0.1)
            
            # Process the recording if we're still in listening mode
            if self.is_listening:
                self.process_recording()
                self.is_listening = False
        except Exception:
            self.is_listening = False
    
    def process_recording(self):
        """Process the recorded audio and generate a transcription"""
        try:
            # Get the recorded audio file path
            temp_audio_path = self.audio_processor.get_recorded_audio_path()
            
            if not temp_audio_path:
                return
            
            # Use the speech recognizer to transcribe the audio
            if self.speech_recognizer and self.speech_recognizer.is_initialized:
                transcription, metadata = self.speech_recognizer.transcribe(temp_audio_path)
            else:
                transcription = "Error: Speech recognition not initialized"
            
            # Clean up the temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass
            
            # Write the transcription to a file for the server to read
            try:
                with open(TRANSCRIPTION_FILE, 'w') as f:
                    f.write(transcription)
            except Exception:
                pass
            
            # Emit the transcription signal
            self.transcription_ready.emit(transcription)
            
        except Exception as e:
            self.transcription_ready.emit(f"Error processing speech: {str(e)}")
    
    def stop_listening(self):
        """Stop listening for audio input."""
        try:
            self.audio_processor.stop_listening()
            self.is_listening = False
            
        except Exception:
            self.is_listening = False