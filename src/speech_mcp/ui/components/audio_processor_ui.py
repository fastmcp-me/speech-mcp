"""
Audio processor UI wrapper for the Speech UI.

This module provides a PyQt wrapper around the AudioProcessor for speech recognition.
"""

import os
import time
import logging
import threading
from PyQt5.QtCore import QObject, pyqtSignal

# Import centralized constants
from speech_mcp.constants import TRANSCRIPTION_FILE

# Import shared audio processor and speech recognition
from speech_mcp.audio_processor import AudioProcessor
from speech_mcp.speech_recognition import SpeechRecognizer

# Setup logging
logger = logging.getLogger(__name__)

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
            logger.info("Initializing speech recognition...")
            
            # Create a speech recognizer instance
            self.speech_recognizer = SpeechRecognizer(model_name="base", device="cpu", compute_type="int8")
            
            if self.speech_recognizer.is_initialized:
                logger.info("Speech recognition initialized successfully")
            else:
                logger.warning("Speech recognition initialization may have failed")
                
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
    
    def start_listening(self):
        """Start listening for audio input."""
        if self.is_listening:
            return
            
        self.is_listening = True
        
        # Start the shared audio processor
        if not self.audio_processor.start_listening():
            logger.error("Failed to start audio processor")
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
        except Exception as e:
            logger.error(f"Error in _listen_and_process: {e}")
            self.is_listening = False
    
    def process_recording(self):
        """Process the recorded audio and generate a transcription"""
        try:
            # Get the recorded audio file path
            temp_audio_path = self.audio_processor.get_recorded_audio_path()
            
            if not temp_audio_path:
                logger.warning("No audio data to process")
                return
            
            logger.info(f"Processing audio file: {temp_audio_path}")
            
            # Use the speech recognizer to transcribe the audio
            if self.speech_recognizer and self.speech_recognizer.is_initialized:
                logger.info("Transcribing audio with speech recognizer...")
                
                transcription, metadata = self.speech_recognizer.transcribe(temp_audio_path)
                
                # Log the transcription details
                logger.info(f"Transcription completed: {transcription}")
                logger.debug(f"Transcription metadata: {metadata}")
            else:
                logger.error("Speech recognizer not initialized")
                transcription = "Error: Speech recognition not initialized"
            
            # Clean up the temporary file
            try:
                logger.debug(f"Removing temporary WAV file: {temp_audio_path}")
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")
            
            # Write the transcription to a file for the server to read
            try:
                logger.debug(f"Writing transcription to file: {TRANSCRIPTION_FILE}")
                with open(TRANSCRIPTION_FILE, 'w') as f:
                    f.write(transcription)
                logger.debug("Transcription file written successfully")
            except Exception as e:
                logger.error(f"Error writing transcription to file: {e}")
            
            # Emit the transcription signal
            self.transcription_ready.emit(transcription)
            
        except Exception as e:
            logger.error(f"Error processing recording: {e}")
            self.transcription_ready.emit(f"Error processing speech: {str(e)}")
    
    def stop_listening(self):
        """Stop listening for audio input."""
        try:
            logger.info("Stopping audio recording")
            self.audio_processor.stop_listening()
            self.is_listening = False
            
        except Exception as e:
            logger.error(f"Error stopping audio recording: {e}")
            self.is_listening = False