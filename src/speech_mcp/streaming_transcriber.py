"""
StreamingTranscriber class for real-time speech transcription using faster-whisper.

This module provides a streaming interface to the faster-whisper speech recognition
system, enabling real-time transcription with word-level timestamps and natural
end-of-speech detection based on word timing patterns.
"""

from typing import Optional, Callable, Dict, Any, Tuple
import threading
import queue
import numpy as np
from faster_whisper import WhisperModel
import time

# Import the centralized logger
from speech_mcp.utils.logger import get_logger
from speech_mcp.constants import (
    STREAMING_END_SILENCE_DURATION,
    STREAMING_INITIAL_WAIT
)

# Get a logger for this module
logger = get_logger(__name__, component="stt")

class StreamingTranscriber:
    """
    Handles real-time streaming transcription using faster-whisper.
    
    This class manages a continuous audio stream, processing chunks of audio data
    as they arrive and providing both partial and final transcriptions. It uses
    word-level timing information to detect natural speech boundaries rather than
    relying on simple silence detection.
    """
    
    def __init__(self, 
                 model_name: str = "base", 
                 device: str = "cpu", 
                 compute_type: str = "int8",
                 language: str = "en",
                 on_partial_transcription: Optional[Callable[[str], None]] = None,
                 on_final_transcription: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        """
        Initialize the StreamingTranscriber.
        
        Args:
            model_name: The name/size of the Whisper model to use (e.g., "base", "small", "medium")
            device: Device to use for computation ("cpu" or "cuda")
            compute_type: Model computation type (e.g., "int8", "float16")
            language: Language code for transcription (e.g., "en" for English)
            on_partial_transcription: Callback for partial transcription updates
            on_final_transcription: Callback for final transcription with metadata
        """
        # Initialize attributes
        self.language = language
        self.on_partial_transcription = on_partial_transcription
        self.on_final_transcription = on_final_transcription
        
        # Audio processing attributes
        self._audio_queue = queue.Queue()
        self._audio_buffer = []
        self._current_transcription = ""
        self._accumulated_transcription = ""  # New: Store all transcribed segments
        self._last_word_time = 0.0
        self._last_word_detected = time.time()
        self._stream_start_time = 0.0  # New: Track when streaming started
        self._is_active = False
        self._processing_thread = None
        self._lock = threading.Lock()
        
        # Load the model
        try:
            logger.info(f"Loading faster-whisper model '{model_name}' on {device}")
            self.model = WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {str(e)}")
            raise
    
    def start_streaming(self) -> bool:
        """
        Start processing the audio stream.
        
        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        if self._is_active:
            logger.warning("Streaming is already active")
            return False
            
        try:
            # Clear any existing data
            self._audio_queue = queue.Queue()
            self._audio_buffer = []
            self._current_transcription = ""
            self._accumulated_transcription = ""  # Reset accumulated transcription
            self._last_word_time = 0.0
            self._last_word_detected = time.time()
            self._stream_start_time = time.time()  # Set stream start time
            
            # Start the processing thread
            self._is_active = True
            self._processing_thread = threading.Thread(
                target=self._process_audio_stream,
                daemon=True
            )
            self._processing_thread.start()
            
            logger.info("Streaming started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {str(e)}")
            self._is_active = False
            return False
    
    def add_audio_chunk(self, audio_chunk: bytes) -> None:
        """
        Add an audio chunk to the processing queue.
        
        Args:
            audio_chunk: Raw audio data as bytes (assumed to be 16-bit PCM)
        """
        if self._is_active:
            self._audio_queue.put(audio_chunk)
    
    def _process_audio_stream(self) -> None:
        """
        Background thread function to process the audio stream.
        
        This continuously processes audio chunks from the queue and updates
        the transcription when enough data is available.
        """
        while self._is_active:
            try:
                # Get audio chunk with timeout to allow checking _is_active
                chunk = self._audio_queue.get(timeout=0.1)
                
                # Convert bytes to numpy array and append to buffer
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                self._audio_buffer.append(audio_data)
                
                # Process buffer when it reaches sufficient size
                # Using 20 chunks = ~1 second of audio at 16kHz
                if len(self._audio_buffer) >= 20:
                    logger.debug(f"Processing buffer with {len(self._audio_buffer)} chunks")
                    self._transcribe_buffer()
                    
                # Check if we're still in initial wait period
                time_since_start = time.time() - self._stream_start_time
                if time_since_start < STREAMING_INITIAL_WAIT:
                    continue
                    
                # Check for end of speech based on word timing
                current_time = time.time()
                time_since_last_word = current_time - self._last_word_detected
                if time_since_last_word > STREAMING_END_SILENCE_DURATION:
                    logger.info(f"No new words detected for {time_since_last_word:.1f} seconds, stopping")
                    # Process any remaining audio and notify through callbacks
                    self.stop_streaming()
                    break
                    
            except queue.Empty:
                # No new audio data, but process buffer if we have enough
                if len(self._audio_buffer) >= 10:
                    logger.debug(f"Processing buffer during quiet period: {len(self._audio_buffer)} chunks")
                    self._transcribe_buffer()
                continue
            except Exception as e:
                logger.error(f"Error processing audio stream: {str(e)}")
                # Continue processing despite errors
                continue
    
    def _transcribe_buffer(self) -> None:
        """
        Transcribe the current audio buffer and update transcription.
        """
        try:
            # Combine audio chunks
            audio_data = np.concatenate(self._audio_buffer)
            
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe using faster-whisper with adjusted thresholds
            segments, info = self.model.transcribe(
                audio_float,
                language=self.language,
                word_timestamps=True,
                # Use more sensitive detection settings
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters={"threshold": 0.3}  # More sensitive VAD
            )
            
            # Log speech detection info
            logger.debug(f"Speech detection info: {info}")
            
            # Process segments and update transcription
            with self._lock:
                new_text = ""
                for segment in segments:
                    new_text += segment.text
                    logger.debug(f"Segment detected: {segment.text}")
                    
                    # Update last word time if words are available
                    if segment.words:
                        # Log each word with its timing
                        for word in segment.words:
                            logger.debug(f"Word: {word.word}, Start: {word.start:.2f}s, End: {word.end:.2f}s")
                        self._last_word_time = segment.words[-1].end
                        self._last_word_detected = time.time()
                        logger.debug(f"Updated last word time to {self._last_word_time:.2f}s")
                
                if new_text:
                    # Append to accumulated transcription and update current
                    self._accumulated_transcription += " " + new_text.strip()
                    self._accumulated_transcription = self._accumulated_transcription.strip()
                    self._current_transcription = self._accumulated_transcription
                    
                    logger.info(f"Updated transcription: {self._current_transcription}")
                    
                    # Call partial transcription callback if provided
                    if self.on_partial_transcription:
                        self.on_partial_transcription(self._current_transcription)
            
            # Clear the buffer after processing
            self._audio_buffer = []
            
        except Exception as e:
            logger.error(f"Error transcribing buffer: {str(e)}")
            if hasattr(e, 'args') and len(e.args) > 0:
                logger.error(f"Error details: {e.args[0]}")
    
    def stop_streaming(self) -> Tuple[str, Dict[str, Any]]:
        """
        Stop streaming and return the final transcription.
        
        Returns:
            Tuple containing:
            - Final transcription text (str)
            - Metadata dictionary with timing information
        """
        if not self._is_active:
            return "", {}
            
        try:
            # Set flag to stop processing thread
            self._is_active = False
            
            # Wait for processing thread to finish if it's not the current thread
            if self._processing_thread and self._processing_thread is not threading.current_thread():
                self._processing_thread.join(timeout=5.0)
            
            # Process any remaining audio in the buffer
            if self._audio_buffer:
                self._transcribe_buffer()
            
            # Get final transcription and metadata
            with self._lock:
                final_text = self._current_transcription
                metadata = {
                    "last_word_time": self._last_word_time,
                    "language": self.language,
                    "time_since_last_word": time.time() - self._last_word_detected
                }
            
            # Call final transcription callback if provided
            if self.on_final_transcription:
                self.on_final_transcription(final_text, metadata)
            
            logger.info("Streaming stopped successfully")
            return final_text, metadata
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {str(e)}")
            return "", {}
        finally:
            # Ensure we clean up
            self._audio_queue = queue.Queue()
            self._audio_buffer = []
            self._current_transcription = ""
            self._last_word_time = 0.0
            self._last_word_detected = time.time()
    
    def get_current_transcription(self) -> str:
        """
        Get the current partial transcription.
        
        Returns:
            str: Current transcription text
        """
        with self._lock:
            return self._current_transcription
    
    def is_active(self) -> bool:
        """
        Check if streaming is currently active.
        
        Returns:
            bool: True if streaming is active, False otherwise
        """
        return self._is_active