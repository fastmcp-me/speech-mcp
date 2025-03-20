"""
Speech recognition module for speech-mcp.

This module provides centralized speech recognition functionality including:
- Model loading and initialization
- Audio transcription (file-based and streaming)
- Fallback mechanisms
- Consistent error handling

It consolidates speech recognition code that was previously duplicated
across server.py and speech_ui.py.
"""

# Import the streaming transcriber
from speech_mcp.streaming_transcriber import StreamingTranscriber

import os
import time
from datetime import timedelta, datetime
from typing import Optional, Tuple, Dict, Any, List, Union, Callable

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="stt")

class SpeechRecognizer:
    """
    Core speech recognition class that handles transcription of audio files.
    
    This class provides a unified interface for speech recognition with fallback mechanisms.
    It supports multiple speech recognition engines, with faster-whisper as the primary engine
    and SpeechRecognition as a fallback.
    """
    
    def __init__(self, model_name: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the speech recognizer.
        
        Args:
            model_name: The name of the faster-whisper model to use (e.g., "base", "small", "medium")
            device: The device to use for inference ("cpu" or "cuda")
            compute_type: The compute type to use for inference ("int8", "float16", "float32")
        """
        self.whisper_model = None
        self.sr_recognizer = None
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.is_initialized = False
        
        # Add streaming transcriber
        self.streaming_transcriber = None
        
        # Initialize the speech recognition models in the background
        self._initialize_speech_recognition()
    
    def _initialize_speech_recognition(self) -> bool:
        """
        Initialize speech recognition models.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.is_initialized:
            logger.info("Speech recognition already initialized")
            return True
        
        # Try to initialize faster-whisper first
        try:
            logger.info(f"Loading faster-whisper speech recognition model '{self.model_name}' on {self.device}...")
            
            import faster_whisper
            # Load the model with the specified parameters
            self.whisper_model = faster_whisper.WhisperModel(
                self.model_name, 
                device=self.device, 
                compute_type=self.compute_type
            )
            
            logger.info("faster-whisper model loaded successfully!")
            
            self.is_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Failed to load faster-whisper: {e}")
            logger.info("Trying to fall back to SpeechRecognition library...")
            
            return self._initialize_speech_recognition_fallback()
        except Exception as e:
            logger.error(f"Error initializing faster-whisper: {e}")
            logger.info("Trying to fall back to SpeechRecognition library...")
            
            return self._initialize_speech_recognition_fallback()
    
    def _initialize_speech_recognition_fallback(self) -> bool:
        """
        Initialize fallback speech recognition using SpeechRecognition library.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            logger.info("Initializing SpeechRecognition fallback...")
            import speech_recognition as sr
            self.sr_recognizer = sr.Recognizer()
            
            logger.info("SpeechRecognition library loaded successfully as fallback!")
            
            self.is_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Failed to load SpeechRecognition: {e}")
            logger.warning("Please install it with: pip install SpeechRecognition")
            
            self.is_initialized = False
            return False
        except Exception as e:
            logger.error(f"Error initializing SpeechRecognition: {e}")
            
            self.is_initialized = False
            return False
    
    def transcribe(self, audio_file_path: str, language: str = "en", 
                  include_timestamps: bool = False, detect_speakers: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe an audio file using the available speech recognition engine.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            language: Language code for transcription (default: "en" for English)
            include_timestamps: Whether to include word-level timestamps (default: False)
            detect_speakers: Whether to attempt speaker detection (default: False)
            
        Returns:
            Tuple containing:
                - The transcribed text (or formatted text with timestamps if include_timestamps=True)
                - A dictionary with metadata about the transcription and timing information
        """
        # Check if the file exists
        if not os.path.exists(audio_file_path):
            error_msg = f"Audio file not found: {audio_file_path}"
            logger.error(error_msg)
            return "", {"error": error_msg, "engine": "none"}
        
        # Ensure speech recognition is initialized
        if not self.is_initialized and not self._initialize_speech_recognition():
            error_msg = "Failed to initialize speech recognition"
            logger.error(error_msg)
            return "", {"error": error_msg, "engine": "none"}
        
        # Try faster-whisper first
        if self.whisper_model is not None:
            try:
                logger.info(f"Transcribing audio with faster-whisper: {audio_file_path}")
                
                transcription_start = time.time()
                
                # Only enable word timestamps if specifically requested
                segments, info = self.whisper_model.transcribe(
                    audio_file_path, 
                    beam_size=5,
                    word_timestamps=include_timestamps or detect_speakers
                )
                
                # Convert segments to list to avoid generator exhaustion
                segments_list = list(segments)
                
                if detect_speakers:
                    # Basic speaker detection using segment clustering
                    speaker_segments = self._detect_speakers(segments_list)
                    
                    # Format output with speaker labels
                    formatted_output = "SPEAKER-AWARE TRANSCRIPT\n"
                    formatted_output += "=====================\n\n"
                    
                    # Track speaker stats
                    speaker_stats = {}
                    current_speaker = None
                    speaker_changes = 0
                    
                    # Process segments with speaker information
                    for segment in speaker_segments:
                        speaker = segment["speaker"]
                        text = segment["text"].strip()
                        start = segment["start"]
                        end = segment["end"]
                        
                        # Track speaker changes
                        if speaker != current_speaker:
                            formatted_output += f"\n[Speaker: {speaker}]\n"
                            current_speaker = speaker
                            if speaker_changes > 0:  # Don't count the first speaker
                                speaker_changes += 1
                        
                        # Add timestamped text
                        timestamp = str(timedelta(seconds=round(start)))
                        formatted_output += f"[{timestamp}] {text}\n"
                        
                        # Update speaker statistics
                        if speaker not in speaker_stats:
                            speaker_stats[speaker] = {
                                "talk_time": 0,
                                "segments": 0,
                                "first_appearance": start,
                                "last_appearance": end
                            }
                        
                        stats = speaker_stats[speaker]
                        stats["talk_time"] += (end - start)
                        stats["segments"] += 1
                        stats["last_appearance"] = max(stats["last_appearance"], end)
                    
                    # Calculate average turn duration
                    total_time = sum(s["talk_time"] for s in speaker_stats.values())
                    avg_turn_duration = total_time / speaker_changes if speaker_changes > 0 else total_time
                    
                    transcription = formatted_output
                    metadata = {
                        "engine": "faster-whisper",
                        "model": self.model_name,
                        "time_taken": time.time() - transcription_start,
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "duration": info.duration,
                        "has_timestamps": True,
                        "has_speakers": True,
                        "speakers": speaker_stats,
                        "speaker_changes": speaker_changes,
                        "average_turn_duration": avg_turn_duration,
                        "segments": speaker_segments
                    }
                    
                elif include_timestamps:
                    # Format timestamped output without speaker detection
                    formatted_output = "TIMESTAMPED TRANSCRIPT\n"
                    formatted_output += "===================\n\n"
                    
                    # Collect segments with timestamps
                    segment_data = []
                    for segment in segments_list:
                        timestamp = str(timedelta(seconds=round(segment.start)))
                        text = segment.text.strip()
                        formatted_output += f"[{timestamp}] {text}\n"
                        
                        segment_data.append({
                            "start": segment.start,
                            "end": segment.end,
                            "text": text,
                            "words": [{"word": w.word, "start": w.start, "end": w.end} 
                                    for w in (segment.words or [])]
                        })
                    
                    transcription = formatted_output
                    metadata = {
                        "engine": "faster-whisper",
                        "model": self.model_name,
                        "time_taken": time.time() - transcription_start,
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "duration": info.duration,
                        "segments": segment_data,
                        "has_timestamps": True,
                        "has_speakers": False
                    }
                else:
                    # Regular transcription without timestamps or speakers
                    transcription = " ".join(segment.text for segment in segments_list).strip()
                    metadata = {
                        "engine": "faster-whisper",
                        "model": self.model_name,
                        "time_taken": time.time() - transcription_start,
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "duration": info.duration,
                        "has_timestamps": False,
                        "has_speakers": False
                    }
                
                logger.info(f"Transcription completed in {metadata['time_taken']:.2f}s")
                logger.debug(f"Transcription info: {info}")
                
                return transcription, metadata
                
            except Exception as e:
                logger.error(f"Error transcribing with faster-whisper: {e}")
                logger.info("Falling back to SpeechRecognition...")
        
        # Fall back to SpeechRecognition if available
        if self.sr_recognizer is not None:
            try:
                import speech_recognition as sr
                
                logger.info(f"Transcribing audio with SpeechRecognition (fallback): {audio_file_path}")
                
                transcription_start = time.time()
                
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.sr_recognizer.record(source)
                    transcription = self.sr_recognizer.recognize_google(audio_data, language=language)
                
                transcription_time = time.time() - transcription_start
                
                logger.info(f"Fallback transcription completed in {transcription_time:.2f}s: {transcription}")
                
                # Return the transcription and metadata
                return transcription, {
                    "engine": "speech_recognition",
                    "api": "google",
                    "time_taken": transcription_time
                }
                
            except Exception as e:
                logger.error(f"Error transcribing with SpeechRecognition: {e}")
        
        # If all methods fail, return an error
        error_msg = "All speech recognition methods failed"
        logger.error(error_msg)
        
        return "", {"error": error_msg, "engine": "none"}
    
    def _detect_speakers(self, segments) -> List[Dict]:
        """
        Basic speaker detection using segment clustering based on timing and content.
        
        This is a simple heuristic approach that:
        1. Detects potential speaker changes based on pauses between segments
        2. Analyzes text patterns that might indicate dialogue
        3. Groups similar segments that likely belong to the same speaker
        
        Args:
            segments: List of transcribed segments from faster-whisper
            
        Returns:
            List of segments with speaker labels
        """
        try:
            MIN_PAUSE = 1.0  # Minimum pause that might indicate speaker change
            DIALOGUE_PATTERNS = [
                "said", "asked", "replied", "answered", "continued",
                ":", "?", "!"  # Punctuation that might indicate dialogue
            ]
            
            speaker_segments = []
            current_speaker = "SPEAKER_00"
            speaker_count = 1
            
            for i, segment in enumerate(segments):
                try:
                    is_speaker_change = False
                    
                    # Check for long pause from previous segment
                    if i > 0:
                        prev_end = segments[i-1].end
                        curr_start = segment.start
                        if curr_start - prev_end > MIN_PAUSE:
                            is_speaker_change = True
                    
                    # Check for dialogue indicators in text
                    text = segment.text.lower() if segment.text else ""
                    for pattern in DIALOGUE_PATTERNS:
                        if pattern in text:
                            is_speaker_change = True
                            break
                    
                    # If this seems like a new speaker, increment speaker count
                    if is_speaker_change:
                        current_speaker = f"SPEAKER_{speaker_count:02d}"
                        speaker_count += 1
                    
                    # Add segment with speaker info
                    speaker_segments.append({
                        "start": getattr(segment, "start", 0),
                        "end": getattr(segment, "end", 0),
                        "text": getattr(segment, "text", "").strip(),
                        "speaker": current_speaker,
                        "words": [{"word": w.word, "start": w.start, "end": w.end}
                                for w in (getattr(segment, "words", []) or [])]
                    })
                except Exception as e:
                    logger.warning(f"Error processing segment {i}: {e}")
                    # Add segment with basic info
                    speaker_segments.append({
                        "start": 0,
                        "end": 0,
                        "text": getattr(segment, "text", "").strip(),
                        "speaker": current_speaker,
                        "words": []
                    })
            
            # Post-process to merge nearby segments from same speaker
            merged_segments = []
            for segment in speaker_segments:
                if not merged_segments:
                    merged_segments.append(segment)
                    continue
                    
                prev = merged_segments[-1]
                if (prev["speaker"] == segment["speaker"] and 
                    segment["start"] - prev["end"] < MIN_PAUSE):
                    # Merge segments
                    prev["end"] = segment["end"]
                    prev["text"] += " " + segment["text"]
                    prev["words"].extend(segment["words"])
                else:
                    merged_segments.append(segment)
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Error in speaker detection: {e}")
            # Return basic segments without speaker detection
            return [{
                "start": getattr(segment, "start", 0),
                "end": getattr(segment, "end", 0),
                "text": getattr(segment, "text", "").strip(),
                "speaker": "SPEAKER_00",
                "words": []
            } for segment in segments]
    
    def start_streaming_transcription(self, 
                                    language: str = "en",
                                    on_partial_transcription: Optional[Callable[[str], None]] = None,
                                    on_final_transcription: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> bool:
        """
        Start streaming transcription using the configured model.
        
        Args:
            language: Language code for transcription (default: "en")
            on_partial_transcription: Callback for partial transcription updates
            on_final_transcription: Callback for final transcription with metadata
            
        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        # Ensure speech recognition is initialized
        if not self.is_initialized and not self._initialize_speech_recognition():
            logger.error("Failed to initialize speech recognition for streaming")
            return False
            
        try:
            # Create streaming transcriber if needed
            if self.streaming_transcriber is None:
                self.streaming_transcriber = StreamingTranscriber(
                    model_name=self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=language,
                    on_partial_transcription=on_partial_transcription,
                    on_final_transcription=on_final_transcription
                )
            
            # Start streaming
            success = self.streaming_transcriber.start_streaming()
            if success:
                logger.info("Streaming transcription started successfully")
            else:
                logger.error("Failed to start streaming transcription")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting streaming transcription: {e}")
            return False
    
    def add_streaming_audio_chunk(self, audio_chunk: bytes) -> None:
        """
        Add an audio chunk to the streaming transcription.
        
        Args:
            audio_chunk: Raw audio data to process
        """
        if self.streaming_transcriber is not None and self.streaming_transcriber.is_active():
            self.streaming_transcriber.add_audio_chunk(audio_chunk)
    
    def stop_streaming_transcription(self) -> Tuple[str, Dict[str, Any]]:
        """
        Stop streaming transcription and get final results.
        
        Returns:
            Tuple containing:
                - The final transcription text
                - A dictionary with metadata about the transcription
        """
        if self.streaming_transcriber is not None:
            try:
                return self.streaming_transcriber.stop_streaming()
            except Exception as e:
                logger.error(f"Error stopping streaming transcription: {e}")
        
        return "", {"error": "No active streaming transcription", "engine": "none"}
    
    def get_current_streaming_transcription(self) -> str:
        """
        Get the current partial transcription.
        
        Returns:
            str: The current partial transcription text
        """
        if self.streaming_transcriber is not None:
            return self.streaming_transcriber.get_current_transcription()
        return ""
    
    def is_streaming_active(self) -> bool:
        """
        Check if streaming transcription is active.
        
        Returns:
            bool: True if streaming is active, False otherwise
        """
        return (self.streaming_transcriber is not None and 
                self.streaming_transcriber.is_active())
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available speech recognition models.
        
        Returns:
            List of dictionaries containing model information
        """
        models = []
        
        # Add faster-whisper models if available
        if self.whisper_model is not None:
            models.extend([
                {"name": "tiny", "engine": "faster-whisper", "description": "Fastest, least accurate"},
                {"name": "base", "engine": "faster-whisper", "description": "Fast, good accuracy"},
                {"name": "small", "engine": "faster-whisper", "description": "Balanced speed and accuracy"},
                {"name": "medium", "engine": "faster-whisper", "description": "Good accuracy, slower"},
                {"name": "large-v2", "engine": "faster-whisper", "description": "Best accuracy, slowest"}
            ])
        
        # Add SpeechRecognition models if available
        if self.sr_recognizer is not None:
            models.append({
                "name": "google", 
                "engine": "speech_recognition", 
                "description": "Google Speech-to-Text API (requires internet)"
            })
        
        return models
    
    def get_current_model(self) -> Dict[str, Any]:
        """
        Get information about the currently active model.
        
        Returns:
            Dictionary containing information about the current model
        """
        if self.whisper_model is not None:
            return {
                "name": self.model_name,
                "engine": "faster-whisper",
                "device": self.device,
                "compute_type": self.compute_type
            }
        elif self.sr_recognizer is not None:
            return {
                "name": "google",
                "engine": "speech_recognition"
            }
        else:
            return {
                "name": "none",
                "engine": "none",
                "error": "No speech recognition model initialized"
            }
    
    def set_model(self, model_name: str, device: Optional[str] = None, compute_type: Optional[str] = None) -> bool:
        """
        Set the speech recognition model to use.
        
        Args:
            model_name: The name of the model to use
            device: The device to use for inference (optional)
            compute_type: The compute type to use for inference (optional)
            
        Returns:
            bool: True if the model was set successfully, False otherwise
        """
        # Update parameters if provided
        if device is not None:
            self.device = device
        
        if compute_type is not None:
            self.compute_type = compute_type
        
        # If the model name is the same and already initialized, no need to reinitialize
        if model_name == self.model_name and self.is_initialized and self.whisper_model is not None:
            return True
        
        # Update the model name
        self.model_name = model_name
        
        # Reset initialization state
        self.is_initialized = False
        self.whisper_model = None
        
        # Reinitialize with the new model
        return self._initialize_speech_recognition()


# Create a singleton instance for easy import
default_recognizer = SpeechRecognizer()

def transcribe_audio(audio_file_path: str, language: str = "en", include_timestamps: bool = False, detect_speakers: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Transcribe an audio file using the default speech recognizer.
    
    This is a convenience function that uses the default recognizer instance.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        language: Language code for transcription (default: "en" for English)
        include_timestamps: Whether to include word-level timestamps (default: False)
        detect_speakers: Whether to attempt speaker detection (default: False)
        
    Returns:
        If include_timestamps=False or detect_speakers=False:
            The transcribed text as a string
        If include_timestamps=True or detect_speakers=True:
            Tuple containing:
                - The formatted text with timestamps/speakers
                - A dictionary with metadata and timing information
    """
    transcription, metadata = default_recognizer.transcribe(
        audio_file_path, 
        language=language,
        include_timestamps=include_timestamps,
        detect_speakers=detect_speakers
    )
    
    # Always return both transcription and metadata since server.py expects it
    return transcription, metadata

def initialize_speech_recognition(
    model_name: str = "base", 
    device: str = "cpu", 
    compute_type: str = "int8"
) -> bool:
    """
    Initialize the default speech recognizer with the specified parameters.
    
    Args:
        model_name: The name of the faster-whisper model to use
        device: The device to use for inference
        compute_type: The compute type to use for inference
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global default_recognizer
    default_recognizer = SpeechRecognizer(model_name, device, compute_type)
    return default_recognizer.is_initialized

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available speech recognition models.
    
    Returns:
        List of dictionaries containing model information
    """
    return default_recognizer.get_available_models()

def get_current_model() -> Dict[str, Any]:
    """
    Get information about the currently active model.
    
    Returns:
        Dictionary containing information about the current model
    """
    return default_recognizer.get_current_model()

def start_streaming_transcription(
    language: str = "en",
    on_partial_transcription: Optional[Callable[[str], None]] = None,
    on_final_transcription: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> bool:
    """
    Start streaming transcription using the default speech recognizer.
    
    Args:
        language: Language code for transcription (default: "en")
        on_partial_transcription: Callback for partial transcription updates
        on_final_transcription: Callback for final transcription with metadata
        
    Returns:
        bool: True if streaming started successfully, False otherwise
    """
    return default_recognizer.start_streaming_transcription(
        language=language,
        on_partial_transcription=on_partial_transcription,
        on_final_transcription=on_final_transcription
    )

def add_streaming_audio_chunk(audio_chunk: bytes) -> None:
    """
    Add an audio chunk to the streaming transcription.
    
    Args:
        audio_chunk: Raw audio data to process
    """
    default_recognizer.add_streaming_audio_chunk(audio_chunk)

def stop_streaming_transcription() -> Tuple[str, Dict[str, Any]]:
    """
    Stop streaming transcription and get final results.
    
    Returns:
        Tuple containing:
            - The final transcription text
            - A dictionary with metadata about the transcription
    """
    return default_recognizer.stop_streaming_transcription()

def get_current_streaming_transcription() -> str:
    """
    Get the current partial transcription.
    
    Returns:
        str: The current partial transcription text
    """
    return default_recognizer.get_current_streaming_transcription()

def is_streaming_active() -> bool:
    """
    Check if streaming transcription is active.
    
    Returns:
        bool: True if streaming is active, False otherwise
    """
    return default_recognizer.is_streaming_active()