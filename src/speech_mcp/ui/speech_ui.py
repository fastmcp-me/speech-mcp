"""
PyQt implementation of the Speech UI.

This module provides a PyQt-based UI for speech recognition and visualization.
"""

import os
import sys
import time
import logging
import threading
import tempfile
import numpy as np
import wave
import pyaudio
import importlib.util
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QPushButton, QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QMetaObject, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont

# Setup logging
logger = logging.getLogger(__name__)

# Path to save speech state - same as in server.py
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speech_state.json")
TRANSCRIPTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "transcription.txt")
RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "response.txt")
COMMAND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui_command.txt")

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Path to audio notification files
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "audio")
START_LISTENING_SOUND = os.path.join(AUDIO_DIR, "start_listening.wav")
STOP_LISTENING_SOUND = os.path.join(AUDIO_DIR, "stop_listening.wav")

class TTSAdapter(QObject):
    """
    Text-to-speech adapter for PyQt UI.
    
    This class provides an interface to use Kokoro for TTS, with a fallback
    to pyttsx3 if Kokoro is not available.
    """
    speaking_finished = pyqtSignal()
    speaking_started = pyqtSignal()
    speaking_progress = pyqtSignal(float)  # Progress between 0.0 and 1.0
    
    def __init__(self):
        super().__init__()
        self.tts_engine = None
        self.is_speaking = False
        self.available_voices = []
        self.current_voice = None
        self.initialize_tts()
    
    def initialize_tts(self):
        """Initialize the TTS engine"""
        try:
            # First try to import the Kokoro adapter
            logger.info("Initializing Kokoro as primary TTS engine")
            
            # Check if Kokoro adapter is available
            if importlib.util.find_spec("speech_mcp.tts_adapters.kokoro_adapter") is not None:
                try:
                    # Import Kokoro adapter
                    from speech_mcp.tts_adapters.kokoro_adapter import KokoroTTS
                    
                    # Import config module if available
                    try:
                        from speech_mcp.config import get_setting, get_env_setting
                        
                        # Get saved voice preference
                        voice = None
                        
                        # First check environment variable
                        env_voice = get_env_setting("SPEECH_MCP_TTS_VOICE")
                        if env_voice:
                            voice = env_voice
                            logger.info(f"Using voice from environment variable: {voice}")
                        else:
                            # Then check config file
                            config_voice = get_setting("tts", "voice", None)
                            if config_voice:
                                voice = config_voice
                                logger.info(f"Using voice from config: {voice}")
                    except ImportError:
                        voice = None
                        logger.info("Config module not available, using default voice")
                    
                    # Initialize with default or saved voice settings
                    if voice:
                        self.tts_engine = KokoroTTS(voice=voice, lang_code="a", speed=1.0)
                    else:
                        self.tts_engine = KokoroTTS(voice="af_heart", lang_code="a", speed=1.0)
                    
                    logger.info("Kokoro TTS adapter initialized successfully")
                    
                    # List of available Kokoro voice models
                    voices = self.tts_engine.get_available_voices()
                    self.available_voices = voices
                    self.current_voice = self.tts_engine.voice  # Use the actual voice that was set
                    logger.debug(f"Available Kokoro TTS voices: {len(voices)}")
                    for i, voice in enumerate(voices):
                        logger.debug(f"Voice {i}: {voice}")
                    
                    return True
                except ImportError as e:
                    logger.warning(f"Failed to import Kokoro adapter: {e}")
                    # Fall back to pyttsx3
                except Exception as e:
                    logger.error(f"Error initializing Kokoro: {e}")
                    # Fall back to pyttsx3
            
            # Fall back to pyttsx3
            logger.info("Falling back to pyttsx3 for TTS")
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                logger.info("pyttsx3 text-to-speech engine initialized")
                
                # Get available voices
                voices = self.tts_engine.getProperty('voices')
                self.available_voices = [f"pyttsx3:{voice.id}" for voice in voices]
                if self.available_voices:
                    self.current_voice = self.available_voices[0]
                logger.debug(f"Available pyttsx3 voices: {len(voices)}")
                for i, voice in enumerate(voices):
                    logger.debug(f"Voice {i}: {voice.id} - {voice.name}")
                
                return True
            except ImportError as e:
                logger.warning(f"pyttsx3 not available: {e}")
            except Exception as e:
                logger.error(f"Error initializing pyttsx3: {e}")
            
            logger.error("No TTS engine available")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            return False
    
    def speak(self, text):
        """Speak the given text"""
        if not text:
            logger.warning("Empty text provided to speak")
            return False
        
        if not self.tts_engine:
            logger.warning("No TTS engine available")
            return False
        
        # Prevent multiple simultaneous speech
        if self.is_speaking:
            logger.warning("Already speaking, ignoring new request")
            return False
        
        logger.info(f"TTSAdapter.speak called with text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"TTSAdapter.speak called with text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        self.is_speaking = True
        self.speaking_started.emit()
        
        # Start speaking in a separate thread
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
        logger.info("Started _speak_thread")
        return True
    
    def _speak_thread(self, text):
        """Thread function for speaking text"""
        try:
            logger.info(f"_speak_thread started for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"_speak_thread started for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Use the appropriate TTS engine
            if hasattr(self.tts_engine, 'speak'):
                # This is the Kokoro adapter
                logger.info("Using Kokoro adapter speak method")
                print("Using Kokoro adapter speak method")
                try:
                    result = self.tts_engine.speak(text)
                    logger.info(f"Kokoro speak result: {result}")
                    print(f"Kokoro speak result: {result}")
                    if not result:
                        logger.error("Kokoro TTS failed")
                        print("Kokoro TTS failed")
                except Exception as e:
                    logger.error(f"Exception in Kokoro speak: {e}", exc_info=True)
                    print(f"Exception in Kokoro speak: {e}")
                    result = False
            else:
                # This is pyttsx3
                logger.info("Using pyttsx3 say method")
                print("Using pyttsx3 say method")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                logger.info("pyttsx3 speech completed")
                print("pyttsx3 speech completed")
            
            logger.info("Speech completed")
            print("Speech completed")
        except Exception as e:
            logger.error(f"Error during text-to-speech: {e}", exc_info=True)
            print(f"Error during text-to-speech: {e}")
        finally:
            self.is_speaking = False
            self.speaking_finished.emit()
            logger.info("Speaking finished signal emitted")
            print("Speaking finished signal emitted")
    
    def set_voice(self, voice_id):
        """Set the voice to use for TTS"""
        if not self.tts_engine:
            logger.warning("No TTS engine available")
            return False
        
        try:
            if self.tts_engine and hasattr(self.tts_engine, 'set_voice'):
                # This is the Kokoro adapter
                result = self.tts_engine.set_voice(voice_id)
                if result:
                    self.current_voice = voice_id
                    logger.info(f"Voice set to: {voice_id}")
                    return True
                else:
                    logger.error(f"Failed to set voice to: {voice_id}")
                    return False
            elif hasattr(self.tts_engine, 'setProperty'):
                # This is pyttsx3
                # Extract the voice ID from the format "pyttsx3:voice_id"
                if voice_id.startswith("pyttsx3:"):
                    voice_id = voice_id.split(":", 1)[1]
                
                # Find the voice object
                for voice in self.tts_engine.getProperty('voices'):
                    if voice.id == voice_id:
                        self.tts_engine.setProperty('voice', voice.id)
                        self.current_voice = f"pyttsx3:{voice.id}"
                        logger.info(f"Voice set to: {voice.name}")
                        return True
                
                logger.error(f"Voice not found: {voice_id}")
                return False
            
            logger.warning("TTS engine does not support voice selection")
            return False
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            return False
    
    def get_available_voices(self):
        """Get a list of available voices"""
        return self.available_voices
    
    def get_current_voice(self):
        """Get the current voice"""
        return self.current_voice

# Path to save speech state - same as in server.py
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speech_state.json")
TRANSCRIPTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "transcription.txt")
RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "response.txt")
COMMAND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui_command.txt")

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Path to audio notification files
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "audio")
START_LISTENING_SOUND = os.path.join(AUDIO_DIR, "start_listening.wav")
STOP_LISTENING_SOUND = os.path.join(AUDIO_DIR, "stop_listening.wav")

# For playing notification sounds
def play_audio_file(file_path):
    """Play an audio file using PyAudio"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return
        
        logger.debug(f"Playing audio notification: {file_path}")
        
        # Open the wave file
        with wave.open(file_path, 'rb') as wf:
            # Create PyAudio instance
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            
            # Read data in chunks and play
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            
            # Close stream and PyAudio
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            logger.debug("Audio notification played successfully")
    except Exception as e:
        logger.error(f"Error playing audio notification: {e}")


class AudioProcessor(QObject):
    """
    Handles audio processing and speech recognition.
    """
    audio_level_updated = pyqtSignal(float)
    transcription_ready = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_listening = False
        self.audio_frames = []
        self.pyaudio = None
        self.stream = None
        self.selected_device_index = None
        self.whisper_model = None
        self._setup_audio()
        
    def _setup_audio(self):
        """Set up audio capture and processing."""
        try:
            logger.info("Setting up audio processing")
            self.pyaudio = pyaudio.PyAudio()
            
            # Log audio device information
            logger.info(f"PyAudio version: {pyaudio.get_portaudio_version()}")
            
            # Get all available audio devices
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            logger.info(f"Found {numdevices} audio devices:")
            
            # Find the best input device
            for i in range(numdevices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                    device_name = device_info.get('name')
                    max_input_channels = device_info.get('maxInputChannels')
                    
                    logger.info(f"Device {i}: {device_name}")
                    logger.info(f"  Max Input Channels: {max_input_channels}")
                    logger.info(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
                    
                    # Only consider input devices
                    if max_input_channels > 0:
                        logger.info(f"Found input device: {device_name}")
                        
                        # Prefer non-default devices as they're often external mics
                        if self.selected_device_index is None or 'default' not in device_name.lower():
                            self.selected_device_index = i
                            logger.info(f"Selected input device: {device_name} (index {i})")
                except Exception as e:
                    logger.warning(f"Error checking device {i}: {e}")
            
            if self.selected_device_index is None:
                logger.warning("No suitable input device found, using default")
            
            # Initialize speech recognition in a background thread
            threading.Thread(target=self._initialize_speech_recognition, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition in a background thread"""
        try:
            logger.info("Loading faster-whisper speech recognition model...")
            
            # Import here to avoid circular imports
            import faster_whisper
            
            # Load the small model for a good balance of speed and accuracy
            # Using CPU as default for compatibility
            self.whisper_model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")
            
            logger.info("faster-whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading faster-whisper model: {e}")
            # Try to load speech_recognition as fallback
            try:
                import speech_recognition as sr
                self.sr = sr
                self.recognizer = sr.Recognizer()
                logger.info("SpeechRecognition library loaded as fallback")
            except ImportError:
                logger.error("Failed to load SpeechRecognition as fallback")
    
    def start_listening(self):
        """Start listening for audio input."""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.audio_frames = []
        
        # Play start listening notification sound
        threading.Thread(target=play_audio_file, args=(START_LISTENING_SOUND,), daemon=True).start()
        
        try:
            logger.info("Starting audio recording")
            
            def audio_callback(in_data, frame_count, time_info, status):
                try:
                    # Check for audio status flags
                    if status:
                        status_flags = []
                        if status & pyaudio.paInputUnderflow:
                            status_flags.append("Input Underflow")
                        if status & pyaudio.paInputOverflow:
                            status_flags.append("Input Overflow")
                        if status & pyaudio.paOutputUnderflow:
                            status_flags.append("Output Underflow")
                        if status & pyaudio.paOutputOverflow:
                            status_flags.append("Output Overflow")
                        if status & pyaudio.paPrimingOutput:
                            status_flags.append("Priming Output")
                        
                        if status_flags:
                            logger.warning(f"Audio callback status flags: {', '.join(status_flags)}")
                    
                    # Store audio data for processing
                    self.audio_frames.append(in_data)
                    
                    # Process audio for visualization
                    self._process_audio_for_visualization(in_data)
                    
                    return (in_data, pyaudio.paContinue)
                    
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
                    return (in_data, pyaudio.paContinue)  # Try to continue despite errors
            
            # Start the audio stream with the selected device
            logger.debug(f"Opening audio stream with FORMAT={FORMAT}, CHANNELS={CHANNELS}, RATE={RATE}, CHUNK={CHUNK}, DEVICE={self.selected_device_index}")
            self.stream = self.pyaudio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.selected_device_index,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback
            )
            
            # Verify stream is active and receiving audio
            if not self.stream.is_active():
                logger.error("Stream created but not active")
                raise Exception("Audio stream is not active")
            
            logger.info("Audio stream initialized and receiving data")
            
            # Start a thread to detect silence and stop recording
            threading.Thread(target=self._detect_silence, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.is_listening = False
    
    def _process_audio_for_visualization(self, audio_data):
        """Process audio data for visualization"""
        try:
            # Convert to numpy array
            data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize the data to range [-1, 1]
            normalized = data.astype(float) / 32768.0
            
            # Take absolute value to get amplitude
            amplitude = np.abs(normalized).mean()
            
            # Apply amplification factor to make the visualization more prominent
            # Increase the factor from 1.0 to 5.0 to make the visualization more visible
            amplification_factor = 5.0
            amplified_amplitude = min(amplitude * amplification_factor, 1.0)  # Clamp to 1.0 max
            
            # Log the original and amplified values occasionally for debugging
            if np.random.random() < 0.01:  # Log roughly 1% of values to avoid flooding logs
                logger.debug(f"Audio amplitude: original={amplitude:.6f}, amplified={amplified_amplitude:.6f}")
            
            # Emit the signal with the amplified amplitude
            self.audio_level_updated.emit(amplified_amplitude)
            
        except Exception as e:
            logger.error(f"Error processing audio for visualization: {e}")
    
    def _detect_silence(self):
        """Detect when the user stops speaking and end recording"""
        try:
            # Wait for initial audio to accumulate
            logger.info("Starting silence detection")
            time.sleep(0.5)
            
            # Adjusted silence detection parameters for longer pauses
            silence_threshold = 0.008  # Reduced threshold to be more sensitive to quiet speech
            silence_duration = 0
            max_silence = 2.0  # 2 seconds of silence to stop recording
            check_interval = 0.1  # Check every 100ms
            
            logger.debug(f"Silence detection parameters: threshold={silence_threshold}, max_silence={max_silence}s, check_interval={check_interval}s")
            
            while self.is_listening and self.stream and silence_duration < max_silence:
                if not self.audio_frames or len(self.audio_frames) < 2:
                    time.sleep(check_interval)
                    continue
                
                # Get the latest audio frame
                latest_frame = self.audio_frames[-1]
                audio_data = np.frombuffer(latest_frame, dtype=np.int16)
                normalized = audio_data.astype(float) / 32768.0
                current_amplitude = np.abs(normalized).mean()
                
                if current_amplitude < silence_threshold:
                    silence_duration += check_interval
                    # Log only when silence is detected
                    if silence_duration >= 1.0 and silence_duration % 1.0 < check_interval:
                        logger.debug(f"Silence detected for {silence_duration:.1f}s, amplitude: {current_amplitude:.6f}")
                else:
                    if silence_duration > 0:
                        logger.debug(f"Speech resumed after {silence_duration:.1f}s of silence, amplitude: {current_amplitude:.6f}")
                    silence_duration = 0
                
                time.sleep(check_interval)
            
            # If we exited because of silence detection
            if self.is_listening and self.stream:
                logger.info(f"Silence threshold reached after {silence_duration:.1f}s, stopping recording")
                self.process_recording()
                self.stop_listening()
            
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
    
    def process_recording(self):
        """Process the recorded audio and generate a transcription"""
        try:
            if not self.audio_frames:
                logger.warning("No audio frames to process")
                return
            
            logger.info(f"Processing {len(self.audio_frames)} audio frames")
            
            # Check if we have enough audio data
            total_audio_time = len(self.audio_frames) * (CHUNK / RATE)
            logger.info(f"Total recorded audio: {total_audio_time:.2f} seconds")
            
            if total_audio_time < 0.5:  # Less than half a second of audio
                logger.warning(f"Audio recording too short ({total_audio_time:.2f}s), may not contain speech")
            
            # Save the recorded audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
                # Create a WAV file from the recorded frames
                logger.debug(f"Creating WAV file at {temp_audio_path}")
                wf = wave.open(temp_audio_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.pyaudio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
                
                # Get file size for logging
                file_size = os.path.getsize(temp_audio_path)
                logger.debug(f"WAV file created, size: {file_size} bytes")
            
            logger.info(f"Audio saved to temporary file: {temp_audio_path}")
            
            # Use faster-whisper to transcribe the audio
            if self.whisper_model:
                logger.info("Transcribing audio with faster-whisper...")
                
                transcription_start = time.time()
                segments, info = self.whisper_model.transcribe(temp_audio_path, beam_size=5)
                
                # Collect all segments to form the complete transcription
                transcription = ""
                for segment in segments:
                    transcription += segment.text + " "
                
                transcription = transcription.strip()
                transcription_time = time.time() - transcription_start
                
                logger.info(f"Transcription completed in {transcription_time:.2f}s: {transcription}")
                logger.debug(f"Transcription info: {info}")
            
            # Fallback to SpeechRecognition if whisper_model is not available
            elif hasattr(self, 'sr') and hasattr(self, 'recognizer'):
                logger.info("Transcribing audio with SpeechRecognition (fallback)...")
                
                with self.sr.AudioFile(temp_audio_path) as source:
                    audio_data = self.recognizer.record(source)
                    transcription = self.recognizer.recognize_google(audio_data)
                
                logger.info(f"Fallback transcription completed: {transcription}")
            
            else:
                logger.error("No speech recognition engine available")
                transcription = "Error: Speech recognition not available"
            
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
            if self.stream:
                logger.debug(f"Stopping audio stream, stream active: {self.stream.is_active()}")
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                logger.info("Audio stream closed successfully")
                
                # Play stop listening notification sound
                threading.Thread(target=play_audio_file, args=(STOP_LISTENING_SOUND,), daemon=True).start()
            
            self.is_listening = False
            
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
            self.is_listening = False


class AudioVisualizer(QWidget):
    """
    Widget for visualizing audio levels and waveforms.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.audio_levels = [0.0] * 50  # Store recent audio levels (reduced from 100 for center-rising visualization)
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # Add a smoothing factor to make the visualization less jumpy
        self.smoothing_factor = 0.3
        self.last_level = 0.0
        
        # Timer for animation
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(30)  # Update at ~30fps
    
    def update_level(self, level):
        """Update with a new audio level."""
        # Apply smoothing to avoid abrupt changes
        smoothed_level = (level * (1.0 - self.smoothing_factor)) + (self.last_level * self.smoothing_factor)
        self.last_level = smoothed_level
        
        # For center-rising visualization, we just need to update the current level
        # We'll shift all values in paintEvent
        self.audio_levels.pop(0)
        self.audio_levels.append(smoothed_level)
    
    def paintEvent(self, event):
        """Draw the audio visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(event.rect(), QColor(30, 30, 30))
        
        # Draw waveform
        width = self.width()
        height = self.height()
        mid_height = height / 2
        
        # Set pen for waveform
        pen = QPen(QColor(0, 200, 255, 180))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw the center-rising waveform
        # We'll draw bars at different positions with heights based on audio levels
        bar_count = 40  # Number of bars to draw
        bar_width = width / bar_count
        bar_spacing = 2  # Pixels between bars
        
        for i in range(bar_count):
            # Calculate the position in the audio_levels array
            # Center bars use the most recent values
            if i < len(self.audio_levels):
                # For bars in the middle, use the most recent levels
                level_idx = len(self.audio_levels) - 1 - i
                if level_idx >= 0:
                    level = self.audio_levels[level_idx]
                else:
                    level = 0.0
            else:
                level = 0.0
                
            # Calculate bar height based on level
            bar_height = level * mid_height * 0.95
            
            # Calculate x position (centered)
            x = (width / 2) + (i * bar_width / 2) - (bar_width / 2)
            x_mirror = (width / 2) - (i * bar_width / 2) - (bar_width / 2)
            
            # Draw the bar (right side)
            if i < bar_count / 2:
                rect = QRectF(x, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                painter.fillRect(rect, QColor(0, 200, 255, 180 - i * 3))
            
            # Draw the mirrored bar (left side)
            if i < bar_count / 2:
                rect_mirror = QRectF(x_mirror, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                painter.fillRect(rect_mirror, QColor(0, 200, 255, 180 - i * 3))
        
        # Draw a thin center line
        painter.setPen(QPen(QColor(100, 100, 100, 100)))
        painter.drawLine(0, int(mid_height), width, int(mid_height))


class AnimatedButton(QPushButton):
    """
    Custom button class with press animation effect.
    """
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFlat(False)
        self.original_style = ""
        self.animation_timer = None
        self.animation_step = 0
        self.animation_steps = 10
        self.animation_direction = 1  # 1 for pressing, -1 for releasing
        self.is_animating = False
        
    def set_style(self, style):
        """Set the button's base style"""
        self.original_style = style
        self.setStyleSheet(style)
        
    def mousePressEvent(self, event):
        """Handle mouse press event with animation"""
        if self.animation_timer is not None:
            self.animation_timer.stop()
            
        self.animation_step = 0
        self.animation_direction = 1
        self.is_animating = True
        
        # Start animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_press)
        self.animation_timer.start(20)  # 20ms per frame
        
        # Call parent handler
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release event with animation"""
        if self.animation_timer is not None:
            self.animation_timer.stop()
            
        self.animation_step = self.animation_steps
        self.animation_direction = -1
        self.is_animating = True
        
        # Start animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_release)
        self.animation_timer.start(20)  # 20ms per frame
        
        # Call parent handler
        super().mouseReleaseEvent(event)
        
    def animate_press(self):
        """Animate button press"""
        if not self.is_animating:
            return
            
        self.animation_step += 1
        
        if self.animation_step >= self.animation_steps:
            self.animation_step = self.animation_steps
            self.animation_timer.stop()
            
        # Calculate animation progress (0.0 to 1.0)
        progress = self.animation_step / self.animation_steps
        
        # Apply visual changes based on progress
        self._apply_animation_style(progress)
        
    def animate_release(self):
        """Animate button release"""
        if not self.is_animating:
            return
            
        self.animation_step -= 1
        
        if self.animation_step <= 0:
            self.animation_step = 0
            self.animation_timer.stop()
            self.is_animating = False
            self.setStyleSheet(self.original_style)
            return
            
        # Calculate animation progress (1.0 to 0.0)
        progress = self.animation_step / self.animation_steps
        
        # Apply visual changes based on progress
        self._apply_animation_style(progress)
        
    def _apply_animation_style(self, progress):
        """Apply animation style based on progress (0.0 to 1.0)"""
        if not self.original_style:
            return
            
        # Extract background color from original style
        bg_color = None
        for style_part in self.original_style.split(";"):
            if "background-color:" in style_part:
                bg_color = style_part.split(":")[1].strip()
                break
                
        if not bg_color:
            return
            
        # Parse color
        if bg_color.startswith("#"):
            # Hex color
            r = int(bg_color[1:3], 16)
            g = int(bg_color[3:5], 16)
            b = int(bg_color[5:7], 16)
            
            # Darken for press effect (reduce by up to 30%)
            darken_factor = 0.7 + (0.3 * (1.0 - progress))
            r = max(0, int(r * darken_factor))
            g = max(0, int(g * darken_factor))
            b = max(0, int(b * darken_factor))
            
            # Create new style with darkened color
            new_bg_color = f"#{r:02x}{g:02x}{b:02x}"
            new_style = self.original_style.replace(bg_color, new_bg_color)
            
            # Add slight inset shadow effect
            shadow_strength = int(progress * 5)
            if shadow_strength > 0:
                new_style += f"; border: none; border-radius: 5px; padding: 8px 16px; font-weight: bold; margin: {shadow_strength}px 0px 0px 0px;"
                
            self.setStyleSheet(new_style)


class PyQtSpeechUI(QMainWindow):
    """
    Main speech UI window implemented with PyQt.
    """
    # Signal for when components are fully loaded
    components_ready = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goose Speech Interface")
        self.resize(500, 300)
        
        # Set initial loading state
        self.tts_ready = False
        self.stt_ready = False
        self.audio_ready = False
        
        # Create UI first (will be in loading state)
        self.setup_ui()
        
        # Create a command file to indicate UI is visible (but not fully ready)
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write("UI_READY")
            logger.info("Created initial UI_READY command file (UI is visible)")
        except Exception as e:
            logger.error(f"Error creating initial command file: {e}")
        
        # Start checking for server commands
        self.command_check_timer = QTimer(self)
        self.command_check_timer.timeout.connect(self.check_for_commands)
        self.command_check_timer.start(100)  # Check every 100ms
        
        # Start checking for response files
        self.response_check_timer = QTimer(self)
        self.response_check_timer.timeout.connect(self.check_for_responses)
        self.response_check_timer.start(100)  # Check every 100ms
        
        # Connect the components_ready signal to update UI
        self.components_ready.connect(self.on_components_ready)
        
        # Initialize components in background threads
        QTimer.singleShot(100, self.initialize_components)
        
    def setup_ui(self):
        """Set up the UI components."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Status label - initially shows loading
        self.status_label = QLabel("Loading...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #f39c12;
        """)
        main_layout.addWidget(self.status_label)
        
        # Audio visualizer
        self.visualizer = AudioVisualizer()
        main_layout.addWidget(self.visualizer)
        
        # Transcription display
        self.transcription_label = QLabel("Initializing speech recognition...")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setWordWrap(True)
        self.transcription_label.setStyleSheet("""
            font-size: 14px;
            color: #ffffff;
            background-color: #2a2a2a;
            border-radius: 5px;
            padding: 10px;
        """)
        main_layout.addWidget(self.transcription_label)
        
        # Voice selection
        voice_layout = QHBoxLayout()
        voice_label = QLabel("Voice:")
        voice_label.setStyleSheet("color: #ffffff;")
        self.voice_combo = QComboBox()
        self.voice_combo.setStyleSheet("""
            background-color: #2a2a2a;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
            padding: 5px;
        """)
        
        # Add loading placeholder
        self.voice_combo.addItem("Loading voices...")
        self.voice_combo.setEnabled(False)
        self.voice_combo.currentIndexChanged.connect(self.on_voice_changed)
        
        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(self.voice_combo, 1)  # 1 = stretch factor
        main_layout.addLayout(voice_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Add Listen button
        self.listen_button = AnimatedButton("Start Listening")
        self.listen_button.clicked.connect(self.toggle_listening)
        self.listen_button.setEnabled(True)  # Enable the button
        self.listen_button.set_style("""
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Use AnimatedButton for Test Voice button
        self.speak_button = AnimatedButton("Test Voice")
        self.speak_button.clicked.connect(self.test_voice)
        self.speak_button.setEnabled(True)  # Enable the button
        self.speak_button.set_style("""
            background-color: #27ae60;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Use AnimatedButton for Close button
        self.close_button = AnimatedButton("Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.set_style("""
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        button_layout.addWidget(self.listen_button)
        button_layout.addWidget(self.speak_button)
        button_layout.addWidget(self.close_button)
        main_layout.addLayout(button_layout)
        
        # Set the main widget
        self.setCentralWidget(main_widget)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def update_voice_list(self):
        """Update the voice selection combo box"""
        # Skip if TTS adapter is not ready yet
        if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
            logger.warning("Cannot update voice list - TTS adapter not ready")
            return
            
        self.voice_combo.clear()
        voices = self.tts_adapter.get_available_voices()
        current_voice = self.tts_adapter.get_current_voice()
        
        if not voices:
            self.voice_combo.addItem("No voices available")
            self.voice_combo.setEnabled(False)
            return
        
        # Add all available voices
        selected_index = 0
        for i, voice in enumerate(voices):
            # Format the voice name for display
            if voice.startswith("pyttsx3:"):
                # For pyttsx3 voices, try to get a more readable name
                voice_id = voice.split(":", 1)[1]
                if hasattr(self.tts_adapter.tts_engine, 'getProperty'):
                    for v in self.tts_adapter.tts_engine.getProperty('voices'):
                        if v.id == voice_id:
                            display_name = f"{v.name} (pyttsx3)"
                            self.voice_combo.addItem(display_name, voice)
                            break
                    else:
                        self.voice_combo.addItem(voice, voice)
                else:
                    self.voice_combo.addItem(voice, voice)
            else:
                # For Kokoro voices, use the voice name directly
                self.voice_combo.addItem(voice, voice)
            
            # Select the current voice
            if voice == current_voice:
                selected_index = i
        
        # Enable the combo box now that it has real data
        self.voice_combo.setEnabled(True)
        
        # Set the current selection
        self.voice_combo.setCurrentIndex(selected_index)
        logger.info(f"Voice combo initialized with {len(voices)} voices, selected: {current_voice}")
        print(f"Voice combo initialized with {len(voices)} voices, selected: {current_voice}")
    
    def initialize_components(self):
        """Initialize components in background threads"""
        logger.info("Starting background initialization of components")
        self.status_label.setText("Loading components...")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f39c12;")
        
        # Start background threads for initialization
        threading.Thread(target=self.initialize_audio_processor, daemon=True).start()
        threading.Thread(target=self.initialize_tts_adapter, daemon=True).start()
    
    def initialize_audio_processor(self):
        """Initialize audio processor in background thread"""
        try:
            logger.info("Initializing audio processor in background")
            self.audio_processor = AudioProcessor()
            self.audio_processor.audio_level_updated.connect(self.update_audio_level)
            self.audio_processor.transcription_ready.connect(self.handle_transcription)
            self.audio_ready = True
            logger.info("Audio processor initialization complete")
            self.check_all_components_ready()
        except Exception as e:
            logger.error(f"Error initializing audio processor: {e}")
    
    def initialize_tts_adapter(self):
        """Initialize TTS adapter in background thread"""
        try:
            logger.info("Initializing TTS adapter in background")
            self.tts_adapter = TTSAdapter()
            self.tts_adapter.speaking_started.connect(self.on_speaking_started)
            self.tts_adapter.speaking_finished.connect(self.on_speaking_finished)
            self.tts_ready = True
            logger.info("TTS adapter initialization complete")
            
            # Update voice list when TTS is ready - use QTimer to call from main thread
            QTimer.singleShot(0, self.update_voice_list)
            
            self.check_all_components_ready()
        except Exception as e:
            logger.error(f"Error initializing TTS adapter: {e}")
    
    def check_all_components_ready(self):
        """Check if all components are ready and emit signal if they are"""
        if self.audio_ready and self.tts_ready:
            logger.info("All components initialized successfully")
            # Use QTimer to safely emit signal from background thread
            QTimer.singleShot(0, lambda: self.components_ready.emit())
    
    def on_components_ready(self):
        """Called when all components are ready"""
        logger.info("All components are ready, updating UI")
        
        # Update status label
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f8c8d;")
        
        # Clear initialization message from transcription label
        if self.transcription_label.text().startswith("Initializing"):
            self.transcription_label.setText("Ready for voice interaction")
        
        # Check if there's a saved voice preference
        self.check_voice_preference()
        
        # Check for any pending commands
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                    if command == "LISTEN":
                        # Only start listening if we have a saved voice preference
                        if self.has_saved_voice_preference():
                            self.start_listening()
                        else:
                            logger.info("Ignoring LISTEN command because no voice preference is saved")
            except Exception as e:
                logger.error(f"Error reading command file: {e}")
    
    def has_saved_voice_preference(self):
        """Check if there's a saved voice preference"""
        try:
            # Check if we have a config module
            if importlib.util.find_spec("speech_mcp.config") is not None:
                from speech_mcp.config import get_setting, get_env_setting
                
                # Check environment variable
                env_voice = get_env_setting("SPEECH_MCP_TTS_VOICE")
                if env_voice:
                    return True
                
                # Check config file
                config_voice = get_setting("tts", "voice", None)
                if config_voice:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking for saved voice preference: {e}")
            return False
    
    def check_voice_preference(self):
        """Check if there's a saved voice preference and play guidance if not"""
        if not self.has_saved_voice_preference():
            logger.info("No saved voice preference found, playing guidance message")
            self.transcription_label.setText("Please select a voice from the dropdown and click 'Test Voice' to hear it.")
            
            # Wait a moment before speaking to ensure UI is fully ready
            QTimer.singleShot(500, self.play_guidance_message)
    
    def play_guidance_message(self):
        """Play a guidance message for first-time users"""
        if hasattr(self, 'tts_adapter') and self.tts_adapter:
            self.tts_adapter.speak("This is the default voice. Please select a voice from the dropdown menu and click Test Voice to hear it.")
            logger.info("Played guidance message for first-time user")
    
    
    def on_voice_changed(self, index):
        """Handle voice selection change"""
        # Skip if TTS adapter is not ready yet
        if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
            return
            
        if index < 0:
            return
        
        voice = self.voice_combo.itemData(index)
        if not voice:
            return
        
        logger.info(f"Voice selection changed to: {voice}")
        self.tts_adapter.set_voice(voice)
    
    def test_voice(self):
        """Test the selected voice"""
        print("Test voice button clicked!")
        logger.info("Test voice button clicked!")
        
        # Skip if TTS adapter is not ready yet
        if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
            logger.warning("Test voice button clicked but TTS not ready yet")
            self.status_label.setText("TTS not ready yet")
            self.transcription_label.setText("TTS not ready yet. Please wait...")
            QTimer.singleShot(2000, lambda: self.status_label.setText("Loading..."))
            return
            
        logger.info(f"TTS adapter exists: {self.tts_adapter is not None}")
        logger.info(f"TTS engine exists: {self.tts_adapter.tts_engine is not None}")
        logger.info(f"Is speaking: {self.tts_adapter.is_speaking}")
        
        if self.tts_adapter.is_speaking:
            logger.warning("Already speaking, ignoring test request")
            return
        
        # Update the transcription label to show we're testing the voice
        self.transcription_label.setText("Testing voice...")
        
        # Log the current voice being tested
        current_voice = self.tts_adapter.get_current_voice()
        logger.info(f"Testing voice: {current_voice}")
        print(f"Testing voice: {current_voice}")
        
        # Speak a test message
        try:
            logger.info("Attempting to speak test message")
            result = self.tts_adapter.speak("This is a test of the selected voice. Hello, I am Goose!")
            logger.info(f"TTS speak result: {result}")
            print(f"TTS speak result: {result}")
            
            if not result:
                logger.error("Failed to start speaking test message")
                self.transcription_label.setText("Error: Failed to test voice")
                QTimer.singleShot(2000, lambda: self.transcription_label.setText("Select a voice and click 'Test Voice' to hear it"))
        except Exception as e:
            logger.error(f"Exception during test voice: {e}", exc_info=True)
            print(f"Exception during test voice: {e}")
            self.transcription_label.setText(f"Error: {str(e)}")
            QTimer.singleShot(3000, lambda: self.transcription_label.setText("Select a voice and click 'Test Voice' to hear it"))
    
    def update_audio_level(self, level):
        """Update the audio level visualization."""
        self.visualizer.update_level(level)
    
    def handle_transcription(self, text):
        """Handle new transcription text."""
        self.transcription_label.setText(text)
        logger.info(f"New transcription: {text}")
    
    def start_listening(self):
        """Start listening mode."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            self.status_label.setText("Speech recognition not ready yet")
            QTimer.singleShot(2000, lambda: self.status_label.setText("Loading..."))
            return
            
        self.audio_processor.start_listening()
        self.status_label.setText("Listening...")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00c8ff;")
    
    def stop_listening(self):
        """Stop listening mode."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            return
            
        self.audio_processor.stop_listening()
        self.status_label.setText("Not Listening")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f8c8d;")
    
    def toggle_listening(self):
        """Toggle between listening and not listening states."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            logger.warning("Listen button clicked but audio processor not ready yet")
            self.status_label.setText("Speech recognition not ready yet")
            self.transcription_label.setText("Speech recognition not ready yet. Please wait...")
            QTimer.singleShot(2000, lambda: self.status_label.setText("Loading..."))
            return
            
        if self.audio_processor.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
    
    def on_speaking_started(self):
        """Called when speaking starts."""
        self.status_label.setText("Speaking...")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2ecc71;")
        self.speak_button.setEnabled(False)
        
    def on_speaking_finished(self):
        """Called when speaking finishes."""
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f8c8d;")
        self.speak_button.setEnabled(True)
    
    def check_for_commands(self):
        """Check for commands from the server."""
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                
                # Process the command
                if command == "LISTEN":
                    logger.info("Received LISTEN command")
                    # If components are not ready, store the command to process later
                    if not hasattr(self, 'audio_processor') or not self.audio_processor:
                        logger.info("Components not ready yet, will process LISTEN command when ready")
                        # Command will be processed in on_components_ready
                        return
                    
                    # Only start listening if we have a saved voice preference
                    if self.has_saved_voice_preference():
                        self.start_listening()
                    else:
                        logger.info("Ignoring LISTEN command because no voice preference is saved")
                        # Play guidance message instead
                        self.check_voice_preference()
                        
                elif command == "IDLE" and hasattr(self, 'audio_processor') and self.audio_processor and self.audio_processor.is_listening:
                    logger.info("Received IDLE command")
                    self.stop_listening()
                elif command == "SPEAK":
                    logger.info("Received SPEAK command")
                    # We'll handle speaking in check_for_responses
                    if hasattr(self, 'tts_adapter') and self.tts_adapter:
                        self.status_label.setText("Speaking...")
                        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2ecc71;")
            except Exception as e:
                logger.error(f"Error reading command file: {e}")
    
    def check_for_responses(self):
        """Check for response files to speak."""
        if os.path.exists(RESPONSE_FILE):
            try:
                # Read the response
                with open(RESPONSE_FILE, 'r') as f:
                    response = f.read().strip()
                
                logger.info(f"Found response to speak: {response[:50]}{'...' if len(response) > 50 else ''}")
                
                # Delete the file immediately to prevent duplicate processing
                try:
                    os.remove(RESPONSE_FILE)
                except Exception as e:
                    logger.warning(f"Error removing response file: {e}")
                
                # If TTS is not ready yet, show a message and return
                if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
                    logger.warning("TTS not ready yet, cannot speak response")
                    self.transcription_label.setText("Response received but TTS not ready yet")
                    return
                
                # Speak the response using the TTS adapter
                if response:
                    self.tts_adapter.speak(response)
                
            except Exception as e:
                logger.error(f"Error processing response file: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop audio processor if it exists
        if hasattr(self, 'audio_processor') and self.audio_processor:
            self.audio_processor.stop_listening()
        
        # Write a UI_CLOSED command to the command file
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write("UI_CLOSED")
            logger.info("Created UI_CLOSED command file")
        except Exception as e:
            logger.error(f"Error creating command file: {e}")
        
        super().closeEvent(event)


def run_ui():
    """Run the PyQt speech UI."""
    app = QApplication(sys.argv)
    window = PyQtSpeechUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Run the UI
    sys.exit(run_ui())