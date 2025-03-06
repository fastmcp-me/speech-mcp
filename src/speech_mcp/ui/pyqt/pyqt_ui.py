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
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QPen, QFont

# Setup logging
logger = logging.getLogger(__name__)

# Path to save speech state - same as in server.py
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "speech_state.json")
TRANSCRIPTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "transcription.txt")
RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "response.txt")
COMMAND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ui_command.txt")

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Path to audio notification files
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "audio")
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
        
        self.is_speaking = True
        self.speaking_started.emit()
        
        # Start speaking in a separate thread
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
        return True
    
    def _speak_thread(self, text):
        """Thread function for speaking text"""
        try:
            logger.info(f"Speaking text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
            
            # Use the appropriate TTS engine
            if hasattr(self.tts_engine, 'speak'):
                # This is the Kokoro adapter
                result = self.tts_engine.speak(text)
                if not result:
                    logger.error("Kokoro TTS failed")
            else:
                # This is pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            logger.info("Speech completed")
        except Exception as e:
            logger.error(f"Error during text-to-speech: {e}")
        finally:
            self.is_speaking = False
            self.speaking_finished.emit()
    
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
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "speech_state.json")
TRANSCRIPTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "transcription.txt")
RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "response.txt")
COMMAND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ui_command.txt")

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Path to audio notification files
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "audio")
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
            
            # Emit the signal with the amplitude
            self.audio_level_updated.emit(amplitude)
            
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
            max_silence = 5.0  # 5 seconds of silence to stop recording
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
        self.audio_levels = [0.0] * 100  # Store recent audio levels
        self.setStyleSheet("background-color: #1e1e1e;")
    
    def update_level(self, level):
        """Update with a new audio level."""
        self.audio_levels.pop(0)
        self.audio_levels.append(level)
        self.update()
    
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
        
        # Draw the waveform lines
        points_per_segment = width / (len(self.audio_levels) - 1)
        for i in range(len(self.audio_levels) - 1):
            x1 = i * points_per_segment
            y1 = mid_height - (self.audio_levels[i] * mid_height)
            x2 = (i + 1) * points_per_segment
            y2 = mid_height - (self.audio_levels[i + 1] * mid_height)
            
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            
            # Mirror the waveform
            y1_mirror = mid_height + (self.audio_levels[i] * mid_height)
            y2_mirror = mid_height + (self.audio_levels[i + 1] * mid_height)
            painter.drawLine(int(x1), int(y1_mirror), int(x2), int(y2_mirror))


class PyQtSpeechUI(QMainWindow):
    """
    Main speech UI window implemented with PyQt.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goose Speech Interface")
        self.resize(500, 300)
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.audio_processor.audio_level_updated.connect(self.update_audio_level)
        self.audio_processor.transcription_ready.connect(self.handle_transcription)
        
        # Initialize TTS adapter
        self.tts_adapter = TTSAdapter()
        self.tts_adapter.speaking_started.connect(self.on_speaking_started)
        self.tts_adapter.speaking_finished.connect(self.on_speaking_finished)
        
        # Create a command file to indicate UI is ready
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write("UI_READY")
            logger.info("Created initial UI_READY command file")
        except Exception as e:
            logger.error(f"Error creating initial command file: {e}")
        
        self.setup_ui()
        
        # Start checking for server commands
        self.command_check_timer = QTimer(self)
        self.command_check_timer.timeout.connect(self.check_for_commands)
        self.command_check_timer.start(100)  # Check every 100ms
        
        # Start checking for response files
        self.response_check_timer = QTimer(self)
        self.response_check_timer.timeout.connect(self.check_for_responses)
        self.response_check_timer.start(100)  # Check every 100ms
        
        # Start with listening mode if requested
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                    if command == "LISTEN":
                        self.start_listening()
            except Exception as e:
                logger.error(f"Error reading command file: {e}")
        
    def setup_ui(self):
        """Set up the UI components."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #7f8c8d;
        """)
        main_layout.addWidget(self.status_label)
        
        # Audio visualizer
        self.visualizer = AudioVisualizer()
        main_layout.addWidget(self.visualizer)
        
        # Transcription display
        self.transcription_label = QLabel("Say something...")
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
        
        # Populate voice combo box
        self.update_voice_list()
        self.voice_combo.currentIndexChanged.connect(self.on_voice_changed)
        
        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(self.voice_combo, 1)  # 1 = stretch factor
        main_layout.addLayout(voice_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.listen_button = QPushButton("Listen")
        self.listen_button.clicked.connect(self.toggle_listening)
        self.listen_button.setStyleSheet("""
            background-color: #00a0e9;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        self.speak_button = QPushButton("Test Voice")
        self.speak_button.clicked.connect(self.test_voice)
        self.speak_button.setStyleSheet("""
            background-color: #27ae60;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.setStyleSheet("""
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
        
        # Set the current selection
        self.voice_combo.setCurrentIndex(selected_index)
        logger.info(f"Voice combo initialized with {len(voices)} voices, selected: {current_voice}")
        print(f"Voice combo initialized with {len(voices)} voices, selected: {current_voice}")
    
    def on_voice_changed(self, index):
        """Handle voice selection change"""
        if index < 0:
            return
        
        voice = self.voice_combo.itemData(index)
        if not voice:
            return
        
        logger.info(f"Voice selection changed to: {voice}")
        self.tts_adapter.set_voice(voice)
    
    def test_voice(self):
        """Test the selected voice"""
        if self.tts_adapter.is_speaking:
            logger.warning("Already speaking, ignoring test request")
            return
        
        self.tts_adapter.speak("This is a test of the selected voice. Hello, I am Goose!")
    
    def update_audio_level(self, level):
        """Update the audio level visualization."""
        self.visualizer.update_level(level)
    
    def handle_transcription(self, text):
        """Handle new transcription text."""
        self.transcription_label.setText(text)
        logger.info(f"New transcription: {text}")
    
    def start_listening(self):
        """Start listening mode."""
        self.audio_processor.start_listening()
        self.status_label.setText("Listening...")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00c8ff;")
        self.listen_button.setText("Stop")
        self.listen_button.setStyleSheet("""
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
    
    def stop_listening(self):
        """Stop listening mode."""
        self.audio_processor.stop_listening()
        self.status_label.setText("Not Listening")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f8c8d;")
        self.listen_button.setText("Listen")
        self.listen_button.setStyleSheet("""
            background-color: #00a0e9;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
    
    def toggle_listening(self):
        """Toggle between listening and not listening states."""
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
                if command == "LISTEN" and not self.audio_processor.is_listening:
                    logger.info("Received LISTEN command")
                    self.start_listening()
                elif command == "IDLE" and self.audio_processor.is_listening:
                    logger.info("Received IDLE command")
                    self.stop_listening()
                elif command == "SPEAK":
                    logger.info("Received SPEAK command")
                    # We'll handle speaking in check_for_responses
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
                
                # Speak the response using the TTS adapter
                if response:
                    self.tts_adapter.speak(response)
                
            except Exception as e:
                logger.error(f"Error processing response file: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
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