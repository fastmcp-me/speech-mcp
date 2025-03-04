import sys
import os
import json
import logging
import time
import threading
import tempfile
import subprocess
import psutil
from typing import Dict, Optional, Callable

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS

mcp = FastMCP("speech")

# Path to save speech state
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_state.json")

# Default speech state
DEFAULT_SPEECH_STATE = {
    "listening": False,
    "speaking": False,
    "last_transcript": "",
    "last_response": "",
    "ui_active": False,
    "ui_process_id": None
}

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech-mcp-server.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load speech state from file or use default
def load_speech_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                return state
        else:
            return DEFAULT_SPEECH_STATE.copy()
    except Exception as e:
        logger.error(f"Error loading speech state: {e}")
        return DEFAULT_SPEECH_STATE.copy()

# Save speech state to file
def save_speech_state(state, create_response_file=False):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        
        # Only create response file if specifically requested
        if create_response_file:
            # Create or update response file for UI communication
            # This helps ensure the UI is properly notified of state changes
            RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "response.txt")
            if state.get("speaking", False):
                # If speaking, write the response to the file for the UI to pick up
                with open(RESPONSE_FILE, 'w') as f:
                    f.write(state.get("last_response", ""))
        
        # Create a special command file to signal state changes to the UI
        COMMAND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_command.txt")
        command = ""
        if state.get("listening", False):
            command = "LISTEN"
        elif state.get("speaking", False):
            command = "SPEAK"
        else:
            command = "IDLE"
        
        with open(COMMAND_FILE, 'w') as f:
            f.write(command)
        
        logger.debug(f"Saved state and sent UI command: {command}")
    except Exception as e:
        logger.error(f"Error saving speech state: {e}")

# Initialize speech state
speech_state = load_speech_state()

# Audio parameters
CHUNK = 1024
FORMAT = None  # Will be set when PyAudio is imported
CHANNELS = 1
RATE = 16000

# Speech recognition and TTS engines
whisper_model = None
tts_engine = None

def initialize_speech_recognition():
    """Initialize speech recognition"""
    global whisper_model
    
    if whisper_model is not None:
        logger.info("Speech recognition already initialized")
        return True
    
    try:
        logger.info("Loading faster-whisper speech recognition model...")
        print("Loading faster-whisper speech recognition model... This may take a moment.")
        
        import faster_whisper
        # Load the small model for a good balance of speed and accuracy
        # Using CPU as default for compatibility
        whisper_model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")
        
        logger.info("faster-whisper model loaded successfully")
        print("faster-whisper speech recognition model loaded successfully!")
        return True
    except ImportError as e:
        logger.error(f"Failed to load faster-whisper: {e}")
        print(f"ERROR: Failed to load faster-whisper module: {e}")
        print("Trying to fall back to SpeechRecognition library...")
        return initialize_speech_recognition_fallback()

def initialize_speech_recognition_fallback():
    """Initialize fallback speech recognition using SpeechRecognition library"""
    try:
        import speech_recognition as sr
        logger.info("SpeechRecognition successfully loaded")
        print("SpeechRecognition library loaded successfully!")
        return True
    except ImportError as e:
        logger.error(f"Failed to load SpeechRecognition: {e}")
        print(f"ERROR: Failed to load SpeechRecognition module: {e}")
        print("Please install it with: pip install SpeechRecognition")
        return False

def initialize_tts():
    """Initialize text-to-speech"""
    global tts_engine
    
    if tts_engine is not None:
        logger.info("TTS already initialized")
        return True
    
    # Always prioritize Kokoro as the primary TTS engine if available
    try:
        print("Initializing Kokoro as primary TTS engine...")
        logger.info("Initializing Kokoro as primary TTS engine")
        
        # Import and initialize Kokoro adapter
        try:
            from speech_mcp.tts_adapters.kokoro_adapter import KokoroTTS
            
            # Initialize with Kokoro voice settings
            tts_engine = KokoroTTS(voice="af_heart", lang_code="a", speed=1.0)
            logger.info("Kokoro TTS adapter initialized successfully as primary TTS engine")
            print("Kokoro TTS adapter initialized successfully as primary TTS engine!")
            
            # Log available voices
            voices = tts_engine.get_available_voices()
            logger.debug(f"Available Kokoro TTS voices: {len(voices)}")
            for i, voice in enumerate(voices):
                logger.debug(f"Voice {i}: {voice}")
            print(f"Available Kokoro voices: {', '.join(voices[:5])}{' and more...' if len(voices) > 5 else ''}")
            return True
        except ImportError as e:
            # If the adapter is available but Kokoro itself is not installed
            logger.warning(f"Kokoro package not available: {e}. Falling back to pyttsx3.")
            print("WARNING: Kokoro package not available. Falling back to pyttsx3.")
            raise ImportError("Kokoro package not installed")
        
    except ImportError as e:
        logger.warning(f"Kokoro adapter not available: {e}. Falling back to pyttsx3.")
        print("WARNING: Kokoro adapter not available. Falling back to pyttsx3.")
        
        # Fall back to pyttsx3
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            logger.info("pyttsx3 text-to-speech engine initialized as fallback")
            print("pyttsx3 text-to-speech engine initialized as fallback!")
            
            # Log available voices
            voices = tts_engine.getProperty('voices')
            logger.debug(f"Available pyttsx3 voices: {len(voices)}")
            for i, voice in enumerate(voices):
                logger.debug(f"Voice {i}: {voice.id} - {voice.name}")
            return True
        except ImportError as e:
            logger.warning(f"pyttsx3 not available: {e}. Text-to-speech will be simulated.")
            print("WARNING: pyttsx3 not available. Text-to-speech will be simulated.")
            return False
        except Exception as e:
            logger.error(f"Error initializing text-to-speech engine: {e}")
            print(f"WARNING: Error initializing text-to-speech: {e}. Text-to-speech will be simulated.")
            return False
    except Exception as e:
        logger.error(f"Error initializing Kokoro TTS adapter: {e}")
        print(f"WARNING: Error initializing Kokoro TTS adapter: {e}. Falling back to pyttsx3.")
        
        # Fall back to pyttsx3
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            logger.info("pyttsx3 text-to-speech engine initialized as fallback")
            print("pyttsx3 text-to-speech engine initialized as fallback!")
            
            # Log available voices
            voices = tts_engine.getProperty('voices')
            logger.debug(f"Available pyttsx3 voices: {len(voices)}")
            for i, voice in enumerate(voices):
                logger.debug(f"Voice {i}: {voice.id} - {voice.name}")
            return True
        except ImportError as e:
            logger.warning(f"pyttsx3 not available: {e}. Text-to-speech will be simulated.")
            print("WARNING: pyttsx3 not available. Text-to-speech will be simulated.")
            return False
        except Exception as e:
            logger.error(f"Error initializing text-to-speech engine: {e}")
            print(f"WARNING: Error initializing text-to-speech: {e}. Text-to-speech will be simulated.")
            return False

def ensure_ui_is_running():
    """Ensure the UI process is running"""
    global speech_state
    
    # Check if UI is already active
    if speech_state.get("ui_active", False) and speech_state.get("ui_process_id"):
        # Check if the process is actually running
        try:
            process_id = speech_state["ui_process_id"]
            if psutil.pid_exists(process_id):
                process = psutil.Process(process_id)
                if process.status() != psutil.STATUS_ZOMBIE:
                    logger.info(f"UI process already running with PID {process_id}")
                    return True
        except Exception as e:
            logger.error(f"Error checking UI process: {e}")
    
    # Check for any existing UI processes by looking for Python processes running speech_mcp.ui
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and len(cmdline) >= 3:
                    if 'python' in cmdline[0].lower() and '-m' in cmdline[1] and 'speech_mcp.ui' in cmdline[2]:
                        # Found an existing UI process
                        logger.info(f"Found existing UI process with PID {proc.info['pid']}")
                        
                        # Update our state to track this process
                        speech_state["ui_active"] = True
                        speech_state["ui_process_id"] = proc.info['pid']
                        save_speech_state(speech_state, False)
                        
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.error(f"Error searching for existing UI processes: {e}")
    
    # No UI process found, we'll need to start one using the launch_ui tool
    logger.info("No UI process found, need to use launch_ui tool")
    return False

def record_audio():
    """Record audio from the microphone and return the audio data"""
    global FORMAT
    
    try:
        import pyaudio
        import numpy as np
        
        if FORMAT is None:
            FORMAT = pyaudio.paInt16
        
        logger.info("Starting audio recording")
        print("\nRecording audio... Speak now.")
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Find the best input device
        selected_device_index = None
        
        # Get all available audio devices
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        logger.info(f"Found {numdevices} audio devices:")
        
        for i in range(numdevices):
            try:
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                device_name = device_info.get('name')
                max_input_channels = device_info.get('maxInputChannels')
                
                logger.info(f"Device {i}: {device_name}")
                logger.info(f"  Max Input Channels: {max_input_channels}")
                
                # Only consider input devices
                if max_input_channels > 0:
                    print(f"Found input device: {device_name}")
                    
                    # Prefer non-default devices as they're often external mics
                    if selected_device_index is None or 'default' not in device_name.lower():
                        selected_device_index = i
                        logger.info(f"Selected input device: {device_name} (index {i})")
                        print(f"Using input device: {device_name}")
            except Exception as e:
                logger.warning(f"Error checking device {i}: {e}")
        
        if selected_device_index is None:
            logger.warning("No suitable input device found, using default")
            print("No suitable input device found, using default")
        
        # Start recording
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=selected_device_index,
            frames_per_buffer=CHUNK
        )
        
        logger.info("Recording started")
        
        # Initialize variables for silence detection
        audio_frames = []
        silence_threshold = 0.008  # Reduced threshold to be more sensitive to quiet speech
        silence_duration = 0
        max_silence = 5.0  # 5 seconds of silence to stop recording (increased from 1.5s)
        check_interval = 0.1
        
        # Record until silence is detected
        while silence_duration < max_silence:
            # Read audio chunk
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_frames.append(data)
            
            # Process audio for silence detection
            audio_data = np.frombuffer(data, dtype=np.int16)
            normalized = audio_data.astype(float) / 32768.0
            amplitude = np.abs(normalized).mean()
            
            # Log amplitude periodically
            if len(audio_frames) % 20 == 0:
                logger.debug(f"Current audio amplitude: {amplitude:.6f}")
                
                # Ensure UI state is still set to listening
                if not speech_state.get("listening", False):
                    speech_state["listening"] = True
                    save_speech_state(speech_state, False)
            
            # Check for silence
            if amplitude < silence_threshold:
                silence_duration += check_interval
                if silence_duration >= 1.0 and silence_duration % 1.0 < check_interval:
                    logger.debug(f"Silence detected for {silence_duration:.1f}s, amplitude: {amplitude:.6f}")
            else:
                if silence_duration > 0:
                    logger.debug(f"Speech resumed after {silence_duration:.1f}s of silence, amplitude: {amplitude:.6f}")
                silence_duration = 0
            
            time.sleep(check_interval)
        
        logger.info(f"Silence threshold reached after {silence_duration:.1f}s, stopping recording")
        print("Silence detected. Processing speech...")
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        logger.info(f"Recording stopped, captured {len(audio_frames)} audio frames")
        
        # Save the recorded audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
            # Create a WAV file from the recorded frames
            logger.debug(f"Creating WAV file at {temp_audio_path}")
            import wave
            wf = wave.open(temp_audio_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(audio_frames))
            wf.close()
            
            # Get file size for logging
            file_size = os.path.getsize(temp_audio_path)
            logger.debug(f"WAV file created, size: {file_size} bytes")
        
        logger.info(f"Audio saved to temporary file: {temp_audio_path}")
        return temp_audio_path
    
    except Exception as e:
        logger.error(f"Error recording audio: {e}", exc_info=True)
        print(f"ERROR: Failed to record audio: {e}")
        raise Exception(f"Error recording audio: {str(e)}")

def transcribe_audio(audio_file_path):
    """Transcribe audio file using faster-whisper"""
    global whisper_model
    
    try:
        if whisper_model is None:
            if not initialize_speech_recognition():
                raise Exception("Failed to initialize speech recognition")
        
        logger.info("Transcribing audio with faster-whisper...")
        print("Transcribing audio with faster-whisper...")
        
        transcription_start = time.time()
        segments, info = whisper_model.transcribe(audio_file_path, beam_size=5)
        
        # Collect all segments to form the complete transcription
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        
        transcription = transcription.strip()
        transcription_time = time.time() - transcription_start
        
        logger.info(f"Transcription completed in {transcription_time:.2f}s: {transcription}")
        logger.debug(f"Transcription info: {info}")
        print(f"Transcription complete: \"{transcription}\"")
        
        # Clean up the temporary file
        try:
            logger.debug(f"Removing temporary WAV file: {audio_file_path}")
            os.unlink(audio_file_path)
        except Exception as e:
            logger.error(f"Error removing temporary file: {e}")
        
        return transcription
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}", exc_info=True)
        print(f"ERROR: Failed to transcribe audio: {e}")
        
        # Try fallback if available
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data)
                logger.info(f"Fallback transcription completed: {transcription}")
                print(f"Fallback transcription complete: \"{transcription}\"")
                return transcription
        except Exception as fallback_error:
            logger.error(f"Fallback transcription also failed: {fallback_error}")
            print(f"ERROR: Fallback transcription also failed: {fallback_error}")
            raise Exception(f"Error transcribing audio: {str(e)}")

def speak_text(text):
    """Speak text using TTS engine"""
    global tts_engine
    
    if not text:
        logger.warning("Empty text provided to speak_text function")
        raise McpError(
            ErrorData(
                INVALID_PARAMS,
                "No text provided to speak."
            )
        )
    
    # Set speaking state
    speech_state["speaking"] = True
    speech_state["last_response"] = text
    
    # Save state but don't create response file - we'll handle TTS directly
    save_speech_state(speech_state, False)
    
    try:
        logger.info(f"Speaking text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"\nSpeaking: \"{text}\"")
        
        # Initialize TTS if not already done
        if tts_engine is None:
            if not initialize_tts():
                # If TTS initialization fails, simulate speech with a delay
                logger.warning("TTS not available, simulating speech timing")
                speaking_duration = len(text) * 0.05  # 50ms per character
                logger.debug(f"Simulating speech for {speaking_duration:.2f} seconds")
                time.sleep(speaking_duration)
                logger.info("Simulated speech completed")
                print("Simulated speech completed.")
                
                # Update state
                speech_state["speaking"] = False
                save_speech_state(speech_state, False)
                return f"Simulated speaking: {text}"
        
        # Use TTS engine to speak text directly without going through the UI
        if hasattr(tts_engine, 'speak'):
            # Use the speak method directly (Kokoro adapter)
            tts_start = time.time()
            
            # Speak the text
            tts_engine.speak(text)
            tts_duration = time.time() - tts_start
            logger.info(f"TTS completed in {tts_duration:.2f} seconds")
            print("Speech completed.")
        else:
            # Use pyttsx3 directly
            tts_start = time.time()
            
            # Speak the text
            tts_engine.say(text)
            tts_engine.runAndWait()
            tts_duration = time.time() - tts_start
            logger.info(f"Speech completed in {tts_duration:.2f} seconds")
            print("Speech completed.")
        
        # Update state
        speech_state["speaking"] = False
        save_speech_state(speech_state, False)
        
        logger.info("Done speaking")
        print("Done speaking.")
        return f"Spoke: {text}"
    
    except Exception as e:
        # Update state on error
        speech_state["speaking"] = False
        save_speech_state(speech_state, False)
        
        logger.error(f"Error during text-to-speech: {e}", exc_info=True)
        print(f"ERROR: Failed to speak text: {e}")
        
        # Simulate speech with a delay as fallback
        speaking_duration = len(text) * 0.05  # 50ms per character
        logger.debug(f"Simulating speech for {speaking_duration:.2f} seconds after error")
        time.sleep(speaking_duration)
        
        return f"Error speaking text: {str(e)}"

def listen_for_speech() -> str:
    """Listen for speech and return transcription"""
    global speech_state
    
    # Set listening state
    speech_state["listening"] = True
    save_speech_state(speech_state, False)
    
    logger.info("Starting to listen for speech input...")
    print("\nListening for speech input... Speak now.")
    
    try:
        # Record audio
        audio_file_path = record_audio()
        
        # Transcribe audio
        transcription = transcribe_audio(audio_file_path)
        
        logger.info(f"Received transcription: {transcription}")
        print(f"Transcription received: \"{transcription}\"")
        
        # Update state
        speech_state["listening"] = False
        speech_state["last_transcript"] = transcription
        save_speech_state(speech_state, False)
        
        return transcription
    
    except Exception as e:
        # Update state on error
        speech_state["listening"] = False
        save_speech_state(speech_state, False)
        
        logger.error(f"Error during speech recognition: {e}", exc_info=True)
        print(f"ERROR: Speech recognition failed: {e}")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"Error during speech recognition: {str(e)}"
            )
        )

def cleanup_ui_process():
    """Clean up the UI process when the server shuts down"""
    global speech_state
    
    if speech_state.get("ui_active", False) and speech_state.get("ui_process_id"):
        try:
            process_id = speech_state["ui_process_id"]
            if psutil.pid_exists(process_id):
                logger.info(f"Terminating UI process with PID {process_id}")
                process = psutil.Process(process_id)
                process.terminate()
                try:
                    process.wait(timeout=3)
                except psutil.TimeoutExpired:
                    logger.warning(f"UI process {process_id} did not terminate, forcing kill")
                    process.kill()
            
            # Update state
            speech_state["ui_active"] = False
            speech_state["ui_process_id"] = None
            save_speech_state(speech_state, False)
            
            logger.info("UI process terminated")
        except Exception as e:
            logger.error(f"Error terminating UI process: {e}")

# Register cleanup function to be called on exit
import atexit
atexit.register(cleanup_ui_process)

@mcp.tool()
def launch_ui() -> str:
    """
    Launch the speech UI.
    
    This will start the speech UI window that shows the microphone status and speech visualization.
    The UI is required for visual feedback during speech recognition.
    
    Returns:
        A message indicating whether the UI was successfully launched.
    """
    global speech_state
    
    logger.info("launch_ui() called")
    print("Launching speech UI...")
    
    # Check if UI is already running
    if ensure_ui_is_running():
        logger.info("UI is already running, no need to launch")
        return "Speech UI is already running."
    
    # Start a new UI process
    try:
        # Check for any existing UI processes first to prevent duplicates
        existing_ui = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and len(cmdline) >= 3:
                    if 'python' in cmdline[0].lower() and '-m' in cmdline[1] and 'speech_mcp.ui' in cmdline[2]:
                        # Found an existing UI process
                        logger.info(f"Found existing UI process with PID {proc.info['pid']}")
                        existing_ui = True
                        
                        # Update our state to track this process
                        speech_state["ui_active"] = True
                        speech_state["ui_process_id"] = proc.info['pid']
                        save_speech_state(speech_state, False)
                        
                        return f"Speech UI is already running with PID {proc.info['pid']}."
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Start a new UI process if none exists
        if not existing_ui:
            logger.info("Starting new UI process...")
            print("Starting speech UI process...")
            
            # Create a command file path to check for UI readiness
            COMMAND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_command.txt")
            
            # Clear any existing command file
            try:
                if os.path.exists(COMMAND_FILE):
                    os.remove(COMMAND_FILE)
            except Exception as e:
                logger.warning(f"Could not clear existing command file: {e}")
            
            # Start the UI process
            ui_process = subprocess.Popen(
                [sys.executable, "-m", "speech_mcp.ui"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Update the speech state
            speech_state["ui_active"] = True
            speech_state["ui_process_id"] = ui_process.pid
            save_speech_state(speech_state, False)
            
            logger.info(f"UI process started with PID {ui_process.pid}")
            print(f"Speech UI started with PID {ui_process.pid}")
            
            # Wait for UI to fully initialize by checking for the UI_READY command
            max_wait_time = 10  # Maximum wait time in seconds
            wait_interval = 0.2  # Check every 200ms
            waited_time = 0
            ui_ready = False
            
            print("Waiting for UI to initialize...")
            logger.info("Waiting for UI to initialize...")
            
            while waited_time < max_wait_time:
                # Check if the process is still running
                if not psutil.pid_exists(ui_process.pid):
                    logger.error("UI process terminated unexpectedly")
                    print("ERROR: UI process terminated unexpectedly")
                    return "ERROR: UI process terminated unexpectedly."
                
                # Check if the command file exists and contains UI_READY
                if os.path.exists(COMMAND_FILE):
                    try:
                        with open(COMMAND_FILE, 'r') as f:
                            command = f.read().strip()
                            if command == "UI_READY":
                                ui_ready = True
                                logger.info("UI reported ready state")
                                print("UI is fully initialized and ready!")
                                break
                    except Exception as e:
                        logger.warning(f"Error reading command file: {e}")
                
                # Wait before checking again
                time.sleep(wait_interval)
                waited_time += wait_interval
                
                # Log progress periodically
                if int(waited_time) % 1 == 0 and waited_time > 0:
                    logger.debug(f"Waiting for UI to initialize: {waited_time:.1f}s elapsed")
                    print(f"Waiting for UI to initialize: {waited_time:.1f}s elapsed")
            
            if ui_ready:
                return f"Speech UI launched successfully with PID {ui_process.pid} and is ready."
            else:
                logger.warning(f"UI did not report ready state within {max_wait_time}s, but process is running")
                print(f"WARNING: UI started but did not report ready state within {max_wait_time}s")
                return f"Speech UI launched with PID {ui_process.pid}, but readiness state is unknown."
    except Exception as e:
        logger.error(f"Error starting UI process: {e}")
        print(f"ERROR: Failed to start speech UI: {e}")
        return f"ERROR: Failed to launch Speech UI: {str(e)}"

@mcp.tool()
def start_conversation() -> str:
    """
    Start a voice conversation by beginning to listen.
    
    This will initialize the speech recognition system and immediately start listening for user input.
    
    Returns:
        The transcription of the user's speech.
    """
    global speech_state
    
    logger.info("Starting new conversation with start_conversation()")
    print("Starting new conversation with start_conversation()")
    
    # Force reset the speech state to avoid any stuck states
    speech_state = DEFAULT_SPEECH_STATE.copy()
    save_speech_state(speech_state, False)
    logger.info("Reset speech state to defaults")
    print("Reset speech state to defaults")
    
    # Initialize speech recognition if not already done
    if not initialize_speech_recognition():
        logger.error("Failed to initialize speech recognition in start_conversation()")
        print("ERROR: Failed to initialize speech recognition in start_conversation()")
        return "ERROR: Failed to initialize speech recognition."
    
    # Check if UI is running but don't launch it automatically
    ui_running = ensure_ui_is_running()
    if not ui_running:
        logger.warning("Speech UI is not running. Use launch_ui() to start it for visual feedback.")
        print("WARNING: Speech UI is not running. Use launch_ui() for visual feedback.")
    
    # Start listening
    try:
        logger.info("Beginning to listen for speech in start_conversation()")
        print("INFO: Beginning to listen for speech in start_conversation()")
        
        # Set listening state before starting to ensure UI shows the correct state
        speech_state["listening"] = True
        save_speech_state(speech_state, False)
        
        # Create a special command file to signal LISTEN state to the UI
        # This ensures the audio blips are played
        COMMAND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_command.txt")
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write("LISTEN")
            logger.debug("Sent LISTEN command to UI")
        except Exception as e:
            logger.error(f"Error writing LISTEN command: {e}")
        
        # Use a queue to get the result from the thread
        import queue
        result_queue = queue.Queue()
        
        def listen_and_queue():
            try:
                result = listen_for_speech()
                result_queue.put(result)
            except Exception as e:
                logger.error(f"Error in listen_and_queue: {e}")
                print(f"ERROR: Error in listen_and_queue: {e}")
                result_queue.put(f"ERROR: {str(e)}")
        
        # Start the thread
        listen_thread = threading.Thread(target=listen_and_queue)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Wait for the result with a timeout
        timeout = 600  # 10 minutes timeout
        try:
            logger.debug(f"Waiting for transcription with {timeout}s timeout")
            print(f"DEBUG: Waiting for transcription with {timeout}s timeout")
            transcription = result_queue.get(timeout=timeout)
            logger.info(f"start_conversation() completed successfully with transcription: {transcription}")
            print(f"INFO: start_conversation() completed successfully with transcription: {transcription}")
            
            # Signal that we're done listening
            speech_state["listening"] = False
            save_speech_state(speech_state, False)
            
            # Create a special command file to signal IDLE state to the UI
            # This ensures the audio blips are played
            try:
                with open(COMMAND_FILE, 'w') as f:
                    f.write("IDLE")
                logger.debug("Sent IDLE command to UI")
            except Exception as e:
                logger.error(f"Error writing IDLE command: {e}")
            
            return transcription
        except queue.Empty:
            logger.error(f"Timeout waiting for transcription after {timeout} seconds")
            print(f"ERROR: Timeout waiting for transcription after {timeout} seconds")
            
            # Update state to stop listening
            speech_state["listening"] = False
            save_speech_state(speech_state, False)
            
            # Signal that we're done listening
            try:
                with open(COMMAND_FILE, 'w') as f:
                    f.write("IDLE")
                logger.debug("Sent IDLE command to UI")
            except Exception as e:
                logger.error(f"Error writing IDLE command: {e}")
            
            # Create an emergency transcription
            emergency_message = f"ERROR: Timeout waiting for speech transcription after {timeout} seconds."
            logger.warning(f"Returning emergency message: {emergency_message}")
            print(f"Returning emergency message: {emergency_message}")
            return emergency_message
    
    except Exception as e:
        logger.error(f"Error starting conversation: {e}", exc_info=True)
        print(f"ERROR: Error starting conversation: {e}")
        
        # Update state to stop listening
        speech_state["listening"] = False
        save_speech_state(speech_state, False)
        
        # Signal that we're done listening
        COMMAND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_command.txt")
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write("IDLE")
            logger.debug("Sent IDLE command to UI")
        except Exception as e:
            logger.error(f"Error writing IDLE command: {e}")
        
        # Return an error message instead of raising an exception
        error_message = f"ERROR: Failed to start conversation: {str(e)}"
        logger.warning(f"Returning error message: {error_message}")
        print(f"Returning error message: {error_message}")
        return error_message

@mcp.tool()
def reply(text: str) -> str:
    """
    Speak the provided text and then listen for a response.
    
    This will speak the given text and then immediately start listening for user input.
    
    Args:
        text: The text to speak to the user
        
    Returns:
        The transcription of the user's response.
    """
    global speech_state
    
    logger.info(f"reply() called with text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"reply() called with text: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # Reset listening and speaking states to ensure we're in a clean state
    speech_state["listening"] = False
    speech_state["speaking"] = False
    save_speech_state(speech_state, False)
    
    # Clear any existing response file to prevent double-speaking
    RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "response.txt")
    try:
        if os.path.exists(RESPONSE_FILE):
            os.remove(RESPONSE_FILE)
            logger.debug("Removed existing response file")
    except Exception as e:
        logger.warning(f"Error removing existing response file: {e}")
    
    # Speak the text
    try:
        logger.info("Speaking text in reply()")
        print("INFO: Speaking text in reply()")
        speak_text(text)
        
        # Add a small delay to ensure speaking is complete
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Error speaking text in reply(): {e}", exc_info=True)
        print(f"ERROR: Error speaking text in reply(): {e}")
        return f"ERROR: Failed to speak text: {str(e)}"
    
    # Check if UI is running but don't launch it automatically
    ui_running = ensure_ui_is_running()
    if not ui_running:
        logger.warning("Speech UI is not running. Use launch_ui() to start it for visual feedback.")
        print("WARNING: Speech UI is not running. Use launch_ui() for visual feedback.")
    
    # Start listening for response
    try:
        logger.info("Beginning to listen for response in reply()")
        print("INFO: Beginning to listen for response in reply()")
        
        # Use a queue to get the result from the thread
        import queue
        result_queue = queue.Queue()
        
        def listen_and_queue():
            try:
                result = listen_for_speech()
                result_queue.put(result)
            except Exception as e:
                logger.error(f"Error in listen_and_queue: {e}")
                print(f"ERROR: Error in listen_and_queue: {e}")
                result_queue.put(f"ERROR: {str(e)}")
        
        # Start the thread
        listen_thread = threading.Thread(target=listen_and_queue)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Wait for the result with a timeout
        timeout = 600  # 10 minutes timeout
        try:
            logger.debug(f"Waiting for transcription with {timeout}s timeout")
            print(f"DEBUG: Waiting for transcription with {timeout}s timeout")
            transcription = result_queue.get(timeout=timeout)
            logger.info(f"reply() completed successfully with transcription: {transcription}")
            print(f"INFO: reply() completed successfully with transcription: {transcription}")
            return transcription
        except queue.Empty:
            logger.error(f"Timeout waiting for transcription after {timeout} seconds")
            print(f"ERROR: Timeout waiting for transcription after {timeout} seconds")
            
            # Update state to stop listening
            speech_state["listening"] = False
            save_speech_state(speech_state, False)
            
            # Create an emergency transcription
            emergency_message = f"ERROR: Timeout waiting for speech transcription after {timeout} seconds."
            logger.warning(f"Returning emergency message: {emergency_message}")
            print(f"Returning emergency message: {emergency_message}")
            return emergency_message
    
    except Exception as e:
        logger.error(f"Error listening for response in reply(): {e}", exc_info=True)
        print(f"ERROR: Error listening for response in reply(): {e}")
        
        # Update state to stop listening
        speech_state["listening"] = False
        save_speech_state(speech_state, False)
        
        # Return an error message instead of raising an exception
        error_message = f"ERROR: Failed to listen for response: {str(e)}"
        logger.warning(f"Returning error message: {error_message}")
        print(f"Returning error message: {error_message}")
        return error_message

@mcp.resource(uri="mcp://speech/usage_guide")
def usage_guide() -> str:
    """
    Return the usage guide for the Speech MCP.
    """
    return """
    # Speech MCP Usage Guide
    
    This MCP extension provides voice interaction capabilities with a simplified interface.
    
    ## How to Use
    
    1. Launch the speech UI for visual feedback (optional but recommended):
       ```
       launch_ui()
       ```
       This starts the visual interface that shows when the microphone is active.
       
    2. Start a conversation:
       ```
       user_input = start_conversation()
       ```
       This initializes the speech recognition system and immediately starts listening for user input.
       Note: The first time you run this, it will download the faster-whisper model which may take a moment.
    
    3. Reply to the user and get their response:
       ```
       user_response = reply("Your response text here")
       ```
       This speaks your response and then listens for the user's reply.
    
    ## Typical Workflow
    
    1. Start the conversation to get the initial user input
    2. Process the transcribed speech
    3. Use the reply function to respond and get the next user input
    4. Repeat steps 2-3 for a continuous conversation
    
    ## Example Conversation Flow
    
    ```python
    # Start the conversation
    user_input = start_conversation()
    
    # Process the input and generate a response
    # ...
    
    # Reply to the user and get their response
    follow_up = reply("Here's my response to your question.")
    
    # Process the follow-up and reply again
    reply("I understand your follow-up question. Here's my answer.")
    ```
    
    ## Tips
    
    - For best results, use a quiet environment and speak clearly
    - Use the `launch_ui()` function to start the visual interface:
      - The UI shows when the microphone is active and listening
      - A blue pulsing circle indicates active listening
      - A green circle indicates the system is speaking
      - Only one UI instance can run at a time (prevents duplicates)
    - The system automatically detects silence to know when you've finished speaking
      - Silence detection waits for 5 seconds of quiet before stopping recording
      - This allows for natural pauses in speech without cutting off
    - The overall listening timeout is set to 10 minutes to allow for extended thinking time or long pauses
    """

@mcp.resource(uri="mcp://speech/kokoro_tts")
def kokoro_tts_guide() -> str:
    """
    Return information about the Kokoro TTS adapter.
    """
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "kokoro_tts_adapter.md"), 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading Kokoro TTS guide: {e}")
        return """
        # Kokoro TTS Adapter
        
        Kokoro is a high-quality neural text-to-speech engine that can be used with speech-mcp.
        
        To install Kokoro, run:
        ```
        python scripts/install_kokoro.py
        ```
        
        For more information, see the documentation in the speech-mcp repository.
        """