import tkinter as tk
import os
import sys
import json
import time
import threading
import logging
import tempfile
import io
from queue import Queue

# Set up logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speech-mcp-ui.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',  # Very simple format for easier parsing
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ]
)
logger = logging.getLogger(__name__)

# Import other dependencies
import numpy as np
import wave
import pyaudio

# For text-to-speech
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_available = True
    logger.info("Text-to-speech engine initialized successfully")
    print("Text-to-speech engine initialized successfully!")
    
    # Log available voices
    voices = tts_engine.getProperty('voices')
    logger.debug(f"Available TTS voices: {len(voices)}")
    for i, voice in enumerate(voices):
        logger.debug(f"Voice {i}: {voice.id} - {voice.name}")
except ImportError as e:
    logger.warning(f"pyttsx3 not available: {e}. Text-to-speech will be simulated.")
    print("WARNING: pyttsx3 not available. Text-to-speech will be simulated.")
    tts_available = False
except Exception as e:
    logger.error(f"Error initializing text-to-speech engine: {e}")
    print(f"WARNING: Error initializing text-to-speech: {e}. Text-to-speech will be simulated.")
    tts_available = False

# These will be imported later when needed
whisper_loaded = False
speech_recognition_loaded = False

# Path to save speech state - same as in server.py
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speech_state.json")
TRANSCRIPTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "transcription.txt")
RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "response.txt")

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

# Import optional dependencies when needed
def load_whisper():
    global whisper_loaded
    try:
        global whisper
        print("Loading Whisper speech recognition model... This may take a moment.")
        import whisper
        whisper_loaded = True
        logger.info("Whisper successfully loaded")
        print("Whisper speech recognition model loaded successfully!")
        return True
    except ImportError as e:
        logger.error(f"Failed to load whisper: {e}")
        print(f"ERROR: Failed to load Whisper module: {e}")
        print("Trying to fall back to SpeechRecognition library...")
        return load_speech_recognition()

def load_speech_recognition():
    global speech_recognition_loaded
    try:
        global sr
        import speech_recognition as sr
        speech_recognition_loaded = True
        logger.info("SpeechRecognition successfully loaded")
        print("SpeechRecognition library loaded successfully!")
        return True
    except ImportError as e:
        logger.error(f"Failed to load SpeechRecognition: {e}")
        print(f"ERROR: Failed to load SpeechRecognition module: {e}")
        print("Please install it with: pip install SpeechRecognition")
        return False

class SimpleSpeechProcessorUI:
    """A very simple speech processor UI that just shows status"""
    def __init__(self, root):
        self.root = root
        self.root.title("Speech MCP")
        self.root.geometry("300x100")
        
        # Initialize basic components
        print("Initializing speech processor...")
        logger.info("Initializing speech processor UI")
        self.ui_active = True
        self.listening = False
        self.speaking = False
        self.last_transcript = ""
        self.last_response = ""
        self.should_update = True
        self.stream = None
        
        # Initialize PyAudio
        print("Initializing audio system...")
        logger.info("Initializing PyAudio system")
        try:
            self.p = pyaudio.PyAudio()
            
            # Log audio device information
            logger.info(f"PyAudio version: {pyaudio.get_portaudio_version()}")
            logger.info(f"Default input device index: {self.p.get_default_input_device_info()['index']}")
            
            # Log all available audio devices
            info = self.p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            logger.debug(f"Found {numdevices} audio devices:")
            
            for i in range(numdevices):
                device_info = self.p.get_device_info_by_host_api_device_index(0, i)
                device_name = device_info.get('name')
                max_input_channels = device_info.get('maxInputChannels')
                max_output_channels = device_info.get('maxOutputChannels')
                
                logger.debug(f"Device {i}: {device_name}")
                logger.debug(f"  Max Input Channels: {max_input_channels}")
                logger.debug(f"  Max Output Channels: {max_output_channels}")
                logger.debug(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
                
                # Print info about input devices
                if max_input_channels > 0:
                    print(f"Found input device: {device_name}")
            
        except Exception as e:
            logger.error(f"Error initializing PyAudio: {e}", exc_info=True)
            print(f"ERROR: Failed to initialize audio system: {e}")
            # Show error in UI
            self.root.after(0, lambda: self.status_label.config(
                text=f"Audio Error: {str(e)[:30]}..."
            ))
        
        # Load speech state
        self.load_speech_state()
        
        # Create the UI components - just a status label
        self.status_label = tk.Label(
            self.root, 
            text="Initializing...", 
            font=('Arial', 16)
        )
        self.status_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Load whisper in a background thread
        print("Checking for speech recognition module...")
        threading.Thread(target=self.initialize_speech_recognition, daemon=True).start()
        
        # Start threads for monitoring state changes
        self.update_thread = threading.Thread(target=self.check_for_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start thread for checking response file
        self.response_thread = threading.Thread(target=self.check_for_responses)
        self.response_thread.daemon = True
        self.response_thread.start()
        
        # Handle window close event
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        print("Speech processor initialization complete!")
        logger.info("Speech processor initialized successfully")
    
    def initialize_speech_recognition(self):
        """Initialize speech recognition in a background thread"""
        if not load_whisper():
            self.root.after(0, lambda: self.status_label.config(
                text="WARNING: Speech recognition not available"
            ))
            return
        
        # Load the whisper model
        try:
            self.root.after(0, lambda: self.status_label.config(
                text="Loading Whisper model..."
            ))
            
            # Load the small model for a good balance of speed and accuracy
            self.whisper_model = whisper.load_model("base")
            
            self.root.after(0, lambda: self.status_label.config(
                text="Ready"
            ))
            
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.root.after(0, lambda: self.status_label.config(
                text=f"Error loading model: {e}"
            ))
    
    def load_speech_state(self):
        """Load the speech state from the file shared with the server"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.ui_active = state.get("ui_active", False)
                    self.listening = state.get("listening", False)
                    self.speaking = state.get("speaking", False)
                    self.last_transcript = state.get("last_transcript", "")
                    self.last_response = state.get("last_response", "")
            else:
                # Default state if file doesn't exist
                self.ui_active = True
                self.listening = False
                self.speaking = False
                self.last_transcript = ""
                self.last_response = ""
                self.save_speech_state()
        except Exception as e:
            logger.error(f"Error loading speech state: {e}")
            # Default state on error
            self.ui_active = True
            self.listening = False
            self.speaking = False
            self.last_transcript = ""
            self.last_response = ""
    
    def save_speech_state(self):
        """Save the speech state to the file shared with the server"""
        try:
            state = {
                "ui_active": self.ui_active,
                "listening": self.listening,
                "speaking": self.speaking,
                "last_transcript": self.last_transcript,
                "last_response": self.last_response
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving speech state: {e}")
    
    def update_ui_from_state(self):
        """Update the UI to reflect the current speech state"""
        if self.listening:
            self.status_label.config(text="Listening...")
        elif self.speaking:
            self.status_label.config(text="Speaking...")
        else:
            self.status_label.config(text="Ready")
    
    def start_listening(self):
        """Start listening for audio input"""
        try:
            logger.info("Starting audio recording")
            
            def audio_callback(in_data, frame_count, time_info, status):
                # Log any audio stream status issues
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
                if hasattr(self, 'audio_frames'):
                    self.audio_frames.append(in_data)
                    
                    # Periodically log audio levels for debugging
                    if len(self.audio_frames) % 20 == 0:  # Log every ~1 second (20 chunks at 1024 samples)
                        try:
                            audio_data = np.frombuffer(in_data, dtype=np.int16)
                            normalized = audio_data.astype(float) / 32768.0
                            amplitude = np.abs(normalized).mean()
                            logger.debug(f"Current audio amplitude: {amplitude:.6f}")
                        except Exception as e:
                            logger.error(f"Error calculating audio level: {e}")
                
                return (in_data, pyaudio.paContinue)
            
            # Initialize audio frames list
            self.audio_frames = []
            
            # Start the audio stream
            logger.debug(f"Opening audio stream with FORMAT={FORMAT}, CHANNELS={CHANNELS}, RATE={RATE}, CHUNK={CHUNK}")
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback
            )
            
            logger.info(f"Audio stream started successfully, stream active: {self.stream.is_active()}")
            print("Microphone activated. Listening for speech...")
            
            # Start a thread to detect silence and stop recording
            threading.Thread(target=self.detect_silence, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}", exc_info=True)
            print(f"Error starting audio: {e}")
            self.listening = False
            self.save_speech_state()
            self.update_ui_from_state()
    
    def detect_silence(self):
        """Detect when the user stops speaking and end recording"""
        try:
            # Wait for initial audio to accumulate
            logger.info("Starting silence detection")
            time.sleep(0.5)
            
            silence_threshold = 0.005  # Reduced from 0.01 to be less sensitive
            silence_duration = 0
            max_silence = 2.0  # Increased from 1.5 to 2.0 seconds
            check_interval = 0.1
            
            logger.debug(f"Silence detection parameters: threshold={silence_threshold}, max_silence={max_silence}s, check_interval={check_interval}s")
            
            # Track audio levels for debugging
            amplitude_history = []
            
            while self.listening and self.stream and silence_duration < max_silence:
                if not hasattr(self, 'audio_frames') or len(self.audio_frames) < 2:
                    time.sleep(check_interval)
                    continue
                
                # Get the latest audio frame
                latest_frame = self.audio_frames[-1]
                audio_data = np.frombuffer(latest_frame, dtype=np.int16)
                normalized = audio_data.astype(float) / 32768.0
                amplitude = np.abs(normalized).mean()
                
                # Add to history (keep last 10 values)
                amplitude_history.append(amplitude)
                if len(amplitude_history) > 10:
                    amplitude_history.pop(0)
                
                if amplitude < silence_threshold:
                    silence_duration += check_interval
                    # Log only when silence is detected
                    if silence_duration >= 0.5 and silence_duration % 0.5 < check_interval:
                        logger.debug(f"Silence detected for {silence_duration:.1f}s, amplitude: {amplitude:.6f}")
                else:
                    if silence_duration > 0:
                        logger.debug(f"Speech resumed after {silence_duration:.1f}s of silence, amplitude: {amplitude:.6f}")
                    silence_duration = 0
                
                time.sleep(check_interval)
            
            # If we exited because of silence detection
            if self.listening and self.stream:
                logger.info(f"Silence threshold reached after {silence_duration:.1f}s, stopping recording")
                logger.debug(f"Final amplitude history: {[f'{a:.6f}' for a in amplitude_history]}")
                self.root.after(0, lambda: self.status_label.config(text="Processing speech..."))
                print("Silence detected. Processing speech...")
                self.process_recording()
                self.stop_listening()
            else:
                if not self.listening:
                    logger.info("Silence detection stopped because listening state changed")
                if not self.stream:
                    logger.info("Silence detection stopped because audio stream was closed")
        
        except Exception as e:
            logger.error(f"Error in silence detection: {e}", exc_info=True)
    
    def process_recording(self):
        """Process the recorded audio and generate a transcription using Whisper"""
        try:
            if not hasattr(self, 'audio_frames') or not self.audio_frames:
                logger.warning("No audio frames to process")
                return
            
            logger.info(f"Processing {len(self.audio_frames)} audio frames")
            
            # Check if we have enough audio data
            total_audio_time = len(self.audio_frames) * (CHUNK / RATE)
            logger.info(f"Total recorded audio: {total_audio_time:.2f} seconds")
            
            if total_audio_time < 0.5:  # Less than half a second of audio
                logger.warning(f"Audio recording too short ({total_audio_time:.2f}s), may not contain speech")
            
            if not hasattr(self, 'whisper_model') or self.whisper_model is None:
                logger.warning("Whisper model not loaded yet")
                self.last_transcript = "Sorry, speech recognition model is still loading. Please try again in a moment."
                with open(TRANSCRIPTION_FILE, 'w') as f:
                    f.write(self.last_transcript)
                return
            
            # Save the recorded audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
                # Create a WAV file from the recorded frames
                logger.debug(f"Creating WAV file at {temp_audio_path}")
                wf = wave.open(temp_audio_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
                
                # Get file size for logging
                file_size = os.path.getsize(temp_audio_path)
                logger.debug(f"WAV file created, size: {file_size} bytes")
            
            logger.info(f"Audio saved to temporary file: {temp_audio_path}")
            
            # Use Whisper to transcribe the audio
            logger.info("Transcribing audio with Whisper...")
            print("Transcribing audio with Whisper...")
            self.root.after(0, lambda: self.status_label.config(text="Transcribing audio..."))
            
            transcription_start = time.time()
            result = self.whisper_model.transcribe(temp_audio_path)
            transcription_time = time.time() - transcription_start
            
            transcription = result["text"].strip()
            
            logger.info(f"Transcription completed in {transcription_time:.2f}s: {transcription}")
            print(f"Transcription complete: \"{transcription}\"")
            
            # Log segments if available
            if "segments" in result:
                logger.debug(f"Transcription segments: {len(result['segments'])}")
                for i, segment in enumerate(result["segments"]):
                    logger.debug(f"Segment {i}: {segment.get('start', '?')}-{segment.get('end', '?')}s: {segment.get('text', '')}")
            
            # Clean up the temporary file
            try:
                logger.debug(f"Removing temporary WAV file: {temp_audio_path}")
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")
            
            # Update the state with the transcription
            self.last_transcript = transcription
            
            # Write the transcription to a file for the server to read
            try:
                logger.debug(f"Writing transcription to file: {TRANSCRIPTION_FILE}")
                with open(TRANSCRIPTION_FILE, 'w') as f:
                    f.write(transcription)
                logger.debug("Transcription file written successfully")
            except Exception as e:
                logger.error(f"Error writing transcription to file: {e}", exc_info=True)
                raise e
            
            # Update state
            self.save_speech_state()
            
        except Exception as e:
            logger.error(f"Error processing recording: {e}", exc_info=True)
            self.last_transcript = f"Error processing speech: {str(e)}"
            with open(TRANSCRIPTION_FILE, 'w') as f:
                f.write(self.last_transcript)
    
    def stop_listening(self):
        """Stop listening for audio input"""
        try:
            logger.info("Stopping audio recording")
            if self.stream:
                logger.debug(f"Stopping audio stream, stream active: {self.stream.is_active()}")
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                print("Microphone deactivated.")
                logger.info("Audio stream closed successfully")
            else:
                logger.debug("No active audio stream to close")
            
            # Update state
            self.listening = False
            self.save_speech_state()
            self.update_ui_from_state()
            
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}", exc_info=True)
            print(f"Error stopping audio: {e}")
            
            # Make sure we update state even if there's an error
            self.listening = False
            self.save_speech_state()
            self.update_ui_from_state()
    
    def check_for_updates(self):
        """Periodically check for updates to the speech state file"""
        last_modified = 0
        if os.path.exists(STATE_FILE):
            last_modified = os.path.getmtime(STATE_FILE)
        
        while self.should_update:
            try:
                if os.path.exists(STATE_FILE):
                    current_modified = os.path.getmtime(STATE_FILE)
                    if current_modified > last_modified:
                        last_modified = current_modified
                        self.load_speech_state()
                        self.root.after(0, self.update_ui_from_state)
            except Exception as e:
                logger.error(f"Error checking for updates: {e}")
            
            time.sleep(0.5)  # Check every half second
    
    def check_for_responses(self):
        """Periodically check for new responses to speak"""
        while self.should_update:
            try:
                if os.path.exists(RESPONSE_FILE):
                    # Read the response
                    logger.debug(f"Found response file: {RESPONSE_FILE}")
                    try:
                        with open(RESPONSE_FILE, 'r') as f:
                            response = f.read().strip()
                        
                        logger.debug(f"Read response text ({len(response)} chars): {response[:100]}{'...' if len(response) > 100 else ''}")
                    except Exception as e:
                        logger.error(f"Error reading response file: {e}", exc_info=True)
                        time.sleep(0.5)
                        continue
                    
                    # Delete the file
                    try:
                        logger.debug("Removing response file")
                        os.remove(RESPONSE_FILE)
                    except Exception as e:
                        logger.warning(f"Error removing response file: {e}")
                    
                    # Process the response
                    if response:
                        self.last_response = response
                        self.speaking = True
                        self.save_speech_state()
                        self.root.after(0, self.update_ui_from_state)
                        
                        logger.info(f"Speaking text ({len(response)} chars): {response[:100]}{'...' if len(response) > 100 else ''}")
                        print(f"Speaking: \"{response}\"")
                        
                        # Use actual text-to-speech if available
                        if tts_available:
                            try:
                                # Use pyttsx3 for actual speech
                                logger.debug("Using pyttsx3 for text-to-speech")
                                
                                # Log TTS settings
                                rate = tts_engine.getProperty('rate')
                                volume = tts_engine.getProperty('volume')
                                voice = tts_engine.getProperty('voice')
                                logger.debug(f"TTS settings - Rate: {rate}, Volume: {volume}, Voice: {voice}")
                                
                                # Speak the text
                                tts_start = time.time()
                                tts_engine.say(response)
                                tts_engine.runAndWait()
                                tts_duration = time.time() - tts_start
                                
                                logger.info(f"Speech completed in {tts_duration:.2f} seconds")
                                print("Speech completed.")
                            except Exception as e:
                                logger.error(f"Error using text-to-speech: {e}", exc_info=True)
                                print(f"Error using text-to-speech: {e}")
                                # Fall back to simulated speech
                                logger.info("Falling back to simulated speech")
                                speaking_duration = len(response) * 0.05  # 50ms per character
                                time.sleep(speaking_duration)
                        else:
                            # Simulate speaking time if TTS not available
                            logger.debug("TTS not available, simulating speech timing")
                            speaking_duration = len(response) * 0.05  # 50ms per character
                            logger.debug(f"Simulating speech for {speaking_duration:.2f} seconds")
                            time.sleep(speaking_duration)
                        
                        # Update state when done speaking
                        self.speaking = False
                        self.save_speech_state()
                        self.root.after(0, self.update_ui_from_state)
                        print("Done speaking.")
                        logger.info("Done speaking")
            except Exception as e:
                logger.error(f"Error checking for responses: {e}", exc_info=True)
            
            time.sleep(0.5)  # Check every half second
    
    def on_close(self):
        """Handle window close event"""
        try:
            logger.info("Shutting down speech processor")
            print("\nShutting down speech processor...")
            self.should_update = False
            
            if self.stream:
                logger.debug("Stopping audio stream")
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    logger.debug("Audio stream closed successfully")
                except Exception as e:
                    logger.error(f"Error closing audio stream: {e}")
            
            logger.debug("Terminating PyAudio")
            try:
                self.p.terminate()
                logger.debug("PyAudio terminated successfully")
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            
            # Update state to indicate UI is closed
            self.ui_active = False
            self.listening = False
            self.speaking = False
            self.save_speech_state()
            
            print("Speech processor shut down successfully.")
            logger.info("Speech processor shut down successfully")
            
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error shutting down speech processor: {e}", exc_info=True)
            print(f"Error during shutdown: {e}")
            self.root.destroy()

def main():
    """Main entry point for the speech processor"""
    try:
        logger.info("Starting Speech MCP Processor")
        print("\n===== Speech MCP Processor =====")
        print("Starting speech recognition system...")
        
        # Log platform information
        import platform
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python version: {platform.python_version()}")
        
        # Log audio-related environment variables
        audio_env_vars = {k: v for k, v in os.environ.items() if 'AUDIO' in k.upper() or 'PULSE' in k.upper() or 'ALSA' in k.upper()}
        if audio_env_vars:
            logger.debug(f"Audio-related environment variables: {json.dumps(audio_env_vars)}")
        
        # Start the UI
        root = tk.Tk()
        app = SimpleSpeechProcessorUI(root)
        logger.info("Starting Tkinter main loop")
        root.mainloop()
        logger.info("Tkinter main loop exited")
    except Exception as e:
        logger.error(f"Error in speech processor main: {e}", exc_info=True)
        print(f"\nERROR: Failed to start speech processor: {e}")

if __name__ == "__main__":
    main()