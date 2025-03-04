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

# Path to audio notification files
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "audio")
START_LISTENING_SOUND = os.path.join(AUDIO_DIR, "start_listening.wav")
STOP_LISTENING_SOUND = os.path.join(AUDIO_DIR, "stop_listening.wav")

# Import other dependencies
import numpy as np
import wave
import pyaudio

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
        print("Loading faster-whisper speech recognition model... This may take a moment.")
        import faster_whisper
        whisper_loaded = True
        logger.info("faster-whisper successfully loaded")
        print("faster-whisper speech recognition model loaded successfully!")
        return True
    except ImportError as e:
        logger.error(f"Failed to load faster-whisper: {e}")
        print(f"ERROR: Failed to load faster-whisper module: {e}")
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
    """A speech processor UI that shows status and audio waveform visualization"""
    def __init__(self, root):
        self.root = root
        self.root.title("Speech MCP - Voice Interface")
        self.root.geometry("500x500")  # Square shape for better circular visualization
        
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
        
        # Audio visualization parameters
        self.waveform_data = []
        self.waveform_max_points = 100  # Number of points to display in waveform
        self.waveform_update_interval = 50  # Update interval in milliseconds
        
        # Initialize PyAudio with explicit device selection
        print("Initializing audio system...")
        logger.info("Initializing PyAudio system")
        try:
            self.p = pyaudio.PyAudio()
            
            # Log audio device information
            logger.info(f"PyAudio version: {pyaudio.get_portaudio_version()}")
            
            # Get all available audio devices
            info = self.p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            logger.info(f"Found {numdevices} audio devices:")
            
            # Find the best input device
            selected_device = None
            selected_device_index = None
            
            for i in range(numdevices):
                try:
                    device_info = self.p.get_device_info_by_host_api_device_index(0, i)
                    device_name = device_info.get('name')
                    max_input_channels = device_info.get('maxInputChannels')
                    
                    logger.info(f"Device {i}: {device_name}")
                    logger.info(f"  Max Input Channels: {max_input_channels}")
                    logger.info(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
                    
                    # Only consider input devices
                    if max_input_channels > 0:
                        print(f"Found input device: {device_name}")
                        
                        # Prefer non-default devices as they're often external mics
                        if not selected_device or 'default' not in device_name.lower():
                            selected_device = device_info
                            selected_device_index = i
                            
                except Exception as e:
                    logger.warning(f"Error checking device {i}: {e}")
            
            if not selected_device:
                raise Exception("No suitable input device found")
            
            logger.info(f"Selected input device: {selected_device['name']} (index {selected_device_index})")
            print(f"Using input device: {selected_device['name']}")
            
            # Store the selected device info for later use
            self.selected_device_index = selected_device_index
            self.selected_device_info = selected_device
            
        except Exception as e:
            logger.error(f"Error initializing PyAudio: {e}", exc_info=True)
            print(f"ERROR: Failed to initialize audio system: {e}")
            # Show error in UI
            self.root.after(0, lambda: self.status_label.config(
                text=f"Audio Error: {str(e)[:30]}..."
            ))
        
        # Load speech state
        self.load_speech_state()
        
        # Create the UI components
        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.main_frame, 
            text="Initializing...", 
            font=('Arial', 16)
        )
        self.status_label.pack(fill="x", pady=(0, 10))
        
        # Waveform canvas
        self.waveform_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.waveform_frame.pack(expand=True, fill="both")
        
        self.waveform_canvas = tk.Canvas(
            self.waveform_frame, 
            bg="#f0f0f0", 
            height=150,
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.waveform_canvas.pack(expand=True, fill="both", padx=5, pady=5)
        
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
        
        # Load the faster-whisper model
        try:
            self.root.after(0, lambda: self.status_label.config(
                text="Loading faster-whisper model..."
            ))
            
            # Import here to avoid circular imports
            import faster_whisper
            
            # Load the small model for a good balance of speed and accuracy
            # Using CPU as default for compatibility
            self.whisper_model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")
            
            self.root.after(0, lambda: self.status_label.config(
                text="Ready"
            ))
            
            logger.info("faster-whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading faster-whisper model: {e}")
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
            # Start visualization if not already running
            self.root.after(0, self.update_waveform)
        elif self.speaking:
            self.status_label.config(text="Speaking...")
            # Start visualization for speaking
            self.root.after(0, self.update_waveform)
        else:
            self.status_label.config(text="Ready")
            # Update visualization to show idle state
            self.root.after(0, self.update_waveform)
    
    def update_waveform(self):
        """Update the circular audio visualization on the canvas"""
        try:
            # Get canvas dimensions
            canvas_width = self.waveform_canvas.winfo_width()
            canvas_height = self.waveform_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet properly sized
                self.root.after(100, self.update_waveform)
                return
            
            # Clear the canvas
            self.waveform_canvas.delete("all")
            
            # Calculate center point
            center_x = canvas_width / 2
            center_y = canvas_height / 2
            
            # Draw background circle
            background_radius = min(canvas_width, canvas_height) * 0.4
            self.waveform_canvas.create_oval(
                center_x - background_radius, center_y - background_radius,
                center_x + background_radius, center_y + background_radius,
                outline="#e0e0e0", width=2, fill="#f8f8f8"
            )
            
            if self.listening or self.speaking:
                # Get current amplitude
                current_amplitude = 0
                if hasattr(self, 'waveform_data') and len(self.waveform_data) > 0:
                    # Use the most recent amplitude value
                    current_amplitude = self.waveform_data[-1]
                
                # Calculate radius based on amplitude
                # Scale the amplitude (typically 0-0.5) to a reasonable range
                # Base radius is 30% of the background circle, max is 90%
                min_radius = background_radius * 0.3
                max_radius = background_radius * 0.9
                
                # Scale amplitude (typically 0-0.5) to radius range
                radius = min_radius + (current_amplitude * (max_radius - min_radius) * 4)
                
                # Ensure radius stays within bounds
                radius = max(min_radius, min(radius, max_radius))
                
                # Draw the amplitude circle
                fill_color = "#4287f5" if self.listening else "#42f587"  # Blue for listening, green for speaking
                self.waveform_canvas.create_oval(
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius,
                    outline="", fill=fill_color
                )
                
                # Draw inner circle (white)
                inner_radius = min_radius * 0.8
                self.waveform_canvas.create_oval(
                    center_x - inner_radius, center_y - inner_radius,
                    center_x + inner_radius, center_y + inner_radius,
                    outline="", fill="white"
                )
                
                # Add icon based on state
                if self.listening:
                    # Draw microphone icon (simple representation)
                    mic_width = inner_radius * 0.6
                    mic_height = inner_radius * 1.2
                    
                    # Microphone body
                    self.waveform_canvas.create_rectangle(
                        center_x - mic_width/2, center_y - mic_height/2,
                        center_x + mic_width/2, center_y + mic_height/4,
                        fill="#555555", outline=""
                    )
                    
                    # Microphone top (rounded)
                    self.waveform_canvas.create_oval(
                        center_x - mic_width/2, center_y - mic_height/2 - mic_width/2,
                        center_x + mic_width/2, center_y - mic_height/2 + mic_width/2,
                        fill="#555555", outline=""
                    )
                    
                    # Stand
                    self.waveform_canvas.create_rectangle(
                        center_x - mic_width/6, center_y + mic_height/4,
                        center_x + mic_width/6, center_y + mic_height/2,
                        fill="#555555", outline=""
                    )
                    
                    # Base
                    self.waveform_canvas.create_rectangle(
                        center_x - mic_width/2, center_y + mic_height/2 - mic_width/6,
                        center_x + mic_width/2, center_y + mic_height/2 + mic_width/6,
                        fill="#555555", outline=""
                    )
                else:
                    # Draw speaker icon for speaking
                    speaker_size = inner_radius * 0.7
                    
                    # Speaker body
                    self.waveform_canvas.create_rectangle(
                        center_x - speaker_size/2, center_y - speaker_size/2,
                        center_x - speaker_size/6, center_y + speaker_size/2,
                        fill="#555555", outline=""
                    )
                    
                    # Speaker cone
                    points = [
                        center_x - speaker_size/6, center_y - speaker_size/2,  # Top left
                        center_x + speaker_size/2, center_y - speaker_size,    # Top right
                        center_x + speaker_size/2, center_y + speaker_size,    # Bottom right
                        center_x - speaker_size/6, center_y + speaker_size/2   # Bottom left
                    ]
                    self.waveform_canvas.create_polygon(points, fill="#555555", outline="")
                    
                    # Sound waves (3 arcs)
                    for i in range(1, 4):
                        arc_size = speaker_size * (0.5 + i * 0.25)
                        self.waveform_canvas.create_arc(
                            center_x, center_y - arc_size/2,
                            center_x + arc_size, center_y + arc_size/2,
                            start=300, extent=120,
                            style="arc", outline="#555555", width=2
                        )
                
                # Draw pulsing rings
                if hasattr(self, 'pulse_count'):
                    self.pulse_count += 1
                    if self.pulse_count > 100:
                        self.pulse_count = 0
                else:
                    self.pulse_count = 0
                
                # Create 3 pulsing rings
                for i in range(3):
                    pulse_phase = (self.pulse_count + i * 33) % 100
                    if pulse_phase < 70:  # Only show rings during part of the cycle
                        # Calculate ring size based on pulse phase
                        ring_size = background_radius * (0.5 + pulse_phase / 70)
                        # Calculate opacity based on pulse phase (fade out as it expands)
                        opacity = int(255 * (1 - pulse_phase / 70))
                        ring_color = f"#{opacity:02x}{opacity:02x}{opacity:02x}"
                        
                        self.waveform_canvas.create_oval(
                            center_x - ring_size, center_y - ring_size,
                            center_x + ring_size, center_y + ring_size,
                            outline=ring_color, width=1, fill=""
                        )
            else:
                # Draw a standby/ready icon in the center
                ready_radius = background_radius * 0.3
                self.waveform_canvas.create_oval(
                    center_x - ready_radius, center_y - ready_radius,
                    center_x + ready_radius, center_y + ready_radius,
                    outline="#cccccc", width=2, fill="#f0f0f0"
                )
                
                # Draw a simple "ready" symbol (play button)
                triangle_size = ready_radius * 0.8
                points = [
                    center_x - triangle_size/2, center_y - triangle_size,
                    center_x - triangle_size/2, center_y + triangle_size,
                    center_x + triangle_size, center_y
                ]
                self.waveform_canvas.create_polygon(points, fill="#cccccc", outline="")
            
            # Schedule the next update if listening or speaking
            if self.listening or self.speaking:
                self.root.after(self.waveform_update_interval, self.update_waveform)
            
        except Exception as e:
            logger.error(f"Error updating visualization: {e}", exc_info=True)
            # Try again after a delay
            self.root.after(self.waveform_update_interval * 2, self.update_waveform)
    
    def process_audio_for_visualization(self, audio_data):
        """Process audio data for visualization"""
        try:
            # Convert to numpy array
            data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize the data to range [-1, 1]
            normalized = data.astype(float) / 32768.0
            
            # Take absolute value to get amplitude
            amplitude = np.abs(normalized).mean()
            
            # Add to waveform data
            self.waveform_data.append(amplitude)
            
            # Keep only the most recent points
            if len(self.waveform_data) > self.waveform_max_points:
                self.waveform_data = self.waveform_data[-self.waveform_max_points:]
        except Exception as e:
            logger.error(f"Error processing audio for visualization: {e}", exc_info=True)
    
    def start_listening(self):
        """Start listening for audio input"""
        try:
            logger.info("Starting audio recording")
            
            # Play start listening notification sound
            threading.Thread(target=play_audio_file, args=(START_LISTENING_SOUND,), daemon=True).start()
            
            # Reset waveform data
            self.waveform_data = []
            
            def audio_callback(in_data, frame_count, time_info, status):
                try:
                    # Log detailed timing information periodically
                    if hasattr(self, 'callback_count'):
                        self.callback_count += 1
                        if self.callback_count % 50 == 0:  # Log every ~50 callbacks
                            logger.debug(f"Audio callback timing - input timestamp: {time_info.get('input_buffer_adc_time', 'N/A')}, "
                                       f"current time: {time_info.get('current_time', 'N/A')}")
                    else:
                        self.callback_count = 1

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
                    if hasattr(self, 'audio_frames'):
                        self.audio_frames.append(in_data)
                        
                        # Process audio for visualization
                        self.process_audio_for_visualization(in_data)
                        
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
                    
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}", exc_info=True)
                    return (in_data, pyaudio.paContinue)  # Try to continue despite errors
            
            # Initialize audio frames list
            self.audio_frames = []
            
            # Start the audio stream with the selected device
            logger.debug(f"Opening audio stream with FORMAT={FORMAT}, CHANNELS={CHANNELS}, RATE={RATE}, CHUNK={CHUNK}, DEVICE={self.selected_device_index}")
            self.stream = self.p.open(
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
            
            # Test audio input
            logger.info("Testing audio input...")
            print("Testing audio input...")
            
            # Wait a moment and check if we're receiving audio
            time.sleep(0.5)
            if not hasattr(self, 'audio_frames') or len(self.audio_frames) == 0:
                logger.error("No audio data received in initial test")
                raise Exception("No audio data being received")
            
            # Check audio levels
            test_frame = self.audio_frames[-1]
            audio_data = np.frombuffer(test_frame, dtype=np.int16)
            normalized = audio_data.astype(float) / 32768.0
            level = np.abs(normalized).mean()
            
            logger.info(f"Initial audio level: {level:.6f}")
            print(f"Audio input level: {level:.6f}")
            
            if level < 0.0001:  # Very low level threshold
                logger.warning("Very low audio input level detected")
                print("Warning: Very low audio input level detected")
            
            logger.info("Audio stream initialized and receiving data")
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
            
            # Adjusted silence detection parameters based on debug tool findings
            silence_threshold = 0.01  # Increased based on debug tool mean amplitude of ~0.026
            silence_duration = 0
            max_silence = 3.0  # Increased to give more time for natural pauses
            check_interval = 0.2  # Increased to reduce CPU usage and allow for smoother detection
            
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
                current_amplitude = np.abs(normalized).mean()
                
                # Use a moving average of recent amplitudes for more stable detection
                if hasattr(self, 'recent_amplitudes') and len(self.recent_amplitudes) > 0:
                    avg_amplitude = sum(self.recent_amplitudes) / len(self.recent_amplitudes)
                else:
                    avg_amplitude = current_amplitude
                
                if avg_amplitude < silence_threshold:
                    silence_duration += check_interval
                    # Log only when silence is detected
                    if silence_duration >= 1.0 and silence_duration % 1.0 < check_interval:
                        logger.debug(f"Silence detected for {silence_duration:.1f}s, avg amplitude: {avg_amplitude:.6f}")
                else:
                    if silence_duration > 0:
                        logger.debug(f"Speech resumed after {silence_duration:.1f}s of silence, amplitude: {avg_amplitude:.6f}")
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
        """Process the recorded audio and generate a transcription using faster-whisper"""
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
                logger.warning("faster-whisper model not loaded yet")
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
            
            # Use faster-whisper to transcribe the audio
            logger.info("Transcribing audio with faster-whisper...")
            print("Transcribing audio with faster-whisper...")
            self.root.after(0, lambda: self.status_label.config(text="Transcribing audio..."))
            
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
            print(f"Transcription complete: \"{transcription}\"")
            
            # Log segments for debugging
            logger.debug("Transcription segments:")
            for i, segment in enumerate(segments):
                logger.debug(f"Segment {i}: {segment.start}-{segment.end}s: {segment.text}")
            
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
                
                # Play stop listening notification sound
                threading.Thread(target=play_audio_file, args=(STOP_LISTENING_SOUND,), daemon=True).start()
            else:
                logger.debug("No active audio stream to close")
            
            # Clear waveform data
            self.waveform_data = []
            
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
                        
                        # Create a simple speaking animation
                        def animate_speaking():
                            if not self.speaking:
                                return
                                
                            # Generate a random amplitude for speaking animation
                            # Use a sine wave with noise for more natural movement
                            import time
                            time_val = time.time() * 3  # Speed factor
                            base_amplitude = 0.1 + 0.1 * np.sin(time_val)
                            noise = 0.05 * np.random.random()
                            amplitude = base_amplitude + noise
                            
                            # Add to waveform data
                            self.waveform_data.append(amplitude)
                            
                            # Keep only the most recent points
                            if len(self.waveform_data) > self.waveform_max_points:
                                self.waveform_data = self.waveform_data[-self.waveform_max_points:]
                            
                            # Update the visualization
                            self.update_waveform()
                            
                            # Schedule the next animation frame if still speaking
                            if self.speaking:
                                self.root.after(50, animate_speaking)
                        
                        # Start the speaking animation
                        self.root.after(0, animate_speaking)
                        
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
                        self.waveform_data = []  # Clear waveform data
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