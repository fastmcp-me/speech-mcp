import os
import json
import time
import threading
import logging
import tempfile
import io
from queue import Queue

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    print("Text-to-speech engine initialized successfully!")
except ImportError:
    print("WARNING: pyttsx3 not available. Text-to-speech will be simulated.")
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

class SimpleSpeechProcessor:
    """A simple speech processor without GUI"""
    def __init__(self):
        # Initialize basic components
        print("Initializing speech processor...")
        self.ui_active = True
        self.listening = False
        self.speaking = False
        self.last_transcript = ""
        self.last_response = ""
        self.should_update = True
        self.stream = None
        
        # Initialize PyAudio
        print("Initializing audio system...")
        self.p = pyaudio.PyAudio()
        
        # Load speech state
        self.load_speech_state()
        
        # Load whisper
        print("Checking for speech recognition module...")
        if not load_whisper():
            print("WARNING: Speech recognition will not be available.")
            print("Please install the Whisper module with: pip install openai-whisper")
            return
        
        # Initialize Whisper model
        self.whisper_model = None
        self.load_whisper_model()
        
        # Start threads for monitoring state changes
        self.update_thread = threading.Thread(target=self.check_for_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start thread for checking response file
        self.response_thread = threading.Thread(target=self.check_for_responses)
        self.response_thread.daemon = True
        self.response_thread.start()
        
        print("Speech processor initialization complete!")
        logger.info("Speech processor initialized successfully")
    
    def load_whisper_model(self):
        """Load the Whisper model in a background thread"""
        try:
            # Load the small model for a good balance of speed and accuracy
            print("Loading Whisper model... This may take a few moments.")
            print("(The model will be downloaded if this is the first time using it)")
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
            print("Whisper model loaded successfully! Ready to transcribe speech.")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            print(f"Error loading Whisper model: {e}")
    
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
    
    def update_state_from_file(self):
        """Update the state based on the current file"""
        # Reload the state from file
        self.load_speech_state()
        
        # Start or stop audio processing based on state
        if self.listening and not self.stream:
            self.start_listening()
        elif not self.listening and self.stream:
            self.stop_listening()
    
    def start_listening(self):
        """Start listening for audio input"""
        try:
            def audio_callback(in_data, frame_count, time_info, status):
                # Store audio data for processing
                if hasattr(self, 'audio_frames'):
                    self.audio_frames.append(in_data)
                
                return (in_data, pyaudio.paContinue)
            
            # Initialize audio frames list
            self.audio_frames = []
            
            # Start the audio stream
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback
            )
            
            print("Microphone activated. Listening for speech...")
            logger.info("Started listening for audio input")
            
            # Start a thread to detect silence and stop recording
            threading.Thread(target=self.detect_silence).start()
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            print(f"Error starting audio: {e}")
            self.listening = False
            self.save_speech_state()
    
    def detect_silence(self):
        """Detect when the user stops speaking and end recording"""
        try:
            # Wait for initial audio to accumulate
            time.sleep(0.5)
            
            silence_threshold = 0.005  # Reduced from 0.01 to be less sensitive
            silence_duration = 0
            max_silence = 2.0  # Increased from 1.5 to 2.0 seconds
            check_interval = 0.1
            
            while self.listening and self.stream and silence_duration < max_silence:
                if not hasattr(self, 'audio_frames') or len(self.audio_frames) < 2:
                    time.sleep(check_interval)
                    continue
                
                # Get the latest audio frame
                latest_frame = self.audio_frames[-1]
                audio_data = np.frombuffer(latest_frame, dtype=np.int16)
                normalized = audio_data.astype(float) / 32768.0
                amplitude = np.abs(normalized).mean()
                
                if amplitude < silence_threshold:
                    silence_duration += check_interval
                else:
                    silence_duration = 0
                
                time.sleep(check_interval)
            
            # If we exited because of silence detection
            if self.listening and self.stream:
                logger.info("Silence detected, stopping recording")
                print("Silence detected. Processing speech...")
                self.process_recording()
                self.stop_listening()
                self.listening = False
                self.save_speech_state()
        
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
    
    def process_recording(self):
        """Process the recorded audio and generate a transcription using Whisper"""
        try:
            if not hasattr(self, 'audio_frames') or not self.audio_frames:
                logger.warning("No audio frames to process")
                return
            
            if self.whisper_model is None:
                logger.warning("Whisper model not loaded yet")
                self.last_transcript = "Sorry, speech recognition model is still loading. Please try again in a moment."
                with open(TRANSCRIPTION_FILE, 'w') as f:
                    f.write(self.last_transcript)
                return
            
            # Save the recorded audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
                # Create a WAV file from the recorded frames
                wf = wave.open(temp_audio_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
            
            logger.info(f"Audio saved to temporary file: {temp_audio_path}")
            
            # Use Whisper to transcribe the audio
            logger.info("Transcribing audio with Whisper...")
            print("Transcribing audio with Whisper...")
            result = self.whisper_model.transcribe(temp_audio_path)
            transcription = result["text"].strip()
            
            logger.info(f"Transcription: {transcription}")
            print(f"Transcription complete: \"{transcription}\"")
            
            # Clean up the temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")
            
            # Update the state with the transcription
            self.last_transcript = transcription
            
            # Write the transcription to a file for the server to read
            with open(TRANSCRIPTION_FILE, 'w') as f:
                f.write(transcription)
            
            # Update state
            self.save_speech_state()
            
        except Exception as e:
            logger.error(f"Error processing recording: {e}")
            self.last_transcript = f"Error processing speech: {str(e)}"
            with open(TRANSCRIPTION_FILE, 'w') as f:
                f.write(self.last_transcript)
    
    def stop_listening(self):
        """Stop listening for audio input"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                print("Microphone deactivated.")
                logger.info("Stopped listening for audio input")
            
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
            print(f"Error stopping audio: {e}")
    
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
                        self.update_state_from_file()
            except Exception as e:
                logger.error(f"Error checking for updates: {e}")
            
            time.sleep(0.5)  # Check every half second
    
    def check_for_responses(self):
        """Periodically check for new responses to speak"""
        while self.should_update:
            try:
                if os.path.exists(RESPONSE_FILE):
                    # Read the response
                    with open(RESPONSE_FILE, 'r') as f:
                        response = f.read().strip()
                    
                    # Delete the file
                    os.remove(RESPONSE_FILE)
                    
                    # Process the response
                    if response:
                        self.last_response = response
                        self.speaking = True
                        self.save_speech_state()
                        
                        logger.info(f"Speaking: {response}")
                        print(f"Speaking: \"{response}\"")
                        
                        # Use actual text-to-speech if available
                        if tts_available:
                            try:
                                # Use pyttsx3 for actual speech
                                tts_engine.say(response)
                                tts_engine.runAndWait()
                                print("Speech completed.")
                            except Exception as e:
                                logger.error(f"Error using text-to-speech: {e}")
                                print(f"Error using text-to-speech: {e}")
                                # Fall back to simulated speech
                                speaking_duration = len(response) * 0.05  # 50ms per character
                                time.sleep(speaking_duration)
                        else:
                            # Simulate speaking time if TTS not available
                            speaking_duration = len(response) * 0.05  # 50ms per character
                            time.sleep(speaking_duration)
                        
                        # Update state when done speaking
                        self.speaking = False
                        self.save_speech_state()
                        print("Done speaking.")
                        logger.info("Done speaking")
            except Exception as e:
                logger.error(f"Error checking for responses: {e}")
            
            time.sleep(0.5)  # Check every half second
    
    def shutdown(self):
        """Clean up resources and shut down"""
        try:
            print("\nShutting down speech processor...")
            self.should_update = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            self.p.terminate()
            
            # Update state to indicate UI is closed
            self.ui_active = False
            self.listening = False
            self.speaking = False
            self.save_speech_state()
            
            print("Speech processor shut down successfully.")
            logger.info("Speech processor shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down speech processor: {e}")
            print(f"Error during shutdown: {e}")

def main():
    """Main entry point for the speech processor"""
    try:
        print("\n===== Speech MCP Processor =====")
        print("Starting speech recognition system...")
        processor = SimpleSpeechProcessor()
        
        print("\nSpeech processor is running. Press Ctrl+C to exit.")
        # Keep the main thread alive
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down")
                processor.shutdown()
                print("Speech processor shut down.")
                break
    except Exception as e:
        logger.error(f"Error in speech processor main: {e}")
        print(f"\nERROR: Failed to start speech processor: {e}")

if __name__ == "__main__":
    main()