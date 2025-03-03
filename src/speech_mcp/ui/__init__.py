import tkinter as tk
from tkinter import ttk
import os
import json
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import pyaudio
import wave
import logging
import math
from queue import Queue
import threading
import random
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class AudioVisualizer:
    def __init__(self, ax, title, color):
        self.ax = ax
        self.title = title
        self.color = color
        self.line, = ax.plot([], [], color=color, lw=2)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(-1, 1)
        self.ax.set_title(title)
        self.ax.axis('off')
        self.data = np.zeros(360)
        self.angles = np.linspace(0, 2*np.pi, 360)
        
    def update(self, frame, data=None):
        if data is not None:
            self.data = data
        
        # Convert to polar coordinates for circle visualization
        x = self.data * np.cos(self.angles)
        y = self.data * np.sin(self.angles)
        
        self.line.set_data(x, y)
        return self.line,

class SpeechUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech MCP")
        self.root.geometry("800x500")
        
        # Load the initial speech state
        self.load_speech_state()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create title label
        title_label = ttk.Label(
            main_frame, 
            text="Voice Interaction", 
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=10)
        
        # Create frame for visualizers
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure for visualizers
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(121, polar=False)
        self.ax2 = self.fig.add_subplot(122, polar=False)
        
        # Create audio visualizers
        self.user_visualizer = AudioVisualizer(self.ax1, "User Speech", "#3498db")
        self.agent_visualizer = AudioVisualizer(self.ax2, "Agent Speech", "#e74c3c")
        
        # Add the figure to the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=10)
        
        # Create status labels
        self.user_status = ttk.Label(
            status_frame, 
            text="User: Idle", 
            font=('Arial', 12)
        )
        self.user_status.pack(side=tk.LEFT, padx=20)
        
        self.agent_status = ttk.Label(
            status_frame, 
            text="Agent: Idle", 
            font=('Arial', 12)
        )
        self.agent_status.pack(side=tk.RIGHT, padx=20)
        
        # Create transcript display
        transcript_frame = ttk.LabelFrame(main_frame, text="Transcript")
        transcript_frame.pack(fill=tk.X, pady=10)
        
        self.transcript_text = tk.Text(transcript_frame, height=4, wrap=tk.WORD)
        self.transcript_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Animation for visualizers
        self.ani1 = animation.FuncAnimation(
            self.fig, self.user_visualizer.update, interval=50, blit=True)
        self.ani2 = animation.FuncAnimation(
            self.fig, self.agent_visualizer.update, interval=50, blit=True)
        
        # Start threads for monitoring state changes
        self.should_update = True
        self.update_thread = threading.Thread(target=self.check_for_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start thread for checking response file
        self.response_thread = threading.Thread(target=self.check_for_responses)
        self.response_thread.daemon = True
        self.response_thread.start()
        
        # Handle window close event
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Update UI state
        self.update_ui_from_state()
    
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
        # Update status labels
        self.user_status.config(
            text=f"User: {'Listening' if self.listening else 'Idle'}")
        self.agent_status.config(
            text=f"Agent: {'Speaking' if self.speaking else 'Idle'}")
        
        # Update transcript display
        self.transcript_text.delete(1.0, tk.END)
        if self.last_transcript:
            self.transcript_text.insert(tk.END, f"User: {self.last_transcript}\n")
        if self.last_response:
            self.transcript_text.insert(tk.END, f"Agent: {self.last_response}")
        
        # Start or stop audio processing based on state
        if self.listening and not self.stream:
            self.start_listening()
        elif not self.listening and self.stream:
            self.stop_listening()
    
    def start_listening(self):
        """Start listening for audio input"""
        try:
            def audio_callback(in_data, frame_count, time_info, status):
                # Process audio data for visualization
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                # Normalize and prepare for visualization
                normalized = audio_data.astype(float) / 32768.0
                
                # Calculate amplitude for visualization
                amplitude = np.abs(normalized).mean() * 3  # Amplify for better visualization
                
                # Create circle data with some variation
                angles = np.linspace(0, 2*np.pi, 360)
                circle_data = np.ones(360) * (0.5 + amplitude)
                # Add some variation
                variation = np.sin(angles * 8) * amplitude * 0.3
                circle_data += variation
                
                # Update visualizer
                self.user_visualizer.update(None, circle_data)
                
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
            
            # Start a thread to detect silence and stop recording
            threading.Thread(target=self.detect_silence).start()
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.listening = False
            self.save_speech_state()
    
    def detect_silence(self):
        """Detect when the user stops speaking and end recording"""
        try:
            # Wait for initial audio to accumulate
            time.sleep(0.5)
            
            silence_threshold = 0.01
            silence_duration = 0
            max_silence = 1.5  # seconds
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
                self.process_recording()
                self.stop_listening()
                self.listening = False
                self.save_speech_state()
        
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
    
    def process_recording(self):
        """Process the recorded audio and generate a transcription"""
        try:
            if not hasattr(self, 'audio_frames') or not self.audio_frames:
                logger.warning("No audio frames to process")
                return
            
            # In a real implementation, this would use whisper to transcribe
            # For now, we'll simulate a transcription
            
            # Generate a simulated transcription
            sample_phrases = [
                "Hello, how can you help me today?",
                "What's the weather like?",
                "Tell me a joke.",
                "I need help with my project.",
                "Can you explain how this works?"
            ]
            
            transcription = random.choice(sample_phrases)
            self.last_transcript = transcription
            
            # Write the transcription to a file for the server to read
            with open(TRANSCRIPTION_FILE, 'w') as f:
                f.write(transcription)
            
            # Update the UI
            self.transcript_text.delete(1.0, tk.END)
            self.transcript_text.insert(tk.END, f"User: {transcription}\n")
            
            # Update state
            self.save_speech_state()
            
        except Exception as e:
            logger.error(f"Error processing recording: {e}")
    
    def stop_listening(self):
        """Stop listening for audio input"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Reset the visualizer
            self.user_visualizer.update(None, np.zeros(360))
            
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
    
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
                    with open(RESPONSE_FILE, 'r') as f:
                        response = f.read().strip()
                    
                    # Delete the file
                    os.remove(RESPONSE_FILE)
                    
                    # Process the response
                    if response:
                        self.last_response = response
                        self.speaking = True
                        self.save_speech_state()
                        
                        # Update UI
                        self.root.after(0, lambda: self.agent_status.config(
                            text="Agent: Speaking"))
                        self.root.after(0, lambda: self.transcript_text.insert(
                            tk.END, f"Agent: {response}"))
                        
                        # Simulate speaking with visualizer
                        self.simulate_speaking(response)
                        
                        # Update state when done speaking
                        self.speaking = False
                        self.save_speech_state()
                        self.root.after(0, lambda: self.agent_status.config(
                            text="Agent: Idle"))
            except Exception as e:
                logger.error(f"Error checking for responses: {e}")
            
            time.sleep(0.5)  # Check every half second
    
    def simulate_speaking(self, text):
        """Simulate speaking with audio visualizer animation"""
        try:
            # Calculate speaking duration based on text length
            duration = len(text) * 0.05  # 50ms per character
            
            # Number of frames to generate
            num_frames = int(duration * 20)  # 20 frames per second
            
            for i in range(num_frames):
                if not self.should_update:
                    break
                
                # Generate a dynamic circle visualization
                t = i / num_frames
                base_amplitude = 0.5 + 0.3 * np.sin(t * 2 * np.pi * 2)
                
                angles = np.linspace(0, 2*np.pi, 360)
                circle_data = np.ones(360) * base_amplitude
                
                # Add some variation based on position in the text
                variation = np.sin(angles * 8 + t * 20) * 0.2
                circle_data += variation
                
                # Update the visualizer
                self.agent_visualizer.update(None, circle_data)
                
                # Update the canvas
                self.canvas.draw_idle()
                
                # Sleep to control animation speed
                time.sleep(0.05)
            
            # Reset the visualizer when done
            self.agent_visualizer.update(None, np.zeros(360))
            
        except Exception as e:
            logger.error(f"Error simulating speaking: {e}")
    
    def on_close(self):
        """Handle window close event"""
        try:
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
            
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error closing UI: {e}")
            self.root.destroy()

def main():
    root = tk.Tk()
    app = SpeechUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()