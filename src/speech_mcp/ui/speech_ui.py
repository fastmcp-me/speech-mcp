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
import importlib.util
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QPushButton, QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QMetaObject, QRectF, Q_ARG
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient

# Import centralized constants
from speech_mcp.constants import (
    STATE_FILE, TRANSCRIPTION_FILE, RESPONSE_FILE, COMMAND_FILE,
    CMD_LISTEN, CMD_SPEAK, CMD_IDLE, CMD_UI_READY, CMD_UI_CLOSED,
    ENV_TTS_VOICE
)

# Import shared audio processor and speech recognition
from speech_mcp.audio_processor import AudioProcessor
from speech_mcp.speech_recognition import SpeechRecognizer

# Setup logging
logger = logging.getLogger(__name__)

class TTSAdapter(QObject):
    """
    Text-to-speech adapter for PyQt UI.
    
    This class provides a Qt wrapper around the TTS adapters to integrate with PyQt signals.
    """
    speaking_finished = pyqtSignal()
    speaking_started = pyqtSignal()
    speaking_progress = pyqtSignal(float)  # Progress between 0.0 and 1.0
    audio_level = pyqtSignal(float)  # Audio level for visualization
    
    def __init__(self):
        super().__init__()
        self.tts_engine = None
        self.is_speaking = False
        self.available_voices = []
        self.current_voice = None
        self.initialize_tts()
    
    def initialize_tts(self):
        """Initialize the TTS engine using the adapter system"""
        try:
            # First try to import the Kokoro adapter
            logger.info("Initializing TTS using adapter system")
            
            # Try to import the TTS adapters
            from speech_mcp.tts_adapters import KokoroTTS, Pyttsx3TTS
            
            # First try Kokoro (our primary TTS engine)
            try:
                logger.info("Trying to initialize Kokoro TTS adapter")
                self.tts_engine = KokoroTTS()
                if self.tts_engine.is_initialized:
                    logger.info("Kokoro TTS adapter initialized successfully")
                else:
                    logger.warning("Kokoro TTS adapter initialization failed")
                    raise ImportError("Kokoro initialization failed")
            except ImportError as e:
                logger.warning(f"Failed to initialize Kokoro TTS adapter: {e}")
                # Fall back to pyttsx3
                try:
                    logger.info("Falling back to pyttsx3 TTS adapter")
                    self.tts_engine = Pyttsx3TTS()
                    if self.tts_engine.is_initialized:
                        logger.info("pyttsx3 TTS adapter initialized successfully")
                    else:
                        logger.warning("pyttsx3 TTS adapter initialization failed")
                        raise ImportError("pyttsx3 initialization failed")
                except ImportError as e:
                    logger.error(f"Failed to initialize pyttsx3 TTS adapter: {e}")
                    self.tts_engine = None
            
            # If we have a TTS engine, get the available voices
            if self.tts_engine:
                self.available_voices = self.tts_engine.get_available_voices()
                self.current_voice = self.tts_engine.voice
                logger.info(f"TTS initialized with {len(self.available_voices)} voices, current voice: {self.current_voice}")
                return True
            else:
                logger.error("No TTS engine available")
                return False
                
        except ImportError as e:
            logger.warning(f"Failed to import TTS adapters: {e}")
            
            # Direct fallback to pyttsx3 if adapters are not available
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                logger.info("pyttsx3 text-to-speech engine initialized directly")
                
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
                logger.error(f"pyttsx3 not available: {e}")
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
        
        # Generate a speech pattern for visualization
        self.generate_speech_pattern(text)
        
        # Start speaking in a separate thread
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
        logger.info("Started _speak_thread")
        return True
    
    def emit_audio_level(self):
        """Emit simulated audio levels for visualization during speech"""
        if not self.is_speaking:
            self.audio_level_timer.stop()
            self.audio_level.emit(0.0)  # Reset to zero when not speaking
            return
        
        # Check if we have a speech pattern
        if hasattr(self, 'speech_pattern') and self.speech_pattern:
            # Get current pattern element
            if self.pattern_index < len(self.speech_pattern):
                level, duration = self.speech_pattern[self.pattern_index]
                
                # Add some randomness for natural variation
                import random
                import math
                
                # Add slight variation to make it more natural
                t = time.time() * 8.0
                variation = 0.1 * math.sin(t) + 0.05 * random.random()
                level = max(0.1, min(0.95, level + variation))
                
                # Emit the level
                self.audio_level.emit(level)
                
                # Decrement duration counter
                if hasattr(self, 'duration_counter'):
                    self.duration_counter -= 1
                else:
                    self.duration_counter = duration
                
                # Move to next pattern element when duration is complete
                if self.duration_counter <= 0:
                    self.pattern_index += 1
                    if self.pattern_index < len(self.speech_pattern):
                        self.duration_counter = self.speech_pattern[self.pattern_index][1]
            else:
                # Loop back to beginning if we've reached the end
                self.pattern_index = 0
                if self.speech_pattern:
                    self.duration_counter = self.speech_pattern[0][1]
        else:
            # Fallback to random pattern if no speech pattern available
            import random
            import math
            
            t = time.time() * 5.0
            base_level = 0.5 + 0.2 * math.sin(t * 1.5)
            variation = 0.3 + 0.15 * math.sin(t * 3.7)
            level = base_level + random.random() * variation
            level = max(0.1, min(0.95, level))
            
            self.audio_level.emit(level)
    
    def _speak_thread(self, text):
        """Thread function for speaking text"""
        try:
            logger.info(f"_speak_thread started for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"_speak_thread started for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Use the TTS engine's speak method
            if hasattr(self.tts_engine, 'speak'):
                # This is one of our adapters
                logger.info("Using TTS adapter speak method")
                print("Using TTS adapter speak method")
                try:
                    result = self.tts_engine.speak(text)
                    logger.info(f"TTS speak result: {result}")
                    print(f"TTS speak result: {result}")
                    if not result:
                        logger.error("TTS failed")
                        print("TTS failed")
                except Exception as e:
                    logger.error(f"Exception in TTS speak: {e}", exc_info=True)
                    print(f"Exception in TTS speak: {e}")
                    result = False
            elif hasattr(self.tts_engine, 'say'):
                # This is direct pyttsx3
                logger.info("Using direct pyttsx3 say method")
                print("Using direct pyttsx3 say method")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                logger.info("pyttsx3 speech completed")
                print("pyttsx3 speech completed")
            else:
                logger.error("TTS engine does not have speak or say method")
                print("TTS engine does not have speak or say method")
            
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
    
    def generate_speech_pattern(self, text):
        """Generate a speech pattern based on the text content"""
        import re
        
        # Initialize speech pattern
        self.speech_pattern = []
        self.pattern_index = 0
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Split sentence into words
            words = sentence.strip().split()
            
            # Start with medium level
            self.speech_pattern.append((0.5, 3))  # (level, duration in frames)
            
            for word in words:
                # Word length affects amplitude
                word_len = len(word)
                
                if word_len <= 2:  # Short words
                    level = 0.3 + 0.1 * word_len
                    duration = 2
                elif word_len <= 5:  # Medium words
                    level = 0.5 + 0.05 * word_len
                    duration = 3
                else:  # Long words
                    level = 0.7 + 0.02 * min(word_len, 15)  # Cap at 15 chars
                    duration = 4 + min(word_len // 3, 4)  # Longer duration for longer words
                
                # Add emphasis for capitalized words
                if word[0].isupper() and len(word) > 1:
                    level = min(level * 1.2, 0.95)
                
                # Add the word's pattern
                self.speech_pattern.append((level, duration))
                
                # Add a brief pause between words
                self.speech_pattern.append((0.2, 1))
            
            # Add longer pause at end of sentence
            self.speech_pattern.append((0.1, 5))
        
        # Add final trailing off
        self.speech_pattern.append((0.3, 2))
        self.speech_pattern.append((0.2, 2))
        self.speech_pattern.append((0.1, 2))
        
        logger.debug(f"Generated speech pattern with {len(self.speech_pattern)} elements")
    
    def set_voice(self, voice_id):
        """Set the voice to use for TTS"""
        if not self.tts_engine:
            logger.warning("No TTS engine available")
            return False
        
        try:
            if hasattr(self.tts_engine, 'set_voice'):
                # This is one of our adapters
                result = self.tts_engine.set_voice(voice_id)
                if result:
                    self.current_voice = voice_id
                    logger.info(f"Voice set to: {voice_id}")
                    return True
                else:
                    logger.error(f"Failed to set voice to: {voice_id}")
                    return False
            elif hasattr(self.tts_engine, 'setProperty'):
                # This is direct pyttsx3
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


class AudioVisualizer(QWidget):
    """
    Widget for visualizing audio levels and waveforms.
    """
    def __init__(self, parent=None, mode="user", width_factor=1.0):
        """
        Initialize the audio visualizer.
        
        Args:
            parent: Parent widget
            mode: "user" or "agent" to determine color scheme
            width_factor: Factor to adjust the width of bars (1.0 = full width, 0.5 = half width)
        """
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.audio_levels = [0.0] * 50  # Store recent audio levels
        self.setStyleSheet("background-color: #1e1e1e;")
        self.mode = mode
        self.width_factor = width_factor
        self.active = False  # Track if visualizer is active
        
        # Set colors based on mode
        if self.mode == "user":
            self.bar_color = QColor(0, 200, 255, 180)  # Blue for user
            self.glow_color = QColor(0, 120, 255, 80)  # Softer blue glow
        else:
            self.bar_color = QColor(0, 255, 100, 200)  # Brighter green for agent
            self.glow_color = QColor(0, 220, 100, 100)  # Stronger green glow
            
        # Inactive colors (grey)
        self.inactive_bar_color = QColor(100, 100, 100, 120)  # Grey for inactive
        self.inactive_glow_color = QColor(80, 80, 80, 60)  # Softer grey glow
        
        # Add a smoothing factor to make the visualization less jumpy
        self.smoothing_factor = 0.3
        self.last_level = 0.0
        
        # Timer for animation
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(30)  # Update at ~30fps
        
        # Animation time for dynamic effects
        self.animation_time = 0.0
    
    def set_active(self, active):
        """Set the visualizer as active or inactive."""
        self.active = active
        self.update()
    
    def update_level(self, level):
        """Update with a new audio level."""
        # Apply smoothing to avoid abrupt changes
        smoothed_level = (level * (1.0 - self.smoothing_factor)) + (self.last_level * self.smoothing_factor)
        self.last_level = smoothed_level
        
        # For center-rising visualization, we just need to update the current level
        # We'll shift all values in paintEvent
        self.audio_levels.pop(0)
        self.audio_levels.append(smoothed_level)
        
        # Update animation time
        self.animation_time += 0.1
    
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
        
        # Choose colors based on active state
        if self.active:
            bar_color = self.bar_color
            glow_color = self.glow_color
        else:
            bar_color = self.inactive_bar_color
            glow_color = self.inactive_glow_color
        
        # Set pen for waveform
        pen = QPen(bar_color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw the center-rising waveform
        # We'll draw bars at different positions with heights based on audio levels
        bar_count = 40  # Number of bars to draw
        bar_width = (width / bar_count) * self.width_factor
        bar_spacing = 2  # Pixels between bars
        
        # Calculate animation phase for dynamic effects
        import math
        phase = self.animation_time % (2 * math.pi)
        
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
                
            # If inactive, flatten the visualization
            if not self.active:
                level = level * 0.2  # Reduce height significantly when inactive
                
            # Add subtle wave effect to make visualization more dynamic
            wave_effect = 0.05 * math.sin(phase + i * 0.2)
            level = max(0.0, min(1.0, level + wave_effect))
            
            # Calculate bar height based on level
            bar_height = level * mid_height * 0.95
            
            # Calculate x position (centered)
            x = (width / 2) + (i * bar_width / 2) - (bar_width / 2)
            x_mirror = (width / 2) - (i * bar_width / 2) - (bar_width / 2)
            
            # Draw the bar (right side)
            if i < bar_count / 2:
                # Draw glow effect first (larger, more transparent)
                glow_rect = QRectF(
                    x - bar_width * 0.2, 
                    mid_height - bar_height * 1.1, 
                    (bar_width - bar_spacing) * 1.4, 
                    bar_height * 2.2
                )
                painter.fillRect(glow_rect, QColor(
                    glow_color.red(), 
                    glow_color.green(), 
                    glow_color.blue(), 
                    80 - i * 2
                ))
                
                # Draw the main bar
                rect = QRectF(x, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                painter.fillRect(rect, QColor(
                    bar_color.red(), 
                    bar_color.green(), 
                    bar_color.blue(), 
                    180 - i * 3
                ))
            
            # Draw the mirrored bar (left side)
            if i < bar_count / 2:
                # Draw glow effect first
                glow_rect_mirror = QRectF(
                    x_mirror - bar_width * 0.2, 
                    mid_height - bar_height * 1.1, 
                    (bar_width - bar_spacing) * 1.4, 
                    bar_height * 2.2
                )
                painter.fillRect(glow_rect_mirror, QColor(
                    glow_color.red(), 
                    glow_color.green(), 
                    glow_color.blue(), 
                    80 - i * 2
                ))
                
                # Draw the main bar
                rect_mirror = QRectF(x_mirror, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                painter.fillRect(rect_mirror, QColor(
                    bar_color.red(), 
                    bar_color.green(), 
                    bar_color.blue(), 
                    180 - i * 3
                ))
        
        # Draw a thin center line with a gradient
        gradient = QLinearGradient(0, mid_height, width, mid_height)
        gradient.setColorAt(0, QColor(100, 100, 100, 0))
        gradient.setColorAt(0.5, QColor(100, 100, 100, 100))
        gradient.setColorAt(1, QColor(100, 100, 100, 0))
        
        painter.setPen(QPen(gradient, 1))
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
        
        # Create a layout for the visualizer labels
        label_layout = QHBoxLayout()
        
        # User label
        user_label = QLabel("User")
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setStyleSheet("""
            font-size: 14px;
            color: #00c8ff;
            font-weight: bold;
        """)
        label_layout.addWidget(user_label, 1)
        
        # Agent label
        agent_label = QLabel("Agent")
        agent_label.setAlignment(Qt.AlignCenter)
        agent_label.setStyleSheet("""
            font-size: 14px;
            color: #00ff64;
            font-weight: bold;
        """)
        label_layout.addWidget(agent_label, 1)
        
        # Add the label layout to the main layout
        main_layout.addLayout(label_layout)
        
        # Create a layout for the visualizers
        visualizer_layout = QHBoxLayout()
        
        # User audio visualizer (blue)
        self.user_visualizer = AudioVisualizer(mode="user", width_factor=1.0)
        visualizer_layout.addWidget(self.user_visualizer, 1)  # Equal ratio
        
        # Agent audio visualizer (green)
        self.agent_visualizer = AudioVisualizer(mode="agent", width_factor=1.0)
        visualizer_layout.addWidget(self.agent_visualizer, 1)  # Equal ratio
        
        # Add the visualizer layout to the main layout
        main_layout.addLayout(visualizer_layout)
        
        # Transcription display
        self.transcription_label = QLabel("Ready for voice interaction")
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
        
        # Add Select Voice button
        self.select_voice_button = AnimatedButton("Save Voice")
        self.select_voice_button.clicked.connect(self.save_selected_voice)
        self.select_voice_button.setEnabled(True)
        self.select_voice_button.setMinimumWidth(120)
        self.select_voice_button.set_style("""
            background-color: #9b59b6;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Use AnimatedButton for Test Voice button
        self.speak_button = AnimatedButton("Test Voice")
        self.speak_button.clicked.connect(self.test_voice)
        self.speak_button.setEnabled(True)
        self.speak_button.setMinimumWidth(120)
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
        self.close_button.setMinimumWidth(120)
        self.close_button.set_style("""
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Add buttons to layout with equal spacing
        button_layout.addStretch(1)
        button_layout.addWidget(self.select_voice_button)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.speak_button)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.close_button)
        button_layout.addStretch(1)
        
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
        
        # Initialize visualizers to inactive state
        self.set_user_visualizer_active(False)
        self.set_agent_visualizer_active(False)
    
    def set_user_visualizer_active(self, active):
        """Set the user visualizer as active or inactive."""
        self.user_visualizer.set_active(active)
    
    def set_agent_visualizer_active(self, active):
        """Set the agent visualizer as active or inactive."""
        self.agent_visualizer.set_active(active)
    
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
        
        # Start background threads for initialization
        threading.Thread(target=self.initialize_audio_processor, daemon=True).start()
        threading.Thread(target=self.initialize_tts_adapter, daemon=True).start()
    
    def initialize_audio_processor(self):
        """Initialize audio processor in background thread"""
        try:
            logger.info("Initializing audio processor in background")
            self.audio_processor = AudioProcessorUI()
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
            
            # Connect audio level signal to agent visualizer
            self.tts_adapter.audio_level.connect(self.update_agent_audio_level)
            
            # Create audio level timer if it doesn't exist yet
            if not hasattr(self.tts_adapter, 'audio_level_timer'):
                self.tts_adapter.audio_level_timer = QTimer()
                self.tts_adapter.audio_level_timer.timeout.connect(self.tts_adapter.emit_audio_level)
                logger.info("Created audio level timer for TTS visualization")
            
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
        
        # Clear initialization message from transcription label
        self.transcription_label.setText("Ready for voice interaction")
        
        # Check for any pending commands
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                    if command == "LISTEN" and self.has_saved_voice_preference():
                        # Start listening since we have a saved voice preference
                        self.start_listening()
            except Exception as e:
                logger.error(f"Error reading command file: {e}")
        
        # If no voice preference is saved, show guidance message
        if not self.has_saved_voice_preference():
            self.transcription_label.setText("Please select a voice from the dropdown and click 'Save Voice' to continue")
            # Wait a moment before speaking to ensure UI is fully ready
            QTimer.singleShot(500, self.play_guidance_message)
    
    def has_saved_voice_preference(self):
        """Check if a voice preference has been saved"""
        try:
            # First check environment variable
            from speech_mcp.config import get_env_setting
            env_voice = get_env_setting(ENV_TTS_VOICE)
            if env_voice:
                logger.info(f"Found voice preference in environment variable: {env_voice}")
                return True
                
            # Then check config file
            from speech_mcp.config import get_setting
            config_voice = get_setting("tts", "voice", None)
            if config_voice:
                logger.info(f"Found voice preference in config file: {config_voice}")
                return True
                
            logger.info("No saved voice preference found")
            return False
        except ImportError:
            logger.warning("Config module not available, assuming no voice preference")
            return False
        except Exception as e:
            logger.error(f"Error checking for saved voice preference: {e}")
            return False
    
    def save_voice_preference(self, voice):
        """Save the selected voice preference to config"""
        try:
            # Save to config file
            from speech_mcp.config import set_setting
            result = set_setting("tts", "voice", voice)
            
            # Also set environment variable for current session
            from speech_mcp.config import set_env_setting
            set_env_setting(ENV_TTS_VOICE, voice)
            
            logger.info(f"Voice preference saved: {voice}")
            return result
        except ImportError:
            logger.error("Config module not available, cannot save voice preference")
            return False
        except Exception as e:
            logger.error(f"Error saving voice preference: {e}")
            return False
    
    def save_selected_voice(self):
        """Save the selected voice and switch to listen mode"""
        # Get the currently selected voice
        index = self.voice_combo.currentIndex()
        if index < 0:
            logger.warning("No voice selected")
            self.transcription_label.setText("Please select a voice from the dropdown")
            return
        
        voice = self.voice_combo.itemData(index)
        if not voice:
            logger.warning("Invalid voice selection")
            self.transcription_label.setText("Please select a valid voice from the dropdown")
            return
        
        logger.info(f"Saving voice preference: {voice}")
        
        # Save the voice preference
        if self.save_voice_preference(voice):
            logger.info("Voice preference saved successfully")
            self.transcription_label.setText(f"Voice '{voice}' saved as your preference")
            
            # Create a UI_READY command file to signal back to the server
            try:
                with open(COMMAND_FILE, 'w') as f:
                    f.write(CMD_UI_READY)
                logger.info("Created UI_READY command file after voice selection")
            except Exception as e:
                logger.error(f"Error creating command file: {e}")
            
            # Test the voice to confirm
            QTimer.singleShot(1000, lambda: self.tts_adapter.speak("Voice preference saved. You can now start listening."))
        else:
            logger.error("Failed to save voice preference")
            self.transcription_label.setText("Failed to save voice preference. Please try again.")
    
    def play_guidance_message(self):
        """Play a guidance message for first-time users"""
        if hasattr(self, 'tts_adapter') and self.tts_adapter:
            # Add a highlight effect to the Select Voice button
            original_style = self.select_voice_button.styleSheet()
            highlight_style = """
                background-color: #e74c3c;
                color: white;
                border: 2px solid #f39c12;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            """
            self.select_voice_button.setStyleSheet(highlight_style)
            
            # Speak the guidance message
            self.tts_adapter.speak("Please select a voice from the dropdown menu and click Save Voice to continue.")
            logger.info("Played guidance message for first-time user")
            
            # Restore the original style after a delay
            QTimer.singleShot(3000, lambda: self.select_voice_button.setStyleSheet(original_style))
    
    
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
            self.transcription_label.setText("TTS not ready yet. Please wait...")
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
        
        # Start the agent animation timer before speaking
        if not hasattr(self, 'agent_animation_timer'):
            self.agent_animation_timer = QTimer(self)
            self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
        self.agent_animation_timer.start(50)  # Update every 50ms
        
        # Activate agent visualizer
        self.set_agent_visualizer_active(True)
        self.set_user_visualizer_active(False)
        
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
                
                # Stop the animation if speaking failed
                if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
                    self.agent_animation_timer.stop()
                    self.agent_visualizer.update_level(0.0)
        except Exception as e:
            logger.error(f"Exception during test voice: {e}", exc_info=True)
            print(f"Exception during test voice: {e}")
            self.transcription_label.setText(f"Error: {str(e)}")
            QTimer.singleShot(3000, lambda: self.transcription_label.setText("Select a voice and click 'Test Voice' to hear it"))
            
            # Stop the animation if an exception occurred
            if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
                self.agent_animation_timer.stop()
                self.agent_visualizer.update_level(0.0)
    
    def update_audio_level(self, level):
        """Update the user audio level visualization."""
        self.user_visualizer.update_level(level)
    
    def update_agent_audio_level(self, level):
        """Update the agent audio level visualization."""
        self.agent_visualizer.update_level(level)
    
    def handle_transcription(self, text):
        """Handle new transcription text."""
        self.transcription_label.setText(f"You: {text}")
        logger.info(f"New transcription: {text}")
    
    def start_listening(self):
        """Start listening mode."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            self.transcription_label.setText("Speech recognition not ready yet")
            return
            
        self.audio_processor.start_listening()
        
        # Activate user visualizer, deactivate agent visualizer
        self.set_user_visualizer_active(True)
        self.set_agent_visualizer_active(False)
    
    def stop_listening(self):
        """Stop listening mode."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            return
            
        self.audio_processor.stop_listening()
        
        # Deactivate user visualizer
        self.set_user_visualizer_active(False)
    
    def on_speaking_started(self):
        """Called when speaking starts."""
        self.speak_button.setEnabled(False)
        
        # Activate agent visualizer, deactivate user visualizer
        self.set_agent_visualizer_active(True)
        self.set_user_visualizer_active(False)
        
        # Start a timer to animate the agent visualizer
        if not hasattr(self, 'agent_animation_timer'):
            self.agent_animation_timer = QTimer(self)
            self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
        self.agent_animation_timer.start(50)  # Update every 50ms
        
    def on_speaking_finished(self):
        """Called when speaking finishes."""
        self.speak_button.setEnabled(True)
        
        # Stop the agent animation timer
        if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
            self.agent_animation_timer.stop()
        
        # Deactivate agent visualizer
        self.set_agent_visualizer_active(False)
            
    def animate_agent_visualizer(self):
        """Animate the agent visualizer with dynamic levels"""
        import random
        import math
        
        # Create a dynamic wave pattern
        t = time.time() * 5.0
        base_level = 0.5 + 0.3 * math.sin(t * 1.5)
        variation = 0.2 * random.random()
        level = base_level + variation
        
        # Ensure level stays within bounds
        level = max(0.1, min(0.95, level))
        
        # Update the agent visualizer
        self.agent_visualizer.update_level(level)
    
    def check_for_commands(self):
        """Check for commands from the server."""
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                
                # Process the command
                if command == CMD_LISTEN:
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
                        # Show guidance message instead
                        self.transcription_label.setText("Please select a voice from the dropdown and click 'Select Voice' to continue")
                        # Wait a moment before speaking to ensure UI is fully ready
                        QTimer.singleShot(500, self.play_guidance_message)
                        
                elif command == CMD_IDLE and hasattr(self, 'audio_processor') and self.audio_processor and self.audio_processor.is_listening:
                    logger.info("Received IDLE command")
                    self.stop_listening()
                elif command == CMD_SPEAK:
                    logger.info("Received SPEAK command")
                    # We'll handle speaking in check_for_responses
                    if hasattr(self, 'tts_adapter') and self.tts_adapter:
                        # Activate agent visualizer
                        self.set_agent_visualizer_active(True)
                        self.set_user_visualizer_active(False)
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
                
                # Display the response text in the transcription label
                self.transcription_label.setText(f"Agent: {response}")
                
                # Start the agent animation timer before speaking
                # This ensures the visualization works even if the TTS signal connection fails
                if not hasattr(self, 'agent_animation_timer'):
                    self.agent_animation_timer = QTimer(self)
                    self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
                self.agent_animation_timer.start(50)  # Update every 50ms
                
                # Speak the response using the TTS adapter
                if response:
                    self.tts_adapter.speak(response)
                
            except Exception as e:
                logger.error(f"Error processing response file: {e}")
                self.transcription_label.setText(f"Error processing response: {str(e)}")
                QTimer.singleShot(3000, lambda: self.transcription_label.setText("Ready for voice interaction"))
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop audio processor if it exists
        if hasattr(self, 'audio_processor') and self.audio_processor:
            self.audio_processor.stop_listening()
        
        # Write a UI_CLOSED command to the command file
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write(CMD_UI_CLOSED)
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