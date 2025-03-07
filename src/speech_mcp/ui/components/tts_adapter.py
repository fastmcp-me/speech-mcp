"""
Text-to-speech adapter for the Speech UI.

This module provides a PyQt wrapper around the TTS adapters.
"""

import os
import time
import logging
import threading
import random
import math
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# Import centralized constants
from speech_mcp.constants import ENV_TTS_VOICE

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
        self._speaking_lock = threading.Lock()  # Add a lock for thread safety
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
        
        # Use a lock to safely check and update speaking state
        with self._speaking_lock:
            if self.is_speaking:
                logger.warning("Already speaking, ignoring new request")
                return False
            
            # Set speaking state before starting thread
            self.is_speaking = True
        
        logger.info(f"TTSAdapter.speak called with text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"TTSAdapter.speak called with text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Emit speaking started signal on the main thread
        self.speaking_started.emit()
        
        # Generate a speech pattern for visualization
        self.generate_speech_pattern(text)
        
        # Start speaking in a separate thread
        speak_thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        speak_thread.start()
        logger.info("Started _speak_thread")
        return True
    
    def emit_audio_level(self):
        """Emit simulated audio levels for visualization during speech"""
        # Use the lock to safely check the speaking state
        with self._speaking_lock:
            is_speaking = self.is_speaking
        
        if not is_speaking:
            if hasattr(self, 'audio_level_timer') and self.audio_level_timer.isActive():
                self.audio_level_timer.stop()
            self.audio_level.emit(0.0)  # Reset to zero when not speaking
            return
        
        # Check if we have a speech pattern
        if hasattr(self, 'speech_pattern') and self.speech_pattern:
            # Get current pattern element
            if self.pattern_index < len(self.speech_pattern):
                level, duration = self.speech_pattern[self.pattern_index]
                
                # Add some randomness for natural variation
                variation = 0.1 * math.sin(time.time() * 8.0) + 0.05 * random.random()
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
            # Use the lock to safely update the speaking state
            with self._speaking_lock:
                self.is_speaking = False
            
            # Emit the signal after releasing the lock
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