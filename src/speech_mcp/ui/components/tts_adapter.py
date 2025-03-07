"""
Text-to-speech adapter for the Speech UI.

This module provides a PyQt wrapper around the TTS adapters.
"""

import os
import time
import threading
import random
import math
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# Import centralized constants
from speech_mcp.constants import ENV_TTS_VOICE

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
            # Try to import the TTS adapters
            from speech_mcp.tts_adapters import KokoroTTS, Pyttsx3TTS
            
            # First try Kokoro (our primary TTS engine)
            try:
                self.tts_engine = KokoroTTS()
                if not self.tts_engine.is_initialized:
                    raise ImportError("Kokoro initialization failed")
            except ImportError:
                # Fall back to pyttsx3
                try:
                    self.tts_engine = Pyttsx3TTS()
                    if not self.tts_engine.is_initialized:
                        raise ImportError("pyttsx3 initialization failed")
                except ImportError:
                    self.tts_engine = None
            
            # If we have a TTS engine, get the available voices
            if self.tts_engine:
                self.available_voices = self.tts_engine.get_available_voices()
                self.current_voice = self.tts_engine.voice
                return True
            else:
                return False
                
        except ImportError:
            # Direct fallback to pyttsx3 if adapters are not available
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                
                # Get available voices
                voices = self.tts_engine.getProperty('voices')
                self.available_voices = [f"pyttsx3:{voice.id}" for voice in voices]
                if self.available_voices:
                    self.current_voice = self.available_voices[0]
                
                return True
            except ImportError:
                pass
            except Exception:
                pass
            
            return False
            
        except Exception:
            return False
    
    def speak(self, text):
        """Speak the given text"""
        if not text:
            return False
        
        if not self.tts_engine:
            return False
        
        # Use a lock to safely check and update speaking state
        with self._speaking_lock:
            if self.is_speaking:
                return False
            
            # Set speaking state before starting thread
            self.is_speaking = True
        
        # Emit speaking started signal on the main thread
        self.speaking_started.emit()
        
        # Start speaking in a separate thread
        speak_thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        speak_thread.start()
        return True
    
    def emit_audio_level(self):
        """Emit audio level signal for visualization"""
        # Use the lock to safely check the speaking state
        with self._speaking_lock:
            is_speaking = self.is_speaking
        
        if not is_speaking:
            if hasattr(self, 'audio_level_timer') and self.audio_level_timer.isActive():
                self.audio_level_timer.stop()
            self.audio_level.emit(0.0)  # Reset to zero when not speaking
            return
        
        # When speaking, we don't need to emit actual levels since we're using pre-recorded patterns
        # Just emit a dummy signal to trigger visualization updates
        self.audio_level.emit(0.5)
    
    def _speak_thread(self, text):
        """Thread function for speaking text"""
        try:
            # Use the TTS engine's speak method
            if hasattr(self.tts_engine, 'speak'):
                # This is one of our adapters
                try:
                    self.tts_engine.speak(text)
                except Exception:
                    pass
            elif hasattr(self.tts_engine, 'say'):
                # This is direct pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
        except Exception:
            pass
        finally:
            # Use the lock to safely update the speaking state
            with self._speaking_lock:
                self.is_speaking = False
            
            # Emit the signal after releasing the lock
            self.speaking_finished.emit()
    
    def set_voice(self, voice_id):
        """Set the voice to use for TTS"""
        if not self.tts_engine:
            return False
        
        try:
            if hasattr(self.tts_engine, 'set_voice'):
                # This is one of our adapters
                result = self.tts_engine.set_voice(voice_id)
                if result:
                    self.current_voice = voice_id
                    return True
                else:
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
                        return True
                
                return False
            
            return False
        except Exception:
            return False
    
    def get_available_voices(self):
        """Get a list of available voices"""
        return self.available_voices
    
    def get_current_voice(self):
        """Get the current voice"""
        return self.current_voice