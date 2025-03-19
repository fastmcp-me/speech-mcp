"""
Pyttsx3 TTS adapter for speech-mcp

This adapter allows the speech-mcp extension to use pyttsx3 for text-to-speech.
It provides a fallback mechanism when more advanced TTS engines are not available.

Usage:
    from speech_mcp.tts_adapters.pyttsx3_adapter import Pyttsx3TTS
    
    # Initialize the TTS engine
    tts = Pyttsx3TTS()
    
    # Speak text
    tts.speak("Hello, world!")
"""

import os
import sys
import threading
from typing import List, Dict, Any, Optional

# Import base adapter class
from speech_mcp.tts_adapters import BaseTTSAdapter

class Pyttsx3TTS(BaseTTSAdapter):
    """
    Text-to-speech adapter for pyttsx3
    
    This class provides an interface to use pyttsx3 for TTS.
    """
    
    def __init__(self, voice: str = None, lang_code: str = "en", speed: float = 1.0):
        """
        Initialize the pyttsx3 TTS adapter
        
        Args:
            voice: The voice to use (default from config or system default)
            lang_code: The language code to use (default: "en" for English)
            speed: The speaking speed (default: 1.0)
        """
        # Call parent constructor to initialize common attributes
        super().__init__(voice, lang_code, speed)
        
        self.engine = None
        self.is_speaking = False
        self._initialize_engine()
    
    def _initialize_engine(self) -> bool:
        """
        Initialize the pyttsx3 engine
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Import pyttsx3
            import pyttsx3
            
            # Initialize engine
            self.engine = pyttsx3.init()
            
            # Set initial properties
            # Convert our speed factor to words per minute (default is around 200)
            rate = int(200 * self.speed)
            self.engine.setProperty('rate', rate)
            
            # If voice was specified, try to set it
            if self.voice:
                # If voice is in our format "pyttsx3:voice_id", extract the ID
                if self.voice.startswith("pyttsx3:"):
                    voice_id = self.voice.split(":", 1)[1]
                else:
                    voice_id = self.voice
                
                # Try to find and set the voice
                for voice in self.engine.getProperty('voices'):
                    if voice.id == voice_id:
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self.is_initialized = True
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def speak(self, text: str) -> bool:
        """
        Speak the given text using pyttsx3
        
        Args:
            text: The text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            return False
        
        if not self.is_initialized or not self.engine:
            return False
        
        # Prevent multiple simultaneous speech
        if self.is_speaking:
            return False
        
        try:
            self.is_speaking = True
            
            # Start speaking in a separate thread to avoid blocking
            threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
            
            return True
        except Exception:
            self.is_speaking = False
            return False
    
    def _speak_thread(self, text: str) -> None:
        """
        Thread function for speaking text
        
        Args:
            text: The text to speak
        """
        try:
            # Add the text to the speech queue
            self.engine.say(text)
            
            # Process the speech queue
            self.engine.runAndWait()
        except Exception:
            pass
        finally:
            self.is_speaking = False
    
    def save_to_file(self, text: str, file_path: str) -> bool:
        """
        Save speech as an audio file using pyttsx3.
        
        Args:
            text: The text to convert to speech
            file_path: Path where to save the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text or not self.is_initialized or not self.engine:
            return False
            
        try:
            # Create a temporary wav file first
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Use pyttsx3's save_to_file
                self.engine.save_to_file(text, temp_path)
                self.engine.runAndWait()
                
                # Convert to the desired format if needed
                import shutil
                shutil.move(temp_path, file_path)
                return True
        except Exception:
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
            return False
    
    def get_available_voices(self) -> List[str]:
        """
        Get a list of available voices
        
        Returns:
            List[str]: List of available voice names in the format "pyttsx3:voice_id"
        """
        voices = []
        
        if not self.is_initialized or not self.engine:
            return voices
        
        try:
            # Get all voices from pyttsx3
            pyttsx3_voices = self.engine.getProperty('voices')
            
            # Format voice IDs with "pyttsx3:" prefix
            for voice in pyttsx3_voices:
                voices.append(f"pyttsx3:{voice.id}")
        except Exception:
            pass
        
        return voices
    
    def set_voice(self, voice: str) -> bool:
        """
        Set the voice to use
        
        Args:
            voice: The voice to use (format: "pyttsx3:voice_id" or just "voice_id")
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized or not self.engine:
            return False
        
        try:
            # Extract the voice ID from the format "pyttsx3:voice_id"
            if voice.startswith("pyttsx3:"):
                voice_id = voice.split(":", 1)[1]
            else:
                voice_id = voice
            
            # Find the voice object
            for v in self.engine.getProperty('voices'):
                if v.id == voice_id:
                    self.engine.setProperty('voice', v.id)
                    
                    # Call parent method to update self.voice and save preference
                    super().set_voice(f"pyttsx3:{voice_id}")
                    return True
            
            return False
        except Exception:
            return False
    
    def set_speed(self, speed: float) -> bool:
        """
        Set the speaking speed
        
        Args:
            speed: The speaking speed (1.0 is normal)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialized or not self.engine:
            return False
        
        try:
            # Call parent method to update self.speed
            super().set_speed(speed)
            
            # pyttsx3 uses words per minute, default is around 200
            # Convert our speed factor to words per minute
            rate = int(200 * speed)
            self.engine.setProperty('rate', rate)
            
            return True
        except Exception:
            return False
