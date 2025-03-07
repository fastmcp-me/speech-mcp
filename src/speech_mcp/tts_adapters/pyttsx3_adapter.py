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
import logging
import threading
from typing import List, Dict, Any, Optional

# Import base adapter class
from speech_mcp.tts_adapters import BaseTTSAdapter

# Set up logging
logger = logging.getLogger(__name__)

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
                        logger.info(f"pyttsx3 voice set to: {voice.name}")
                        break
            
            self.is_initialized = True
            logger.info("pyttsx3 TTS engine initialized successfully")
            return True
        except ImportError as e:
            logger.error(f"Failed to import pyttsx3: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing pyttsx3: {e}")
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
            logger.warning("Empty text provided to speak")
            return False
        
        if not self.is_initialized or not self.engine:
            logger.error("pyttsx3 engine not initialized")
            return False
        
        # Prevent multiple simultaneous speech
        if self.is_speaking:
            logger.warning("Already speaking, ignoring new request")
            return False
        
        logger.info(f"Speaking text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
        
        try:
            self.is_speaking = True
            
            # Start speaking in a separate thread to avoid blocking
            threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
            
            return True
        except Exception as e:
            logger.error(f"Error starting speech thread: {e}")
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
            
            logger.info("pyttsx3 speech completed")
        except Exception as e:
            logger.error(f"Error during text-to-speech: {e}")
        finally:
            self.is_speaking = False
    
    def get_available_voices(self) -> List[str]:
        """
        Get a list of available voices
        
        Returns:
            List[str]: List of available voice names in the format "pyttsx3:voice_id"
        """
        voices = []
        
        if not self.is_initialized or not self.engine:
            logger.warning("pyttsx3 engine not initialized, cannot get voices")
            return voices
        
        try:
            # Get all voices from pyttsx3
            pyttsx3_voices = self.engine.getProperty('voices')
            
            # Format voice IDs with "pyttsx3:" prefix
            for voice in pyttsx3_voices:
                voices.append(f"pyttsx3:{voice.id}")
            
            logger.info(f"Found {len(voices)} pyttsx3 voices")
        except Exception as e:
            logger.error(f"Error getting pyttsx3 voices: {e}")
        
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
            logger.error("pyttsx3 engine not initialized")
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
                    
                    logger.info(f"pyttsx3 voice set to: {v.name}")
                    return True
            
            logger.error(f"Voice not found: {voice}")
            return False
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
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
            logger.error("pyttsx3 engine not initialized")
            return False
        
        try:
            # Call parent method to update self.speed
            super().set_speed(speed)
            
            # pyttsx3 uses words per minute, default is around 200
            # Convert our speed factor to words per minute
            rate = int(200 * speed)
            self.engine.setProperty('rate', rate)
            
            logger.info(f"pyttsx3 rate set to: {rate} words per minute")
            return True
        except Exception as e:
            logger.error(f"Error setting pyttsx3 rate: {e}")
            return False