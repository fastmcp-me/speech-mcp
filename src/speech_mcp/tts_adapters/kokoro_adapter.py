"""
Kokoro TTS adapter for speech-mcp

This adapter allows the speech-mcp extension to use Kokoro for text-to-speech.
It provides a fallback mechanism to use pyttsx3 if Kokoro is not available.

Usage:
    from speech_mcp.tts_adapters.kokoro_adapter import KokoroTTS
    
    # Initialize the TTS engine
    tts = KokoroTTS()
    
    # Speak text
    tts.speak("Hello, world!")
"""

import os
import sys
import logging
import tempfile
import time
import threading
import importlib.util
from typing import Optional, Dict, Any, List

# Set up logging
logger = logging.getLogger(__name__)

class KokoroTTS:
    """
    Text-to-speech adapter for Kokoro
    
    This class provides an interface to use Kokoro for TTS, with a fallback
    to pyttsx3 if Kokoro is not available.
    """
    
    def __init__(self, voice: str = "af_heart", lang_code: str = "a", speed: float = 1.0):
        """
        Initialize the Kokoro TTS adapter
        
        Args:
            voice: The voice to use (default: "af_heart")
            lang_code: The language code to use (default: "a" for American English)
            speed: The speaking speed (default: 1.0)
        """
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.kokoro_available = False
        self.pipeline = None
        self.pyttsx3_engine = None
        
        # Try to import Kokoro
        try:
            # Check if Kokoro is installed
            if importlib.util.find_spec("kokoro") is not None:
                try:
                    # Import Kokoro
                    from kokoro import KPipeline
                    self.pipeline = KPipeline(lang_code=self.lang_code)
                    self.kokoro_available = True
                    logger.info(f"Kokoro TTS initialized successfully with voice={voice}, lang_code={lang_code}, speed={speed}")
                    print(f"Kokoro TTS initialized successfully with voice={voice}!")
                except ImportError as e:
                    logger.warning(f"Failed to import Kokoro module: {e}")
                    print(f"Failed to import Kokoro module: {e}")
                    self._setup_pyttsx3_fallback()
                except Exception as e:
                    logger.error(f"Error initializing Kokoro: {e}")
                    print(f"Error initializing Kokoro: {e}")
                    self._setup_pyttsx3_fallback()
            else:
                logger.warning("Kokoro not found, falling back to pyttsx3")
                print("Kokoro not found, falling back to pyttsx3")
                self._setup_pyttsx3_fallback()
        except ImportError:
            logger.warning("Failed to import Kokoro, falling back to pyttsx3")
            print("Failed to import Kokoro, falling back to pyttsx3")
            self._setup_pyttsx3_fallback()
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            print(f"Error initializing Kokoro TTS: {e}")
            self._setup_pyttsx3_fallback()
    
    def _setup_pyttsx3_fallback(self):
        """Set up pyttsx3 as a fallback TTS engine"""
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            logger.info("pyttsx3 fallback initialized successfully")
            print("pyttsx3 fallback initialized successfully!")
        except ImportError:
            logger.error("pyttsx3 not available for fallback")
            print("pyttsx3 not available for fallback")
            self.pyttsx3_engine = None
        except Exception as e:
            logger.error(f"Error initializing pyttsx3 fallback: {e}")
            print(f"Error initializing pyttsx3 fallback: {e}")
            self.pyttsx3_engine = None
    
    def speak(self, text: str) -> bool:
        """
        Speak the given text using Kokoro or fallback to pyttsx3
        
        Args:
            text: The text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            logger.warning("Empty text provided to speak")
            return False
        
        logger.info(f"Speaking text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Speaking: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        
        # Try Kokoro first - this is our primary TTS engine
        if self.kokoro_available and self.pipeline is not None:
            try:
                logger.info(f"Using Kokoro for TTS with voice={self.voice}, speed={self.speed}")
                print(f"Using Kokoro voice: {self.voice}")
                
                # Generate audio using Kokoro
                generator = self.pipeline(
                    text, voice=self.voice,
                    speed=self.speed
                )
                
                # Process each segment
                for i, (gs, ps, audio) in enumerate(generator):
                    # Save audio to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                        
                        # Save audio data to file
                        import soundfile as sf
                        sf.write(temp_audio_path, audio, 24000)
                        
                        # Play audio using a system command
                        if sys.platform == "darwin":  # macOS
                            os.system(f"afplay {temp_audio_path}")
                        elif sys.platform == "win32":  # Windows
                            os.system(f"start /min powershell -c (New-Object Media.SoundPlayer '{temp_audio_path}').PlaySync()")
                        else:  # Linux and others
                            os.system(f"aplay {temp_audio_path}")
                        
                        # Clean up
                        try:
                            os.unlink(temp_audio_path)
                        except:
                            pass
                
                logger.info("Kokoro TTS completed successfully")
                return True
            except Exception as e:
                logger.error(f"Error using Kokoro for TTS: {e}")
                print(f"Error using Kokoro for TTS: {e}")
                print("Falling back to pyttsx3...")
                # Fall back to pyttsx3
        else:
            logger.info("Kokoro not available, using pyttsx3 fallback")
            print("Kokoro not available, using pyttsx3 fallback")
        
        # Fall back to pyttsx3 if Kokoro failed or is not available
        if self.pyttsx3_engine is not None:
            try:
                logger.info("Using pyttsx3 fallback for TTS")
                print("Using pyttsx3 fallback for TTS")
                self.pyttsx3_engine.say(text)
                self.pyttsx3_engine.runAndWait()
                logger.info("pyttsx3 TTS completed successfully")
                return True
            except Exception as e:
                logger.error(f"Error using pyttsx3 fallback: {e}")
                print(f"Error using pyttsx3 fallback: {e}")
        else:
            logger.warning("pyttsx3 fallback not available")
            print("pyttsx3 fallback not available")
        
        # If we got here, both Kokoro and pyttsx3 failed
        logger.error("All TTS methods failed")
        print("All TTS methods failed")
        return False
    
    def get_available_voices(self) -> List[str]:
        """
        Get a list of available voices
        
        Returns:
            List[str]: List of available voice names
        """
        voices = []
        
        # Get Kokoro voices if available
        if self.kokoro_available and self.pipeline is not None:
            try:
                # Kokoro doesn't have a direct API to get voices, so we'll list the common ones
                voices = [
                    "af_heart", "af_chill", "af_robot", "af_bright", "af_serious",
                    "am_casual", "am_calm", "am_serious", "am_happy"
                ]
                logger.info(f"Found {len(voices)} Kokoro voices")
                print(f"Found {len(voices)} Kokoro voices")
                
                # Log the current voice
                logger.info(f"Current Kokoro voice: {self.voice}")
                print(f"Current Kokoro voice: {self.voice}")
            except Exception as e:
                logger.error(f"Error getting Kokoro voices: {e}")
                print(f"Error getting Kokoro voices: {e}")
        else:
            logger.info("Kokoro not available, listing only fallback voices")
            print("Kokoro not available, listing only fallback voices")
        
        # Get pyttsx3 voices if available
        if self.pyttsx3_engine is not None:
            try:
                pyttsx3_voices = self.pyttsx3_engine.getProperty('voices')
                for voice in pyttsx3_voices:
                    voices.append(f"pyttsx3:{voice.id}")
                logger.info(f"Found {len(pyttsx3_voices)} pyttsx3 voices")
                print(f"Found {len(pyttsx3_voices)} pyttsx3 voices")
            except Exception as e:
                logger.error(f"Error getting pyttsx3 voices: {e}")
                print(f"Error getting pyttsx3 voices: {e}")
        
        return voices
    
    def set_voice(self, voice: str) -> bool:
        """
        Set the voice to use
        
        Args:
            voice: The voice to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if the voice is a pyttsx3 voice
            if voice.startswith("pyttsx3:") and self.pyttsx3_engine is not None:
                voice_id = voice.split(":", 1)[1]
                logger.info(f"Setting pyttsx3 voice to: {voice_id}")
                print(f"Setting pyttsx3 voice to: {voice_id}")
                
                # Find the voice object
                for v in self.pyttsx3_engine.getProperty('voices'):
                    if v.id == voice_id:
                        self.pyttsx3_engine.setProperty('voice', v.id)
                        logger.info(f"pyttsx3 voice set to: {v.name}")
                        print(f"pyttsx3 voice set to: {v.name}")
                        break
                
                # Also update the internal voice name
                self.voice = voice
                return True
            else:
                # Assume it's a Kokoro voice
                old_voice = self.voice
                self.voice = voice
                logger.info(f"Kokoro voice changed from {old_voice} to {voice}")
                print(f"Kokoro voice changed from {old_voice} to {voice}")
                return True
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            print(f"Error setting voice: {e}")
            return False
    
    def set_speed(self, speed: float) -> bool:
        """
        Set the speaking speed
        
        Args:
            speed: The speaking speed (1.0 is normal)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            old_speed = self.speed
            self.speed = speed
            logger.info(f"Speed changed from {old_speed} to {speed}")
            print(f"TTS speed changed from {old_speed} to {speed}")
            
            # Also set speed for pyttsx3 if available
            if self.pyttsx3_engine is not None:
                try:
                    # pyttsx3 uses words per minute, default is around 200
                    # Convert our speed factor to words per minute
                    rate = int(200 * speed)
                    self.pyttsx3_engine.setProperty('rate', rate)
                    logger.info(f"pyttsx3 rate set to: {rate} words per minute")
                    print(f"pyttsx3 rate set to: {rate} words per minute")
                except Exception as e:
                    logger.error(f"Error setting pyttsx3 rate: {e}")
                    print(f"Error setting pyttsx3 rate: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error setting speed: {e}")
            print(f"Error setting speed: {e}")
            return False