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

# Import base adapter class
from speech_mcp.tts_adapters import BaseTTSAdapter

# Import pyttsx3 adapter for fallback
try:
    from speech_mcp.tts_adapters.pyttsx3_adapter import Pyttsx3TTS
except ImportError:
    Pyttsx3TTS = None

# Import centralized constants
from speech_mcp.constants import ENV_TTS_VOICE

# Set up logging
logger = logging.getLogger(__name__)

class KokoroTTS(BaseTTSAdapter):
    """
    Text-to-speech adapter for Kokoro
    
    This class provides an interface to use Kokoro for TTS, with a fallback
    to pyttsx3 if Kokoro is not available.
    """
    
    def __init__(self, voice: str = None, lang_code: str = "a", speed: float = 1.0):
        """
        Initialize the Kokoro TTS adapter
        
        Args:
            voice: The voice to use (default from config or "af_heart")
            lang_code: The language code to use (default: "a" for American English)
            speed: The speaking speed (default: 1.0)
        """
        # Call parent constructor to initialize common attributes
        super().__init__(voice, lang_code, speed)
        
        # Set default voice if none provided
        if self.voice is None:
            self.voice = "af_heart"
        
        self.kokoro_available = False
        self.pipeline = None
        self.fallback_tts = None
        
        # Initialize Kokoro
        self._initialize_kokoro()
        
        # If Kokoro initialization failed, set up fallback
        if not self.kokoro_available:
            self._setup_fallback()
    
    def _initialize_kokoro(self) -> bool:
        """
        Initialize Kokoro TTS engine
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if Kokoro is installed
            if importlib.util.find_spec("kokoro") is not None:
                try:
                    # Import Kokoro
                    from kokoro import KPipeline
                    self.pipeline = KPipeline(lang_code=self.lang_code)
                    self.kokoro_available = True
                    self.is_initialized = True
                    logger.info(f"Kokoro TTS initialized successfully with voice={self.voice}, lang_code={self.lang_code}, speed={self.speed}")
                    print(f"Kokoro TTS initialized successfully with voice={self.voice}!")
                    return True
                except ImportError as e:
                    logger.warning(f"Failed to import Kokoro module: {e}")
                    print(f"Failed to import Kokoro module: {e}")
                except Exception as e:
                    logger.error(f"Error initializing Kokoro: {e}")
                    print(f"Error initializing Kokoro: {e}")
            else:
                logger.warning("Kokoro not found, will use fallback")
                print("Kokoro not found, will use fallback")
        except ImportError:
            logger.warning("Failed to import Kokoro, will use fallback")
            print("Failed to import Kokoro, will use fallback")
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            print(f"Error initializing Kokoro TTS: {e}")
        
        return False
    
    def _setup_fallback(self) -> bool:
        """
        Set up fallback TTS engine (pyttsx3)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to use the Pyttsx3TTS adapter
            if Pyttsx3TTS is not None:
                self.fallback_tts = Pyttsx3TTS(voice=None, lang_code=self.lang_code, speed=self.speed)
                if self.fallback_tts.is_initialized:
                    logger.info("pyttsx3 fallback initialized successfully")
                    print("pyttsx3 fallback initialized successfully!")
                    self.is_initialized = True
                    return True
            
            # If Pyttsx3TTS adapter is not available, try direct import
            import pyttsx3
            from speech_mcp.tts_adapters.pyttsx3_adapter import Pyttsx3TTS
            self.fallback_tts = Pyttsx3TTS(voice=None, lang_code=self.lang_code, speed=self.speed)
            if self.fallback_tts.is_initialized:
                logger.info("pyttsx3 fallback initialized successfully")
                print("pyttsx3 fallback initialized successfully!")
                self.is_initialized = True
                return True
        except ImportError:
            logger.error("pyttsx3 not available for fallback")
            print("pyttsx3 not available for fallback")
        except Exception as e:
            logger.error(f"Error initializing pyttsx3 fallback: {e}")
            print(f"Error initializing pyttsx3 fallback: {e}")
        
        self.fallback_tts = None
        return False
    
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
                try:
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
                    logger.error(f"Error in Kokoro pipeline: {e}", exc_info=True)
                    print(f"Error in Kokoro pipeline: {e}")
                    # Fall back to pyttsx3
                    raise
            except Exception as e:
                logger.error(f"Error using Kokoro for TTS: {e}")
                print(f"Error using Kokoro for TTS: {e}")
                print("Falling back to pyttsx3...")
                # Fall back to pyttsx3
        else:
            logger.info("Kokoro not available, using fallback")
            print("Kokoro not available, using fallback")
        
        # Fall back to pyttsx3 if Kokoro failed or is not available
        if self.fallback_tts is not None:
            try:
                logger.info("Using pyttsx3 fallback for TTS")
                print("Using pyttsx3 fallback for TTS")
                return self.fallback_tts.speak(text)
            except Exception as e:
                logger.error(f"Error using pyttsx3 fallback: {e}")
                print(f"Error using pyttsx3 fallback: {e}")
        else:
            logger.warning("No fallback TTS available")
            print("No fallback TTS available")
        
        # If we got here, both Kokoro and fallback failed
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
                # List of available Kokoro voice models
                voices = [
                    # American Female voices
                    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", 
                    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
                    
                    # American Male voices
                    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                    "am_michael", "am_onyx", "am_puck", "am_santa",
                    
                    # British Female voices
                    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                    
                    # British Male voices
                    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
                    
                    # Other English voices
                    "ef_dora", "em_alex", "em_santa",
                    
                    # French voice
                    "ff_siwis",
                    
                    # Hindi voices
                    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
                    
                    # Italian voices
                    "if_sara", "im_nicola",
                    
                    # Japanese voices
                    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
                    
                    # Portuguese voices
                    "pf_dora", "pm_alex", "pm_santa",
                    
                    # Chinese voices
                    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"
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
            logger.info("Kokoro not available")
            print("Kokoro not available")
        
        # Get fallback voices if available
        if self.fallback_tts is not None:
            try:
                fallback_voices = self.fallback_tts.get_available_voices()
                voices.extend(fallback_voices)
                logger.info(f"Found {len(fallback_voices)} fallback voices")
                print(f"Found {len(fallback_voices)} fallback voices")
            except Exception as e:
                logger.error(f"Error getting fallback voices: {e}")
                print(f"Error getting fallback voices: {e}")
        
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
            # Check if the voice is for the fallback TTS
            if voice.startswith("pyttsx3:") and self.fallback_tts is not None:
                result = self.fallback_tts.set_voice(voice)
                if result:
                    # Update our own voice property and save preference
                    super().set_voice(voice)
                return result
            else:
                # Assume it's a Kokoro voice
                old_voice = self.voice
                
                # Update voice property and save preference
                super().set_voice(voice)
                
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
            # Update speed property
            super().set_speed(speed)
            
            # Also set speed for fallback TTS if available
            if self.fallback_tts is not None:
                try:
                    self.fallback_tts.set_speed(speed)
                except Exception as e:
                    logger.error(f"Error setting fallback TTS speed: {e}")
                    print(f"Error setting fallback TTS speed: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error setting speed: {e}")
            print(f"Error setting speed: {e}")
            return False