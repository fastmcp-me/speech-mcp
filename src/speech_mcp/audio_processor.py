"""
Audio processing module for speech-mcp.

This module provides centralized audio processing functionality including:
- Audio device selection
- Audio recording
- Audio playback
- Audio level visualization
- Silence detection
"""

import os
import time
import logging
import tempfile
import threading
import wave
import numpy as np
import pyaudio
from typing import Optional, List, Tuple, Callable, Any, Dict

# Import centralized constants
from speech_mcp.constants import (
    CHUNK, FORMAT, CHANNELS, RATE,
    SILENCE_THRESHOLD, MAX_SILENCE_DURATION, SILENCE_CHECK_INTERVAL,
    START_LISTENING_SOUND, STOP_LISTENING_SOUND
)

# Setup logging
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Core audio processing class that handles device selection, recording, and playback.
    
    This class provides the shared audio functionality used by both the server and UI components.
    """
    
    def __init__(self, on_audio_level: Optional[Callable[[float], None]] = None):
        """
        Initialize the audio processor.
        
        Args:
            on_audio_level: Optional callback function that receives audio level updates (0.0 to 1.0)
        """
        self.pyaudio = None
        self.stream = None
        self.selected_device_index = None
        self.is_listening = False
        self.audio_frames = []
        self.on_audio_level = on_audio_level
        self._setup_audio()
    
    def _setup_audio(self) -> None:
        """Set up audio capture and processing."""
        try:
            logger.info("Setting up audio processing")
            self.pyaudio = pyaudio.PyAudio()
            
            # Log audio device information
            logger.info(f"PyAudio version: {pyaudio.get_portaudio_version()}")
            
            # Get all available audio devices
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            logger.info(f"Found {numdevices} audio devices:")
            
            # Find the best input device
            for i in range(numdevices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                    device_name = device_info.get('name')
                    max_input_channels = device_info.get('maxInputChannels')
                    
                    logger.info(f"Device {i}: {device_name}")
                    logger.info(f"  Max Input Channels: {max_input_channels}")
                    logger.info(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
                    
                    # Only consider input devices
                    if max_input_channels > 0:
                        logger.info(f"Found input device: {device_name}")
                        print(f"Found input device: {device_name}")
                        
                        # Prefer non-default devices as they're often external mics
                        if self.selected_device_index is None or 'default' not in device_name.lower():
                            self.selected_device_index = i
                            logger.info(f"Selected input device: {device_name} (index {i})")
                            print(f"Using input device: {device_name}")
                except Exception as e:
                    logger.warning(f"Error checking device {i}: {e}")
            
            if self.selected_device_index is None:
                logger.warning("No suitable input device found, using default")
                print("No suitable input device found, using default")
            
        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
            print(f"Error setting up audio: {e}")
    
    def start_listening(self, callback: Optional[Callable] = None) -> bool:
        """
        Start listening for audio input.
        
        Args:
            callback: Optional callback function to call when audio data is received
            
        Returns:
            bool: True if listening started successfully, False otherwise
        """
        if self.is_listening:
            logger.info("Already listening, ignoring start_listening call")
            return True
            
        self.is_listening = True
        self.audio_frames = []
        
        # Play start listening notification sound
        threading.Thread(target=self.play_audio_file, args=(START_LISTENING_SOUND,), daemon=True).start()
        
        try:
            logger.info("Starting audio recording")
            print("\nRecording audio... Speak now.")
            
            def audio_callback(in_data, frame_count, time_info, status):
                try:
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
                    self.audio_frames.append(in_data)
                    
                    # Process audio for visualization
                    self._process_audio_for_visualization(in_data)
                    
                    # Call user-provided callback if available
                    if callback:
                        callback(in_data)
                    
                    return (in_data, pyaudio.paContinue)
                    
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
                    return (in_data, pyaudio.paContinue)  # Try to continue despite errors
            
            # Start the audio stream with the selected device
            logger.debug(f"Opening audio stream with FORMAT={FORMAT}, CHANNELS={CHANNELS}, RATE={RATE}, CHUNK={CHUNK}, DEVICE={self.selected_device_index}")
            self.stream = self.pyaudio.open(
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
                self.is_listening = False
                return False
            
            logger.info("Audio stream initialized and receiving data")
            
            # Start a thread to detect silence and stop recording
            threading.Thread(target=self._detect_silence, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            print(f"ERROR: Failed to start audio recording: {e}")
            self.is_listening = False
            return False
    
    def _process_audio_for_visualization(self, audio_data: bytes) -> None:
        """
        Process audio data for visualization.
        
        Args:
            audio_data: Raw audio data from PyAudio
        """
        try:
            # Convert to numpy array
            data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize the data to range [-1, 1]
            normalized = data.astype(float) / 32768.0
            
            # Take absolute value to get amplitude
            amplitude = np.abs(normalized).mean()
            
            # Apply amplification factor to make the visualization more prominent
            # Increase the factor from 1.0 to 5.0 to make the visualization more visible
            amplification_factor = 5.0
            amplified_amplitude = min(amplitude * amplification_factor, 1.0)  # Clamp to 1.0 max
            
            # Log the original and amplified values occasionally for debugging
            if np.random.random() < 0.01:  # Log roughly 1% of values to avoid flooding logs
                logger.debug(f"Audio amplitude: original={amplitude:.6f}, amplified={amplified_amplitude:.6f}")
            
            # Call the audio level callback if provided
            if self.on_audio_level:
                self.on_audio_level(amplified_amplitude)
            
        except Exception as e:
            logger.error(f"Error processing audio for visualization: {e}")
    
    def _detect_silence(self) -> None:
        """
        Detect when the user stops speaking and end recording.
        
        This method runs in a separate thread and monitors audio levels to detect
        when the user has stopped speaking.
        """
        try:
            # Wait for initial audio to accumulate
            logger.info("Starting silence detection")
            time.sleep(0.5)
            
            # Initialize silence detection parameters
            silence_duration = 0
            
            logger.debug(f"Silence detection parameters: threshold={SILENCE_THRESHOLD}, max_silence={MAX_SILENCE_DURATION}s, check_interval={SILENCE_CHECK_INTERVAL}s")
            
            while self.is_listening and self.stream and silence_duration < MAX_SILENCE_DURATION:
                if not self.audio_frames or len(self.audio_frames) < 2:
                    time.sleep(SILENCE_CHECK_INTERVAL)
                    continue
                
                # Get the latest audio frame
                latest_frame = self.audio_frames[-1]
                audio_data = np.frombuffer(latest_frame, dtype=np.int16)
                normalized = audio_data.astype(float) / 32768.0
                current_amplitude = np.abs(normalized).mean()
                
                if current_amplitude < SILENCE_THRESHOLD:
                    silence_duration += SILENCE_CHECK_INTERVAL
                    # Log only when silence is detected
                    if silence_duration >= 1.0 and silence_duration % 1.0 < SILENCE_CHECK_INTERVAL:
                        logger.debug(f"Silence detected for {silence_duration:.1f}s, amplitude: {current_amplitude:.6f}")
                else:
                    if silence_duration > 0:
                        logger.debug(f"Speech resumed after {silence_duration:.1f}s of silence, amplitude: {current_amplitude:.6f}")
                    silence_duration = 0
                
                time.sleep(SILENCE_CHECK_INTERVAL)
            
            # If we exited because of silence detection
            if self.is_listening and self.stream:
                logger.info(f"Silence threshold reached after {silence_duration:.1f}s, stopping recording")
                print("Silence detected. Processing speech...")
                self.stop_listening()
            
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
    
    def stop_listening(self) -> None:
        """
        Stop listening for audio input.
        
        Returns:
            None
        """
        try:
            logger.info("Stopping audio recording")
            if self.stream:
                logger.debug(f"Stopping audio stream, stream active: {self.stream.is_active()}")
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                logger.info("Audio stream closed successfully")
                
                # Play stop listening notification sound
                threading.Thread(target=self.play_audio_file, args=(STOP_LISTENING_SOUND,), daemon=True).start()
            
            self.is_listening = False
            
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
            self.is_listening = False
    
    def get_recorded_audio_path(self) -> Optional[str]:
        """
        Save the recorded audio to a temporary WAV file and return the path.
        
        Returns:
            str: Path to the temporary WAV file, or None if an error occurred
        """
        if not self.audio_frames:
            logger.warning("No audio frames to process")
            return None
        
        try:
            logger.info(f"Processing {len(self.audio_frames)} audio frames")
            
            # Check if we have enough audio data
            total_audio_time = len(self.audio_frames) * (CHUNK / RATE)
            logger.info(f"Total recorded audio: {total_audio_time:.2f} seconds")
            
            if total_audio_time < 0.5:  # Less than half a second of audio
                logger.warning(f"Audio recording too short ({total_audio_time:.2f}s), may not contain speech")
            
            # Save the recorded audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
                # Create a WAV file from the recorded frames
                logger.debug(f"Creating WAV file at {temp_audio_path}")
                wf = wave.open(temp_audio_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.pyaudio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
                
                # Get file size for logging
                file_size = os.path.getsize(temp_audio_path)
                logger.debug(f"WAV file created, size: {file_size} bytes")
            
            logger.info(f"Audio saved to temporary file: {temp_audio_path}")
            return temp_audio_path
            
        except Exception as e:
            logger.error(f"Error saving audio to file: {e}")
            return None
    
    def record_audio(self) -> Optional[str]:
        """
        Record audio from the microphone and return the path to the audio file.
        
        This is a blocking method that handles the entire recording process including
        starting recording, detecting silence, and stopping recording.
        
        Returns:
            str: Path to the recorded audio file, or None if an error occurred
        """
        if not self.start_listening():
            return None
        
        # Wait for recording to complete (silence detection will stop it)
        while self.is_listening:
            time.sleep(0.1)
        
        # Get the recorded audio file path
        return self.get_recorded_audio_path()
    
    def play_audio_file(self, file_path: str) -> bool:
        """
        Play an audio file using PyAudio.
        
        Args:
            file_path: Path to the audio file to play
            
        Returns:
            bool: True if the file was played successfully, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return False
            
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
                return True
        except Exception as e:
            logger.error(f"Error playing audio notification: {e}")
            return False
    
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """
        Get a list of available audio input devices.
        
        Returns:
            List of dictionaries containing device information
        """
        devices = []
        
        try:
            if not self.pyaudio:
                self._setup_audio()
                
            if not self.pyaudio:
                logger.error("PyAudio not initialized")
                return devices
                
            # Get all available audio devices
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            for i in range(numdevices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                    max_input_channels = device_info.get('maxInputChannels')
                    
                    # Only include input devices
                    if max_input_channels > 0:
                        devices.append({
                            'index': i,
                            'name': device_info.get('name'),
                            'channels': max_input_channels,
                            'sample_rate': device_info.get('defaultSampleRate')
                        })
                except Exception as e:
                    logger.warning(f"Error getting device info for device {i}: {e}")
                    
            return devices
            
        except Exception as e:
            logger.error(f"Error getting available devices: {e}")
            return devices
    
    def set_device_index(self, device_index: int) -> bool:
        """
        Set the audio input device by index.
        
        Args:
            device_index: Index of the audio device to use
            
        Returns:
            bool: True if the device was set successfully, False otherwise
        """
        try:
            # Check if the device exists
            if not self.pyaudio:
                self._setup_audio()
                
            if not self.pyaudio:
                logger.error("PyAudio not initialized")
                return False
                
            try:
                device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, device_index)
                if device_info.get('maxInputChannels') > 0:
                    self.selected_device_index = device_index
                    logger.info(f"Selected input device: {device_info.get('name')} (index {device_index})")
                    return True
                else:
                    logger.error(f"Device {device_index} is not an input device")
                    return False
            except Exception as e:
                logger.error(f"Error setting device index {device_index}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting device index: {e}")
            return False
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the audio processor.
        
        This should be called when the audio processor is no longer needed.
        """
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
                
            logger.info("Audio processor resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up audio processor: {e}")