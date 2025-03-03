import subprocess
import sys
import os
import json
import logging
import time
from typing import Dict, Optional

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS

mcp = FastMCP("speech")

# Path to save speech state
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_state.json")
TRANSCRIPTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcription.txt")
RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "response.txt")

# Default speech state
DEFAULT_SPEECH_STATE = {
    "ui_active": False,
    "ui_process": None,
    "listening": False,
    "speaking": False,
    "last_transcript": "",
    "last_response": ""
}

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech-mcp-server.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load speech state from file or use default
def load_speech_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # UI process can't be serialized, so we set it to None
                state["ui_process"] = None
                return state
        else:
            return DEFAULT_SPEECH_STATE.copy()
    except Exception as e:
        logger.error(f"Error loading speech state: {e}")
        return DEFAULT_SPEECH_STATE.copy()

# Save speech state to file
def save_speech_state(state):
    try:
        # Create a copy of the state without the UI process
        state_copy = state.copy()
        state_copy.pop("ui_process", None)
        
        with open(STATE_FILE, 'w') as f:
            json.dump(state_copy, f)
    except Exception as e:
        logger.error(f"Error saving speech state: {e}")

# Initialize speech state
speech_state = load_speech_state()

def ensure_ui_running():
    """Ensure that the UI is running, start it if not"""
    global speech_state
    
    if speech_state["ui_active"] and speech_state["ui_process"] is not None:
        # Check if the process is still running
        if speech_state["ui_process"].poll() is None:
            return True
    
    # UI is not running, start it
    try:
        # Kill any existing UI process
        if speech_state["ui_process"] is not None:
            try:
                speech_state["ui_process"].terminate()
            except:
                pass
        
        # Start a new UI process
        python_executable = sys.executable
        script_path = os.path.join(os.path.dirname(__file__), "ui", "__init__.py")
        speech_state["ui_process"] = subprocess.Popen([python_executable, script_path])
        speech_state["ui_active"] = True
        
        # Save the updated state
        save_speech_state(speech_state)
        
        # Give the UI time to start up
        time.sleep(1)
        
        return True
    except Exception as e:
        logger.error(f"Failed to start UI: {e}")
        return False

@mcp.tool()
def start_voice_mode() -> str:
    """
    Start the voice mode UI with audio visualizers.
    
    This will open a separate window with audio visualizers for both user and agent speech.
    """
    global speech_state
    
    # Reload speech state to ensure we have the latest
    speech_state = load_speech_state()
    
    if ensure_ui_running():
        return "Voice mode activated. The UI is now open with audio visualizers. Note that Whisper model loading may take a moment on first use."
    else:
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                "Failed to start the voice mode UI."
            )
        )

@mcp.tool()
def listen() -> str:
    """
    Start listening for user speech and return the transcription.
    
    This will activate the microphone and visualize the audio input until the user stops speaking.
    Returns:
        A string containing the transcription of the user's speech.
    """
    global speech_state
    
    # Reload speech state to ensure we have the latest
    speech_state = load_speech_state()
    
    # Ensure the UI is running
    if not ensure_ui_running():
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                "Failed to start the voice mode UI for listening."
            )
        )
    
    # Set listening state
    speech_state["listening"] = True
    save_speech_state(speech_state)
    
    try:
        # Wait for the transcription file to be created by the UI
        logger.info("Waiting for speech input and transcription...")
        
        # Delete any existing transcription file to avoid using old data
        if os.path.exists(TRANSCRIPTION_FILE):
            os.remove(TRANSCRIPTION_FILE)
        
        timeout = 60  # 60 seconds timeout
        start_time = time.time()
        
        while not os.path.exists(TRANSCRIPTION_FILE) and time.time() - start_time < timeout:
            time.sleep(0.5)
        
        if not os.path.exists(TRANSCRIPTION_FILE):
            speech_state["listening"] = False
            save_speech_state(speech_state)
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    "Timeout waiting for speech transcription."
                )
            )
        
        # Read the transcription
        with open(TRANSCRIPTION_FILE, 'r') as f:
            transcription = f.read().strip()
        
        logger.info(f"Received transcription: {transcription}")
        
        # Delete the file to prepare for the next transcription
        os.remove(TRANSCRIPTION_FILE)
        
        # Update state
        speech_state["listening"] = False
        speech_state["last_transcript"] = transcription
        save_speech_state(speech_state)
        
        return transcription
    except Exception as e:
        # Update state on error
        speech_state["listening"] = False
        save_speech_state(speech_state)
        
        logger.error(f"Error during speech recognition: {e}")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"Error during speech recognition: {str(e)}"
            )
        )

@mcp.tool()
def speak(text: str) -> str:
    """
    Convert text to speech and play it through the audio visualizer.
    
    Args:
        text: The text to be spoken
        
    Returns:
        A confirmation message
    """
    global speech_state
    
    # Reload speech state to ensure we have the latest
    speech_state = load_speech_state()
    
    # Ensure the UI is running
    if not ensure_ui_running():
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                "Failed to start the voice mode UI for speaking."
            )
        )
    
    if not text:
        raise McpError(
            ErrorData(
                INVALID_PARAMS,
                "No text provided to speak."
            )
        )
    
    # Set speaking state
    speech_state["speaking"] = True
    speech_state["last_response"] = text
    save_speech_state(speech_state)
    
    try:
        logger.info(f"Speaking: {text}")
        
        # Write the text to a file for the UI to process
        with open(RESPONSE_FILE, 'w') as f:
            f.write(text)
        
        # Give the UI time to process and "speak" the text
        # We'll estimate the speaking time based on text length
        # Average speaking rate is about 150 words per minute or 2.5 words per second
        # Assuming an average of 5 characters per word
        words = len(text) / 5
        speaking_time = words / 2.5  # Time in seconds
        
        # Add a small buffer
        speaking_time += 1.0
        
        # Wait for the estimated speaking time
        time.sleep(speaking_time)
        
        # Update state
        speech_state["speaking"] = False
        save_speech_state(speech_state)
        
        return f"Spoke: {text}"
    except Exception as e:
        # Update state on error
        speech_state["speaking"] = False
        save_speech_state(speech_state)
        
        logger.error(f"Error during text-to-speech: {e}")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"Error during text-to-speech: {str(e)}"
            )
        )

@mcp.tool()
def get_speech_state() -> str:
    """
    Get the current state of the speech system.
    
    Returns:
        A string representation of the current speech state
    """
    global speech_state
    
    # Reload speech state to ensure we have the latest
    speech_state = load_speech_state()
    
    state_str = f"""
Speech UI Active: {speech_state["ui_active"]}
Currently Listening: {speech_state["listening"]}
Currently Speaking: {speech_state["speaking"]}
Last Transcript: "{speech_state["last_transcript"]}"
Last Response: "{speech_state["last_response"]}"
"""
    
    return state_str

@mcp.resource(uri="mcp://speech/usage_guide")
def usage_guide() -> str:
    """
    Return the usage guide for the Speech MCP.
    """
    return """
    # Speech MCP Usage Guide
    
    This MCP extension provides voice interaction capabilities with audio visualization.
    
    ## How to Use
    
    1. Start the voice mode UI:
       ```
       start_voice_mode()
       ```
       This opens a window with two audio visualizers - one for user input and one for agent output.
       Note: The first time you run this, it will download the Whisper model which may take a moment.
    
    2. Listen for user speech:
       ```
       transcript = listen()
       ```
       This activates the microphone and visualizes the audio input until the user stops speaking.
       The function returns the transcription of the speech using OpenAI's Whisper model.
    
    3. Respond with speech:
       ```
       speak("Your response text here")
       ```
       This converts the text to speech and plays it through the audio visualizer.
    
    4. Check the current state:
       ```
       get_speech_state()
       ```
       This returns information about the current state of the speech system.
    
    ## Typical Workflow
    
    1. Start the voice mode UI
    2. Listen for user input
    3. Process the transcribed speech
    4. Respond with speech
    5. Listen again for the next user input
    
    ## Example Conversation Flow
    
    ```python
    # Start the voice mode
    start_voice_mode()
    
    # Listen for the user's question
    user_input = listen()
    
    # Process the input and generate a response
    # ...
    
    # Speak the response
    speak("Here's my response to your question.")
    
    # Listen for follow-up
    next_input = listen()
    ```
    
    ## Tips
    
    - The UI visualizes audio in real-time using circular audio visualizers
    - The left visualizer shows user input, the right one shows agent output
    - For best results, use a quiet environment and speak clearly
    - The system automatically detects silence to know when you've finished speaking
    """