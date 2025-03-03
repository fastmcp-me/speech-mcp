import subprocess
import sys
import os
import json
import logging
import time
import threading
import asyncio
from typing import Dict, Optional, Callable

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
            logger.info("UI process is already running")
            return True
        else:
            logger.warning("UI process has terminated unexpectedly")
    
    # UI is not running, start it
    try:
        # Kill any existing UI process
        if speech_state["ui_process"] is not None:
            try:
                logger.info("Terminating existing UI process")
                speech_state["ui_process"].terminate()
                time.sleep(1)  # Give it time to terminate
            except Exception as e:
                logger.error(f"Error terminating UI process: {e}")
        
        # Start a new UI process
        python_executable = sys.executable
        
        # Use the module import approach instead of direct file path
        logger.info(f"Starting UI process with Python module import")
        print(f"Starting UI process with Python module import")
        
        # Start the process with stdout and stderr redirected
        speech_state["ui_process"] = subprocess.Popen(
            [python_executable, "-m", "speech_mcp.ui"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Start threads to monitor the process output
        def log_output(stream, level):
            for line in stream:
                # Clean up the line to remove any file references that might be in the output
                clean_line = line.strip()
                if level == "info":
                    logger.info(f"UI: {clean_line}")
                    print(f"UI: {clean_line}")
                else:
                    logger.error(f"UI Error: {clean_line}")
                    print(f"UI Error: {clean_line}")
        
        threading.Thread(target=log_output, args=(speech_state["ui_process"].stdout, "info"), daemon=True).start()
        threading.Thread(target=log_output, args=(speech_state["ui_process"].stderr, "error"), daemon=True).start()
        
        speech_state["ui_active"] = True
        
        # Save the updated state
        save_speech_state(speech_state)
        
        # Give the UI time to start up
        time.sleep(2)  # Increased from 1 to 2 seconds
        
        # Check if the process is still running
        if speech_state["ui_process"].poll() is not None:
            exit_code = speech_state["ui_process"].poll()
            logger.error(f"UI process exited immediately with code {exit_code}")
            return False
        
        logger.info("UI process started successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start UI: {e}")
        print(f"Failed to start UI: {e}")
        return False

def listen_for_speech() -> str:
    """Internal function to listen for speech and return transcription"""
    global speech_state
    
    # Set listening state
    speech_state["listening"] = True
    save_speech_state(speech_state)
    
    logger.info("Starting to listen for speech input...")
    print("\nListening for speech input... Speak now.")
    
    try:
        # Wait for the transcription file to be created by the UI
        logger.info("Waiting for speech input and transcription...")
        
        # Delete any existing transcription file to avoid using old data
        if os.path.exists(TRANSCRIPTION_FILE):
            os.remove(TRANSCRIPTION_FILE)
        
        max_timeout = 600  # 10 minutes maximum timeout
        start_time = time.time()
        
        # Wait for transcription file to appear
        while not os.path.exists(TRANSCRIPTION_FILE) and time.time() - start_time < max_timeout:
            time.sleep(0.5)
            # Print a message every 30 seconds to indicate we're still waiting
            if (time.time() - start_time) % 30 < 0.5:
                print(f"Still waiting for speech input... ({int(time.time() - start_time)} seconds elapsed)")
        
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
        print(f"Transcription received: \"{transcription}\"")
        
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

def speak_text(text: str) -> str:
    """Internal function to speak text"""
    global speech_state
    
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
        print(f"\nSpeaking: \"{text}\"")
        
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
        
        print("Done speaking.")
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

# Background task for listening
async def listen_task(callback: Callable[[str], None]):
    """Background task for listening to speech"""
    try:
        # Ensure the UI is running
        if not ensure_ui_running():
            callback(f"Error: Failed to start the speech recognition system.")
            return
        
        # Set listening state
        speech_state["listening"] = True
        save_speech_state(speech_state)
        
        logger.info("Starting to listen for speech input in background task...")
        print("\nListening for speech input... Speak now.")
        
        # Delete any existing transcription file to avoid using old data
        if os.path.exists(TRANSCRIPTION_FILE):
            os.remove(TRANSCRIPTION_FILE)
        
        # Start a separate thread to monitor for the transcription file
        def monitor_transcription():
            try:
                start_time = time.time()
                max_wait_time = 600  # 10 minutes
                
                while not os.path.exists(TRANSCRIPTION_FILE) and time.time() - start_time < max_wait_time:
                    time.sleep(0.5)
                    # Print a message every 30 seconds to indicate we're still waiting
                    if (time.time() - start_time) % 30 < 0.5:
                        print(f"Still waiting for speech input... ({int(time.time() - start_time)} seconds elapsed)")
                
                if not os.path.exists(TRANSCRIPTION_FILE):
                    speech_state["listening"] = False
                    save_speech_state(speech_state)
                    callback("Error: Timeout waiting for speech transcription.")
                    return
                
                # Read the transcription
                with open(TRANSCRIPTION_FILE, 'r') as f:
                    transcription = f.read().strip()
                
                logger.info(f"Received transcription: {transcription}")
                print(f"Transcription received: \"{transcription}\"")
                
                # Delete the file to prepare for the next transcription
                os.remove(TRANSCRIPTION_FILE)
                
                # Update state
                speech_state["listening"] = False
                speech_state["last_transcript"] = transcription
                save_speech_state(speech_state)
                
                # Call the callback with the transcription
                callback(transcription)
            except Exception as e:
                # Update state on error
                speech_state["listening"] = False
                save_speech_state(speech_state)
                
                logger.error(f"Error during speech recognition: {e}")
                callback(f"Error: {str(e)}")
        
        # Start the monitoring thread
        threading.Thread(target=monitor_transcription, daemon=True).start()
        
    except Exception as e:
        logger.error(f"Error in listen_task: {e}")
        callback(f"Error: {str(e)}")

# Active background tasks
background_tasks = {}

@mcp.tool()
def start_conversation() -> str:
    """
    Start a voice conversation by launching the UI and beginning to listen.
    
    This will initialize the speech recognition system and immediately start listening for user input.
    
    Returns:
        The transcription of the user's speech.
    """
    global speech_state
    
    # Reload speech state to ensure we have the latest
    speech_state = load_speech_state()
    
    # Start the UI
    if not ensure_ui_running():
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                "Failed to start the speech recognition system."
            )
        )
    
    # Give the UI a moment to fully initialize
    time.sleep(2)
    
    # Start listening - using direct approach for this initial call
    try:
        transcription = listen_for_speech()
        return transcription
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"Error starting conversation: {str(e)}"
            )
        )

@mcp.tool()
def reply(text: str) -> str:
    """
    Speak the provided text and then listen for a response.
    
    This will speak the given text and then immediately start listening for user input.
    
    Args:
        text: The text to speak to the user
        
    Returns:
        The transcription of the user's response.
    """
    global speech_state
    
    # Reload speech state to ensure we have the latest
    speech_state = load_speech_state()
    
    # Ensure the UI is running
    if not ensure_ui_running():
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                "Failed to start the speech recognition system."
            )
        )
    
    # Speak the text
    try:
        speak_text(text)
    except Exception as e:
        logger.error(f"Error speaking text: {e}")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"Error speaking text: {str(e)}"
            )
        )
    
    # Start listening for response - using direct approach
    try:
        transcription = listen_for_speech()
        return transcription
    except Exception as e:
        logger.error(f"Error listening for response: {e}")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"Error listening for response: {str(e)}"
            )
        )

@mcp.resource(uri="mcp://speech/usage_guide")
def usage_guide() -> str:
    """
    Return the usage guide for the Speech MCP.
    """
    return """
    # Speech MCP Usage Guide
    
    This MCP extension provides voice interaction capabilities with a simplified interface.
    
    ## How to Use
    
    1. Start a conversation:
       ```
       user_input = start_conversation()
       ```
       This initializes the speech recognition system, launches the UI, and immediately starts listening for user input.
       Note: The first time you run this, it will download the Whisper model which may take a moment.
    
    2. Reply to the user and get their response:
       ```
       user_response = reply("Your response text here")
       ```
       This speaks your response and then listens for the user's reply.
    
    ## Typical Workflow
    
    1. Start the conversation to get the initial user input
    2. Process the transcribed speech
    3. Use the reply function to respond and get the next user input
    4. Repeat steps 2-3 for a continuous conversation
    
    ## Example Conversation Flow
    
    ```python
    # Start the conversation
    user_input = start_conversation()
    
    # Process the input and generate a response
    # ...
    
    # Reply to the user and get their response
    follow_up = reply("Here's my response to your question.")
    
    # Process the follow-up and reply again
    reply("I understand your follow-up question. Here's my answer.")
    ```
    
    ## Tips
    
    - For best results, use a quiet environment and speak clearly
    - The system automatically detects silence to know when you've finished speaking
    - The listening timeout is set to 10 minutes to allow for natural pauses in conversation
    """