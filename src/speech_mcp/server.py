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
    
    logger.debug("Checking UI process status")
    if speech_state["ui_active"] and speech_state["ui_process"] is not None:
        # Check if the process is still running
        if speech_state["ui_process"].poll() is None:
            logger.info("UI process is already running")
            return True
        else:
            exit_code = speech_state["ui_process"].poll()
            logger.warning(f"UI process has terminated unexpectedly with exit code: {exit_code}")
    else:
        logger.debug(f"UI active status: {speech_state['ui_active']}, UI process exists: {speech_state['ui_process'] is not None}")
    
    # UI is not running, start it
    try:
        # Kill any existing UI process
        if speech_state["ui_process"] is not None:
            try:
                logger.info(f"Terminating existing UI process (PID: {speech_state['ui_process'].pid})")
                speech_state["ui_process"].terminate()
                time.sleep(1)  # Give it time to terminate
                
                # Check if it's still running
                if speech_state["ui_process"].poll() is None:
                    logger.warning("Process didn't terminate, attempting to kill")
                    speech_state["ui_process"].kill()
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error terminating UI process: {e}")
        
        # Start a new UI process
        python_executable = sys.executable
        logger.debug(f"Using Python executable: {python_executable}")
        
        # Use the module import approach instead of direct file path
        logger.info(f"Starting UI process with Python module import")
        print(f"Starting UI process with Python module import")
        
        # Log environment variables that might affect audio
        audio_env_vars = {k: v for k, v in os.environ.items() if 'AUDIO' in k.upper() or 'PULSE' in k.upper() or 'ALSA' in k.upper()}
        if audio_env_vars:
            logger.debug(f"Audio-related environment variables: {json.dumps(audio_env_vars)}")
        
        # Start the process with stdout and stderr redirected
        cmd = [python_executable, "-m", "speech_mcp.ui"]
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        # Redirect stderr to stdout to simplify handling
        speech_state["ui_process"] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            bufsize=1  # Line buffered
        )
        
        logger.info(f"Started UI process with PID: {speech_state['ui_process'].pid}")
        
        # Start a thread to monitor the process output
        def log_output():
            try:
                buffer = ""
                for line in speech_state["ui_process"].stdout:
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    
                    # Accumulate lines for context
                    buffer = (buffer + "\n" + clean_line).strip()
                    if len(buffer.split("\n")) > 5:  # Keep only last 5 lines
                        buffer = "\n".join(buffer.split("\n")[-5:])
                    
                    # Check if this is an error message
                    if ("error" in clean_line.lower() or "exception" in clean_line.lower() or "traceback" in clean_line.lower()) and not any(info_marker in clean_line for info_marker in ["INFO", "DEBUG", "WARNING"]):
                        logger.error(f"UI Error: {clean_line}")
                        print(f"UI Error: {clean_line}")
                    else:
                        # Treat as info by default
                        logger.info(f"UI: {clean_line}")
                        print(f"UI: {clean_line}")
            except Exception as e:
                logger.error(f"Error in log output thread: {e}", exc_info=True)
        
        threading.Thread(target=log_output, daemon=True).start()
        
        speech_state["ui_active"] = True
        
        # Save the updated state
        save_speech_state(speech_state)
        
        # Give the UI time to start up
        startup_time = 2  # seconds
        logger.debug(f"Waiting {startup_time} seconds for UI to initialize")
        time.sleep(startup_time)
        
        # Check if the process is still running
        if speech_state["ui_process"].poll() is not None:
            exit_code = speech_state["ui_process"].poll()
            logger.error(f"UI process exited immediately with code {exit_code}")
            
            # Try to get any output from the process
            try:
                stdout_output = speech_state["ui_process"].stdout.read()
                if stdout_output:
                    logger.error(f"UI process output: {stdout_output}")
            except Exception as e:
                logger.error(f"Could not retrieve process output: {e}")
                
            return False
        
        logger.info("UI process started successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start UI: {e}", exc_info=True)
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
            logger.debug(f"Removing existing transcription file: {TRANSCRIPTION_FILE}")
            os.remove(TRANSCRIPTION_FILE)
        else:
            logger.debug(f"No existing transcription file found at: {TRANSCRIPTION_FILE}")
        
        max_timeout = 600  # 10 minutes maximum timeout
        start_time = time.time()
        
        # Wait for transcription file to appear
        logger.debug("Beginning wait loop for transcription file")
        while not os.path.exists(TRANSCRIPTION_FILE) and time.time() - start_time < max_timeout:
            time.sleep(0.5)
            # Print a message every 30 seconds to indicate we're still waiting
            if (time.time() - start_time) % 30 < 0.5:
                elapsed = int(time.time() - start_time)
                logger.debug(f"Still waiting for transcription file after {elapsed} seconds")
                print(f"Still waiting for speech input... ({elapsed} seconds elapsed)")
                
                # Check UI process status periodically
                if speech_state["ui_process"] is not None:
                    if speech_state["ui_process"].poll() is not None:
                        exit_code = speech_state["ui_process"].poll()
                        logger.error(f"UI process terminated while waiting for transcription with exit code: {exit_code}")
                        # Try to restart the UI
                        logger.info("Attempting to restart UI process")
                        if not ensure_ui_running():
                            raise Exception("Failed to restart UI process while waiting for transcription")
        
        if not os.path.exists(TRANSCRIPTION_FILE):
            logger.error(f"Timeout waiting for transcription file after {int(time.time() - start_time)} seconds")
            speech_state["listening"] = False
            save_speech_state(speech_state)
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    "Timeout waiting for speech transcription."
                )
            )
        
        # Read the transcription
        logger.debug(f"Transcription file found at: {TRANSCRIPTION_FILE}, reading content")
        try:
            with open(TRANSCRIPTION_FILE, 'r') as f:
                transcription = f.read().strip()
            
            # Check if transcription is empty
            if not transcription:
                logger.warning("Transcription file exists but is empty")
                transcription = ""
                
        except Exception as e:
            logger.error(f"Error reading transcription file: {e}", exc_info=True)
            raise Exception(f"Error reading transcription file: {str(e)}")
        
        logger.info(f"Received transcription: {transcription}")
        print(f"Transcription received: \"{transcription}\"")
        
        # Delete the file to prepare for the next transcription
        try:
            logger.debug(f"Removing transcription file: {TRANSCRIPTION_FILE}")
            os.remove(TRANSCRIPTION_FILE)
        except Exception as e:
            logger.warning(f"Error removing transcription file: {e}")
        
        # Update state
        speech_state["listening"] = False
        speech_state["last_transcript"] = transcription
        save_speech_state(speech_state)
        
        return transcription
    except Exception as e:
        # Update state on error
        speech_state["listening"] = False
        save_speech_state(speech_state)
        
        logger.error(f"Error during speech recognition: {e}", exc_info=True)
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
        logger.warning("Empty text provided to speak_text function")
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
        logger.info(f"Speaking text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"\nSpeaking: \"{text}\"")
        
        # Check if UI process is running before attempting to speak
        if speech_state["ui_process"] is None or speech_state["ui_process"].poll() is not None:
            logger.warning("UI process is not running, attempting to restart before speaking")
            if not ensure_ui_running():
                raise Exception("Failed to start UI process for text-to-speech")
        
        # Write the text to a file for the UI to process
        logger.debug(f"Writing text to response file: {RESPONSE_FILE}")
        try:
            with open(RESPONSE_FILE, 'w') as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Error writing to response file: {e}", exc_info=True)
            raise Exception(f"Error writing to response file: {str(e)}")
        
        # Give the UI time to process and "speak" the text
        # We'll estimate the speaking time based on text length
        # Average speaking rate is about 150 words per minute or 2.5 words per second
        # Assuming an average of 5 characters per word
        words = len(text) / 5
        speaking_time = words / 2.5  # Time in seconds
        
        # Add a small buffer
        speaking_time += 1.0
        
        logger.debug(f"Estimated speaking time: {speaking_time:.2f} seconds for {words:.1f} words")
        
        # Wait for the estimated speaking time
        start_wait = time.time()
        time.sleep(speaking_time)
        actual_wait = time.time() - start_wait
        logger.debug(f"Waited {actual_wait:.2f} seconds for speech to complete")
        
        # Check if response file still exists (UI might delete it when done)
        if os.path.exists(RESPONSE_FILE):
            logger.debug("Response file still exists after waiting, attempting to remove")
            try:
                os.remove(RESPONSE_FILE)
                logger.debug("Response file removed successfully")
            except Exception as e:
                logger.warning(f"Error removing response file: {e}")
        
        # Update state
        speech_state["speaking"] = False
        save_speech_state(speech_state)
        
        # Check UI process status after speaking
        if speech_state["ui_process"] is not None and speech_state["ui_process"].poll() is not None:
            exit_code = speech_state["ui_process"].poll()
            logger.warning(f"UI process terminated during speech with exit code: {exit_code}")
        
        logger.info("Done speaking")
        print("Done speaking.")
        return f"Spoke: {text}"
    except Exception as e:
        # Update state on error
        speech_state["speaking"] = False
        save_speech_state(speech_state)
        
        logger.error(f"Error during text-to-speech: {e}", exc_info=True)
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
        logger.info("Starting background listen task")
        
        # Ensure the UI is running
        if not ensure_ui_running():
            logger.error("Failed to start the speech recognition system in background task")
            callback(f"Error: Failed to start the speech recognition system.")
            return
        
        # Set listening state
        speech_state["listening"] = True
        save_speech_state(speech_state)
        
        logger.info("Starting to listen for speech input in background task...")
        print("\nListening for speech input... Speak now.")
        
        # Delete any existing transcription file to avoid using old data
        if os.path.exists(TRANSCRIPTION_FILE):
            logger.debug(f"Removing existing transcription file: {TRANSCRIPTION_FILE}")
            os.remove(TRANSCRIPTION_FILE)
        else:
            logger.debug(f"No existing transcription file found at: {TRANSCRIPTION_FILE}")
        
        # Start a separate thread to monitor for the transcription file
        def monitor_transcription():
            try:
                start_time = time.time()
                max_wait_time = 600  # 10 minutes
                
                logger.debug("Beginning wait loop for transcription file in background task")
                while not os.path.exists(TRANSCRIPTION_FILE) and time.time() - start_time < max_wait_time:
                    time.sleep(0.5)
                    # Print a message every 30 seconds to indicate we're still waiting
                    if (time.time() - start_time) % 30 < 0.5:
                        elapsed = int(time.time() - start_time)
                        logger.debug(f"Background task: Still waiting for transcription file after {elapsed} seconds")
                        print(f"Still waiting for speech input... ({elapsed} seconds elapsed)")
                        
                        # Check UI process status periodically
                        if speech_state["ui_process"] is not None:
                            if speech_state["ui_process"].poll() is not None:
                                exit_code = speech_state["ui_process"].poll()
                                logger.error(f"UI process terminated while waiting for transcription with exit code: {exit_code}")
                                # Try to restart the UI
                                logger.info("Attempting to restart UI process from background task")
                                if not ensure_ui_running():
                                    logger.error("Failed to restart UI process in background task")
                                    speech_state["listening"] = False
                                    save_speech_state(speech_state)
                                    callback("Error: Failed to restart UI process while waiting for transcription.")
                                    return
                
                if not os.path.exists(TRANSCRIPTION_FILE):
                    logger.error(f"Timeout waiting for transcription file in background task after {int(time.time() - start_time)} seconds")
                    speech_state["listening"] = False
                    save_speech_state(speech_state)
                    callback("Error: Timeout waiting for speech transcription.")
                    return
                
                # Read the transcription
                logger.debug(f"Transcription file found at: {TRANSCRIPTION_FILE}, reading content")
                try:
                    with open(TRANSCRIPTION_FILE, 'r') as f:
                        transcription = f.read().strip()
                    
                    # Check if transcription is empty
                    if not transcription:
                        logger.warning("Transcription file exists but is empty in background task")
                        transcription = ""
                        
                except Exception as e:
                    logger.error(f"Error reading transcription file in background task: {e}", exc_info=True)
                    speech_state["listening"] = False
                    save_speech_state(speech_state)
                    callback(f"Error: {str(e)}")
                    return
                
                logger.info(f"Received transcription in background task: {transcription}")
                print(f"Transcription received: \"{transcription}\"")
                
                # Delete the file to prepare for the next transcription
                try:
                    logger.debug(f"Removing transcription file: {TRANSCRIPTION_FILE}")
                    os.remove(TRANSCRIPTION_FILE)
                except Exception as e:
                    logger.warning(f"Error removing transcription file in background task: {e}")
                
                # Update state
                speech_state["listening"] = False
                speech_state["last_transcript"] = transcription
                save_speech_state(speech_state)
                
                # Call the callback with the transcription
                logger.debug("Calling callback with transcription")
                callback(transcription)
            except Exception as e:
                # Update state on error
                speech_state["listening"] = False
                save_speech_state(speech_state)
                
                logger.error(f"Error during speech recognition in background task: {e}", exc_info=True)
                callback(f"Error: {str(e)}")
        
        # Start the monitoring thread
        logger.debug("Starting background thread to monitor for transcription")
        threading.Thread(target=monitor_transcription, daemon=True).start()
        
    except Exception as e:
        logger.error(f"Error in listen_task: {e}", exc_info=True)
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
    
    logger.info("Starting new conversation with start_conversation()")
    
    # Reload speech state to ensure we have the latest
    speech_state = load_speech_state()
    
    # Start the UI
    logger.debug("Ensuring UI is running for start_conversation()")
    if not ensure_ui_running():
        logger.error("Failed to start the speech recognition system in start_conversation()")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                "Failed to start the speech recognition system."
            )
        )
    
    # Give the UI a moment to fully initialize
    logger.debug("Waiting for UI to fully initialize")
    time.sleep(2)
    
    # Start listening - using direct approach for this initial call
    try:
        logger.info("Beginning to listen for speech in start_conversation()")
        transcription = listen_for_speech()
        logger.info(f"start_conversation() completed successfully with transcription: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Error starting conversation: {e}", exc_info=True)
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
    
    logger.info(f"reply() called with text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Reload speech state to ensure we have the latest
    speech_state = load_speech_state()
    
    # Ensure the UI is running
    logger.debug("Ensuring UI is running for reply()")
    if not ensure_ui_running():
        logger.error("Failed to start the speech recognition system in reply()")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                "Failed to start the speech recognition system."
            )
        )
    
    # Speak the text
    try:
        logger.info("Speaking text in reply()")
        speak_text(text)
    except Exception as e:
        logger.error(f"Error speaking text in reply(): {e}", exc_info=True)
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"Error speaking text: {str(e)}"
            )
        )
    
    # Start listening for response - using direct approach
    try:
        logger.info("Beginning to listen for response in reply()")
        transcription = listen_for_speech()
        logger.info(f"reply() completed successfully with transcription: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Error listening for response in reply(): {e}", exc_info=True)
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