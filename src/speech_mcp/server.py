import subprocess
import sys
import os
import json
import logging
import time
import threading
import asyncio
import queue
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
    print("DEBUG: Checking UI process status")
    if speech_state["ui_active"] and speech_state["ui_process"] is not None:
        # Check if the process is still running
        if speech_state["ui_process"].poll() is None:
            logger.info("UI process is already running")
            print("INFO: UI process is already running")
            return True
        else:
            exit_code = speech_state["ui_process"].poll()
            logger.warning(f"UI process has terminated unexpectedly with exit code: {exit_code}")
            print(f"WARNING: UI process has terminated unexpectedly with exit code: {exit_code}")
    else:
        logger.debug(f"UI active status: {speech_state['ui_active']}, UI process exists: {speech_state['ui_process'] is not None}")
        print(f"DEBUG: UI active status: {speech_state['ui_active']}, UI process exists: {speech_state['ui_process'] is not None}")
    
    # UI is not running, start it
    try:
        # Kill any existing UI process
        if speech_state["ui_process"] is not None:
            try:
                logger.info(f"Terminating existing UI process (PID: {speech_state['ui_process'].pid})")
                print(f"INFO: Terminating existing UI process (PID: {speech_state['ui_process'].pid})")
                speech_state["ui_process"].terminate()
                time.sleep(1)  # Give it time to terminate
                
                # Check if it's still running
                if speech_state["ui_process"].poll() is None:
                    logger.warning("Process didn't terminate, attempting to kill")
                    print("WARNING: Process didn't terminate, attempting to kill")
                    speech_state["ui_process"].kill()
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error terminating UI process: {e}")
                print(f"ERROR: Error terminating UI process: {e}")
        
        # Start a new UI process
        python_executable = sys.executable
        logger.debug(f"Using Python executable: {python_executable}")
        print(f"DEBUG: Using Python executable: {python_executable}")
        
        # Use the module import approach instead of direct file path
        logger.info(f"Starting UI process with Python module import")
        print(f"INFO: Starting UI process with Python module import")
        
        # Log environment variables that might affect audio
        audio_env_vars = {k: v for k, v in os.environ.items() if 'AUDIO' in k.upper() or 'PULSE' in k.upper() or 'ALSA' in k.upper()}
        if audio_env_vars:
            logger.debug(f"Audio-related environment variables: {json.dumps(audio_env_vars)}")
            print(f"DEBUG: Audio-related environment variables: {json.dumps(audio_env_vars)}")
        
        # Start the process with stdout and stderr redirected
        cmd = [python_executable, "-m", "speech_mcp.ui"]
        logger.debug(f"Running command: {' '.join(cmd)}")
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        
        # Check if the module is importable before starting process
        try:
            import importlib
            importlib.import_module("speech_mcp.ui")
            logger.debug("Successfully imported speech_mcp.ui module")
            print("DEBUG: Successfully imported speech_mcp.ui module")
        except ImportError as e:
            logger.error(f"Cannot import speech_mcp.ui module: {e}")
            print(f"ERROR: Cannot import speech_mcp.ui module: {e}")
            # Try to find the module location
            try:
                import speech_mcp
                logger.debug(f"speech_mcp module location: {speech_mcp.__file__}")
                print(f"DEBUG: speech_mcp module location: {speech_mcp.__file__}")
            except Exception as e2:
                logger.error(f"Error finding speech_mcp module: {e2}")
                print(f"ERROR: Error finding speech_mcp module: {e2}")
        
        # Redirect stderr to stdout to simplify handling
        speech_state["ui_process"] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            bufsize=1  # Line buffered
        )
        
        logger.info(f"Started UI process with PID: {speech_state['ui_process'].pid}")
        print(f"INFO: Started UI process with PID: {speech_state['ui_process'].pid}")
        
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
                print(f"ERROR: Error in log output thread: {e}")
        
        threading.Thread(target=log_output, daemon=True).start()
        
        speech_state["ui_active"] = True
        
        # Save the updated state
        save_speech_state(speech_state)
        
        # Give the UI time to start up
        startup_time = 2  # seconds
        logger.debug(f"Waiting {startup_time} seconds for UI to initialize")
        print(f"DEBUG: Waiting {startup_time} seconds for UI to initialize")
        time.sleep(startup_time)
        
        # Check if the process is still running
        if speech_state["ui_process"].poll() is not None:
            exit_code = speech_state["ui_process"].poll()
            logger.error(f"UI process exited immediately with code {exit_code}")
            print(f"ERROR: UI process exited immediately with code {exit_code}")
            
            # Try to get any output from the process
            try:
                stdout_output = speech_state["ui_process"].stdout.read()
                if stdout_output:
                    logger.error(f"UI process output: {stdout_output}")
                    print(f"ERROR: UI process output: {stdout_output}")
            except Exception as e:
                logger.error(f"Could not retrieve process output: {e}")
                print(f"ERROR: Could not retrieve process output: {e}")
                
            return False
        
        logger.info("UI process started successfully")
        print("INFO: UI process started successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start UI: {e}", exc_info=True)
        print(f"ERROR: Failed to start UI: {e}")
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
        
        # Create a test transcription file to check if file system permissions are working
        try:
            logger.debug("Testing file system permissions with test transcription file")
            print("Testing file system permissions...")
            with open(TRANSCRIPTION_FILE, 'w') as f:
                f.write("TEST_TRANSCRIPTION_FILE_PERMISSION_CHECK")
            
            # Check if the file was created
            if os.path.exists(TRANSCRIPTION_FILE):
                logger.debug("Test transcription file created successfully")
                print("File system permissions OK")
                # Remove the test file
                os.remove(TRANSCRIPTION_FILE)
            else:
                logger.error("Failed to create test transcription file")
                print("ERROR: Failed to create test transcription file")
                raise Exception("Failed to create test transcription file - check file system permissions")
        except Exception as e:
            logger.error(f"Error testing file system permissions: {e}")
            print(f"ERROR: File system permission test failed: {e}")
            raise Exception(f"File system permission test failed: {e}")
        
        # Check if UI is actually running
        if speech_state["ui_process"] is None or speech_state["ui_process"].poll() is not None:
            logger.error("UI process is not running at the start of listen_for_speech")
            print("ERROR: UI process is not running, attempting to restart")
            
            # Try to restart the UI
            if not ensure_ui_running():
                logger.error("Failed to start UI process for speech recognition")
                print("ERROR: Failed to start UI process for speech recognition")
                speech_state["listening"] = False
                save_speech_state(speech_state)
                raise Exception("Failed to start UI process for speech recognition")
        
        # Create a direct transcription file as a fallback mechanism
        # This helps diagnose if the UI process is failing to create the file
        logger.debug("Creating fallback transcription mechanism")
        print("Creating fallback transcription mechanism...")
        
        # Start a thread to create a fallback transcription after a delay
        def create_fallback_transcription():
            try:
                # Wait for 10 seconds before creating fallback
                time.sleep(10)
                
                # Check if transcription file already exists (created by UI)
                if os.path.exists(TRANSCRIPTION_FILE):
                    logger.debug("Transcription file already exists, no need for fallback")
                    return
                
                # If we're still listening and no file exists, create a fallback
                if speech_state["listening"]:
                    logger.warning("Creating fallback transcription file after 10s")
                    print("WARNING: Creating fallback transcription file")
                    with open(TRANSCRIPTION_FILE, 'w') as f:
                        f.write("FALLBACK_TRANSCRIPTION: UI process may not be functioning correctly")
            except Exception as e:
                logger.error(f"Error in fallback transcription thread: {e}")
        
        # Start the fallback thread
        fallback_thread = threading.Thread(target=create_fallback_transcription)
        fallback_thread.daemon = True
        fallback_thread.start()
        
        # Shorter timeout for better responsiveness
        max_timeout = 30  # 30 seconds maximum timeout (reduced from 600s)
        start_time = time.time()
        
        # Wait for transcription file to appear
        logger.debug("Beginning wait loop for transcription file")
        while not os.path.exists(TRANSCRIPTION_FILE) and time.time() - start_time < max_timeout:
            time.sleep(0.5)
            # Print a message every 5 seconds to indicate we're still waiting
            if (time.time() - start_time) % 5 < 0.5:
                elapsed = int(time.time() - start_time)
                logger.debug(f"Still waiting for transcription file after {elapsed} seconds")
                print(f"Still waiting for speech input... ({elapsed} seconds elapsed)")
                
                # Check UI process status more frequently
                if speech_state["ui_process"] is not None:
                    if speech_state["ui_process"].poll() is not None:
                        exit_code = speech_state["ui_process"].poll()
                        logger.error(f"UI process terminated while waiting for transcription with exit code: {exit_code}")
                        print(f"ERROR: UI process terminated with exit code: {exit_code}")
                        
                        # Try to restart the UI
                        logger.info("Attempting to restart UI process")
                        print("Attempting to restart UI process...")
                        if not ensure_ui_running():
                            logger.error("Failed to restart UI process")
                            print("ERROR: Failed to restart UI process")
                            speech_state["listening"] = False
                            save_speech_state(speech_state)
                            raise Exception("Failed to restart UI process while waiting for transcription")
        
        if not os.path.exists(TRANSCRIPTION_FILE):
            logger.error(f"Timeout waiting for transcription file after {int(time.time() - start_time)} seconds")
            print(f"ERROR: Timeout waiting for transcription file after {int(time.time() - start_time)} seconds")
            
            # Create an emergency transcription file
            try:
                logger.warning("Creating emergency transcription file due to timeout")
                print("Creating emergency transcription file due to timeout")
                with open(TRANSCRIPTION_FILE, 'w') as f:
                    f.write("ERROR: Timeout waiting for speech transcription. The UI process may not be functioning correctly.")
            except Exception as e:
                logger.error(f"Error creating emergency transcription file: {e}")
                print(f"ERROR: Failed to create emergency transcription file: {e}")
            
            # Wait a moment for the file to be created
            time.sleep(1)
            
            # If still no file, then give up
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
        logger.debug(f"Transcription file found at: {TRANSCRIPTION_FILE}, reading content")
        try:
            with open(TRANSCRIPTION_FILE, 'r') as f:
                transcription = f.read().strip()
            
            # Check if transcription is empty
            if not transcription:
                logger.warning("Transcription file exists but is empty")
                print("WARNING: Transcription file exists but is empty")
                transcription = "ERROR: Empty transcription received"
                
        except Exception as e:
            logger.error(f"Error reading transcription file: {e}", exc_info=True)
            print(f"ERROR: Failed to read transcription file: {e}")
            raise Exception(f"Error reading transcription file: {str(e)}")
        
        logger.info(f"Received transcription: {transcription}")
        print(f"Transcription received: \"{transcription}\"")
        
        # Delete the file to prepare for the next transcription
        try:
            logger.debug(f"Removing transcription file: {TRANSCRIPTION_FILE}")
            os.remove(TRANSCRIPTION_FILE)
        except Exception as e:
            logger.warning(f"Error removing transcription file: {e}")
            print(f"WARNING: Failed to remove transcription file: {e}")
        
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
        print(f"ERROR: Speech recognition failed: {e}")
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
    print("Starting new conversation with start_conversation()")
    
    # Force reset the speech state to avoid any stuck states
    speech_state = DEFAULT_SPEECH_STATE.copy()
    save_speech_state(speech_state)
    logger.info("Reset speech state to defaults")
    print("Reset speech state to defaults")
    
    # Check if there's an existing UI process
    try:
        # Find any existing UI processes
        import psutil
        ui_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and len(proc.info['cmdline']) > 1:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'speech_mcp.ui' in cmdline:
                        ui_processes.append(proc)
                        logger.info(f"Found existing UI process: PID {proc.pid}, {cmdline}")
                        print(f"Found existing UI process: PID {proc.pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Kill any existing UI processes
        for proc in ui_processes:
            try:
                logger.info(f"Terminating existing UI process: PID {proc.pid}")
                print(f"Terminating existing UI process: PID {proc.pid}")
                proc.terminate()
                gone, alive = psutil.wait_procs([proc], timeout=3)
                if alive:
                    logger.warning(f"Process {proc.pid} still alive after terminate, killing")
                    print(f"Process {proc.pid} still alive after terminate, killing")
                    for p in alive:
                        p.kill()
            except Exception as e:
                logger.error(f"Error terminating process {proc.pid}: {e}")
                print(f"Error terminating process {proc.pid}: {e}")
    except ImportError:
        logger.warning("psutil not available, skipping existing process check")
        print("psutil not available, skipping existing process check")
    except Exception as e:
        logger.error(f"Error checking for existing processes: {e}")
        print(f"Error checking for existing processes: {e}")
    
    # Start the UI
    logger.debug("Ensuring UI is running for start_conversation()")
    print("DEBUG: Ensuring UI is running for start_conversation()")
    if not ensure_ui_running():
        logger.error("Failed to start the speech recognition system in start_conversation()")
        print("ERROR: Failed to start the speech recognition system in start_conversation()")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                "Failed to start the speech recognition system."
            )
        )
    
    # Give the UI a moment to fully initialize
    logger.debug("Waiting for UI to fully initialize")
    print("DEBUG: Waiting for UI to fully initialize")
    time.sleep(3)  # Increased from 2s to 3s
    
    # Check if UI process is still running
    if speech_state["ui_process"] is None or speech_state["ui_process"].poll() is not None:
        exit_code = speech_state["ui_process"].poll() if speech_state["ui_process"] else None
        logger.error(f"UI process is not running after initialization, exit code: {exit_code}")
        print(f"ERROR: UI process is not running after initialization, exit code: {exit_code}")
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"UI process failed to start or terminated immediately with exit code: {exit_code}"
            )
        )
    
    # Start listening - using direct approach for this initial call
    try:
        logger.info("Beginning to listen for speech in start_conversation()")
        print("INFO: Beginning to listen for speech in start_conversation()")
        
        # Use a queue to get the result from the thread
        result_queue = queue.Queue()
        
        def listen_and_queue():
            try:
                result = listen_for_speech()
                result_queue.put(result)
            except Exception as e:
                logger.error(f"Error in listen_and_queue: {e}")
                print(f"ERROR: Error in listen_and_queue: {e}")
                result_queue.put(f"ERROR: {str(e)}")
        
        # Start the thread
        listen_thread = threading.Thread(target=listen_and_queue)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Wait for the result with a timeout
        timeout = 30  # 30 seconds timeout
        try:
            logger.debug(f"Waiting for transcription with {timeout}s timeout")
            print(f"DEBUG: Waiting for transcription with {timeout}s timeout")
            transcription = result_queue.get(timeout=timeout)
            logger.info(f"start_conversation() completed successfully with transcription: {transcription}")
            print(f"INFO: start_conversation() completed successfully with transcription: {transcription}")
            return transcription
        except queue.Empty:
            logger.error(f"Timeout waiting for transcription after {timeout} seconds")
            print(f"ERROR: Timeout waiting for transcription after {timeout} seconds")
            
            # Update state to stop listening
            speech_state["listening"] = False
            save_speech_state(speech_state)
            
            # Create an emergency transcription
            emergency_message = f"ERROR: Timeout waiting for speech transcription after {timeout} seconds."
            logger.warning(f"Returning emergency message: {emergency_message}")
            print(f"Returning emergency message: {emergency_message}")
            return emergency_message
    except Exception as e:
        logger.error(f"Error starting conversation: {e}", exc_info=True)
        print(f"ERROR: Error starting conversation: {e}")
        
        # Update state to stop listening
        speech_state["listening"] = False
        save_speech_state(speech_state)
        
        # Return an error message instead of raising an exception
        error_message = f"ERROR: Failed to start conversation: {str(e)}"
        logger.warning(f"Returning error message: {error_message}")
        print(f"Returning error message: {error_message}")
        return error_message

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
    print(f"reply() called with text: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # Reset listening and speaking states to ensure we're in a clean state
    speech_state["listening"] = False
    speech_state["speaking"] = False
    save_speech_state(speech_state)
    
    # Ensure the UI is running
    logger.debug("Ensuring UI is running for reply()")
    print("DEBUG: Ensuring UI is running for reply()")
    if not ensure_ui_running():
        logger.error("Failed to start the speech recognition system in reply()")
        print("ERROR: Failed to start the speech recognition system in reply()")
        return "ERROR: Failed to start the speech recognition system."
    
    # Check if UI process is still running
    if speech_state["ui_process"] is None or speech_state["ui_process"].poll() is not None:
        exit_code = speech_state["ui_process"].poll() if speech_state["ui_process"] else None
        logger.error(f"UI process is not running, exit code: {exit_code}")
        print(f"ERROR: UI process is not running, exit code: {exit_code}")
        return f"ERROR: UI process is not running, exit code: {exit_code}"
    
    # Speak the text
    try:
        logger.info("Speaking text in reply()")
        print("INFO: Speaking text in reply()")
        speak_text(text)
    except Exception as e:
        logger.error(f"Error speaking text in reply(): {e}", exc_info=True)
        print(f"ERROR: Error speaking text in reply(): {e}")
        return f"ERROR: Failed to speak text: {str(e)}"
    
    # Start listening for response - using direct approach
    try:
        logger.info("Beginning to listen for response in reply()")
        print("INFO: Beginning to listen for response in reply()")
        
        # Use a queue to get the result from the thread
        result_queue = queue.Queue()
        
        def listen_and_queue():
            try:
                result = listen_for_speech()
                result_queue.put(result)
            except Exception as e:
                logger.error(f"Error in listen_and_queue: {e}")
                print(f"ERROR: Error in listen_and_queue: {e}")
                result_queue.put(f"ERROR: {str(e)}")
        
        # Start the thread
        listen_thread = threading.Thread(target=listen_and_queue)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Wait for the result with a timeout
        timeout = 30  # 30 seconds timeout
        try:
            logger.debug(f"Waiting for transcription with {timeout}s timeout")
            print(f"DEBUG: Waiting for transcription with {timeout}s timeout")
            transcription = result_queue.get(timeout=timeout)
            logger.info(f"reply() completed successfully with transcription: {transcription}")
            print(f"INFO: reply() completed successfully with transcription: {transcription}")
            return transcription
        except queue.Empty:
            logger.error(f"Timeout waiting for transcription after {timeout} seconds")
            print(f"ERROR: Timeout waiting for transcription after {timeout} seconds")
            
            # Update state to stop listening
            speech_state["listening"] = False
            save_speech_state(speech_state)
            
            # Create an emergency transcription
            emergency_message = f"ERROR: Timeout waiting for speech transcription after {timeout} seconds."
            logger.warning(f"Returning emergency message: {emergency_message}")
            print(f"Returning emergency message: {emergency_message}")
            return emergency_message
    except Exception as e:
        logger.error(f"Error listening for response in reply(): {e}", exc_info=True)
        print(f"ERROR: Error listening for response in reply(): {e}")
        
        # Update state to stop listening
        speech_state["listening"] = False
        save_speech_state(speech_state)
        
        # Return an error message instead of raising an exception
        error_message = f"ERROR: Failed to listen for response: {str(e)}"
        logger.warning(f"Returning error message: {error_message}")
        print(f"Returning error message: {error_message}")
        return error_message

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
       Note: The first time you run this, it will download the faster-whisper model which may take a moment.
    
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