# Speech MCP Changes

## Fixes for Current Issues

### 1. Listen Function Timeout Issue
- Increased the timeout from 60 seconds to 10 minutes
- Added progress messages during listening to show that the system is still waiting
- Reduced silence detection threshold from 0.01 to 0.005 to make it less sensitive
- Increased maximum silence duration from 1.5 to 2.0 seconds before ending recording

### 2. Speak Function Not Producing Audio
- Added pyttsx3 as a dependency for text-to-speech functionality
- Implemented actual speech synthesis using pyttsx3 instead of just simulating speech with delays
- Added fallback to simulation if text-to-speech fails

### 3. UI Not Opening
- Enhanced error handling and logging in the UI startup process
- Added detailed logging of UI process output to help diagnose issues
- Increased the startup wait time from 1 to 2 seconds
- Added checks to verify if the UI process is still running after startup
- Improved process termination handling for existing UI processes
- **Fixed path issue:** Changed UI process startup to use Python module import (`python -m speech_mcp.ui`) instead of direct file path
- **Fixed log output:** Improved log output formatting to clean up error messages
- **Added minimal UI:** Implemented a simple status window that shows when the system is listening or speaking

### 4. Simplified API
- Reduced the API to just two main functions:
  - `start_conversation()`: Launches the UI and immediately starts listening
  - `reply(text)`: Speaks the provided text and then listens for a response
- Removed separate `start_voice_mode()`, `listen()`, and `speak()` functions
- Simplified the workflow for voice conversations

## How to Test

1. Make sure all dependencies are installed:
   ```
   source .venv/bin/activate
   uv pip install -e .
   ```

2. Run the speech-mcp server:
   ```
   speech-mcp
   ```

3. Start a conversation:
   ```
   user_input = start_conversation()
   ```

4. Reply to the user and get their response:
   ```
   user_response = reply("This is a test of the speech synthesis system")
   ```

## Troubleshooting

If you encounter issues:

1. Check the log files:
   - `/Users/mnovich/Development/speech-mcp/src/speech_mcp/speech-mcp.log`
   - `/Users/mnovich/Development/speech-mcp/src/speech_mcp/speech-mcp-server.log`

2. Make sure the UI process is running:
   ```
   ps aux | grep speech_mcp
   ```

3. If the UI is not running, check for error messages in the logs and try restarting the server.