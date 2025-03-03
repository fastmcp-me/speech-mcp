# Speech MCP

A Goose MCP extension for voice interaction with audio visualization.

## Overview

Speech MCP provides a voice interface for Goose, allowing users to interact through speech rather than text. It includes:

- Real-time audio processing for speech recognition
- Local speech-to-text using faster-whisper (a faster implementation of OpenAI's Whisper model)
- Text-to-speech capabilities 
- Simple command-line interface for voice interaction

## Features

- **Voice Input**: Capture and transcribe user speech using faster-whisper
- **Voice Output**: Convert agent responses to speech
- **Continuous Conversation**: Automatically listen for user input after agent responses
- **Silence Detection**: Automatically stops recording when the user stops speaking
- **Robust Error Handling**: Graceful recovery from common failure modes

## Installation

### Option 1: Quick Install (One-Click)

Click the link below if you have Goose installed:

`goose://extension?cmd=uvx&arg=speech-mcp&id=speech_mcp&name=Speech%20Interface&description=Voice%20interaction%20with%20audio%20visualization%20for%20Goose`

### Option 2: Using Goose CLI (recommended)

Start Goose with your extension enabled:

```bash
# If you installed via PyPI
goose session --with-extension "speech-mcp"

# Or if you want to use a local development version
goose session --with-extension "python -m speech_mcp"
```

### Option 3: Manual setup in Goose

1. Run `goose configure`
2. Select "Add Extension" from the menu
3. Choose "Command-line Extension"
4. Enter a name (e.g., "Speech Interface")
5. For the command, enter: `speech-mcp`
6. Follow the prompts to complete the setup

### Option 4: Manual Installation

1. Clone this repository
2. Install dependencies:
   ```
   uv pip install -e .
   ```

## Dependencies

- Python 3.10+
- PyAudio (for audio capture)
- faster-whisper (for speech-to-text)
- NumPy (for audio processing)
- Pydub (for audio processing)
- pyttsx3 (for text-to-speech)
- psutil (for process management)

## Usage

To use this MCP with Goose, you can:

1. Start a conversation:
   ```python
   user_input = start_conversation()
   ```

2. Reply to the user and get their response:
   ```python
   user_response = reply("Your response text here")
   ```

## Typical Workflow

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

## Troubleshooting

If you encounter issues with the extension freezing or not responding:

1. **Check the logs**: Look at the log files in `src/speech_mcp/` for detailed error messages.
2. **Reset the state**: If the extension seems stuck, try deleting `src/speech_mcp/speech_state.json` or setting all states to `false`.
3. **Use the direct command**: Instead of `uv run speech-mcp`, use the installed package with `speech-mcp` directly.
4. **Check audio devices**: Ensure your microphone is properly configured and accessible to Python.
5. **Verify dependencies**: Make sure all required dependencies are installed correctly.

## Recent Fixes

- **Improved error handling**: Better recovery from common failure modes
- **Timeout management**: Reduced timeouts and added fallback mechanisms
- **Process management**: Better handling of UI process startup and termination
- **State consistency**: Added state reset mechanisms to avoid getting stuck
- **Fallback transcription**: Added emergency transcription when UI process fails
- **Debugging output**: Enhanced logging and console output for troubleshooting

## Technical Details

### Speech-to-Text

The MCP uses faster-whisper for speech recognition:
- Uses the "base" model for a good balance of accuracy and speed
- Processes audio locally without sending data to external services
- Automatically detects when the user has finished speaking
- Provides improved performance over the original Whisper implementation

## License

[MIT License](LICENSE)