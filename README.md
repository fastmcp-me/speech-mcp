# Speech MCP

A Goose MCP extension for voice interaction with audio visualization.

## Overview

Speech MCP provides a voice interface for Goose, allowing users to interact through speech rather than text. It includes:

- Real-time audio processing for speech recognition
- Local speech-to-text using OpenAI's Whisper model
- Text-to-speech capabilities 
- Simple command-line interface for voice interaction

## Features

- **Voice Input**: Capture and transcribe user speech using Whisper
- **Voice Output**: Convert agent responses to speech
- **Continuous Conversation**: Automatically listen for user input after agent responses
- **Silence Detection**: Automatically stops recording when the user stops speaking

## Installation

### Option 1: Quick Install (One-Click)

Click the link below if you have Goose installed:

`goose://extension?cmd=uvx&arg=speech-mcp&id=speech_mcp&name=Speech%20Interface&description=Voice%20interaction%20with%20audio%20visualization%20for%20Goose`

### Option 2: Using Goose CLI (recommended)

Start Goose with your extension enabled:

```bash
# If you installed via PyPI
goose session --with-extension "uvx speech-mcp"

# Or if you want to use a local development version
goose session --with-extension "python -m speech_mcp"
```

### Option 3: Manual setup in Goose

1. Run `goose configure`
2. Select "Add Extension" from the menu
3. Choose "Command-line Extension"
4. Enter a name (e.g., "Speech Interface")
5. For the command, enter: `uvx speech-mcp`
6. Follow the prompts to complete the setup

### Option 4: Manual Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -e .
   ```

## Dependencies

- Python 3.10+
- PyAudio (for audio capture)
- OpenAI Whisper (for speech-to-text)
- NumPy (for audio processing)
- Pydub (for audio processing)

## Usage

To use this MCP with Goose, you can:

1. Start the voice mode:
   ```
   start_voice_mode()
   ```

2. Listen for user input:
   ```
   transcript = listen()
   ```

3. Respond with speech:
   ```
   speak("Your response text")
   ```

4. Get the current state:
   ```
   get_speech_state()
   ```

## Typical Workflow

```python
# Start the voice interface
start_voice_mode()

# Listen for user input
transcript = listen()

# Process the transcript and generate a response
# ...

# Speak the response
speak("Here is my response")

# Automatically listen again
transcript = listen()
```

## Technical Details

### Speech-to-Text

The MCP uses OpenAI's Whisper model for speech recognition:
- Uses the "base" model for a good balance of accuracy and speed
- Processes audio locally without sending data to external services
- Automatically detects when the user has finished speaking

## License

[MIT License](LICENSE)