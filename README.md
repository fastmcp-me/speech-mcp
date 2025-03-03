# Speech MCP

A Goose MCP extension for voice interaction with audio visualization.

## Overview

Speech MCP provides a voice interface for Goose, allowing users to interact through speech rather than text. It includes:

- Real-time audio visualization for both user and agent speech
- Local speech-to-text using Whisper
- Local text-to-speech capabilities
- A simple but effective UI with circular audio visualizers

## Features

- **Voice Input**: Capture and transcribe user speech
- **Voice Output**: Convert agent responses to speech
- **Audio Visualization**: Visual feedback for both user and agent audio
- **Continuous Conversation**: Automatically listen for user input after agent responses

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -e .
   ```

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

## Requirements

- Python 3.10+
- PyAudio
- OpenAI Whisper
- NumPy
- Matplotlib
- Tkinter (usually included with Python)

## License

[MIT License](LICENSE)