# Speech MCP

A Goose MCP extension for voice interaction with audio visualization.

## Overview

Speech MCP provides a voice interface for Goose, allowing users to interact through speech rather than text. It includes:

- Real-time audio visualization for both user and agent speech
- Local speech-to-text using OpenAI's Whisper model
- Text-to-speech capabilities with audio visualization
- A simple but effective UI with circular audio visualizers

## Features

- **Voice Input**: Capture and transcribe user speech using Whisper
- **Voice Output**: Convert agent responses to speech
- **Audio Visualization**: Visual feedback for both user and agent audio
- **Continuous Conversation**: Automatically listen for user input after agent responses
- **Silence Detection**: Automatically stops recording when the user stops speaking

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -e .
   ```

## Dependencies

- Python 3.10+
- PyAudio (for audio capture)
- OpenAI Whisper (for speech-to-text)
- NumPy and Matplotlib (for audio visualization)
- Pydub (for audio processing)
- Tkinter (for UI components)

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

### Audio Visualization

- Real-time circular visualizers show audio activity
- Left visualizer displays user's speech input
- Right visualizer displays agent's speech output
- Amplitude and variations in the audio affect the visualization pattern

## License

[MIT License](LICENSE)