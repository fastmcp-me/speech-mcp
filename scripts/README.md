# Speech MCP Scripts

This directory contains utility scripts for the speech-mcp extension.

## Available Scripts

### `install_kokoro.py`

Installs Kokoro TTS and its dependencies for use with speech-mcp.

```bash
python install_kokoro.py
```

Options:
- `--venv PATH`: Specify a custom path for the virtual environment
- `--no-venv`: Install in the current Python environment
- `--force`: Force reinstallation even if already installed

### `test_kokoro.py`

Tests the Kokoro TTS adapter to verify that it's working correctly.

```bash
python test_kokoro.py
```

This script will:
1. Import the Kokoro adapter
2. Initialize the TTS engine
3. List available voices
4. Speak some test text

If Kokoro is not available, it will fall back to pyttsx3.