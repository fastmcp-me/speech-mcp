# Kokoro TTS for speech-mcp

This guide explains how to use Kokoro TTS with the speech-mcp extension for Goose.

## What is Kokoro?

Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.

## Installation

There are two ways to install Kokoro TTS for speech-mcp:

### Option 1: Using pip with optional dependencies

You can install Kokoro TTS directly with pip using the optional dependencies:

```bash
pip install speech-mcp[kokoro]     # Basic Kokoro support with English
pip install speech-mcp[ja]         # Add Japanese support
pip install speech-mcp[zh]         # Add Chinese support
pip install speech-mcp[all]        # All languages and features
```

### Option 2: Using the installation script

Alternatively, you can run the installation script:

```bash
python scripts/install_kokoro.py
```

This script will:
1. Create a virtual environment for Kokoro (by default at `~/.speech-mcp/kokoro-venv`)
2. Install Kokoro and its dependencies
3. Configure it for use with speech-mcp

### Installation Options

You can customize the installation with these options:

- `--venv PATH`: Specify a custom path for the virtual environment
- `--no-venv`: Install in the current Python environment instead of creating a virtual environment
- `--force`: Force reinstallation even if already installed

Example:
```bash
python scripts/install_kokoro.py --venv ~/my-kokoro-env
```

## Available Voices

Kokoro comes with several voice styles:

- `af_heart`: Female voice with warm, natural tone (default)
- `af_chill`: Female voice with relaxed, calm tone
- `af_robot`: Female voice with robotic, synthetic tone
- `af_bright`: Female voice with bright, cheerful tone
- `af_serious`: Female voice with serious, formal tone
- `am_casual`: Male voice with casual, relaxed tone
- `am_calm`: Male voice with calm, soothing tone
- `am_serious`: Male voice with serious, formal tone
- `am_happy`: Male voice with happy, upbeat tone

## Language Support

Kokoro supports multiple languages:

- 🇺🇸 'a': American English (default)
- 🇬🇧 'b': British English
- 🇪🇸 'e': Spanish
- 🇫🇷 'f': French
- 🇮🇳 'h': Hindi
- 🇮🇹 'i': Italian
- 🇯🇵 'j': Japanese (requires `pip install misaki[ja]`)
- 🇧🇷 'p': Brazilian Portuguese
- 🇨🇳 'z': Mandarin Chinese (requires `pip install misaki[zh]`)

## Customizing Voices

To customize the voice used by the speech-mcp extension, you need to modify the `KokoroTTS` initialization in the `tts_adapters/kokoro_adapter.py` file:

```python
# Change these parameters to customize the voice
tts_engine = KokoroTTS(
    voice="af_heart",  # Change to any voice from the list above
    lang_code="a",     # Change to any language code from the list above
    speed=1.0          # Adjust speed (0.5 = slower, 1.5 = faster)
)
```

## Troubleshooting

If you encounter issues with Kokoro TTS:

1. **Check installation**: Verify that Kokoro was installed correctly by running:
   ```
   python -c "import kokoro; print(kokoro.__version__)"
   ```

2. **Check logs**: Look at the log files in `src/speech_mcp/` for detailed error messages.

3. **Fallback mechanism**: The system will automatically fall back to pyttsx3 if Kokoro fails.

4. **Manual installation**: If the installation script fails, try installing Kokoro manually:
   ```
   pip install kokoro>=0.8.4 soundfile torch misaki[en]
   ```

5. **Network issues**: If you're behind a corporate firewall or proxy, you might need to configure pip to use a specific index URL:
   ```
   pip install kokoro --index-url https://pypi.org/simple
   ```

## Resources

- [Kokoro GitHub Repository](https://github.com/hexgrad/kokoro)
- [Kokoro on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
- [Kokoro Samples](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/SAMPLES.md)