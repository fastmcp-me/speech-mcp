# Kokoro TTS Adapter for speech-mcp

This resource provides information about the Kokoro TTS adapter for speech-mcp.

## Overview

The Kokoro TTS adapter allows speech-mcp to use Kokoro for high-quality text-to-speech synthesis. Kokoro is an open-weight TTS model with 82 million parameters that delivers comparable quality to larger models while being significantly faster and more cost-efficient.

## Installation

There are two ways to install Kokoro TTS:

### Option 1: Using pip with optional dependencies

```bash
pip install speech-mcp[kokoro]     # Basic Kokoro support with English
pip install speech-mcp[ja]         # Add Japanese support
pip install speech-mcp[zh]         # Add Chinese support
pip install speech-mcp[all]        # All languages and features
```

### Option 2: Using the installation script

```bash
python scripts/install_kokoro.py
```

## Features

- High-quality neural text-to-speech
- Multiple voice styles (casual, serious, robot, bright, etc.)
- Multiple languages (English, Japanese, Chinese, Spanish, etc.)
- Lightweight model (82M parameters) that runs efficiently on CPU
- Apache-licensed weights for use in any project

## Available Voices

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

- 🇺🇸 'a': American English (default)
- 🇬🇧 'b': British English
- 🇪🇸 'e': Spanish
- 🇫🇷 'f': French
- 🇮🇳 'h': Hindi
- 🇮🇹 'i': Italian
- 🇯🇵 'j': Japanese (requires `pip install misaki[ja]`)
- 🇧🇷 'p': Brazilian Portuguese
- 🇨🇳 'z': Mandarin Chinese (requires `pip install misaki[zh]`)

## Documentation

For more information, see the [Kokoro TTS Guide](../docs/kokoro-tts-guide.md).