# Local Installation Guide

This document provides instructions for installing the speech-mcp extension from a local wheel file.

## Prerequisites

- Python 3.10 or higher
- UVX (Universal Extension Manager)
- Goose Desktop App

## Installation Options

### Option 1: Using the installation script

The easiest way to install the extension locally is to use the provided installation script:

```bash
# Just install the extension
./install_local.sh

# Install and start a Goose session
./install_local.sh --goose
```

### Option 2: Generate a deeplink

You can generate a deeplink that can be used to install the extension:

```bash
./generate_deeplink.sh
```

This will generate a deeplink and copy it to your clipboard. You can then paste it into your browser to install the extension.

### Option 3: Manual installation

If you prefer to install the extension manually, you can use the following commands:

```bash
# Build the wheel
python -m build

# Get the wheel file path
WHEEL_FILE=$(ls -t dist/*.whl | head -1)

# Install with UVX
uvx "$PWD/$WHEEL_FILE"

# Or start a Goose session with the extension
goose session --with-extension "uvx $PWD/$WHEEL_FILE"
```

## Troubleshooting

If you encounter any issues during installation, check the following:

1. Make sure you have the latest version of UVX installed:

```bash
pip install -U uvx
```

2. Check if the wheel file was built correctly:

```bash
ls -la dist/*.whl
```

3. Try installing with pip directly to see if there are any dependency issues:

```bash
pip install ./dist/*.whl
```

4. Check the logs for any error messages:

```bash
cat ~/.speech-mcp/logs/speech-mcp.log
```

## Additional Resources

- [TTS Initialization Fix](./TTS_INITIALIZATION_FIX.md): Instructions for fixing TTS initialization issues
- [Test Scripts](./test_kokoro.py): Scripts for testing Kokoro TTS functionality