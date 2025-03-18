#!/bin/bash
# Script to generate a deeplink for local testing of the speech-mcp extension

set -e  # Exit on error

# Change to the project directory
cd "$(dirname "$0")"

# Build the wheel
echo "Building wheel..."
python -m build

# Get the wheel file path
WHEEL_FILE=$(ls -t dist/*.whl | head -1)
WHEEL_PATH="$(pwd)/$WHEEL_FILE"

# URL encode the wheel path
ENCODED_PATH=$(python -c "import urllib.parse; print(urllib.parse.quote('$WHEEL_PATH'))")

# Generate the deeplink
DEEPLINK="goose://extension?cmd=uvx&arg=-p&arg=3.10.14&arg=$ENCODED_PATH&id=speech_mcp&name=Speech%20Interface&description=Voice%20interaction%20with%20audio%20visualization%20for%20Goose"

echo "Generated deeplink:"
echo "$DEEPLINK"

# Copy to clipboard if pbcopy is available (macOS)
if command -v pbcopy &> /dev/null; then
    echo "$DEEPLINK" | pbcopy
    echo "Deeplink copied to clipboard!"
fi

echo "You can now paste this deeplink into your browser to install the extension."
