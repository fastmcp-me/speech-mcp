"""
TTS adapters for speech-mcp

This package contains adapters for various text-to-speech engines.
"""

# Import adapters
try:
    from .kokoro_adapter import KokoroTTS
except ImportError:
    # Kokoro adapter not available
    pass