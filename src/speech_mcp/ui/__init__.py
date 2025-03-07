"""
UI module for the Speech MCP.

This module provides the UI component of the Speech MCP.
"""

import sys
import logging
from speech_mcp.ui.pyqt import run_ui

# Setup logging
logger = logging.getLogger(__name__)

def main():
    """Run the Speech UI."""
    logger.info("Starting Speech UI from __init__.py")
    return run_ui()

# Export the PyQt UI components
from speech_mcp.ui.pyqt import PyQtSpeechUI, run_ui

__all__ = [
    'main',
    'PyQtSpeechUI',
    'run_ui'
]