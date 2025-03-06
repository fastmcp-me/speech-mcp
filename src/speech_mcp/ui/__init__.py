"""
Speech UI module for speech-mcp.

This module provides the user interface for the speech MCP extension.
The implementation uses PyQt for a modern, responsive UI with audio visualization.
"""

import logging
import sys
import os

# Set up logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speech-mcp-ui.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import PyQt UI components directly from speech_ui.py
from .speech_ui import PyQtSpeechUI, AudioProcessor, AudioVisualizer, run_ui

def main():
    """Main entry point for the speech processor UI"""
    try:
        logger.info("Starting Speech MCP Processor (PyQt UI)")
        print("\n===== Speech MCP Processor (PyQt UI) =====")
        print("Starting speech recognition system...")
        
        # Run the PyQt UI
        run_ui()
            
    except Exception as e:
        logger.error(f"Error in speech processor main: {e}", exc_info=True)
        print(f"\nERROR: Failed to start speech processor: {e}")

if __name__ == "__main__":
    main()