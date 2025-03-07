"""
Main UI module for the Speech MCP.

This module provides the main entry point for the UI component of the Speech MCP.
"""

import os
import sys
import logging
from speech_mcp.ui.pyqt import run_ui

# Setup logging
logger = logging.getLogger(__name__)

def main():
    """Run the Speech UI."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speech-mcp-ui.log"),
        filemode='a'
    )
    
    # Add console handler for debugging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logger.info("Starting Speech UI")
    
    # Run the PyQt UI
    return run_ui()

if __name__ == "__main__":
    sys.exit(main())