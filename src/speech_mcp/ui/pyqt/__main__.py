"""
Entry point for running the PyQt UI directly.

This module allows the PyQt UI to be run directly for testing.
"""

import sys
import logging
from speech_mcp.ui.pyqt.pyqt_ui import run_ui

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Run the UI
    sys.exit(run_ui())