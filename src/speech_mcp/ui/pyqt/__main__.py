#!/usr/bin/env python
"""
Test script to run the PyQt UI directly.
"""

import os
import sys
import logging

if __name__ == "__main__":
    # Configure logging
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "speech-mcp-ui-pyqt.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting PyQt Speech UI")
    
    # Import after logging is configured
    from .pyqt_ui import run_ui
    
    # Run the UI
    sys.exit(run_ui())