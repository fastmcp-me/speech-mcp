import argparse
import logging
import sys
import os
import signal
import atexit
from .server import mcp, cleanup_ui_process

# Set up logging to both console and file
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'speech-mcp.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Ensure UI process is cleaned up on exit
atexit.register(cleanup_ui_process)

# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    logging.info(f"Received signal {sig}, shutting down...")
    cleanup_ui_process()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Speech MCP: Voice interaction with speech recognition."""
    logging.info("Starting Speech MCP server...")
    try:
        parser = argparse.ArgumentParser(
            description="Voice interaction with speech recognition."
        )
        parser.parse_args()
        logging.info("Running MCP server...")
        mcp.run()
    except Exception as e:
        logging.exception(f"Error running MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()