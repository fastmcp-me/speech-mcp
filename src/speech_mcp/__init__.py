import argparse
import sys
import os
import signal
import atexit
from .server import mcp, cleanup_ui_process

# Ensure UI process is cleaned up on exit
atexit.register(cleanup_ui_process)

# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    cleanup_ui_process()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Speech MCP: Voice interaction with speech recognition."""
    try:
        parser = argparse.ArgumentParser(
            description="Voice interaction with speech recognition."
        )
        parser.parse_args()
        mcp.run()
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()