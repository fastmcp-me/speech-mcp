# Speech MCP Logging System

This document describes the centralized logging system for the speech-mcp project.

## Overview

The speech-mcp project uses a centralized logging system that provides consistent logging across all components. The logging system writes logs to a deterministic location on disk and provides different log files for different components of the system.

## Log Files

All log files are stored in the `~/.speech-mcp/logs/` directory. The following log files are available:

- `speech-mcp.log`: Main log file for general messages
- `speech-mcp-server.log`: Log file for server-related messages
- `speech-mcp-ui.log`: Log file for UI-related messages
- `speech-mcp-tts.log`: Log file for text-to-speech related messages
- `speech-mcp-stt.log`: Log file for speech-to-text related messages

## Log Rotation

Log files are automatically rotated when they reach 10MB in size. Up to 5 backup log files are kept for each log file.

## Usage

To use the logging system in your code, import the `get_logger` function from the `speech_mcp.utils.logger` module:

```python
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__)

# Or specify a component to determine the log file
logger = get_logger(__name__, component="server")
```

Then use the logger to log messages at different levels:

```python
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

To log exceptions with traceback:

```python
try:
    # Some code that might raise an exception
    raise ValueError("Example error")
except Exception as e:
    logger.exception("An error occurred")
```

## Log Level

The default log level is `INFO`. You can change the log level for all loggers using the `set_log_level` function:

```python
from speech_mcp.utils.logger import set_log_level
import logging

set_log_level(logging.DEBUG)
```

## Accessing Log Files

You can get a dictionary of log file paths using the `get_log_files` function:

```python
from speech_mcp.utils.logger import get_log_files

log_files = get_log_files()
print(log_files["server"])  # Prints the path to the server log file
```

## Stack Traces on Signal Termination

The system is configured to dump stack traces when it receives a SIGINT or SIGTERM signal. This is useful for debugging issues where the application might be stuck or deadlocked. The stack traces are printed to stderr.