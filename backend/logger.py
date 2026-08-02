"""
Logging Configuration Module for SentimentIQ

Configures structured logging with formatting, timestamps, log levels,
and optional file logging.
"""

import os
import logging
import sys
from backend.config import LOG_LEVEL, BASE_DIR

# Determine log level
numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

# Create logs directory
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "sentimentiq.log")

# Custom formatter
log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Root logger setup
logger = logging.getLogger("SentimentIQ")
logger.setLevel(numeric_level)
logger.propagate = False

if not logger.handlers:
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"[Logger] Warning: Could not create file log handler: {e}")


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger prefixed with module_name."""
    return logger.getChild(module_name)
