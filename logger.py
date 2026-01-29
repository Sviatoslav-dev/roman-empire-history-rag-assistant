import logging
import os
import sys
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for the project.

    - Uses LOG_LEVEL environment variable (default: INFO).
    - Adds a StreamHandler with a compact formatter the first time a
      logger is requested.
    - Sets propagate=False to avoid duplicate messages when root logger
      is configured elsewhere.

    Note: write normal log output to stdout (not stderr) so that IDEs
    or consoles that color stderr in red don't render all logs as errors.
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()

    if level_name not in valid_levels:
        level_name = "INFO"

    level = getattr(logging, level_name, logging.INFO)

    logger_name = name or "roman_empire"
    logger = logging.getLogger(logger_name)

    # Configure handler only once per logger
    if not logger.handlers:
        # Use stdout so normal INFO/debug logs are not colored as stderr in some IDEs
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        # Let the logger's level control what is emitted; handler should not filter further
        handler.setLevel(logging.NOTSET)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
