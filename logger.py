"""Project-wide logger factory.

Place a single small module at project root so other modules can import
get_logger and obtain a preconfigured logger. Respects LOG_LEVEL env var.
"""
from __future__ import annotations

import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger for the project.

    - Uses LOG_LEVEL environment variable (default: INFO).
    - Adds a StreamHandler with a compact formatter the first time a
      logger is requested.
    - Sets propagate=False to avoid duplicate messages when root logger
      is configured elsewhere.
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
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger

