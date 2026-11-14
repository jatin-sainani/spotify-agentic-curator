"""Structured logging configuration for spotify-agentic-curator."""

import logging
import sys
from typing import Optional

from .config import settings


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """Setup structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in logs

    Returns:
        Configured logger instance
    """
    if level is None:
        level = settings.log_level

    # Windows consoles default to a legacy code page (e.g. cp1252); replace
    # characters it can't encode (arrows, emoji, non-Latin track names)
    # instead of raising UnicodeEncodeError mid-run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format string
    if format_string is None:
        time_format = "%(asctime)s " if include_timestamp else ""
        format_string = f"{time_format}%(levelname)s - %(name)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,  # stdout is for command output and the MCP stdio protocol
    )

    # Create logger for this module
    logger = logging.getLogger("spotify_agentic_curator")
    logger.setLevel(numeric_level)

    return logger


# Global logger instance
logger = setup_logging()
