"""
utils.py — Shared helper functions used across the entire pipeline.

Functions here are stateless and side-effect-free (except file I/O),
so they are safe to import and call from any module.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger with consistent formatting.

    Every module should call this at the top:
        logger = get_logger(__name__)

    Using __name__ automatically sets the logger name to the module's
    dotted path (e.g., "ingestion.who_source"), which makes log lines
    easy to trace back to their source.

    Args:
        name: Logger name, typically the module's __name__.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called more than once
    # for the same name (e.g., during tests that reimport modules).
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    # formatter applies the timestamp + level + name + message layout
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler — always on
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — only if LOG_FILE is set in config
    if LOG_FILE is not None:
        # Ensure the log file's parent directory exists before writing
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent messages from bubbling up to the root logger (avoids duplicate output)
    logger.propagate = False

    return logger


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """
    Serialize `data` to a JSON file, creating parent directories if needed.

    Uses ensure_ascii=False so international characters (e.g., country names)
    are stored as-is rather than as escaped Unicode sequences.

    Args:
        data:      Any JSON-serialisable object (dict, list, str, int, …).
        file_path: Destination path for the .json file.
        indent:    Number of spaces for pretty-printing (default 2).
    """
    # Create the directory tree if it doesn't exist yet
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 'with open()' ensures the file is closed even if an error occurs mid-write
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        # default=str converts non-serialisable types (e.g., datetime, Path)
        # to strings instead of raising TypeError


def load_json(file_path: Path) -> Any:
    """
    Load and parse a JSON file, returning the Python object it contains.

    Args:
        file_path: Path to the .json file to read.

    Returns:
        Parsed Python object (dict, list, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp_string(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Return the current local datetime as a formatted string.

    Used for naming output files so runs don't overwrite each other.
    Example return value: "20240915_143022"

    Args:
        fmt: strftime format string. Default gives YYYYMMDD_HHMMSS.

    Returns:
        Formatted datetime string.
    """
    return datetime.now().strftime(fmt)
