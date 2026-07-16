"""
core/utils.py — Shared utility helpers used across all HealthPipe AI v2 modules.

Provides:
  - get_logger(name)       : returns a configured logger (dedup-guarded)
  - save_json(obj, path)   : serialize any object to JSON, dates/paths → strings
  - load_json(path)        : deserialize JSON from a file
  - timestamp_string()     : ISO-8601 timestamp suitable for filenames
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import ROOT_DIR


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with console and rotating file handlers.

    Uses a dedup guard so calling get_logger('foo') multiple times does NOT
    attach duplicate handlers — important when modules are hot-reloaded in
    Streamlit or re-imported in tests.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Guard: only add handlers the first time this logger is configured.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above to avoid swamping Streamlit output.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — DEBUG and above; written to pipeline.log in project root.
    log_path: Path = ROOT_DIR / "pipeline.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid double-logging.
    logger.propagate = False

    return logger


def save_json(obj: Any, path: Path) -> None:
    """Serialize *obj* to JSON and write to *path*, creating parent dirs if needed.

    Uses ``default=str`` so datetime, Path, and other non-serializable types
    are automatically converted to their string representations rather than
    raising TypeError.

    Args:
        obj:  Any JSON-serializable object (dict, list, etc.).
        path: Destination file path (pathlib.Path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Deserialize JSON from *path* and return the Python object.

    Args:
        path: Source file path (pathlib.Path or str).

    Returns:
        Parsed Python object (dict, list, etc.).

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def timestamp_string() -> str:
    """Return a UTC timestamp string safe for use in file names.

    Format: ``YYYYMMDD_HHMMSS`` (e.g. ``20260710_143022``).

    Returns:
        Formatted UTC timestamp string.
    """
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
