"""
ingestion/base_source.py — Abstract base class for every data source in HealthPipe AI v2.

All concrete sources (FileSource, API sources, DB sources) must inherit from BaseSource
and implement the three abstract methods below.  The registry pattern means pipeline
code never needs to know which source it is talking to.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


class BaseSource(ABC):
    """Abstract base for all data-source connectors.

    Subclasses set ``source_type`` at the class level to one of:
    ``"api"``, ``"file"``, or ``"database"``.  The default is ``"generic"``.

    Args:
        name:        Short, unique identifier used as the registry key.
        description: Human-readable description shown in the UI.
    """

    # Subclasses override this to identify the category of source.
    source_type: str = "generic"

    def __init__(self, name: str, description: str) -> None:
        """Initialise shared metadata fields.

        Args:
            name:        Unique registry key for this source.
            description: Human-readable label used in the UI source selector.
        """
        self.name: str = name
        self.description: str = description
        # Set after a successful extract(); None until first extraction.
        self.last_extract_time: Optional[datetime] = None
        # Row count from the most recent extract(); -1 until first extraction.
        self.last_record_count: int = -1

    # ------------------------------------------------------------------
    # Abstract interface — every source must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """Test connectivity or readiness for this source.

        For file-based sources this is trivially True; for API/DB sources
        it should perform a lightweight ping/handshake.

        Returns:
            True if the source is reachable and ready, False otherwise.
        """

    @abstractmethod
    def extract(self, **kwargs) -> pd.DataFrame:
        """Pull data from the source and return it as a DataFrame.

        Implementations must update ``last_extract_time`` and
        ``last_record_count`` before returning.

        Args:
            **kwargs: Source-specific parameters (e.g. filepath, filters).

        Returns:
            A pandas DataFrame; empty DataFrame on failure (never raises).
        """

    # ------------------------------------------------------------------
    # Concrete — subclasses may call super() and extend the dict
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict:
        """Return a dict of metadata for this source.

        Subclasses may call ``super().get_metadata()`` and extend the
        returned dict with source-specific fields.

        Returns:
            Dict with keys: name, description, source_type,
            last_extract_time, last_record_count.
        """
        return {
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type,
            "last_extract_time": (
                self.last_extract_time.isoformat()
                if self.last_extract_time is not None
                else None
            ),
            "last_record_count": self.last_record_count,
        }

    def _record_extract(self, df: pd.DataFrame) -> None:
        """Update bookkeeping fields after an extraction.

        Call this inside every concrete ``extract()`` before returning.

        Args:
            df: The DataFrame that was just extracted.
        """
        self.last_extract_time = datetime.now(tz=timezone.utc)
        self.last_record_count = len(df)
