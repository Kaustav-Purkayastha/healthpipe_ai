"""
ingestion/registry.py — Central registry for all HealthPipe AI v2 data sources.

Sources register themselves (or are auto-registered) and pipeline code
retrieves them by name, never by class reference.  This decouples the
pipeline from any specific connector implementation.

Usage:
    from ingestion.registry import SourceRegistry
    registry = SourceRegistry()            # auto-registers built-ins
    source = registry.get("file")          # FileSource instance
    df = source.extract(filepath="...")
"""

from __future__ import annotations

from typing import Optional

from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)


class SourceRegistry:
    """Maintains a name → BaseSource mapping for the whole application.

    Args:
        auto_register: When True (default), built-in sources are registered
                       automatically during __init__.  Pass False in unit tests
                       that want a clean, empty registry.
    """

    def __init__(self, auto_register: bool = True) -> None:
        """Initialise registry, optionally auto-registering built-in sources."""
        self._sources: dict[str, BaseSource] = {}

        if auto_register:
            self._register_builtins()

    # ------------------------------------------------------------------
    # Built-in auto-registration
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """Register every built-in source that ships with HealthPipe AI v2.

        Sources are imported lazily (inside this method) so missing optional
        drivers produce a friendly warning rather than an import-time crash.

        # EXTENSION POINT — add new built-in sources below this comment:
        """
        # Step 02: file-based source (CSV/TSV/JSON/Parquet/XLSX)
        from ingestion.file_source import FileSource
        self.register(FileSource())

        # Step 07: public API sources
        from ingestion.who_source import WHOSource
        self.register(WHOSource())

        from ingestion.openfda_source import OpenFDASource
        self.register(OpenFDASource())

        from ingestion.cms_source import CMSMedicareSource
        self.register(CMSMedicareSource())

        from ingestion.cdc_cdi_source import CDCChronicDiseaseSource
        self.register(CDCChronicDiseaseSource())

        from ingestion.brfss_source import BRFSSSource
        self.register(BRFSSSource())

        from ingestion.census_source import CensusPopulationSource
        self.register(CensusPopulationSource())

        # Step 17 (validation upgrade): CDC PLACES county-level source
        from ingestion.places_source import CDCPlacesSource
        self.register(CDCPlacesSource())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, source: BaseSource) -> None:
        """Add a source to the registry, keyed by its ``name`` attribute.

        Overwrites silently if a source with the same name already exists
        (allows hot-reloading in Streamlit without crashing).

        Args:
            source: Any concrete BaseSource subclass instance.
        """
        self._sources[source.name] = source
        _log.debug("Registered source '%s' (type=%s)", source.name, source.source_type)

    def get(self, name: str) -> Optional[BaseSource]:
        """Retrieve a source by name.

        Args:
            name: The registry key (matches BaseSource.name).

        Returns:
            The registered source, or None if not found.
        """
        return self._sources.get(name)

    def list_sources(self) -> list[dict]:
        """Return metadata dicts for every registered source.

        Returns:
            List of dicts from each source's get_metadata() method,
            sorted alphabetically by name.
        """
        return [
            source.get_metadata()
            for source in sorted(self._sources.values(), key=lambda s: s.name)
        ]

    def check_all_connections(self) -> dict[str, bool]:
        """Ping every registered source and return a name → bool map.

        Returns:
            Dict mapping source name to True (reachable) or False (not reachable).
        """
        results: dict[str, bool] = {}
        for name, source in self._sources.items():
            try:
                results[name] = source.connect()
            except Exception as exc:  # noqa: BLE001 — any error → False, never crash
                _log.warning("Connection check failed for '%s': %s", name, exc)
                results[name] = False
        return results
