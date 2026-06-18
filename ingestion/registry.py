"""
registry.py — Central registry for all data sources.

The Registry Pattern lets you register sources by name and retrieve them later
without importing the specific class. This decouples the rest of the pipeline
from the concrete source implementations.

Usage:
    registry = SourceRegistry()         # auto-registers WHO, OpenFDA, CSV
    who = registry.get("who")           # returns a WHOSource instance
    df = who.extract(indicator="life_expectancy")
"""

from core.utils import get_logger
from ingestion.base_source import BaseSource
from ingestion.who_source import WHOSource
from ingestion.openfda_source import OpenFDASource
from ingestion.csv_source import CSVSource

logger = get_logger(__name__)


class SourceRegistry:
    """
    A named collection of data sources.

    Think of it like a phone book: you register sources by name,
    then look them up later with registry.get("who"). This makes it
    easy to add new sources without changing the pipeline code.
    """

    def __init__(self, auto_register: bool = True) -> None:
        """
        Initialize the registry with an empty source dictionary.

        Args:
            auto_register: If True (default), automatically register the
                           built-in sources (WHO, OpenFDA, CSV) at creation.
                           Set to False for testing with custom sources.
        """
        # _sources is a dict mapping name → source instance
        # e.g., {"who": WHOSource(), "openfda": OpenFDASource()}
        self._sources: dict[str, BaseSource] = {}

        if auto_register:
            self._register_defaults()

    def register(self, source: BaseSource) -> None:
        """
        Add a data source to the registry.

        If a source with the same name already exists, it is replaced
        and a warning is logged.

        Args:
            source: Any object that inherits from BaseSource.
        """
        if source.name in self._sources:
            logger.warning(
                f"Replacing existing source '{source.name}' in registry"
            )

        self._sources[source.name] = source
        logger.info(f"Registered source: '{source.name}' ({source.description})")

    def get(self, name: str) -> BaseSource | None:
        """
        Retrieve a source by its registered name.

        Args:
            name: The name the source was registered with (e.g., "who").

        Returns:
            The source instance, or None if no source with that name exists.
        """
        source = self._sources.get(name)
        if source is None:
            logger.error(
                f"Source '{name}' not found in registry. "
                f"Available: {list(self._sources.keys())}"
            )
        return source

    def list_sources(self) -> list[dict]:
        """
        Return a list of all registered sources with their details.

        Returns:
            List of dicts, each containing "name", "description", and "type".
        """
        return [
            {
                "name": source.name,
                "description": source.description,
                "type": type(source).__name__,
            }
            for source in self._sources.values()
        ]

    def check_all_connections(self) -> dict[str, bool]:
        """
        Test connectivity for every registered source.

        Calls connect() on each source and collects the results.
        Useful as a health check before starting a pipeline run.

        Returns:
            Dict mapping source name to True/False connection status.
            Example: {"who": True, "openfda": True, "csv": True}
        """
        results: dict[str, bool] = {}

        for name, source in self._sources.items():
            logger.info(f"Checking connection: '{name}'...")
            try:
                results[name] = source.connect()
            except Exception as e:
                # Catch-all so one broken source doesn't stop the health check
                logger.error(
                    f"Connection check failed for '{name}': {e}"
                )
                results[name] = False

        # Log a summary
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        logger.info(
            f"Connection check complete: {passed}/{total} sources OK"
        )
        return results

    def _register_defaults(self) -> None:
        """
        Register the three built-in data sources: WHO, OpenFDA, and CSV.

        Called automatically during __init__ unless auto_register=False.
        """
        self.register(WHOSource())
        self.register(OpenFDASource())
        self.register(CSVSource())
        logger.info(
            f"Default sources registered: {list(self._sources.keys())}"
        )

    def __len__(self) -> int:
        """Return the number of registered sources."""
        return len(self._sources)

    def __repr__(self) -> str:
        """Developer-friendly string showing all registered source names."""
        names = list(self._sources.keys())
        return f"<SourceRegistry(sources={names})>"
