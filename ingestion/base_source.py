"""
base_source.py — Abstract base class for all data sources.

Every data source (WHO API, OpenFDA API, local CSV) inherits from this class.
This guarantees they all have the same interface: connect(), extract(), get_metadata().
Downstream code can work with any source without knowing its type — that's polymorphism.
"""

from abc import ABC, abstractmethod  # ABC = Abstract Base Class
import pandas as pd


class BaseSource(ABC):
    """
    Abstract base class that every data source must inherit from.

    Subclasses MUST implement the three abstract methods below.
    Trying to instantiate a subclass without implementing them
    raises TypeError at construction time — so bugs are caught early.
    """

    def __init__(self, name: str, description: str) -> None:
        """
        Initialize the source with a human-readable name and description.

        Args:
            name:        Short identifier (e.g., "who", "openfda", "csv").
            description: One-line summary of what this source provides.
        """
        self.name = name
        self.description = description

    @abstractmethod
    def connect(self) -> bool:
        """
        Verify that the data source is reachable.

        For APIs: make a lightweight test request to check connectivity.
        For files: check that the file path exists on disk.

        Returns:
            True if connection is successful, False otherwise.
        """
        pass  # Subclasses must override this — calling it directly raises TypeError

    @abstractmethod
    def extract(self, **kwargs) -> pd.DataFrame:
        """
        Pull data from the source and return it as a pandas DataFrame.

        Keyword arguments vary by source type (e.g., indicator code for WHO,
        search term for OpenFDA, file path for CSV).

        Args:
            **kwargs: Source-specific parameters for the extraction.

        Returns:
            A pandas DataFrame containing the extracted data.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Return a dictionary describing the source and its current state.

        Useful for auditing: lets you log which source produced which data,
        when, and with what parameters.

        Returns:
            Dict with keys like "name", "source_type", "record_count", etc.
        """
        pass

    def __repr__(self) -> str:
        """
        Developer-friendly string for debugging.

        When you print(source) or inspect it in a debugger, you'll see
        something like: <WHOSource(name='who')> instead of a memory address.
        """
        # type(self).__name__ gets the actual subclass name, not "BaseSource"
        return f"<{type(self).__name__}(name='{self.name}')>"
