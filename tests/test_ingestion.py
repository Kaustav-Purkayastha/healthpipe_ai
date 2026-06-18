"""
test_ingestion.py -- Tests for the ingestion layer (CSVSource and SourceRegistry).

Covers:
    - CSVSource reading the 20-row test fixture
    - CSVSource reading the 309K-row real dataset
    - CSVSource returning an empty DataFrame for a missing file
    - SourceRegistry listing the 3 built-in sources
    - SourceRegistry returning None for an unknown source name
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from core.config import ROOT_DIR
from ingestion.csv_source import CSVSource
from ingestion.registry import SourceRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSVSource tests
# ---------------------------------------------------------------------------

class TestCSVSource:
    """Tests for the CSVSource file reader."""

    def test_csv_source_reads_fixture(self, test_fixture_path: Path) -> None:
        """CSVSource should read the 20-row test fixture without errors."""
        source = CSVSource()
        # extract() accepts a filepath string, not a Path object
        df = source.extract(filepath=str(test_fixture_path))

        # Fixture has exactly 20 rows (19 unique + 1 duplicate)
        assert len(df) == 20, (
            f"Expected 20 rows from test fixture, got {len(df)}"
        )
        # Verify expected columns are present
        assert "patient_id" in df.columns, "Missing 'patient_id' column"
        assert "age" in df.columns, "Missing 'age' column"
        logger.info(f"Fixture loaded: {len(df)} rows, {len(df.columns)} cols")

    def test_csv_source_reads_real_dataset(self) -> None:
        """CSVSource should read the 309K-row real CSV dataset."""
        real_csv = ROOT_DIR / "data" / "sample" / "U.S._Chronic_Disease_Indicators.csv"
        if not real_csv.exists():
            pytest.skip("Real dataset not found -- skipping large file test")

        source = CSVSource()
        df = source.extract(filepath=str(real_csv))

        # The U.S. Chronic Disease Indicators dataset has ~309K rows
        assert len(df) >= 309_000, (
            f"Expected >= 309,000 rows from real dataset, got {len(df)}"
        )
        logger.info(f"Real dataset loaded: {len(df)} rows")

    def test_csv_source_missing_file(self) -> None:
        """CSVSource should return an empty DataFrame for a nonexistent file."""
        source = CSVSource()
        # Pass a path that does not exist on disk
        df = source.extract(filepath="nonexistent/fake_data.csv")

        # extract() returns pd.DataFrame() when the file is missing
        assert isinstance(df, pd.DataFrame), "Expected a DataFrame return"
        assert df.empty, "Expected an empty DataFrame for a missing file"
        logger.info("Missing file correctly returned empty DataFrame")


# ---------------------------------------------------------------------------
# SourceRegistry tests
# ---------------------------------------------------------------------------

class TestSourceRegistry:
    """Tests for the SourceRegistry central registry."""

    def test_registry_lists_sources(self) -> None:
        """Registry should auto-register exactly 3 built-in sources."""
        registry = SourceRegistry()  # auto_register=True by default
        sources = registry.list_sources()

        # Built-in sources: who, openfda, csv
        assert len(sources) == 3, (
            f"Expected 3 registered sources, got {len(sources)}"
        )
        # Verify each expected source name is present
        source_names = {s["name"] for s in sources}
        assert "who" in source_names, "WHO source not registered"
        assert "openfda" in source_names, "OpenFDA source not registered"
        assert "csv" in source_names, "CSV source not registered"
        logger.info(f"Registry sources: {source_names}")

    def test_registry_get_unknown(self) -> None:
        """Registry.get() should return None for an unregistered source name."""
        registry = SourceRegistry()
        result = registry.get("nonexistent_source")

        assert result is None, (
            "Expected None for unknown source, got a real object"
        )
        logger.info("Unknown source correctly returned None")
