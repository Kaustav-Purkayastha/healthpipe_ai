"""
conftest.py -- Shared pytest fixtures for the HealthPipe AI test suite.

Fixtures defined here are automatically available to every test file
without needing an explicit import -- pytest discovers conftest.py
files and injects their fixtures by name.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from core.config import ROOT_DIR

# Use a module-level logger instead of print() for test diagnostics
logger = logging.getLogger(__name__)


@pytest.fixture
def test_fixture_path() -> Path:
    """
    Return the absolute Path to the 20-row test fixture CSV.

    The fixture lives at data/sample/test_fixture.csv and contains
    intentional quality issues (nulls, duplicate, negative age,
    mixed casing, empty string) for testing agent logic.
    """
    # Build the path relative to the project root defined in config
    path = ROOT_DIR / "data" / "sample" / "test_fixture.csv"
    return path


@pytest.fixture
def sample_dataframe(test_fixture_path: Path) -> pd.DataFrame:
    """
    Read the test fixture CSV into a pandas DataFrame.

    Depends on the test_fixture_path fixture so the path is
    resolved in one place.  Returns the raw 20-row DataFrame
    with all quality issues intact (no cleaning applied).
    """
    logger.info(f"Loading test fixture from: {test_fixture_path}")
    # low_memory=False forces pandas to scan full columns before
    # guessing types, avoiding mixed-type warnings
    df = pd.read_csv(test_fixture_path, low_memory=False)
    return df
