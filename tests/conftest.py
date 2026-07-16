"""
tests/conftest.py — Shared pytest fixtures for HealthPipe AI v2.

Provides:
  sample_dir          — Path to data/sample/
  fixture_csv         — Path to test_fixture.csv
  fixture_json        — Path to test_fixture.json
  fixture_parquet     — Path to test_fixture.parquet
  fixture_xlsx        — Path to test_fixture.xlsx

If the fixture files are missing (e.g. fresh clone), they are generated
automatically by calling data/sample/make_fixtures.build_fixtures().
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so imports like `from core.config`
# work when pytest is run from any directory.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SAMPLE_DIR  # noqa: E402 — after sys.path setup


def _ensure_fixtures_exist() -> None:
    """Generate fixture files if any of the four expected files are absent."""
    expected = [
        SAMPLE_DIR / "test_fixture.csv",
        SAMPLE_DIR / "test_fixture.json",
        SAMPLE_DIR / "test_fixture.parquet",
        SAMPLE_DIR / "test_fixture.xlsx",
    ]
    if not all(p.exists() for p in expected):
        # Import and run the generator — importable by design.
        sys.path.insert(0, str(SAMPLE_DIR.parent))  # so `make_fixtures` resolves
        from data.sample.make_fixtures import build_fixtures
        build_fixtures(output_dir=SAMPLE_DIR)


# Generate once at collection time (not inside a fixture) so path fixtures
# can safely reference files that exist before any test runs.
_ensure_fixtures_exist()


# ---------------------------------------------------------------------------
# Directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_dir() -> Path:
    """Return the Path to data/sample/."""
    return SAMPLE_DIR


# ---------------------------------------------------------------------------
# Per-format path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixture_csv(sample_dir: Path) -> Path:
    """Path to data/sample/test_fixture.csv."""
    return sample_dir / "test_fixture.csv"


@pytest.fixture(scope="session")
def fixture_json(sample_dir: Path) -> Path:
    """Path to data/sample/test_fixture.json."""
    return sample_dir / "test_fixture.json"


@pytest.fixture(scope="session")
def fixture_parquet(sample_dir: Path) -> Path:
    """Path to data/sample/test_fixture.parquet."""
    return sample_dir / "test_fixture.parquet"


@pytest.fixture(scope="session")
def fixture_xlsx(sample_dir: Path) -> Path:
    """Path to data/sample/test_fixture.xlsx."""
    return sample_dir / "test_fixture.xlsx"
