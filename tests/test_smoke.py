"""
tests/test_smoke.py — Fast, hermetic smoke tests for the Step 01 scaffold.

These tests make no network calls and touch no filesystem outside the project
root.  They are meant to pass immediately after `pip install -r requirements-core.txt`.
"""

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Import smoke
# ---------------------------------------------------------------------------

def test_config_imports() -> None:
    """core.config must be importable without errors."""
    import core.config  # noqa: F401  (import side-effect check)


def test_utils_imports() -> None:
    """core.utils must be importable without errors."""
    import core.utils  # noqa: F401


# ---------------------------------------------------------------------------
# ROOT_DIR
# ---------------------------------------------------------------------------

def test_root_dir_exists() -> None:
    """ROOT_DIR must resolve to an existing directory on disk."""
    from core.config import ROOT_DIR

    assert isinstance(ROOT_DIR, Path), "ROOT_DIR should be a pathlib.Path"
    assert ROOT_DIR.exists(), f"ROOT_DIR does not exist: {ROOT_DIR}"
    assert ROOT_DIR.is_dir(), f"ROOT_DIR is not a directory: {ROOT_DIR}"


# ---------------------------------------------------------------------------
# FIPS_TO_STATE
# ---------------------------------------------------------------------------

def test_fips_to_state_has_51_entries() -> None:
    """FIPS_TO_STATE must contain exactly 51 entries (50 states + DC)."""
    from core.config import FIPS_TO_STATE

    assert len(FIPS_TO_STATE) == 51, (
        f"Expected 51 entries in FIPS_TO_STATE, got {len(FIPS_TO_STATE)}"
    )


def test_fips_to_state_keys_are_zero_padded() -> None:
    """All FIPS keys must be 2-character zero-padded strings."""
    from core.config import FIPS_TO_STATE

    for code in FIPS_TO_STATE:
        assert len(code) == 2 and code.isdigit(), (
            f"FIPS key '{code}' is not a 2-digit zero-padded string"
        )


def test_fips_dc_present() -> None:
    """DC must be in FIPS_TO_STATE under code '11'."""
    from core.config import FIPS_TO_STATE

    assert FIPS_TO_STATE.get("11") == "DC", "FIPS '11' should map to 'DC'"


def test_state_abbrs_has_51_entries() -> None:
    """STATE_ABBRS must contain exactly 51 entries (50 states + DC)."""
    from core.config import STATE_ABBRS

    assert len(STATE_ABBRS) == 51, (
        f"Expected 51 entries in STATE_ABBRS, got {len(STATE_ABBRS)}"
    )


# ---------------------------------------------------------------------------
# URL constants all start with http
# ---------------------------------------------------------------------------

URL_CONSTANTS = [
    "OLLAMA_BASE_URL",
    "GEMINI_BASE_URL",
    "WHO_BASE_URL",
    "OPENFDA_BASE_URL",
    "CMS_BASE_URL",
    "CDC_CDI_URL",
    "CDC_BRFSS_URL",
    "CENSUS_ACS5_URL",
]


@pytest.mark.parametrize("const_name", URL_CONSTANTS)
def test_url_constants_start_with_http(const_name: str) -> None:
    """Every URL constant must start with 'http' (http:// or https://)."""
    import core.config as cfg

    value: str = getattr(cfg, const_name)
    assert value.startswith("http"), (
        f"{const_name} = {value!r} does not start with 'http'"
    )


# ---------------------------------------------------------------------------
# QUALITY_GRADE_BANDS ordering is sane
# ---------------------------------------------------------------------------

def test_quality_grade_bands_ordering() -> None:
    """Grade A threshold must be > B, and B > C (stricter grade = higher score)."""
    from core.config import QUALITY_GRADE_BANDS

    assert QUALITY_GRADE_BANDS["A"] > QUALITY_GRADE_BANDS["B"], (
        "Grade A threshold must be greater than grade B threshold"
    )
    assert QUALITY_GRADE_BANDS["B"] > QUALITY_GRADE_BANDS["C"], (
        "Grade B threshold must be greater than grade C threshold"
    )


def test_quality_grade_bands_values() -> None:
    """Verify exact threshold values match the spec: A=90, B=75, C=60."""
    from core.config import QUALITY_GRADE_BANDS

    assert QUALITY_GRADE_BANDS["A"] == 90.0
    assert QUALITY_GRADE_BANDS["B"] == 75.0
    assert QUALITY_GRADE_BANDS["C"] == 60.0


def test_quality_gate_default_min_grade() -> None:
    """QUALITY_GATE_DEFAULT_MIN_GRADE must be 'B' per spec."""
    from core.config import QUALITY_GATE_DEFAULT_MIN_GRADE

    assert QUALITY_GATE_DEFAULT_MIN_GRADE == "B"
