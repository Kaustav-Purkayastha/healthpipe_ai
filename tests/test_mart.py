"""
tests/test_mart.py — Offline tests for analytics/mart_builder.py.

All source extract() calls are monkeypatched — no API access needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.config import STATE_ABBRS
from analytics.mart_builder import MartBuilder


# ---------------------------------------------------------------------------
# Mock data builders
# ---------------------------------------------------------------------------

def _make_census_frame() -> pd.DataFrame:
    """51 state rows with 1M population each."""
    rows = [{"state_abbr": s, "population": 1_000_000} for s in STATE_ABBRS]
    return pd.DataFrame(rows)


def _make_cdi_frame() -> pd.DataFrame:
    """CDI rows with:
    - FL: 2023 null, 2022=12.5, 2021=11.8  → vintage rule picks 2022
    - US aggregate row (should be excluded by MartBuilder)
    - All other states: 2023 value = 10.0
    """
    rows = []
    for state in STATE_ABBRS:
        if state == "FL":
            rows.append({
                "locationabbr": "FL", "yearstart": "2023",
                "datavalue": float("nan"), "is_suppressed": False,
            })
            rows.append({
                "locationabbr": "FL", "yearstart": "2022",
                "datavalue": 12.5, "is_suppressed": False,
            })
            rows.append({
                "locationabbr": "FL", "yearstart": "2021",
                "datavalue": 11.8, "is_suppressed": False,
            })
        else:
            rows.append({
                "locationabbr": state, "yearstart": "2023",
                "datavalue": 10.0, "is_suppressed": False,
            })
    # Aggregate row — MartBuilder must exclude this
    rows.append({
        "locationabbr": "US", "yearstart": "2023",
        "datavalue": 11.5, "is_suppressed": False,
    })
    return pd.DataFrame(rows)


def _make_cms_frame() -> pd.DataFrame:
    """CMS rows with one NPI per state, plus a ZZ row to be excluded."""
    rows = []
    for i, state in enumerate(STATE_ABBRS):
        rows.append({
            "state": state,
            "npi": f"{1000000000 + i}",
            "total_medicare_payment": 1_000_000.0,
        })
    # Foreign ZZ row — must be excluded before aggregation
    rows.append({
        "state": "ZZ",
        "npi": "0000000001",
        "total_medicare_payment": 999.0,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mart_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MartBuilder:
    """MartBuilder with mocked sources and isolated cache dir."""
    census_df = _make_census_frame()
    cdi_df = _make_cdi_frame()
    cms_df = _make_cms_frame()

    monkeypatch.setattr(
        "ingestion.census_source.CensusPopulationSource.extract",
        lambda self, **kw: census_df,
    )
    monkeypatch.setattr(
        "ingestion.cdc_cdi_source.CDCChronicDiseaseSource.extract",
        lambda self, **kw: cdi_df,
    )
    monkeypatch.setattr(
        "ingestion.cms_source.CMSMedicareSource.extract",
        lambda self, **kw: cms_df,
    )
    monkeypatch.setattr(
        "ingestion.cms_source.CMSMedicareSource.connect",
        lambda self: True,
    )
    monkeypatch.setattr(
        "ingestion.cms_source.CMSMedicareSource.get_metadata",
        lambda self: {"total_rows_in_dataset": 1_300_000},
    )

    return MartBuilder(cache_dir=tmp_path / "cache")


# ===========================================================================
# build() — structure
# ===========================================================================

class TestBuild:
    """MartBuilder.build() must return 51 rows with correct joins."""

    def test_returns_51_rows(self, mart_builder: MartBuilder) -> None:
        df = mart_builder.build(refresh=True)
        assert len(df) == 51

    def test_state_abbr_unique(self, mart_builder: MartBuilder) -> None:
        df = mart_builder.build(refresh=True)
        assert df["state_abbr"].nunique() == 51

    def test_all_canonical_states_present(self, mart_builder: MartBuilder) -> None:
        df = mart_builder.build(refresh=True)
        assert set(df["state_abbr"]) == set(STATE_ABBRS)

    def test_no_aggregate_codes_in_mart(self, mart_builder: MartBuilder) -> None:
        """US, UW, ZZ and territory codes must not appear in the mart."""
        df = mart_builder.build(refresh=True)
        bad = {"US", "UW", "ZZ", "PR", "VI", "GU"}
        assert df["state_abbr"].isin(bad).sum() == 0

    def test_state_fips_populated(self, mart_builder: MartBuilder) -> None:
        """Every state must have a state_fips value."""
        df = mart_builder.build(refresh=True)
        assert df["state_fips"].notna().all()

    def test_derived_medicare_spend_per_capita(self, mart_builder: MartBuilder) -> None:
        """Hand-verify one state: CA has pop=1M and payment=1M → spend/cap=1.0."""
        df = mart_builder.build(refresh=True)
        ca = df[df["state_abbr"] == "CA"].iloc[0]
        # population=1_000_000, total_medicare_payment=1_000_000 → spend/cap=1.0
        assert abs(ca["medicare_spend_per_capita"] - 1.0) < 0.01

    def test_derived_providers_per_100k(self, mart_builder: MartBuilder) -> None:
        """One provider per 1M pop → 0.1 providers per 100k."""
        df = mart_builder.build(refresh=True)
        ca = df[df["state_abbr"] == "CA"].iloc[0]
        # 1 NPI / 1_000_000 * 100_000 = 0.1
        assert abs(ca["providers_per_100k"] - 0.1) < 0.01


# ===========================================================================
# Vintage-year (latest-non-null) rule
# ===========================================================================

class TestVintageRule:
    """MartBuilder must pick the most recent non-null CDI value per state."""

    def test_fl_gets_2022_vintage(self, mart_builder: MartBuilder) -> None:
        """FL: 2023 is null → should pick 2022 value (12.5) and tag vintage=2022."""
        df = mart_builder.build(refresh=True)
        fl = df[df["state_abbr"] == "FL"].iloc[0]
        assert abs(fl["diabetes_prevalence_pct"] - 12.5) < 0.001
        assert int(fl["diabetes_vintage"]) == 2022

    def test_non_fl_states_get_2023_vintage(self, mart_builder: MartBuilder) -> None:
        """States with a 2023 value should have vintage_year=2023."""
        df = mart_builder.build(refresh=True)
        non_fl = df[df["state_abbr"] == "CA"].iloc[0]
        assert int(non_fl["diabetes_vintage"]) == 2023


# ===========================================================================
# Exclusion of aggregate / foreign codes
# ===========================================================================

class TestExclusionRules:
    """Aggregate and foreign codes must be excluded before joining."""

    def test_us_row_excluded_from_cdi(self, mart_builder: MartBuilder) -> None:
        """'US' locationabbr in CDI mock must not appear as a state row."""
        df = mart_builder.build(refresh=True)
        assert "US" not in df["state_abbr"].values

    def test_zz_row_excluded_from_cms(self, mart_builder: MartBuilder) -> None:
        """'ZZ' state in CMS mock must not appear in the mart."""
        df = mart_builder.build(refresh=True)
        assert "ZZ" not in df["state_abbr"].values


# ===========================================================================
# quality_check
# ===========================================================================

class TestQualityCheck:
    """quality_check() must flag deliberately broken mart data."""

    def test_passes_on_clean_mart(self, mart_builder: MartBuilder) -> None:
        """A well-formed mart must get at least grade B."""
        df = mart_builder.build(refresh=True)
        sc = mart_builder.quality_check(df)
        assert sc["grade"] in {"A", "B"}

    def test_flags_negative_population(self, mart_builder: MartBuilder) -> None:
        """Negative population must fail the mart_population_positive check."""
        df = mart_builder.build(refresh=True)
        broken = df.copy()
        broken.loc[0, "population"] = -1  # inject bad value
        sc = mart_builder.quality_check(broken)
        pop_check = next(c for c in sc["checks"] if c["check"] == "mart_population_positive")
        assert pop_check["passed"] is False

    def test_flags_wrong_row_count(self, mart_builder: MartBuilder) -> None:
        """A mart with 50 rows must fail the row_count==51 check."""
        df = mart_builder.build(refresh=True)
        short = df.head(50)
        sc = mart_builder.quality_check(short)
        rc_check = next(c for c in sc["checks"] if c["check"] == "mart_row_count")
        assert rc_check["passed"] is False

    def test_flags_out_of_band_prevalence(self, mart_builder: MartBuilder) -> None:
        """Prevalence > 60 must fail the sanity-band check."""
        df = mart_builder.build(refresh=True)
        bad = df.copy()
        bad.loc[0, "diabetes_prevalence_pct"] = 99.0  # impossible value
        sc = mart_builder.quality_check(bad)
        band_check = next(
            c for c in sc["checks"] if "diabetes_prevalence_pct_band" in c["check"]
        )
        assert band_check["passed"] is False


# ===========================================================================
# insight_briefing — facts computation
# ===========================================================================

class TestInsightBriefing:
    """insight_briefing() must compute quadrant correctly on a synthetic mart."""

    def _make_simple_mart(self) -> pd.DataFrame:
        """4-state mart with known spend and diabetes values for quadrant test."""
        return pd.DataFrame({
            "state_abbr": ["AL", "AK", "AZ", "AR"],
            "population": [1_000_000] * 4,
            # Diabetes: AL=15 (high), AK=8 (low), AZ=14 (high), AR=7 (low)
            "diabetes_prevalence_pct": [15.0, 8.0, 14.0, 7.0],
            # Spend: AL=100 (low), AK=200 (high), AZ=90 (low), AR=250 (high)
            "medicare_spend_per_capita": [100.0, 200.0, 90.0, 250.0],
        })

    def test_quadrant_count(self) -> None:
        """AL and AZ have above-median diabetes AND below-median spend → 2 in quadrant."""
        mart = self._make_simple_mart()
        builder = MartBuilder(cache_dir=Path("/tmp"))  # cache_dir unused for briefing
        result = builder.insight_briefing(None, mart)
        facts = result["facts"]
        # Medians: diabetes=11 (mid between 8,14,15,7 sorted=7,8,14,15 → median=11)
        # Medians: spend=150 (mid between 90,100,200,250 sorted=90,100,200,250 → median=150)
        # High diabetes (>11): AL=15, AZ=14
        # Low spend (<150): AL=100, AZ=90
        # Both: AL, AZ → count=2
        assert facts["high_burden_low_spend_count"] == 2
        assert set(facts["high_burden_low_spend_states"]) == {"AL", "AZ"}

    def test_fallback_when_router_none(self) -> None:
        """router=None must use template text and generated_by='rule-based fallback'."""
        mart = self._make_simple_mart()
        builder = MartBuilder(cache_dir=Path("/tmp"))
        result = builder.insight_briefing(None, mart)
        assert result["generated_by"] == "rule-based fallback"
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 20

    def test_facts_dict_has_required_keys(self) -> None:
        """facts dict must contain all documented keys."""
        mart = self._make_simple_mart()
        builder = MartBuilder(cache_dir=Path("/tmp"))
        result = builder.insight_briefing(None, mart)
        required = {
            "top3_spend_per_capita", "bottom3_spend_per_capita",
            "top3_diabetes_prevalence", "corr_diabetes_spend",
            "high_burden_low_spend_states", "high_burden_low_spend_count",
        }
        assert required.issubset(result["facts"].keys())

    def test_ai_router_called_for_briefing(self) -> None:
        """When a router is provided, router.generate must be called."""
        mart = self._make_simple_mart()
        builder = MartBuilder(cache_dir=Path("/tmp"))

        mock_router = MagicMock()
        mock_router.generate.return_value = ("AI briefing text.", "ollama")

        result = builder.insight_briefing(mock_router, mart)
        mock_router.generate.assert_called_once()
        assert result["text"] == "AI briefing text."


# ===========================================================================
# Caching — second call must NOT re-fetch
# ===========================================================================

class TestCaching:
    """After build(refresh=True), build(refresh=False) must use the cache."""

    def test_second_build_does_not_call_extract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second build(refresh=False) must read from parquet cache, not call extract()."""
        call_counts: dict[str, int] = {"census": 0, "cdi": 0, "cms": 0}

        census_df = _make_census_frame()
        cdi_df = _make_cdi_frame()
        cms_df = _make_cms_frame()

        def mock_census_extract(self, **kw):
            call_counts["census"] += 1
            return census_df

        def mock_cdi_extract(self, **kw):
            call_counts["cdi"] += 1
            return cdi_df

        def mock_cms_extract(self, **kw):
            call_counts["cms"] += 1
            return cms_df

        monkeypatch.setattr("ingestion.census_source.CensusPopulationSource.extract", mock_census_extract)
        monkeypatch.setattr("ingestion.cdc_cdi_source.CDCChronicDiseaseSource.extract", mock_cdi_extract)
        monkeypatch.setattr("ingestion.cms_source.CMSMedicareSource.extract", mock_cms_extract)
        monkeypatch.setattr("ingestion.cms_source.CMSMedicareSource.connect", lambda self: True)
        monkeypatch.setattr(
            "ingestion.cms_source.CMSMedicareSource.get_metadata",
            lambda self: {"total_rows_in_dataset": 1_300_000},
        )

        builder = MartBuilder(cache_dir=tmp_path / "cache")

        # First build — populates cache
        builder.build(refresh=True)
        counts_after_first = dict(call_counts)

        # Second build — should read from cache, not call extract()
        builder.build(refresh=False)
        counts_after_second = dict(call_counts)

        # Counts must not have changed on the second call
        assert counts_after_second == counts_after_first, (
            f"Sources were called again on cache-hit build: "
            f"{counts_after_second} vs {counts_after_first}"
        )
