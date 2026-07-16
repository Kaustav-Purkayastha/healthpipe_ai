"""
tests/test_mart_upgrades.py — Offline tests for the validation-mission mart upgrades.

Covers:
  5b — measure catalog parsing + slug rules; MartBuilder per-measure column
       generation incl. the <40-state warn path; build metadata.
  5c — CDC PLACES source: mixed-case datavaluetypeid trap, string FIPS, US exclusion.
All HTTP is mocked — no network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.config import STATE_ABBRS


# ===========================================================================
# 5c — CDC PLACES source
# ===========================================================================

class TestPlacesSource:
    """CDCPlacesSource cleans PLACES rows and respects the mixed-case trap."""

    def _raw_page(self) -> list[dict]:
        # Mixed-case datavaluetypeid is the trap (CrdPrv / AgeAdjPrv), unlike CDI.
        return [
            {"year": "2023", "stateabbr": "AR", "locationname": "Drew",
             "locationid": "05043", "measureid": "DIABETES",  # leading zero!
             "datavaluetypeid": "CrdPrv", "data_value": "12.5",
             "totalpop18plus": "3500000"},
            {"year": "2023", "stateabbr": "TX", "locationname": "Loving",
             "locationid": "48301", "measureid": "DIABETES",
             "datavaluetypeid": "CrdPrv", "data_value": "",   # suppressed → NaN
             "totalpop18plus": "64"},
            {"year": "2023", "stateabbr": "US", "locationname": "United States",
             "locationid": "59",    "measureid": "DIABETES",
             "datavaluetypeid": "CrdPrv", "data_value": "11.0",
             "totalpop18plus": "250000000"},  # national roll-up → must be dropped
        ]

    def test_clean_types_and_exclusions(self) -> None:
        from ingestion.places_source import CDCPlacesSource
        df = CDCPlacesSource._clean(pd.DataFrame(self._raw_page()))
        # 'US' roll-up dropped.
        assert (df["stateabbr"] == "US").sum() == 0
        assert len(df) == 2
        # locationid stays a string with leading zeros intact (FIPS "05043").
        assert df["locationid"].tolist() == ["05043", "48301"]
        assert all(isinstance(v, str) for v in df["locationid"])
        # data_value → float, empty string → NaN.
        assert df.loc[df["locationname"] == "Drew", "data_value"].iloc[0] == 12.5
        assert pd.isna(df.loc[df["locationname"] == "Loving", "data_value"].iloc[0])
        # totalpop18plus → nullable int.
        assert df["totalpop18plus"].tolist() == [3_500_000, 64]

    def test_extract_builds_where_and_no_year_filter(self) -> None:
        """extract must filter state/measure/type + exclude US, and NOT filter year."""
        from ingestion.places_source import CDCPlacesSource

        captured: dict = {}

        def _fake_get(url, params=None, timeout=None):
            captured["params"] = params
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = lambda: None
            resp.json = lambda: self._raw_page() if params.get("$offset", 0) == 0 else []
            return resp

        with patch("ingestion.places_source.requests.get", _fake_get):
            df = CDCPlacesSource().extract(
                state_abbr="TX", measure_id="DIABETES", data_value_type="CrdPrv"
            )

        where = captured["params"]["$where"]
        assert "stateabbr != 'US'" in where
        assert "stateabbr='TX'" in where
        assert "measureid='DIABETES'" in where
        assert "datavaluetypeid='CrdPrv'" in where
        assert "year" not in where.lower()  # data year is 2023, not the 2025 release
        assert len(df) == 2  # US row excluded

    def test_registered_in_registry(self) -> None:
        from ingestion.registry import SourceRegistry
        reg = SourceRegistry()
        src = reg.get("cdc_places")
        assert src is not None
        assert src.source_type == "api"


# ===========================================================================
# 5b — measure catalog
# ===========================================================================

class TestMeasureCatalog:
    """get_available_measures parses the CDI $group response and slugs measures."""

    def test_measure_slug_defaults_and_custom(self) -> None:
        from analytics.measure_catalog import measure_slug
        assert measure_slug("DIA01") == "diabetes"
        assert measure_slug("NPW14") == "obesity"
        assert measure_slug("TOB04") == "smoking"
        assert measure_slug("AST01") == "ast01"  # non-default → lowercased id

    def test_catalog_parses_live_response(self, tmp_path, monkeypatch) -> None:
        from analytics import measure_catalog
        monkeypatch.setattr(measure_catalog, "CACHE_DIR", tmp_path)

        payload = [
            {"questionid": "DIA01", "question": "Diabetes", "topic": "Diabetes"},
            {"questionid": "AST01", "question": "Asthma", "topic": "Asthma"},
        ]

        def _fake_get(url, params=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = lambda: None
            resp.json = lambda: payload
            return resp

        monkeypatch.setattr("analytics.measure_catalog.requests.get", _fake_get)
        df = measure_catalog.get_available_measures(refresh=True)
        assert set(df["questionid"]) == {"DIA01", "AST01"}
        assert {"questionid", "question", "topic"}.issubset(df.columns)
        # Cached to parquet for offline reuse.
        assert (tmp_path / "cdi_measure_catalog.parquet").exists()

    def test_catalog_only_offers_crude_prevalence_measures(self, tmp_path, monkeypatch) -> None:
        """The picker must only offer measures pull_cdi_measure() can actually fetch.

        pull_cdi_measure() hardcodes datavaluetypeid='CRDPREV'. Rate/count-only
        indicators (e.g. CAN02 breast cancer mortality, reported as a per-100k
        rate with no Crude Prevalence value) would silently pull back zero rows
        if offered — the mart column would be 100% NaN with no visible error.
        The catalog query must filter server-side on CRDPREV so this can't happen.
        """
        from analytics import measure_catalog
        monkeypatch.setattr(measure_catalog, "CACHE_DIR", tmp_path)

        captured: dict = {}

        def _fake_get(url, params=None, timeout=None):
            captured["params"] = params
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = lambda: None
            resp.json = lambda: [
                {"questionid": "DIA01", "question": "Diabetes", "topic": "Diabetes"},
            ]
            return resp

        monkeypatch.setattr("analytics.measure_catalog.requests.get", _fake_get)
        measure_catalog.get_available_measures(refresh=True)
        assert captured["params"]["$where"] == "datavaluetypeid='CRDPREV'"

    def test_catalog_offline_fallback(self, tmp_path, monkeypatch) -> None:
        """API down + no cache → the three verified defaults, never a crash."""
        from analytics import measure_catalog
        monkeypatch.setattr(measure_catalog, "CACHE_DIR", tmp_path)

        def _boom(url, params=None, timeout=None):
            raise ConnectionError("offline")

        monkeypatch.setattr("analytics.measure_catalog.requests.get", _boom)
        df = measure_catalog.get_available_measures(refresh=True)
        assert set(df["questionid"]) == {"DIA01", "NPW14", "TOB04"}


# ===========================================================================
# 5b — MartBuilder measure generalization + 5a build metadata
# ===========================================================================

def _census_frame() -> pd.DataFrame:
    return pd.DataFrame([{"state_abbr": s, "population": 1_000_000} for s in STATE_ABBRS])


def _cdi_frame(states: list[str]) -> pd.DataFrame:
    """Raw CDI-shaped frame (one 2023 row per given state)."""
    return pd.DataFrame([
        {"locationabbr": s, "yearstart": "2023", "datavalue": 10.0,
         "is_suppressed": False}
        for s in states
    ])


def _cms_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"state": s, "npi": f"{1000000000 + i}", "total_medicare_payment": 1_000_000.0}
         for i, s in enumerate(STATE_ABBRS)]
    )


@pytest.fixture
def builder(tmp_path, monkeypatch):
    """MartBuilder with mocked sources; CDI coverage varies by question id."""
    from analytics.mart_builder import MartBuilder

    monkeypatch.setattr(
        "ingestion.census_source.CensusPopulationSource.extract",
        lambda self, **kw: _census_frame(),
    )

    def _cdi_extract(self, question_id=None, **kw):
        # SPARSE1 covers only 10 states → triggers the <40 warn path.
        states = STATE_ABBRS if question_id != "SPARSE1" else STATE_ABBRS[:10]
        return _cdi_frame(states)

    monkeypatch.setattr(
        "ingestion.cdc_cdi_source.CDCChronicDiseaseSource.extract", _cdi_extract
    )
    monkeypatch.setattr(
        "ingestion.cms_source.CMSMedicareSource.extract", lambda self, **kw: _cms_frame()
    )
    monkeypatch.setattr("ingestion.cms_source.CMSMedicareSource.connect", lambda self: True)
    monkeypatch.setattr(
        "ingestion.cms_source.CMSMedicareSource.get_metadata",
        lambda self: {"total_rows_in_dataset": 1_300_000},
    )
    return MartBuilder(cache_dir=tmp_path / "cache")


class TestMartMeasures:
    """build() generalizes to any measure set and records build metadata."""

    def test_default_measures_keep_friendly_columns(self, builder) -> None:
        df = builder.build(refresh=True)
        for col in ["diabetes_prevalence_pct", "obesity_prevalence_pct",
                    "smoking_prevalence_pct", "diabetes_vintage"]:
            assert col in df.columns, f"missing backward-compat column {col}"

    def test_custom_measure_uses_slug_column(self, builder) -> None:
        df = builder.build(refresh=True, measures=["DIA01", "AST01"])
        assert "diabetes_prevalence_pct" in df.columns  # default keeps friendly name
        assert "ast01_prevalence_pct" in df.columns      # non-default → slug
        assert "obesity_prevalence_pct" not in df.columns  # not selected

    def test_build_metadata_populated(self, builder) -> None:
        builder.build(refresh=True)
        meta = builder.build_metadata
        assert meta["measures"] == ["DIA01", "NPW14", "TOB04"]
        assert meta["sample_mode"] is True          # full_cms defaulted False
        assert meta["census_rows"] == 51
        assert meta["cms_rows"] == 51                # mock: one provider per state
        assert meta["cdc_rows"] == 51 * 3            # three measures × 51 states
        assert "built_at" in meta and meta["built_at"]

    def test_low_coverage_measure_flagged(self, builder) -> None:
        """A measure covering <40 states lands in low_coverage_measures."""
        builder.build(refresh=True, measures=["DIA01", "SPARSE1"])
        low = builder.build_metadata["low_coverage_measures"]
        assert any(m["questionid"] == "SPARSE1" for m in low)
        assert all(m["questionid"] != "DIA01" for m in low)  # 51 states → not flagged

    def test_to_duckdb_writes_build_meta(self, builder, tmp_path, monkeypatch) -> None:
        import analytics.mart_builder as mb
        monkeypatch.setattr(mb, "_MART_DIR", tmp_path / "mart")
        monkeypatch.setattr(mb, "DATABASE_PATH", tmp_path / "hp.duckdb")
        df = builder.build(refresh=True)
        builder.to_duckdb(df)
        meta_path = tmp_path / "mart" / "build_meta.json"
        assert meta_path.exists()
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["measures"] == ["DIA01", "NPW14", "TOB04"]
        assert meta["cdc_rows"] == 153

    def test_insight_briefing_survives_missing_diabetes(self, builder) -> None:
        """A measure set without diabetes must not crash insight_briefing."""
        df = builder.build(refresh=True, measures=["NPW14"])  # obesity only
        result = builder.insight_briefing(None, df)
        assert result["generated_by"] == "rule-based fallback"
        assert result["facts"]["high_burden_low_spend_count"] == 0
