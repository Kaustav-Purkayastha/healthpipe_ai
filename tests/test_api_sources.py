"""
tests/test_api_sources.py — Offline unit tests for all six API sources (Step 7).

ALL HTTP calls are mocked via unittest.mock.patch("requests.get").
No internet connection required. Live-API tests are marked @pytest.mark.network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from ingestion.cms_source import CMSMedicareSource
from ingestion.cdc_cdi_source import CDCChronicDiseaseSource
from ingestion.brfss_source import BRFSSSource
from ingestion.census_source import CensusPopulationSource
from ingestion.who_source import WHOSource
from ingestion.openfda_source import OpenFDASource


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_resp(status: int = 200, body=None) -> MagicMock:
    """Return a MagicMock response with .status_code and .json()."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body if body is not None else {}
    if status >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ===========================================================================
# CMSMedicareSource
# ===========================================================================

class TestCMSMedicareSource:
    """Tests for CMSMedicareSource connect/extract."""

    _STATS_BODY = {"total": 1_300_000}

    _ROW = {
        "Rndrng_NPI": "1234567890",
        "Rndrng_Prvdr_Last_Org_Name": "SMITH",
        "Rndrng_Prvdr_Type": "Internal Medicine",
        "Rndrng_Prvdr_State_Abrvtn": "MD",
        "Rndrng_Prvdr_State_FIPS": "24",
        "Rndrng_Prvdr_Cntry": "US",
        "Tot_Benes": "150",
        "Tot_Srvcs": "300",
        "Tot_Mdcr_Pymt_Amt": "12000.00",
        "Tot_Mdcr_Stdzd_Amt": "11500.00",
    }

    def test_connect_returns_true_and_caches_total(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, self._STATS_BODY)):
            src = CMSMedicareSource()
            assert src.connect() is True
            assert src._total_rows == 1_300_000

    def test_connect_false_on_error(self) -> None:
        import requests as rq
        with patch("requests.get", side_effect=rq.exceptions.ConnectionError("refused")):
            assert CMSMedicareSource().connect() is False

    def test_metadata_includes_total_rows(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, self._STATS_BODY)):
            src = CMSMedicareSource()
            src.connect()
            meta = src.get_metadata()
            assert meta["total_rows_in_dataset"] == 1_300_000

    def test_extract_paginates_and_stops_on_short_page(self) -> None:
        full_page = [self._ROW.copy() for _ in range(3)]
        short_page = [self._ROW.copy()]  # shorter than page size → stop

        # Patch CMS_PAGE_SIZE to 3 so 'full' and 'short' make sense in the test
        with patch("ingestion.cms_source.CMS_PAGE_SIZE", 3), \
             patch("requests.get", side_effect=[
                 _mock_resp(200, full_page),
                 _mock_resp(200, short_page),
             ]):
            src = CMSMedicareSource()
            df = src.extract(max_records=100)

        assert len(df) == 4

    def test_extract_excludes_zz_rows(self) -> None:
        zz_row = {**self._ROW, "Rndrng_Prvdr_Cntry": "ZZ", "Rndrng_Prvdr_State_Abrvtn": "ZZ"}
        page = [self._ROW.copy(), zz_row]
        with patch("requests.get", return_value=_mock_resp(200, page)):
            df = CMSMedicareSource().extract(max_records=100)
        # ZZ row must be excluded
        assert "ZZ" not in df.get("country", pd.Series()).tolist()
        if "country" in df.columns:
            assert all(df["country"] == "US")

    def test_extract_renames_columns(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW.copy()])):
            df = CMSMedicareSource().extract(max_records=10)
        assert "npi" in df.columns
        assert "provider_specialty" in df.columns
        assert "state" in df.columns
        assert "total_medicare_payment" in df.columns

    def test_extract_casts_numerics(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW.copy()])):
            df = CMSMedicareSource().extract(max_records=10)
        assert pd.api.types.is_float_dtype(df["total_medicare_payment"]) or \
               pd.api.types.is_numeric_dtype(df["total_medicare_payment"])

    def test_extract_keeps_npi_as_string(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW.copy()])):
            df = CMSMedicareSource().extract(max_records=10)
        # String dtype (object or StringDtype in pandas 3) — not numeric
        assert pd.api.types.is_string_dtype(df["npi"]) or df["npi"].dtype == object
        assert df["npi"].iloc[0] == "1234567890"

    def test_extract_state_filter_in_params(self) -> None:
        calls_made: list = []

        def _capture(url, params=None, **kw):
            calls_made.append(params or {})
            return _mock_resp(200, [self._ROW.copy()])

        with patch("requests.get", side_effect=_capture):
            CMSMedicareSource().extract(state="MD", max_records=5)

        assert any("filter[Rndrng_Prvdr_State_Abrvtn]" in p for p in calls_made)
        assert any(p.get("filter[Rndrng_Prvdr_State_Abrvtn]") == "MD" for p in calls_made)

    def test_extract_returns_empty_on_no_records(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, [])):
            df = CMSMedicareSource().extract(max_records=100)
        assert df.empty


# ===========================================================================
# CDCChronicDiseaseSource
# ===========================================================================

class TestCDCChronicDiseaseSource:
    """Tests for CDCChronicDiseaseSource connect/extract."""

    _ROW = {
        "yearstart": "2023",
        "locationabbr": "MD",
        "locationid": "24",
        "topic": "Diabetes",
        "questionid": "DIA01",
        "question": "Prevalence of diagnosed diabetes among adults",
        "datavaluetype": "Crude Prevalence",
        "datavaluetypeid": "CRDPREV",
        "datavalue": "10.5",
        "datavalueunit": "%",
        "stratificationcategory1": "Overall",
        "stratification1": "Overall",
        "datavaluefootnotesymbol": "",
    }

    def test_connect_true_on_200(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW])):
            assert CDCChronicDiseaseSource().connect() is True

    def test_where_contains_question_id(self) -> None:
        captured: list = []

        def _cap(url, params=None, **kw):
            captured.append(params or {})
            return _mock_resp(200, [])

        with patch("requests.get", side_effect=_cap):
            CDCChronicDiseaseSource().extract(question_id="DIA01", max_records=10)

        assert any("DIA01" in str(p.get("$where", "")) for p in captured)

    def test_where_contains_year(self) -> None:
        captured: list = []

        def _cap(url, params=None, **kw):
            captured.append(params or {})
            return _mock_resp(200, [])

        with patch("requests.get", side_effect=_cap):
            CDCChronicDiseaseSource().extract(year="2023", max_records=10)

        assert any("2023" in str(p.get("$where", "")) for p in captured)

    def test_datavalue_cast_to_float(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW.copy()])):
            df = CDCChronicDiseaseSource().extract(max_records=10)
        assert pd.api.types.is_float_dtype(df["datavalue"])

    def test_is_suppressed_flag(self) -> None:
        suppressed_row = {**self._ROW, "datavalue": "", "datavaluefootnotesymbol": "*"}
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW.copy(), suppressed_row])):
            df = CDCChronicDiseaseSource().extract(max_records=10)
        assert "is_suppressed" in df.columns
        assert df["is_suppressed"].sum() == 1

    def test_excludes_us_location(self) -> None:
        us_row = {**self._ROW, "locationabbr": "US"}
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW.copy(), us_row])):
            df = CDCChronicDiseaseSource().extract(max_records=10)
        assert "US" not in df["locationabbr"].tolist()

    def test_pagination_stops_on_short_page(self) -> None:
        full_page = [self._ROW.copy() for _ in range(3)]
        short_page = [self._ROW.copy()]  # short page → stop

        # Patch SOCRATA_PAGE_SIZE so 3 rows counts as a full page
        with patch("ingestion.cdc_cdi_source.SOCRATA_PAGE_SIZE", 3), \
             patch("requests.get", side_effect=[
                 _mock_resp(200, full_page),
                 _mock_resp(200, short_page),
             ]):
            df = CDCChronicDiseaseSource().extract(max_records=100)

        assert len(df) == 4

    def test_class_constants_correct(self) -> None:
        assert CDCChronicDiseaseSource.DIABETES == "DIA01"
        assert CDCChronicDiseaseSource.OBESITY == "NPW14"
        assert CDCChronicDiseaseSource.SMOKING == "TOB04"


# ===========================================================================
# BRFSSSource
# ===========================================================================

class TestBRFSSSource:
    """Tests for BRFSSSource connect/extract."""

    _ROW = {
        "year": "2023",
        "locationabbr": "MD",
        "locationdesc": "Maryland",
        "topic": "BMI Categories",
        "response": "Obese (BMI 30.0 - 99.8)",
        "break_out": "Overall",
        "break_out_category": "Overall",
        "data_value": "31.5",
        "data_value_type": "Value",
        "confidence_limit_low": "29.8",
        "confidence_limit_high": "33.2",
        "sample_size": "1200",
    }

    def test_connect_true_on_200(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW])):
            assert BRFSSSource().connect() is True

    def test_where_contains_topic(self) -> None:
        captured: list = []

        def _cap(url, params=None, **kw):
            captured.append(params or {})
            return _mock_resp(200, [])

        with patch("requests.get", side_effect=_cap):
            BRFSSSource().extract(topic="BMI Categories", max_records=10)

        assert any("BMI Categories" in str(p.get("$where", "")) for p in captured)

    def test_where_contains_year(self) -> None:
        captured: list = []

        def _cap(url, params=None, **kw):
            captured.append(params or {})
            return _mock_resp(200, [])

        with patch("requests.get", side_effect=_cap):
            BRFSSSource().extract(year="2023", max_records=10)

        assert any("2023" in str(p.get("$where", "")) for p in captured)

    def test_overall_only_adds_break_out_filter(self) -> None:
        captured: list = []

        def _cap(url, params=None, **kw):
            captured.append(params or {})
            return _mock_resp(200, [])

        with patch("requests.get", side_effect=_cap):
            BRFSSSource().extract(overall_only=True, max_records=10)

        assert any("break_out='Overall'" in str(p.get("$where", "")) for p in captured)

    def test_excludes_us_and_uw(self) -> None:
        us_row = {**self._ROW, "locationabbr": "US"}
        uw_row = {**self._ROW, "locationabbr": "UW"}
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW.copy(), us_row, uw_row])):
            df = BRFSSSource().extract(max_records=100)
        assert "US" not in df["locationabbr"].tolist()
        assert "UW" not in df["locationabbr"].tolist()

    def test_data_value_cast_to_float(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, [self._ROW.copy()])):
            df = BRFSSSource().extract(max_records=10)
        assert pd.api.types.is_float_dtype(df["data_value"])


# ===========================================================================
# CensusPopulationSource
# ===========================================================================

class TestCensusPopulationSource:
    """Tests for CensusPopulationSource connect/extract."""

    # Realistic Census API response: header row + 3 data rows (incl. PR)
    _CENSUS_BODY = [
        ["NAME", "B01003_001E", "state"],
        ["Alabama", "5108468", "01"],
        ["Alaska", "733583", "02"],
        ["Puerto Rico", "3250897", "72"],  # not in FIPS_TO_STATE → must be dropped
    ]

    def test_connect_false_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """connect() must return False and warn when CENSUS_API_KEY is empty."""
        monkeypatch.setattr("ingestion.census_source.CENSUS_API_KEY", "")
        assert CensusPopulationSource().connect() is False

    def test_connect_true_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ingestion.census_source.CENSUS_API_KEY", "fake-key")
        with patch("requests.get", return_value=_mock_resp(200, [["NAME", "B01003_001E", "state"], ["Alabama", "5108468", "01"]])):
            assert CensusPopulationSource().connect() is True

    def test_header_row_not_in_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """First row of Census response is a header — must not appear as data."""
        monkeypatch.setattr("ingestion.census_source.CENSUS_API_KEY", "fake-key")
        with patch("requests.get", return_value=_mock_resp(200, self._CENSUS_BODY)):
            df = CensusPopulationSource().extract()
        # "NAME" should appear as a column, not as a state_name value
        assert "NAME" not in df["state_name"].tolist()
        assert "Alabama" in df["state_name"].tolist()

    def test_drops_non_fips_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Puerto Rico (FIPS 72) is not in FIPS_TO_STATE — must be dropped."""
        monkeypatch.setattr("ingestion.census_source.CENSUS_API_KEY", "fake-key")
        with patch("requests.get", return_value=_mock_resp(200, self._CENSUS_BODY)):
            df = CensusPopulationSource().extract()
        assert "Puerto Rico" not in df["state_name"].tolist()
        # Only Alabama (01) and Alaska (02) should remain
        assert len(df) == 2

    def test_population_is_integer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """population column must be numeric (int/Int64), not string."""
        monkeypatch.setattr("ingestion.census_source.CENSUS_API_KEY", "fake-key")
        with patch("requests.get", return_value=_mock_resp(200, self._CENSUS_BODY)):
            df = CensusPopulationSource().extract()
        assert pd.api.types.is_integer_dtype(df["population"])

    def test_state_fips_stays_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """state_fips must remain a string to preserve leading zeros."""
        monkeypatch.setattr("ingestion.census_source.CENSUS_API_KEY", "fake-key")
        with patch("requests.get", return_value=_mock_resp(200, self._CENSUS_BODY)):
            df = CensusPopulationSource().extract()
        # Check string dtype (pandas 3 may return StringDtype not object)
        assert pd.api.types.is_string_dtype(df["state_fips"]) or df["state_fips"].dtype == object
        # Leading zero must be preserved
        assert "01" in df["state_fips"].tolist()

    def test_state_abbr_added(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """state_abbr column must be added via FIPS crosswalk."""
        monkeypatch.setattr("ingestion.census_source.CENSUS_API_KEY", "fake-key")
        with patch("requests.get", return_value=_mock_resp(200, self._CENSUS_BODY)):
            df = CensusPopulationSource().extract()
        assert "state_abbr" in df.columns
        assert "AL" in df["state_abbr"].tolist()
        assert "AK" in df["state_abbr"].tolist()

    def test_extract_skips_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """extract() with no key must return empty DataFrame (not crash)."""
        monkeypatch.setattr("ingestion.census_source.CENSUS_API_KEY", "")
        df = CensusPopulationSource().extract()
        assert df.empty


# ===========================================================================
# Registry auto-registration
# ===========================================================================

class TestRegistryAutoRegister:
    """All 6 API sources must be registered after SourceRegistry() init."""

    def test_all_api_sources_registered(self) -> None:
        from ingestion.registry import SourceRegistry
        r = SourceRegistry()
        names = {s["name"] for s in r.list_sources()}
        # cdc_places added by the validation-mission county drill-down upgrade (5c).
        expected = {"file", "who", "openfda", "cms_medicare", "cdc_cdi",
                    "cdc_brfss", "census", "cdc_places"}
        assert expected == names

    def test_list_sources_returns_all_metadata(self) -> None:
        from ingestion.registry import SourceRegistry
        r = SourceRegistry()
        assert len(r.list_sources()) == 8  # file + 6 API sources + cdc_places


# ===========================================================================
# WHOSource (spot checks — v1 logic preserved)
# ===========================================================================

class TestWHOSourceAdapted:
    """Spot checks that the v1 logic survived the v2 adaptation."""

    def test_connect_true_on_200(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, {"value": []})):
            assert WHOSource().connect() is True

    def test_extract_renames_spatialdim(self) -> None:
        row = {"SpatialDim": "IND", "TimeDim": "2020", "NumericValue": 70.5}
        page = {"value": [row]}
        with patch("requests.get", return_value=_mock_resp(200, page)):
            df = WHOSource().extract(indicator="WHOSIS_000001", max_records=5)
        assert "country_code" in df.columns


# ===========================================================================
# OpenFDASource (spot checks — v1 logic preserved)
# ===========================================================================

class TestOpenFDASourceAdapted:
    """Spot checks that the v1 logic survived the v2 adaptation."""

    _EVENT = {
        "safetyreportid": "123",
        "receivedate": "20230101",
        "serious": "1",
        "patient": {
            "patientonsetage": "45",
            "patientsex": "1",
            "drug": [{"medicinalproduct": "Aspirin", "openfda": {"brand_name": ["Aspirin"]}}],
            "reaction": [{"reactionmeddrapt": "Headache"}],
        },
    }

    def test_connect_true_on_200(self) -> None:
        with patch("requests.get", return_value=_mock_resp(200, {"results": []})):
            assert OpenFDASource().connect() is True

    def test_extract_flattens_event(self) -> None:
        body = {"results": [self._EVENT]}
        with patch("requests.get", return_value=_mock_resp(200, body)):
            df = OpenFDASource().extract(search_term="aspirin", max_records=5)
        assert "safety_report_id" in df.columns
        assert "reactions" in df.columns
