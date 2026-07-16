"""
tests/test_agents.py — Hermetic unit tests for all four agent classes.

No network calls, no live databases.  All tests use the sample fixtures
produced by conftest.py / make_fixtures.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agents.profiler import ProfilerAgent
from agents.transformer import TransformerAgent
from agents.quality_checker import QualityCheckerAgent
from agents.documenter import DocumenterAgent
from ingestion.file_source import FileSource

# Expected columns in the raw fixture (before transformation)
FIXTURE_COLS = {"patient_id", "age", "state", "diagnosis", "visit_date", "cost_usd"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(fixture_csv: Path) -> pd.DataFrame:
    """Return the raw 20-row test fixture DataFrame."""
    return FileSource().extract(filepath=str(fixture_csv))


# ===========================================================================
# ProfilerAgent
# ===========================================================================

class TestProfilerAgent:
    """Tests for ProfilerAgent.run() and detect_pii_columns()."""

    def test_profile_has_required_top_level_keys(self, fixture_csv: Path) -> None:
        """Profile dict must contain all expected top-level keys."""
        df = _load_fixture(fixture_csv)
        profile = ProfilerAgent().run(df, "test")
        required = {"dataset_name", "profiled_at", "overview", "columns",
                     "quality_issues", "correlations", "pii_columns"}
        assert required.issubset(profile.keys())

    def test_overview_row_count(self, fixture_csv: Path) -> None:
        """overview.row_count must equal the DataFrame length."""
        df = _load_fixture(fixture_csv)
        profile = ProfilerAgent().run(df, "test")
        assert profile["overview"]["row_count"] == len(df)

    def test_profile_has_column_entry_per_column(self, fixture_csv: Path) -> None:
        """Profile columns dict must have one entry per DataFrame column."""
        df = _load_fixture(fixture_csv)
        profile = ProfilerAgent().run(df, "test")
        assert set(profile["columns"].keys()) == set(df.columns)

    def test_null_count_diagnosis(self, fixture_csv: Path) -> None:
        """diagnosis column has 3 planted None values — profiler must count them."""
        df = _load_fixture(fixture_csv)
        profile = ProfilerAgent().run(df, "test")
        diagnosis_null = profile["columns"]["diagnosis"]["null_count"]
        assert diagnosis_null == 3

    def test_pii_detects_email_column(self) -> None:
        """A column named 'email_addr' with email values must be flagged as PII."""
        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "email_addr": ["alice@example.com", "bob@test.org", "carol@health.io"],
            "cost_usd": [100.0, 200.0, 300.0],
        })
        pii = ProfilerAgent().detect_pii_columns(df)
        flagged = {p["column"] for p in pii}
        assert "email_addr" in flagged

    def test_pii_does_not_flag_cost_usd(self) -> None:
        """cost_usd contains floats — must NOT be flagged as PII."""
        df = pd.DataFrame({
            "email_addr": ["alice@example.com", "bob@test.org"],
            "cost_usd": [100.0, 200.0],
        })
        pii = ProfilerAgent().detect_pii_columns(df)
        flagged = {p["column"] for p in pii}
        assert "cost_usd" not in flagged

    def test_pii_detects_ssn_by_value_pattern(self) -> None:
        """A column with SSN-formatted values must be detected via value pattern."""
        df = pd.DataFrame({
            "record_number": ["123-45-6789", "987-65-4321", "111-22-3333"],
        })
        pii = ProfilerAgent().detect_pii_columns(df)
        # 'record_number' has no name hint but values match SSN pattern
        flagged = {p["column"] for p in pii}
        assert "record_number" in flagged

    def test_pii_column_output_has_required_keys(self) -> None:
        """Each PII finding dict must have column/reason/confidence keys."""
        df = pd.DataFrame({
            "email": ["alice@example.com", "bob@test.org"],
        })
        pii = ProfilerAgent().detect_pii_columns(df)
        assert len(pii) >= 1
        for entry in pii:
            assert "column" in entry
            assert "reason" in entry
            assert "confidence" in entry

    def test_pii_included_in_profile_dict(self, fixture_csv: Path) -> None:
        """pii_columns key must be present in the profile dict."""
        df = _load_fixture(fixture_csv)
        profile = ProfilerAgent().run(df, "test")
        assert "pii_columns" in profile
        assert isinstance(profile["pii_columns"], list)


# ===========================================================================
# TransformerAgent
# ===========================================================================

class TestTransformerAgent:
    """Tests for TransformerAgent.run()."""

    def test_dedup_reduces_20_to_19(self, fixture_csv: Path) -> None:
        """Fixture has 1 duplicate row (row 20 = row 19) → 19 rows after dedup."""
        df = _load_fixture(fixture_csv)
        assert len(df) == 20  # confirm fixture starts at 20
        clean = TransformerAgent().run(df, "test")
        assert len(clean) == 19

    def test_snake_case_columns_preserved(self, fixture_csv: Path) -> None:
        """Fixture columns are already snake_case — they must survive transformation."""
        df = _load_fixture(fixture_csv)
        clean = TransformerAgent().run(df, "test")
        # Original columns should all appear (possibly lower-cased, already snake)
        for col in FIXTURE_COLS:
            assert col in clean.columns

    def test_negative_age_survives_transform(self, fixture_csv: Path) -> None:
        """Transformer does not fix value ranges — age=-5 must survive."""
        df = _load_fixture(fixture_csv)
        clean = TransformerAgent().run(df, "test")
        # After dedup, age=-5 row should still be present
        assert (clean["age"] < 0).any(), "Negative age should survive transformation"

    def test_metadata_columns_added(self, fixture_csv: Path) -> None:
        """_loaded_at and _source columns must be added by the transformer."""
        df = _load_fixture(fixture_csv)
        clean = TransformerAgent().run(df, "test")
        assert "_loaded_at" in clean.columns
        assert "_source" in clean.columns

    def test_transform_log_non_empty(self, fixture_csv: Path) -> None:
        """get_transform_summary() must return at least one step."""
        df = _load_fixture(fixture_csv)
        transformer = TransformerAgent()
        transformer.run(df, "test")
        log = transformer.get_transform_summary()
        assert isinstance(log, list)
        assert len(log) >= 1

    def test_transform_log_has_step_fields(self, fixture_csv: Path) -> None:
        """Each log entry must have step/action/detail/timestamp keys."""
        df = _load_fixture(fixture_csv)
        transformer = TransformerAgent()
        transformer.run(df, "test")
        for entry in transformer.get_transform_summary():
            assert "step" in entry
            assert "action" in entry
            assert "detail" in entry
            assert "timestamp" in entry


# ===========================================================================
# QualityCheckerAgent
# ===========================================================================

class TestQualityCheckerAgent:
    """Tests for QualityCheckerAgent.run()."""

    def _clean_fixture(self, fixture_csv: Path) -> pd.DataFrame:
        """Return the transformed fixture (19 rows, nulls filled)."""
        raw = FileSource().extract(filepath=str(fixture_csv))
        return TransformerAgent().run(raw, "test")

    def test_scorecard_has_required_keys(self, fixture_csv: Path) -> None:
        """Scorecard must have score/grade/checks/total_checks keys."""
        clean = self._clean_fixture(fixture_csv)
        sc = QualityCheckerAgent().run(clean, "test")
        required = {"score", "grade", "checks", "total_checks",
                     "checks_passed", "checks_failed"}
        assert required.issubset(sc.keys())

    def test_grade_is_valid_letter(self, fixture_csv: Path) -> None:
        """Grade must be one of A, B, C, F."""
        clean = self._clean_fixture(fixture_csv)
        sc = QualityCheckerAgent().run(clean, "test")
        assert sc["grade"] in {"A", "B", "C", "F"}

    def test_score_is_percentage(self, fixture_csv: Path) -> None:
        """Score must be a float between 0 and 100."""
        clean = self._clean_fixture(fixture_csv)
        sc = QualityCheckerAgent().run(clean, "test")
        assert 0.0 <= sc["score"] <= 100.0

    def test_negative_age_fails_check(self, fixture_csv: Path) -> None:
        """no_negatives_age check must fail because age=-5 survives transformation."""
        clean = self._clean_fixture(fixture_csv)
        sc = QualityCheckerAgent().run(clean, "test")
        age_neg_check = next(
            (c for c in sc["checks"] if c["check"] == "no_negatives_age"), None
        )
        assert age_neg_check is not None, "no_negatives_age check must be present"
        assert age_neg_check["passed"] is False

    def test_checks_passed_plus_failed_equals_total(self, fixture_csv: Path) -> None:
        """checks_passed + checks_failed must equal total_checks."""
        clean = self._clean_fixture(fixture_csv)
        sc = QualityCheckerAgent().run(clean, "test")
        assert sc["checks_passed"] + sc["checks_failed"] == sc["total_checks"]


# ===========================================================================
# DocumenterAgent
# ===========================================================================

class TestDocumenterAgent:
    """Tests for DocumenterAgent.run() — must work with zero AI dependencies."""

    def test_documenter_runs_without_ai(self, fixture_csv: Path) -> None:
        """DocumenterAgent must complete without any AI/LLM calls."""
        raw = FileSource().extract(filepath=str(fixture_csv))
        clean = TransformerAgent().run(raw, "test")
        # If core.llm is still imported this will raise ImportError — that's a bug.
        docs = DocumenterAgent().run(clean, "test_fixture")
        assert docs is not None

    def test_data_dictionary_one_entry_per_column(self, fixture_csv: Path) -> None:
        """data_dictionary must have one entry per column in the DataFrame."""
        raw = FileSource().extract(filepath=str(fixture_csv))
        clean = TransformerAgent().run(raw, "test")
        docs = DocumenterAgent().run(clean, "test_fixture")
        dd = docs["data_dictionary"]
        assert len(dd) == len(clean.columns)

    def test_data_dictionary_entry_keys(self, fixture_csv: Path) -> None:
        """Each data dictionary entry must have the required keys."""
        raw = FileSource().extract(filepath=str(fixture_csv))
        clean = TransformerAgent().run(raw, "test")
        docs = DocumenterAgent().run(clean, "test_fixture")
        required = {"column_name", "data_type", "nullable", "null_count",
                     "unique_count", "sample_values", "description"}
        for entry in docs["data_dictionary"]:
            assert required.issubset(entry.keys())

    def test_docs_has_all_sections(self, fixture_csv: Path) -> None:
        """Docs dict must contain data_dictionary, schema, lineage, quality_summary."""
        raw = FileSource().extract(filepath=str(fixture_csv))
        clean = TransformerAgent().run(raw, "test")
        docs = DocumenterAgent().run(clean, "test_fixture")
        required_sections = {
            "dataset_name", "data_dictionary", "schema", "lineage",
            "quality_summary", "usage_notes",
        }
        assert required_sections.issubset(docs.keys())
