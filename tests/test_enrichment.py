"""
tests/test_enrichment.py — Offline tests for core.enrich (Step 5 AI enrichment).

All router calls are mocked — no Ollama or Gemini connection needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from core.enrich import (
    _DESCRIPTION_BATCH_SIZE,
    _rule_based_description,
    describe_columns,
    explain_issues,
    generate_briefing,
)
from ingestion.file_source import FileSource


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock_router(response: str | None = "AI generated text") -> MagicMock:
    """Return a mock AIRouter whose generate() always returns (response, 'ollama')."""
    router = MagicMock()
    router.generate.return_value = (response, "ollama")
    return router


def _fake_profile(row_count: int = 20, col_count: int = 6) -> dict:
    return {
        "overview": {
            "row_count": row_count,
            "column_count": col_count,
            "completeness_score": 92.5,
            "duplicate_rows": 1,
            "duplicate_percentage": 5.0,
        },
        "columns": {},
        "quality_issues": [],
        "pii_columns": [],
    }


def _fake_scorecard(grade: str = "A", score: float = 94.7) -> dict:
    return {
        "grade": grade,
        "score": score,
        "total_checks": 19,
        "checks_passed": 18,
        "checks_failed": 1,
        "checks": [
            {
                "check": "no_negatives_age",
                "passed": False,
                "value": 1,
                "threshold": 0,
                "detail": "Column 'age': 1 negative values found",
            },
            {
                "check": "overall_completeness",
                "passed": True,
                "value": 92.5,
                "threshold": 70.0,
                "detail": "92.5% complete",
            },
        ],
    }


# ===========================================================================
# generate_briefing
# ===========================================================================

class TestGenerateBriefing:
    """Tests for core.enrich.generate_briefing()."""

    def test_returns_required_keys(self) -> None:
        """Result must have text, generated_by, and latency_s keys."""
        router = _mock_router("Executive summary here.")
        result = generate_briefing(router, "patients", _fake_profile(), _fake_scorecard())
        assert {"text", "generated_by", "latency_s"}.issubset(result.keys())

    def test_prompt_contains_fact_sheet_numbers(self) -> None:
        """The prompt sent to the router must contain key facts (row count, grade)."""
        captured: list[str] = []
        router = MagicMock()

        def _capture(task, prompt, **kw):
            captured.append(prompt)
            return ("AI text", "ollama")

        router.generate.side_effect = _capture
        generate_briefing(router, "patients", _fake_profile(row_count=20), _fake_scorecard(grade="A", score=94.7))

        assert len(captured) == 1
        prompt = captured[0]
        assert "20" in prompt         # row_count
        assert "94.7" in prompt       # score
        assert "A" in prompt          # grade

    def test_prompt_does_not_contain_raw_data_values(self) -> None:
        """The prompt must not contain actual data cell values from the DataFrame.

        WHY: generate_briefing receives only aggregated profile/scorecard dicts,
        so no raw row values can leak into the prompt by construction.
        """
        captured: list[str] = []

        def _capture(task, prompt, **kw):
            captured.append(prompt)
            return ("text", "ollama")

        router = MagicMock()
        router.generate.side_effect = _capture

        generate_briefing(router, "test_fixture", _fake_profile(), _fake_scorecard())
        prompt = captured[0]

        # These are example cell values from the fixture — they must not appear.
        assert "P001" not in prompt
        assert "Hypertension" not in prompt
        assert "alice@example.com" not in prompt

    def test_generated_by_includes_model_name_for_ollama(self) -> None:
        """generated_by must name the local model when Ollama responds."""
        router = _mock_router("Some briefing text.")
        result = generate_briefing(router, "ds", _fake_profile(), _fake_scorecard())
        assert "local" in result["generated_by"]

    def test_fallback_when_router_returns_none(self) -> None:
        """When generate() returns None, result uses rule-based fallback text."""
        router = _mock_router(None)
        result = generate_briefing(router, "patients", _fake_profile(row_count=42), _fake_scorecard(grade="B"))
        assert result["generated_by"] == "rule-based fallback"
        assert "42" in result["text"]   # row count in fallback template
        assert "B" in result["text"]    # grade in fallback template
        assert result["latency_s"] == 0.0


# ===========================================================================
# describe_columns
# ===========================================================================

class TestDescribeColumns:
    """Tests for core.enrich.describe_columns()."""

    def test_batching_34_columns_makes_5_calls(self) -> None:
        """34 columns / batch_size=7 → ceil(34/7) = 5 generate() calls."""
        df = pd.DataFrame({f"col_{i}": range(3) for i in range(34)})
        router = _mock_router("col_0: A field\ncol_1: Another\n")
        describe_columns(router, df, {"columns": {}})
        assert router.generate.call_count == 5

    def test_all_columns_have_a_description(self) -> None:
        """Every column must end up with at least a rule-based description."""
        df = pd.DataFrame({f"col_{i}": range(3) for i in range(10)})
        # Router returns empty string — forces full fallback
        router = _mock_router("")
        result = describe_columns(router, df, {"columns": {}})
        for col in df.columns:
            assert col in result
            assert isinstance(result[col], str)
            assert len(result[col]) > 0

    def test_well_formed_ai_line_is_accepted(self) -> None:
        """A correctly formatted 'col_name: description' line must be stored."""
        df = pd.DataFrame({"patient_id": ["P001"], "age": [34]})
        router = _mock_router("patient_id: Unique patient identifier\nage: Age in years\n")
        result = describe_columns(router, df, {"columns": {}})
        assert result["patient_id"] == "Unique patient identifier"
        assert result["age"] == "Age in years"

    def test_malformed_line_falls_back_per_column(self) -> None:
        """A line whose key doesn't match any real column must be ignored."""
        df = pd.DataFrame({"cost_usd": [100.0], "state": ["AL"]})
        # "not_a_real_column" is not in df.columns → must be ignored
        router = _mock_router("not_a_real_column: Ignore this\ncost_usd: Dollar cost\n")
        result = describe_columns(router, df, {"columns": {}})
        # cost_usd was parsed correctly
        assert result["cost_usd"] == "Dollar cost"
        # state had no valid line → got rule-based fallback
        assert "state" in result
        assert len(result["state"]) > 0

    def test_line_without_colon_is_ignored(self) -> None:
        """Lines without a colon separator must not cause errors or bad entries."""
        df = pd.DataFrame({"age": [34]})
        router = _mock_router("this line has no colon at all\nage: Age in years\n")
        result = describe_columns(router, df, {"columns": {}})
        assert result["age"] == "Age in years"

    def test_router_none_response_falls_back_for_whole_batch(self) -> None:
        """When generate() returns None, every column in that batch gets fallback."""
        df = pd.DataFrame({"age": [34], "cost_usd": [100.0]})
        router = _mock_router(None)
        result = describe_columns(router, df, {"columns": {}})
        # Both columns should have rule-based descriptions (non-empty strings)
        assert len(result["age"]) > 0
        assert len(result["cost_usd"]) > 0


# ===========================================================================
# explain_issues
# ===========================================================================

class TestExplainIssues:
    """Tests for core.enrich.explain_issues()."""

    _SCHEMA = [
        {"column_name": "age", "column_type": "INTEGER"},
        {"column_name": "diagnosis", "column_type": "VARCHAR"},
    ]

    _ISSUES = [
        {
            "check": "no_negatives_age",
            "passed": False,
            "value": 1,
            "threshold": 0,
            "detail": "Column 'age': 1 negative values found",
        },
        {
            "check": "null_rate_diagnosis",
            "passed": False,
            "value": 15.0,
            "threshold": 20.0,
            "detail": "Column 'diagnosis': 15.0% null",
        },
        {
            "check": "overall_completeness",
            "passed": True,  # passed — must be excluded
            "value": 92.5,
            "threshold": 70.0,
            "detail": "92.5% complete",
        },
    ]

    def test_result_has_required_keys(self) -> None:
        """Each result item must have issue/explanation/suggested_fix/generated_by."""
        router = _mock_router("Explanation: Bad data.\nFix: df.dropna()")
        results = explain_issues(router, self._ISSUES, self._SCHEMA, top_n=5)
        for item in results:
            assert {"issue", "explanation", "suggested_fix", "generated_by"}.issubset(item.keys())

    def test_passed_checks_are_excluded(self) -> None:
        """Issues with passed=True must not appear in the output."""
        router = _mock_router("Explanation: Bad.\nFix: fix()")
        results = explain_issues(router, self._ISSUES, self._SCHEMA, top_n=5)
        check_names = [r["issue"].get("check") for r in results]
        assert "overall_completeness" not in check_names

    def test_top_n_limits_results(self) -> None:
        """At most top_n items are returned."""
        router = _mock_router("Explanation: Bad.\nFix: fix()")
        results = explain_issues(router, self._ISSUES, self._SCHEMA, top_n=1)
        assert len(results) <= 1

    def test_severity_ordering_completeness_before_null_rate(self) -> None:
        """overall_completeness (severity 0) must rank before null_rate (severity 2)."""
        issues = [
            {"check": "null_rate_col", "passed": False, "detail": "10% null"},
            {"check": "overall_completeness", "passed": False, "detail": "50% complete"},
        ]
        router = _mock_router("Explanation: text.\nFix: fix()")
        results = explain_issues(router, issues, self._SCHEMA, top_n=5)
        # overall_completeness has lower severity score → appears first
        assert results[0]["issue"]["check"] == "overall_completeness"

    def test_fallback_fires_when_router_is_none(self) -> None:
        """With router=None, all results must use rule-based fallback."""
        results = explain_issues(None, self._ISSUES, self._SCHEMA, top_n=5)
        for item in results:
            assert item["generated_by"] == "rule-based fallback"
            assert len(item["explanation"]) > 0
            assert len(item["suggested_fix"]) > 0

    def test_fallback_fires_when_router_returns_none(self) -> None:
        """When router.generate() returns None, fallback templates must be used."""
        router = _mock_router(None)
        results = explain_issues(router, self._ISSUES, self._SCHEMA, top_n=5)
        for item in results:
            assert item["generated_by"] == "rule-based fallback"

    def test_null_rate_fallback_template_mentions_column(self) -> None:
        """Null-rate fallback explanation must reference the column name."""
        issues = [{"check": "null_rate_diagnosis", "passed": False, "detail": "15% null"}]
        results = explain_issues(None, issues, self._SCHEMA, top_n=5)
        assert "diagnosis" in results[0]["explanation"]

    def test_empty_issues_returns_empty_list(self) -> None:
        """Empty issues list (all passed) → empty results."""
        passed_issues = [{"check": "ok", "passed": True}]
        results = explain_issues(None, passed_issues, self._SCHEMA, top_n=5)
        assert results == []


# ===========================================================================
# Rule-based description helper
# ===========================================================================

class TestRuleBasedDescription:
    """Unit tests for the _rule_based_description() helper."""

    def test_id_column(self) -> None:
        assert "identifier" in _rule_based_description("patient_id", "object").lower()

    def test_date_column(self) -> None:
        assert "date" in _rule_based_description("visit_date", "object").lower()

    def test_cost_column(self) -> None:
        assert "monetary" in _rule_based_description("cost_usd", "float64").lower() or \
               "cost" in _rule_based_description("cost_usd", "float64").lower()

    def test_state_column(self) -> None:
        result = _rule_based_description("state", "object")
        assert len(result) > 0

    def test_numeric_fallback(self) -> None:
        """Unknown numeric column falls back to 'Numeric measurement field'."""
        result = _rule_based_description("xyz_val", "float64")
        assert "numeric" in result.lower()

    def test_generic_fallback(self) -> None:
        """Completely unknown column returns a non-empty fallback."""
        result = _rule_based_description("qwerty", "object")
        assert len(result) > 0


# ===========================================================================
# Pipeline integration — router=None passes cleanly
# ===========================================================================

class TestPipelineWithoutRouter:
    """run_pipeline with router=None must include briefing/issue_explanations keys."""

    def test_pipeline_router_none_has_briefing_key(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """briefing key must be present (None) when no router is supplied."""
        from core.pipeline import run_pipeline

        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(
            df, "test_fixture", db_path=tmp_path / "hp.duckdb", router=None
        )
        assert "briefing" in result
        assert result["briefing"] is None

    def test_pipeline_router_none_has_issue_explanations_key(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """issue_explanations key must be present (None) when no router is supplied."""
        from core.pipeline import run_pipeline

        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(
            df, "test_fixture", db_path=tmp_path / "hp.duckdb", router=None
        )
        assert "issue_explanations" in result
        assert result["issue_explanations"] is None

    def test_pipeline_enable_ai_enrichment_false_skips_enrichment(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """enable_ai_enrichment=False with a real router must still skip enrichment."""
        from core.pipeline import run_pipeline
        from core.router import AIRouter

        router = AIRouter()
        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(
            df,
            "test_fixture",
            db_path=tmp_path / "hp.duckdb",
            router=router,
            enable_ai_enrichment=False,
        )
        assert result["briefing"] is None
        assert result["issue_explanations"] is None

    def test_pipeline_no_router_passes_end_to_end(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """Full pipeline run with router=None must complete without errors."""
        from core.pipeline import run_pipeline

        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(
            df, "test_fixture", db_path=tmp_path / "hp.duckdb", router=None
        )
        assert result["scorecard"] is not None
        assert result["clean_df"] is not None
        assert len(result["clean_df"]) == 19
