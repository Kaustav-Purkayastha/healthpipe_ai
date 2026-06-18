"""
test_agents.py -- Tests for the four pipeline agents.

Covers:
    - ProfilerAgent: output structure, null detection
    - TransformerAgent: duplicate removal, column standardization, null handling
    - QualityCheckerAgent: scorecard structure, negative-age detection
    - DocumenterAgent: data dictionary generation, markdown file creation
"""

import logging
import re
from pathlib import Path

import pandas as pd
import pytest

from core.config import ROOT_DIR, DOCS_DIR
from agents.profiler import ProfilerAgent
from agents.transformer import TransformerAgent
from agents.quality_checker import QualityCheckerAgent
from agents.documenter import DocumenterAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProfilerAgent tests
# ---------------------------------------------------------------------------

class TestProfilerAgent:
    """Tests for the data profiling agent."""

    def test_profiler_output_structure(self, sample_dataframe: pd.DataFrame) -> None:
        """Profiler output dict should contain the four required top-level keys."""
        profiler = ProfilerAgent()
        profile = profiler.run(sample_dataframe, "test_fixture")

        # The profile dict must have these keys for downstream agents
        required_keys = {"overview", "columns", "quality_issues", "correlations"}
        actual_keys = set(profile.keys())
        # Check that every required key is present
        assert required_keys.issubset(actual_keys), (
            f"Missing keys: {required_keys - actual_keys}"
        )
        # overview should itself be a dict with row_count
        assert "row_count" in profile["overview"], (
            "overview missing 'row_count'"
        )
        logger.info(f"Profile keys: {list(profile.keys())}")

    def test_profiler_detects_nulls(self, sample_dataframe: pd.DataFrame) -> None:
        """Profiler should detect null values present in the test fixture."""
        profiler = ProfilerAgent()
        profile = profiler.run(sample_dataframe, "test_fixture_nulls")

        # The fixture has 3 null cells: diagnosis(row6), blood_pressure(row9),
        # cholesterol(row15).  The profiler tracks nulls per-column in 'columns'.
        columns = profile["columns"]

        # Sum up total nulls across all columns
        total_nulls = sum(
            col_info.get("null_count", 0) for col_info in columns.values()
        )
        # At least 3 nulls should be detected from the fixture
        assert total_nulls >= 3, (
            f"Expected at least 3 nulls in fixture, profiler found {total_nulls}"
        )
        logger.info(f"Total nulls detected by profiler: {total_nulls}")


# ---------------------------------------------------------------------------
# TransformerAgent tests
# ---------------------------------------------------------------------------

class TestTransformerAgent:
    """Tests for the data transformation agent."""

    def test_transformer_removes_duplicates(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Transformer should drop the 1 duplicate row (20 -> 19 rows)."""
        transformer = TransformerAgent()
        # run() chains all transform steps including remove_duplicates
        clean_df = transformer.run(sample_dataframe, "test_dedup")

        # Fixture has 20 rows with 1 exact duplicate (P002 appears twice)
        assert len(clean_df) == 19, (
            f"Expected 19 rows after dedup, got {len(clean_df)}"
        )
        logger.info(f"Rows after dedup: {len(clean_df)}")

    def test_transformer_standardizes_columns(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """All column names should be snake_case after transformation."""
        transformer = TransformerAgent()
        clean_df = transformer.run(sample_dataframe, "test_cols")

        # snake_case pattern: lowercase letters, digits, and underscores only.
        # Metadata columns (_loaded_at, _source) start with underscore which
        # is valid snake_case.
        snake_case_pattern = re.compile(r"^_?[a-z][a-z0-9]*(_[a-z0-9]+)*$")

        for col in clean_df.columns:
            assert snake_case_pattern.match(col), (
                f"Column '{col}' is not valid snake_case"
            )
        logger.info(f"All {len(clean_df.columns)} columns are snake_case")

    def test_transformer_handles_nulls(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Numeric columns should have zero nulls after transformation."""
        transformer = TransformerAgent()
        clean_df = transformer.run(sample_dataframe, "test_nulls")

        # After handle_nulls(), numeric columns get median-filled
        numeric_cols = clean_df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            null_count = clean_df[col].isna().sum()
            assert null_count == 0, (
                f"Numeric column '{col}' still has {null_count} nulls "
                f"after transformation"
            )
        logger.info(
            f"All {len(numeric_cols)} numeric columns are null-free"
        )


# ---------------------------------------------------------------------------
# QualityCheckerAgent tests
# ---------------------------------------------------------------------------

class TestQualityCheckerAgent:
    """Tests for the quality checking agent."""

    def test_quality_checker_scoring(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Scorecard should contain score, grade, and checks keys."""
        checker = QualityCheckerAgent()
        scorecard = checker.run(sample_dataframe, "test_scoring")

        # These three keys are consumed by the report generator
        assert "score" in scorecard, "Scorecard missing 'score'"
        assert "grade" in scorecard, "Scorecard missing 'grade'"
        assert "checks" in scorecard, "Scorecard missing 'checks'"

        # Score should be a number between 0 and 100
        assert 0 <= scorecard["score"] <= 100, (
            f"Score {scorecard['score']} is outside 0-100 range"
        )
        # Grade should be one of the valid letters
        assert scorecard["grade"] in ("A", "B", "C", "F"), (
            f"Unexpected grade: {scorecard['grade']}"
        )
        logger.info(
            f"Quality score: {scorecard['score']}% "
            f"(Grade {scorecard['grade']})"
        )

    def test_quality_checker_finds_negative_age(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Quality checker should flag the negative age value in the fixture."""
        checker = QualityCheckerAgent()
        scorecard = checker.run(sample_dataframe, "test_neg_age")

        # The fixture has age=-5 in row 8 (P008).
        # _check_value_ranges flags negatives in columns whose name
        # contains "age", "count", "rate", etc.
        checks = scorecard["checks"]

        # Find the check that looks at negatives in the age column
        age_checks = [
            c for c in checks
            if "age" in c.get("check", "").lower()
            and "negative" in c.get("check", "").lower()
        ]

        # There should be a no_negatives_age check that failed
        assert len(age_checks) > 0, (
            "No negative-age check found in scorecard checks"
        )
        # The check should have failed because age=-5 exists
        failed_age = [c for c in age_checks if not c["passed"]]
        assert len(failed_age) > 0, (
            "Negative-age check passed but fixture has age=-5"
        )
        logger.info(
            f"Negative age correctly flagged: {failed_age[0]}"
        )


# ---------------------------------------------------------------------------
# DocumenterAgent tests
# ---------------------------------------------------------------------------

class TestDocumenterAgent:
    """Tests for the auto-documentation agent."""

    def test_documenter_generates_dictionary(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Data dictionary should have one entry per DataFrame column."""
        documenter = DocumenterAgent()
        docs = documenter.run(sample_dataframe, "test_dict")

        dictionary = docs.get("data_dictionary", [])
        # One dictionary entry per column in the input DataFrame
        assert len(dictionary) == len(sample_dataframe.columns), (
            f"Dictionary has {len(dictionary)} entries but DataFrame "
            f"has {len(sample_dataframe.columns)} columns"
        )
        # Each entry should have the required fields
        for entry in dictionary:
            assert "column_name" in entry, "Dictionary entry missing 'column_name'"
            assert "data_type" in entry, "Dictionary entry missing 'data_type'"
            assert "description" in entry, "Dictionary entry missing 'description'"

        logger.info(
            f"Dictionary generated: {len(dictionary)} column entries"
        )

    def test_documenter_generates_markdown(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Documenter should create a markdown file in outputs/docs/."""
        documenter = DocumenterAgent()
        # run() saves both JSON and Markdown to DOCS_DIR
        docs = documenter.run(sample_dataframe, "test_md")

        # The markdown file should exist at outputs/docs/docs_test_md.md
        md_path = DOCS_DIR / "docs_test_md.md"
        assert md_path.exists(), (
            f"Markdown file not created at {md_path}"
        )
        # File should have real content, not be empty
        content = md_path.read_text(encoding="utf-8")
        assert len(content) > 100, (
            f"Markdown file is suspiciously short: {len(content)} chars"
        )
        logger.info(
            f"Markdown doc created: {md_path} ({len(content)} chars)"
        )
