"""
tests/test_pipeline.py — Hermetic tests for core.pipeline.run_pipeline().

Uses tmp_path so every test gets its own isolated DuckDB file.
No network calls; fixtures are the standard sample files from conftest.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.pipeline import run_pipeline, sanitize_table_name
from core.database import DuckDBManager
from ingestion.file_source import FileSource


# ---------------------------------------------------------------------------
# sanitize_table_name
# ---------------------------------------------------------------------------

class TestSanitizeTableName:
    """Unit tests for the table name sanitizer."""

    def test_removes_extension(self) -> None:
        assert sanitize_table_name("test_fixture.csv") == "test_fixture_csv"

    def test_replaces_spaces(self) -> None:
        assert sanitize_table_name("My Dataset 2024") == "my_dataset_2024"

    def test_replaces_hyphens(self) -> None:
        assert sanitize_table_name("who-life-expectancy") == "who_life_expectancy"

    def test_collapses_repeated_underscores(self) -> None:
        assert sanitize_table_name("a___b") == "a_b"

    def test_lowercases(self) -> None:
        assert sanitize_table_name("MyTable") == "mytable"

    def test_strips_leading_trailing_underscores(self) -> None:
        assert sanitize_table_name("_hidden_") == "hidden"


# ---------------------------------------------------------------------------
# run_pipeline — result structure
# ---------------------------------------------------------------------------

class TestPipelineResultStructure:
    """run_pipeline must return a dict with all documented keys."""

    def test_result_has_all_keys(self, fixture_csv: Path, tmp_path: Path) -> None:
        """Result dict must contain all expected keys."""
        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(df, "test_fixture", db_path=tmp_path / "hp.duckdb")
        required = {
            "profile", "clean_df", "transform_log", "scorecard",
            "docs", "table_name", "gate_blocked",
        }
        assert required.issubset(result.keys())

    def test_gate_blocked_false_by_default(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """gate_blocked must be False when no quality_gate_min_grade is set."""
        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(df, "test_fixture", db_path=tmp_path / "hp.duckdb")
        assert result["gate_blocked"] is False


# ---------------------------------------------------------------------------
# run_pipeline — end-to-end data flow
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    """Full pipeline run on the CSV fixture."""

    def test_clean_df_has_19_rows(self, fixture_csv: Path, tmp_path: Path) -> None:
        """After dedup the clean DataFrame must have 19 rows."""
        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(df, "test_fixture", db_path=tmp_path / "hp.duckdb")
        assert len(result["clean_df"]) == 19

    def test_table_loaded_in_duckdb(self, fixture_csv: Path, tmp_path: Path) -> None:
        """The sanitized table name must appear in DuckDB after the pipeline."""
        df = FileSource().extract(filepath=str(fixture_csv))
        db_file = tmp_path / "hp.duckdb"
        result = run_pipeline(df, "test_fixture", db_path=db_file)
        with DuckDBManager(db_path=db_file) as db:
            tables = db.list_tables()
        assert result["table_name"] in tables

    def test_table_row_count_matches_clean_df(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """Row count in DuckDB must equal the clean_df length."""
        df = FileSource().extract(filepath=str(fixture_csv))
        db_file = tmp_path / "hp.duckdb"
        result = run_pipeline(df, "test_fixture", db_path=db_file)
        with DuckDBManager(db_path=db_file) as db:
            count = db.query(
                f"SELECT COUNT(*) AS n FROM {result['table_name']}"
            )["n"][0]
        assert count == len(result["clean_df"])

    def test_profile_present_in_result(self, fixture_csv: Path, tmp_path: Path) -> None:
        """Profile must be populated with pii_columns key."""
        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(df, "test_fixture", db_path=tmp_path / "hp.duckdb")
        assert result["profile"] is not None
        assert "pii_columns" in result["profile"]

    def test_scorecard_grade_set(self, fixture_csv: Path, tmp_path: Path) -> None:
        """Scorecard must contain a grade in {A, B, C, F}."""
        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(df, "test_fixture", db_path=tmp_path / "hp.duckdb")
        assert result["scorecard"]["grade"] in {"A", "B", "C", "F"}


# ---------------------------------------------------------------------------
# run_pipeline — progress callback
# ---------------------------------------------------------------------------

class TestPipelineProgressCallback:
    """The progress_callback must be called for every pipeline stage."""

    def test_callback_receives_all_stages(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """Callback must be called with each of the five stage names."""
        calls: list[tuple[str, str]] = []
        df = FileSource().extract(filepath=str(fixture_csv))
        run_pipeline(
            df,
            "test_fixture",
            db_path=tmp_path / "hp.duckdb",
            progress_callback=lambda step, status: calls.append((step, status)),
        )
        stages_called = {step for step, _ in calls}
        assert {"profile", "transform", "quality", "document", "load"}.issubset(
            stages_called
        )

    def test_callback_receives_starting_and_done(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """Each stage must emit both 'starting' and 'done' statuses."""
        calls: list[tuple[str, str]] = []
        df = FileSource().extract(filepath=str(fixture_csv))
        run_pipeline(
            df,
            "test_fixture",
            db_path=tmp_path / "hp.duckdb",
            progress_callback=lambda step, status: calls.append((step, status)),
        )
        statuses = {status for _, status in calls}
        assert "starting" in statuses
        assert "done" in statuses

    def test_no_callback_does_not_crash(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """Passing progress_callback=None must not crash."""
        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(
            df,
            "test_fixture",
            db_path=tmp_path / "hp.duckdb",
            progress_callback=None,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# run_pipeline — quality gate
# ---------------------------------------------------------------------------

class TestQualityGate:
    """quality_gate_min_grade blocks the DuckDB load when grade is worse."""

    def _make_bad_df(self) -> pd.DataFrame:
        """Return a DataFrame that will score well below A (< 90%) after transformation.

        WHY: the transformer fills nulls and standardises types, so purely
        null-based bad data comes out looking clean.  Instead we use columns
        that contain negative values in positive-hinted names (age, rate, count).
        Those negative values SURVIVE the transformer, causing multiple checks
        to fail and pushing the score below the A threshold of 90%.
        """
        return pd.DataFrame({
            # Non-unique IDs with different row values so not all rows dedup'd
            "report_id": ["R001", "R001", "R002", "R002", "R003",
                          "R003", "R004", "R004", "R005", "R005"],
            # ALL negative ages → fails no_negatives_age every time
            "age":   [-100, -50, -30, -20, -10, -5, -3, -1, -2, -200],
            # ALL negative rates → fails no_negatives_rate
            "rate":  [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            # ALL negative counts → fails no_negatives_count
            "count": [-100, -200, -300, -400, -500, -600, -700, -800, -900, -1000],
        })

    def test_gate_blocks_when_grade_below_minimum(self, tmp_path: Path) -> None:
        """Pipeline must set gate_blocked=True and skip DuckDB load."""
        bad_df = self._make_bad_df()
        result = run_pipeline(
            bad_df,
            "bad_dataset",
            db_path=tmp_path / "hp.duckdb",
            quality_gate_min_grade="A",  # very strict — bad_df will never score >= 90
        )
        assert result["gate_blocked"] is True

    def test_gate_blocked_table_not_in_duckdb(self, tmp_path: Path) -> None:
        """When gate blocks, the table must NOT be created in DuckDB."""
        bad_df = self._make_bad_df()
        db_file = tmp_path / "hp.duckdb"
        result = run_pipeline(
            bad_df,
            "bad_dataset",
            db_path=db_file,
            quality_gate_min_grade="A",
        )
        assert result["gate_blocked"] is True
        # DuckDB file might not even exist; if it does, table should be absent.
        if db_file.exists():
            with DuckDBManager(db_path=db_file) as db:
                tables = db.list_tables()
            assert result["table_name"] not in tables

    def test_gate_not_blocked_without_grade(
        self, fixture_csv: Path, tmp_path: Path
    ) -> None:
        """No quality gate → gate_blocked must always be False."""
        df = FileSource().extract(filepath=str(fixture_csv))
        result = run_pipeline(
            df,
            "test_fixture",
            db_path=tmp_path / "hp.duckdb",
            quality_gate_min_grade=None,
        )
        assert result["gate_blocked"] is False

    def test_gate_docs_still_generated_when_blocked(self, tmp_path: Path) -> None:
        """Even when gate blocks, docs must still be generated."""
        bad_df = self._make_bad_df()
        result = run_pipeline(
            bad_df,
            "bad_dataset",
            db_path=tmp_path / "hp.duckdb",
            quality_gate_min_grade="A",
        )
        assert result["gate_blocked"] is True
        assert result["docs"] is not None
