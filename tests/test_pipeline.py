"""
test_pipeline.py -- Integration tests that run the full pipeline end-to-end.

Covers:
    - Full pipeline on the 20-row test fixture (ingest -> profile ->
      transform -> quality check -> document -> load to DuckDB)
    - Full pipeline on the real 309K-row U.S. Chronic Disease Indicators CSV
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from core.config import ROOT_DIR
from ingestion.csv_source import CSVSource
from agents.profiler import ProfilerAgent
from agents.transformer import TransformerAgent
from agents.quality_checker import QualityCheckerAgent
from agents.documenter import DocumenterAgent
from core.database import DuckDBManager

logger = logging.getLogger(__name__)


class TestFullPipeline:
    """End-to-end integration tests for the complete pipeline."""

    def test_full_pipeline_fixture(
        self, test_fixture_path: Path, tmp_path: Path
    ) -> None:
        """
        Run every pipeline stage on the 20-row test fixture.

        Stages: ingest -> profile -> transform -> quality check ->
        document -> load to DuckDB.  Validates output at each stage.
        """
        # -- Stage 1: Ingest --
        source = CSVSource()
        raw_df = source.extract(filepath=str(test_fixture_path))
        assert len(raw_df) == 20, f"Ingest failed: expected 20 rows, got {len(raw_df)}"
        logger.info(f"[Ingest] {len(raw_df)} rows loaded")

        # -- Stage 2: Profile --
        profiler = ProfilerAgent()
        profile = profiler.run(raw_df, "integration_fixture")
        assert "overview" in profile, "Profile missing 'overview' key"
        assert profile["overview"]["row_count"] == 20, (
            "Profile row_count does not match ingested data"
        )
        logger.info(f"[Profile] completeness={profile['overview']['completeness_score']}%")

        # -- Stage 3: Transform --
        transformer = TransformerAgent()
        clean_df = transformer.run(raw_df, "integration_fixture")
        # Should be 19 rows after removing the 1 duplicate
        assert len(clean_df) == 19, (
            f"Transform failed: expected 19 rows, got {len(clean_df)}"
        )
        transform_log = transformer.get_transform_summary()
        # The transformer runs 6 steps (standardize, dedup, convert,
        # nulls, clean text, add metadata)
        assert len(transform_log) == 6, (
            f"Expected 6 transform steps, got {len(transform_log)}"
        )
        logger.info(f"[Transform] {len(clean_df)} rows, {len(transform_log)} steps")

        # -- Stage 4: Quality Check --
        checker = QualityCheckerAgent()
        scorecard = checker.run(clean_df, "integration_fixture")
        assert "score" in scorecard, "Scorecard missing 'score'"
        assert "grade" in scorecard, "Scorecard missing 'grade'"
        logger.info(
            f"[Quality] score={scorecard['score']}%, "
            f"grade={scorecard['grade']}"
        )

        # -- Stage 5: Document --
        documenter = DocumenterAgent()
        docs = documenter.run(
            clean_df,
            "integration_fixture",
            source_metadata=source.get_metadata(),
            profile_data=profile,
            transform_log=transform_log,
            quality_scorecard=scorecard,
        )
        # Dictionary should have one entry per column in the cleaned DataFrame
        assert len(docs["data_dictionary"]) == len(clean_df.columns), (
            "Dictionary column count does not match cleaned DataFrame"
        )
        logger.info(
            f"[Document] {len(docs['data_dictionary'])} columns documented"
        )

        # -- Stage 6: Load to DuckDB --
        # Use tmp_path so the test database does not pollute the project
        db_path = tmp_path / "test_pipeline.duckdb"
        with DuckDBManager(db_path=db_path) as db:
            db.load_dataframe(clean_df, "integration_fixture")
            tables = db.list_tables()
            assert "integration_fixture" in tables, (
                f"Table not found in DuckDB. Tables: {tables}"
            )
            # Verify row count matches via SQL
            result = db.query("SELECT COUNT(*) AS cnt FROM integration_fixture")
            row_count = result["cnt"].iloc[0]
            assert row_count == 19, (
                f"DuckDB row count mismatch: expected 19, got {row_count}"
            )
        logger.info(f"[DuckDB] Table loaded with {row_count} rows at {db_path}")

    def test_full_pipeline_real_data(self, tmp_path: Path) -> None:
        """
        Run the full pipeline on the real 309K-row CSV dataset.

        Skipped automatically if the real dataset file is not present.
        """
        real_csv = ROOT_DIR / "data" / "sample" / "U.S._Chronic_Disease_Indicators.csv"
        if not real_csv.exists():
            pytest.skip("Real dataset not found -- skipping large file test")

        # -- Ingest --
        source = CSVSource()
        raw_df = source.extract(filepath=str(real_csv))
        assert len(raw_df) >= 309_000, (
            f"Expected >= 309,000 rows, got {len(raw_df)}"
        )
        logger.info(f"[Ingest] {len(raw_df)} rows from real dataset")

        # -- Profile --
        profiler = ProfilerAgent()
        profile = profiler.run(raw_df, "cdi_real")
        assert profile["overview"]["row_count"] == len(raw_df), (
            "Profile row_count does not match ingested data"
        )

        # -- Transform --
        transformer = TransformerAgent()
        clean_df = transformer.run(raw_df, "cdi_real")
        # After dedup the count should be <= original
        assert len(clean_df) <= len(raw_df), (
            "Transformed data has more rows than raw -- unexpected"
        )
        logger.info(f"[Transform] {len(raw_df)} -> {len(clean_df)} rows")

        # -- Quality Check --
        checker = QualityCheckerAgent()
        scorecard = checker.run(clean_df, "cdi_real")
        assert scorecard["grade"] in ("A", "B", "C", "F"), (
            f"Invalid grade: {scorecard['grade']}"
        )

        # -- Document --
        documenter = DocumenterAgent()
        docs = documenter.run(
            clean_df,
            "cdi_real",
            source_metadata=source.get_metadata(),
            profile_data=profile,
            transform_log=transformer.get_transform_summary(),
            quality_scorecard=scorecard,
        )
        assert len(docs["data_dictionary"]) > 0, (
            "Data dictionary is empty for real dataset"
        )

        # -- Load to DuckDB --
        db_path = tmp_path / "test_real_pipeline.duckdb"
        with DuckDBManager(db_path=db_path) as db:
            db.load_dataframe(clean_df, "cdi_real")
            tables = db.list_tables()
            assert "cdi_real" in tables, (
                f"Table not found. Tables: {tables}"
            )
        logger.info(
            f"[Full pipeline] Real dataset complete: "
            f"score={scorecard['score']}%, grade={scorecard['grade']}"
        )
