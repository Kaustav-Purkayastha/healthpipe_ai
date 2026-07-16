"""
core/pipeline.py — Orchestrates the full HealthPipe AI v2 deterministic pipeline.

Stages (in order):
    1. Profile      — ProfilerAgent.run()
    2. Transform    — TransformerAgent.run()
    3. Quality      — QualityCheckerAgent.run()
    4. Document     — DocumenterAgent.run()
    5. Load         — DuckDBManager.load_dataframe()

The optional ``progress_callback`` is called as callback(step_name, status)
before and after every stage.  WHY: the Streamlit UI (Step 4) drives its live
agent status panel through this hook — the callback updates the UI without the
pipeline knowing anything about Streamlit.

The optional ``quality_gate_min_grade`` blocks the DuckDB load when the
scorecard grade is below the requested minimum.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from core.config import DATABASE_PATH, QUALITY_GRADE_BANDS
from core.database import DuckDBManager
from core.utils import get_logger
from agents.profiler import ProfilerAgent
from agents.transformer import TransformerAgent
from agents.quality_checker import QualityCheckerAgent
from agents.documenter import DocumenterAgent

# AIRouter imported lazily inside run_pipeline to keep the import graph clean
# and allow router=None callers to skip loading providers entirely.

_log = get_logger(__name__)

# Numeric rank for each grade — used to compare grades in the quality gate.
_GRADE_RANK: dict[str, int] = {"A": 4, "B": 3, "C": 2, "F": 1}


def sanitize_table_name(name: str) -> str:
    """Convert any string to a safe DuckDB table name.

    Replaces non-alphanumeric characters with underscores, collapses
    consecutive underscores, strips leading/trailing underscores, and
    lowercases the result.

    Args:
        name: Raw dataset or file name (e.g. "test_fixture.csv").

    Returns:
        Safe table name string (e.g. "test_fixture_csv").
    """
    # Replace anything that isn't a letter or digit with an underscore.
    s = re.sub(r"[^a-zA-Z0-9]", "_", name)
    # Collapse consecutive underscores into one.
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def run_pipeline(
    df: pd.DataFrame,
    dataset_name: str,
    source_metadata: Optional[dict] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    quality_gate_min_grade: Optional[str] = None,
    db_path: Optional[Path] = None,
    router=None,              # AIRouter | None — avoids import at module level
    enable_ai_enrichment: bool = True,
) -> dict:
    """Orchestrate the full profile → transform → quality → [enrich] → document → load pipeline.

    Args:
        df:                     Raw DataFrame to process.
        dataset_name:           Human-readable name used for table names and reports.
        source_metadata:        Dict from source.get_metadata() (optional; passed to
                                DocumenterAgent for lineage tracking).
        progress_callback:      Optional callable(step_name: str, status: str).
                                Called with status="starting" before each stage and
                                status="done" after.  The Streamlit UI hooks here.
        quality_gate_min_grade: If set (e.g. "B"), the DuckDB load is skipped when
                                the scorecard grade is worse than this grade.
                                "gate_blocked": True is set in the result.
        db_path:                Path to the DuckDB file.  Defaults to DATABASE_PATH
                                from config (outputs/healthpipe.duckdb).
        router:                 Optional AIRouter for AI enrichment (Step 5).
                                When None or enable_ai_enrichment=False, enrichment
                                is skipped and result keys are present but None.
        enable_ai_enrichment:   Set False to skip AI enrichment even when a router
                                is supplied (e.g. --no-ai CLI flag).

    Returns:
        Dict with keys:
            profile            — output of ProfilerAgent.run()
            clean_df           — transformed DataFrame
            transform_log      — list of transformation step dicts
            scorecard          — output of QualityCheckerAgent.run()
            docs               — output of DocumenterAgent.run()
            table_name         — sanitized DuckDB table name
            gate_blocked       — True if quality gate prevented the load, else False
            briefing           — dict from generate_briefing(), or None
            issue_explanations — list from explain_issues(), or None
            starter_questions  — list[str] of chat starter questions, or None
    """
    _cb = progress_callback or _noop_callback
    table_name = sanitize_table_name(dataset_name)
    _log.info("Pipeline starting for '%s' → table '%s'", dataset_name, table_name)

    result: dict = {
        "profile": None,
        "clean_df": None,
        "transform_log": [],
        "scorecard": None,
        "docs": None,
        "table_name": table_name,
        "gate_blocked": False,
        "briefing": None,
        "issue_explanations": None,
        "starter_questions": None,
    }

    # ------------------------------------------------------------------
    # Stage 1: Profile (on the raw data)
    # ------------------------------------------------------------------
    _cb("profile", "starting")
    profiler = ProfilerAgent()
    profile = profiler.run(df, dataset_name)
    result["profile"] = profile
    _cb("profile", "done")

    # ------------------------------------------------------------------
    # Stage 2: Transform
    # ------------------------------------------------------------------
    _cb("transform", "starting")
    transformer = TransformerAgent()
    clean_df = transformer.run(df, dataset_name)
    transform_log = transformer.get_transform_summary()
    result["clean_df"] = clean_df
    result["transform_log"] = transform_log
    _cb("transform", "done")

    # ------------------------------------------------------------------
    # Stage 3: Quality check (on the clean data)
    # ------------------------------------------------------------------
    _cb("quality", "starting")
    checker = QualityCheckerAgent()
    scorecard = checker.run(clean_df, dataset_name)
    result["scorecard"] = scorecard
    _cb("quality", "done")

    # ------------------------------------------------------------------
    # Stage 4: AI Enrichment (optional — requires router + flag)
    # ------------------------------------------------------------------
    column_descriptions: dict | None = None
    if router is not None and enable_ai_enrichment:
        from core.enrich import describe_columns, explain_issues, generate_briefing  # lazy import
        _cb("ai_enrichment", "starting")
        try:
            column_descriptions = describe_columns(router, clean_df, profile)
            briefing = generate_briefing(router, dataset_name, profile, scorecard)
            failed_checks = [c for c in scorecard.get("checks", []) if not c.get("passed", True)]
            df_schema = [
                {"column_name": col, "column_type": str(dtype)}
                for col, dtype in clean_df.dtypes.items()
            ]
            issue_explanations = explain_issues(router, failed_checks, df_schema, top_n=5)
            result["briefing"] = briefing
            result["issue_explanations"] = issue_explanations

            # Precompute chat starter questions now, while the local model is warm.
            # Persisted as an artifact so the chat screen loads them instantly with
            # no spinner and no repeat cost (built from clean_df — the table is not
            # yet loaded into DuckDB at this stage).
            from core.analyst import starter_questions_from_df  # lazy import
            result["starter_questions"] = starter_questions_from_df(
                router, clean_df, table_name
            )
        except Exception as exc:  # noqa: BLE001 — enrichment failure is non-fatal
            _log.warning("AI enrichment failed (non-fatal): %s", exc)
        _cb("ai_enrichment", "done")

    # ------------------------------------------------------------------
    # Quality gate — block DuckDB load if grade is below minimum
    # ------------------------------------------------------------------
    if quality_gate_min_grade is not None:
        actual_grade = scorecard.get("grade", "F")
        if _grade_worse_than(actual_grade, quality_gate_min_grade):
            _log.warning(
                "Quality gate blocked: grade=%s < min=%s — DuckDB load skipped",
                actual_grade,
                quality_gate_min_grade,
            )
            result["gate_blocked"] = True
            # Still generate docs even when gate blocks, for transparency.
            _cb("document", "starting")
            documenter = DocumenterAgent()
            docs = documenter.run(
                clean_df,
                dataset_name,
                source_metadata=source_metadata,
                profile_data=profile,
                transform_log=transform_log,
                quality_scorecard=scorecard,
                descriptions=column_descriptions,
            )
            result["docs"] = docs
            _cb("document", "done")
            return result

    # ------------------------------------------------------------------
    # Stage 5: Document
    # ------------------------------------------------------------------
    _cb("document", "starting")
    documenter = DocumenterAgent()
    docs = documenter.run(
        clean_df,
        dataset_name,
        source_metadata=source_metadata,
        profile_data=profile,
        transform_log=transform_log,
        quality_scorecard=scorecard,
        descriptions=column_descriptions,
    )
    result["docs"] = docs
    _cb("document", "done")

    # ------------------------------------------------------------------
    # Stage 5: Load to DuckDB
    # ------------------------------------------------------------------
    _cb("load", "starting")
    _effective_db_path = db_path or DATABASE_PATH
    _effective_db_path.parent.mkdir(parents=True, exist_ok=True)
    with DuckDBManager(db_path=_effective_db_path) as db:
        db.load_dataframe(clean_df, table_name)
    _cb("load", "done")

    _log.info(
        "Pipeline complete: %d rows → table '%s'",
        len(clean_df),
        table_name,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _noop_callback(step_name: str, status: str) -> None:
    """No-op progress callback used when caller passes None."""


def _grade_worse_than(actual: str, minimum: str) -> bool:
    """Return True if *actual* grade is strictly worse than *minimum* grade.

    Args:
        actual:  The scorecard's letter grade.
        minimum: The required minimum letter grade.

    Returns:
        True if actual rank < minimum rank.
    """
    return _GRADE_RANK.get(actual, 0) < _GRADE_RANK.get(minimum, 0)
