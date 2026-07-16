"""
core/history.py — Pipeline run recording and retrieval.

Every onboarding run is recorded in a DuckDB table `pipeline_runs` and its
artifact JSONs are saved to `outputs/runs/{run_id}/`.  The Dashboard and
Run History screens read from here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import OUTPUTS_DIR
from core.utils import get_logger, load_json, save_json, timestamp_string

_log = get_logger(__name__)

_RUNS_DIR: Path = OUTPUTS_DIR / "runs"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          VARCHAR,
    dataset_name    VARCHAR,
    source_name     VARCHAR,
    started_at      VARCHAR,
    duration_s      DOUBLE,
    rows_in         INTEGER,
    rows_out        INTEGER,
    grade           VARCHAR,
    score           DOUBLE,
    gate_blocked    BOOLEAN,
    table_name      VARCHAR
)
"""


def record_run(db, run_meta: dict, artifacts: dict) -> str:
    """Record one pipeline run in DuckDB and save artifact JSONs.

    Args:
        db:         Open DuckDBManager connection.
        run_meta:   Dict with keys: dataset_name, source_name, started_at,
                    duration_s, rows_in, rows_out, grade, score,
                    gate_blocked, table_name.
        artifacts:  Dict of optional artifact dicts — any subset of:
                    profile, scorecard, docs, briefing, issue_explanations,
                    starter_questions.

    Returns:
        run_id string (e.g. "run_20260711_143022").
    """
    run_id = f"run_{timestamp_string()}"

    # Ensure the table exists
    db._conn.execute(_CREATE_TABLE_SQL)

    db._conn.execute(
        "INSERT INTO pipeline_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            run_id,
            run_meta.get("dataset_name", ""),
            run_meta.get("source_name", ""),
            run_meta.get("started_at", ""),
            float(run_meta.get("duration_s", 0.0)),
            int(run_meta.get("rows_in", 0)),
            int(run_meta.get("rows_out", 0)),
            run_meta.get("grade", ""),
            float(run_meta.get("score", 0.0)),
            bool(run_meta.get("gate_blocked", False)),
            run_meta.get("table_name", ""),
        ],
    )

    # Save artifact JSONs
    run_dir = _RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for key in ["profile", "scorecard", "docs", "briefing", "issue_explanations", "starter_questions"]:
        value = artifacts.get(key)
        if value is not None:
            try:
                save_json(value, run_dir / f"{key}.json")
            except Exception as exc:
                _log.warning("Failed to save artifact '%s' for run %s: %s", key, run_id, exc)

    _log.info("Recorded run %s (grade=%s rows=%s)", run_id, run_meta.get("grade"), run_meta.get("rows_out"))
    return run_id


def list_runs(db, limit: int = 50) -> pd.DataFrame:
    """Return the most recent pipeline runs, newest first.

    Args:
        db:    Open DuckDBManager.
        limit: Maximum rows to return.

    Returns:
        DataFrame with all pipeline_runs columns, or empty DataFrame if the
        table doesn't exist yet.
    """
    try:
        return db.query(
            f"SELECT * FROM pipeline_runs "
            f"ORDER BY started_at DESC LIMIT {int(limit)}"
        )
    except Exception:
        # Table doesn't exist yet — no runs recorded
        return pd.DataFrame()


def save_artifact(run_id: str, key: str, value: object) -> None:
    """Persist a single artifact JSON for a run, overwriting any existing file.

    Used by screens that regenerate one artifact on demand (e.g. Issue Triage
    explaining a single issue) — keeps the run's artifact directory the single
    source of truth so a page refresh shows the updated content.

    Args:
        run_id: The run identifier returned by record_run().
        key:    Artifact key (one of profile/scorecard/docs/briefing/
                issue_explanations).
        value:  JSON-serialisable artifact payload.
    """
    run_dir = _RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(value, run_dir / f"{key}.json")


def load_artifacts(run_id: str) -> dict:
    """Load saved artifact JSONs for a given run_id.

    Args:
        run_id: The run identifier returned by record_run().

    Returns:
        Dict with whichever artifact keys were saved:
        profile, scorecard, docs, briefing, issue_explanations.
        Missing keys are silently omitted.
    """
    run_dir = _RUNS_DIR / run_id
    result: dict = {}

    for key in ["profile", "scorecard", "docs", "briefing", "issue_explanations", "starter_questions"]:
        path = run_dir / f"{key}.json"
        if path.exists():
            try:
                result[key] = load_json(path)
            except Exception as exc:
                _log.warning("Failed to load artifact '%s' for run %s: %s", key, run_id, exc)

    return result
