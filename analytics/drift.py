"""
analytics/drift.py — Schema-drift comparison between two historical runs.

Answers "what changed between loads?" — the question a data engineer asks when
an upstream feed silently changes shape.  Pure functions over two runs' saved
artifacts (profile + scorecard); NO database access, because historical DuckDB
tables get replaced on re-load while the artifacts are immutable snapshots.
"""

from __future__ import annotations

# How many null-rate deltas to surface (largest absolute change first).
_TOP_NULL_DELTAS = 10


def _columns(artifacts: dict) -> dict:
    """Return the profile's per-column stats dict ({} when absent)."""
    return (artifacts.get("profile") or {}).get("columns") or {}


def _row_count(artifacts: dict):
    """Return the profiled row count, or None when unavailable."""
    return ((artifacts.get("profile") or {}).get("overview") or {}).get("row_count")


def compare_runs(artifacts_a: dict, artifacts_b: dict) -> dict:
    """Compare two runs' artifacts and return a structured drift summary.

    Args:
        artifacts_a: The BEFORE run's artifacts (profile + scorecard).
        artifacts_b: The AFTER run's artifacts.

    Returns:
        Dict with keys:
          added_columns    — columns in B not in A (sorted)
          removed_columns  — columns in A not in B (sorted)
          type_changes     — [{column, from, to}] for common columns whose dtype differs
          null_rate_deltas — [{column, from_pct, to_pct, delta}], top 10 by |delta|
          row_delta        — B rows − A rows (None if either unknown)
          score_delta      — B score − A score (None if either unknown)
          grade_change     — "A→B" style string, or None when unchanged/unknown
    """
    cols_a = _columns(artifacts_a)
    cols_b = _columns(artifacts_b)
    names_a = set(cols_a)
    names_b = set(cols_b)

    added_columns = sorted(names_b - names_a)
    removed_columns = sorted(names_a - names_b)
    common = names_a & names_b

    # Type changes on shared columns.
    type_changes: list[dict] = []
    for col in sorted(common):
        from_type = str(cols_a[col].get("dtype", ""))
        to_type = str(cols_b[col].get("dtype", ""))
        if from_type != to_type:
            type_changes.append({"column": col, "from": from_type, "to": to_type})

    # Null-rate deltas on shared columns — only non-zero changes, top-N by |delta|.
    null_rate_deltas: list[dict] = []
    for col in common:
        from_pct = float(cols_a[col].get("null_percentage", 0.0) or 0.0)
        to_pct = float(cols_b[col].get("null_percentage", 0.0) or 0.0)
        delta = round(to_pct - from_pct, 2)
        if abs(delta) > 1e-9:
            null_rate_deltas.append({
                "column": col,
                "from_pct": from_pct,
                "to_pct": to_pct,
                "delta": delta,
            })
    null_rate_deltas.sort(key=lambda d: abs(d["delta"]), reverse=True)
    null_rate_deltas = null_rate_deltas[:_TOP_NULL_DELTAS]

    # Row delta.
    rows_a = _row_count(artifacts_a)
    rows_b = _row_count(artifacts_b)
    row_delta = (
        rows_b - rows_a
        if isinstance(rows_a, (int, float)) and isinstance(rows_b, (int, float))
        else None
    )

    # Score / grade from the scorecard artifact.
    sc_a = artifacts_a.get("scorecard") or {}
    sc_b = artifacts_b.get("scorecard") or {}
    score_a = sc_a.get("score")
    score_b = sc_b.get("score")
    score_delta = (
        round(float(score_b) - float(score_a), 2)
        if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float))
        else None
    )
    grade_a = sc_a.get("grade")
    grade_b = sc_b.get("grade")
    grade_change = (
        f"{grade_a}→{grade_b}"
        if grade_a and grade_b and grade_a != grade_b
        else None
    )

    return {
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "type_changes": type_changes,
        "null_rate_deltas": null_rate_deltas,
        "row_delta": row_delta,
        "score_delta": score_delta,
        "grade_change": grade_change,
    }


def has_drift(diff: dict) -> bool:
    """Return True when *diff* (from compare_runs) shows any schema-level change.

    Row/score deltas alone are treated as normal variation, not schema drift —
    this mirrors what a DE means by "did the shape change?".
    """
    return bool(
        diff.get("added_columns")
        or diff.get("removed_columns")
        or diff.get("type_changes")
        or diff.get("null_rate_deltas")
    )
