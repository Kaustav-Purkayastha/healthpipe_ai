"""
scripts/smoke_db.py — End-to-end DatabaseSource + pipeline smoke test.

Sequence:
  1. Build demo_clinic.db (if missing)
  2. DatabaseSource.configure(sqlite) → connect → list_tables (expect 3)
  3. extract(table="claims")
  4. run_pipeline on the claims DataFrame
  5. Print quality grade + DuckDB table name

This is the live-demo dress rehearsal for the DB ingestion lane.

Usage:
    python scripts/smoke_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline import run_pipeline           # noqa: E402
from ingestion.database_source import DatabaseSource  # noqa: E402

DEMO_DB = ROOT / "data" / "sample" / "demo_clinic.db"


def _ensure_demo_db() -> None:
    """Build demo_clinic.db if it doesn't already exist."""
    if DEMO_DB.exists():
        return
    print("  demo_clinic.db not found — building it...")
    sys.path.insert(0, str(ROOT / "data" / "sample"))
    from make_demo_db import build_demo_db  # noqa: PLC0415
    counts = build_demo_db(DEMO_DB)
    for tbl, n in counts.items():
        print(f"    {tbl}: {n} rows")
    print()


def main() -> None:
    print("\nHealthPipe AI v2 — Database Source Smoke Test")
    print("=" * 56)

    _ensure_demo_db()

    src = DatabaseSource()
    src.configure("sqlite", database=str(DEMO_DB))

    # ----------------------------------------------------------------
    # 1. Connect
    # ----------------------------------------------------------------
    print("Connecting to SQLite demo_clinic.db ...")
    ok = src.connect()
    print(f"  connected: {ok}")
    if not ok:
        print("ERROR: could not connect — aborting")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2. List tables
    # ----------------------------------------------------------------
    tables = src.list_tables()
    print(f"  tables: {tables}")
    assert set(tables) == {"patients", "claims", "providers"}, (
        f"Expected 3 tables, got: {tables}"
    )

    # ----------------------------------------------------------------
    # 3. Extract claims
    # ----------------------------------------------------------------
    print("\nExtracting claims table...")
    df = src.extract(table="claims")
    print(f"  rows: {len(df)}  cols: {len(df.columns)}")
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"

    # ----------------------------------------------------------------
    # 4. Run pipeline
    # ----------------------------------------------------------------
    print("\nRunning pipeline on claims...")
    from core.config import OUTPUTS_DIR  # noqa: PLC0415
    db_path = OUTPUTS_DIR / "demo_clinic_pipeline.duckdb"
    result = run_pipeline(
        df,
        dataset_name="demo_claims",
        source_metadata=src.get_metadata(),
        db_path=db_path,
        router=None,
    )

    scorecard = result.get("scorecard") or {}
    grade = scorecard.get("grade", "N/A")
    score = scorecard.get("score", "N/A")
    table_name = result.get("table_name", "N/A")
    clean_rows = len(result["clean_df"]) if result.get("clean_df") is not None else "N/A"

    print()
    print("=" * 56)
    print("SMOKE DB SUMMARY")
    print("=" * 56)
    print(f"  Rows extracted   : {len(df)}")
    print(f"  Rows after clean : {clean_rows}")
    print(f"  Quality grade    : {grade} ({score}%)")
    print(f"  DuckDB table     : {table_name}")
    print("=" * 56)
    print("\nPASS — database lane end-to-end complete.\n")


if __name__ == "__main__":
    main()
