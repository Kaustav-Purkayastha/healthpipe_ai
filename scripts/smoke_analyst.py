"""
scripts/smoke_analyst.py — Analyst engine smoke test for HealthPipe AI v2.

Loads the test_fixture table (running the pipeline first if missing), then
runs 6 benchmark questions through ask() and prints a results table.

Exit code 0 if >=5/6 questions executed successfully, else 1.

Usage:
    python scripts/smoke_analyst.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analyst import ask, starter_questions  # noqa: E402
from core.config import DATABASE_PATH             # noqa: E402
from core.database import DuckDBManager           # noqa: E402
from core.router import AIRouter                  # noqa: E402

TABLE = "test_fixture"

QUESTIONS = [
    "How many rows are there?",
    "What are the distinct diagnosis values?",
    "Which state has the most records?",
    "What is the average cost_usd per state?",
    "Top 3 diagnoses by count",
    "Which patients have a negative age?",
]


def _ensure_table(db: DuckDBManager) -> None:
    """Run the pipeline to load test_fixture if it doesn't exist yet."""
    tables = db.list_tables()
    if TABLE in tables:
        return

    print(f"  Table '{TABLE}' not found — running pipeline to create it...")
    from core.pipeline import run_pipeline
    from ingestion.file_source import FileSource
    from core.config import SAMPLE_DIR

    csv_path = SAMPLE_DIR / "test_fixture.csv"
    if not csv_path.exists():
        from data.sample.make_fixtures import build_fixtures
        build_fixtures()

    df = FileSource().extract(filepath=str(csv_path))
    run_pipeline(df, TABLE, db_path=DATABASE_PATH)
    print(f"  Pipeline complete — '{TABLE}' loaded.\n")


def _truncate(s: str, n: int = 55) -> str:
    """Truncate a string to n chars for table display."""
    s = s.replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


def main() -> None:
    # Windows consoles / piped stdout default to cp1252, which can't encode the
    # ✓/✗ marks and the em-dash in this table — force UTF-8 so the smoke doesn't
    # crash on the target platform.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    print("\nHealthPipe AI v2 — Analyst Engine Smoke Test")
    print("=" * 70)

    router = AIRouter()
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DuckDBManager(db_path=DATABASE_PATH) as db:
        _ensure_table(db)

        # Print starter questions as a bonus check
        print("Starter questions suggested by local model:")
        starters = starter_questions(router, db, TABLE)
        for i, q in enumerate(starters, 1):
            print(f"  {i}. {q}")
        print()

        # Header
        print(f"{'Question':<40} {'Valid':<6} {'Rows':<6} {'Ret':<4} {'Lat':>6}  SQL")
        print("-" * 70)

        successes = 0

        for question in QUESTIONS:
            result = ask(router, db, TABLE, question)

            rows = len(result["df"]) if result["df"] is not None else "-"
            valid_str = "✓" if result["valid"] else "✗"
            lat_str = f"{result['latency_s']:.1f}s"
            sql_short = _truncate(result["sql"] or result["error"] or "")
            q_short = _truncate(question, 38)

            print(
                f"{q_short:<40} {valid_str:<6} {str(rows):<6} "
                f"{result['retries']:<4} {lat_str:>6}  {sql_short}"
            )

            if result["df"] is not None:
                successes += 1

            # Warn on semantic red flags
            if result["valid"] and result["sql"]:
                sql_up = result["sql"].upper()
                if "per" in question.lower() or "by" in question.lower():
                    if "GROUP BY" not in sql_up:
                        print(
                            f"  ⚠ WARNING: question contains 'per'/'by' but SQL "
                            f"has no GROUP BY — possible semantic error"
                        )

            if result["narration"]:
                print(f"  → {result['narration'][:120]}")

        print("-" * 70)
        print(f"\nResult: {successes}/{len(QUESTIONS)} executed successfully.")

        if successes >= 5:
            print("PASS — analyst engine is functional.\n")
            sys.exit(0)
        else:
            print("FAIL — fewer than 5/6 questions succeeded.\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
