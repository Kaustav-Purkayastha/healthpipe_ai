"""
scripts/build_mart.py — CLI for building the reporting_state_health mart.

Usage:
    python scripts/build_mart.py              # fast: reads from cache
    python scripts/build_mart.py --refresh    # re-pulls all source data
    python scripts/build_mart.py --full-cms   # pulls all ~1.3M CMS providers
    python scripts/build_mart.py --no-ai      # skips AI briefing (template only)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.mart_builder import MartBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the reporting_state_health mart"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-pull all source data, ignoring cache.",
    )
    parser.add_argument(
        "--full-cms",
        action="store_true",
        help="Pull all ~1.3M CMS providers (slow; default is 300k sample).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip AI briefing — use template text only.",
    )
    parser.add_argument(
        "--measures",
        nargs="+",
        default=None,
        metavar="QUESTIONID",
        help="CDI question ids to include (default: DIA01 NPW14 TOB04). "
             "The three defaults keep friendly column names.",
    )
    args = parser.parse_args()

    print("\nHealthPipe AI v2 — Reporting Mart Builder")
    print("=" * 60)

    builder = MartBuilder()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    print("Building mart...")
    mart_df = builder.build(
        refresh=args.refresh, full_cms=args.full_cms, measures=args.measures
    )
    build_time = time.monotonic() - t0
    print(f"  Shape : {mart_df.shape}")
    print(f"  Time  : {build_time:.1f}s")

    # ------------------------------------------------------------------
    # Quality check
    # ------------------------------------------------------------------
    print("\nRunning quality checks...")
    scorecard = builder.quality_check(mart_df)
    print(f"  Grade : {scorecard['grade']} ({scorecard['score']}%)")
    print(f"  Passed: {scorecard['checks_passed']}/{scorecard['total_checks']}")

    failed = [c for c in scorecard["checks"] if not c["passed"]]
    if failed:
        print("  Failed checks:")
        for c in failed[:5]:
            print(f"    ✗ {c['check']}: {c['detail']}")

    # ------------------------------------------------------------------
    # Insight briefing
    # ------------------------------------------------------------------
    print("\nGenerating insight briefing...")
    if not args.no_ai:
        from core.router import AIRouter  # noqa: PLC0415
        router = AIRouter()
    else:
        router = None

    briefing = builder.insight_briefing(router, mart_df)
    facts = briefing["facts"]

    print("\nKey Facts:")
    top3 = [r["state_abbr"] for r in facts["top3_spend_per_capita"]]
    bot3 = [r["state_abbr"] for r in facts["bottom3_spend_per_capita"]]
    diab3 = [r["state_abbr"] for r in facts["top3_diabetes_prevalence"]]
    print(f"  Top 3 Medicare spend/capita  : {top3}")
    print(f"  Bottom 3 Medicare spend/capita: {bot3}")
    print(f"  Top 3 diabetes prevalence    : {diab3}")
    print(f"  Diabetes ↔ spend correlation : {facts.get('corr_diabetes_spend')}")
    print(
        f"  High burden / low spend      : {facts['high_burden_low_spend_count']} states "
        f"({', '.join(facts['high_burden_low_spend_states'][:5])}...)"
    )

    print(
        f"\nBriefing [{briefing['generated_by']}] ({briefing['latency_s']:.1f}s):"
    )
    for line in briefing["text"].splitlines():
        print(f"  {line}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print("\nSaving artifacts...")
    builder.to_duckdb(mart_df)
    print("  DuckDB table : reporting_state_health")
    print("  Parquet      : outputs/mart/reporting_state_health.parquet")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
