"""
run_local.py — Mini CLI for running the HealthPipe AI v2 pipeline locally.

Usage:
    python run_local.py --file data/sample/test_fixture.csv
    python run_local.py --file data/sample/test_fixture.parquet --no-ai
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.file_source import FileSource  # noqa: E402
from core.pipeline import run_pipeline         # noqa: E402


def _progress(step: str, status: str) -> None:
    """Print-based progress callback — one line per stage transition."""
    arrow = ">>>" if status == "starting" else "   "
    icon = "..." if status == "starting" else "done"
    print(f"  {arrow} [{icon}] {step}")


def main() -> None:
    """Parse args, ingest the file, run the pipeline, and print a summary."""
    parser = argparse.ArgumentParser(
        description="HealthPipe AI v2 — local pipeline runner",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to a CSV, TSV, JSON, Parquet, or XLSX file to process.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Registry source name to extract from (e.g. cms_medicare, who, census).",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="State filter passed to API sources that support it (2-letter code).",
    )
    parser.add_argument(
        "--year",
        default=None,
        help="Year filter passed to API sources that support it.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum records to extract from an API source.",
    )
    parser.add_argument(
        "--min-grade",
        default=None,
        help="Quality gate: skip DuckDB load if grade is below this (A/B/C).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        default=False,
        help="Skip AI enrichment and use rule-based descriptions/briefing only.",
    )
    args = parser.parse_args()

    file_path = Path(args.file) if args.file else None
    if file_path is None and args.source is None:
        print("ERROR: provide --file or --source", file=sys.stderr)
        sys.exit(1)

    if file_path is not None and not file_path.exists():
        print(f"ERROR: File not found — {file_path}", file=sys.stderr)
        sys.exit(1)

    # Derive dataset name
    if file_path:
        dataset_name = file_path.stem
    else:
        dataset_name = args.source

    print(f"\nHealthPipe AI v2 — Local Pipeline")
    print(f"  File    : {file_path}")
    print(f"  Dataset : {dataset_name}")
    if args.min_grade:
        print(f"  Quality gate: min grade = {args.min_grade}")
    print(f"  AI mode : {'disabled (--no-ai)' if args.no_ai else 'enabled'}")
    print()

    # ----------------------------------------------------------------
    # Ingest (file or API source)
    # ----------------------------------------------------------------
    print("Ingesting...")
    if file_path:
        source = FileSource()
        df = source.extract(filepath=str(file_path))
        source_meta = source.get_metadata()
    else:
        from ingestion.registry import SourceRegistry
        registry = SourceRegistry()
        source = registry.get(args.source)
        if source is None:
            print(f"ERROR: Source '{args.source}' not found in registry.", file=sys.stderr)
            sys.exit(1)
        extract_kwargs: dict = {}
        if args.state:
            extract_kwargs["state"] = args.state
        if args.year:
            extract_kwargs["year"] = args.year
        if args.max_records:
            extract_kwargs["max_records"] = args.max_records
        df = source.extract(**extract_kwargs)
        source_meta = source.get_metadata()

    if df.empty:
        print("ERROR: Source produced an empty DataFrame — check source/filters.",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")
    print()

    # ----------------------------------------------------------------
    # Build AIRouter (unless --no-ai)
    # ----------------------------------------------------------------
    router = None
    if not args.no_ai:
        from core.router import AIRouter  # noqa: E402
        router = AIRouter()

    # ----------------------------------------------------------------
    # Run pipeline
    # ----------------------------------------------------------------
    print("Running pipeline stages:")
    result = run_pipeline(
        df=df,
        dataset_name=dataset_name,
        source_metadata=source_meta,
        progress_callback=_progress,
        quality_gate_min_grade=args.min_grade or None,
        router=router,
        enable_ai_enrichment=not args.no_ai,
    )
    print()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    clean_df = result["clean_df"]
    scorecard = result["scorecard"] or {}
    profile = result["profile"] or {}
    pii = profile.get("pii_columns", [])

    print("=" * 56)
    print("PIPELINE SUMMARY")
    print("=" * 56)
    print(f"  Rows in            : {len(df)}")
    print(f"  Rows out (clean)   : {len(clean_df) if clean_df is not None else 'N/A'}")
    print(f"  Quality grade      : {scorecard.get('grade', 'N/A')}  "
          f"({scorecard.get('score', 'N/A')}%)")
    print(f"  PII columns found  : {len(pii)}")
    for p in pii:
        print(f"    - {p['column']}  ({p['reason']}, {p['confidence']})")

    if result.get("gate_blocked"):
        print(f"  Quality gate       : BLOCKED — DuckDB load skipped")
    else:
        print(f"  DuckDB table       : {result.get('table_name', 'N/A')}")

    # ---- AI Briefing ----
    briefing = result.get("briefing")
    if briefing:
        print()
        print(f"  AI Briefing [{briefing.get('generated_by', '')}] "
              f"({briefing.get('latency_s', 0):.1f}s):")
        for line in briefing.get("text", "").splitlines():
            print(f"    {line}")

    # ---- First issue explanation ----
    issue_exps = result.get("issue_explanations") or []
    if issue_exps:
        first = issue_exps[0]
        print()
        print(f"  Top issue explanation [{first.get('generated_by', '')}]:")
        check = first.get("issue", {}).get("check", "")
        print(f"    Issue  : {check}")
        print(f"    Why    : {first.get('explanation', '')[:200]}")
        print(f"    Fix    : {first.get('suggested_fix', '')}")

    print("=" * 56)
    print()


if __name__ == "__main__":
    main()

