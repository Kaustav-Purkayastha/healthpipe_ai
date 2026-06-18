"""
main.py — CLI entry point for the HealthPipe AI pipeline.

Orchestrates the full data pipeline:
    1. Ingest data from a source (WHO API, OpenFDA API, or local file)
    2. Profile the raw data (statistics, outliers, correlations)
    3. Transform (clean, standardize, fill nulls)
    4. Quality check (score and grade the data)
    5. Document (data dictionary, schema, lineage → JSON + Markdown)
    6. Generate HTML report (self-contained dashboard)
    7. Load into DuckDB (analytics-ready table)

Usage examples:
    python main.py --source who --indicator life_expectancy --countries IND USA BRA
    python main.py --source openfda --search aspirin --max-records 200
    python main.py --source csv_local --filepath data/sample/chronic_disease_sample.csv
    python main.py --source who --use-crew --indicator life_expectancy
"""

import argparse
import re
import sys
import webbrowser
from pathlib import Path

# Add project root to import path so "from core.xxx" works when
# running directly with "python main.py"
sys.path.insert(0, str(Path(__file__).parent))

from core.utils import get_logger
from core.database import DuckDBManager
from core.report import ReportGenerator
from ingestion.registry import SourceRegistry
from agents.profiler import ProfilerAgent
from agents.transformer import TransformerAgent
from agents.quality_checker import QualityCheckerAgent
from agents.documenter import DocumenterAgent
from agents.crew import run_crew

logger = get_logger("main")


def sanitize_table_name(name: str) -> str:
    """
    Convert a dataset name into a valid DuckDB table name.

    DuckDB table names must be alphanumeric + underscores. This function
    replaces anything else with underscores and collapses duplicates.

    Args:
        name: Raw dataset name (e.g., "WHO Life-Expectancy (2020)").

    Returns:
        Sanitized name (e.g., "who_life_expectancy_2020").
    """
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)
    # Collapse multiple consecutive underscores into one
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores and lowercase
    return sanitized.strip("_").lower()


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Orchestrate the full pipeline: ingest → profile → transform → QC → document → load.

    Each step feeds its output into the next. Metadata from earlier steps
    is passed to the documenter for a complete lineage trail.

    Args:
        args: Parsed CLI arguments from argparse.
    """
    # Determine the dataset name — user-provided or default to source name
    dataset_name = args.name or args.source

    logger.info("=" * 60)
    logger.info(f"HealthPipe AI — Starting pipeline for '{dataset_name}'")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Ingest — Get data from the selected source
    # ------------------------------------------------------------------
    logger.info("\n[Step 1/7] Ingesting data...")

    registry = SourceRegistry()

    # Map the CLI source name to the registry name
    # "csv_local" in the CLI maps to "csv" in the registry
    source_map = {"who": "who", "openfda": "openfda", "csv_local": "csv"}
    registry_name = source_map.get(args.source, args.source)

    source = registry.get(registry_name)
    if source is None:
        logger.error(f"Source '{args.source}' not found in registry")
        return

    # Test connectivity before extracting
    if not source.connect():
        logger.error(f"Cannot connect to '{args.source}' — aborting pipeline")
        return

    # Build extraction kwargs based on the source type
    extract_kwargs = _build_extract_kwargs(args)
    raw_df = source.extract(**extract_kwargs)

    if raw_df.empty:
        logger.error("No data extracted — aborting pipeline")
        return

    logger.info(f"Ingested {len(raw_df)} rows, {len(raw_df.columns)} columns")

    # ------------------------------------------------------------------
    # Crew mode: delegate steps 2-6 to the multi-agent Crew orchestrator
    # ------------------------------------------------------------------
    if getattr(args, "use_crew", False):
        logger.info("\n[Crew Mode] Delegating to multi-agent orchestrator...")
        results = run_crew(
            raw_df,
            dataset_name,
            source_metadata=source.get_metadata(),
        )
        scorecard = results["scorecard"]
        table_name = sanitize_table_name(dataset_name)

        # Open the HTML report in the user's default browser
        crew_report_path = Path("outputs/reports") / f"report_{dataset_name}.html"
        if crew_report_path.exists():
            webbrowser.open(crew_report_path.resolve().as_uri())

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE (Crew Mode)")
        logger.info("=" * 60)
        logger.info(f"  Source:       {args.source}")
        logger.info(f"  Dataset:      {dataset_name}")
        logger.info(f"  Raw rows:     {len(raw_df)}")
        logger.info(f"  Clean rows:   {len(results['clean_df'])}")
        logger.info(f"  Quality:      {scorecard['score']}% (Grade {scorecard['grade']})")
        logger.info(f"  DuckDB table: {table_name}")
        logger.info(f"  HTML report:  outputs/reports/report_{dataset_name}.html")
        logger.info(f"  Reports:      outputs/reports/")
        logger.info(f"  Docs:         outputs/docs/")
        logger.info("=" * 60)
        return

    # ------------------------------------------------------------------
    # Step 2: Profile — Analyze the raw data
    # ------------------------------------------------------------------
    logger.info("\n[Step 2/7] Profiling raw data...")

    profiler = ProfilerAgent()
    profile = profiler.run(raw_df, dataset_name)

    overview = profile["overview"]
    logger.info(
        f"Profile: {overview['row_count']} rows, "
        f"{overview['completeness_score']}% complete, "
        f"{overview['duplicate_rows']} duplicates, "
        f"{len(profile['quality_issues'])} issues"
    )

    # ------------------------------------------------------------------
    # Step 3: Transform — Clean and standardize
    # ------------------------------------------------------------------
    logger.info("\n[Step 3/7] Transforming data...")

    transformer = TransformerAgent()
    clean_df = transformer.run(raw_df, dataset_name)
    transform_log = transformer.get_transform_summary()

    logger.info(
        f"Transform: {len(raw_df)} → {len(clean_df)} rows, "
        f"{len(transform_log)} steps applied"
    )

    # ------------------------------------------------------------------
    # Step 4: Quality Check — Score and grade
    # ------------------------------------------------------------------
    logger.info("\n[Step 4/7] Running quality checks...")

    checker = QualityCheckerAgent()
    scorecard = checker.run(clean_df, dataset_name)

    logger.info(
        f"Quality: {scorecard['score']}% (Grade {scorecard['grade']}) — "
        f"{scorecard['checks_passed']}/{scorecard['total_checks']} passed"
    )

    # ------------------------------------------------------------------
    # Step 5: Document — Generate data dictionary and docs
    # ------------------------------------------------------------------
    logger.info("\n[Step 5/7] Generating documentation...")

    documenter = DocumenterAgent()
    docs = documenter.run(
        clean_df,
        dataset_name,
        source_metadata=source.get_metadata(),
        profile_data=profile,
        transform_log=transform_log,
        quality_scorecard=scorecard,
    )

    logger.info(
        f"Docs: {len(docs['data_dictionary'])} columns documented, "
        f"{len(docs['usage_notes'])} usage notes"
    )

    # ------------------------------------------------------------------
    # Step 6: Generate HTML report
    # ------------------------------------------------------------------
    logger.info("\n[Step 6/7] Generating HTML report...")

    report_gen = ReportGenerator()
    report_path = report_gen.generate(
        dataset_name=dataset_name,
        overview=overview,
        scorecard=scorecard,
        data_dictionary=docs.get("data_dictionary", []),
        column_profiles=profile.get("columns", {}),
        transform_log=transform_log,
        quality_issues=profile.get("quality_issues", []),
        correlations=profile.get("correlations", []),
        usage_notes=docs.get("usage_notes", []),
    )

    logger.info(f"HTML report: {report_path}")

    # Open the report in the user's default browser automatically
    webbrowser.open(report_path.as_uri())

    # ------------------------------------------------------------------
    # Step 7: Load into DuckDB
    # ------------------------------------------------------------------
    logger.info("\n[Step 7/7] Loading into DuckDB...")

    table_name = sanitize_table_name(dataset_name)

    with DuckDBManager() as db:
        db.load_dataframe(clean_df, table_name)
        tables = db.list_tables()
        logger.info(f"DuckDB tables: {tables}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Source:       {args.source}")
    logger.info(f"  Dataset:      {dataset_name}")
    logger.info(f"  Raw rows:     {len(raw_df)}")
    logger.info(f"  Clean rows:   {len(clean_df)}")
    logger.info(f"  Quality:      {scorecard['score']}% (Grade {scorecard['grade']})")
    logger.info(f"  DuckDB table: {table_name}")
    logger.info(f"  HTML report:  {report_path}")
    logger.info(f"  Reports:      outputs/reports/")
    logger.info(f"  Docs:         outputs/docs/")
    logger.info("=" * 60)


def _build_extract_kwargs(args: argparse.Namespace) -> dict:
    """
    Build the keyword arguments for source.extract() based on CLI args.

    Each source type expects different parameters:
        - WHO: indicator, countries, max_records
        - OpenFDA: search_term, max_records
        - CSV: filepath

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dict of keyword arguments to pass to source.extract().
    """
    kwargs: dict = {}

    if args.source == "who":
        kwargs["indicator"] = args.indicator or "life_expectancy"
        kwargs["max_records"] = args.max_records
        if args.countries:
            kwargs["countries"] = args.countries

    elif args.source == "openfda":
        kwargs["search_term"] = args.search or "aspirin"
        kwargs["max_records"] = args.max_records

    elif args.source == "csv_local":
        if args.filepath:
            kwargs["filepath"] = args.filepath
        # Use chunked reading for large files (>10 MB)
        filepath = Path(args.filepath) if args.filepath else None
        if filepath and filepath.exists() and filepath.stat().st_size > 10_000_000:
            kwargs["chunk_size"] = 50000
            logger.info(
                f"Large file detected ({filepath.stat().st_size / 1_000_000:.1f} MB) "
                f"— using chunked reading"
            )

    return kwargs


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argparse CLI parser with all arguments and help text.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="healthpipe",
        description="HealthPipe AI — Multi-agent healthcare data pipeline",
        # epilog appears after the argument list in --help output
        epilog="""
examples:
  # WHO API — life expectancy for specific countries
  python main.py --source who --indicator life_expectancy --countries IND USA BRA

  # WHO API — all countries, limit to 500 records
  python main.py --source who --indicator neonatal_mortality --max-records 500

  # OpenFDA — adverse events for a specific drug
  python main.py --source openfda --search aspirin --max-records 200

  # Local CSV file
  python main.py --source csv_local --filepath data/sample/chronic_disease_sample.csv

  # Custom dataset name (used in reports, docs, and DuckDB table name)
  python main.py --source who --name my_who_data --indicator measles_immunization

  # Use multi-agent Crew mode with AI summaries (requires Ollama)
  python main.py --source who --use-crew --indicator life_expectancy
        """,
        # RawDescriptionHelpFormatter preserves the whitespace in epilog
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--source",
        required=True,
        choices=["who", "openfda", "csv_local"],
        help="Data source to ingest from",
    )

    # Optional arguments
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom dataset name (default: same as source name). "
             "Used in reports, docs, and DuckDB table name.",
    )

    # WHO-specific arguments
    parser.add_argument(
        "--indicator",
        type=str,
        default="life_expectancy",
        help="WHO indicator name or code "
             "(choices: life_expectancy, neonatal_mortality, "
             "tuberculosis_incidence, measles_immunization). "
             "Default: life_expectancy",
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        type=str,
        default=None,
        help="ISO 3-letter country codes to filter "
             "(e.g., IND USA BRA CHN ZAF). Default: all countries.",
    )

    # OpenFDA-specific arguments
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="OpenFDA drug search term (e.g., aspirin, ibuprofen). "
             "Default: aspirin",
    )

    # CSV-specific arguments
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="Path to a local CSV, TSV, or JSON file. "
             "Default: data/sample/chronic_disease_sample.csv",
    )

    # Shared arguments
    parser.add_argument(
        "--max-records",
        type=int,
        default=5000,
        help="Maximum number of records to fetch from APIs. "
             "Default: 5000. Ignored for local files.",
    )

    # AI orchestration mode
    parser.add_argument(
        "--use-crew",
        action="store_true",
        default=False,
        help="Use multi-agent Crew orchestration with AI-generated summaries "
             "(requires Ollama running locally). Default: direct agent calls.",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate source-specific arguments
    if args.source == "csv_local" and args.filepath:
        filepath = Path(args.filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
