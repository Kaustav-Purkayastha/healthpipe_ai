"""
scripts/smoke_sources.py — API source connectivity and extract smoke test.

Runs connect() + a small extract() on every registered API source and
prints a results table.  Census prints SKIPPED cleanly when key is missing.

Exit 0 if all connected sources returned rows; exit 1 otherwise.

Usage:
    python scripts/smoke_sources.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.registry import SourceRegistry          # noqa: E402
from ingestion.cdc_cdi_source import CDCChronicDiseaseSource  # noqa: E402
from ingestion.brfss_source import BRFSSSource         # noqa: E402


def _row(name: str, connected: str, rows: str, cols: str, secs: str) -> None:
    print(f"  {name:<18} {connected:<10} {rows:<8} {cols:<8} {secs}")


def main() -> None:
    print("\nHealthPipe AI v2 — API Sources Smoke Test")
    print("=" * 65)
    print(f"  {'Source':<18} {'Connected':<10} {'Rows':<8} {'Cols':<8} {'Time'}")
    print("-" * 65)

    registry = SourceRegistry()
    api_sources = [
        s for s in registry._sources.values()
        if s.source_type == "api"
    ]

    failures: list[str] = []

    for source in api_sources:
        name = source.name

        # Census: skip gracefully when key is missing
        if name == "census":
            connected = source.connect()
            if not connected:
                _row(name, "SKIPPED", "-", "-", "-")
                continue

        t0 = time.monotonic()
        connected = source.connect()
        conn_str = "✓ yes" if connected else "✗ no"

        if not connected:
            _row(name, conn_str, "-", "-", f"{time.monotonic()-t0:.1f}s")
            continue

        # Small targeted extract per source
        try:
            if name == "cms_medicare":
                df = source.extract(state="MD", max_records=500)
            elif name == "cdc_cdi":
                df = source.extract(
                    question_id=CDCChronicDiseaseSource.DIABETES,
                    year="2023",
                    max_records=200,
                )
            elif name == "cdc_brfss":
                df = source.extract(
                    **BRFSSSource.OBESITY,
                    year="2023",
                    max_records=200,
                )
            elif name == "census":
                df = source.extract()
            elif name == "who":
                df = source.extract(indicator="life_expectancy", max_records=200)
            elif name == "openfda":
                df = source.extract(search_term="aspirin", max_records=50)
            else:
                df = source.extract(max_records=100)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            _row(name, conn_str, "ERROR", str(exc)[:20], f"{elapsed:.1f}s")
            failures.append(name)
            continue

        elapsed = time.monotonic() - t0
        rows = len(df) if df is not None else 0
        cols = len(df.columns) if df is not None and not df.empty else 0
        row_str = str(rows)
        col_str = str(cols)

        _row(name, conn_str, row_str, col_str, f"{elapsed:.1f}s")

        if rows == 0 and connected:
            print(f"    ⚠ WARNING: connected but 0 rows returned for {name}")
            failures.append(name)

    print("-" * 65)
    if failures:
        print(f"\nFAIL — {len(failures)} source(s) returned 0 rows or errored: {failures}\n")
        sys.exit(1)
    else:
        print("\nPASS — all connected sources returned data.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
