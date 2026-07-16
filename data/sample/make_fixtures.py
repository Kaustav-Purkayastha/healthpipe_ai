"""
data/sample/make_fixtures.py — One-time fixture generator for HealthPipe AI v2 tests.

Builds a 20-row "fake patient" DataFrame with deliberate data-quality issues
baked in (negative age, mixed-case state codes, None diagnoses, duplicate row,
zero cost) so that later quality-check steps have something interesting to find.

Run directly to regenerate fixtures:
    python data/sample/make_fixtures.py

The ``build_fixtures()`` function is importable so tests/conftest.py can call
it automatically when the fixture files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Where the generated files land.
SAMPLE_DIR: Path = Path(__file__).parent


def build_fixtures(output_dir: Path = SAMPLE_DIR) -> dict[str, Path]:
    """Build the fake-patient DataFrame and write it in four formats.

    The DataFrame intentionally contains:
    - One negative age (-5)  → catches range-validation bugs.
    - Mixed-case state codes ("ny" vs "NY")  → catches normalisation bugs.
    - Three None diagnoses  → catches null-handling bugs.
    - One zero cost  → catches zero-value handling.
    - Row 20 is an exact duplicate of row 19  → catches dedup bugs.

    Args:
        output_dir: Directory to write the fixture files into.
                    Defaults to the same directory as this script.

    Returns:
        Dict mapping format name to the written Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build the raw rows (19 unique + 1 duplicate)
    # ------------------------------------------------------------------

    rows: list[dict] = [
        # patient_id  age   state  diagnosis             visit_date    cost_usd
        {"patient_id": "P001", "age": 34,  "state": "AL", "diagnosis": "Hypertension",    "visit_date": "2024-01-05", "cost_usd": 120.50},
        {"patient_id": "P002", "age": 52,  "state": "CA", "diagnosis": "Diabetes",        "visit_date": "2024-01-07", "cost_usd": 340.00},
        {"patient_id": "P003", "age": 28,  "state": "ny", "diagnosis": "Asthma",          "visit_date": "2024-01-09", "cost_usd": 88.75},  # lowercase state
        {"patient_id": "P004", "age": 67,  "state": "TX", "diagnosis": None,              "visit_date": "2024-01-11", "cost_usd": 512.00},  # None diagnosis
        {"patient_id": "P005", "age": -5,  "state": "FL", "diagnosis": "Obesity",         "visit_date": "2024-01-14", "cost_usd": 200.00},  # negative age
        {"patient_id": "P006", "age": 44,  "state": "GA", "diagnosis": "COPD",            "visit_date": "2024-01-16", "cost_usd": 275.00},
        {"patient_id": "P007", "age": 39,  "state": "OH", "diagnosis": "Depression",      "visit_date": "2024-01-18", "cost_usd": 0.00},    # zero cost
        {"patient_id": "P008", "age": 71,  "state": "PA", "diagnosis": None,              "visit_date": "2024-01-20", "cost_usd": 640.00},  # None diagnosis
        {"patient_id": "P009", "age": 55,  "state": "Il", "diagnosis": "Heart Disease",   "visit_date": "2024-01-22", "cost_usd": 890.00},  # mixed case state
        {"patient_id": "P010", "age": 31,  "state": "MI", "diagnosis": "Stroke",          "visit_date": "2024-01-24", "cost_usd": 1200.00},
        {"patient_id": "P011", "age": 48,  "state": "NC", "diagnosis": "Alzheimer's",     "visit_date": "2024-01-26", "cost_usd": 430.00},
        {"patient_id": "P012", "age": 60,  "state": "VA", "diagnosis": "Cancer",          "visit_date": "2024-01-28", "cost_usd": 2500.00},
        {"patient_id": "P013", "age": 22,  "state": "AZ", "diagnosis": None,              "visit_date": "2024-01-30", "cost_usd": 55.00},   # None diagnosis
        {"patient_id": "P014", "age": 37,  "state": "CO", "diagnosis": "Arthritis",       "visit_date": "2024-02-01", "cost_usd": 310.00},
        {"patient_id": "P015", "age": 43,  "state": "WA", "diagnosis": "Hypertension",    "visit_date": "2024-02-03", "cost_usd": 145.00},
        {"patient_id": "P016", "age": 58,  "state": "MN", "diagnosis": "Diabetes",        "visit_date": "2024-02-05", "cost_usd": 375.00},
        {"patient_id": "P017", "age": 29,  "state": "OR", "diagnosis": "Asthma",          "visit_date": "2024-02-07", "cost_usd": 95.00},
        {"patient_id": "P018", "age": 76,  "state": "NV", "diagnosis": "COPD",            "visit_date": "2024-02-09", "cost_usd": 780.00},
        # Row 19 — will be duplicated as row 20
        {"patient_id": "P019", "age": 50,  "state": "DC", "diagnosis": "Depression",      "visit_date": "2024-02-11", "cost_usd": 220.00},
        # Row 20 — exact duplicate of row 19 (for dedup testing)
        {"patient_id": "P019", "age": 50,  "state": "DC", "diagnosis": "Depression",      "visit_date": "2024-02-11", "cost_usd": 220.00},
    ]

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Write in all four formats
    # ------------------------------------------------------------------

    paths: dict[str, Path] = {}

    csv_path = output_dir / "test_fixture.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    json_path = output_dir / "test_fixture.json"
    df.to_json(json_path, orient="records", indent=2)
    paths["json"] = json_path

    parquet_path = output_dir / "test_fixture.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    paths["parquet"] = parquet_path

    xlsx_path = output_dir / "test_fixture.xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    paths["xlsx"] = xlsx_path

    return paths


if __name__ == "__main__":
    written = build_fixtures()
    for fmt, path in written.items():
        print(f"  [{fmt}] {path}")
    print("Done — 4 fixture files written.")
