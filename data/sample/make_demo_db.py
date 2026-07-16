"""
data/sample/make_demo_db.py — Builds data/sample/demo_clinic.db for demos.

Deterministically seeded with random.seed(42) so every run produces the same
data — important for reproducible smoke tests and live demo walkthroughs.

Tables:
  patients  (200 rows) — patient demographics + insurance plan
  claims    (1000 rows) — insurance claim transactions
  providers (50 rows)  — provider NPI + specialty

Run directly:
    python data/sample/make_demo_db.py
"""

from __future__ import annotations

import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on sys.path so core imports work when run directly.
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH: Path = Path(__file__).parent / "demo_clinic.db"

# Deterministic seed — comment: random.seed(42) guarantees the same synthetic
# data every run, making smoke tests and demos fully reproducible.
random.seed(42)


def _rand_date(start: date, end: date) -> str:
    """Return a random ISO date string between start and end (inclusive)."""
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def build_demo_db(db_path: Path = DB_PATH) -> dict[str, int]:
    """Create (or overwrite) the demo SQLite clinic database.

    Args:
        db_path: Destination .db file path.

    Returns:
        Dict of {table_name: row_count} for confirmation output.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # always rebuild fresh for reproducibility

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # ------------------------------------------------------------------
    # patients (200 rows)
    # ------------------------------------------------------------------
    cur.execute("""
        CREATE TABLE patients (
            patient_id   TEXT PRIMARY KEY,
            age          INTEGER,
            state        TEXT,
            plan_type    TEXT,
            enrolled_date TEXT
        )
    """)

    states_weighted = (
        ["CA"] * 30 + ["TX"] * 25 + ["FL"] * 20 + ["NY"] * 20 +
        ["OH"] * 15 + ["PA"] * 15 + ["IL"] * 15 + ["GA"] * 10
    )
    plan_types = ["HMO", "PPO", "EPO"]
    enroll_start = date(2019, 1, 1)
    enroll_end = date(2024, 6, 30)

    patients: list[tuple] = []
    for i in range(1, 201):
        pid = f"P{i:04d}"
        # Plant 3 intentionally negative ages (for quality-check demo)
        if i in (17, 53, 142):
            age = random.randint(-5, -1)
        else:
            age = random.randint(18, 89)
        state = random.choice(states_weighted)
        plan = random.choice(plan_types)
        enrolled = _rand_date(enroll_start, enroll_end)
        patients.append((pid, age, state, plan, enrolled))

    cur.executemany(
        "INSERT INTO patients VALUES (?,?,?,?,?)", patients
    )

    # ------------------------------------------------------------------
    # providers (50 rows)
    # ------------------------------------------------------------------
    cur.execute("""
        CREATE TABLE providers (
            npi        TEXT PRIMARY KEY,
            specialty  TEXT,
            state      TEXT
        )
    """)

    specialties = [
        "Internal Medicine", "Family Medicine", "Cardiology",
        "Orthopedics", "Oncology", "Neurology", "Psychiatry",
        "Radiology", "Anesthesiology", "Emergency Medicine",
    ]
    providers: list[tuple] = []
    used_npis: set[str] = set()
    for i in range(50):
        while True:
            npi = str(random.randint(1_000_000_000, 9_999_999_999))
            if npi not in used_npis:
                used_npis.add(npi)
                break
        spec = random.choice(specialties)
        state = random.choice(states_weighted)
        providers.append((npi, spec, state))

    cur.executemany("INSERT INTO providers VALUES (?,?,?)", providers)

    # ------------------------------------------------------------------
    # claims (1000 rows)
    # ------------------------------------------------------------------
    cur.execute("""
        CREATE TABLE claims (
            claim_id          TEXT PRIMARY KEY,
            patient_id        TEXT REFERENCES patients(patient_id),
            procedure_code    TEXT,
            billed_amount     REAL,
            allowed_amount    REAL,
            paid_amount       REAL,
            claim_date        TEXT,
            status            TEXT
        )
    """)

    procedure_codes = [
        "99213", "99214", "99215",  # Office visits
        "93000", "93010",            # ECG
        "71046",                     # Chest X-ray
        "85025",                     # CBC
        "80053",                     # Comprehensive metabolic panel
        "36415",                     # Venipuncture
        "99283",                     # ED visit
    ]
    claim_start = date(2020, 1, 1)
    claim_end = date(2024, 12, 31)
    patient_ids = [p[0] for p in patients]

    claims: list[tuple] = []
    for i in range(1, 1001):
        cid = f"CLM{i:06d}"
        pid = random.choice(patient_ids)
        proc = random.choice(procedure_codes)
        billed = round(random.uniform(50, 2500), 2)
        allowed = round(billed * random.uniform(0.60, 0.80), 2)  # ≈70% of billed
        paid = round(allowed * random.uniform(0.80, 0.90), 2)     # ≈85% of allowed
        cdate = _rand_date(claim_start, claim_end)
        # Status: ~8% DENIED, ~7% PENDING, rest PAID
        r = random.random()
        if r < 0.08:
            status = "DENIED"
        elif r < 0.15:
            status = "PENDING"
        else:
            status = "PAID"
        claims.append((cid, pid, proc, billed, allowed, paid, cdate, status))

    cur.executemany(
        "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?)", claims
    )

    conn.commit()
    conn.close()

    counts = {
        "patients": len(patients),
        "providers": len(providers),
        "claims": len(claims),
    }
    return counts


if __name__ == "__main__":
    counts = build_demo_db()
    for table, n in counts.items():
        print(f"  {table:<12} {n} rows")
    print(f"\nDemo DB written to: {DB_PATH}")
