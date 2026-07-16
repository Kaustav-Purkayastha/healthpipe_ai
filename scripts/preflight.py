"""
scripts/preflight.py — the ONE demo-readiness command.

Run before a demo:  ``python scripts/preflight.py``

Prints a table of ✅/⚠️/❌ rows, each with a fix hint.  Exit code is 0 unless a
hard failure (❌) exists, so it doubles as a CI/local gate.

Status meanings:
    ✅ ok    — ready.
    ⚠️  warn  — optional/degraded; the app still runs (fallbacks or cached data).
    ❌ fail  — hard blocker; fix before demoing.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Make the repo importable when run as `python scripts/preflight.py`.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK = "ok"
WARN = "warn"
FAIL = "fail"

_SYMBOL = {OK: "✅", WARN: "⚠️", FAIL: "❌"}

# Exact versions the app was pinned + verified against (requirements-core.txt).
# duckdb is a floor (>=1.1.0), so it is checked for import + a version only.
_EXPECTED_PINS: dict[str, str] = {
    "pandas": "3.0.3",
    "pyarrow": "25.0.0",
    "flask": "3.0.0",
}
_TESTED_PYTHON = (3, 14, 3)
_MIN_PYTHON = (3, 11)


@dataclass
class Check:
    """One preflight row.

    Attributes:
        status: OK / WARN / FAIL.
        name:   Short check name (left column).
        detail: What was found (middle column).
        hint:   How to fix it (only meaningful for WARN/FAIL).
    """

    status: str
    name: str
    detail: str
    hint: str = ""


# ---------------------------------------------------------------------------
# Individual checks — each returns a Check so tests can call them directly.
# ---------------------------------------------------------------------------

def check_python() -> Check:
    """Python must be 3.11+; warn if it isn't exactly the tested 3.14.3."""
    v = sys.version_info
    cur = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < _MIN_PYTHON:
        return Check(
            FAIL, "Python version", f"{cur} (need 3.11+)",
            "Install Python 3.11 or newer (3.14.3 is the tested build).",
        )
    if (v.major, v.minor, v.micro) != _TESTED_PYTHON:
        return Check(
            WARN, "Python version", f"{cur} (tested on 3.14.3)",
            "Runs on 3.11+, but 3.14.3 is the version every pin was verified on.",
        )
    return Check(OK, "Python version", f"{cur} (tested build)")


def check_core_imports() -> Check:
    """Core packages must import and match their pinned versions."""
    missing: list[str] = []
    mismatched: list[str] = []
    seen: list[str] = []

    for pkg, expected in _EXPECTED_PINS.items():
        try:
            actual = md.version(pkg)
        except md.PackageNotFoundError:
            missing.append(pkg)
            continue
        if actual != expected:
            mismatched.append(f"{pkg} {actual}≠{expected}")
        else:
            seen.append(f"{pkg} {actual}")

    # duckdb is a floor pin — just require import + a version string.
    try:
        seen.append(f"duckdb {md.version('duckdb')}")
    except md.PackageNotFoundError:
        missing.append("duckdb")

    if missing:
        return Check(
            FAIL, "Core imports", f"missing: {', '.join(missing)}",
            "pip install -r requirements-core.txt",
        )
    if mismatched:
        return Check(
            WARN, "Core imports", f"version drift: {', '.join(mismatched)}",
            "pip install -r requirements-core.txt  (pins are the verified set)",
        )
    return Check(OK, "Core imports", ", ".join(seen))


def check_fixtures() -> Check:
    """The four test-fixture files must exist (used by the offline test suite)."""
    from core.config import SAMPLE_DIR

    expected = [
        SAMPLE_DIR / "test_fixture.csv",
        SAMPLE_DIR / "test_fixture.json",
        SAMPLE_DIR / "test_fixture.parquet",
        SAMPLE_DIR / "test_fixture.xlsx",
    ]
    missing = [p.name for p in expected if not p.exists()]
    if missing:
        return Check(
            WARN, "Test fixtures", f"missing: {', '.join(missing)}",
            "python data/sample/make_fixtures.py",
        )
    return Check(OK, "Test fixtures", "csv · json · parquet · xlsx present")


def check_demo_db() -> Check:
    """The SQLite demo clinic DB backs the 'Databases' onboarding demo."""
    from core.config import SAMPLE_DIR

    db_path = SAMPLE_DIR / "demo_clinic.db"
    if not db_path.exists():
        return Check(
            WARN, "Demo clinic DB", "demo_clinic.db not found",
            "python data/sample/make_demo_db.py",
        )
    return Check(OK, "Demo clinic DB", "demo_clinic.db present")


def check_duckdb_writable() -> Check:
    """The DuckDB output directory must be writable (the app persists there)."""
    from core.config import DATABASE_PATH

    try:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        probe = DATABASE_PATH.parent / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(
            FAIL, "DuckDB writable", f"cannot write to {DATABASE_PATH.parent}: {exc}",
            "Fix directory permissions for the outputs/ folder.",
        )
    return Check(OK, "DuckDB writable", str(DATABASE_PATH.parent))


def check_audit_writable() -> Check:
    """The append-only AI audit log must be writable."""
    from core.audit import AUDIT_FILE

    try:
        AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_FILE.open("a", encoding="utf-8"):
            pass  # open in append mode without writing a record
    except OSError as exc:
        return Check(
            FAIL, "Audit log writable", f"cannot write {AUDIT_FILE}: {exc}",
            "Fix directory permissions for the outputs/ folder.",
        )
    return Check(OK, "Audit log writable", str(AUDIT_FILE))


def check_ollama() -> Check:
    """Local Ollama + gemma3:4b power all data-touching AI (briefings, chips…)."""
    from core.config import OLLAMA_MODEL
    from core.providers import OllamaProvider

    if OllamaProvider().is_available():
        return Check(OK, "Ollama (local AI)", f"{OLLAMA_MODEL} reachable")
    return Check(
        WARN, "Ollama (local AI)", f"{OLLAMA_MODEL} not reachable",
        "Start Ollama + `ollama pull gemma3:4b`. "
        "Without it, local AI features use rule-based fallbacks.",
    )


def check_gemini_key() -> Check:
    """Gemini key is optional — chat falls back to local when absent."""
    from core.config import GEMINI_API_KEY

    if GEMINI_API_KEY:
        return Check(OK, "GEMINI_API_KEY", "set (cloud chat enabled)")
    return Check(
        WARN, "GEMINI_API_KEY", "not set",
        "Optional — chat runs local-only without it. Add to .env to enable cloud.",
    )


def check_census_key() -> Check:
    """Census key is optional — cached mart still works without it."""
    from core.config import CENSUS_API_KEY

    if CENSUS_API_KEY:
        return Check(OK, "CENSUS_API_KEY", "set (census/mart pulls enabled)")
    return Check(
        WARN, "CENSUS_API_KEY", "not set",
        "Optional — census/mart live pulls need it; a cached mart still works.",
    )


def check_mart_artifacts() -> Check:
    """The reporting mart parquet — cached so the demo needs no live pulls."""
    from core.config import OUTPUTS_DIR

    parquet = OUTPUTS_DIR / "mart" / "reporting_state_health.parquet"
    if not parquet.exists():
        return Check(
            WARN, "Mart artifacts", "reporting_state_health.parquet not found",
            "python scripts/build_mart.py  (run once before the demo)",
        )
    return Check(OK, "Mart artifacts", "reporting_state_health.parquet present")


def check_mart_staleness() -> Check:
    """Surface mart freshness — warn when older than 7 days or never built (5a)."""
    from core.config import OUTPUTS_DIR

    meta_path = OUTPUTS_DIR / "mart" / "build_meta.json"
    hint = "python scripts/build_mart.py --refresh  (see docs/SCHEDULING.md)"
    if not meta_path.exists():
        return Check(WARN, "Mart freshness", "no build_meta.json yet", hint)
    try:
        built_at = json.loads(meta_path.read_text(encoding="utf-8")).get("built_at")
        built = datetime.fromisoformat(built_at) if built_at else None
    except (OSError, ValueError, TypeError):
        built = None
    if built is None:
        return Check(WARN, "Mart freshness", "unreadable built_at", hint)
    if built.tzinfo is None:
        built = built.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - built).days
    if age_days > 7:
        return Check(WARN, "Mart freshness", f"built {age_days} days ago", hint)
    return Check(OK, "Mart freshness", f"built {age_days} day(s) ago")


# Order shown in the table.
_CHECKS = (
    check_python,
    check_core_imports,
    check_fixtures,
    check_demo_db,
    check_duckdb_writable,
    check_audit_writable,
    check_ollama,
    check_gemini_key,
    check_census_key,
    check_mart_artifacts,
    check_mart_staleness,
)


def run_all() -> list[Check]:
    """Run every check and return the list of Check results (never raises)."""
    results: list[Check] = []
    for fn in _CHECKS:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 — a broken check must not crash preflight
            results.append(
                Check(FAIL, fn.__name__, f"check errored: {exc}", "See traceback / report.")
            )
    return results


def _print_table(results: list[Check]) -> None:
    """Print the results as an aligned table with per-row fix hints."""
    name_w = max((len(c.name) for c in results), default=10)
    print("\nHealthPipe AI — Preflight")
    print("=" * 60)
    for c in results:
        print(f"{_SYMBOL[c.status]}  {c.name.ljust(name_w)}  {c.detail}")
        if c.status != OK and c.hint:
            print(f"      ↳ fix: {c.hint}")
    print("=" * 60)

    n_fail = sum(1 for c in results if c.status == FAIL)
    n_warn = sum(1 for c in results if c.status == WARN)
    n_ok = sum(1 for c in results if c.status == OK)
    print(f"{n_ok} ok · {n_warn} warn · {n_fail} fail")
    if n_fail:
        print("❌ Not demo-ready — resolve the failures above.")
    elif n_warn:
        print("⚠️  Demo-ready with degraded/optional features (see warnings).")
    else:
        print("✅ All systems go.")


def main() -> int:
    """Run all checks, print the table, and return an exit code (0 = no ❌)."""
    # Windows consoles default to cp1252 — force UTF-8 so the emoji don't crash.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    results = run_all()
    _print_table(results)
    return 1 if any(c.status == FAIL for c in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
