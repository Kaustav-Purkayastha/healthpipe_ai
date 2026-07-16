"""core/server.py — Flask API server for HealthPipe AI v2.

The application server. Runs on port 8501 and serves:
  - Login page and authentication (/api/login)
  - HTML/CSS/JS frontend pages (/pages/*)
  - REST API endpoints for core functionality (/api/*)
"""

from __future__ import annotations

import os
import re
import threading
from typing import Optional
from flask import Flask, request, jsonify, send_file, send_from_directory
from core.auth import check_credentials
from core.utils import get_logger

_log = get_logger(__name__)

# Get the project root (parent of 'core' directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
PAGES_DIR = os.path.join(STATIC_DIR, "pages")
LOGIN_HTML = os.path.join(STATIC_DIR, "login.html")

# Module-level singletons for the Flask process — the router and source registry
# are built once and reused across requests.
_router = None


_registry = None


def _get_router():
    """Lazily construct one AIRouter for this Flask process's lifetime."""
    global _router
    if _router is None:
        from core.router import AIRouter
        _router = AIRouter()
    return _router


def _get_registry():
    """Lazily construct one SourceRegistry (built-in sources) for this process."""
    global _registry
    if _registry is None:
        from ingestion.registry import SourceRegistry
        _registry = SourceRegistry()
    return _registry


def _api_param_schema(name: str) -> list[dict]:
    """Return the form-field schema for an API source's extract parameters.

    Mirrors the per-source parameter forms from the original Onboard screen so
    the web UI can render the right inputs and post back values the extractor
    understands. Unknown sources get an empty schema (pulled with defaults).
    """
    if name == "who":
        from ingestion.who_source import WHO_INDICATORS  # lazy
        opts = [{"value": code, "label": lbl} for code, lbl in WHO_INDICATORS.items()]
        return [
            {"key": "indicator", "label": "Indicator", "type": "select", "options": opts},
            {"key": "countries", "label": "Countries (space-separated ISO3, blank = all)",
             "type": "text", "optional": True},
        ]
    if name == "openfda":
        return [{"key": "search_term", "label": "Drug search term", "type": "text", "default": "aspirin"}]
    if name == "cms_medicare":
        from core.config import STATE_ABBRS  # lazy
        opts = [{"value": "", "label": "All states"}] + [{"value": s, "label": s} for s in sorted(STATE_ABBRS)]
        return [
            {"key": "state", "label": "State (optional)", "type": "select", "options": opts, "optional": True},
            {"key": "specialty", "label": "Specialty (optional)", "type": "text", "optional": True},
        ]
    if name == "cdc_cdi":
        from ingestion.cdc_cdi_source import CDCChronicDiseaseSource  # lazy
        opts = [
            {"value": CDCChronicDiseaseSource.DIABETES, "label": "Diabetes"},
            {"value": CDCChronicDiseaseSource.OBESITY, "label": "Obesity"},
            {"value": CDCChronicDiseaseSource.SMOKING, "label": "Smoking"},
        ]
        return [
            {"key": "question_id", "label": "Measure", "type": "select", "options": opts},
            {"key": "year", "label": "Year", "type": "number", "default": "2023",
             "min": 2011, "max": 2024, "step": 1},
        ]
    if name in ("cdc_brfss", "brfss"):
        return [
            {"key": "preset", "label": "Preset", "type": "select",
             "options": [{"value": "obesity", "label": "Obesity"}, {"value": "smoking", "label": "Smoking"}]},
            {"key": "year", "label": "Year", "type": "number", "default": "2023",
             "min": 2011, "max": 2024, "step": 1},
        ]
    return []  # census / places / others — no parameters


def _api_extract_kwargs(name: str, params: dict) -> dict:
    """Translate posted form params into the source's extract() kwargs."""
    p = params or {}
    kw: dict = {}
    if name == "who":
        if p.get("indicator"):
            kw["indicator"] = p["indicator"]
        countries = (p.get("countries") or "").strip()
        if countries:
            kw["countries"] = countries.split()
    elif name == "openfda":
        kw["search_term"] = p.get("search_term") or "aspirin"
    elif name == "cms_medicare":
        if p.get("state"):
            kw["state"] = p["state"]
        if (p.get("specialty") or "").strip():
            kw["specialty"] = p["specialty"].strip()
    elif name == "cdc_cdi":
        if p.get("question_id"):
            kw["question_id"] = p["question_id"]
        kw["year"] = p.get("year") or "2023"
    elif name in ("cdc_brfss", "brfss"):
        from ingestion.brfss_source import BRFSSSource  # lazy
        presets = {"obesity": BRFSSSource.OBESITY, "smoking": BRFSSSource.SMOKING}
        chosen = presets.get(p.get("preset") or "")
        if isinstance(chosen, dict):
            kw.update(chosen)
        kw["year"] = p.get("year") or "2023"
    return kw


# Words that describe the mart's two fixed columns (not part of the CDI
# catalog, so a keyword-overlap check against the catalog alone would miss
# a perfectly valid request like "rank states by Medicare spend").
_FIXED_COLUMN_TERMS: tuple[str, ...] = (
    "spend", "spending", "cost", "medicare", "census", "population", "provider",
)


def _parquet_rows(path) -> Optional[int]:
    """Return a parquet file's row count from its footer metadata (no full load).

    Used to show the "how much raw data" funnel on the sources panel. Returns
    None if the file is missing/unreadable so the caller can omit the figure.
    """
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    except Exception:  # noqa: BLE001 — missing/corrupt cache → just omit the number
        return None


def _prompt_matches_catalog(prompt: str, catalog) -> bool:
    """Cheap relevance guard: does the prompt touch anything actually available?

    Runs before the (potentially slow) AI planning call so a prompt with zero
    overlap with the CDI catalog or the mart's fixed columns gets an honest
    "not possible with current data sources" instead of a fabricated plan.
    This mirrors the keyword logic in analytics.mart_planner's offline
    heuristic, applied a step earlier so we can short-circuit entirely.
    """
    low = prompt.lower()
    if any(term in low for term in _FIXED_COLUMN_TERMS):
        return True
    prompt_words = set(re.findall(r"[a-z]{4,}", low))
    catalog_words: set[str] = set()
    for row in catalog.itertuples():
        catalog_words.update(re.findall(r"[a-z]{4,}", f"{row.question} {row.topic}".lower()))
    return bool(prompt_words & catalog_words)


# Phrasings that ask for something the data's GRAIN can't deliver, even when the
# measure names match. The mart is state-level crude prevalence for the overall
# adult population — so it has no individuals and no demographic breakdowns.
_INDIVIDUAL_PAT = re.compile(
    r"\b(individual|individuals|person|persons|people|someone|somebody|patient|patients"
    r"|the same (person|people|individual)|also (suffer|suffers|have|has|had|experience|experiences|get|gets)"
    r"|who (also|have|has)|do they also|does (he|she|they) also|both conditions?|co-?occur)\b",
    re.IGNORECASE,
)
_DEMOGRAPHIC_PAT = re.compile(
    r"\b(females?|wom[ae]n|males?|\bmen\b|\bman\b|by age|age group|age-group|by race|racial"
    r"|ethnicit(y|ies)|by gender|by sex|by income|low[- ]income|among (wom[ae]n|men)|demographic)\b",
    re.IGNORECASE,
)


def _scope_caveat(prompt: str) -> Optional[str]:
    """Return an honest caveat when the prompt asks beyond the data's grain.

    The measure words may match the catalog (so the request is "buildable" as a
    state-level comparison), but the phrasing wants individual-level co-occurrence
    ("does the same person have both") or a demographic slice ("for females") —
    neither of which state-level, overall-population aggregates can answer. We
    surface this instead of silently planning as if the literal question is met.

    Returns None when no such over-reach is detected.
    """
    p = prompt or ""
    individual = bool(_INDIVIDUAL_PAT.search(p))
    demographic = bool(_DEMOGRAPHIC_PAT.search(p))
    if not (individual or demographic):
        return None

    parts: list[str] = []
    if individual:
        parts.append(
            "this mart holds state-level averages, not individual records — it can't tell you "
            "whether the same people have two conditions at once"
        )
    if demographic:
        parts.append(
            "the figures are for the overall adult population, not broken down by sex, age, race, "
            "or income — it can't filter to a group like \"females\""
        )
    what = " and ".join(parts)
    return (
        f"Heads up — {what}. What I can build is a state-level comparison: whether states high in "
        "one measure tend to be high in the other. That's a state-level association, not proof "
        "about individuals."
    )


# ---------------------------------------------------------------------------
# Per-mart persistence: a small JSON registry so every built mart survives a
# logout and reappears in the "My marts" list, re-openable without a rebuild.
# ---------------------------------------------------------------------------
_MART_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "mart")
_REGISTRY_PATH = os.path.join(_MART_OUT_DIR, "registry.json")
_registry_lock = threading.Lock()

# DuckDB tables that are internal/reporting, not user-onboarded source datasets.
_RESERVED_TABLES: frozenset[str] = frozenset({"reporting_state_health"})

# Friendly titles for reserved reporting tables (used for their AI enrichment /
# data dictionary, since they have no registry entry to borrow a title from).
_RESERVED_LABELS: dict[str, str] = {
    "reporting_state_health": "State health reporting table",
}


def _slugify_table(title: str, measures: list[str]) -> str:
    """Derive a safe, per-mart DuckDB table name from the title (or measures).

    e.g. "Diabetes vs Medicare spend" -> "mart_diabetes_vs_medicare_spend".
    Rebuilding the same-titled mart reuses the same table (an upsert), so the
    registry never accumulates duplicates for the same conceptual mart.
    """
    base = re.sub(r"[^a-z0-9]+", "_", (title or "").lower()).strip("_")
    if not base or base in {"untitled_mart", "custom_state_health_mart"}:
        base = "_".join(str(m).lower() for m in (measures or [])[:3])
    base = re.sub(r"_+", "_", base).strip("_")[:40] or "custom"
    return f"mart_{base}"


def _registry_load() -> list:
    """Load the mart registry list (empty list if absent/corrupt)."""
    try:
        import json
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001 — missing/corrupt registry → start fresh
        return []


def _norm_state_abbr(value) -> Optional[str]:
    """Map a state value (abbr or full name, any case) to a USPS abbreviation."""
    from analytics.mart_planner import _STATE_NAMES  # abbr -> full
    s = str(value).strip()
    if s.upper() in _STATE_NAMES:
        return s.upper()
    full_to_abbr = {v.lower(): k for k, v in _STATE_NAMES.items()}
    return full_to_abbr.get(s.lower())


def _external_col_name(spec: dict) -> str:
    """Deterministic mart column name for a joined external metric."""
    base = re.sub(r"[^a-z0-9]+", "_", f"{spec.get('table','')}_{spec.get('column','')}".lower()).strip("_")
    return f"ext_{base}"[:60]


def _external_measure_series(spec: dict) -> dict:
    """Aggregate an onboarded table to one value per state (abbr -> value).

    Runs entirely on the local DuckDB — the external metric is grouped by its
    detected state column and aggregated (mean/sum/count), then state keys are
    normalised to USPS abbreviations so it can left-join onto the 51-state mart.
    """
    from core.database import DuckDBManager
    from core.config import DATABASE_PATH

    table = spec.get("table")
    col = spec.get("column")
    sc = spec.get("state_col")
    if not (table and col and sc):
        return {}
    agg = (spec.get("aggregation") or "mean").lower()
    agg_sql = {"mean": "AVG", "avg": "AVG", "sum": "SUM", "count": "COUNT"}.get(agg, "AVG")

    with DuckDBManager(db_path=DATABASE_PATH) as db:
        if table not in db.list_tables():
            return {}
        df = db.query(f'SELECT "{sc}" AS s, {agg_sql}("{col}") AS v FROM "{table}" GROUP BY "{sc}"')

    out: dict = {}
    for _, r in df.iterrows():
        abbr = _norm_state_abbr(r["s"])
        if abbr and r["v"] is not None:
            out[abbr] = round(float(r["v"]), 2)
    return out


_SOURCE_PROFILES_PATH = os.path.join(_MART_OUT_DIR, "source_profiles.json")
_profiles_lock = threading.Lock()

# AI enrichment (briefing + AI-written column descriptions) is computed on-device
# in the background at onboard time and cached here, keyed by table, so it can be
# viewed later by opening the dataset — it never blocks onboarding.
_ENRICHMENT_PATH = os.path.join(_MART_OUT_DIR, "enrichment.json")
_enrichment_lock = threading.Lock()


def _enrichment_load() -> dict:
    try:
        import json
        with open(_ENRICHMENT_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _enrichment_save(data: dict) -> None:
    import json
    os.makedirs(_MART_OUT_DIR, exist_ok=True)
    with open(_ENRICHMENT_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _store_enrichment(table: str, briefing: Optional[dict], column_docs: list) -> None:
    with _enrichment_lock:
        data = _enrichment_load()
        data[table] = {
            "briefing": briefing,
            "column_docs": column_docs,
            "generated_by": (briefing or {}).get("generated_by", "gemma3:4b (local)"),
        }
        _enrichment_save(data)


def _get_enrichment(table: str) -> Optional[dict]:
    return _enrichment_load().get(table)


# ----- Data-quality store -------------------------------------------------
# The pipeline profiles + quality-checks every table at ingest (ProfilerAgent +
# QualityCheckerAgent). That work used to be thrown away after shaping the grade;
# we now persist a compact scorecard per table here so the onboard UI can show
# the FULL quality profile (per-check pass/fail across completeness, nulls,
# duplicates, type consistency, ranges, uniqueness) — not just the letter grade.
_QUALITY_PATH = os.path.join(_MART_OUT_DIR, "quality.json")
_quality_lock = threading.Lock()

# check-name prefix -> human dimension label, for grouping the per-check list.
_QUALITY_DIMENSIONS: list[tuple[str, str]] = [
    ("overall_completeness", "Completeness"),
    ("duplicate_rows", "Duplicates"),
    ("null_rate_", "Missing values"),
    ("type_consistency_", "Type consistency"),
    ("value_range_", "Value ranges"),
    ("uniqueness_", "Uniqueness"),
]


def _quality_load() -> dict:
    try:
        import json
        with open(_QUALITY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _json_native(o):
    """json.dump default for numpy scalars (profiler stats are np.float64 etc.).

    numpy scalars expose .item() → a native Python number; anything else falls
    back to str so a stray type can never break the (best-effort) quality save.
    """
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return o.item()
        except Exception:  # noqa: BLE001
            pass
    return str(o)


def _quality_save(data: dict) -> None:
    import json
    os.makedirs(_MART_OUT_DIR, exist_ok=True)
    with open(_QUALITY_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, default=_json_native)


def _dimension_for(check_name: str) -> str:
    for prefix, label in _QUALITY_DIMENSIONS:
        if check_name == prefix or check_name.startswith(prefix):
            return label
    return "Other"


def _build_quality_record(scorecard: Optional[dict], profile: Optional[dict]) -> Optional[dict]:
    """Distill a ProfilerAgent profile + QualityCheckerAgent scorecard into the
    compact, UI-ready record we persist and serve. Returns None if there's no
    scorecard to summarize."""
    scorecard = scorecard or {}
    profile = profile or {}
    checks = scorecard.get("checks") or []
    if not scorecard and not checks:
        return None
    overview = profile.get("overview") or {}
    # Group checks by dimension with a pass/total tally, keeping each check's
    # human-readable detail line (already produced by the quality checker).
    dims: dict = {}
    for c in checks:
        label = _dimension_for(str(c.get("check", "")))
        d = dims.setdefault(label, {"dimension": label, "passed": 0, "total": 0, "checks": []})
        d["total"] += 1
        if c.get("passed"):
            d["passed"] += 1
        d["checks"].append({
            "check": c.get("check", ""),
            "passed": bool(c.get("passed")),
            "detail": c.get("detail", ""),
        })
    return {
        "grade": scorecard.get("grade", ""),
        "score": scorecard.get("score", 0.0),
        "total_checks": scorecard.get("total_checks", len(checks)),
        "checks_passed": scorecard.get("checks_passed", sum(1 for c in checks if c.get("passed"))),
        "checks_failed": scorecard.get("checks_failed",
                                       sum(1 for c in checks if not c.get("passed"))),
        "overview": {
            "row_count": overview.get("row_count"),
            "column_count": overview.get("column_count"),
            "completeness_score": overview.get("completeness_score"),
            "duplicate_percentage": overview.get("duplicate_percentage"),
        },
        "pii_columns": profile.get("pii_columns") or [],
        "quality_issues": profile.get("quality_issues") or [],
        "dimensions": list(dims.values()),
    }


def _store_quality(table: str, scorecard: Optional[dict], profile: Optional[dict]) -> None:
    """Persist a table's distilled quality record (best-effort, no-op if empty)."""
    record = _build_quality_record(scorecard, profile)
    if record is None:
        return
    with _quality_lock:
        data = _quality_load()
        data[table] = record
        _quality_save(data)


def _get_quality(table: str) -> Optional[dict]:
    return _quality_load().get(table)


# ----- Dashboard helpers (ported, UI-free, from the Streamlit screens) -----
# These mirror ui/screens/{triage,audit,dashboard}.py + analytics/{drift,lineage}
# so the Dashboard page can consolidate run history, triage, lineage and the
# privacy audit without importing Streamlit.

_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "gemma", "local"})
_NON_PROVIDERS: frozenset[str] = frozenset({"none", ""})

# Per-column check-name prefixes → so "null_rate_email" surfaces column "email".
_TRIAGE_COLUMN_PREFIXES: tuple[str, ...] = (
    "null_rate_", "no_negatives_", "extreme_outliers_",
    "type_consistency_", "uniqueness_", "value_range_",
)
# severity → (sort rank, label, kind).  Failed checks + criticals sort first.
_TRIAGE_SEVERITY: dict[str, tuple[int, str, str]] = {
    "critical": (0, "Critical", "fail"),
    "fail": (0, "Failed check", "fail"),
    "warning": (1, "Warning", "warn"),
    "info": (2, "Info", "info"),
}


def _is_cloud_provider(provider: str) -> bool:
    p = str(provider).lower()
    return p not in _LOCAL_PROVIDERS and p not in _NON_PROVIDERS


def _audit_metrics(records: list) -> dict:
    """Headline audit metrics (ported from ui/screens/audit.audit_metrics)."""
    total = len(records)
    local = sum(1 for r in records if str(r.get("provider", "")).lower() in _LOCAL_PROVIDERS)
    cloud = sum(1 for r in records if _is_cloud_provider(r.get("provider", "")))
    total_redactions = sum(int(r.get("redaction_count", 0) or 0) for r in records)
    # THE proof number: characters CLOUD calls carried — cloud-only, never summed
    # with local, because the claim is about what left the machine.
    cloud_chars = sum(int(r.get("prompt_chars", 0) or 0)
                      for r in records if _is_cloud_provider(r.get("provider", "")))
    return {
        "total_calls": total,
        "local_calls": local,
        "cloud_calls": cloud,
        "local_pct": round(local / total * 100, 1) if total else 0.0,
        "cloud_pct": round(cloud / total * 100, 1) if total else 0.0,
        "total_redactions": total_redactions,
        "cloud_prompt_chars": cloud_chars,
    }


def _proof_line(m: dict) -> str:
    """The privacy proof sentence (ported from ui/screens/audit.proof_line)."""
    return (
        f"Cloud calls carried {m['cloud_prompt_chars']:,} characters of "
        f"schema + questions and 0 data rows. "
        f"{m['total_redactions']} PII item(s) were redacted before leaving this machine."
    )


def _check_passed(c: dict) -> bool:
    val = c.get("passed", True)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().upper() not in {"FAIL", "FALSE", "0", "NO"}
    return bool(val)


def _triage_column(check_name: str) -> str:
    for prefix in _TRIAGE_COLUMN_PREFIXES:
        if check_name.startswith(prefix):
            return check_name[len(prefix):]
    return "—"


def _merge_issues(profile: dict, scorecard: dict) -> list:
    """Merge profiler quality issues + failed scorecard checks, severity-sorted.

    Ported from ui/screens/triage.merge_issues — same normalised shape and the
    same synthetic check_name so explanations match by key.
    """
    issues: list = []
    for qi in (profile or {}).get("quality_issues", []) or []:
        sev = str(qi.get("severity", "info")).lower()
        rank, label, kind = _TRIAGE_SEVERITY.get(sev, _TRIAGE_SEVERITY["info"])
        issue_type = str(qi.get("issue", "issue"))
        column = str(qi.get("column", "—"))
        issues.append({
            "rank": rank, "label": label, "kind": kind,
            "title": issue_type.replace("_", " "), "column": column,
            "detail": str(qi.get("detail", "")),
            "check_name": f"{issue_type}_{column}",
        })
    for c in (scorecard or {}).get("checks", []) or []:
        if _check_passed(c):
            continue
        check = str(c.get("check", "check"))
        rank, label, kind = _TRIAGE_SEVERITY["fail"]
        issues.append({
            "rank": rank, "label": label, "kind": kind,
            "title": check.replace("_", " "), "column": _triage_column(check),
            "detail": str(c.get("detail", "")), "check_name": check,
        })
    issues.sort(key=lambda i: i["rank"])
    return issues


def _triage_schema(profile: dict, docs: dict) -> list:
    """Lightweight column name/type list for the explain-issue prompt."""
    data_dict = (docs or {}).get("data_dictionary", []) or []
    if data_dict:
        return [{"column_name": e.get("column_name", "?"),
                 "column_type": e.get("data_type", "?")} for e in data_dict]
    return [{"column_name": name, "column_type": "unknown"}
            for name in (profile or {}).get("columns", {}).keys()]


def _lineage_nodes(artifacts: dict) -> list:
    """Ordered lineage flow (source → transforms → table), from run artifacts.

    Same source fields as analytics/lineage.build_lineage_dot, but returned as
    a node list the web UI renders as an HTML flow (no Graphviz dependency).
    """
    docs = (artifacts or {}).get("docs") or {}
    profile = (artifacts or {}).get("profile") or {}
    lineage = docs.get("lineage") or {}
    source = lineage.get("source") or {}
    schema = docs.get("schema") or {}
    transforms = lineage.get("transformations") or []
    dataset_name = lineage.get("dataset_name") or docs.get("dataset_name") or "dataset"

    nodes: list = []
    src_rows = source.get("last_record_count")
    if not isinstance(src_rows, int) or src_rows < 0:
        src_rows = (profile.get("overview") or {}).get("row_count")
    src_type = source.get("source_type") or source.get("type") or "source"
    src_detail = src_type + (f" · {src_rows:,} rows"
                             if isinstance(src_rows, int) and src_rows >= 0 else "")
    nodes.append({"kind": "source", "title": str(source.get("name") or dataset_name),
                  "detail": src_detail})
    for step in transforms:
        nodes.append({"kind": "transform", "title": str(step.get("action", "step")),
                      "detail": str(step.get("detail", ""))})
    out_rows = schema.get("row_count")
    nodes.append({"kind": "table",
                  "title": f"DuckDB · {schema.get('table_name') or dataset_name}",
                  "detail": (f"{out_rows:,} rows" if isinstance(out_rows, int) else "")})
    return nodes


def _initials(name: str) -> str:
    """Up-to-two-letter uppercase initials for the avatar (e.g. 'admin' → 'AD')."""
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", str(name or "")) if p]
    if not parts:
        return "OP"
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return parts[0][:2].upper()


def _run_record(row: dict) -> dict:
    """Coerce one pipeline_runs row (numpy-typed) into a JSON-safe dict."""
    def _i(v):
        try:
            return int(v)
        except Exception:  # noqa: BLE001
            return 0

    def _f(v):
        try:
            return round(float(v), 2)
        except Exception:  # noqa: BLE001
            return 0.0
    return {
        "run_id": str(row.get("run_id", "")),
        "dataset_name": str(row.get("dataset_name", "")),
        "source_name": str(row.get("source_name", "")),
        "started_at": str(row.get("started_at", "")),
        "duration_s": _f(row.get("duration_s")),
        "rows_in": _i(row.get("rows_in")),
        "rows_out": _i(row.get("rows_out")),
        "grade": str(row.get("grade", "")),
        "score": _f(row.get("score")),
        "gate_blocked": bool(row.get("gate_blocked")),
        "table_name": str(row.get("table_name", "")),
    }


def _profile_onboarded_table(table: str) -> Optional[dict]:
    """Deterministic on-device profile of an onboarded table.

    Returns columns, detected US-state column, distinct-state count, row count,
    and per-numeric-column stats — the raw material for the mart relevance
    verdict. Returns None if the table can't be read.
    """
    from core.database import DuckDBManager
    from core.config import DATABASE_PATH
    from analytics.mart_planner import _STATE_ABBR_SET, _STATE_FULL_SET

    try:
        with DuckDBManager(db_path=DATABASE_PATH) as db:
            if table not in db.list_tables():
                return None
            schema = db.get_schema(table)
            n_rows = int(db.query(f'SELECT COUNT(*) AS n FROM "{table}"').iloc[0]["n"])
            columns = [{"name": c["column_name"], "type": str(c["column_type"])} for c in schema]

            state_col, best = None, 0
            for c in columns:
                if not any(k in c["type"].upper() for k in ("CHAR", "VARCHAR", "STRING", "TEXT")):
                    continue
                try:
                    vals = db.query(f'SELECT DISTINCT "{c["name"]}" AS v FROM "{table}" LIMIT 80')["v"].tolist()
                except Exception:  # noqa: BLE001
                    continue
                norm = [str(v).strip() for v in vals if v is not None]
                if not norm:
                    continue
                hits = sum(1 for v in norm if v.upper() in _STATE_ABBR_SET or v.lower() in _STATE_FULL_SET)
                if hits and hits / len(norm) >= 0.6 and hits > best:
                    best, state_col = hits, c["name"]

            n_states = 0
            if state_col:
                try:
                    n_states = int(db.query(
                        f'SELECT COUNT(DISTINCT "{state_col}") AS n FROM "{table}"').iloc[0]["n"])
                except Exception:  # noqa: BLE001
                    n_states = best

            numeric_stats: dict = {}
            for c in columns:
                if any(k in c["type"].upper() for k in ("INT", "DOUBLE", "DECIMAL", "FLOAT", "REAL", "NUMERIC", "BIGINT")):
                    try:
                        row = db.query(
                            f'SELECT MIN("{c["name"]}") mn, MAX("{c["name"]}") mx, AVG("{c["name"]}") av FROM "{table}"'
                        ).iloc[0]
                        numeric_stats[c["name"]] = {
                            "min": float(row["mn"]) if row["mn"] is not None else 0.0,
                            "max": float(row["mx"]) if row["mx"] is not None else 0.0,
                            "mean": float(row["av"]) if row["av"] is not None else 0.0,
                        }
                    except Exception:  # noqa: BLE001
                        continue
    except Exception:  # noqa: BLE001
        _log.exception("_profile_onboarded_table failed for %s", table)
        return None

    return {"columns": columns, "state_col": state_col, "n_states": n_states,
            "n_rows": n_rows, "numeric_stats": numeric_stats}


def _mart_source_profile(table: str, dataset_name: str = "") -> Optional[dict]:
    """Full mart-readiness profile for an onboarded table (profile + AI verdict).

    Runs the deterministic profile, then — only when the table plausibly fits
    (has a state column + a numeric metric) — asks the LOCAL model to judge
    relevance and name usable metrics. The result is what the mart shows and the
    planner suggests from, so this work happens ONCE at onboard time, not per view.
    """
    from analytics.mart_planner import analyze_onboarded_source

    profile = _profile_onboarded_table(table)
    if profile is None:
        return None

    has_metric = any(profile.get("numeric_stats"))
    if profile.get("state_col") and has_metric:
        verdict = analyze_onboarded_source(_get_router(), table, profile)
    else:
        # No state key / no numeric metric → deterministic "not state-joinable".
        verdict = {
            "relevant": False, "joinable": bool(profile.get("state_col")),
            "state_col": profile.get("state_col"), "metrics": [],
            "reason": ("Has a state column but no numeric metric to aggregate." if profile.get("state_col")
                       else "No US-state column, so it can't join into a state-level mart."),
            "generated_by": "rule-based",
        }
    return {
        "table": table,
        "dataset_name": dataset_name or table,
        "n_rows": profile.get("n_rows", 0),
        "n_fields": len(profile.get("columns", [])),
        "state_col": profile.get("state_col"),
        "n_states": profile.get("n_states", 0),
        "relevant": bool(verdict.get("relevant")),
        "reason": verdict.get("reason", ""),
        "metrics": verdict.get("metrics", []),
        "generated_by": verdict.get("generated_by", ""),
    }


def _source_profiles_load() -> dict:
    try:
        import json
        with open(_SOURCE_PROFILES_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _source_profiles_save(profiles: dict) -> None:
    import json
    os.makedirs(_MART_OUT_DIR, exist_ok=True)
    with open(_SOURCE_PROFILES_PATH, "w", encoding="utf-8") as fh:
        json.dump(profiles, fh)


def _external_suggestions(prompt: str) -> list:
    """Onboarded metrics whose names/meanings overlap the user's request.

    Pure local keyword matching against the STORED profiles — no cloud, no
    re-analysis. Powers "Generate a mart with AI" auto-suggesting relevant
    onboarded data for a related question.
    """
        # Generic terms that appear in almost every metric description — they must
        # NOT drive a match, or every onboarded metric would "match" any prompt.
    _STOP = {
        "state", "states", "per", "average", "avg", "mean", "sum", "count", "rate",
        "pct", "percent", "percentage", "data", "dataset", "across", "show", "compare",
        "the", "and", "for", "with", "each", "people", "adults", "adult", "usa",
        "these", "this", "that", "into", "from", "over", "under", "than", "more",
        "less", "how", "does", "what", "which", "value", "values", "number", "total",
        "burden", "spend", "medicare", "health",
    }
    words = set(re.findall(r"[a-z]{4,}", (prompt or "").lower())) - _STOP
    if not words:
        return []
    out, seen = [], set()
    for table, prof in _source_profiles_load().items():
        if not prof.get("relevant"):
            continue
        for m in prof.get("metrics", []):
            col = str(m.get("column", ""))
            agg = str(m.get("aggregation", "mean"))
            hay = set(re.findall(r"[a-z]{4,}", f"{col} {m.get('meaning','')} {prof.get('dataset_name','')}".lower())) - _STOP
            if not (words & hay):
                continue
            ext_id = f"ext::{table}::{col}::{agg}"
            if ext_id in seen:
                continue
            seen.add(ext_id)
            out.append({
                "id": ext_id, "label": col.replace("_", " ").strip().capitalize(),
                "table": table, "column": col, "aggregation": agg,
                "state_col": prof.get("state_col"), "meaning": m.get("meaning", ""),
                "dataset_name": prof.get("dataset_name", table),
            })
    return out[:6]


def _store_source_profile(table: str, dataset_name: str = "") -> Optional[dict]:
    """Compute + persist a table's mart profile (best-effort). Returns it or None."""
    prof = _mart_source_profile(table, dataset_name)
    if prof is None:
        return None
    with _profiles_lock:
        profiles = _source_profiles_load()
        profiles[table] = prof
        _source_profiles_save(profiles)
    return prof


# Live profiling status (table -> "profiling"|"ready"|"error") so the onboard UI
# can say "profiling…" then "profiling complete" while it happens in the background.
_profiling_status: dict = {}
_profiling_status_lock = threading.Lock()

# Live AI-enrichment status, tracked SEPARATELY from profiling so the UI can tell
# "never enriched" (absent) from "actively generating" (running) from "failed"
# (error, with the reason). table -> "running"|"done"|"error".
_enrich_status: dict = {}
_enrich_error: dict = {}
_enrich_status_lock = threading.Lock()


def _set_profiling_status(table: str, status: str) -> None:
    with _profiling_status_lock:
        _profiling_status[table] = status


def _get_profiling_status(table: str) -> Optional[str]:
    with _profiling_status_lock:
        return _profiling_status.get(table)


def _set_enrich_status(table: str, status: str, error: str = "") -> None:
    with _enrich_status_lock:
        _enrich_status[table] = status
        if error:
            _enrich_error[table] = error
        elif status != "error":
            _enrich_error.pop(table, None)


def _get_enrich_status(table: str) -> tuple:
    with _enrich_status_lock:
        return _enrich_status.get(table), _enrich_error.get(table, "")


def _start_background_profile(table: str, dataset_name: str = "",
                              clean_df=None, profile: Optional[dict] = None,
                              scorecard: Optional[dict] = None) -> None:
    """Enrich + profile a freshly onboarded table on a daemon thread (non-blocking).

    Onboarding returns immediately. In the background we (1) run AI enrichment
    on-device — a briefing narrative + per-column descriptions from the local
    model — and cache it so it can be viewed by opening the dataset, then
    (2) compute the mart-readiness profile (state detection + local relevance
    verdict). Status flips to "ready" only when both are done, so the UI can
    report "analysis complete".
    """
    _set_profiling_status(table, "profiling")
    if clean_df is not None:
        _set_enrich_status(table, "running")
    threading.Thread(
        target=lambda: _enrich_and_profile(table, dataset_name, clean_df, profile, scorecard),
        daemon=True,
    ).start()


def _enrich_and_profile(table: str, dataset_name: str, clean_df,
                        profile: Optional[dict], scorecard: Optional[dict],
                        do_profile: bool = True) -> None:
    """Synchronous enrich (if df given) + optional mart-profile for one table.

    Shared by the per-onboard background thread, the mart-build enrichment, and
    the startup sweep. Runs the local model for the briefing + column
    descriptions, caches them, then (for onboarded sources) computes the
    mart-readiness profile. ``do_profile=False`` for built marts — a mart is the
    query target itself, not a candidate source to fold into another mart.
    """
    import traceback as _tb
    # Persist the quality scorecard whenever we have one (sweep / enrich_now /
    # mart-enrich all pass a freshly computed profile + scorecard), so re-opening
    # an older dataset backfills its quality profile too.
    if scorecard or profile:
        _store_quality(table, scorecard, profile)
    if clean_df is not None:
        try:
            from core.enrich import describe_columns, generate_briefing
            router = _get_router()
            descs = describe_columns(router, clean_df, profile or {})
            briefing = generate_briefing(
                router, dataset_name or table, profile or {}, scorecard or {})
            column_docs = [{"name": k, "description": v} for k, v in descs.items()]
            _store_enrichment(table, briefing, column_docs)
            _set_enrich_status(table, "done")
            _log.info("enrichment completed for %s", table)
        except Exception as e:  # noqa: BLE001 — enrichment is best-effort
            msg = f"{type(e).__name__}: {e}"
            _set_enrich_status(table, "error", msg)
            _log.warning("enrichment failed for %s: %s\n%s", table, msg, _tb.format_exc())
    if not do_profile:
        return
    try:
        _store_source_profile(table, dataset_name)
        _set_profiling_status(table, "ready")
        _log.info("profiling ready for %s", table)
    except Exception as e:  # noqa: BLE001
        _log.warning("profiling failed for %s: %s", table, str(e))
        _set_profiling_status(table, "error")


def _start_mart_enrich(table: str, title: str, mart_df) -> None:
    """Enrich a freshly built mart on a daemon thread (column docs + briefing).

    Enrichment only — a mart is a query target, not a source to profile for
    inclusion in another mart, so the mart-readiness profile is skipped.
    """
    if mart_df is None:
        return
    _set_enrich_status(table, "running")

    def _run():
        prof, score = _compute_profile_scorecard(mart_df, title or table)
        _enrich_and_profile(table, title or table, mart_df, prof, score, do_profile=False)

    threading.Thread(target=_run, daemon=True).start()


def _compute_profile_scorecard(df, dataset_name: str) -> tuple:
    """Run the profiler + quality checker on a DataFrame (best-effort).

    Used when enriching a dataset that was onboarded before enrichment existed, so
    the briefing reflects the REAL row/column counts and grade instead of empty
    placeholders. Returns (profile, scorecard); either may be {} on failure.
    """
    profile, scorecard = {}, {}
    try:
        from agents.profiler import ProfilerAgent
        profile = ProfilerAgent().run(df, dataset_name) or {}
    except Exception:  # noqa: BLE001
        _log.warning("profiler failed for %s", dataset_name)
    try:
        from agents.quality_checker import QualityCheckerAgent
        scorecard = QualityCheckerAgent().run(df, dataset_name) or {}
    except Exception:  # noqa: BLE001
        _log.warning("quality checker failed for %s", dataset_name)
    return profile, scorecard


def _start_enrich_sweep() -> None:
    """On startup, enrich any loaded dataset that has no enrichment yet.

    Runs SEQUENTIALLY in a single daemon thread (one model call at a time, so the
    local model isn't hammered) so every dataset gets its AI enrichment
    automatically — the user never has to open it to trigger generation.
    """
    def _run():
        try:
            from core.database import DuckDBManager
            from core.config import DATABASE_PATH
            profiles = _source_profiles_load()
            enriched = set(_enrichment_load().keys())
            # Onboarded sources (profiled) AND built marts (from the registry) that
            # have no enrichment yet. Marts are enrichment-only (do_profile=False).
            mart_titles = {m.get("table_name"): m.get("title", "") for m in _registry_load()}
            # Pre-built reporting tables (e.g. reporting_state_health) belong to
            # neither list but still need a data dictionary — treat them like marts.
            for _rt in _RESERVED_TABLES:
                mart_titles.setdefault(_rt, _RESERVED_LABELS.get(_rt, _rt))
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                tables = set(db.list_tables())
                src_pending = [t for t in profiles if t not in enriched and t in tables]
                mart_pending = [t for t in mart_titles
                                if t and t not in enriched and t in tables]
                want = src_pending + [t for t in mart_pending if t not in src_pending]
                dfs = {}
                for t in want:
                    try:
                        dfs[t] = db.query(f'SELECT * FROM "{t}"')
                    except Exception:  # noqa: BLE001
                        pass
            if not want:
                return
            _log.info("enrich sweep: %d table(s) pending", len(want))
            # Mark all pending "running" up front so the UI shows "generating"
            # and won't fire duplicate enrich requests if opened mid-sweep.
            for t in want:
                if dfs.get(t) is not None:
                    _set_enrich_status(t, "running")
            for t in want:
                df = dfs.get(t)
                if df is None:
                    continue
                if t in _enrichment_load():  # got enriched meanwhile (e.g. user opened it)
                    continue
                is_mart = t in mart_titles
                ds_name = mart_titles.get(t) if is_mart else profiles.get(t, {}).get("dataset_name", t)
                prof, score = _compute_profile_scorecard(df, ds_name or t)
                _enrich_and_profile(t, ds_name or t, df, prof, score, do_profile=not is_mart)
        except Exception:  # noqa: BLE001
            _log.exception("enrich sweep failed")

    threading.Thread(target=_run, daemon=True).start()


def _registry_upsert(entry: dict) -> None:
    """Insert/replace a mart entry (keyed by table_name), newest first."""
    import json
    with _registry_lock:
        marts = [m for m in _registry_load() if m.get("table_name") != entry.get("table_name")]
        marts.insert(0, entry)
        os.makedirs(_MART_OUT_DIR, exist_ok=True)
        with open(_REGISTRY_PATH, "w", encoding="utf-8") as fh:
            json.dump(marts, fh)


def _onboard_run_and_record(dataset_name, df, source_meta, min_grade):
    """Run a DataFrame through the pipeline + register it; return a UI summary.

    Shared by every onboard lane (API / file / database). The pipeline runs
    WITHOUT synchronous AI enrichment so ingestion stays fast; enrichment (briefing
    + AI column descriptions) then runs automatically in the background and is
    cached for viewing by opening the dataset. The optional quality gate can block
    the DuckDB load when the grade is below ``min_grade``.
    """
    import time as _time
    from core.pipeline import run_pipeline
    from core.database import DuckDBManager
    from core.config import DATABASE_PATH
    from core import history

    t0 = _time.monotonic()
    result = run_pipeline(
        df, dataset_name=dataset_name, source_metadata=source_meta,
        quality_gate_min_grade=min_grade, router=None, enable_ai_enrichment=False,
    )
    duration = round(_time.monotonic() - t0, 2)

    scorecard = result.get("scorecard") or {}
    profile = result.get("profile") or {}
    clean_df = result.get("clean_df")
    table_name = result.get("table_name")
    rows_out = int(len(clean_df)) if clean_df is not None else 0
    columns = [str(c) for c in clean_df.columns] if clean_df is not None else []
    gate_blocked = bool(result.get("gate_blocked", False))

    # Persist the full quality scorecard + profile so the onboard UI can show the
    # per-check quality profile (not just the letter grade) for this source.
    if table_name:
        _store_quality(table_name, scorecard, profile)

    try:
        from datetime import datetime, timezone
        with DuckDBManager(db_path=DATABASE_PATH) as db:
            history.record_run(db, {
                "dataset_name": dataset_name,
                "source_name": (source_meta or {}).get("name", "") or (source_meta or {}).get("source_type", ""),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "duration_s": duration,
                "rows_in": int(len(df)),
                "rows_out": rows_out,
                "grade": scorecard.get("grade", ""),
                "score": float(scorecard.get("score", 0.0)),
                "gate_blocked": gate_blocked,
                "table_name": table_name if not gate_blocked else "",  # not loaded → not listable
            }, {
                "profile": result.get("profile"),
                "scorecard": scorecard,
                "docs": result.get("docs"),
            })
    except Exception:  # noqa: BLE001 — registry write is best-effort
        _log.exception("_onboard_run_and_record: record_run failed")

    # Enrich (AI briefing + column descriptions) and profile the loaded table in
    # the BACKGROUND, so onboarding returns immediately. Status flips to "ready"
    # when both are done; the enrichment is cached and viewable by opening the
    # dataset, and the mart reads the cached profile without re-analyzing.
    if table_name and not gate_blocked:
        _start_background_profile(table_name, dataset_name, clean_df, profile, scorecard)

    return {
        "ok": True,
        "dataset_name": dataset_name,
        "table_name": table_name,
        "rows_in": int(len(df)),
        "rows_out": rows_out,
        "grade": scorecard.get("grade", ""),
        "score": scorecard.get("score", 0.0),
        "duration_s": duration,
        "columns": columns,
        "gate_blocked": gate_blocked,
    }


def create_flask_app() -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
    app.secret_key = "demo-key-not-secure"  # demo only

    # ===== STATIC PAGES =====
    @app.route("/")
    def login_page():
        """Serve the login page."""
        return send_file(LOGIN_HTML, mimetype="text/html")

    @app.route("/pages/<path:filename>")
    def serve_page(filename):
        """Serve HTML pages from /pages directory."""
        if not filename.endswith('.html'):
            filename += '.html'
        page_path = os.path.join(PAGES_DIR, filename)
        if os.path.exists(page_path):
            return send_file(page_path, mimetype="text/html")
        return jsonify({"error": "Page not found"}), 404

    # ===== AUTHENTICATION API =====
    @app.route("/api/login", methods=["POST"])
    def login():
        """Validate credentials and return status.

        Request JSON: {"username": "...", "password": "..."}
        Response: {"ok": true/false, "redirect": "..."}
        """
        data = request.get_json() or {}
        username = data.get("username", "")
        password = data.get("password", "")

        if check_credentials(username, password):
            # Land on the Dashboard — the command centre (quality, history,
            # triage, lineage, privacy audit).
            return jsonify({
                "ok": True,
                "redirect": "http://localhost:8501/pages/dashboard.html"
            })
        return jsonify({"ok": False, "message": "Invalid credentials"}), 401

    # ===== MART API =====
    @app.route("/api/mart/sources", methods=["GET"])
    def mart_sources():
        """Describe what the mart can be built from — including how much raw data.

        Reads the cached parquet row counts so the UI can show the funnel up
        front: hundreds of thousands of source records distilled into 51 state
        rows.

        Response: {"sources": [{"name","kind","detail","volume"}, ...],
                   "grain": "...", "funnel": "..."}
        """
        from analytics.measure_catalog import get_available_measures
        from core.config import CACHE_DIR

        catalog = get_available_measures()
        n_topics = int(catalog["topic"].nunique()) if not catalog.empty else 0

        cms_rows = _parquet_rows(CACHE_DIR / "mart_cms_raw.parquet")
        census_rows = _parquet_rows(CACHE_DIR / "mart_census.parquet")
        cdc_rows = 0
        for p in CACHE_DIR.glob("mart_cdi_*.parquet"):
            n = _parquet_rows(p)
            if n:
                cdc_rows += n

        def vol(n, noun):
            return f"{n:,} {noun}" if n else None

        total = (cms_rows or 0) + (census_rows or 0) + cdc_rows
        n_sources = sum(1 for v in (cdc_rows, cms_rows, census_rows) if v)
        funnel = (
            f"{n_sources} public sources · ≈{total:,} raw records → 51 state rows "
            f"(50 states + DC). Every mart lands at this same grain."
            if total else None
        )

        return jsonify({
            "sources": [
                {
                    "name": "CDC Chronic Disease Indicators",
                    "kind": "measures",
                    "detail": f"{n_topics} topics · {len(catalog)} measures · latest crude prevalence per state",
                    "volume": vol(cdc_rows, "cached indicator rows"),
                },
                {
                    "name": "CMS Medicare Payments",
                    "kind": "fixed",
                    "detail": "Provider payments aggregated to per-state Medicare spend",
                    "volume": vol(cms_rows, "provider records"),
                },
                {
                    "name": "US Census Bureau",
                    "kind": "fixed",
                    "detail": "Population by state (for per-capita rates)",
                    "volume": vol(census_rows, "state population records"),
                },
            ],
            "grain": "51 rows — 50 states + DC",
            "funnel": funnel,
        })

    @app.route("/api/mart/catalog", methods=["GET"])
    def mart_catalog():
        """Return the real CDI measure catalog, topic-grouped, for the measure picker.

        Response: {"topics": {"<topic>": [{"id","label"}, ...]}, "defaults": [...]}
        """
        from analytics.measure_catalog import get_available_measures, DEFAULT_MEASURE_IDS

        catalog = get_available_measures()
        topics: dict[str, list] = {}
        for row in catalog.itertuples():
            topics.setdefault(row.topic, []).append({"id": row.questionid, "label": row.question})
        return jsonify({"topics": topics, "defaults": DEFAULT_MEASURE_IDS})

    @app.route("/api/mart/plan", methods=["POST"])
    def mart_plan():
        """Plan a mart from a natural-language prompt — the real planner, not a mock.

        Validates the prompt against the actual catalog first (cheap, no AI call);
        an unrelated prompt gets an honest "not possible" instead of a fabricated
        plan. Otherwise calls analytics.mart_planner.plan_mart(), which tries the
        cloud-eligible router (Gemini first, Ollama fallback) then a keyword
        heuristic — the prompt and catalog only, never data rows.

        Request JSON: {"prompt": "..."}
        Response (irrelevant prompt): {"possible": false, "message": "..."}
        Response (planned): {"possible": true, "title", "measures": [{"id","label"}, ...],
                   "primary_measure", "primary_label", "narrative_focus",
                   "provider", "used_fallback", "latency_s", "redaction_count"}
        """
        from analytics.measure_catalog import get_available_measures, DEFAULT_MEASURE_IDS
        from analytics.mart_planner import plan_mart

        data = request.get_json() or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "empty_prompt"}), 400

        catalog = get_available_measures()

        # Onboarded metrics the request seems to touch (local keyword match).
        ext_suggestions = _external_suggestions(prompt)
        cdi_matches = _prompt_matches_catalog(prompt, catalog)

        if not cdi_matches and not ext_suggestions:
            return jsonify({
                "possible": False,
                "message": (
                    "That doesn't match anything available — CDC chronic-disease measures, "
                    "CMS Medicare spend, Census population, or the data you've onboarded. "
                    "Try one of the suggestions, or name a specific measure."
                ),
            })

        label_by_id = {row.questionid: row.question for row in catalog.itertuples()}

        if cdi_matches:
            try:
                router = _get_router()
                spec, meta = plan_mart(router, prompt, catalog, DEFAULT_MEASURE_IDS)
            except Exception:  # noqa: BLE001 — AI/network failure must not 500 the demo
                _log.exception("mart_plan: plan_mart() failed")
                return jsonify({
                    "possible": False,
                    "message": "The planner hit an error reaching the AI provider. Try again in a moment.",
                }), 502
            measures = [{"id": m, "label": label_by_id.get(m, m)} for m in spec.measures]
            plan_fields = {
                "title": spec.title,
                "primary_measure": spec.primary_measure,
                "primary_label": label_by_id.get(spec.primary_measure, spec.primary_measure),
                "narrative_focus": spec.narrative_focus,
                "provider": meta.get("provider", "none"),
                "used_fallback": meta.get("used_fallback", True),
                "latency_s": meta.get("latency_s", 0.0),
                "redaction_count": meta.get("redaction_count", 0),
            }
        else:
            # Only onboarded data matched — offer the external suggestions alone.
            measures = []
            first = ext_suggestions[0]
            plan_fields = {
                "title": first["label"], "primary_measure": "",
                "primary_label": "", "narrative_focus": "policy",
                "provider": "none", "used_fallback": True, "latency_s": 0.0, "redaction_count": 0,
            }

        return jsonify({
            "possible": True,
            "measures": measures,
            "external_suggestions": ext_suggestions,
            # Honest limitation note when the question over-reaches the data grain.
            "caveat": _scope_caveat(prompt),
            **plan_fields,
        })

    @app.route("/api/mart/build", methods=["POST"])
    def mart_build():
        """Build the mart from cached/live source data and compute real facts.

        With ONE measure: single-measure facts (that measure vs Medicare spend).
        With 2+ measures: a derived COMBINED-burden analysis — the measures are
        normalised and averaged into a 0-100 burden index, correlated against
        spend, with multi-burden hotspots and pairwise relationships. So picking
        several measures yields combined meaning, not just parallel columns.

        Request JSON: {"measures": [...], "primary_measure": "DIA01", "narrative_focus": "payer", "title": "..."}
        Response: {"mode": "single"|"composite", "states_covered", "measure_col",
                   "measure_label", "facts", "composite", "briefing", "build_metadata"}
        """
        from analytics.mart_builder import MartBuilder
        from analytics.mart_planner import (
            compute_report_facts, compute_composite_facts, compute_scatter, plan_explore,
        )
        from analytics.measure_catalog import get_available_measures, measure_slug, DEFAULT_MEASURE_IDS

        data = request.get_json() or {}
        measures = data.get("measures") or list(DEFAULT_MEASURE_IDS)
        external = data.get("external_measures") or []
        ext_by_id = {e.get("id"): e for e in external if e.get("id")}
        primary = data.get("primary_measure") or measures[0]
        focus = data.get("narrative_focus") or "payer"
        title = data.get("title") or "Custom State Health Mart"
        prompt = (data.get("prompt") or "").strip()
        caveat = data.get("caveat") or None

        # CDC measures feed the deterministic builder; "ext::" ids are onboarded
        # metrics joined in afterwards.
        cdi_measures = [m for m in measures if not str(m).startswith("ext::")]

        try:
            builder = MartBuilder()
            mart_df = builder.build(measures=cdi_measures)
        except Exception:  # noqa: BLE001 — e.g. no network for an uncached measure
            _log.exception("mart_build: MartBuilder.build() failed")
            return jsonify({
                "error": "build_failed",
                "message": "Could not build with the selected measures (likely no cached data and no network).",
            }), 502

        catalog = get_available_measures()
        label_by_id = {row.questionid: row.question for row in catalog.itertuples()}

        # Join any onboarded (external) metrics as extra per-state columns.
        ext_cols: dict = {}   # id -> (column_name, label)
        for m in measures:
            if str(m).startswith("ext::") and m in ext_by_id:
                spec = ext_by_id[m]
                col = _external_col_name(spec)
                try:
                    series = _external_measure_series(spec)
                    if series:
                        mart_df[col] = mart_df["state_abbr"].map(series)
                        ext_cols[m] = (col, spec.get("label") or spec.get("column") or col)
                except Exception:  # noqa: BLE001 — a bad external metric shouldn't kill the build
                    _log.exception("mart_build: external join failed for %s", m)

        # Unified (id, column, label) for every chosen measure, order preserved.
        chosen: list[tuple] = []
        for m in measures:
            if m in ext_cols:
                chosen.append((m, ext_cols[m][0], ext_cols[m][1]))
            elif not str(m).startswith("ext::"):
                chosen.append((m, f"{measure_slug(m)}_prevalence_pct", label_by_id.get(m, m)))
        if not chosen:  # everything dropped — fall back to defaults so we never 500
            for m in list(DEFAULT_MEASURE_IDS):
                chosen.append((m, f"{measure_slug(m)}_prevalence_pct", label_by_id.get(m, m)))

        measure_ids = [c[0] for c in chosen]
        measure_cols = [c[1] for c in chosen]
        measure_labels = [c[2] for c in chosen]
        prim = next((c for c in chosen if c[0] == primary), chosen[0])
        primary_col, primary_label = prim[1], prim[2]

        # Primary-measure facts always computed (drive single mode + top-states).
        facts = compute_report_facts(mart_df, primary_col)
        router = _get_router()

        composite = None
        mode = "single"
        if len(chosen) >= 2:
            composite = compute_composite_facts(mart_df, measure_cols, measure_labels)
            if composite.get("composite_top"):
                mode = "composite"

        # Real per-state scatter (x = burden, y = Medicare spend).
        scatter = compute_scatter(mart_df, measure_cols, primary_col, mode)
        if scatter is not None:
            scatter["x_label"] = (
                "Combined burden index (0–100)" if mode == "composite"
                else f"{primary_label} (%)"
            )
            scatter["y_label"] = "Medicare spend / capita"

        # AI-decided, mart-specific Explore: a cloud director (schema only) picks
        # up to 5 components; the local model narrates the value-based ones.
        try:
            explore = plan_explore(
                router, mode, scatter, facts, composite or {}, measure_labels,
                primary_label, primary_col, title, focus,
            )
        except Exception:  # noqa: BLE001 — never fatal; fall back to no components
            _log.exception("mart_build: plan_explore failed")
            explore = {"components": [], "director": {"provider": "none", "used_fallback": True, "latency_s": 0.0}, "briefing": None}

        briefing = explore.get("briefing") or {"text": "", "generated_by": "unavailable", "latency_s": 0.0}

        # Persist as its OWN queryable DuckDB table (per-mart name), so the SQL
        # console / Ask-the-data CTA is truthful and each mart is distinct.
        table_name = None
        try:
            table_name = _slugify_table(title, measures)
            builder.to_duckdb(mart_df, table_name=table_name)
            # Enrich the mart on-device (column dictionary + briefing) so the
            # chat/SQL schema views show descriptions, just like onboarded sources.
            _start_mart_enrich(table_name, title, mart_df)
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            _log.exception("mart_build: to_duckdb() failed")
            table_name = None

        result = {
            "mode": mode,
            "states_covered": int(len(mart_df)),
            "measure_col": primary_col,
            "measure_label": primary_label,
            "measures": measure_ids,
            "measure_labels": measure_labels,
            "primary_measure": prim[0],
            "narrative_focus": focus,
            "title": title,
            "prompt": prompt,
            "caveat": caveat,
            "facts": facts,
            "composite": composite,
            "scatter": scatter,
            "briefing": briefing,
            "components": explore.get("components", []),
            "director": explore.get("director", {}),
            "table_name": table_name,
            "build_metadata": builder.build_metadata,
        }

        # Register the built mart so it persists across sessions + reopens fast.
        if table_name:
            try:
                from datetime import datetime, timezone
                _registry_upsert({
                    "table_name": table_name,
                    "title": title,
                    "prompt": prompt,
                    "mode": mode,
                    "measure_labels": measure_labels,
                    "states_covered": int(len(mart_df)),
                    "built_at": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                })
            except Exception:  # noqa: BLE001 — registry write is best-effort
                _log.exception("mart_build: registry upsert failed")

        return jsonify(result)

    @app.route("/api/mart/ask", methods=["POST"])
    def mart_ask():
        """Answer a natural-language question against the built mart table.

        Reuses the same NL→SQL pipeline as "Ask the data": PII-scrub → generate
        SQL (cloud-eligible) → validate SELECT-only → run on DuckDB → narrate.

        Request JSON: {"question": "...", "table": "reporting_state_health"}
        Response: {"sql", "columns", "rows", "narration", "provider", "error", "row_count"}
        """
        from core import analyst
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        data = request.get_json() or {}
        question = (data.get("question") or "").strip()
        table = data.get("table") or "reporting_state_health"
        if not question:
            return jsonify({"error": "empty_question"}), 400

        try:
            router = _get_router()
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                if table not in db.list_tables():
                    return jsonify({"error": "no_table",
                                    "message": "Build a mart first — no table to query yet."}), 400
                result = analyst.ask(router, db, table, question)
        except Exception:  # noqa: BLE001 — never 500 the demo on an AI/DB hiccup
            _log.exception("mart_ask failed")
            return jsonify({"error": "ask_failed",
                            "message": "The query engine hit an error. Try rephrasing."}), 502

        df = result.get("df")
        columns, rows = [], []
        if df is not None:
            columns = [str(c) for c in df.columns]
            rows = df.head(100).to_dict("records")

        return jsonify({
            "sql": result.get("sql", ""),
            "valid": result.get("valid", False),
            "columns": columns,
            "rows": rows,
            "row_count": int(len(df)) if df is not None else 0,
            "narration": result.get("narration"),
            "provider": result.get("provider_used", "none"),
            "error": result.get("error"),
            "latency_s": result.get("latency_s", 0.0),
            # PII the scrubber removed before the query left the machine. Aggregates
            # only ({kind,count}) — never the matched values.
            "redactions": result.get("redactions", []),
            "scrubbed": bool(result.get("redactions")),
        })

    @app.route("/api/mart/list", methods=["GET"])
    def mart_list():
        """List previously built marts for the "My marts" gallery.

        Returns only lightweight summaries (not the full result payload), newest
        first. A mart is shown only if its DuckDB table still exists, so the list
        can't offer a mart that would fail to reopen.

        Response: {"marts": [{"table_name","title","prompt","mode",
                   "measure_labels","states_covered","built_at"}, ...]}
        """
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        marts = _registry_load()
        live: set[str] = set()
        try:
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                live = set(db.list_tables())
        except Exception:  # noqa: BLE001 — no DB yet → nothing to list
            live = set()

        out = []
        for m in marts:
            if m.get("table_name") not in live:
                continue
            out.append({k: m.get(k) for k in (
                "table_name", "title", "prompt", "mode", "measure_labels",
                "states_covered", "built_at",
            )})
        return jsonify({"marts": out})

    @app.route("/api/mart/open", methods=["GET"])
    def mart_open():
        """Return the full stored Explore result for a previously built mart.

        Lets "My marts" reopen a mart instantly (no rebuild, no new AI calls) and
        survives logout, since the result was persisted at build time.

        Query: ?table=<table_name>
        Response: the same payload shape as /api/mart/build, or 404.
        """
        table = (request.args.get("table") or "").strip()
        for m in _registry_load():
            if m.get("table_name") == table and m.get("result"):
                return jsonify(m["result"])
        return jsonify({"error": "not_found", "message": "No stored mart for that table."}), 404

    @app.route("/api/mart/onboarded", methods=["GET"])
    def mart_onboarded():
        """List datasets onboarded via the Onboard screen (honest, from history).

        Reads the pipeline_runs registry (not a raw table dump), so only genuinely
        onboarded datasets show — and only if their DuckDB table still exists. The
        mart itself is still built from the three curated feeds; these are candidates
        the user can ask the AI to evaluate for inclusion.

        Response: {"sources": [{"table","dataset_name","source_name","rows","grade"}, ...]}
        """
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH
        from core import history

        out = []
        try:
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                live = set(db.list_tables())
                runs = history.list_runs(db, 200)
                seen: set[str] = set()
                if not runs.empty:
                    for _, r in runs.iterrows():
                        t = str(r.get("table_name") or "")
                        if not t or t in seen or t not in live:
                            continue
                        if t in _RESERVED_TABLES or t.startswith("mart_") or t == "pipeline_runs":
                            continue
                        seen.add(t)
                        out.append({
                            "table": t,
                            "dataset_name": str(r.get("dataset_name") or t),
                            "source_name": str(r.get("source_name") or ""),
                            "rows": int(r.get("rows_out") or 0),
                            "grade": str(r.get("grade") or ""),
                            "started_at": str(r.get("started_at") or ""),
                        })
        except Exception:  # noqa: BLE001 — no DB/history yet → empty list
            _log.exception("mart_onboarded failed")
            out = []
        return jsonify({"sources": out})

    @app.route("/api/mart/onboarded_profiles", methods=["GET"])
    def mart_onboarded_profiles():
        """Return pre-analyzed profiles for onboarded sources (mart's optional lane).

        Reads the cached profiles written at onboard time. For any onboarded table
        that predates profiling (no cache yet), it computes + caches the profile on
        first read, so the list is always complete. The mart shows these directly —
        no per-view analysis.

        Response: {"sources": [{table,dataset_name,n_rows,n_fields,state_col,
                   n_states,relevant,reason,metrics,generated_by}, ...]}
        """
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH
        from core import history

        live: set[str] = set()
        runs_by_table: dict = {}
        try:
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                live = set(db.list_tables())
                runs = history.list_runs(db, 200)
                if not runs.empty:
                    for _, r in runs.iterrows():
                        t = str(r.get("table_name") or "")
                        if t and t not in runs_by_table:
                            runs_by_table[t] = str(r.get("dataset_name") or t)
        except Exception:  # noqa: BLE001
            pass

        stored = _source_profiles_load()
        out = []
        for t, ds in runs_by_table.items():
            if t not in live or t in _RESERVED_TABLES or t.startswith("mart_") or t == "pipeline_runs":
                continue
            prof = stored.get(t)
            if prof is None:
                prof = _store_source_profile(t, ds)  # compute + cache on first read
            if prof:
                out.append(prof)
        # useful (can-improve) first, then by rows
        out.sort(key=lambda p: (0 if p.get("relevant") else 1, -int(p.get("n_rows") or 0)))
        return jsonify({"sources": out})

    # ===== ONBOARD API =====
    @app.route("/api/onboard/catalog", methods=["GET"])
    def onboard_catalog():
        """Describe the three ingestion lanes for the Onboard UI.

        Response: {"api_sources":[{name,label,params}], "db_engines":[{id,label,
                   installed,pip}], "file_formats":[...]}
        """
        from core.driver_manager import DRIVER_SPECS, is_driver_installed

        api_sources = []
        try:
            reg = _get_registry()
            for name, src in sorted(reg._sources.items()):
                # "api" + the API-backed "generic" pull sources (WHO, openFDA);
                # everything file/db-based is handled by its own lane.
                if getattr(src, "source_type", "") not in ("api", "generic"):
                    continue
                api_sources.append({
                    "name": name,
                    "label": getattr(src, "description", None) or name,
                    "params": _api_param_schema(name),
                })
        except Exception:  # noqa: BLE001 — registry issues shouldn't blank the page
            _log.exception("onboard_catalog: registry failed")

        db_engines = []
        for eid, spec in DRIVER_SPECS.items():
            if spec.get("url_template") is None and eid not in ("databricks", "bigquery"):
                continue  # object_storage etc. — not a table-query lane
            try:
                installed = is_driver_installed(eid)
            except Exception:  # noqa: BLE001
                installed = False
            db_engines.append({
                "id": eid, "label": spec.get("label", eid),
                "installed": installed, "pip": spec.get("pip", []),
                "default_port": spec.get("default_port"),
            })

        return jsonify({
            "api_sources": api_sources,
            "db_engines": db_engines,
            "file_formats": ["csv", "tsv", "json", "parquet", "xlsx"],
        })

    @app.route("/api/onboard/run_api", methods=["POST"])
    def onboard_run_api():
        """Pull from a registered API source and run it through the pipeline.

        Request JSON: {source, params:{}, dataset_name, max_records, min_grade}
        """
        data = request.get_json() or {}
        source_name = (data.get("source") or "").strip()
        dataset_name = (data.get("dataset_name") or source_name or "api_dataset").strip()
        params = data.get("params") or {}
        max_records = int(data.get("max_records") or 1000)
        min_grade = data.get("min_grade") or None

        reg = _get_registry()
        source = reg.get(source_name)
        if source is None or getattr(source, "source_type", "") not in ("api", "generic"):
            return jsonify({"error": "bad_source", "message": "Unknown API source."}), 400

        try:
            kwargs = _api_extract_kwargs(source_name, params)
            df = source.extract(max_records=max_records, **kwargs)
        except Exception:  # noqa: BLE001 — network/param failure
            _log.exception("onboard_run_api: extract failed")
            return jsonify({"error": "extract_failed",
                            "message": "The source pull failed — check parameters and connectivity."}), 502
        if df is None or df.empty:
            return jsonify({"error": "empty", "message": "The source returned no rows for those parameters."}), 400

        try:
            summary = _onboard_run_and_record(dataset_name, df, source.get_metadata(), min_grade)
        except Exception:  # noqa: BLE001
            _log.exception("onboard_run_api: pipeline failed")
            return jsonify({"error": "pipeline_failed", "message": "The pipeline hit an error."}), 502
        return jsonify(summary)

    @app.route("/api/onboard/upload", methods=["POST"])
    def onboard_upload():
        """Ingest one or more uploaded files (CSV/TSV/JSON/Parquet/XLSX).

        Each file is saved to the uploads cache, read by FileSource (format
        detected by extension), run through the pipeline, and registered.

        Multipart form: file=<one or more files>, optional min_grade.
        Response: {"ok", "results":[<per-file summary or error>]}
        """
        from core.config import CACHE_DIR
        from ingestion.file_source import FileSource

        files = request.files.getlist("file")
        if not files:
            return jsonify({"error": "no_file", "message": "No file was uploaded."}), 400

        min_grade = request.form.get("min_grade") or None

        uploads = CACHE_DIR / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        fs = FileSource()
        results = []

        for f in files:
            if not f or not f.filename:
                continue
            fname = os.path.basename(f.filename)
            dataset_name = os.path.splitext(fname)[0] or "onboarded_dataset"
            dest = uploads / fname
            try:
                f.save(str(dest))
                df = fs.extract(filepath=str(dest))
            except Exception:  # noqa: BLE001 — unsupported/corrupt file
                _log.exception("onboard_upload: could not read %s", fname)
                results.append({"ok": False, "dataset_name": dataset_name,
                                "message": f"Could not read {fname} — unsupported or malformed file."})
                continue
            if df is None or df.empty:
                results.append({"ok": False, "dataset_name": dataset_name,
                                "message": f"{fname} has no rows."})
                continue
            try:
                meta = {"source_type": "file", "name": "file upload", "filename": fname}
                results.append(_onboard_run_and_record(dataset_name, df, meta, min_grade))
            except Exception:  # noqa: BLE001
                _log.exception("onboard_upload: pipeline failed for %s", fname)
                results.append({"ok": False, "dataset_name": dataset_name,
                                "message": f"The pipeline hit an error processing {fname}."})

        return jsonify({"ok": True, "results": results})

    @app.route("/api/onboard/db_test", methods=["POST"])
    def onboard_db_test():
        """Test a database connection and list its tables.

        Request JSON: {engine_id, conn:{...}}  (conn holds host/port/database/user/
        password or sqlite path, etc. — the user's own credentials, used only to
        connect; never stored or logged.)
        Response: {"ok", "tables":[...]} or {"error","message"}
        """
        from core.driver_manager import DRIVER_SPECS, is_driver_installed
        from ingestion.database_source import DatabaseSource

        data = request.get_json() or {}
        engine_id = (data.get("engine_id") or "").strip()
        conn = data.get("conn") or {}
        if engine_id not in DRIVER_SPECS:
            return jsonify({"error": "bad_engine", "message": "Unknown database engine."}), 400
        if not is_driver_installed(engine_id):
            pins = DRIVER_SPECS[engine_id].get("pip", [])
            return jsonify({"error": "driver_missing",
                            "message": f"Driver not installed. Run: pip install {' '.join(pins)}"}), 400
        try:
            src = DatabaseSource()
            src.configure(engine_id=engine_id, **{k: v for k, v in conn.items() if v not in (None, "")})
            if not src.connect():
                return jsonify({"error": "connect_failed",
                                "message": "Connection failed — check host, credentials, and driver."}), 400
            tables = src.list_tables()
        except Exception:  # noqa: BLE001
            _log.exception("onboard_db_test failed")
            return jsonify({"error": "connect_failed",
                            "message": "Connection failed — check host, credentials, and driver."}), 400
        return jsonify({"ok": True, "tables": tables})

    @app.route("/api/onboard/run_db", methods=["POST"])
    def onboard_run_db():
        """Extract a table (or SQL query) from a database and run the pipeline.

        Request JSON: {engine_id, conn:{...}, table|query, dataset_name, max_rows,
                       min_grade}
        """
        from core.driver_manager import DRIVER_SPECS, is_driver_installed
        from ingestion.database_source import DatabaseSource

        data = request.get_json() or {}
        engine_id = (data.get("engine_id") or "").strip()
        conn = data.get("conn") or {}
        table = (data.get("table") or "").strip() or None
        query = (data.get("query") or "").strip() or None
        dataset_name = (data.get("dataset_name") or table or "db_extract").strip()
        max_rows = int(data.get("max_rows") or 100_000)
        min_grade = data.get("min_grade") or None

        if engine_id not in DRIVER_SPECS or not is_driver_installed(engine_id):
            return jsonify({"error": "driver_missing", "message": "Driver not installed for that engine."}), 400
        if not (table or query):
            return jsonify({"error": "no_target", "message": "Pick a table or write a query."}), 400

        try:
            src = DatabaseSource()
            src.configure(engine_id=engine_id, **{k: v for k, v in conn.items() if v not in (None, "")})
            if not src.connect():
                return jsonify({"error": "connect_failed", "message": "Connection failed."}), 400
            df = src.extract(table=table, query=query, max_rows=max_rows) if table else src.extract(query=query, max_rows=max_rows)
        except Exception:  # noqa: BLE001
            _log.exception("onboard_run_db: extract failed")
            return jsonify({"error": "extract_failed", "message": "The extract failed — check the table/query."}), 502
        if df is None or df.empty:
            return jsonify({"error": "empty", "message": "The extract returned no rows."}), 400

        try:
            summary = _onboard_run_and_record(dataset_name, df, src.get_metadata(), min_grade)
        except Exception:  # noqa: BLE001
            _log.exception("onboard_run_db: pipeline failed")
            return jsonify({"error": "pipeline_failed", "message": "The pipeline hit an error."}), 502
        return jsonify(summary)

    @app.route("/api/onboard/profile_status", methods=["GET"])
    def onboard_profile_status():
        """Report background-profiling status for a just-onboarded table.

        Response: {"status": "profiling"|"ready"|"error"|"unknown",
                   "relevant": bool, "n_metrics": int, "reason": str}
        """
        table = (request.args.get("table") or "").strip()
        status = _get_profiling_status(table)
        prof = _source_profiles_load().get(table)
        if prof is not None and status in (None, "ready"):
            return jsonify({
                "status": "ready",
                "relevant": bool(prof.get("relevant")),
                "n_metrics": len(prof.get("metrics", [])),
                "reason": prof.get("reason", ""),
            })
        if status is None and prof is None:
            return jsonify({"status": "unknown"})
        return jsonify({"status": status or "profiling"})

    @app.route("/api/onboard/delete", methods=["POST"])
    def onboard_delete():
        """Delete an onboarded dataset (remove from DuckDB, cache, enrichment, profiles).

        Request JSON: {table}
        Response: {"ok": true}
        """
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        data = request.get_json() or {}
        table = (data.get("table") or "").strip()
        if not table:
            return jsonify({"error": "no_table"}), 400

        try:
            # Drop from DuckDB
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                if table in db.list_tables():
                    db.execute(f'DROP TABLE IF EXISTS "{table}"')

            # Remove enrichment cache
            with _enrichment_lock:
                enr = _enrichment_load()
                enr.pop(table, None)
                _enrichment_save(enr)

            # Remove profile cache
            with _profiles_lock:
                prof = _source_profiles_load()
                prof.pop(table, None)
                _source_profiles_save(prof)

            # Remove quality cache
            with _quality_lock:
                q = _quality_load()
                q.pop(table, None)
                _quality_save(q)

            # Clear profiling status
            _set_profiling_status(table, None)

            # Remove all pipeline_runs history rows for this table so the
            # dashboard no longer shows stale entries after a delete.
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                try:
                    db.execute(
                        "DELETE FROM pipeline_runs WHERE table_name = ?",
                        table,
                    )
                except Exception:
                    pass  # pipeline_runs may not exist yet; that's fine

            _log.info("deleted dataset: %s", table)
            return jsonify({"ok": True})
        except Exception:  # noqa: BLE001
            _log.exception("delete failed for %s", table)
            return jsonify({"error": "failed"}), 500

    @app.route("/api/onboard/enrich_now", methods=["POST"])
    def onboard_enrich_now():
        """Trigger enrichment now (non-blocking) for an existing dataset.

        Request JSON: {table}
        Response: {"ok": true, "status": "enriching"}
        """
        data = request.get_json() or {}
        table = (data.get("table") or "").strip()
        if not table:
            return jsonify({"error": "no_table", "message": "Specify a table name."}), 400

        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        try:
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                if table not in db.list_tables():
                    return jsonify({"error": "not_found", "message": f"Table {table} not found."}), 404
                clean_df = db.query(f'SELECT * FROM "{table}"')
            # Compute a real profile + scorecard from the loaded table so the
            # briefing reflects actual row/column counts and grade (not empty
            # placeholders), then enrich on-device in the background.
            ds_name = _source_profiles_load().get(table, {}).get("dataset_name", table)
            prof, scorecard = _compute_profile_scorecard(clean_df, ds_name)
            _start_background_profile(table, ds_name, clean_df, prof, scorecard)
            return jsonify({"ok": True, "status": "enriching"})
        except Exception:  # noqa: BLE001
            _log.exception("enrich_now failed for %s", table)
            return jsonify({"error": "failed", "message": "Could not trigger enrichment."}), 500

    @app.route("/api/onboard/enrichment", methods=["GET"])
    def onboard_enrichment():
        """Return the cached AI enrichment for an onboarded table.

        Enrichment runs on-device in the background at onboard time; this lets the
        UI show it when a dataset is opened. While it's still being generated,
        returns ready=false so the UI can say "still running on-device".

        Response: {"ok", "ready", "briefing": {...}|null, "column_docs": [...],
                   "generated_by": str}
        """
        table = (request.args.get("table") or "").strip()
        enr = _get_enrichment(table)
        if enr is None:
            # Not cached. Report a precise status so the UI knows whether to WAIT
            # (something is running) or TRIGGER it (nothing has ever run):
            #   running  — a background job is generating it now
            #   error    — the last attempt failed (message in `error`)
            #   absent   — never enriched; the UI should kick it off
            estatus, eerr = _get_enrich_status(table)
            if estatus == "running":
                status = "running"
            elif estatus == "error":
                status = "error"
            else:
                status = "absent"
            return jsonify({"ok": True, "ready": False,
                            "status": status, "error": eerr,
                            "briefing": None, "column_docs": []})
        return jsonify({
            "ok": True, "ready": True,
            "briefing": enr.get("briefing"),
            "column_docs": enr.get("column_docs", []),
            "generated_by": enr.get("generated_by", ""),
        })

    @app.route("/api/onboard/quality", methods=["GET"])
    def onboard_quality():
        """Return a table's data-quality profile (the pipeline's scorecard).

        Deterministic + on-device: the profiler + quality checker are pandas-only,
        so unlike AI enrichment this is fast and always available. Served from the
        cache; computed + cached on first read for any table that predates the
        quality store (or was onboarded before it existed).

        Query: ?table=<name>
        Response: {"ok", "quality": {...}|null}
        """
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        table = (request.args.get("table") or "").strip()
        if not table:
            return jsonify({"error": "no_table"}), 400
        q = _get_quality(table)
        if q is not None:
            return jsonify({"ok": True, "quality": q})
        # Not cached — compute it now from the loaded table and cache it.
        try:
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                if table not in db.list_tables():
                    return jsonify({"ok": True, "quality": None})
                df = db.query(f'SELECT * FROM "{table}"')
            ds_name = _source_profiles_load().get(table, {}).get("dataset_name", table)
            profile, scorecard = _compute_profile_scorecard(df, ds_name)
            _store_quality(table, scorecard, profile)
            return jsonify({"ok": True, "quality": _get_quality(table)})
        except Exception:  # noqa: BLE001
            _log.exception("onboard_quality failed for %s", table)
            return jsonify({"ok": True, "quality": None})

    # ===== DRIVER INSTALL (database lane) =====
    @app.route("/api/onboard/install_driver", methods=["POST"])
    def onboard_install_driver():
        """Install the pinned DB driver for an engine, one click (pinned pip).

        Runs the same pinned-only ``install_driver`` policy as the original app,
        then re-checks availability so the UI can unlock the connection form
        without a manual reload.

        Request JSON: {engine_id}
        Response: {"ok", "installed", "message", "log"}
        """
        from core.driver_manager import DRIVER_SPECS, install_driver, is_driver_installed

        data = request.get_json() or {}
        engine_id = (data.get("engine_id") or "").strip()
        if engine_id not in DRIVER_SPECS:
            return jsonify({"ok": False, "message": "Unknown database engine."}), 400
        try:
            success, output = install_driver(engine_id)
        except Exception:  # noqa: BLE001
            _log.exception("install_driver failed for %s", engine_id)
            return jsonify({"ok": False, "installed": False,
                            "message": "The install command hit an error.", "log": ""}), 500
        installed = is_driver_installed(engine_id)
        label = DRIVER_SPECS[engine_id].get("label", engine_id)
        if success and installed:
            msg = f"{label} driver installed — you can connect now."
        elif success:
            msg = f"Install finished, but {label} still isn't importable. A restart may be needed."
        else:
            msg = f"Could not install the {label} driver. See the log."
        return jsonify({"ok": bool(success), "installed": bool(installed),
                        "message": msg, "log": output or ""})

    # ===== QUERY SURFACE (Ask the data + SQL console) =====
    @app.route("/api/query/tables", methods=["GET"])
    def query_tables():
        """List queryable tables for the chat/SQL pickers (marts + onboarded).

        Excludes bookkeeping tables (pipeline_runs). Marks each table's kind so
        the UI can group them: "mart" (built reporting marts) vs "source"
        (onboarded datasets). Row counts come from a cheap COUNT(*).

        Response: {"tables": [{"table","label","kind","rows"}, ...]}
        """
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        out = []
        try:
            marts = {m.get("table_name"): m.get("title") for m in _registry_load()}
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                for t in db.list_tables():
                    if t == "pipeline_runs":
                        continue
                    try:
                        rows = int(db.query(f'SELECT COUNT(*) AS n FROM "{t}"').iloc[0]["n"])
                    except Exception:  # noqa: BLE001
                        rows = 0
                    is_mart = t in _RESERVED_TABLES or t.startswith("mart_") or t in marts
                    out.append({
                        "table": t,
                        "label": marts.get(t) or t,
                        "kind": "mart" if is_mart else "source",
                        "rows": rows,
                    })
        except Exception:  # noqa: BLE001
            _log.exception("query_tables failed")
            out = []
        # Marts first, then onboarded sources; alphabetical within each group.
        out.sort(key=lambda r: (r["kind"] != "mart", r["table"].lower()))
        return jsonify({"tables": out})

    @app.route("/api/query/schema", methods=["GET"])
    def query_schema():
        """Return a table's columns with AI descriptions where available.

        Prefers the on-device enrichment column descriptions (so the schema peek
        matches what the dataset page shows); falls back to bare name/type.

        Query: ?table=<name>
        Response: {"columns": [{"name","type","description"}], "n_rows": int}
        """
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        table = (request.args.get("table") or "").strip()
        if not table:
            return jsonify({"error": "no_table"}), 400
        enr = _get_enrichment(table) or {}
        desc_by_col = {c.get("name"): c.get("description", "")
                       for c in (enr.get("column_docs") or [])}
        cols, n_rows = [], 0
        try:
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                if table not in db.list_tables():
                    return jsonify({"error": "no_table", "message": "That table no longer exists."}), 404
                for c in db.get_schema(table):
                    name = c["column_name"]
                    cols.append({"name": name, "type": str(c["column_type"]),
                                 "description": desc_by_col.get(name, "")})
                try:
                    n_rows = int(db.query(f'SELECT COUNT(*) AS n FROM "{table}"').iloc[0]["n"])
                except Exception:  # noqa: BLE001
                    n_rows = 0
        except Exception:  # noqa: BLE001
            _log.exception("query_schema failed for %s", table)
            return jsonify({"error": "failed"}), 500
        return jsonify({"columns": cols, "n_rows": n_rows})

    @app.route("/api/query/starters", methods=["GET"])
    def query_starters():
        """Return instant, schema-based starter questions for a table.

        router=None → pure template generation (no model call, no wait), matching
        the chat screen's instant fallback behaviour.

        Query: ?table=<name>
        Response: {"questions": [...]}
        """
        from core import analyst
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        table = (request.args.get("table") or "").strip()
        if not table:
            return jsonify({"questions": []})
        try:
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                if table not in db.list_tables():
                    return jsonify({"questions": []})
                qs = analyst.starter_questions(None, db, table)
        except Exception:  # noqa: BLE001
            _log.exception("query_starters failed for %s", table)
            qs = []
        return jsonify({"questions": list(qs)})

    @app.route("/api/sql/run", methods=["POST"])
    def sql_run():
        """Execute a read-only SQL query for the SQL console.

        Read-only BY DESIGN: every query passes analyst.validate_sql first (the
        same SELECT-only policy as the AI path), so mutations are rejected, not
        executed. Nothing leaves the machine.

        Request JSON: {sql}
        Response: {"ok","columns","rows","row_count","elapsed_ms"} or
                  {"ok":false,"rejected":bool,"reason":str}
        """
        import time as _time
        from core import analyst
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH

        data = request.get_json() or {}
        raw = (data.get("sql") or "").strip()
        if not raw:
            return jsonify({"ok": False, "rejected": True, "reason": "Write a query first."}), 400

        cleaned = analyst.clean_sql(raw)
        valid, reason = analyst.validate_sql(cleaned)
        if not valid:
            return jsonify({"ok": False, "rejected": True,
                            "reason": reason or "The console is read-only — SELECT queries only."})
        try:
            t0 = _time.monotonic()
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                df = db.query(cleaned)
            elapsed_ms = round((_time.monotonic() - t0) * 1000.0, 1)
        except Exception as exc:  # noqa: BLE001
            _log.exception("sql_run failed")
            err = "; ".join(str(a) for a in exc.args) or str(exc)
            return jsonify({"ok": False, "rejected": False, "reason": f"DuckDB error: {err}"})
        columns = [str(c) for c in df.columns]
        rows = df.head(500).to_dict("records")
        return jsonify({"ok": True, "sql": cleaned, "columns": columns, "rows": rows,
                        "row_count": int(len(df)), "elapsed_ms": elapsed_ms})

    # ===== DASHBOARD API =====
    def _load_runs(limit: int = 500) -> list:
        """Return recorded pipeline runs as JSON-safe records, newest first."""
        from core import history
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH
        try:
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                df = history.list_runs(db, limit)
            if df is None or df.empty:
                return []
            return [_run_record(r) for r in df.to_dict("records")]
        except Exception:  # noqa: BLE001
            _log.exception("dashboard: list_runs failed")
            return []

    @app.route("/api/dashboard/runs", methods=["GET"])
    def dashboard_runs():
        """List recorded pipeline runs, newest first (real data now).

        Response: {"runs": [{run_id, dataset_name, grade, rows_in, rows_out, ...}]}
        """
        return jsonify({"runs": _load_runs(500)})

    @app.route("/api/dashboard/overview", methods=["GET"])
    def dashboard_overview():
        """Headline stats + privacy summary for the Dashboard hero row.

        Response: {datasets, marts, total_runs, avg_score, top_grade, grade_dist,
                   last_run, privacy:{cloud_prompt_chars, total_redactions,
                   cloud_calls, local_calls, proof}}
        """
        from core.database import DuckDBManager
        from core.config import DATABASE_PATH
        from core import audit

        runs = _load_runs(500)
        loaded = [r for r in runs if not r["gate_blocked"]]
        grade_dist: dict = {}
        for r in loaded:
            g = r["grade"] or "—"
            grade_dist[g] = grade_dist.get(g, 0) + 1
        scores = [r["score"] for r in loaded if r["score"]]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        top_grade = max(grade_dist, key=grade_dist.get) if grade_dist else "—"
        last_run = runs[0]["started_at"] if runs else ""

        datasets = marts = 0
        try:
            registry_marts = {m.get("table_name") for m in _registry_load()}
            with DuckDBManager(db_path=DATABASE_PATH) as db:
                for t in db.list_tables():
                    if t == "pipeline_runs":
                        continue
                    if t in _RESERVED_TABLES or t.startswith("mart_") or t in registry_marts:
                        marts += 1
                    else:
                        datasets += 1
        except Exception:  # noqa: BLE001
            _log.exception("dashboard_overview: table count failed")

        m = _audit_metrics(audit.read_audit(1000))
        return jsonify({
            "datasets": datasets, "marts": marts,
            "total_runs": len(runs), "avg_score": avg_score,
            "top_grade": top_grade, "grade_dist": grade_dist, "last_run": last_run,
            "privacy": {
                "cloud_prompt_chars": m["cloud_prompt_chars"],
                "total_redactions": m["total_redactions"],
                "cloud_calls": m["cloud_calls"], "local_calls": m["local_calls"],
                "proof": _proof_line(m),
            },
        })

    @app.route("/api/dashboard/run/<run_id>", methods=["GET"])
    def dashboard_run(run_id):
        """Full detail for one run: quality, triage issues, lineage, briefing.

        Response: {ok, meta, quality, issues, lineage, briefing, column_docs,
                   explanations}
        """
        from core import history

        artifacts = history.load_artifacts(run_id)
        profile = artifacts.get("profile") or {}
        scorecard = artifacts.get("scorecard") or {}
        docs = artifacts.get("docs") or {}

        meta = next((r for r in _load_runs(500) if r["run_id"] == run_id), {})
        table_name = meta.get("table_name") or ""

        # Briefing + data dictionary: prefer the on-device enrichment store
        # (kept fresh + cached), fall back to the run's saved artifacts.
        enr = _get_enrichment(table_name) if table_name else None
        briefing = (enr or {}).get("briefing") or artifacts.get("briefing")
        column_docs = (enr or {}).get("column_docs") or []
        if not column_docs and docs.get("data_dictionary"):
            column_docs = [{"name": e.get("column_name", ""),
                            "description": e.get("description", "") or e.get("business_meaning", "")}
                           for e in docs.get("data_dictionary", [])]

        return jsonify({
            "ok": True,
            "meta": meta,
            "quality": _build_quality_record(scorecard, profile),
            "issues": _merge_issues(profile, scorecard),
            "lineage": _lineage_nodes(artifacts),
            "briefing": briefing,
            "column_docs": column_docs,
            "explanations": artifacts.get("issue_explanations") or [],
        })

    @app.route("/api/dashboard/audit", methods=["GET"])
    def dashboard_audit():
        """Privacy receipts: proof line, headline metrics, recent AI calls.

        Response: {proof, metrics, records:[{ts,task,provider,model,latency_s,
                   prompt_chars,redaction_count,success}]}
        """
        from core import audit
        records = audit.read_audit(500)
        m = _audit_metrics(records)
        return jsonify({"proof": _proof_line(m), "metrics": m, "records": records[:80]})

    @app.route("/api/dashboard/compare", methods=["GET"])
    def dashboard_compare():
        """Schema-drift comparison between two runs (analytics/drift).

        Query: ?a=<run_id>&b=<run_id>
        Response: {ok, diff:{...}, has_drift}
        """
        from core import history
        from analytics import drift

        a = (request.args.get("a") or "").strip()
        b = (request.args.get("b") or "").strip()
        if not a or not b:
            return jsonify({"ok": False, "error": "need_two_runs"}), 400
        diff = drift.compare_runs(history.load_artifacts(a), history.load_artifacts(b))
        return jsonify({"ok": True, "diff": diff, "has_drift": drift.has_drift(diff)})

    @app.route("/api/dashboard/triage/explain", methods=["POST"])
    def dashboard_triage_explain():
        """Explain ONE quality issue with the local model, persist, return it.

        On-device only (value-touching task). Request JSON: {run_id, check_name,
        detail}. Response: {ok, explanation:{explanation, suggested_fix,
        generated_by, issue}}
        """
        from core import history, enrich

        data = request.get_json() or {}
        run_id = (data.get("run_id") or "").strip()
        check_name = (data.get("check_name") or "").strip()
        detail = data.get("detail") or ""
        if not run_id or not check_name:
            return jsonify({"ok": False, "error": "bad_request"}), 400

        artifacts = history.load_artifacts(run_id)
        profile = artifacts.get("profile") or {}
        docs = artifacts.get("docs") or {}
        check_dict = {"check": check_name, "passed": False, "detail": detail}
        try:
            new_items = enrich.explain_issues(
                _get_router(), [check_dict], _triage_schema(profile, docs), top_n=1)
        except Exception:  # noqa: BLE001
            _log.exception("triage explain failed for %s", check_name)
            return jsonify({"ok": False, "error": "explain_failed"}), 500
        if not new_items:
            return jsonify({"ok": False, "error": "no_explanation"}), 500
        existing = artifacts.get("issue_explanations") or []
        updated = [it for it in existing
                   if str(it.get("issue", {}).get("check", "")) != check_name]
        updated.extend(new_items)
        try:
            history.save_artifact(run_id, "issue_explanations", updated)
        except Exception:  # noqa: BLE001 — persistence is best-effort
            _log.warning("could not persist explanation for %s", run_id)
        return jsonify({"ok": True, "explanation": new_items[0]})

    # ===== SESSION / OPERATOR =====
    @app.route("/api/whoami", methods=["GET"])
    def whoami():
        """Identify the signed-in operator for the topbar avatar menu.

        Single-operator demo: reports the configured master username (or a
        generic 'operator' label when auth is disabled). Response:
        {username, initials, authed}
        """
        from core import config
        username = config.MASTER_USERNAME or "operator"
        return jsonify({
            "username": username,
            "initials": _initials(username),
            "authed": bool(config.MASTER_USERNAME and config.MASTER_PASSWORD),
        })

    # ===== MODELS / ROUTING MODE =====
    @app.route("/api/models", methods=["GET"])
    def models_list():
        """List the AI models and the current routing mode for the sidebar panel.

        Response: {"mode": "hybrid"|"local_only",
                   "providers": [{id,label,kind,model,available}, ...]}
        """
        from core.config import OLLAMA_MODEL, GEMINI_MODEL
        router = _get_router()
        try:
            gem = bool(router._gemini.is_available())
        except Exception:  # noqa: BLE001
            gem = False
        try:
            oll = bool(router._ollama.is_available())
        except Exception:  # noqa: BLE001
            oll = False
        return jsonify({
            "mode": "local_only" if router.force_local else "hybrid",
            "providers": [
                {"id": "gemini", "label": "Gemini", "kind": "cloud", "model": GEMINI_MODEL, "available": gem},
                {"id": "ollama", "label": "Ollama", "kind": "local", "model": OLLAMA_MODEL, "available": oll},
            ],
        })

    @app.route("/api/models/mode", methods=["POST"])
    def models_mode():
        """Set the routing mode.

        'hybrid'     — cloud (Gemini) for metadata-only tasks, local for anything
                       touching data values (default).
        'local_only' — everything runs on the local model; nothing goes to cloud.

        Request JSON: {"mode": "hybrid"|"local_only"}
        """
        data = request.get_json() or {}
        mode = (data.get("mode") or "").strip()
        if mode not in ("hybrid", "local_only"):
            return jsonify({"error": "bad_mode", "message": "mode must be 'hybrid' or 'local_only'"}), 400
        _get_router().force_local = (mode == "local_only")
        return jsonify({"ok": True, "mode": mode})

    # ===== HEALTH CHECK =====
    @app.route("/health")
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok"})

    # Proactively enrich any already-loaded dataset that has no enrichment yet, so
    # AI enrichment is available without the user ever opening the dataset.
    _start_enrich_sweep()

    return app


def start_flask_server() -> None:
    """Start Flask server on port 8501 in a background thread."""
    app = create_flask_app()
    thread = threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1",
            port=8501,
            debug=False,
            use_reloader=False,
            threaded=True
        ),
        daemon=True
    )
    thread.start()
