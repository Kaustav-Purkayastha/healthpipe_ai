"""
core/enrich.py — AI enrichment functions for HealthPipe AI v2.

Three functions that use the AIRouter to add AI-generated narrative to pipeline
output.  Every function degrades gracefully to rule-based text when the router
returns None (Ollama down, no key, rate-limited).  Every AI call is audit-logged.

Privacy invariants (enforced here and by the router):
  - BRIEFING / COLUMN_DESCRIPTIONS / ISSUE_EXPLANATION → local Ollama only.
  - Sample data values are included only in local prompts; cloud never sees them.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import pandas as pd

from core.audit import log_ai_call
from core.config import OLLAMA_MODEL
from core.router import AIRouter, TaskType
from core.utils import get_logger

_log = get_logger(__name__)

# Columns are batched in groups of 7 for description generation.
# WHY: one prompt per column would mean 20-30+ Ollama calls (quadrupled latency);
# batching to 7 was benchmarked as the best quality/speed tradeoff on CPU inference.
_DESCRIPTION_BATCH_SIZE = 7

# Severity rank for quality-checker check names (lower number = more severe).
_CHECK_SEVERITY: dict[str, int] = {
    "overall_completeness": 0,
    "duplicate_rows": 1,
    "null_rate": 2,
    "no_negatives": 3,
    "extreme_outliers": 4,
    "type_consistency": 5,
    "uniqueness": 6,
}


# ---------------------------------------------------------------------------
# 1.  Executive briefing
# ---------------------------------------------------------------------------

def generate_briefing(
    router: AIRouter,
    dataset_name: str,
    profile: dict,
    scorecard: dict,
) -> dict:
    """Generate a ~150-word executive briefing of a dataset.

    Builds a FACT SHEET from computed aggregates only — no raw data rows —
    then prompts the router (task BRIEFING, always local) for prose.
    Falls back to a template when the router returns None.

    Args:
        router:       AIRouter instance (router.generate may return None).
        dataset_name: Human-readable dataset name.
        profile:      Output of ProfilerAgent.run().
        scorecard:    Output of QualityCheckerAgent.run().

    Returns:
        Dict with keys:
            text         — briefing prose.
            generated_by — e.g. "gemma3:4b (local)" or "rule-based fallback".
            latency_s    — wall-clock seconds for the AI call (0.0 for fallback).
    """
    overview = profile.get("overview", {})
    row_count = overview.get("row_count", 0)
    col_count = overview.get("column_count", 0)
    completeness = overview.get("completeness_score", 0.0)
    dup_rows = overview.get("duplicate_rows", 0)

    grade = scorecard.get("grade", "N/A")
    score = scorecard.get("score", 0.0)
    checks_failed = scorecard.get("checks_failed", 0)
    total_checks = scorecard.get("total_checks", 0)

    failed_checks = [
        c.get("detail", c.get("check", "unknown"))
        for c in scorecard.get("checks", [])
        if not c.get("passed", True)
    ][:3]

    pii_cols = [p.get("column", "") for p in profile.get("pii_columns", [])]

    fact_sheet = (
        f"DATASET: {dataset_name}\n"
        f"Rows: {row_count}  |  Columns: {col_count}\n"
        f"Completeness: {completeness}%\n"
        f"Duplicate rows: {dup_rows}\n"
        f"Quality grade: {grade} ({score}%)\n"
        f"Failed checks: {checks_failed}/{total_checks}\n"
        f"Top issues: {'; '.join(failed_checks) if failed_checks else 'None'}\n"
        f"PII columns: {', '.join(pii_cols) if pii_cols else 'None detected'}\n"
    )

    prompt = (
        "You are a data steward. Using ONLY the facts below — do not invent columns "
        "or numbers — write a ~150-word executive briefing of this dataset: what it "
        "contains, its main quality concerns, and a fit-for-use verdict.\n\n"
        f"{fact_sheet}"
    )

    t0 = time.monotonic()
    text, provider_used = router.generate(
        TaskType.BRIEFING, prompt, max_tokens=300
    )
    latency = time.monotonic() - t0

    log_ai_call(
        task=TaskType.BRIEFING,
        provider=provider_used,
        model=OLLAMA_MODEL if provider_used == "ollama" else provider_used,
        latency_s=latency,
        prompt_chars=len(prompt),
        redaction_count=0,
        success=text is not None,
    )

    if text is not None:
        generated_by = f"{OLLAMA_MODEL} (local)" if provider_used == "ollama" else f"{provider_used} (cloud)"
        return {"text": text, "generated_by": generated_by, "latency_s": round(latency, 2)}

    # --- Rule-based fallback ---
    issue_str = "; ".join(failed_checks) if failed_checks else "no major issues"
    pii_str = f"PII detected in: {', '.join(pii_cols)}." if pii_cols else "No PII columns detected."
    fallback_text = (
        f'Dataset "{dataset_name}" contains {row_count:,} records across {col_count} columns '
        f"with {completeness}% overall completeness. "
        f"{dup_rows} duplicate rows were found. "
        f"Quality grade: {grade} ({score}%) — {checks_failed} of {total_checks} checks failed. "
        f"Main issues: {issue_str}. "
        f"{pii_str} "
        f"{'Recommend review before production use.' if grade in ('C', 'F') else 'Dataset meets quality standards for use.'}"
    )
    return {"text": fallback_text, "generated_by": "rule-based fallback", "latency_s": 0.0}


# ---------------------------------------------------------------------------
# 2.  Batched column descriptions
# ---------------------------------------------------------------------------

def describe_columns(
    router: AIRouter,
    df: pd.DataFrame,
    profile: dict,
) -> dict[str, str]:
    """Generate a plain-English description for every column in *df*.

    Columns are batched into groups of ``_DESCRIPTION_BATCH_SIZE`` (7) to
    minimise round-trips.  The router always routes COLUMN_DESCRIPTIONS to
    local Ollama — sample values in the prompt must never reach a cloud API.
    Parsing is defensive: only lines whose key exactly matches a real column
    are accepted; the rest fall back to rule-based heuristics.

    Args:
        router:  AIRouter instance.
        df:      The DataFrame to describe.
        profile: Output of ProfilerAgent.run() (used for null % etc.).

    Returns:
        Dict mapping column name → description string.
    """
    columns = list(df.columns)
    col_profiles = profile.get("columns", {})
    descriptions: dict[str, str] = {}

    batches = [
        columns[i: i + _DESCRIPTION_BATCH_SIZE]
        for i in range(0, len(columns), _DESCRIPTION_BATCH_SIZE)
    ]

    for batch in batches:
        chunk_lines: list[str] = []
        for col in batch:
            dtype = str(df[col].dtype)
            samples = df[col].dropna().unique()[:3].tolist()
            null_pct = col_profiles.get(col, {}).get("null_percentage", 0.0)
            chunk_lines.append(
                f"- {col}  dtype={dtype}  null%={null_pct}  "
                f"samples={[str(s) for s in samples]}"
            )

        prompt = (
            "For each column below write a brief description (<=20 words).\n"
            "Output EXACTLY one line per column in this format:\n"
            "column_name: description\n\n"
            "Columns:\n"
            + "\n".join(chunk_lines)
        )

        t0 = time.monotonic()
        text, provider_used = router.generate(
            TaskType.COLUMN_DESCRIPTIONS, prompt, max_tokens=250
        )
        latency = time.monotonic() - t0

        log_ai_call(
            task=TaskType.COLUMN_DESCRIPTIONS,
            provider=provider_used,
            model=OLLAMA_MODEL if provider_used == "ollama" else provider_used,
            latency_s=latency,
            prompt_chars=len(prompt),
            redaction_count=0,
            success=text is not None,
        )

        valid_cols = set(batch)
        if text is not None:
            for line in text.splitlines():
                if ":" not in line:
                    continue
                key, _, desc = line.partition(":")
                key = key.strip()
                desc = desc.strip()
                # Only accept keys that exactly match a real column in this batch.
                if key in valid_cols and desc:
                    descriptions[key] = desc

        # Fallback for any column the AI didn't cover or gave a bad line for.
        for col in batch:
            if col not in descriptions:
                descriptions[col] = _rule_based_description(col, str(df[col].dtype))

    return descriptions


# ---------------------------------------------------------------------------
# 3.  Issue explanations
# ---------------------------------------------------------------------------

def explain_issues(
    router: Optional[AIRouter],
    quality_issues: list[dict],
    df_schema: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Explain the top quality issues in plain English with suggested fixes.

    Issues are sorted by inferred severity (completeness failures first,
    then null-rate, then outliers, etc.) and the top *top_n* are explained.
    Falls back to template explanations when the router is None or returns None.

    Args:
        router:         AIRouter or None (None → all fallback, no AI call).
        quality_issues: List of check dicts from QualityCheckerAgent (may
                        include passed=True items; they are filtered out).
        df_schema:      Column name/type info used as context in the prompt.
        top_n:          Maximum number of issues to explain.

    Returns:
        List of dicts with keys: issue, explanation, suggested_fix, generated_by.
    """
    # Filter to failed checks only.
    failed = [c for c in quality_issues if not c.get("passed", True)]

    # Sort by severity — completeness first, then null-rate, etc.
    failed.sort(key=lambda c: _check_severity_score(c.get("check", "")))

    top_issues = failed[:top_n]

    schema_summary = "\n".join(
        f"  {col.get('column_name', '?')} ({col.get('column_type', '?')})"
        for col in (df_schema or [])[:10]  # first 10 columns for context
    )

    results: list[dict] = []

    for issue in top_issues:
        check_name = issue.get("check", "unknown")
        detail = issue.get("detail", "")

        if router is None:
            results.append({
                "issue": issue,
                "explanation": _fallback_explanation(check_name, detail),
                "suggested_fix": _fallback_fix(check_name),
                "generated_by": "rule-based fallback",
            })
            continue

        prompt = (
            f"You are a data quality analyst reviewing a dataset.\n\n"
            f"Issue: {check_name}\n"
            f"Detail: {detail}\n"
            f"Schema (first 10 columns):\n{schema_summary}\n\n"
            f"In 2-3 sentences explain WHY this matters for data analysis.\n"
            f"Then provide ONE suggested fix as pandas code or SQL.\n\n"
            f"Format your response as:\n"
            f"Explanation: <2-3 sentences>\n"
            f"Fix: <one-liner pandas or SQL>"
        )

        t0 = time.monotonic()
        text, provider_used = router.generate(
            TaskType.ISSUE_EXPLANATION, prompt, max_tokens=150
        )
        latency = time.monotonic() - t0

        log_ai_call(
            task=TaskType.ISSUE_EXPLANATION,
            provider=provider_used,
            model=OLLAMA_MODEL if provider_used == "ollama" else provider_used,
            latency_s=latency,
            prompt_chars=len(prompt),
            redaction_count=0,
            success=text is not None,
        )

        if text is not None:
            explanation, fix = _parse_issue_response(text, check_name)
            generated_by = (
                f"{OLLAMA_MODEL} (local)" if provider_used == "ollama"
                else f"{provider_used} (cloud)"
            )
        else:
            explanation = _fallback_explanation(check_name, detail)
            fix = _fallback_fix(check_name)
            generated_by = "rule-based fallback"

        results.append({
            "issue": issue,
            "explanation": explanation,
            "suggested_fix": fix,
            "generated_by": generated_by,
        })

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_severity_score(check_name: str) -> int:
    """Return a severity rank (lower = more severe) for a check name."""
    for key, score in _CHECK_SEVERITY.items():
        if key in check_name:
            return score
    return 9  # least severe / unknown


def _rule_based_description(col_name: str, dtype: str) -> str:
    """Generate a description from column name keywords and pandas dtype.

    Args:
        col_name: Column name string.
        dtype:    Pandas dtype string (e.g. "int64", "object").

    Returns:
        Short description string.
    """
    col_lower = col_name.lower()

    if any(x in col_lower for x in ["_id", "id_", " id"]) or col_lower.endswith("id"):
        return "Unique identifier field"
    if any(x in col_lower for x in ["date", "time", "day", "month", "year", "dob"]):
        return "Date or time field"
    if any(x in col_lower for x in ["first", "last", "full_name", "fname", "lname"]):
        return "Person name field"
    if "name" in col_lower:
        return "Name or label field"
    if any(x in col_lower for x in ["email", "mail"]):
        return "Email address field"
    if any(x in col_lower for x in ["phone", "tel", "mobile"]):
        return "Phone number field"
    if any(x in col_lower for x in ["addr", "address", "street", "city", "zip", "postal"]):
        return "Geographic address field"
    if "state" in col_lower:
        return "US state abbreviation"
    if any(x in col_lower for x in ["amount", "cost", "price", "fee", "usd", "total", "revenue"]):
        return "Monetary amount or cost field"
    if any(x in col_lower for x in ["count", "qty", "quantity", "num_"]):
        return "Count or quantity field"
    if any(x in col_lower for x in ["rate", "pct", "percent", "ratio", "score"]):
        return "Rate, percentage, or score field"
    if any(x in col_lower for x in ["code", "type", "category", "status", "flag"]):
        return "Categorical code or classification"
    if any(x in col_lower for x in ["diagnosis", "condition", "disease"]):
        return "Medical diagnosis or condition"
    if any(x in col_lower for x in ["age"]):
        return "Age in years"
    if col_lower.startswith("_"):
        return "Pipeline metadata column"
    if "int" in dtype or "float" in dtype:
        return "Numeric measurement field"
    return "Data field — review column contents for context"


def _fallback_explanation(check_name: str, detail: str = "") -> str:
    """Return a template explanation for a quality issue type."""
    if "completeness" in check_name:
        return (
            "Low overall completeness means a significant proportion of data cells "
            "are missing, which can bias any analysis built on this dataset. "
            "Downstream models and aggregations may produce unreliable results."
        )
    if "duplicate" in check_name:
        return (
            "Duplicate rows artificially inflate counts and metrics, causing "
            "incorrect aggregations and skewed distributions. "
            "They often indicate data-ingestion or merge errors that should be investigated."
        )
    if "null_rate" in check_name:
        col = check_name.replace("null_rate_", "")
        return (
            f"Column '{col}' has a high proportion of missing values, "
            "which reduces the usable rows for any analysis involving it. "
            "Imputation or exclusion should be considered based on domain context."
        )
    if "no_negatives" in check_name:
        col = check_name.replace("no_negatives_", "")
        return (
            f"Negative values found in '{col}', which should only contain positive numbers. "
            "These likely represent data entry errors or sensor anomalies. "
            "Analyses such as averages and totals will be distorted."
        )
    if "outlier" in check_name:
        col = check_name.replace("extreme_outliers_", "")
        return (
            f"Extreme statistical outliers in '{col}' can distort means, "
            "standard deviations, and regression models significantly. "
            "Verify whether these are genuine observations or measurement errors."
        )
    if "type_consistency" in check_name:
        col = check_name.replace("type_consistency_", "")
        return (
            f"Column '{col}' contains a mix of numeric and non-numeric values, "
            "preventing correct numeric operations. "
            "This usually indicates inconsistent data entry or a merge of incompatible sources."
        )
    return (
        f"Quality issue detected: {detail or check_name}. "
        "Review the affected column or dataset to determine root cause and remediation."
    )


def _fallback_fix(check_name: str) -> str:
    """Return a template one-liner fix for a quality issue type."""
    if "completeness" in check_name:
        return "df.dropna(thresh=int(len(df.columns) * 0.5), inplace=True)"
    if "duplicate" in check_name:
        return "df = df.drop_duplicates().reset_index(drop=True)"
    if "null_rate" in check_name:
        col = check_name.replace("null_rate_", "")
        return f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
    if "no_negatives" in check_name:
        col = check_name.replace("no_negatives_", "")
        return f"df = df[df['{col}'] >= 0]"
    if "outlier" in check_name:
        col = check_name.replace("extreme_outliers_", "")
        return f"df = df[df['{col}'].between(df['{col}'].quantile(0.01), df['{col}'].quantile(0.99))]"
    if "type_consistency" in check_name:
        col = check_name.replace("type_consistency_", "")
        return f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
    return "# Review column manually and apply domain-specific cleaning"


def _parse_issue_response(text: str, check_name: str) -> tuple[str, str]:
    """Parse the LLM's explanation + fix response into two strings.

    Looks for 'Explanation:' and 'Fix:' labels; falls back gracefully if
    the model deviates from the requested format.

    Args:
        text:       Raw LLM response text.
        check_name: Check name used for fallback fix when parsing fails.

    Returns:
        Tuple of (explanation, suggested_fix).
    """
    explanation = ""
    fix = ""

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        lower = stripped.lower()
        if lower.startswith("explanation:"):
            explanation = stripped[len("explanation:"):].strip()
        elif lower.startswith("fix:"):
            fix = stripped[len("fix:"):].strip()
            # If the model put the code on the next line(s) after a fence opener,
            # collect those lines until the closing fence or end of text.
            if fix.startswith("```"):
                code_lines = []
                i += 1
                while i < len(lines):
                    l = lines[i]
                    if l.strip().startswith("```"):
                        break
                    code_lines.append(l)
                    i += 1
                fix = "\n".join(code_lines).strip()
        i += 1

    # Strip any remaining fence markers the model may have inlined.
    fix = re.sub(r"^```[^\n]*\n?", "", fix)
    fix = re.sub(r"\n?```\s*$", "", fix).strip()

    if not explanation:
        explanation = text.strip()
    if not fix:
        fix = _fallback_fix(check_name)

    return explanation, fix
