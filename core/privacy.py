"""
core/privacy.py — Deterministic, zero-AI PII scrubber for HealthPipe AI v2.

Two public helpers:
  scrub()              — replace PII matches with [REDACTED:<kind>] placeholders.
  cloud_safe_schema()  — render schema as "name (type)" lines; NO sample values.

Privacy invariant: actual data values must NEVER reach cloud APIs.
These functions are the enforcement gate for that invariant.
"""

from __future__ import annotations

import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# PII detection patterns (compiled once at module load for performance)
# ---------------------------------------------------------------------------

# NPI false-positive tradeoff: \b\d{10}\b matches ANY standalone 10-digit integer,
# not only valid National Provider Identifiers.  In healthcare data this is the
# correct tradeoff: false positives (random 10-digit numbers) are rare compared
# to the cost of sending a real NPI to a cloud API.

PII_PATTERNS: dict[str, re.Pattern] = {
    # Standard email address: user@domain.tld
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        re.IGNORECASE,
    ),
    # US phone number — requires at least one separator (space, dash, or dot)
    # between the area code and the exchange.  This deliberately excludes bare
    # 10-digit strings so those are caught by the NPI pattern instead.
    # Uses (?<!\d) instead of \b at the start because \b does not match before
    # a parenthesis '(' — a common area-code delimiter.
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[\s\-.])?(?:\(\d{3}\)|\d{3})[\s\-.]\d{3}[\s\-.]?\d{4}\b",
    ),
    # US Social Security Number: NNN-NN-NNNN
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # 10-digit standalone numeric ID (NPI-like).  \b on both sides ensures we
    # only match bare 10-digit integers, not substrings of longer numbers.
    "npi": re.compile(r"\b\d{10}\b"),
}


def scrub(text: str) -> tuple[str, list[dict]]:
    """Replace PII patterns in *text* with ``[REDACTED:<kind>]`` placeholders.

    WHY we return kind+count, not matched values: storing the matched values
    in the audit log would turn that log into a PII repository — the log must
    document THAT redaction occurred and HOW MUCH, but must never contain the
    sensitive values themselves.

    Args:
        text: Input string that may contain PII.

    Returns:
        Tuple of:
            cleaned_text — Input with every PII match replaced.
            redactions   — List of ``{"kind": str, "count": int}`` aggregates,
                           sorted by kind, omitting kinds with zero matches.
    """
    counts: defaultdict[str, int] = defaultdict(int)

    cleaned = text
    for kind, pattern in PII_PATTERNS.items():

        def _replace(match: re.Match, _kind: str = kind) -> str:
            """Closure that increments the counter and returns the placeholder."""
            counts[_kind] += 1
            return f"[REDACTED:{_kind}]"

        cleaned = pattern.sub(_replace, cleaned)

    redactions = [
        {"kind": k, "count": v}
        for k, v in sorted(counts.items())
        if v > 0
    ]
    return cleaned, redactions


def cloud_safe_schema(schema: list[dict]) -> str:
    """Render a table schema as ``column_name (type)`` lines only.

    This is the ONLY schema renderer permitted in cloud-bound prompts.
    It deliberately omits sample values, statistics, null counts, and any
    other field that could contain actual data content.

    Args:
        schema: List of dicts with ``column_name`` and ``column_type`` keys,
                as returned by ``DuckDBManager.get_schema()``.

    Returns:
        Multi-line string: one ``column_name (column_type)`` per line.
    """
    return "\n".join(
        f"{col.get('column_name', '?')} ({col.get('column_type', '?')})"
        for col in schema
    )
