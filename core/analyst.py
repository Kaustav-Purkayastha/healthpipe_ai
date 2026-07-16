"""
core/analyst.py — Guardrailed NL→SQL analyst engine for HealthPipe AI v2.

Pipeline:  question → PII scrub → SQL generation (routed) → validate →
           execute on DuckDB → narrate results (local only) → return.

Every guardrail here was driven by benchmark failures on this exact machine
(gemma3:4b): missed GROUP BY, ID-column filtering, extra prose in output.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import pandas as pd

from core.audit import log_ai_call
from core.config import OLLAMA_MODEL
from core.database import DuckDBManager
from core import privacy
from core.router import AIRouter, TaskType
from core.utils import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt guardrails — verbatim; each rule fixes a benchmarked failure mode
# ---------------------------------------------------------------------------

SQL_PROMPT_RULES: str = """\
RULES (follow exactly):
1. Output ONLY one valid DuckDB SQL SELECT query. No explanations, no markdown fences.
2. Prefer human-readable text columns (e.g. topic, question, location names) over ID/code columns ending in _id or _cd.
3. For topical/keyword questions use ILIKE '%keyword%' on text columns.
4. If the question says 'per X' or 'by X', you MUST GROUP BY X.
5. Limit results to 50 rows unless the question asks otherwise.
"""

# Mutation/DDL keywords that must never appear in analyst-submitted SQL.
_FORBIDDEN_KEYWORDS: list[str] = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "ATTACH", "COPY", "PRAGMA", "INSTALL", "LOAD", "EXPORT", "CALL",
]


# ---------------------------------------------------------------------------
# 1. Schema context builder
# ---------------------------------------------------------------------------

def build_schema_context(
    db: DuckDBManager,
    table: str,
    include_samples: bool,
) -> str:
    """Build a schema description string for inclusion in LLM prompts.

    Args:
        db:              Open DuckDBManager connection.
        table:           Table to describe.
        include_samples: When True (LOCAL prompts only), each line includes up
                         to 3 sample values truncated to 40 chars.
                         When False (CLOUD prompts), delegate to
                         ``privacy.cloud_safe_schema`` — names+types ONLY.
                         This is THE enforcement point for the privacy invariant.

    Returns:
        Multi-line schema string ready for prompt injection.
    """
    schema = db.get_schema(table)

    if not include_samples:
        # Cloud path: names and types ONLY — no actual data values ever reach cloud.
        return privacy.cloud_safe_schema(schema)

    # Local path: include sample values from the table.
    try:
        sample_df = db.query(f'SELECT * FROM "{table}" LIMIT 5')
    except Exception:
        # If the table query fails, fall back to schema-only to stay non-crashing.
        return privacy.cloud_safe_schema(schema)

    lines: list[str] = []
    for col_info in schema:
        col_name = col_info["column_name"]
        col_type = col_info["column_type"]

        if col_name in sample_df.columns:
            raw_samples = sample_df[col_name].dropna().unique()[:3].tolist()
            samples_str = ", ".join(str(v)[:40] for v in raw_samples)
            lines.append(f"{col_name} ({col_type}) — samples: {samples_str}")
        else:
            lines.append(f"{col_name} ({col_type})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. SQL cleaner
# ---------------------------------------------------------------------------

def clean_sql(raw: str) -> str:
    """Strip prose and markdown fences; extract the bare SQL statement.

    Handles any fence tag (```sql, ```duckdb, ```) because the benchmark saw
    ``duckdb`` fences in production output from gemma3:4b.

    Steps:
      1. Remove all markdown code fences (opening and closing, any language tag).
      2. Find the first SELECT or WITH keyword (case-insensitive).
      3. Cut at the first semicolon.
      4. Collapse all whitespace.

    Args:
        raw: Raw LLM output string.

    Returns:
        Clean single-line SQL string (may be empty if no SELECT/WITH found).
    """
    # Remove opening fences like ```sql, ```duckdb, ```
    text = re.sub(r"```[a-zA-Z]*\n?", "", raw)
    # Remove any remaining closing fences
    text = text.replace("```", "")

    # Find the first SELECT or WITH keyword
    match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if match:
        text = text[match.start():]

    # Cut at the first semicolon (avoid multiple statements)
    if ";" in text:
        text = text[: text.index(";")]

    # Collapse whitespace to a single line
    return " ".join(text.split()).strip()


# ---------------------------------------------------------------------------
# 3. SQL validator
# ---------------------------------------------------------------------------

def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate that *sql* is a safe, single SELECT or WITH statement.

    Checks (in order):
      - Non-empty after stripping.
      - Starts with SELECT or WITH (prevents injection via leading comments).
      - No forbidden mutation/DDL keywords as whole words (case-insensitive).
      - No semicolons (prevents statement chaining).

    Args:
        sql: The cleaned SQL string to validate.

    Returns:
        Tuple (is_valid: bool, reason: str).  reason is "" when valid.
    """
    stripped = sql.strip()

    if not stripped:
        return False, "SQL is empty after cleaning"

    upper = stripped.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, f"SQL must start with SELECT or WITH, got: {stripped[:40]!r}"

    for kw in _FORBIDDEN_KEYWORDS:
        # Whole-word match so 'drop_table' column name isn't rejected
        if re.search(rf"\b{kw}\b", stripped, re.IGNORECASE):
            return False, f"Forbidden keyword in SQL: {kw}"

    if ";" in stripped:
        return False, "Multiple statements detected (';' present)"

    return True, ""


# ---------------------------------------------------------------------------
# 4. Internal narration helper
# ---------------------------------------------------------------------------

def narrate(
    router: AIRouter,
    question: str,
    result_df: pd.DataFrame,
) -> Optional[str]:
    """Narrate query results in 1-2 sentences using the LOCAL model.

    The prompt contains real result rows (df.head(10)), so it MUST route local.
    The router guarantees this: NARRATION → Ollama only.
    Returns None gracefully if the local model is unavailable.

    Args:
        router:     AIRouter instance.
        question:   The original user question.
        result_df:  The DataFrame returned by the SQL query.

    Returns:
        Short narration string, or None on failure.
    """
    table_str = result_df.head(10).to_string(index=False)
    prompt = (
        f"Question: {question}\n\n"
        f"Query results:\n{table_str}\n\n"
        f"Answer in 1-2 sentences using only these results."
    )
    text, _ = router.generate(TaskType.NARRATION, prompt, max_tokens=100)
    return text


# ---------------------------------------------------------------------------
# 5. Main orchestrator
# ---------------------------------------------------------------------------

def ask(
    router: AIRouter,
    db: DuckDBManager,
    table: str,
    question: str,
    include_samples_local: bool = True,
    force_local: bool = False,
) -> dict:
    """Answer a natural-language question against a DuckDB table.

    Full pipeline:
        scrub question → pick provider → build schema context →
        generate SQL → validate → execute → [retry on error] →
        narrate → audit log → return result dict.

    Args:
        router:               AIRouter instance.
        db:                   Open DuckDBManager for query execution.
        table:                Target table name.
        question:             Natural-language question from the user.
        include_samples_local: Pass sample values to the LOCAL model prompt.
                              Always False for cloud; this flag only applies
                              when the local Ollama provider is selected.
        force_local:          When True, always use the Ollama provider
                              regardless of Gemini availability.
                              Powers the UI's "Local only" mode.

    Returns:
        Dict with keys: question, scrubbed, redactions, sql, valid, df,
        error, narration, provider_used, retries, latency_s.
    """
    t_start = time.monotonic()

    # a. PII scrub the question before it goes anywhere
    scrubbed_q, redactions = privacy.scrub(question)

    def _empty(error: str, sql: str = "", valid: bool = False, provider: str = "none") -> dict:
        return {
            "question": question,
            "scrubbed": bool(redactions),
            "redactions": redactions,
            "sql": sql,
            "valid": valid,
            "df": None,
            "error": error,
            "narration": None,
            "provider_used": provider,
            "retries": 0,
            "latency_s": round(time.monotonic() - t_start, 2),
        }

    # b. Route provider and build schema context.
    if force_local:
        # "Local only" UI toggle: always use Ollama regardless of Gemini availability.
        _local = router._ollama
        if not _local.is_available():
            return _empty("Local model (Ollama) is not available", provider="ollama")
        # Local model may see actual sample values in the prompt.
        use_samples = include_samples_local
    else:
        provider = router.pick(TaskType.CHAT_SQL)
        if provider is None:
            return _empty("No AI provider available for CHAT_SQL")
        # THE privacy invariant: cloud (Gemini) NEVER receives sample data values.
        # Only the local model (Ollama) may see actual cell contents in the prompt.
        use_samples = (
            False if provider.name == "gemini" else include_samples_local
        )

    schema_context = build_schema_context(db, table, include_samples=use_samples)

    # c. Build full prompt
    prompt = (
        f"{SQL_PROMPT_RULES}\n"
        f"Table: {table}\n"
        f"Schema:\n{schema_context}\n\n"
        f"Question: {scrubbed_q}\n"
        f"SQL:"
    )

    # Generate SQL — force_local bypasses router.generate() to stay on Ollama.
    if force_local:
        raw_text = _local.generate(prompt, max_tokens=200)
        provider_used = _local.name
    else:
        raw_text, provider_used = router.generate(
            TaskType.CHAT_SQL, prompt, max_tokens=200
        )

    if raw_text is None:
        return _empty("AI provider returned no response", provider=provider_used)

    # d. Clean and validate
    sql = clean_sql(raw_text)
    valid, reason = validate_sql(sql)
    if not valid:
        _log.warning("Generated SQL failed validation: %s | SQL: %s", reason, sql)
        log_ai_call(
            task=TaskType.CHAT_SQL,
            provider=provider_used,
            model=OLLAMA_MODEL if provider_used == "ollama" else provider_used,
            latency_s=round(time.monotonic() - t_start, 2),
            prompt_chars=len(prompt),
            redaction_count=len(redactions),
            success=False,
        )
        return {
            "question": question,
            "scrubbed": bool(redactions),
            "redactions": redactions,
            "sql": sql,
            "valid": False,
            "df": None,
            "error": f"SQL validation failed: {reason}",
            "narration": None,
            "provider_used": provider_used,
            "retries": 0,
            "latency_s": round(time.monotonic() - t_start, 2),
        }

    # e. Execute — one retry on DuckDB error
    retries = 0
    result_df: Optional[pd.DataFrame] = None
    error: Optional[str] = None

    try:
        result_df = db.query(sql)
    except Exception as exc:
        error_msg = str(exc)
        _log.warning("SQL execution failed (attempt 1): %s", error_msg)

        # ONE retry — send original question + failed SQL + error for correction.
        # Benchmark showed the model needs all three to correct course.
        retry_prompt = (
            f"{SQL_PROMPT_RULES}\n"
            f"Table: {table}\n"
            f"Schema:\n{schema_context}\n\n"
            f"The following SQL query failed:\n{sql}\n"
            f"Error: {error_msg}\n\n"
            f"Original question: {scrubbed_q}\n"
            f"Write a corrected SQL query.\n"
            f"SQL:"
        )

        if force_local:
            raw2 = _local.generate(retry_prompt, max_tokens=200)
            provider_used = _local.name
        else:
            raw2, provider_used = router.generate(
                TaskType.CHAT_SQL, retry_prompt, max_tokens=200
            )
        retries = 1

        if raw2 is None:
            error = f"Retry failed — provider returned no response. Original error: {error_msg}"
        else:
            sql2 = clean_sql(raw2)
            valid2, reason2 = validate_sql(sql2)
            if not valid2:
                error = f"Retry SQL invalid: {reason2}"
            else:
                try:
                    result_df = db.query(sql2)
                    sql = sql2
                    error = None
                except Exception as exc2:
                    error = f"Retry execution failed: {exc2}"

    # f. Narrate results (local only — router guarantees this)
    narration: Optional[str] = None
    if result_df is not None:
        narration = narrate(router, question, result_df)

    latency = round(time.monotonic() - t_start, 2)

    log_ai_call(
        task=TaskType.CHAT_SQL,
        provider=provider_used,
        model=OLLAMA_MODEL if provider_used == "ollama" else provider_used,
        latency_s=latency,
        prompt_chars=len(prompt),
        redaction_count=len(redactions),
        success=result_df is not None,
    )

    return {
        "question": question,
        "scrubbed": bool(redactions),
        "redactions": redactions,
        "sql": sql,
        "valid": True,
        "df": result_df,
        "error": error,
        "narration": narration,
        "provider_used": provider_used,
        "retries": retries,
        "latency_s": latency,
    }


# ---------------------------------------------------------------------------
# 6. Starter question generator
# ---------------------------------------------------------------------------

def _starter_templates(table: str, raw_schema: list[dict]) -> list[str]:
    """Build 5 schema-based template questions (instant, no AI).

    Args:
        table:      Target table name.
        raw_schema: List of {column_name, column_type} dicts.

    Returns:
        List of exactly 5 template question strings.
    """
    first_text_col = next(
        (c["column_name"] for c in raw_schema
         if any(t in c["column_type"].upper() for t in ["VARCHAR", "TEXT", "STRING", "CHAR"])),
        table,
    )
    first_numeric_col = next(
        (c["column_name"] for c in raw_schema
         if any(t in c["column_type"].upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT", "REAL"])),
        "value",
    )
    return [
        f"How many rows are in {table}?",
        f"What are the distinct values of {first_text_col}?",
        f"Which {first_text_col} has the most records?",
        f"What is the average {first_numeric_col} by {first_text_col}?",
        f"What is the maximum {first_numeric_col} in the dataset?",
    ]


def _build_starters(
    router: Optional[AIRouter],
    table: str,
    raw_schema: list[dict],
    schema_with_samples: str,
) -> list[str]:
    """Shared starter-question logic: local-AI generation with template fallback.

    Prompts the LOCAL model (NARRATION task) for questions, parses question-like
    lines, and pads with templates to exactly 5.  Returns templates directly when
    the router is None or returns no text.  This is the single source of truth for
    both the db-backed and DataFrame-backed entry points.

    Args:
        router:              AIRouter, or None (pure template fallback).
        table:               Target table name.
        raw_schema:          List of {column_name, column_type} dicts (templates).
        schema_with_samples: Schema string with sample values (LOCAL prompt only).

    Returns:
        List of exactly 5 question strings.
    """
    templates = _starter_templates(table, raw_schema)

    if router is None:
        return templates[:5]

    prompt = (
        f"Table: {table}\n"
        f"Schema:\n{schema_with_samples}\n\n"
        f"Write exactly 5 short analytical questions a data analyst would ask "
        f"about this table. One question per line. No numbering, no bullets."
    )

    text, _ = router.generate(TaskType.NARRATION, prompt, max_tokens=200)

    if text is None:
        return templates[:5]

    # Parse response — accept lines that look like questions.
    questions: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("0123456789.-) ").strip()
        if len(cleaned) > 10 and "?" in cleaned:
            questions.append(cleaned)

    # Pad with templates until we have exactly 5.
    for tmpl in templates:
        if len(questions) >= 5:
            break
        if tmpl not in questions:
            questions.append(tmpl)

    return questions[:5]


def starter_questions(
    router: Optional[AIRouter],
    db: DuckDBManager,
    table: str,
) -> list[str]:
    """Generate 5 analytical starter questions for a table from DuckDB.

    Used as the chat screen's INSTANT fallback (router=None → templates only).
    Onboarded tables are precomputed at pipeline time via
    ``starter_questions_from_df`` and served from the run artifact instead.

    Args:
        router: AIRouter, or None (pure template fallback).
        db:     Open DuckDBManager.
        table:  Target table name.

    Returns:
        List of exactly 5 question strings.
    """
    raw_schema = db.get_schema(table)
    schema_with_samples = build_schema_context(db, table, include_samples=True)
    return _build_starters(router, table, raw_schema, schema_with_samples)


def starter_questions_from_df(
    router: Optional[AIRouter],
    df: pd.DataFrame,
    table: str,
) -> list[str]:
    """Generate 5 starter questions from an in-memory DataFrame (pipeline path).

    Called during onboarding's AI-enrichment stage, when the local model is
    already warm — the "idle time" the demo waits through anyway.  The result is
    persisted as a run artifact so the chat screen loads it instantly with no
    spinner and no repeat cost.  Schema + sample values are built from *df*
    because the table is not yet loaded into DuckDB at enrichment time.

    Args:
        router: AIRouter, or None (pure template fallback).
        df:     The (clean) DataFrame to derive schema and samples from.
        table:  Target table name.

    Returns:
        List of exactly 5 question strings.
    """
    raw_schema = [
        {"column_name": str(col), "column_type": str(dtype)}
        for col, dtype in df.dtypes.items()
    ]
    # Build a samples string mirroring build_schema_context's LOCAL format.
    lines: list[str] = []
    for col in df.columns:
        samples = df[col].dropna().unique()[:3].tolist()
        samples_str = ", ".join(str(v)[:40] for v in samples)
        if samples_str:
            lines.append(f"{col} ({df[col].dtype}) — samples: {samples_str}")
        else:
            lines.append(f"{col} ({df[col].dtype})")
    schema_with_samples = "\n".join(lines)

    return _build_starters(router, table, raw_schema, schema_with_samples)
