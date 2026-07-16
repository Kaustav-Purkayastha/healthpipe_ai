"""
core/audit.py — Append-only audit log for every AI call in HealthPipe AI v2.

Each call appends one JSON line to ``outputs/ai_audit.jsonl``.  The module
never raises — audit failure must not break the pipeline or the UI.

Recorded fields: timestamp, task, provider, model, latency, prompt_chars,
redaction_count, success.  Actual prompt text is NEVER stored.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from core.config import OUTPUTS_DIR
from core.utils import get_logger

_log = get_logger(__name__)

AUDIT_FILE: Path = OUTPUTS_DIR / "ai_audit.jsonl"


def log_ai_call(
    task: str,
    provider: str,
    model: str,
    latency_s: float,
    prompt_chars: int,
    redaction_count: int,
    success: bool,
) -> None:
    """Append one JSON audit record for an AI call.

    Deliberately never raises — a logging failure must not break the calling
    pipeline stage.  Failures are written to the application logger as warnings.

    Args:
        task:            TaskType constant (e.g. ``"chat_sql"``).
        provider:        Provider name (``"ollama"`` or ``"gemini"``).
        model:           Model identifier used (e.g. ``"gemma3:4b"``).
        latency_s:       Wall-clock seconds the generate() call took.
        prompt_chars:    Character length of the prompt (NOT the prompt text).
        redaction_count: Number of PII tokens replaced before sending.
        success:         True if the provider returned a non-None response.
    """
    record: dict = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "task": task,
        "provider": provider,
        "model": model,
        "latency_s": round(latency_s, 3),
        "prompt_chars": prompt_chars,
        "redaction_count": redaction_count,
        "success": success,
    }
    try:
        AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 — audit failure is non-fatal
        _log.warning("Audit log write failed (non-fatal): %s", exc)


def read_audit(limit: int = 100) -> list[dict]:
    """Return the most recent AI audit records, newest first.

    Args:
        limit: Maximum number of records to return (default 100).

    Returns:
        List of audit record dicts ordered newest-first.
        Returns an empty list when the file is absent or unreadable.
    """
    if not AUDIT_FILE.exists():
        return []

    try:
        lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
        records: list[dict] = []
        # Iterate lines in reverse (newest at the bottom of the JSONL file).
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
            if len(records) >= limit:
                break
        return records
    except Exception as exc:  # noqa: BLE001 — read failure is non-fatal
        _log.warning("Audit log read failed: %s", exc)
        return []
