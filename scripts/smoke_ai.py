"""
scripts/smoke_ai.py — Provider health check and AI smoke test for Step 4.

Prints a provider availability table, then runs one generation per key task
type via the AIRouter.  Exits 0 regardless of provider availability — absence
of a provider is expected in offline environments.

Usage:
    python scripts/smoke_ai.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add the project root to sys.path so imports work when run directly.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import GEMINI_MODEL, OLLAMA_MODEL  # noqa: E402
from core.providers import GeminiProvider, OllamaProvider  # noqa: E402
from core.router import AIRouter, TaskType  # noqa: E402
from core.audit import log_ai_call  # noqa: E402


def _row(label: str, value: str) -> None:
    """Print one left-aligned table row."""
    print(f"  {label:<24} {value}")


def _smoke_generate(
    router: AIRouter,
    task: str,
    prompt: str,
    max_tokens: int = 40,
) -> None:
    """Run one generation, print result, write to audit log."""
    t0 = time.monotonic()
    text, provider_used = router.generate(task, prompt, max_tokens=max_tokens)
    latency = time.monotonic() - t0

    if provider_used == "none" or text is None:
        print(f"  [skipped] No provider available for task '{task}'")
        return

    _row("Provider:", provider_used)
    _row("Latency:", f"{latency:.2f}s")
    _row("Response:", repr(text[:120]))

    log_ai_call(
        task=task,
        provider=provider_used,
        model=provider_used,
        latency_s=latency,
        prompt_chars=len(prompt),
        redaction_count=0,
        success=True,
    )


def main() -> None:
    """Run the provider health table and smoke generations."""
    print("\nHealthPipe AI v2 — AI Provider Smoke Test")
    print("=" * 56)

    ollama = OllamaProvider()
    gemini = GeminiProvider()

    ollama_ok = ollama.is_available()
    gemini_ok = gemini.is_available()

    print("\nProvider Health:")
    _row("Ollama available:", str(ollama_ok))
    if ollama_ok:
        _row("Ollama model:", OLLAMA_MODEL)
    _row("Gemini available:", str(gemini_ok))
    if gemini_ok:
        _row("Gemini model:", GEMINI_MODEL)

    print()

    router = AIRouter()

    # --- BRIEFING: always routes to local Ollama ---
    print("── BRIEFING (local only) ─────────────────────────")
    _smoke_generate(
        router,
        TaskType.BRIEFING,
        "Summarise this dataset in one sentence: 19 patient records, mostly complete.",
    )
    print()

    # --- CHAT_SQL: prefers Gemini, falls back to Ollama ---
    print("── CHAT_SQL (gemini preferred, ollama fallback) ──")
    _smoke_generate(
        router,
        TaskType.CHAT_SQL,
        "Write a SQL query to count rows in a table named patients.",
        max_tokens=60,
    )
    print()

    print("Audit entries appended to: outputs/ai_audit.jsonl")
    print("=" * 56)
    print()


if __name__ == "__main__":
    main()
