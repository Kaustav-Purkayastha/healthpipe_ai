"""
core/router.py — Task-based AI provider router for HealthPipe AI v2.

Enforces the privacy architecture: Gemini (cloud) is used ONLY for CHAT_SQL
because that task's prompt contains schema metadata only — never actual data
rows.  Every other task type sees actual data values and therefore routes to
local Ollama only.
"""

from __future__ import annotations

from typing import Optional

from core.config import GEMINI_DAILY_LIMIT, GEMINI_RPM_LIMIT, GEMINI_USAGE_FILE
from core.providers import GeminiProvider, LLMProvider, OllamaProvider
from core.rate_limit import CloudRateLimiter
from core.utils import get_logger

_log = get_logger(__name__)


class TaskType:
    """Task type constants for the AI router.

    These are plain string constants (not an enum) so callers can also pass
    raw strings without importing TaskType explicitly.
    """

    CHAT_SQL = "chat_sql"
    """NL→SQL generation — prompt contains schema metadata only, no data rows.
    May route to cloud (Gemini) when available."""

    MART_PLAN = "mart_plan"
    """NL→mart-spec planning — prompt contains the user's request + the measure
    CATALOG (questionid/question/topic) and fixed schema column names only, never
    data rows.  Cloud-safe by the same reasoning as CHAT_SQL, so it may route to
    Gemini when available (with local fallback)."""

    BRIEFING = "briefing"
    """Dataset summary narrative — sees actual profiling numbers.  Local only."""

    COLUMN_DESCRIPTIONS = "column_descriptions"
    """Per-column descriptions — sees sample values.  Local only."""

    ISSUE_EXPLANATION = "issue_explanation"
    """Quality issue explanations — sees actual check results.  Local only."""

    NARRATION = "narration"
    """Free-form data narrative — sees actual data values.  Local only."""


# Tasks whose prompts contain ONLY schema/catalog metadata + a user question —
# never actual data rows — so they may route to the cloud (Gemini) first with a
# local fallback.  Every other task sees real data values and is local-only.
_CLOUD_ELIGIBLE_TASKS: frozenset[str] = frozenset({TaskType.CHAT_SQL, TaskType.MART_PLAN})


class AIRouter:
    """Routes AI generation tasks to the appropriate LLM provider.

    Privacy invariant (non-negotiable):
        CHAT_SQL, MART_PLAN → Gemini preferred (prompt = schema/catalog metadata
                              + the user's question only; no data rows).
        ALL OTHER TASKS → Ollama only.

    WHY cloud is never used for the other tasks: BRIEFING, NARRATION,
    COLUMN_DESCRIPTIONS, and ISSUE_EXPLANATION all receive actual data values
    (sample values, profiling statistics, cell contents, aggregated mart facts)
    as prompt context.  Sending those to a cloud API would violate the privacy
    invariant stated in copilot-instructions.md — actual data rows must never
    leave the machine.  MART_PLAN is cloud-eligible because it only translates a
    request into a spec from the measure catalog; the narration that follows a
    plan sees aggregated facts and therefore uses BRIEFING (local).
    """

    def __init__(self) -> None:
        """Instantiate providers + the cloud rate limiter — no network calls yet."""
        self._ollama: LLMProvider = OllamaProvider()
        self._gemini: LLMProvider = GeminiProvider()
        # When True, even cloud-eligible tasks route to the local model — the
        # user's "on-device only" mode. Default is hybrid (cloud for
        # metadata-only tasks, local for everything touching data values).
        self.force_local: bool = False
        # Caps cloud usage (per-minute + per-day) so the app falls back to local
        # BEFORE hitting a server-side 429.  See core.rate_limit.
        self._cloud_limiter = CloudRateLimiter(
            GEMINI_RPM_LIMIT, GEMINI_DAILY_LIMIT, GEMINI_USAGE_FILE
        )

    def pick(self, task: str) -> Optional[LLMProvider]:
        """Select the best available provider for a given task.

        Args:
            task: A TaskType constant string.

        Returns:
            A ready LLMProvider instance, or None if no provider is available.
        """
        if task in _CLOUD_ELIGIBLE_TASKS and not self.force_local:
            # Cloud-eligible (CHAT_SQL, MART_PLAN): prefer cloud when available
            # AND within our rate limits.  When over the per-minute or daily cap
            # we deliberately choose local so we never provoke the free-tier 429.
            # (check() does not record.)
            if self._gemini.is_available() and self._cloud_limiter.check()[0]:
                return self._gemini
            if self._ollama.is_available():
                return self._ollama
            return None

        # ALL OTHER TASKS: local Ollama only.
        # Cloud providers MUST NOT be used here — these prompts may contain
        # actual data values which must never reach a cloud API.
        if self._ollama.is_available():
            return self._ollama
        return None

    def generate(
        self,
        task: str,
        prompt: str,
        **kwargs,
    ) -> tuple[Optional[str], str]:
        """Generate text for a task using the appropriate provider.

        For CHAT_SQL: if Gemini was selected but returns None (rate-limit or
        transient error), automatically falls back to Ollama and reports the
        provider that actually produced the result.

        Args:
            task:    A TaskType constant string.
            prompt:  The prompt to send to the provider.
            **kwargs: Forwarded to provider.generate() (max_tokens, temperature).

        Returns:
            Tuple of (text_or_None, provider_name_used).
            ``provider_name_used`` is ``"none"`` when no provider is available.
        """
        provider = self.pick(task)

        if provider is None:
            _log.warning("AIRouter: no provider available for task '%s'", task)
            return None, "none"

        # Count the cloud call against our rate limits BEFORE sending it, so a
        # burst is throttled even if some calls fail.  Local calls are free.
        if provider is self._gemini:
            self._cloud_limiter.record()

        result = provider.generate(prompt, **kwargs)

        # Cloud-eligible fallback: if Gemini was tried and returned None, try Ollama.
        if result is None and task in _CLOUD_ELIGIBLE_TASKS and provider is self._gemini:
            _log.info(
                "%s: Gemini returned None — falling back to Ollama", task
            )
            if self._ollama.is_available():
                result = self._ollama.generate(prompt, **kwargs)
                return result, self._ollama.name
            return None, "none"

        return result, provider.name

    def cloud_availability(self) -> tuple[bool, str, int]:
        """Report whether a cloud call would be made right now, for the UI badge.

        Combines provider reachability with the rate-limit state so the chat
        screen can show either the cloud badge (+ remaining daily budget) or the
        local badge (+ the specific reason cloud is paused).

        Returns:
            Tuple (cloud_usable_now, reason, remaining_today):
              - cloud_usable_now: True only when Gemini is reachable AND within
                the per-minute and daily caps.
              - reason: "" when usable; otherwise a human-readable explanation
                ("cloud not configured", or a rate/daily-limit message).
              - remaining_today: cloud calls left in today's budget.
        """
        if not self._gemini.is_available():
            return (False, "cloud not configured", 0)
        allowed, reason = self._cloud_limiter.check()
        return (allowed, reason, self._cloud_limiter.remaining_today())
