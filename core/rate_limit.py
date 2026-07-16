"""
core/rate_limit.py — Client-side rate limiter for cloud (Gemini) chat calls.

Protects the free-tier quota by capping cloud usage BEFORE a request is sent,
so the app proactively falls back to local gemma instead of provoking a
server-side HTTP 429.  Two independent limits:

  - Per-minute (RPM): a sliding 60-second window, kept in memory.
  - Per-day:          a running counter persisted to a small date-keyed JSON
                      file, so it survives app restarts and resets at midnight.

The limiter never raises — all file I/O is guarded and fails open (treats an
unreadable counter as zero), because the router's existing 429 fallback is the
ultimate backstop.  Only actual cloud calls are recorded; local calls are free.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from core.utils import get_logger, load_json, save_json

_log = get_logger(__name__)

_WINDOW_SECONDS: float = 60.0


class CloudRateLimiter:
    """Enforces per-minute and per-day caps on cloud LLM calls.

    Args:
        rpm_limit:    Maximum cloud requests allowed in any 60-second window.
        daily_limit:  Maximum cloud requests allowed per calendar day.
        usage_file:   Path to the JSON file persisting the daily counter.
        monotonic_fn: Injectable monotonic clock (seconds) for the minute window.
                      Defaults to time.monotonic; overridden in tests.
        date_fn:      Injectable "YYYY-MM-DD" provider for the day key.
                      Defaults to local date; overridden in tests.
    """

    def __init__(
        self,
        rpm_limit: int,
        daily_limit: int,
        usage_file: Path,
        monotonic_fn: Callable[[], float] = time.monotonic,
        date_fn: Callable[[], str] | None = None,
    ) -> None:
        self._rpm = int(rpm_limit)
        self._daily = int(daily_limit)
        self._usage_file = Path(usage_file)
        self._monotonic = monotonic_fn
        self._date_fn = date_fn or (lambda: datetime.now().strftime("%Y-%m-%d"))
        # Monotonic timestamps of recent cloud calls (for the sliding window).
        self._minute_calls: deque[float] = deque()

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _recent_minute_count(self) -> int:
        """Return the number of cloud calls within the last 60 seconds.

        Also prunes timestamps that have fallen out of the window.
        """
        now = self._monotonic()
        while self._minute_calls and (now - self._minute_calls[0]) >= _WINDOW_SECONDS:
            self._minute_calls.popleft()
        return len(self._minute_calls)

    def _read_daily(self) -> int:
        """Return today's persisted cloud-call count (0 if missing/stale/unreadable).

        A file whose stored date is not today counts as zero — that is how the
        daily budget resets at midnight without any scheduled job.
        """
        try:
            data = load_json(self._usage_file)
            if isinstance(data, dict) and data.get("date") == self._date_fn():
                return int(data.get("count", 0))
        except (FileNotFoundError, ValueError, OSError, TypeError):
            # Missing or corrupt file → fail open (treat as a fresh day).
            pass
        return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> tuple[bool, str]:
        """Return (allowed, reason) WITHOUT recording a call.

        Daily limit is checked first (it's the harder ceiling and gives the more
        useful "resets tomorrow" message).  reason is "" when allowed.
        """
        used_today = self._read_daily()
        if used_today >= self._daily:
            return (
                False,
                f"Daily cloud limit reached ({used_today}/{self._daily}) — "
                f"using local until tomorrow.",
            )
        if self._recent_minute_count() >= self._rpm:
            return (
                False,
                f"Cloud rate limit ({self._rpm}/min) reached — using local for a moment.",
            )
        return (True, "")

    def record(self) -> None:
        """Record one cloud call against both the minute window and daily count."""
        self._minute_calls.append(self._monotonic())
        today = self._date_fn()
        used_today = self._read_daily()  # 0 when the stored date is not today
        try:
            save_json({"date": today, "count": used_today + 1}, self._usage_file)
        except OSError as exc:  # non-fatal — the minute window still applies
            _log.warning("Rate limiter: failed to persist daily count: %s", exc)

    def used_today(self) -> int:
        """Return how many cloud calls have been made so far today."""
        return self._read_daily()

    def remaining_today(self) -> int:
        """Return how many cloud calls remain in today's budget (clamped ≥ 0)."""
        return max(0, self._daily - self._read_daily())

    def snapshot(self) -> dict:
        """Return a display-friendly view of current usage."""
        used = self._read_daily()
        return {
            "used_today": used,
            "daily_limit": self._daily,
            "remaining_today": max(0, self._daily - used),
            "per_minute_used": self._recent_minute_count(),
            "rpm_limit": self._rpm,
        }
