"""
tests/test_rate_limit.py — Unit tests for core.rate_limit.CloudRateLimiter.

Clocks are injected (monotonic_fn, date_fn) so the sliding minute-window and the
daily reset are deterministic without real waiting or the real calendar.  The
daily counter is persisted to a tmp file per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.rate_limit import CloudRateLimiter


class _Clock:
    """Mutable fake monotonic clock (seconds)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _limiter(tmp_path: Path, rpm: int = 5, daily: int = 200,
             clock: _Clock | None = None, day: str = "2026-07-11") -> tuple:
    """Build a limiter with injected clocks and a tmp usage file."""
    clk = clock or _Clock()
    day_box = {"d": day}
    lim = CloudRateLimiter(
        rpm_limit=rpm,
        daily_limit=daily,
        usage_file=tmp_path / "usage.json",
        monotonic_fn=clk,
        date_fn=lambda: day_box["d"],
    )
    return lim, clk, day_box


# ===========================================================================
# Per-minute window
# ===========================================================================

class TestPerMinute:
    """The sliding 60-second window caps requests-per-minute."""

    def test_allows_up_to_rpm(self, tmp_path: Path) -> None:
        lim, _clk, _ = _limiter(tmp_path, rpm=5, daily=1000)
        for _ in range(5):
            assert lim.check()[0] is True
            lim.record()
        # 6th within the same minute is blocked with the rate-limit reason.
        allowed, reason = lim.check()
        assert allowed is False
        assert "rate limit" in reason.lower()

    def test_window_slides_after_60s(self, tmp_path: Path) -> None:
        clk = _Clock()
        lim, _clk, _ = _limiter(tmp_path, rpm=2, daily=1000, clock=clk)
        lim.record()
        lim.record()
        assert lim.check()[0] is False  # 2/min hit
        clk.advance(61)                 # window slides past both calls
        assert lim.check()[0] is True

    def test_daily_checked_before_minute(self, tmp_path: Path) -> None:
        """When both are exceeded, the daily reason wins (harder ceiling)."""
        lim, _clk, _ = _limiter(tmp_path, rpm=1, daily=1)
        lim.record()  # uses the single daily + minute slot
        allowed, reason = lim.check()
        assert allowed is False
        assert "daily" in reason.lower()


# ===========================================================================
# Per-day counter (persisted, date-keyed)
# ===========================================================================

class TestPerDay:
    """The persisted daily counter caps requests-per-day and resets by date."""

    def test_blocks_at_daily_limit(self, tmp_path: Path) -> None:
        lim, _clk, _ = _limiter(tmp_path, rpm=1000, daily=3)
        for _ in range(3):
            lim.record()
        allowed, reason = lim.check()
        assert allowed is False
        assert "3/3" in reason

    def test_remaining_today_math(self, tmp_path: Path) -> None:
        lim, _clk, _ = _limiter(tmp_path, rpm=1000, daily=10)
        lim.record()
        lim.record()
        assert lim.remaining_today() == 8

    def test_resets_on_new_day(self, tmp_path: Path) -> None:
        clk = _Clock()
        lim, _clk, day_box = _limiter(tmp_path, rpm=1000, daily=2, clock=clk)
        lim.record()
        lim.record()
        assert lim.check()[0] is False        # day 1 exhausted
        day_box["d"] = "2026-07-12"           # calendar rolls over
        assert lim.check()[0] is True          # fresh budget
        assert lim.remaining_today() == 2

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        """A second limiter over the same file sees the prior day's count."""
        lim1, _c1, _ = _limiter(tmp_path, daily=10)
        lim1.record()
        lim1.record()
        lim2, _c2, _ = _limiter(tmp_path, daily=10)
        assert lim2.used_today() == 2


# ===========================================================================
# Robustness — never raises, fails open
# ===========================================================================

class TestRobustness:
    """Missing/corrupt usage file must not raise and must fail open."""

    def test_missing_file_treated_as_fresh(self, tmp_path: Path) -> None:
        lim, _clk, _ = _limiter(tmp_path, daily=5)
        # No file written yet.
        assert lim.used_today() == 0
        assert lim.check()[0] is True

    def test_corrupt_file_fails_open(self, tmp_path: Path) -> None:
        usage = tmp_path / "usage.json"
        usage.write_text("{not valid json", encoding="utf-8")
        lim = CloudRateLimiter(5, 5, usage, date_fn=lambda: "2026-07-11")
        # Corrupt content → treated as zero, still allowed.
        assert lim.used_today() == 0
        assert lim.check()[0] is True

    def test_snapshot_shape(self, tmp_path: Path) -> None:
        lim, _clk, _ = _limiter(tmp_path, rpm=5, daily=200)
        lim.record()
        snap = lim.snapshot()
        assert snap["used_today"] == 1
        assert snap["daily_limit"] == 200
        assert snap["remaining_today"] == 199
        assert snap["rpm_limit"] == 5
        assert snap["per_minute_used"] == 1
