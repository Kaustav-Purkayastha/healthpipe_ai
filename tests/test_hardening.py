"""
tests/test_hardening.py — Offline tests for Step 18 hardening + docs.

Covers the preflight check functions (called directly with monkeypatched
conditions), the requirements files' formatting, and the presence + section
structure of the README and build log.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts import preflight

ROOT = Path(__file__).parent.parent


# ===========================================================================
# Preflight — structured check results
# ===========================================================================

class TestPreflightChecks:
    """Each check returns a structured Check; conditions map to ok/warn/fail."""

    def test_python_check_ok_on_supported_version(self) -> None:
        # The suite runs on a 3.11+ interpreter, so this is never a hard fail.
        result = preflight.check_python()
        assert result.status in (preflight.OK, preflight.WARN)
        assert result.status != preflight.FAIL

    def test_core_imports_present(self) -> None:
        # Core deps are installed to run the tests → not a FAIL.
        result = preflight.check_core_imports()
        assert result.status != preflight.FAIL

    def test_ollama_down_is_warn_not_fail(self, monkeypatch) -> None:
        """Ollama unreachable must degrade to a warning, never a hard failure."""
        monkeypatch.setattr(
            "core.providers.OllamaProvider.is_available", lambda self: False
        )
        result = preflight.check_ollama()
        assert result.status == preflight.WARN
        assert "fallback" in result.hint.lower()

    def test_missing_fixture_gives_fix_hint(self, monkeypatch, tmp_path) -> None:
        """A missing fixture must warn and hand back the make-fixtures command."""
        monkeypatch.setattr("core.config.SAMPLE_DIR", tmp_path)  # empty dir
        result = preflight.check_fixtures()
        assert result.status == preflight.WARN
        assert "make_fixtures.py" in result.hint

    def test_missing_demo_db_gives_fix_hint(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("core.config.SAMPLE_DIR", tmp_path)
        result = preflight.check_demo_db()
        assert result.status == preflight.WARN
        assert "make_demo_db.py" in result.hint

    def test_missing_gemini_key_is_warn_with_local_note(self, monkeypatch) -> None:
        monkeypatch.setattr("core.config.GEMINI_API_KEY", "")
        result = preflight.check_gemini_key()
        assert result.status == preflight.WARN
        assert "local-only" in result.hint.lower()

    def test_missing_mart_gives_build_hint(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("core.config.OUTPUTS_DIR", tmp_path)
        result = preflight.check_mart_artifacts()
        assert result.status == preflight.WARN
        assert "build_mart.py" in result.hint

    def test_run_all_returns_checks(self) -> None:
        results = preflight.run_all()
        assert len(results) == len(preflight._CHECKS)
        assert all(isinstance(c, preflight.Check) for c in results)
        assert all(c.status in (preflight.OK, preflight.WARN, preflight.FAIL)
                   for c in results)

    def test_exit_code_zero_when_no_failures(self, monkeypatch) -> None:
        monkeypatch.setattr(
            preflight, "run_all",
            lambda: [preflight.Check(preflight.OK, "x", "fine"),
                     preflight.Check(preflight.WARN, "y", "meh", "hint")],
        )
        assert preflight.main() == 0

    def test_exit_code_one_when_a_failure_exists(self, monkeypatch) -> None:
        monkeypatch.setattr(
            preflight, "run_all",
            lambda: [preflight.Check(preflight.OK, "x", "fine"),
                     preflight.Check(preflight.FAIL, "z", "broken", "fix it")],
        )
        assert preflight.main() == 1


# ===========================================================================
# Requirements files
# ===========================================================================

class TestRequirements:
    """Every requirements line is a comment, blank, or a pin-formatted spec."""

    _PIN_RE = re.compile(r"^[A-Za-z0-9_.\[\]\-]+\s*(==|>=|<=|~=|!=|<|>)\s*[0-9][\w.]*$")

    @pytest.mark.parametrize("fname", [
        "requirements-core.txt",
        "requirements-connectors.txt",
        "requirements-bigquery.txt",
    ])
    def test_lines_are_comment_or_pin(self, fname: str) -> None:
        path = ROOT / fname
        assert path.exists(), f"{fname} missing"
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            assert self._PIN_RE.match(line), f"Unpinned/odd line in {fname}: {raw!r}"

    @pytest.mark.parametrize("fname", [
        "requirements-core.txt",
        "requirements-connectors.txt",
        "requirements-bigquery.txt",
    ])
    def test_has_header_comment(self, fname: str) -> None:
        """Each tier must open with an explanatory header comment."""
        path = ROOT / fname
        first = path.read_text(encoding="utf-8").splitlines()[0].strip()
        assert first.startswith("#"), f"{fname} must start with a header comment"


# ===========================================================================
# Docs — README
# ===========================================================================

class TestDocs:
    """The README exists and carries the expected section structure."""

    def test_readme_exists_with_sections(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        required = [
            "The Problem",
            "Architecture",
            "The AI Design",
            "Feature Matrix",
            "The Reporting Mart",
            "Quick Start",
            "Development Process",
            "Roadmap",
            "License",
        ]
        for heading in required:
            assert heading in readme, f"README missing section: {heading}"

    def test_readme_architecture_is_plain_ascii(self) -> None:
        """The architecture diagram must not use box-drawing unicode."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        # Box-drawing range U+2500–U+257F breaks across renderers — forbid it.
        offenders = [ch for ch in readme if "─" <= ch <= "╿"]
        assert not offenders, f"README uses box-drawing chars: {set(offenders)}"

    def test_readme_privacy_invariant_present(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "never data rows" in readme.lower() or "never cross" in readme.lower()

    def test_license_is_apache(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        assert "Apache License" in license_text
