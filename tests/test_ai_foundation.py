"""
tests/test_ai_foundation.py — Hermetic tests for Step 4 AI foundation.

ALL network calls are mocked via unittest.mock.patch("requests.get"/"post").
No real Ollama or Gemini connection is required or made.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests.exceptions

from core.providers import GeminiProvider, OllamaProvider
from core.router import AIRouter, TaskType
from core.privacy import PII_PATTERNS, cloud_safe_schema, scrub
import core.audit as audit_module


# ---------------------------------------------------------------------------
# Mock response helpers
# ---------------------------------------------------------------------------

def _mock_response(status: int = 200, body: Any = None) -> MagicMock:
    """Return a MagicMock that behaves like a requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body or {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# Canned Gemini response body — matches the documented API shape
_GEMINI_BODY = {
    "candidates": [
        {"content": {"parts": [{"text": "SELECT COUNT(*) FROM patients;"}]}}
    ]
}

# Canned Ollama /api/tags response (model is present)
_OLLAMA_TAGS_BODY = {"models": [{"name": "gemma3:4b"}]}

# Canned Ollama /api/generate response
_OLLAMA_GEN_BODY = {"response": "Hello, world!"}


# ===========================================================================
# OllamaProvider
# ===========================================================================

class TestOllamaProvider:
    """Tests for OllamaProvider.is_available() and .generate()."""

    def test_available_when_model_present(self) -> None:
        """200 response with matching model name → is_available() is True."""
        with patch("requests.get", return_value=_mock_response(200, _OLLAMA_TAGS_BODY)):
            assert OllamaProvider().is_available() is True

    def test_not_available_when_model_absent(self) -> None:
        """200 response but model not in list → is_available() is False."""
        body = {"models": [{"name": "llama3:8b"}]}
        with patch("requests.get", return_value=_mock_response(200, body)):
            assert OllamaProvider().is_available() is False

    def test_not_available_when_non_200(self) -> None:
        """Non-200 status → is_available() is False."""
        with patch("requests.get", return_value=_mock_response(503)):
            assert OllamaProvider().is_available() is False

    def test_not_available_on_connection_error(self) -> None:
        """ConnectionError → is_available() returns False, does not raise."""
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
            assert OllamaProvider().is_available() is False

    def test_generate_returns_stripped_text(self) -> None:
        """Successful POST → generate() returns the stripped response text."""
        with patch("requests.post", return_value=_mock_response(200, _OLLAMA_GEN_BODY)):
            result = OllamaProvider().generate("Say hi")
        assert result == "Hello, world!"

    def test_generate_returns_none_on_connection_error(self) -> None:
        """ConnectionError during generate() → returns None, does not raise."""
        with patch(
            "requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            assert OllamaProvider().generate("hi") is None

    def test_generate_returns_none_on_http_error(self) -> None:
        """HTTP error status → generate() returns None."""
        with patch("requests.post", return_value=_mock_response(500)):
            assert OllamaProvider().generate("hi") is None

    def test_generate_returns_none_on_missing_key(self) -> None:
        """Response body missing 'response' key → generate() returns None."""
        with patch("requests.post", return_value=_mock_response(200, {"text": "oops"})):
            assert OllamaProvider().generate("hi") is None


# ===========================================================================
# GeminiProvider
# ===========================================================================

class TestGeminiProvider:
    """Tests for GeminiProvider.is_available() and .generate()."""

    def test_not_available_when_key_empty(self) -> None:
        """Empty GEMINI_API_KEY → is_available() is False without any network call."""
        with patch("core.providers.GEMINI_API_KEY", ""):
            provider = GeminiProvider()
            with patch("requests.get") as mock_get:
                assert provider.is_available() is False
                mock_get.assert_not_called()

    def test_available_when_key_set_and_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-empty key + 200 response → is_available() is True."""
        monkeypatch.setattr("core.providers.GEMINI_API_KEY", "fake-key")
        provider = GeminiProvider()
        with patch("requests.get", return_value=_mock_response(200)):
            assert provider.is_available() is True

    def test_not_available_when_api_returns_403(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-empty key + 403 response → is_available() is False."""
        monkeypatch.setattr("core.providers.GEMINI_API_KEY", "bad-key")
        provider = GeminiProvider()
        with patch("requests.get", return_value=_mock_response(403)):
            assert provider.is_available() is False

    def test_availability_cached_after_first_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_available() called twice on same instance → requests.get called once."""
        monkeypatch.setattr("core.providers.GEMINI_API_KEY", "fake-key")
        provider = GeminiProvider()
        with patch("requests.get", return_value=_mock_response(200)) as mock_get:
            provider.is_available()
            provider.is_available()
        assert mock_get.call_count == 1, "Should only check availability once (cached)"

    def test_generate_parses_documented_response_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """generate() must parse candidates[0].content.parts[0].text."""
        monkeypatch.setattr("core.providers.GEMINI_API_KEY", "fake-key")
        provider = GeminiProvider()
        with patch("requests.post", return_value=_mock_response(200, _GEMINI_BODY)):
            result = provider.generate("count patients")
        assert result == "SELECT COUNT(*) FROM patients;"

    def test_generate_returns_none_on_429(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HTTP 429 (rate limit) → generate() returns None without raising."""
        monkeypatch.setattr("core.providers.GEMINI_API_KEY", "fake-key")
        provider = GeminiProvider()
        with patch("requests.post", return_value=_mock_response(429)):
            assert provider.generate("hi") is None

    def test_generate_returns_none_when_key_empty(self) -> None:
        """Empty key → generate() returns None immediately, no network call."""
        with patch("core.providers.GEMINI_API_KEY", ""):
            provider = GeminiProvider()
            with patch("requests.post") as mock_post:
                assert provider.generate("hi") is None
                mock_post.assert_not_called()

    def test_generate_returns_none_on_empty_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty candidates list → generate() returns None gracefully."""
        monkeypatch.setattr("core.providers.GEMINI_API_KEY", "fake-key")
        provider = GeminiProvider()
        with patch("requests.post", return_value=_mock_response(200, {"candidates": []})):
            assert provider.generate("hi") is None

    def test_generate_returns_none_on_connection_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Network error during generate() → returns None."""
        monkeypatch.setattr("core.providers.GEMINI_API_KEY", "fake-key")
        provider = GeminiProvider()
        with patch(
            "requests.post",
            side_effect=requests.exceptions.ConnectionError("timeout"),
        ):
            assert provider.generate("hi") is None


# ===========================================================================
# AIRouter
# ===========================================================================

def _make_router(
    ollama_available: bool = True,
    gemini_available: bool = True,
    ollama_response: str | None = "ollama answer",
    gemini_response: str | None = "gemini answer",
) -> AIRouter:
    """Build an AIRouter with injected mock providers."""
    router = AIRouter()

    router._ollama = MagicMock()
    router._ollama.name = "ollama"
    router._ollama.is_available.return_value = ollama_available
    router._ollama.generate.return_value = ollama_response

    router._gemini = MagicMock()
    router._gemini.name = "gemini"
    router._gemini.is_available.return_value = gemini_available
    router._gemini.generate.return_value = gemini_response

    # Permissive rate limiter by default — routing tests exercise provider
    # selection, not quota enforcement (those have dedicated tests).
    router._cloud_limiter = MagicMock()
    router._cloud_limiter.check.return_value = (True, "")
    router._cloud_limiter.remaining_today.return_value = 999

    return router


class TestAIRouter:
    """Tests for AIRouter.pick() and .generate()."""

    def test_chat_sql_picks_gemini_when_available(self) -> None:
        """CHAT_SQL + both available → pick() returns the Gemini provider."""
        router = _make_router(ollama_available=True, gemini_available=True)
        provider = router.pick(TaskType.CHAT_SQL)
        assert provider is router._gemini

    def test_chat_sql_falls_back_to_ollama_when_gemini_unavailable(self) -> None:
        """CHAT_SQL + Gemini unavailable → pick() returns Ollama."""
        router = _make_router(ollama_available=True, gemini_available=False)
        provider = router.pick(TaskType.CHAT_SQL)
        assert provider is router._ollama

    def test_chat_sql_returns_none_when_both_unavailable(self) -> None:
        """CHAT_SQL + both unavailable → pick() returns None."""
        router = _make_router(ollama_available=False, gemini_available=False)
        assert router.pick(TaskType.CHAT_SQL) is None

    def test_briefing_never_picks_gemini_even_when_available(self) -> None:
        """BRIEFING + both available → pick() returns Ollama (not Gemini)."""
        router = _make_router(ollama_available=True, gemini_available=True)
        provider = router.pick(TaskType.BRIEFING)
        assert provider is router._ollama
        assert provider is not router._gemini

    def test_narration_never_picks_gemini(self) -> None:
        """NARRATION + both available → pick() returns Ollama."""
        router = _make_router(ollama_available=True, gemini_available=True)
        assert router.pick(TaskType.NARRATION) is router._ollama

    def test_column_descriptions_never_picks_gemini(self) -> None:
        """COLUMN_DESCRIPTIONS + both available → pick() returns Ollama."""
        router = _make_router(ollama_available=True, gemini_available=True)
        assert router.pick(TaskType.COLUMN_DESCRIPTIONS) is router._ollama

    def test_issue_explanation_never_picks_gemini(self) -> None:
        """ISSUE_EXPLANATION + both available → pick() returns Ollama."""
        router = _make_router(ollama_available=True, gemini_available=True)
        assert router.pick(TaskType.ISSUE_EXPLANATION) is router._ollama

    def test_generate_chat_sql_returns_gemini_text_and_name(self) -> None:
        """generate() for CHAT_SQL with gemini available returns gemini result."""
        router = _make_router(gemini_response="SELECT 1")
        text, provider_name = router.generate(TaskType.CHAT_SQL, "count rows")
        assert text == "SELECT 1"
        assert provider_name == "gemini"

    def test_generate_chat_sql_fallback_when_gemini_returns_none(self) -> None:
        """CHAT_SQL: Gemini picked but returns None → falls back to Ollama text."""
        router = _make_router(
            ollama_available=True,
            gemini_available=True,
            ollama_response="SELECT COUNT(*) FROM t",
            gemini_response=None,  # Gemini fails
        )
        text, provider_name = router.generate(TaskType.CHAT_SQL, "count rows")
        assert text == "SELECT COUNT(*) FROM t"
        assert provider_name == "ollama"

    def test_generate_returns_none_string_when_no_provider(self) -> None:
        """No providers available → generate() returns (None, 'none')."""
        router = _make_router(ollama_available=False, gemini_available=False)
        text, provider_name = router.generate(TaskType.BRIEFING, "hello")
        assert text is None
        assert provider_name == "none"

    def test_generate_briefing_uses_ollama_and_returns_name(self) -> None:
        """generate() for BRIEFING returns Ollama result with 'ollama' name."""
        router = _make_router(ollama_response="Great dataset!")
        text, provider_name = router.generate(TaskType.BRIEFING, "describe data")
        assert text == "Great dataset!"
        assert provider_name == "ollama"


class TestAIRouterRateLimit:
    """AIRouter honours the cloud rate limiter for CHAT_SQL."""

    def test_chat_sql_falls_back_to_local_when_rate_limited(self) -> None:
        """Gemini available but limiter blocks → pick() returns Ollama."""
        router = _make_router(ollama_available=True, gemini_available=True)
        router._cloud_limiter.check.return_value = (False, "Daily cloud limit reached")
        assert router.pick(TaskType.CHAT_SQL) is router._ollama

    def test_generate_records_cloud_call_when_gemini_used(self) -> None:
        """A CHAT_SQL call routed to Gemini must be recorded against the limiter."""
        router = _make_router(gemini_response="SELECT 1")
        router.generate(TaskType.CHAT_SQL, "count rows")
        router._cloud_limiter.record.assert_called_once()

    def test_generate_local_call_not_recorded(self) -> None:
        """A local (BRIEFING) call must NOT touch the cloud limiter."""
        router = _make_router(ollama_response="hello")
        router.generate(TaskType.BRIEFING, "describe")
        router._cloud_limiter.record.assert_not_called()

    def test_cloud_availability_reports_limit_reason(self) -> None:
        """cloud_availability reflects the limiter's block reason + remaining."""
        router = _make_router(gemini_available=True)
        router._cloud_limiter.check.return_value = (False, "Daily cloud limit reached")
        router._cloud_limiter.remaining_today.return_value = 0
        usable, reason, remaining = router.cloud_availability()
        assert usable is False
        assert "Daily cloud limit" in reason
        assert remaining == 0

    def test_cloud_availability_not_configured(self) -> None:
        """When Gemini is unreachable, cloud_availability reports not configured."""
        router = _make_router(gemini_available=False)
        usable, reason, remaining = router.cloud_availability()
        assert usable is False
        assert reason == "cloud not configured"


# ===========================================================================
# Privacy — scrub() and cloud_safe_schema()
# ===========================================================================

class TestPrivacyScrub:
    """Tests for core.privacy.scrub()."""

    def test_scrub_email(self) -> None:
        """Email address is replaced with [REDACTED:email]."""
        text = "Contact alice@example.com for help"
        cleaned, redactions = scrub(text)
        assert "alice@example.com" not in cleaned
        assert "[REDACTED:email]" in cleaned
        assert any(r["kind"] == "email" for r in redactions)

    def test_scrub_phone_with_dashes(self) -> None:
        """US phone in NNN-NNN-NNNN format is replaced."""
        text = "call 555-867-5309 now"
        cleaned, redactions = scrub(text)
        assert "555-867-5309" not in cleaned
        assert "[REDACTED:phone]" in cleaned
        assert any(r["kind"] == "phone" for r in redactions)

    def test_scrub_phone_with_parens(self) -> None:
        """US phone in (NNN) NNN-NNNN format is replaced."""
        text = "reach us at (800) 555-1234"
        cleaned, redactions = scrub(text)
        assert "(800) 555-1234" not in cleaned
        assert "[REDACTED:phone]" in cleaned

    def test_scrub_ssn(self) -> None:
        """SSN in NNN-NN-NNNN format is replaced."""
        text = "SSN: 123-45-6789 on file"
        cleaned, redactions = scrub(text)
        assert "123-45-6789" not in cleaned
        assert "[REDACTED:ssn]" in cleaned
        assert any(r["kind"] == "ssn" for r in redactions)

    def test_scrub_npi_bare_10_digits(self) -> None:
        """Standalone 10-digit number (NPI-like) is replaced."""
        text = "NPI 1234567890 registered"
        cleaned, redactions = scrub(text)
        assert "1234567890" not in cleaned
        assert "[REDACTED:npi]" in cleaned
        assert any(r["kind"] == "npi" for r in redactions)

    def test_scrub_count_is_correct_for_multiple_matches(self) -> None:
        """Two email addresses produce count=2 in the redactions list."""
        text = "From alice@a.com to bob@b.org"
        cleaned, redactions = scrub(text)
        email_entry = next(r for r in redactions if r["kind"] == "email")
        assert email_entry["count"] == 2

    def test_scrub_matched_values_absent_from_redactions_list(self) -> None:
        """The actual matched values must NOT appear in the redactions output.

        WHY: The audit log stores redactions; storing matched values would
        turn the log into a PII repository.
        """
        original_email = "alice@example.com"
        text = f"email: {original_email}"
        cleaned, redactions = scrub(text)
        # Verify the original value is not in the cleaned text
        assert original_email not in cleaned
        # Verify the original value is not present anywhere in the redactions list
        redactions_str = json.dumps(redactions)
        assert original_email not in redactions_str
        # The redactions list should only contain kind and count
        for entry in redactions:
            assert set(entry.keys()) == {"kind", "count"}

    def test_scrub_no_pii_returns_empty_redactions(self) -> None:
        """Text with no PII → cleaned text unchanged, empty redactions list."""
        text = "The quick brown fox jumps over the lazy dog."
        cleaned, redactions = scrub(text)
        assert cleaned == text
        assert redactions == []

    def test_pii_patterns_dict_has_required_keys(self) -> None:
        """PII_PATTERNS must contain email, phone, ssn, and npi entries."""
        assert {"email", "phone", "ssn", "npi"}.issubset(PII_PATTERNS.keys())


class TestCloudSafeSchema:
    """Tests for core.privacy.cloud_safe_schema()."""

    def test_contains_column_names(self) -> None:
        """Output must include every column name from the schema."""
        schema = [
            {"column_name": "patient_id", "column_type": "VARCHAR"},
            {"column_name": "age", "column_type": "INTEGER"},
        ]
        result = cloud_safe_schema(schema)
        assert "patient_id" in result
        assert "age" in result

    def test_contains_column_types(self) -> None:
        """Output must include the column type."""
        schema = [{"column_name": "age", "column_type": "INTEGER"}]
        result = cloud_safe_schema(schema)
        assert "INTEGER" in result

    def test_does_not_contain_sample_values(self) -> None:
        """Output must NOT contain sample values even if present in schema dict."""
        schema = [
            {
                "column_name": "diagnosis",
                "column_type": "VARCHAR",
                "sample_values": ["Hypertension", "Diabetes"],  # must be excluded
            }
        ]
        result = cloud_safe_schema(schema)
        assert "Hypertension" not in result
        assert "Diabetes" not in result
        assert "diagnosis" in result

    def test_one_line_per_column(self) -> None:
        """Output has exactly one line per schema entry."""
        schema = [
            {"column_name": "a", "column_type": "INTEGER"},
            {"column_name": "b", "column_type": "VARCHAR"},
            {"column_name": "c", "column_type": "FLOAT"},
        ]
        lines = cloud_safe_schema(schema).strip().splitlines()
        assert len(lines) == 3

    def test_format_is_name_space_type_in_parens(self) -> None:
        """Each line must be in 'column_name (column_type)' format."""
        schema = [{"column_name": "age", "column_type": "INTEGER"}]
        result = cloud_safe_schema(schema)
        assert result.strip() == "age (INTEGER)"


# ===========================================================================
# Audit log
# ===========================================================================

class TestAudit:
    """Tests for core.audit.log_ai_call() and read_audit()."""

    def _write_call(self, audit_path: Path, **overrides: Any) -> None:
        """Helper: write one audit record to a temp file."""
        defaults = dict(
            task="briefing",
            provider="ollama",
            model="gemma3:4b",
            latency_s=0.5,
            prompt_chars=42,
            redaction_count=0,
            success=True,
        )
        defaults.update(overrides)
        # Temporarily redirect AUDIT_FILE to the tmp path
        original = audit_module.AUDIT_FILE
        audit_module.AUDIT_FILE = audit_path
        try:
            audit_module.log_ai_call(**defaults)
        finally:
            audit_module.AUDIT_FILE = original

    def test_log_ai_call_creates_file(self, tmp_path: Path) -> None:
        """log_ai_call() must create the JSONL file if it doesn't exist."""
        audit_file = tmp_path / "ai_audit.jsonl"
        self._write_call(audit_file)
        assert audit_file.exists()

    def test_log_ai_call_appends_valid_json_line(self, tmp_path: Path) -> None:
        """Each call must write exactly one parseable JSON line."""
        audit_file = tmp_path / "ai_audit.jsonl"
        self._write_call(audit_file)
        lines = [l for l in audit_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert isinstance(record, dict)

    def test_log_ai_call_record_has_required_fields(self, tmp_path: Path) -> None:
        """Audit record must contain all required keys."""
        audit_file = tmp_path / "ai_audit.jsonl"
        self._write_call(audit_file, task="chat_sql", provider="gemini")
        record = json.loads(audit_file.read_text().strip())
        required = {"ts", "task", "provider", "model", "latency_s",
                     "prompt_chars", "redaction_count", "success"}
        assert required.issubset(record.keys())

    def test_log_ai_call_does_not_store_prompt_text(self, tmp_path: Path) -> None:
        """The audit record must NOT store the prompt text — only its length."""
        audit_file = tmp_path / "ai_audit.jsonl"
        self._write_call(audit_file, prompt_chars=99)
        record = json.loads(audit_file.read_text().strip())
        assert record["prompt_chars"] == 99
        assert "prompt" not in record  # key 'prompt' must not exist

    def test_read_audit_returns_newest_first(self, tmp_path: Path) -> None:
        """read_audit() must return records newest-first."""
        audit_file = tmp_path / "ai_audit.jsonl"
        original = audit_module.AUDIT_FILE
        audit_module.AUDIT_FILE = audit_file
        try:
            audit_module.log_ai_call(
                task="briefing", provider="ollama", model="m",
                latency_s=0.1, prompt_chars=10, redaction_count=0, success=True,
            )
            time.sleep(0.01)  # ensure distinct timestamps
            audit_module.log_ai_call(
                task="chat_sql", provider="gemini", model="m",
                latency_s=0.2, prompt_chars=20, redaction_count=0, success=True,
            )
            records = audit_module.read_audit()
        finally:
            audit_module.AUDIT_FILE = original

        assert len(records) == 2
        # Newest record (chat_sql) must come first
        assert records[0]["task"] == "chat_sql"
        assert records[1]["task"] == "briefing"

    def test_read_audit_returns_empty_list_when_file_absent(
        self, tmp_path: Path
    ) -> None:
        """read_audit() on a non-existent file must return []."""
        original = audit_module.AUDIT_FILE
        audit_module.AUDIT_FILE = tmp_path / "nonexistent.jsonl"
        try:
            result = audit_module.read_audit()
        finally:
            audit_module.AUDIT_FILE = original
        assert result == []

    def test_log_ai_call_never_raises_on_bad_path(self) -> None:
        """log_ai_call() with an unwritable path must not raise."""
        original = audit_module.AUDIT_FILE
        # Use an impossible path (file named like a directory that doesn't exist)
        audit_module.AUDIT_FILE = Path("/nonexistent_root_dir/deep/audit.jsonl")
        try:
            # Should complete silently (logged as warning internally)
            audit_module.log_ai_call(
                task="test", provider="ollama", model="m",
                latency_s=0.0, prompt_chars=5, redaction_count=0, success=False,
            )
        finally:
            audit_module.AUDIT_FILE = original
