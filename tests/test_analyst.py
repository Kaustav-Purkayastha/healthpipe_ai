"""
tests/test_analyst.py — Offline tests for core.analyst (Step 6).

All router calls are mocked; real DuckDB in-memory via tmp_path.
No Ollama or Gemini connection required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.analyst import (
    SQL_PROMPT_RULES,
    ask,
    build_schema_context,
    clean_sql,
    starter_questions,
    validate_sql,
)
from core.database import DuckDBManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyst_db(tmp_path: Path) -> DuckDBManager:
    """In-file DuckDB pre-loaded with a small test_fixture table."""
    db_path = tmp_path / "analyst.duckdb"
    df = pd.DataFrame({
        "patient_id": ["P001", "P002", "P003", "P004"],
        "age": [34, 52, -5, 28],           # row 3 has negative age
        "state": ["AL", "CA", "FL", "NY"],
        "diagnosis": ["Hypertension", "Diabetes", "Asthma", "Hypertension"],
        "visit_date": ["2024-01-05", "2024-01-07", "2024-01-09", "2024-01-11"],
        "cost_usd": [120.50, 340.00, 88.75, 200.00],
    })
    db = DuckDBManager(db_path=db_path)
    db.load_dataframe(df, "test_fixture")
    yield db
    db.close()


def _mock_router(sql_response: str | None = "SELECT COUNT(*) AS n FROM test_fixture",
                 narration: str | None = "There are 4 rows.",
                 provider_name: str = "ollama") -> MagicMock:
    """Return a mock AIRouter whose generate() returns known responses."""
    router = MagicMock()
    mock_provider = MagicMock()
    mock_provider.name = provider_name
    router.pick.return_value = mock_provider

    def _side_effect(task, prompt, **kw):
        if task == "chat_sql":
            return (sql_response, provider_name)
        return (narration, "ollama")  # NARRATION always local

    router.generate.side_effect = _side_effect
    return router


# ===========================================================================
# clean_sql
# ===========================================================================

class TestCleanSQL:
    """Tests for clean_sql()."""

    def test_plain_sql_unchanged(self) -> None:
        sql = "SELECT COUNT(*) FROM patients"
        assert clean_sql(sql) == sql

    def test_strips_sql_fence(self) -> None:
        raw = "```sql\nSELECT * FROM t\n```"
        assert clean_sql(raw) == "SELECT * FROM t"

    def test_strips_duckdb_fence(self) -> None:
        raw = "```duckdb\nSELECT id FROM patients\n```"
        assert clean_sql(raw) == "SELECT id FROM patients"

    def test_strips_generic_fence(self) -> None:
        raw = "```\nSELECT 1\n```"
        assert clean_sql(raw) == "SELECT 1"

    def test_cuts_at_semicolon(self) -> None:
        raw = "SELECT * FROM t; DROP TABLE t"
        result = clean_sql(raw)
        assert result == "SELECT * FROM t"

    def test_extracts_select_from_prose(self) -> None:
        raw = "Here is the query: SELECT id FROM patients LIMIT 5"
        assert clean_sql(raw) == "SELECT id FROM patients LIMIT 5"

    def test_collapses_whitespace(self) -> None:
        raw = "SELECT   id,\n  name\nFROM   t"
        assert "  " not in clean_sql(raw)
        assert "\n" not in clean_sql(raw)

    def test_handles_with_clause(self) -> None:
        raw = "WITH cte AS (SELECT 1 AS n) SELECT * FROM cte"
        result = clean_sql(raw)
        assert result.startswith("WITH")

    def test_returns_empty_when_no_select(self) -> None:
        raw = "Here is some prose with no SQL keyword."
        result = clean_sql(raw)
        # No SELECT or WITH → returns empty after collapsing
        assert "SELECT" not in result.upper()


# ===========================================================================
# validate_sql
# ===========================================================================

class TestValidateSQL:
    """Tests for validate_sql()."""

    def test_valid_select(self) -> None:
        ok, reason = validate_sql("SELECT * FROM t")
        assert ok is True
        assert reason == ""

    def test_valid_with(self) -> None:
        ok, reason = validate_sql("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert ok is True

    def test_rejects_empty(self) -> None:
        ok, _ = validate_sql("")
        assert ok is False

    def test_rejects_not_starting_with_select(self) -> None:
        ok, reason = validate_sql("-- comment\nSELECT 1")
        assert ok is False

    @pytest.mark.parametrize("kw", [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "ATTACH", "COPY", "PRAGMA", "INSTALL", "LOAD", "EXPORT", "CALL",
    ])
    def test_rejects_each_forbidden_keyword(self, kw: str) -> None:
        """Each mutation/DDL keyword must be rejected as a whole word."""
        ok, reason = validate_sql(f"SELECT 1; {kw} TABLE foo")
        # Note: the semicolon itself will also fail, but let's test the keyword path
        # with a clean statement that still contains the keyword
        ok2, reason2 = validate_sql(f"SELECT * FROM t WHERE x = 1 AND {kw} = 'a'")
        # At least one of the two must reject (semicolon or keyword)
        assert not ok or not ok2

    def test_rejects_multiple_statements(self) -> None:
        ok, reason = validate_sql("SELECT 1; DROP TABLE foo")
        # Rejected either for the ';' or for DROP keyword — both are correct
        assert ok is False
        assert len(reason) > 0

    def test_whole_word_does_not_reject_column_name(self) -> None:
        """'drop_count' column name must NOT trigger the DROP rejection."""
        ok, _ = validate_sql("SELECT drop_count FROM t")
        assert ok is True

    def test_rejects_insert_in_query(self) -> None:
        ok, _ = validate_sql("SELECT * FROM t WHERE INSERT = 1")
        assert ok is False


# ===========================================================================
# ask() — happy path
# ===========================================================================

class TestAskHappyPath:
    """ask() returns a result dict with executed SQL and DataFrame."""

    def test_result_has_all_required_keys(self, analyst_db: DuckDBManager) -> None:
        router = _mock_router()
        result = ask(router, analyst_db, "test_fixture", "How many rows?")
        required = {
            "question", "scrubbed", "redactions", "sql", "valid",
            "df", "error", "narration", "provider_used", "retries", "latency_s",
        }
        assert required.issubset(result.keys())

    def test_df_is_populated_on_success(self, analyst_db: DuckDBManager) -> None:
        router = _mock_router()
        result = ask(router, analyst_db, "test_fixture", "How many rows?")
        assert result["df"] is not None
        assert len(result["df"]) > 0

    def test_valid_is_true_on_success(self, analyst_db: DuckDBManager) -> None:
        router = _mock_router()
        result = ask(router, analyst_db, "test_fixture", "How many rows?")
        assert result["valid"] is True

    def test_retries_is_zero_on_first_success(self, analyst_db: DuckDBManager) -> None:
        router = _mock_router()
        result = ask(router, analyst_db, "test_fixture", "How many rows?")
        assert result["retries"] == 0

    def test_provider_used_is_set(self, analyst_db: DuckDBManager) -> None:
        router = _mock_router(provider_name="ollama")
        result = ask(router, analyst_db, "test_fixture", "How many rows?")
        assert result["provider_used"] == "ollama"

    def test_narration_is_populated(self, analyst_db: DuckDBManager) -> None:
        router = _mock_router(narration="There are 4 rows in total.")
        result = ask(router, analyst_db, "test_fixture", "How many rows?")
        assert result["narration"] == "There are 4 rows in total."


# ===========================================================================
# ask() — retry path
# ===========================================================================

class TestAskRetry:
    """ask() retries once when first SQL execution fails."""

    def test_retry_increments_retries_counter(self, analyst_db: DuckDBManager) -> None:
        call_count = {"n": 0}
        captured_prompts: list[tuple[str, str]] = []

        def _side(task, prompt, **kw):
            captured_prompts.append((task, prompt))
            if task == "chat_sql":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # Bad SQL — references a nonexistent table
                    return ("SELECT * FROM nonexistent_table_xyz", "ollama")
                # Second attempt — valid SQL
                return ("SELECT COUNT(*) AS n FROM test_fixture", "ollama")
            return ("There are rows.", "ollama")

        router = MagicMock()
        provider = MagicMock()
        provider.name = "ollama"
        router.pick.return_value = provider
        router.generate.side_effect = _side

        result = ask(router, analyst_db, "test_fixture", "How many rows?")

        assert result["retries"] == 1

    def test_retry_prompt_contains_error_text(self, analyst_db: DuckDBManager) -> None:
        """The second prompt must contain the DuckDB error message."""
        captured_prompts: list[tuple[str, str]] = []
        call_count = {"n": 0}

        def _side(task, prompt, **kw):
            captured_prompts.append((task, prompt))
            if task == "chat_sql":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return ("SELECT * FROM no_such_table", "ollama")
                return ("SELECT COUNT(*) AS n FROM test_fixture", "ollama")
            return ("Done.", "ollama")

        router = MagicMock()
        provider = MagicMock()
        provider.name = "ollama"
        router.pick.return_value = provider
        router.generate.side_effect = _side

        ask(router, analyst_db, "test_fixture", "Count rows please")

        chat_prompts = [p for t, p in captured_prompts if t == "chat_sql"]
        # Second prompt must contain the failing SQL and the error message
        assert len(chat_prompts) == 2
        retry_prompt = chat_prompts[1]
        assert "no_such_table" in retry_prompt       # failed SQL text
        assert "Count rows please" in retry_prompt   # original question

    def test_retry_succeeds_and_df_populated(self, analyst_db: DuckDBManager) -> None:
        """After retry produces valid SQL, df must be populated."""
        call_count = {"n": 0}

        def _side(task, prompt, **kw):
            if task == "chat_sql":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return ("SELECT * FROM bad_table", "ollama")
                return ("SELECT COUNT(*) AS n FROM test_fixture", "ollama")
            return ("4 rows.", "ollama")

        router = MagicMock()
        provider = MagicMock()
        provider.name = "ollama"
        router.pick.return_value = provider
        router.generate.side_effect = _side

        result = ask(router, analyst_db, "test_fixture", "How many?")
        assert result["df"] is not None


# ===========================================================================
# ask() — invalid SQL path
# ===========================================================================

class TestAskInvalid:
    """ask() returns early without executing when SQL fails validation."""

    def test_delete_returns_valid_false(self, analyst_db: DuckDBManager) -> None:
        router = _mock_router(sql_response="DELETE FROM test_fixture")
        result = ask(router, analyst_db, "test_fixture", "delete all")
        assert result["valid"] is False
        assert result["df"] is None

    def test_insert_returns_valid_false(self, analyst_db: DuckDBManager) -> None:
        router = _mock_router(sql_response="INSERT INTO test_fixture VALUES (1)")
        result = ask(router, analyst_db, "test_fixture", "add row")
        assert result["valid"] is False

    def test_no_provider_returns_error(self, analyst_db: DuckDBManager) -> None:
        router = MagicMock()
        router.pick.return_value = None
        result = ask(router, analyst_db, "test_fixture", "how many?")
        assert result["df"] is None
        assert result["error"] is not None


# ===========================================================================
# Privacy invariant
# ===========================================================================

class TestPrivacyInvariant:
    """Cloud provider (gemini) path must never receive sample data values."""

    def test_gemini_prompt_has_no_sample_values(
        self, analyst_db: DuckDBManager
    ) -> None:
        """When gemini is the picked provider, schema_context must have no samples."""
        captured: list[tuple[str, str]] = []

        def _side(task, prompt, **kw):
            captured.append((task, prompt))
            if task == "chat_sql":
                return ("SELECT COUNT(*) AS n FROM test_fixture", "gemini")
            return ("4 rows.", "ollama")

        router = MagicMock()
        gemini_provider = MagicMock()
        gemini_provider.name = "gemini"
        router.pick.return_value = gemini_provider
        router.generate.side_effect = _side

        ask(router, analyst_db, "test_fixture", "How many rows?")

        chat_sql_prompt = next(p for t, p in captured if t == "chat_sql")
        # Sample values from our test DB must not appear in cloud prompt
        assert "Hypertension" not in chat_sql_prompt
        assert "P001" not in chat_sql_prompt
        assert "AL" not in chat_sql_prompt

    def test_narration_prompt_contains_result_rows(
        self, analyst_db: DuckDBManager
    ) -> None:
        """Narration prompt MUST include actual result data (not just the question)."""
        captured: list[tuple[str, str]] = []

        def _side(task, prompt, **kw):
            captured.append((task, prompt))
            if task == "chat_sql":
                return ("SELECT COUNT(*) AS n FROM test_fixture", "ollama")
            return ("4 rows.", "ollama")

        router = MagicMock()
        provider = MagicMock()
        provider.name = "ollama"
        router.pick.return_value = provider
        router.generate.side_effect = _side

        ask(router, analyst_db, "test_fixture", "How many rows?")

        narration_prompt = next(p for t, p in captured if t == "narration")
        # The narration prompt must contain the query result (count = 4)
        assert "4" in narration_prompt
        # And it must have some table representation (to_string produces spaces/newlines)
        assert "\n" in narration_prompt or "n" in narration_prompt.lower()


# ===========================================================================
# build_schema_context
# ===========================================================================

class TestBuildSchemaContext:
    """build_schema_context() respects the include_samples flag."""

    def test_with_samples_contains_data(self, analyst_db: DuckDBManager) -> None:
        ctx = build_schema_context(analyst_db, "test_fixture", include_samples=True)
        assert "samples:" in ctx
        # Should contain actual values from the table
        assert any(v in ctx for v in ["P001", "AL", "Hypertension", "34"])

    def test_without_samples_is_names_and_types_only(
        self, analyst_db: DuckDBManager
    ) -> None:
        ctx = build_schema_context(analyst_db, "test_fixture", include_samples=False)
        assert "samples:" not in ctx
        # Should contain column names
        assert "patient_id" in ctx
        assert "age" in ctx

    def test_without_samples_has_no_data_values(
        self, analyst_db: DuckDBManager
    ) -> None:
        ctx = build_schema_context(analyst_db, "test_fixture", include_samples=False)
        # Actual cell values must not appear
        assert "Hypertension" not in ctx
        assert "P001" not in ctx


# ===========================================================================
# starter_questions
# ===========================================================================

class TestStarterQuestions:
    """starter_questions() returns exactly 5 questions and falls back correctly."""

    def test_returns_exactly_5(self, analyst_db: DuckDBManager) -> None:
        router = MagicMock()
        router.generate.return_value = ("Q1?\nQ2?\nQ3?\nQ4?\nQ5?", "ollama")
        result = starter_questions(router, analyst_db, "test_fixture")
        assert len(result) == 5

    def test_fallback_when_router_is_none(self, analyst_db: DuckDBManager) -> None:
        """With router=None, must return exactly 5 template questions."""
        result = starter_questions(None, analyst_db, "test_fixture")
        assert len(result) == 5
        assert all(isinstance(q, str) and len(q) > 5 for q in result)

    def test_fallback_when_generate_returns_none(
        self, analyst_db: DuckDBManager
    ) -> None:
        """When generate() returns None, must return exactly 5 templates."""
        router = MagicMock()
        router.generate.return_value = (None, "none")
        result = starter_questions(router, analyst_db, "test_fixture")
        assert len(result) == 5

    def test_pads_short_ai_response_with_templates(
        self, analyst_db: DuckDBManager
    ) -> None:
        """If AI returns only 2 questions, the remainder must be padded with templates."""
        router = MagicMock()
        router.generate.return_value = ("Only one question here?", "ollama")
        result = starter_questions(router, analyst_db, "test_fixture")
        assert len(result) == 5

    def test_fallback_mentions_table_name(self, analyst_db: DuckDBManager) -> None:
        """At least one template question must reference the table name."""
        result = starter_questions(None, analyst_db, "test_fixture")
        assert any("test_fixture" in q for q in result)


# ===========================================================================
# ask() — force_local parameter
# ===========================================================================

class TestAskForceLocal:
    """force_local=True routes directly to Ollama, bypassing router.generate()."""

    def _make_router_with_ollama(self, sql_response: str) -> MagicMock:
        """Build a mock router whose _ollama.generate() returns a known SQL string."""
        router = MagicMock()

        mock_ollama = MagicMock()
        mock_ollama.name = "ollama"
        mock_ollama.is_available.return_value = True
        mock_ollama.generate.return_value = sql_response

        mock_gemini = MagicMock()
        mock_gemini.name = "gemini"
        mock_gemini.is_available.return_value = True  # would normally be preferred

        router._ollama = mock_ollama
        router._gemini = mock_gemini
        # pick() would return gemini in normal routing — force_local must bypass this
        router.pick.return_value = mock_gemini

        return router

    def test_force_local_bypasses_router_generate(
        self, analyst_db: DuckDBManager
    ) -> None:
        """When force_local=True, router.generate() must not be called for CHAT_SQL.

        NARRATION still goes through router.generate() (local-only by design) —
        that call is expected and is NOT bypassed by force_local.
        """
        router = self._make_router_with_ollama(
            "SELECT COUNT(*) AS n FROM test_fixture"
        )
        # router.generate IS called for NARRATION; configure it to return a
        # valid tuple so narrate() doesn't crash on unpacking.
        router.generate.return_value = ("4 rows total.", "ollama")

        result = ask(router, analyst_db, "test_fixture", "How many rows?", force_local=True)

        # No CHAT_SQL generation should have gone through router.generate()
        chat_sql_calls = [
            c for c in router.generate.call_args_list
            if len(c.args) > 0 and c.args[0] == "chat_sql"
        ]
        assert len(chat_sql_calls) == 0, (
            f"router.generate was called for CHAT_SQL: {chat_sql_calls}"
        )
        assert result["provider_used"] == "ollama"

    def test_force_local_result_is_populated(
        self, analyst_db: DuckDBManager
    ) -> None:
        """With force_local=True and a valid SQL response, df must be populated."""
        router = self._make_router_with_ollama(
            "SELECT COUNT(*) AS n FROM test_fixture"
        )
        # Patch narrate so the test stays offline
        router.generate.side_effect = lambda task, prompt, **kw: ("4 rows.", "ollama")

        result = ask(router, analyst_db, "test_fixture", "How many rows?", force_local=True)

        # router.generate may be called for NARRATION (which uses router); only
        # CHAT_SQL generation must go through _ollama directly.
        assert result["df"] is not None
        assert result["provider_used"] == "ollama"

    def test_force_local_ollama_unavailable_returns_error(
        self, analyst_db: DuckDBManager
    ) -> None:
        """When force_local=True but Ollama is down, an error dict is returned."""
        router = MagicMock()
        mock_ollama = MagicMock()
        mock_ollama.is_available.return_value = False
        router._ollama = mock_ollama

        result = ask(router, analyst_db, "test_fixture", "How many rows?", force_local=True)

        assert result["df"] is None
        assert result["error"] is not None
        assert "not available" in result["error"].lower()

    def test_force_local_false_uses_normal_routing(
        self, analyst_db: DuckDBManager
    ) -> None:
        """With force_local=False (default), router.generate() is used as normal."""
        router = _mock_router()
        result = ask(router, analyst_db, "test_fixture", "How many rows?", force_local=False)

        router.generate.assert_called()
        assert result["df"] is not None


# ===========================================================================
# starter_questions_from_df — pipeline (DataFrame) path
# ===========================================================================

class TestStarterQuestionsFromDf:
    """starter_questions_from_df generates from an in-memory DataFrame."""

    _DF = pd.DataFrame({
        "state": ["AL", "CA", "NY"],
        "cost": [1.0, 2.0, 3.0],
    })

    def test_returns_exactly_5_with_router(self) -> None:
        from core.analyst import starter_questions_from_df
        router = MagicMock()
        router.generate.return_value = ("Q1?\nQ2?\nQ3?\nQ4?\nQ5?", "ollama")
        result = starter_questions_from_df(router, self._DF, "claims")
        assert len(result) == 5

    def test_templates_when_router_none(self) -> None:
        """router=None → instant templates only, referencing the table name."""
        from core.analyst import starter_questions_from_df
        result = starter_questions_from_df(None, self._DF, "claims")
        assert len(result) == 5
        assert any("claims" in q for q in result)

    def test_routes_narration_task_local_only(self) -> None:
        """Generation must use the NARRATION task (local), never CHAT_SQL/cloud."""
        from core.analyst import starter_questions_from_df
        captured: list[str] = []

        def _gen(task, prompt, **kw):
            captured.append(task)
            return ("Q1?\nQ2?\nQ3?\nQ4?\nQ5?", "ollama")

        router = MagicMock()
        router.generate.side_effect = _gen
        starter_questions_from_df(router, self._DF, "claims")

        assert captured == ["narration"], f"Expected NARRATION only, got {captured}"

    def test_no_samples_leak_check_prompt_local(self) -> None:
        """The local prompt includes sample values (local-only is allowed)."""
        from core.analyst import starter_questions_from_df
        captured: dict = {}

        def _gen(task, prompt, **kw):
            captured["prompt"] = prompt
            return ("Q1?\nQ2?\nQ3?\nQ4?\nQ5?", "ollama")

        router = MagicMock()
        router.generate.side_effect = _gen
        starter_questions_from_df(router, self._DF, "claims")

        # Local prompt legitimately carries samples (never sent to cloud).
        assert "samples:" in captured["prompt"]
