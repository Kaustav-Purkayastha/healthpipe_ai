"""
tests/test_mart_planner.py — Offline tests for the AI mart generator (planner).

Covers the NL→spec planning (JSON parse, validation, measure filtering, heuristic
fallback), the local-only narration, and the router change that makes MART_PLAN
cloud-eligible alongside CHAT_SQL. All AI is mocked — no network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


def _catalog() -> pd.DataFrame:
    return pd.DataFrame([
        {"questionid": "DIA01", "question": "Diabetes among adults", "topic": "Diabetes"},
        {"questionid": "NPW14", "question": "Obesity among adults", "topic": "Nutrition"},
        {"questionid": "TOB04", "question": "Current cigarette smoking", "topic": "Tobacco"},
        {"questionid": "CAN09", "question": "Mammography use", "topic": "Cancer"},
    ])


# ===========================================================================
# Router: MART_PLAN is cloud-eligible; narration stays local
# ===========================================================================

class TestMartPlanRouting:
    def _router(self, gemini_ok=True, ollama_ok=True):
        from core.router import AIRouter
        r = AIRouter()
        r._gemini = MagicMock(); r._gemini.name = "gemini"
        r._gemini.is_available.return_value = gemini_ok
        r._gemini.generate.return_value = "{}"
        r._ollama = MagicMock(); r._ollama.name = "ollama"
        r._ollama.is_available.return_value = ollama_ok
        r._ollama.generate.return_value = "{}"
        r._cloud_limiter = MagicMock()
        r._cloud_limiter.check.return_value = (True, "")
        return r

    def test_mart_plan_prefers_gemini(self) -> None:
        from core.router import TaskType
        r = self._router()
        assert r.pick(TaskType.MART_PLAN) is r._gemini

    def test_mart_plan_falls_back_to_ollama(self) -> None:
        from core.router import TaskType
        r = self._router(gemini_ok=False)
        assert r.pick(TaskType.MART_PLAN) is r._ollama

    def test_mart_plan_records_cloud_call(self) -> None:
        from core.router import TaskType
        r = self._router()
        r.generate(TaskType.MART_PLAN, "plan")
        r._cloud_limiter.record.assert_called_once()

    def test_mart_plan_gemini_none_falls_back(self) -> None:
        from core.router import TaskType
        r = self._router()
        r._gemini.generate.return_value = None
        r._ollama.generate.return_value = "local json"
        text, provider = r.generate(TaskType.MART_PLAN, "plan")
        assert text == "local json"
        assert provider == "ollama"


# ===========================================================================
# plan_mart — JSON parse + validation
# ===========================================================================

class TestPlanMart:
    def _router_returning(self, text):
        r = MagicMock()
        r.generate.return_value = (text, "ollama")
        return r

    def test_parses_valid_json(self) -> None:
        from analytics.mart_planner import plan_mart
        raw = (
            '{"measures": ["NPW14", "TOB04"], "primary_measure": "TOB04", '
            '"narrative_focus": "payer", "title": "Obesity vs Smoking"}'
        )
        spec, meta = plan_mart(self._router_returning(raw), "obesity and smoking", _catalog(), ["DIA01"])
        assert spec.measures == ["NPW14", "TOB04"]
        assert spec.primary_measure == "TOB04"
        assert spec.narrative_focus == "payer"
        assert spec.source == "ai"
        assert meta["used_fallback"] is False

    def test_strips_markdown_fences(self) -> None:
        from analytics.mart_planner import plan_mart
        raw = '```json\n{"measures": ["DIA01"], "primary_measure": "DIA01"}\n```'
        spec, _ = plan_mart(self._router_returning(raw), "diabetes", _catalog(), ["DIA01"])
        assert spec.measures == ["DIA01"]

    def test_drops_hallucinated_measures(self) -> None:
        """Measure ids not in the catalog must be filtered out."""
        from analytics.mart_planner import plan_mart
        raw = '{"measures": ["DIA01", "ZZZ99"], "primary_measure": "ZZZ99"}'
        spec, _ = plan_mart(self._router_returning(raw), "x", _catalog(), ["DIA01"])
        assert spec.measures == ["DIA01"]
        # primary falls back to a valid measure when the requested one was invalid.
        assert spec.primary_measure == "DIA01"

    def test_invalid_focus_defaults_to_payer(self) -> None:
        from analytics.mart_planner import plan_mart
        raw = '{"measures": ["DIA01"], "primary_measure": "DIA01", "narrative_focus": "banana"}'
        spec, _ = plan_mart(self._router_returning(raw), "x", _catalog(), ["DIA01"])
        assert spec.narrative_focus == "payer"

    def test_bad_json_uses_heuristic(self) -> None:
        from analytics.mart_planner import plan_mart
        spec, meta = plan_mart(self._router_returning("not json at all"),
                               "obesity please", _catalog(), ["DIA01"])
        assert meta["used_fallback"] is True
        assert spec.source == "heuristic"
        assert "NPW14" in spec.measures  # matched 'obesity' → NPW14

    def test_no_router_uses_heuristic(self) -> None:
        from analytics.mart_planner import plan_mart
        spec, meta = plan_mart(None, "smoking trends", _catalog(), ["DIA01"])
        assert spec.source == "heuristic"
        assert "TOB04" in spec.measures  # 'smoking' → TOB04

    def test_heuristic_default_when_nothing_matches(self) -> None:
        from analytics.mart_planner import plan_mart
        spec, _ = plan_mart(None, "zzzzz qqqqq", _catalog(), ["DIA01"])
        assert spec.measures == ["DIA01"]  # falls back to the provided default


# ===========================================================================
# Prompt safety — PII scrub before any provider + injection delimiting
# ===========================================================================

class TestPlanPromptSafety:
    def test_scrubs_pii_before_provider(self) -> None:
        """A patient identifier in the request must be redacted before it reaches the model."""
        from analytics.mart_planner import plan_mart
        captured: dict = {}

        def fake_generate(task, prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"measures": ["DIA01"], "primary_measure": "DIA01"}', "gemini"

        router = MagicMock()
        router.generate.side_effect = fake_generate
        spec, meta = plan_mart(
            router, "diabetes burden for patient SSN 123-45-6789", _catalog(), ["DIA01"]
        )
        # The raw SSN must never appear in the prompt sent to the provider.
        assert "123-45-6789" not in captured["prompt"]
        assert "[REDACTED:ssn]" in captured["prompt"]
        # Redaction is counted (for the audit), never the value itself.
        assert meta["redaction_count"] == 1

    def test_pii_not_retained_in_heuristic_title(self) -> None:
        """With no router, the offline heuristic title must also be scrubbed."""
        from analytics.mart_planner import plan_mart
        spec, meta = plan_mart(None, "smoking 123-45-6789 trends", _catalog(), ["DIA01"])
        assert "123-45-6789" not in spec.title
        assert meta["redaction_count"] == 1

    def test_prompt_wraps_request_in_delimiters(self) -> None:
        """The user text is fenced in <request> tags with a data-not-instructions guard."""
        from analytics.mart_planner import _build_plan_prompt
        prompt = _build_plan_prompt("obesity and smoking", _catalog())
        assert "<request>" in prompt and "</request>" in prompt
        assert "obesity and smoking" in prompt
        assert "never" in prompt.lower() and "instructions" in prompt.lower()


# ===========================================================================
# narrate_report — deterministic facts + local narration
# ===========================================================================

class TestNarrateReport:
    def _mart(self) -> pd.DataFrame:
        return pd.DataFrame({
            "state_abbr": ["AL", "AK", "AZ", "AR"],
            "diabetes_prevalence_pct": [15.0, 8.0, 14.0, 7.0],
            "medicare_spend_per_capita": [100.0, 200.0, 90.0, 250.0],
        })

    def test_compute_report_facts_quadrant(self) -> None:
        from analytics.mart_planner import compute_report_facts
        facts = compute_report_facts(self._mart(), "diabetes_prevalence_pct")
        assert facts["quadrant_count"] == 2
        assert set(facts["quadrant_states"]) == {"AL", "AZ"}
        assert facts["top3_spend"][0]["state_abbr"] == "AR"

    def test_narrate_uses_template_without_router(self) -> None:
        from analytics.mart_planner import MartSpec, narrate_report
        spec = MartSpec(measures=["DIA01"], primary_measure="DIA01", title="T")
        out = narrate_report(None, self._mart(), spec, "diabetes_prevalence_pct", "Diabetes")
        assert out["generated_by"] == "rule-based fallback"
        assert len(out["text"]) > 20

    def test_narrate_calls_router_briefing_local(self) -> None:
        """narrate_report must route via BRIEFING (local), never a cloud task."""
        from analytics.mart_planner import MartSpec, narrate_report
        from core.router import TaskType
        router = MagicMock()
        router.generate.return_value = ("AI narrative.", "ollama")
        spec = MartSpec(measures=["DIA01"], primary_measure="DIA01", title="T")
        out = narrate_report(router, self._mart(), spec, "diabetes_prevalence_pct", "Diabetes")
        assert out["text"] == "AI narrative."
        task_arg = router.generate.call_args[0][0]
        assert task_arg == TaskType.BRIEFING  # LOCAL-only task, not MART_PLAN/CHAT_SQL
