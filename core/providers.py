"""
core/providers.py — LLM provider abstractions for HealthPipe AI v2.

Two concrete providers:
  OllamaProvider — local Ollama REST API, default for all non-SQL tasks.
  GeminiProvider — Google Gemini REST API, used ONLY for NL→SQL chat.

Uses plain requests HTTP calls — no ollama SDK or google-generativeai package.
Returning None from generate() signals failure; callers must always have a
fallback path.  Neither provider ever raises.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import requests
import requests.exceptions

from core.config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)
from core.utils import get_logger

_log = get_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for all LLM provider integrations.

    Subclasses implement ``is_available()`` (lightweight reachability check)
    and ``generate()`` (text generation).  Both methods must be safe to call
    at any time: ``generate()`` returns None on failure, never raises.
    """

    name: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is reachable and ready to generate.

        Should be fast (timeout ≤ 5 s).  May be called repeatedly.
        """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Generate text from *prompt*.

        Args:
            prompt:      The input prompt string.
            max_tokens:  Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Generated text string, or None on any failure.  Never raises.
        """


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider — uses the Ollama REST API over localhost.

    Appropriate for all tasks that involve actual data values (profiling,
    narration, column descriptions) because nothing leaves the machine.
    """

    name: str = "ollama"

    def is_available(self) -> bool:
        """Return True if Ollama is running AND OLLAMA_MODEL is loaded.

        Args: none.
        Returns: bool.
        """
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False
            # Each element has a "name" field like "gemma3:4b"
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(OLLAMA_MODEL in m for m in models)
        except Exception:
            # Any network error → not available
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Generate text using the local Ollama instance.

        Args:
            prompt:      Input prompt.
            max_tokens:  Passed as num_predict to Ollama options.
            temperature: Sampling temperature.

        Returns:
            Stripped response string, or None on any error.
        """
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            KeyError,
        ) as exc:
            _log.warning("OllamaProvider.generate failed: %s", exc)
            return None


class GeminiProvider(LLMProvider):
    """Google Gemini REST API provider — used ONLY for NL→SQL chat.

    Reads GEMINI_API_KEY and GEMINI_MODEL from core.config (loaded from .env).
    Availability is checked once and cached for the process lifetime.
    """

    name: str = "gemini"

    def __init__(self) -> None:
        """Initialise with an unchecked availability cache."""
        # Cache: None = not yet checked, True/False = cached result.
        # WHY: is_available() makes a real network call.  Calling it per-request
        # would add latency to every AI call; one check per app run is sufficient.
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Return True if GEMINI_API_KEY is set and the API responds with 200.

        Result is cached for the process lifetime after the first call.

        Returns:
            bool — True if the Gemini API is accessible with the configured key.
        """
        if self._available is not None:
            return self._available

        if not GEMINI_API_KEY:
            self._available = False
            return False

        try:
            resp = requests.get(
                f"{GEMINI_BASE_URL}/models",
                params={"key": GEMINI_API_KEY},
                timeout=5,
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False

        return self._available

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """Generate text via the Gemini REST API.

        Args:
            prompt:      Input prompt (must contain schema only — no data rows).
            max_tokens:  maxOutputTokens for generationConfig.
            temperature: Sampling temperature.

        Returns:
            Generated text string, or None on HTTP 429, network error,
            missing key, or unexpected response shape.
        """
        if not GEMINI_API_KEY:
            return None

        url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
        try:
            resp = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                    },
                },
                timeout=30,
            )

            if resp.status_code == 429:
                # Free-tier rate limit hit.  Return None so the AIRouter can
                # fall back to the local Ollama provider transparently.
                _log.warning(
                    "GeminiProvider: HTTP 429 rate limit — router will fall back to local"
                )
                return None

            resp.raise_for_status()
            data = resp.json()

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
        ) as exc:
            _log.warning("GeminiProvider.generate network error: %s", exc)
            return None
        except ValueError as exc:
            # JSON decode failure
            _log.warning("GeminiProvider.generate JSON parse error: %s", exc)
            return None

        # Navigate the documented response structure safely.
        # Any missing key or empty list → return None rather than crash.
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None
            text = parts[0].get("text", "").strip()
            return text or None
        except (KeyError, IndexError, TypeError) as exc:
            _log.warning("GeminiProvider: unexpected response shape — %s", exc)
            return None
