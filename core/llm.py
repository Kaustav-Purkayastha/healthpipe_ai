"""
llm.py — Helper module for querying the local Ollama LLM.

Ollama runs a local HTTP server at http://localhost:11434 that accepts
POST requests with a prompt and returns generated text. This module
wraps that interaction with timeout handling and graceful fallback.

If Ollama is not running, every function returns None instead of crashing,
so the rest of the pipeline can fall back to rule-based logic.
"""

import requests

from core.utils import get_logger

logger = get_logger(__name__)

# Ollama's default local API endpoint
OLLAMA_URL: str = "http://localhost:11434/api/generate"

# Default model — gemma3:4b is a small, fast model suitable for
# generating short descriptions without needing a GPU
DEFAULT_MODEL: str = "gemma3:4b"


def query_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> str | None:
    """
    Send a prompt to the local Ollama server and return the response text.

    Ollama's /api/generate endpoint accepts a JSON body with "model",
    "prompt", and "stream" fields. Setting stream=False makes it return
    the full response in one shot instead of token-by-token.

    Args:
        prompt:  The text prompt to send to the model.
        model:   Which Ollama model to use (default: gemma3:4b).
        timeout: Max seconds to wait for a response. Ollama can be slow
                 on CPU-only machines, so 30s is a reasonable default.

    Returns:
        The model's response text as a string, or None if Ollama is
        unavailable or the request fails.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                # stream=False returns the complete response as one JSON object
                # instead of streaming tokens one at a time
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()

        result = response.json()
        # Ollama returns the generated text in the "response" field
        text = result.get("response", "").strip()

        if text:
            return text

        logger.warning("Ollama returned an empty response")
        return None

    except requests.ConnectionError:
        # Ollama server is not running — this is expected in CI or
        # on machines without Ollama installed
        logger.warning(
            "Ollama not reachable at localhost:11434 — "
            "falling back to rule-based logic"
        )
        return None
    except requests.Timeout:
        logger.warning(
            f"Ollama request timed out after {timeout}s — "
            f"falling back to rule-based logic"
        )
        return None
    except requests.HTTPError as e:
        logger.warning(f"Ollama HTTP error: {e}")
        return None
    except ValueError:
        # response.json() failed — invalid JSON from Ollama
        logger.warning("Ollama returned invalid JSON")
        return None


def is_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """
    Check if Ollama is running and the specified model is loaded.

    Makes a lightweight request to the /api/tags endpoint, which lists
    all available models without generating any text.

    Args:
        model: Model name to check for (default: gemma3:4b).

    Returns:
        True if Ollama is running and the model is available.
    """
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        # data["models"] is a list of dicts, each with a "name" field
        available = [m.get("name", "") for m in data.get("models", [])]

        if model in available:
            logger.info(f"Ollama available with model '{model}'")
            return True
        else:
            logger.warning(
                f"Ollama running but model '{model}' not found. "
                f"Available: {available}"
            )
            return False

    except (requests.ConnectionError, requests.Timeout):
        logger.warning("Ollama not reachable")
        return False
    except (requests.HTTPError, ValueError):
        return False
