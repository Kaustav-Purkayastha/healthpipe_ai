"""
analytics/measure_catalog.py — the catalog of CDI measures the mart can include.

Queries the CDC Chronic Disease Indicators API for the distinct
(questionid, question, topic) triples so the mart screen can offer a
topic-grouped picker.  Result is cached to parquet — the live query runs once,
then the demo (and offline runs) read the cache.

The three verified default measures (DIA01/NPW14/TOB04) keep FRIENDLY column
slugs (diabetes/obesity/smoking) for backward compatibility with the existing
mart charts and facts; any other measure uses its lowercased questionid.
"""

from __future__ import annotations

import pandas as pd
import requests

from core.config import CACHE_DIR, CDC_CDI_URL
from core.utils import get_logger

_log = get_logger(__name__)

_CACHE_NAME = "cdi_measure_catalog"

# The three live-verified defaults (order matters — used as the picker default).
DEFAULT_MEASURE_IDS: list[str] = ["DIA01", "NPW14", "TOB04"]

# Friendly column slugs for the defaults (backward compatibility). Every other
# measure derives its slug from the lowercased questionid.
_FRIENDLY_SLUGS: dict[str, str] = {
    "DIA01": "diabetes",
    "NPW14": "obesity",
    "TOB04": "smoking",
}


def measure_slug(question_id: str) -> str:
    """Return the mart column slug for a CDI questionid.

    Defaults keep their friendly name (DIA01 → 'diabetes'); everything else uses
    the lowercased questionid (e.g. 'AST01' → 'ast01'), giving columns like
    ``ast01_prevalence_pct`` / ``ast01_vintage``.

    Args:
        question_id: CDI questionid (e.g. "DIA01").

    Returns:
        Column-name slug string.
    """
    return _FRIENDLY_SLUGS.get(question_id, question_id.lower())


def _default_catalog() -> pd.DataFrame:
    """Offline fallback catalog — just the three verified defaults."""
    return pd.DataFrame([
        {"questionid": "DIA01", "question": "Diabetes among adults",
         "topic": "Diabetes"},
        {"questionid": "NPW14", "question": "Obesity among adults",
         "topic": "Nutrition, Physical Activity, and Weight Status"},
        {"questionid": "TOB04", "question": "Current cigarette smoking among adults",
         "topic": "Tobacco"},
    ])


def get_available_measures(refresh: bool = False) -> pd.DataFrame:
    """Return the catalog of CDI measures (questionid, question, topic).

    Reads the parquet cache unless *refresh* is True.  On a live pull it queries
    the CDI API with a $group so only distinct measures come back, then caches.
    Falls back to the cache (then to the three defaults) if the live call fails —
    so the picker always has something to show, even offline.

    Args:
        refresh: When True, ignore the cache and re-query the API.

    Returns:
        DataFrame with columns questionid, question, topic (topic-sorted).
    """
    cache_path = CACHE_DIR / f"{_CACHE_NAME}.parquet"

    if not refresh and cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:  # noqa: BLE001 — corrupt cache → fall through to live
            _log.warning("Measure-catalog cache unreadable; re-querying.")

    try:
        resp = requests.get(
            CDC_CDI_URL,
            params={
                "$select": "questionid,question,topic",
                "$group": "questionid,question,topic",
                # Only offer measures that publish Crude Prevalence — the value
                # type pull_cdi_measure() hardcodes. Rate/count-only indicators
                # (e.g. CAN02 breast cancer mortality, a per-100k rate) would
                # otherwise silently pull back zero rows if selected, since the
                # mart's *_prevalence_pct columns assume a prevalence measure.
                "$where": "datavaluetypeid='CRDPREV'",
                "$limit": 1000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        if not df.empty and {"questionid", "question", "topic"}.issubset(df.columns):
            df = (
                df[["questionid", "question", "topic"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["topic", "question"])
                .reset_index(drop=True)
            )
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)
            except Exception:  # noqa: BLE001 — caching is best-effort
                _log.warning("Could not cache measure catalog.")
            return df
    except Exception as exc:  # noqa: BLE001 — offline / API down → fallback
        _log.warning("Measure-catalog live query failed: %s", exc)

    # Fallbacks: stale cache, then the hardcoded defaults.
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:  # noqa: BLE001
            pass
    return _default_catalog()
