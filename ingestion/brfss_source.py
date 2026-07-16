"""
ingestion/brfss_source.py — BRFSS (Behavioral Risk Factor Surveillance System) source.

LIVE-VERIFIED FACTS (do not change):
- Endpoint: data.cdc.gov/resource/dttw-5yxu.json (CDC_BRFSS_URL in config)
- JSON field names are LOWERCASE: year, locationabbr, topic, response, data_value, etc.
- Filter on topic STRING — never on topicid alone.
  VERIFIED TRAP: topicid casing differs across vintages ('Topic09' pre-2015 vs
  'TOPIC09' after), making topicid an unreliable filter key.
- BRFSS data contains both 'US' (national aggregate) AND 'UW' (national median
  across states). Both must be excluded — they are not state-level totals.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from core.config import CDC_BRFSS_URL, EXCLUDED_LOCATION_CODES, SOCRATA_PAGE_SIZE
from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)


class BRFSSSource(BaseSource):
    """BRFSS chronic-condition prevalence data via the CDC Socrata API.

    Verified filter combinations (class constants):
        OBESITY  = {"topic": "BMI Categories",
                     "response": "Obese (BMI 30.0 - 99.8)"}
        SMOKING  = {"topic": "Current Smoker Status",
                     "response": "Yes"}

    Usage:
        src = BRFSSSource()
        if src.connect():
            df = src.extract(**BRFSSSource.OBESITY, year="2023")
    """

    source_type: str = "api"

    # Verified live filter combos
    OBESITY: dict[str, str] = {
        "topic": "BMI Categories",
        "response": "Obese (BMI 30.0 - 99.8)",
    }
    SMOKING: dict[str, str] = {
        "topic": "Current Smoker Status",
        "response": "Yes",
    }

    def __init__(self) -> None:
        """Initialise with registry name and description."""
        super().__init__(
            name="cdc_brfss",
            description="CDC BRFSS Prevalence Data 2011+ (Socrata)",
        )

    def connect(self) -> bool:
        """Return True if the BRFSS endpoint responds with 200 for a 1-row probe.

        Returns:
            bool
        """
        try:
            resp = requests.get(CDC_BRFSS_URL, params={"$limit": 1}, timeout=10)
            resp.raise_for_status()
            _log.info("CDC BRFSS API connected")
            return True
        except Exception as exc:  # noqa: BLE001
            _log.error("CDC BRFSS connect failed: %s", exc)
            return False

    def extract(
        self,
        topic: str | None = None,
        year: str = "2024",
        response: str | None = None,
        overall_only: bool = True,
        max_records: int = 10000,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch BRFSS prevalence rows.

        IMPORTANT: Filter on topic STRING (not topicid) to avoid casing issues
        across BRFSS vintages.

        Args:
            topic:        BRFSS topic string (e.g. "BMI Categories").
            year:         4-digit year string (e.g. "2024").
            response:     Response category to filter (e.g. "Obese (BMI 30.0 - 99.8)").
            overall_only: When True, filter to break_out='Overall' (excludes sex/race/age).
            max_records:  Maximum rows to return.
            **kwargs:     Ignored.

        Returns:
            Cleaned DataFrame with data_value cast to float.
        """
        where_parts: list[str] = []
        if year:
            where_parts.append(f"year='{year}'")
        if topic:
            # Filter on topic STRING not topicid (verified trap — see module docstring)
            where_parts.append(f"topic='{topic}'")
        if response:
            where_parts.append(f"response='{response}'")
        if overall_only:
            # 'Overall' break_out excludes sex/race/age stratifications
            where_parts.append("break_out='Overall'")

        where_clause = " AND ".join(where_parts) if where_parts else None

        _log.info(
            "Extracting BRFSS: topic=%s year=%s response=%s max=%d",
            topic, year, response, max_records,
        )

        all_records: list[dict] = []
        offset = 0

        while len(all_records) < max_records:
            params: dict = {
                "$limit": min(SOCRATA_PAGE_SIZE, max_records - len(all_records)),
                "$offset": offset,
            }
            if where_clause:
                params["$where"] = where_clause

            try:
                resp = requests.get(CDC_BRFSS_URL, params=params, timeout=30)
                resp.raise_for_status()
                page: list[dict] = resp.json()
            except Exception as exc:  # noqa: BLE001
                _log.error("BRFSS page fetch failed at offset=%d: %s", offset, exc)
                break

            if not page:
                _log.info("BRFSS: empty page at offset=%d — done", offset)
                break

            all_records.extend(page)
            _log.info(
                "BRFSS page: %d rows (total so far: %d)", len(page), len(all_records)
            )

            if len(page) < SOCRATA_PAGE_SIZE:
                break  # Last page

            offset += len(page)

        if not all_records:
            _log.warning("BRFSS: no records returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_records[:max_records])

        # Exclude aggregate rows.
        # BRFSS has BOTH 'US' (national aggregate) AND 'UW' (national median across
        # states) — both are non-state rows and must be excluded for state-level work.
        if "locationabbr" in df.columns:
            df = df[~df["locationabbr"].isin(EXCLUDED_LOCATION_CODES)]

        # Cast data_value: empty string → NA → float
        if "data_value" in df.columns:
            df["data_value"] = df["data_value"].replace("", pd.NA)
            df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

        df = df.reset_index(drop=True)
        self._record_extract(df)
        _log.info("BRFSS extraction complete: %d rows", len(df))
        return df
