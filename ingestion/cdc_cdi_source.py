"""
ingestion/cdc_cdi_source.py — CDC Chronic Disease Indicators (CDI) via Socrata.

LIVE-VERIFIED FACTS (do not change):
- Endpoint: data.cdc.gov/resource/hksd-2xuw.json (CDC_CDI_URL in config)
- JSON field names are LOWERCASE: yearstart, locationabbr, datavalue, etc.
- datavalue arrives as a STRING; empty string means suppressed (use is_suppressed flag).
- questionid is case-sensitive: "DIA01", "NPW14", "TOB04".
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from core.config import CDC_CDI_URL, EXCLUDED_LOCATION_CODES, SOCRATA_PAGE_SIZE
from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)


class CDCChronicDiseaseSource(BaseSource):
    """CDC Chronic Disease Indicators via the Socrata JSON API.

    Verified question IDs (class constants — do not rename):
        DIABETES = "DIA01"
        OBESITY  = "NPW14"  (topic: 'Nutrition, Physical Activity, and Weight
                              Status' — NOT the 'Obesity' topic)
        SMOKING  = "TOB04"

    Usage:
        src = CDCChronicDiseaseSource()
        if src.connect():
            df = src.extract(
                question_id=CDCChronicDiseaseSource.DIABETES,
                year="2023",
            )
    """

    source_type: str = "api"

    # Verified live question IDs — do not alter casing
    DIABETES: str = "DIA01"
    OBESITY: str = "NPW14"   # topic='Nutrition, Physical Activity, and Weight Status'
    SMOKING: str = "TOB04"

    def __init__(self) -> None:
        """Initialise with registry name and description."""
        super().__init__(
            name="cdc_cdi",
            description="CDC Chronic Disease Indicators (Socrata)",
        )

    def connect(self) -> bool:
        """Return True if the CDI endpoint responds with 200 for a 1-row probe.

        Returns:
            bool
        """
        try:
            resp = requests.get(CDC_CDI_URL, params={"$limit": 1}, timeout=10)
            resp.raise_for_status()
            _log.info("CDC CDI API connected")
            return True
        except Exception as exc:  # noqa: BLE001
            _log.error("CDC CDI connect failed: %s", exc)
            return False

    def extract(
        self,
        question_id: str | None = None,
        year: str | None = None,
        data_value_type: str = "CRDPREV",
        overall_only: bool = True,
        max_records: int = 10000,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch CDC Chronic Disease Indicator rows.

        Builds a Socrata $where clause from the supplied filters.
        Paginates with $limit / $offset.

        Args:
            question_id:     Socrata questionid code (e.g. "DIA01").
            year:            4-digit year string e.g. "2023".
            data_value_type: datavaluetypeid filter (default "CRDPREV" = crude
                             prevalence — the most comparable cross-state measure).
            overall_only:    When True, filter to stratificationcategory1='Overall'
                             (excludes sex/race/age breakdowns).
            max_records:     Maximum rows to return.
            **kwargs:        Ignored.

        Returns:
            Cleaned DataFrame with is_suppressed bool column added.
        """
        where_parts: list[str] = []
        if question_id:
            where_parts.append(f"questionid='{question_id}'")
        if year:
            where_parts.append(f"yearstart='{year}'")
        if data_value_type:
            where_parts.append(f"datavaluetypeid='{data_value_type}'")
        if overall_only:
            where_parts.append("stratificationcategory1='Overall'")

        where_clause = " AND ".join(where_parts) if where_parts else None

        _log.info(
            "Extracting CDC CDI: question=%s year=%s where=%s max=%d",
            question_id, year, where_clause, max_records,
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
                resp = requests.get(CDC_CDI_URL, params=params, timeout=30)
                resp.raise_for_status()
                page: list[dict] = resp.json()
            except Exception as exc:  # noqa: BLE001
                _log.error("CDC CDI page fetch failed at offset=%d: %s", offset, exc)
                break

            if not page:
                _log.info("CDC CDI: empty page at offset=%d — done", offset)
                break

            all_records.extend(page)
            _log.info(
                "CDC CDI page: %d rows (total so far: %d)", len(page), len(all_records)
            )

            if len(page) < SOCRATA_PAGE_SIZE:
                break  # Last page

            offset += len(page)

        if not all_records:
            _log.warning("CDC CDI: no records returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_records[:max_records])

        # Exclude aggregate / territory location codes
        if "locationabbr" in df.columns:
            df = df[~df["locationabbr"].isin(EXCLUDED_LOCATION_CODES)]

        # Cast datavalue: empty string → NA → float
        if "datavalue" in df.columns:
            df["datavalue"] = df["datavalue"].replace("", pd.NA)
            df["datavalue"] = pd.to_numeric(df["datavalue"], errors="coerce")

        # is_suppressed: datavalue is NA AND footnote symbol is "*"
        if "datavaluefootnotesymbol" in df.columns:
            df["is_suppressed"] = (
                df["datavalue"].isna()
                & (df["datavaluefootnotesymbol"].fillna("") == "*")
            )
        else:
            df["is_suppressed"] = False

        df = df.reset_index(drop=True)
        self._record_extract(df)
        _log.info("CDC CDI extraction complete: %d rows", len(df))
        return df
