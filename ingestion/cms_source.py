"""
ingestion/cms_source.py — CMS Medicare Provider Summary (CY2024) data source.

Endpoint: data.cms.gov Data API v1, keyless.
Dataset: "Medicare Physician & Other Practitioners — by Provider" (CY2024).
~1.3 million rows — one row per provider NPI.

IMPORTANT live-verified facts (do not change):
- Response is a BARE JSON list (no envelope dict).
- All numeric values arrive as STRINGS — cast with pd.to_numeric(errors='coerce').
- Foreign/unknown providers have Rndrng_Prvdr_Cntry='ZZ' — filtered out post-extract.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from core.config import (
    CMS_BASE_URL,
    CMS_PAGE_SIZE,
    CMS_PROVIDER_SUMMARY_ID,
    CMS_RATE_LIMIT_SECONDS,
    STATE_ABBRS,
)
from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)

# CMS column → friendly column name
_RENAME_MAP: dict[str, str] = {
    "Rndrng_NPI": "npi",
    "Rndrng_Prvdr_Last_Org_Name": "provider_last_name",
    "Rndrng_Prvdr_First_Name": "provider_first_name",
    "Rndrng_Prvdr_MI": "provider_middle_initial",
    "Rndrng_Prvdr_Crdntls": "credentials",
    "Rndrng_Prvdr_Gndr": "gender",
    "Rndrng_Prvdr_Type": "provider_specialty",
    "Rndrng_Prvdr_State_Abrvtn": "state",
    "Rndrng_Prvdr_State_FIPS": "state_fips",
    "Rndrng_Prvdr_Zip5": "zip_code",
    "Rndrng_Prvdr_Cntry": "country",
    "Tot_Benes": "total_beneficiaries",
    "Tot_Srvcs": "total_services",
    "Tot_Mdcr_Pymt_Amt": "total_medicare_payment",
    "Tot_Mdcr_Stdzd_Amt": "total_medicare_payment_std",
}

# Columns that must stay as strings (leading zeros, identifiers)
_STRING_COLUMNS: set[str] = {"npi", "state_fips", "zip_code"}

# Numeric columns to cast from string
_NUMERIC_COLUMNS: list[str] = [
    "total_beneficiaries", "total_services",
    "total_medicare_payment", "total_medicare_payment_std",
]


class CMSMedicareSource(BaseSource):
    """CMS Medicare Provider Summary data (CY2024, by Provider).

    Usage:
        src = CMSMedicareSource()
        if src.connect():
            df = src.extract(state="MD", max_records=500)
    """

    source_type: str = "api"

    def __init__(self) -> None:
        """Initialise with registry name and description."""
        super().__init__(
            name="cms_medicare",
            description="CMS Medicare Provider & Service Summary (CY2024)",
        )
        self._total_rows: int = -1  # cached from stats endpoint

    def connect(self) -> bool:
        """Ping the dataset stats endpoint; cache total_rows for metadata.

        Returns:
            True if the endpoint responds with 200.
        """
        url = f"{CMS_BASE_URL}/{CMS_PROVIDER_SUMMARY_ID}/data/stats"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self._total_rows = int(data.get("total", -1))
            _log.info("CMS Medicare connected — total_rows=%d", self._total_rows)
            return True
        except Exception as exc:  # noqa: BLE001
            _log.error("CMS Medicare connect failed: %s", exc)
            return False

    def extract(
        self,
        state: str | None = None,
        specialty: str | None = None,
        max_records: int = 5000,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch CMS Medicare provider summary rows.

        Args:
            state:       2-letter USPS state code to filter server-side (optional).
            specialty:   Provider specialty string to filter server-side (optional).
            max_records: Maximum rows to return (safety cap).
            **kwargs:    Ignored for registry interface compatibility.

        Returns:
            Cleaned DataFrame with renamed columns and numeric types cast.
        """
        _log.info(
            "Extracting CMS Medicare: state=%s specialty=%s max=%d",
            state, specialty, max_records,
        )
        url = f"{CMS_BASE_URL}/{CMS_PROVIDER_SUMMARY_ID}/data"
        all_records: list[dict] = []
        offset = 0

        while len(all_records) < max_records:
            remaining = max_records - len(all_records)
            params: dict = {
                "size": min(CMS_PAGE_SIZE, remaining),
                "offset": offset,
            }
            # Server-side filters (requests encodes bracket notation correctly)
            if state:
                params["filter[Rndrng_Prvdr_State_Abrvtn]"] = state
            if specialty:
                params["filter[Rndrng_Prvdr_Type]"] = specialty

            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                page: list[dict] = resp.json()
            except Exception as exc:  # noqa: BLE001
                _log.error("CMS page fetch failed at offset=%d: %s", offset, exc)
                break

            if not page:
                _log.info("CMS: empty page at offset=%d — done", offset)
                break

            all_records.extend(page)
            _log.info(
                "CMS page fetched: %d rows (total so far: %d)",
                len(page), len(all_records),
            )

            # Short page means last page — stop early
            if len(page) < CMS_PAGE_SIZE:
                break

            offset += len(page)
            time.sleep(CMS_RATE_LIMIT_SECONDS)

        if not all_records:
            _log.warning("CMS Medicare: no records returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_records[:max_records])

        # Post-filter: drop foreign / ZZ rows.
        # Verified: CMS data includes rows with Rndrng_Prvdr_Cntry='ZZ' (foreign
        # providers who billed Medicare). Keep only 'US' + known state codes.
        if "Rndrng_Prvdr_Cntry" in df.columns:
            df = df[df["Rndrng_Prvdr_Cntry"] == "US"]
        if "Rndrng_Prvdr_State_Abrvtn" in df.columns:
            df = df[df["Rndrng_Prvdr_State_Abrvtn"].isin(STATE_ABBRS)]

        df = df.rename(columns=_RENAME_MAP)

        # Cast numerics — values arrive as strings; empty strings → NA first
        df = df.replace("", pd.NA)
        for col in _NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep ID columns as strings (leading zeros in FIPS / NPI)
        for col in _STRING_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = df.reset_index(drop=True)
        self._record_extract(df)
        _log.info("CMS extraction complete: %d rows", len(df))
        return df

    def get_metadata(self) -> dict:
        """Return metadata including cached total_rows from stats endpoint.

        Returns:
            Dict extending the base metadata with CMS-specific fields.
        """
        meta = super().get_metadata()
        meta["dataset_id"] = CMS_PROVIDER_SUMMARY_ID
        meta["total_rows_in_dataset"] = self._total_rows
        return meta
