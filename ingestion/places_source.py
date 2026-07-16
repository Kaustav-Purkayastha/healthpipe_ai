"""
ingestion/places_source.py — CDC PLACES county-level prevalence via Socrata.

LIVE-VERIFIED FACTS (do not "correct" these):
- Endpoint: data.cdc.gov/resource/swc5-untb.json (CDC_PLACES_URL in config).
- JSON field names are LOWERCASE (stateabbr, locationname, locationid, ...).
- locationid is a 5-digit county FIPS **string** (e.g. "06061") — keep as string
  (leading zeros matter).
- data_value and totalpop18plus arrive as **strings** — cast defensively.
- datavaluetypeid is MIXED case, unlike CDI's uppercase CRDPREV:
    'CrdPrv'   = crude prevalence
    'AgeAdjPrv'= age-adjusted prevalence
  (This is the trap: do NOT uppercase it.)
- The 2025 release carries year="2023" data, so we do NOT filter on year.
- 'US' is a national roll-up row and territories are absent — exclude stateabbr=='US'.
"""

from __future__ import annotations

import pandas as pd
import requests

from core.config import CDC_PLACES_URL, SOCRATA_PAGE_SIZE
from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)


class CDCPlacesSource(BaseSource):
    """CDC PLACES county-level prevalence estimates via the Socrata JSON API.

    Verified measure IDs (class constants — do not rename):
        OBESITY  = "OBESITY"
        DIABETES = "DIABETES"
        SMOKING  = "CSMOKING"   (current smoking among adults)
        BPHIGH   = "BPHIGH"     (high blood pressure)

    datavaluetypeid selects crude vs age-adjusted prevalence:
        CRUDE        = "CrdPrv"
        AGE_ADJUSTED = "AgeAdjPrv"

    Usage:
        src = CDCPlacesSource()
        if src.connect():
            df = src.extract(state_abbr="TX", measure_id="DIABETES")
    """

    source_type: str = "api"

    # Verified live measure IDs — uppercase, unlike datavaluetypeid.
    OBESITY: str = "OBESITY"
    DIABETES: str = "DIABETES"
    SMOKING: str = "CSMOKING"
    BPHIGH: str = "BPHIGH"

    # datavaluetypeid values (MIXED case — verified live).
    CRUDE: str = "CrdPrv"
    AGE_ADJUSTED: str = "AgeAdjPrv"

    def __init__(self) -> None:
        """Initialise with registry name and description."""
        super().__init__(
            name="cdc_places",
            description="CDC PLACES county-level prevalence (Socrata)",
        )

    def connect(self) -> bool:
        """Return True if the PLACES endpoint responds with 200 for a 1-row probe.

        Returns:
            bool
        """
        try:
            resp = requests.get(CDC_PLACES_URL, params={"$limit": 1}, timeout=10)
            resp.raise_for_status()
            _log.info("CDC PLACES API connected")
            return True
        except Exception as exc:  # noqa: BLE001 — any error → not reachable
            _log.error("CDC PLACES connect failed: %s", exc)
            return False

    def extract(
        self,
        state_abbr: str | None = None,
        measure_id: str | None = None,
        data_value_type: str = "CrdPrv",
        max_records: int = 10000,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch CDC PLACES county rows, cleaned and typed.

        Args:
            state_abbr:      2-letter state to filter (e.g. "TX"). None = all states.
            measure_id:      PLACES measureid (e.g. "DIABETES"). None = all measures.
            data_value_type: datavaluetypeid — "CrdPrv" (crude, default) or
                             "AgeAdjPrv" (age-adjusted). MIXED case; passed verbatim.
            max_records:     Maximum rows to return.
            **kwargs:        Ignored.

        Returns:
            DataFrame with cleaned columns: stateabbr, locationname, locationid
            (str FIPS), measureid, datavaluetypeid, data_value (float),
            totalpop18plus (int), year.  Empty DataFrame on failure.
        """
        # We deliberately do NOT filter on year: the current release stores 2023
        # in the year field, so a year==2025 filter would return nothing.
        where_parts: list[str] = ["stateabbr != 'US'"]  # drop national roll-up
        if state_abbr:
            where_parts.append(f"stateabbr='{state_abbr}'")
        if measure_id:
            where_parts.append(f"measureid='{measure_id}'")
        if data_value_type:
            where_parts.append(f"datavaluetypeid='{data_value_type}'")
        where_clause = " AND ".join(where_parts)

        _log.info(
            "Extracting CDC PLACES: state=%s measure=%s type=%s max=%d",
            state_abbr, measure_id, data_value_type, max_records,
        )

        all_records: list[dict] = []
        offset = 0
        while len(all_records) < max_records:
            params: dict = {
                "$limit": min(SOCRATA_PAGE_SIZE, max_records - len(all_records)),
                "$offset": offset,
                "$where": where_clause,
            }
            try:
                resp = requests.get(CDC_PLACES_URL, params=params, timeout=30)
                resp.raise_for_status()
                page: list[dict] = resp.json()
            except Exception as exc:  # noqa: BLE001 — never raise from a source
                _log.error("CDC PLACES page fetch failed at offset=%d: %s", offset, exc)
                break

            if not page:
                break
            all_records.extend(page)
            if len(page) < SOCRATA_PAGE_SIZE:
                break
            offset += len(page)

        if not all_records:
            _log.warning("CDC PLACES: no records returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_records[:max_records])
        df = self._clean(df)
        self._record_extract(df)
        _log.info("CDC PLACES extraction complete: %d rows", len(df))
        return df

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Type-cast PLACES fields defensively (all arrive as strings).

        Keeps locationid as a string (5-digit county FIPS with leading zeros);
        casts data_value → float and totalpop18plus → nullable int.
        """
        # Defensive: exclude any 'US' roll-up that slipped through.
        if "stateabbr" in df.columns:
            df = df[df["stateabbr"] != "US"].copy()

        if "data_value" in df.columns:
            df["data_value"] = df["data_value"].replace("", pd.NA)
            df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

        if "totalpop18plus" in df.columns:
            df["totalpop18plus"] = pd.to_numeric(
                df["totalpop18plus"], errors="coerce"
            ).astype("Int64")

        # locationid stays a string — leading zeros are significant for FIPS.
        if "locationid" in df.columns:
            df["locationid"] = df["locationid"].astype(str)

        return df.reset_index(drop=True)
