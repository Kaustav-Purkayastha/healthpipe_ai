"""
ingestion/census_source.py — US Census ACS5 population data source.

LIVE-VERIFIED FACTS (do not change):
- Endpoint: CENSUS_ACS5_URL (https://api.census.gov/data/2023/acs/acs5)
- Requires CENSUS_API_KEY in .env (free key at https://api.census.gov/data/key_signup.html)
- Response is a JSON array-of-arrays. FIRST ROW IS THE HEADER:
    [["NAME","B01003_001E","state"], ["Alabama","5108468","01"], ...]
  All values are strings — cast population to int.
- state column is the FIPS code (zero-padded 2-char string, e.g. "01").
- Rows for territories (Puerto Rico "72", etc.) are included in the API response
  but must be dropped (not in FIPS_TO_STATE crosswalk).
"""

from __future__ import annotations

import pandas as pd
import requests

from core.config import (
    CENSUS_ACS5_URL,
    CENSUS_API_KEY,
    CENSUS_POPULATION_VAR,
    FIPS_TO_STATE,
)
from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)


class CensusPopulationSource(BaseSource):
    """ACS5 state-level population from the US Census Bureau API.

    Requires CENSUS_API_KEY set in .env; connect() returns False when key missing.

    Usage:
        src = CensusPopulationSource()
        if src.connect():
            df = src.extract()  # returns 51 rows (50 states + DC)
    """

    source_type: str = "api"

    def __init__(self) -> None:
        """Initialise with registry name and description."""
        super().__init__(
            name="census",
            description="US Census ACS5 State Population (2023)",
        )

    def connect(self) -> bool:
        """Return True if the Census API key is configured and the endpoint responds.

        Returns False immediately (with a clear warning) when no key is set,
        so the registry can report "not configured" rather than "unreachable".

        Returns:
            bool
        """
        if not CENSUS_API_KEY:
            _log.warning(
                "Census API key not set — add CENSUS_API_KEY to .env "
                "(free key: https://api.census.gov/data/key_signup.html)"
            )
            return False

        try:
            # Probe with a minimal request (1-variable, 1-state)
            resp = requests.get(
                CENSUS_ACS5_URL,
                params={
                    "get": f"NAME,{CENSUS_POPULATION_VAR}",
                    "for": "state:01",  # Alabama — lightweight probe
                    "key": CENSUS_API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            _log.info("Census API connected")
            return True
        except Exception as exc:  # noqa: BLE001
            _log.error("Census API connect failed: %s", exc)
            return False

    def extract(self, **kwargs) -> pd.DataFrame:
        """Fetch ACS5 population estimates for all 50 states + DC.

        The API returns an array-of-arrays where the first element is a header row.
        Territories (Puerto Rico, etc.) are included by the API but dropped by the
        FIPS crosswalk filter.

        Args:
            **kwargs: Ignored (for registry interface compatibility).

        Returns:
            DataFrame with 51 rows (50 states + DC) and columns:
                state_name, population (int), state_fips (str), state_abbr (str).
        """
        if not CENSUS_API_KEY:
            _log.warning("Census extract skipped — no API key configured")
            return pd.DataFrame()

        params = {
            "get": f"NAME,{CENSUS_POPULATION_VAR}",
            "for": "state:*",
            "key": CENSUS_API_KEY,
        }

        try:
            resp = requests.get(CENSUS_ACS5_URL, params=params, timeout=30)
            resp.raise_for_status()
            rows: list[list[str]] = resp.json()
        except Exception as exc:  # noqa: BLE001
            _log.error("Census API extract failed: %s", exc)
            return pd.DataFrame()

        if len(rows) < 2:
            _log.warning("Census API returned fewer than 2 rows (no data)")
            return pd.DataFrame()

        # IMPORTANT: First row is the header — rest are data rows.
        # All values are strings as returned by the Census API.
        header = rows[0]
        data = rows[1:]

        df = pd.DataFrame(data, columns=header)

        # Rename Census column names to friendly names
        df = df.rename(columns={
            "NAME": "state_name",
            CENSUS_POPULATION_VAR: "population",
            "state": "state_fips",   # FIPS code as zero-padded string
        })

        # Keep state_fips as string (leading zeros critical for e.g. "01" Alabama)
        df["state_fips"] = df["state_fips"].astype(str)

        # Map FIPS → USPS abbreviation; drop rows not in the crosswalk (territories)
        df["state_abbr"] = df["state_fips"].map(FIPS_TO_STATE)
        df = df.dropna(subset=["state_abbr"]).reset_index(drop=True)

        # Cast population to int (arrives as string from the Census API)
        df["population"] = pd.to_numeric(df["population"], errors="coerce").astype(
            "Int64"
        )

        self._record_extract(df)
        _log.info("Census extraction complete: %d states", len(df))
        return df
