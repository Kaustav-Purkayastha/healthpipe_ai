"""
who_source.py — Data source for the WHO Global Health Observatory (GHO) API.

The WHO GHO API uses the OData protocol. Data is paginated with $top (page size)
and $skip (offset) query parameters. No API key is required.

Example URL:
    https://ghoapi.azureedge.net/api/WHOSIS_000001?$top=1000&$skip=0
    → returns up to 1000 life expectancy records starting from record 0
"""

import time
from datetime import datetime

import pandas as pd
import requests

from core.config import (
    WHO_BASE_URL,
    WHO_INDICATORS,
    WHO_PAGE_SIZE,
    WHO_RATE_LIMIT_SECONDS,
)
from core.utils import get_logger
from ingestion.base_source import BaseSource

logger = get_logger(__name__)

# Maps friendly names to WHO indicator codes — lets users type "life_expectancy"
# instead of remembering "WHOSIS_000001"
FRIENDLY_TO_CODE: dict = {friendly: code for code, friendly in WHO_INDICATORS.items()}
# Also keep the reverse mapping for lookups
CODE_TO_FRIENDLY: dict = WHO_INDICATORS


class WHOSource(BaseSource):
    """
    Connects to the WHO GHO OData API and extracts health indicator data.

    Usage:
        source = WHOSource()
        source.connect()
        df = source.extract(indicator="life_expectancy", countries=["IND", "USA"])
    """

    def __init__(self) -> None:
        """Initialize with a fixed name and description."""
        super().__init__(
            name="who",
            description="WHO Global Health Observatory API"
        )
        # Track the most recent extraction for get_metadata()
        self._last_extract_time: str | None = None
        self._last_record_count: int = 0
        self._last_indicator: str | None = None

    def connect(self) -> bool:
        """
        Test connectivity by requesting the API root endpoint.

        The root URL returns a list of available indicators.
        If that responds with 200, the API is reachable.

        Returns:
            True if the API responds successfully, False otherwise.
        """
        try:
            # timeout=10 prevents hanging forever if the server is down
            response = requests.get(WHO_BASE_URL, timeout=10)
            # raise_for_status() throws an exception for 4xx/5xx HTTP codes
            response.raise_for_status()
            logger.info("WHO GHO API connection successful")
            return True
        except requests.ConnectionError:
            logger.error("WHO GHO API unreachable — check your internet connection")
            return False
        except requests.Timeout:
            logger.error("WHO GHO API timed out after 10 seconds")
            return False
        except requests.HTTPError as e:
            logger.error(f"WHO GHO API returned HTTP error: {e}")
            return False

    def extract(
        self,
        indicator: str = "life_expectancy",
        countries: list[str] | None = None,
        max_records: int = 5000,
    ) -> pd.DataFrame:
        """
        Fetch records for a specific WHO health indicator.

        Handles OData pagination automatically — keeps requesting pages until
        all data is fetched or max_records is reached.

        Args:
            indicator:   Friendly name (e.g., "life_expectancy") or raw code
                         (e.g., "WHOSIS_000001").
            countries:   Optional list of ISO 3-letter country codes to filter
                         (e.g., ["IND", "USA", "BRA"]). None = all countries.
            max_records: Safety cap on total records to fetch. Prevents
                         accidentally downloading millions of rows.

        Returns:
            DataFrame with columns: country_code, year, value, indicator_name,
            plus any other fields the API returns.
        """
        # Resolve friendly name to WHO code if needed
        indicator_code = self._resolve_indicator_code(indicator)
        logger.info(
            f"Extracting WHO data: indicator={indicator_code}, "
            f"countries={countries}, max_records={max_records}"
        )

        all_records: list[dict] = []
        skip = 0  # OData offset — starts at 0, incremented by page size each loop

        while len(all_records) < max_records:
            # Build the paginated URL: /api/WHOSIS_000001?$top=1000&$skip=0
            url = f"{WHO_BASE_URL}/{indicator_code}"
            params = {"$top": WHO_PAGE_SIZE, "$skip": skip}

            page_data = self._fetch_page(url, params)
            if page_data is None:
                # _fetch_page logs the error — break and return what we have
                break

            # WHO wraps records in a "value" key
            records = page_data.get("value", [])

            if not records:
                # Empty page means we've exhausted the data
                logger.info(
                    f"No more records at skip={skip}. "
                    f"Total fetched: {len(all_records)}"
                )
                break

            all_records.extend(records)
            skip += WHO_PAGE_SIZE  # Move to the next page

            logger.info(
                f"Fetched page: {len(records)} records "
                f"(total so far: {len(all_records)})"
            )

            # Respect WHO rate limit — wait before the next request
            time.sleep(WHO_RATE_LIMIT_SECONDS)

        # Trim to max_records if we overshot on the last page
        all_records = all_records[:max_records]

        if not all_records:
            logger.warning(f"No records returned for indicator {indicator_code}")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Rename WHO's column names to friendlier ones
        df = self._rename_columns(df)

        # Filter to requested countries if specified
        if countries and "country_code" in df.columns:
            # .isin() checks if each value is in the list — like SQL's IN clause
            df = df[df["country_code"].isin(countries)].reset_index(drop=True)

        # Add a column showing which indicator this data is for
        friendly_name = CODE_TO_FRIENDLY.get(indicator_code, indicator_code)
        df["indicator_name"] = friendly_name

        # Update metadata for get_metadata()
        self._last_extract_time = datetime.now().isoformat()
        self._last_record_count = len(df)
        self._last_indicator = friendly_name

        logger.info(
            f"WHO extraction complete: {len(df)} records for '{friendly_name}'"
        )
        return df

    def get_metadata(self) -> dict:
        """
        Return a summary of this source and the most recent extraction.

        Returns:
            Dict with source info and last extraction stats.
        """
        return {
            "name": self.name,
            "description": self.description,
            "source_type": "api",
            "base_url": WHO_BASE_URL,
            "available_indicators": list(WHO_INDICATORS.values()),
            "last_extract_time": self._last_extract_time,
            "last_record_count": self._last_record_count,
            "last_indicator": self._last_indicator,
        }

    def _resolve_indicator_code(self, indicator: str) -> str:
        """
        Convert a friendly indicator name to its WHO code.

        If the input is already a valid WHO code, return it as-is.

        Args:
            indicator: Friendly name ("life_expectancy") or code ("WHOSIS_000001").

        Returns:
            The WHO indicator code string.
        """
        if indicator in FRIENDLY_TO_CODE:
            # User passed a friendly name — look up the code
            return FRIENDLY_TO_CODE[indicator]
        if indicator in CODE_TO_FRIENDLY:
            # User already passed a valid WHO code — use it directly
            return indicator
        # Unknown indicator — use it raw and let the API handle the error
        logger.warning(
            f"Unknown indicator '{indicator}' — passing to API as-is"
        )
        return indicator

    def _fetch_page(self, url: str, params: dict) -> dict | None:
        """
        Make a single GET request to the WHO API.

        Args:
            url:    Full endpoint URL (e.g., https://ghoapi.../api/WHOSIS_000001).
            params: Query parameters dict (e.g., {"$top": 1000, "$skip": 0}).

        Returns:
            Parsed JSON response as a dict, or None if the request failed.
        """
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError:
            logger.error("Lost connection to WHO API during request")
            return None
        except requests.Timeout:
            logger.error(f"WHO API request timed out: {url}")
            return None
        except requests.HTTPError as e:
            logger.error(f"WHO API HTTP error: {e}")
            return None
        except ValueError:
            # response.json() raises ValueError if the body isn't valid JSON
            logger.error("WHO API returned invalid JSON")
            return None

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename WHO's raw column names to more readable ones.

        WHO returns columns like "SpatialDim", "TimeDim", "NumericValue".
        We rename them to "country_code", "year", "value" for consistency.

        Args:
            df: Raw DataFrame from the API.

        Returns:
            DataFrame with renamed columns (unmatched columns are kept as-is).
        """
        column_map = {
            "SpatialDim": "country_code",
            "TimeDim": "year",
            "NumericValue": "value",
            "Dim1": "dimension_1",
            "Dim1Type": "dimension_1_type",
        }
        # .rename() only renames columns that exist — ignores missing keys safely
        return df.rename(columns=column_map)
