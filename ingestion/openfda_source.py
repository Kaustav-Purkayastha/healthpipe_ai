"""
openfda_source.py — Data source for the OpenFDA Drug Adverse Events API.

OpenFDA returns deeply nested JSON. A single adverse event record can contain
multiple drugs, reactions, and patient info nested several levels deep.
This source flattens that structure into one row per event.

Key challenge: the API returns 429 (Too Many Requests) if you hit it too fast.
We handle that with exponential backoff — wait 1s, then 2s, then 4s.
"""

import time
from datetime import datetime

import pandas as pd
import requests

from core.config import OPENFDA_BASE_URL
from core.utils import get_logger
from ingestion.base_source import BaseSource

logger = get_logger(__name__)

# Local constants (not in v2 config — defined here to keep logic intact)
_OPENFDA_PAGE_SIZE: int = 100          # OpenFDA hard cap per request
_OPENFDA_RATE_LIMIT_SECONDS: float = 0.5
_OPENFDA_MAX_RETRIES: int = 3
_OPENFDA_RETRY_BACKOFF_BASE: float = 1.0  # exponential backoff base in seconds


class OpenFDASource(BaseSource):
    """
    Connects to the OpenFDA Drug Adverse Event API and extracts event records.

    The API returns nested JSON with patient, drug, and reaction data.
    This source flattens it into a tabular format suitable for analysis.

    Usage:
        source = OpenFDASource()
        source.connect()
        df = source.extract(search_term="aspirin", max_records=200)
    """

    def __init__(self) -> None:
        """Initialize with a fixed name and description."""
        super().__init__(
            name="openfda",
            description="OpenFDA Drug Adverse Events API"
        )
        self._last_extract_time: str | None = None
        self._last_record_count: int = 0
        self._last_search_term: str | None = None

    def connect(self) -> bool:
        """
        Test connectivity by requesting 1 record from the drug events endpoint.

        Returns:
            True if the API responds successfully, False otherwise.
        """
        try:
            url = f"{OPENFDA_BASE_URL}"
            # Just fetch 1 record to confirm the API is alive
            response = requests.get(url, params={"limit": 1}, timeout=10)
            response.raise_for_status()
            logger.info("OpenFDA API connection successful")
            return True
        except requests.ConnectionError:
            logger.error(
                "OpenFDA API unreachable — check your internet connection"
            )
            return False
        except requests.Timeout:
            logger.error("OpenFDA API timed out after 10 seconds")
            return False
        except requests.HTTPError as e:
            logger.error(f"OpenFDA API returned HTTP error: {e}")
            return False

    def extract(
        self,
        search_term: str = "aspirin",
        date_range: tuple[str, str] | None = None,
        max_records: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch adverse event records matching a search term.

        Uses limit/skip pagination. Each page returns up to 100 records
        (OpenFDA hard cap). Automatically retries on 429 errors.

        Args:
            search_term: Drug name or search query (e.g., "aspirin").
            date_range:  Optional tuple of (start, end) dates in YYYYMMDD
                         format (e.g., ("20200101", "20231231")).
            max_records: Safety cap on total records to fetch.

        Returns:
            DataFrame with flattened adverse event data.
        """
        logger.info(
            f"Extracting OpenFDA data: search='{search_term}', "
            f"date_range={date_range}, max_records={max_records}"
        )

        # Build the search query string for the API
        search_query = self._build_search_query(search_term, date_range)

        all_records: list[dict] = []
        skip = 0

        while len(all_records) < max_records:
            # Calculate how many records to request this page
            # (don't request more than we still need)
            remaining = max_records - len(all_records)
            limit = min(_OPENFDA_PAGE_SIZE, remaining)

            page_data = self._fetch_page_with_retry(
                search_query, limit=limit, skip=skip
            )

            if page_data is None:
                break

            # OpenFDA wraps records in a "results" key
            results = page_data.get("results", [])

            if not results:
                logger.info(
                    f"No more results at skip={skip}. "
                    f"Total fetched: {len(all_records)}"
                )
                break

            # Flatten each nested event into a flat dict
            for event in results:
                flat = self._flatten_event(event)
                all_records.append(flat)

            skip += limit
            logger.info(
                f"Fetched page: {len(results)} events "
                f"(total so far: {len(all_records)})"
            )

            # Respect rate limit
            time.sleep(_OPENFDA_RATE_LIMIT_SECONDS)

        if not all_records:
            logger.warning(
                f"No records returned for search term '{search_term}'"
            )
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Update metadata via BaseSource helper
        self._record_extract(df)
        self._last_search_term = search_term

        logger.info(
            f"OpenFDA extraction complete: {len(df)} records "
            f"for '{search_term}'"
        )
        return df

    def get_metadata(self) -> dict:
        """
        Return a summary of this source and the most recent extraction.

        Returns:
            Dict with source info and last extraction stats.
        """
        meta = super().get_metadata()
        meta["base_url"] = OPENFDA_BASE_URL
        meta["last_search_term"] = self._last_search_term
        return meta

    def _build_search_query(
        self,
        search_term: str,
        date_range: tuple[str, str] | None = None,
    ) -> str:
        """
        Build the OpenFDA search query string.

        OpenFDA uses a custom search syntax:
            patient.drug.openfda.brand_name:"aspirin"
        Date ranges use bracket notation:
            receivedate:[20200101+TO+20231231]

        Args:
            search_term: Drug name to search for.
            date_range:  Optional (start, end) date tuple.

        Returns:
            URL-ready search query string.
        """
        # Search across the brand_name field in the drug's openfda section
        query = f'patient.drug.openfda.brand_name:"{search_term}"'

        if date_range is not None:
            start_date, end_date = date_range
            # +AND+ combines multiple search conditions in OpenFDA syntax
            query += (
                f"+AND+receivedate:[{start_date}+TO+{end_date}]"
            )

        return query

    def _fetch_page_with_retry(
        self,
        search_query: str,
        limit: int,
        skip: int,
    ) -> dict | None:
        """
        Fetch one page of results, retrying on 429 with exponential backoff.

        Exponential backoff means the wait time doubles each attempt:
            Attempt 1: wait 1 second
            Attempt 2: wait 2 seconds
            Attempt 3: wait 4 seconds

        Args:
            search_query: The OpenFDA search query string.
            limit:        Number of records to fetch (max 100).
            skip:         Number of records to skip (pagination offset).

        Returns:
            Parsed JSON response as dict, or None if all retries failed.
        """
        url = f"{OPENFDA_BASE_URL}"
        params = {
            "search": search_query,
            "limit": limit,
            "skip": skip,
        }

        for attempt in range(_OPENFDA_MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 429:
                    # 429 = Too Many Requests — server says "slow down"
                    # Calculate wait time: base * 2^attempt → 1s, 2s, 4s
                    wait = _OPENFDA_RETRY_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"Rate limited (429). Waiting {wait}s "
                        f"(attempt {attempt + 1}/{_OPENFDA_MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue  # Retry the same request

                response.raise_for_status()
                return response.json()

            except requests.ConnectionError:
                logger.error("Lost connection to OpenFDA API")
                return None
            except requests.Timeout:
                logger.error(f"OpenFDA request timed out (attempt {attempt + 1})")
                if attempt < _OPENFDA_MAX_RETRIES - 1:
                    # Wait before retrying on timeout too
                    wait = _OPENFDA_RETRY_BACKOFF_BASE * (2 ** attempt)
                    time.sleep(wait)
                    continue
                return None
            except requests.HTTPError as e:
                # 404 often means "no results" rather than a real error
                if response.status_code == 404:
                    logger.info("OpenFDA returned 404 — no matching records")
                    return None
                logger.error(f"OpenFDA HTTP error: {e}")
                return None
            except ValueError:
                logger.error("OpenFDA returned invalid JSON")
                return None

        # All retries exhausted
        logger.error(
            f"OpenFDA request failed after {_OPENFDA_MAX_RETRIES} attempts"
        )
        return None

    def _flatten_event(self, event: dict) -> dict:
        """
        Flatten a deeply nested adverse event record into a flat dictionary.

        OpenFDA event JSON is nested like this:
            {
                "safetyreportid": "123",
                "patient": {
                    "patientonsetage": "45",
                    "drug": [{"openfda": {"brand_name": ["Aspirin"]}}],
                    "reaction": [{"reactionmeddrapt": "Headache"}]
                }
            }

        We flatten it into:
            {"safety_report_id": "123", "patient_age": "45",
             "brand_name": "Aspirin", "reactions": "Headache"}

        Args:
            event: Raw event dict from the API response.

        Returns:
            Flat dict with one key per field (no nesting).
        """
        flat: dict = {}

        # Top-level fields
        flat["safety_report_id"] = event.get("safetyreportid", "")
        flat["receive_date"] = event.get("receivedate", "")
        flat["serious"] = event.get("serious", "")
        flat["sender_organization"] = event.get(
            "companynumb", ""
        )

        # Patient-level fields — nested one level deep
        # .get("patient", {}) returns empty dict if "patient" key is missing,
        # so the next .get() call won't crash
        patient = event.get("patient", {})
        flat["patient_age"] = patient.get("patientonsetage", "")
        flat["patient_sex"] = patient.get("patientsex", "")
        flat["patient_weight"] = patient.get("patientweight", "")

        # Drugs — nested inside patient, it's a list of dicts
        drugs = patient.get("drug", [])
        if drugs:
            # Take the first drug's info (events can have multiple drugs)
            first_drug = drugs[0]
            flat["drug_name"] = first_drug.get("medicinalproduct", "")
            flat["drug_indication"] = first_drug.get(
                "drugindication", ""
            )

            # Brand name is nested even deeper: drug -> openfda -> brand_name
            openfda = first_drug.get("openfda", {})
            brand_names = openfda.get("brand_name", [])
            # brand_name is a list — join into a single string
            flat["brand_name"] = "; ".join(brand_names) if brand_names else ""

            generic_names = openfda.get("generic_name", [])
            flat["generic_name"] = (
                "; ".join(generic_names) if generic_names else ""
            )
        else:
            flat["drug_name"] = ""
            flat["drug_indication"] = ""
            flat["brand_name"] = ""
            flat["generic_name"] = ""

        # Reactions — also a list inside patient
        reactions = patient.get("reaction", [])
        # Extract reaction names and join them: "Headache; Nausea; Dizziness"
        reaction_names = [
            r.get("reactionmeddrapt", "") for r in reactions
        ]
        flat["reactions"] = "; ".join(reaction_names)

        return flat
