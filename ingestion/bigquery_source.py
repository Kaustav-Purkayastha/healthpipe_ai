"""
ingestion/bigquery_source.py — Google BigQuery data source.

Uses the google-cloud-bigquery client directly (not SQLAlchemy) — this avoids
pulling in sqlalchemy-bigquery and its additional dependencies.

All Google Cloud imports are LAZY (inside methods) — the module is importable
without the heavy BigQuery packages (~13 Google packages incl. grpcio).

Security: credentials_path is stored but never logged; it is passed to the
BigQuery client via the GOOGLE_APPLICATION_CREDENTIALS environment variable,
which is Google's standard service-account auth mechanism.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from core.driver_manager import is_driver_installed
from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)


class BigQuerySource(BaseSource):
    """Google BigQuery data source.

    Authenticates via a service-account JSON key file.  Install the driver
    first (it's in the bigquery tier due to size)::

        from core.driver_manager import install_driver
        install_driver("bigquery")

    Usage::

        src = BigQuerySource()
        src.configure(
            project_id="my-gcp-project",
            credentials_path="/path/to/service_account.json",
        )
        if src.connect():
            df = src.extract("SELECT * FROM dataset.table LIMIT 100")
    """

    source_type: str = "database"

    def __init__(self) -> None:
        """Initialise with fixed registry name."""
        super().__init__(name="bigquery", description="Google BigQuery")
        self._project_id: Optional[str] = None
        self._credentials_path: Optional[str] = None
        self._client = None  # google.cloud.bigquery.Client

    def configure(
        self,
        project_id: str,
        credentials_path: Optional[str] = None,
    ) -> None:
        """Store BigQuery connection parameters.

        Args:
            project_id:       GCP project ID (e.g. ``my-project-123``).
            credentials_path: Path to a service-account JSON key file.  Sets
                              GOOGLE_APPLICATION_CREDENTIALS env var, which is
                              Google's standard mechanism for ADC (Application
                              Default Credentials).  When None the client uses
                              whatever ADC is already active (gcloud auth, etc.).
        """
        self._project_id = project_id
        self._credentials_path = credentials_path
        self._client = None

    def connect(self) -> bool:
        """Authenticate and run a test query (SELECT 1) against BigQuery.

        Returns:
            True on success, False on any error.
        """
        if not is_driver_installed("bigquery"):
            _log.warning(
                "BigQuery driver not installed. Run: "
                "from core.driver_manager import install_driver; "
                "install_driver('bigquery')"
            )
            return False

        try:
            from google.cloud import bigquery  # noqa: PLC0415

            # Set GOOGLE_APPLICATION_CREDENTIALS so the BigQuery client picks up
            # the service-account key — Google's standard ADC auth mechanism.
            if self._credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._credentials_path

            self._client = bigquery.Client(project=self._project_id)

            # Lightweight probe with a 15-second timeout
            job = self._client.query("SELECT 1")
            job.result(timeout=15)

            _log.info("BigQuerySource connected: project=%s", self._project_id)
            return True

        except Exception as exc:  # noqa: BLE001
            _log.error(
                "BigQuerySource connect failed: project=%s — %s",
                self._project_id,
                exc,
            )
            self._client = None
            return False

    def extract(
        self,
        query: str = "",
        max_rows: int = 100_000,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute a BigQuery SQL query and return results as a DataFrame.

        Uses ``job.result(max_results=max_rows).to_dataframe()``.  The
        ``db-dtypes`` package (in the bigquery pip tier) is required for
        to_dataframe() to handle BigQuery-specific types like NUMERIC and DATE.

        Args:
            query:    BigQuery SQL statement.
            max_rows: Maximum rows to return (passed to result()).
            **kwargs: Ignored.

        Returns:
            DataFrame (empty on error).
        """
        if self._client is None:
            _log.error("BigQuerySource.extract: not connected — call connect() first")
            return pd.DataFrame()

        if not query:
            _log.error("BigQuerySource.extract: query must not be empty")
            return pd.DataFrame()

        try:
            job = self._client.query(query)
            # db-dtypes is required for to_dataframe() to convert BigQuery types
            df = job.result(max_results=max_rows).to_dataframe()
            self._record_extract(df)
            _log.info("BigQuerySource extract: %d rows", len(df))
            return df

        except Exception as exc:  # noqa: BLE001
            _log.error("BigQuerySource.extract failed: %s", exc)
            return pd.DataFrame()

    def get_metadata(self) -> dict:
        """Return metadata — credentials path is omitted for security.

        Returns:
            Dict with project_id and base metadata.
        """
        meta = super().get_metadata()
        meta["project_id"] = self._project_id
        # credentials_path intentionally absent
        return meta
