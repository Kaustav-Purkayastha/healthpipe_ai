"""
ingestion/databricks_source.py — Databricks SQL Warehouse connector.

Uses the native databricks-sql-connector DBAPI directly (not SQLAlchemy) because
the Databricks dialect for SQLAlchemy requires a separate package that was not
in the tested-and-pinned dependency matrix.

All databricks imports are LAZY (inside methods) — this module must be importable
even when databricks-sql-connector is not installed.

Security: access_token is stored name-mangled and never written to logs or metadata.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.driver_manager import is_driver_installed
from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)


class DatabricksSource(BaseSource):
    """Databricks SQL Warehouse data source.

    Connects via the Databricks SQL Connector DBAPI and executes arbitrary SQL
    queries.  Install the driver first::

        from core.driver_manager import install_driver
        install_driver("databricks")

    Usage::

        src = DatabricksSource()
        src.configure(
            server_hostname="dbc-xxxxxx.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/xxxxxxxx",
            access_token="dapi...",
        )
        if src.connect():
            df = src.extract("SELECT * FROM sales.orders LIMIT 100")
    """

    source_type: str = "database"

    def __init__(self) -> None:
        """Initialise with fixed registry name."""
        super().__init__(name="databricks", description="Databricks SQL Warehouse")
        self._server_hostname: Optional[str] = None
        self._http_path: Optional[str] = None
        # Name-mangled: never logged, never in get_metadata()
        self.__access_token: Optional[str] = None
        self._connection = None  # databricks.sql Connection

    def configure(
        self,
        server_hostname: str,
        http_path: str,
        access_token: str,
    ) -> None:
        """Store Databricks connection parameters.

        Args:
            server_hostname: Databricks workspace hostname (e.g.
                             ``dbc-xxxxxx.azuredatabricks.net``).
            http_path:       SQL warehouse HTTP path (e.g.
                             ``/sql/1.0/warehouses/xxxxxxxx``).
            access_token:    Personal access token — kept in memory only.
        """
        self._server_hostname = server_hostname
        self._http_path = http_path
        self.__access_token = access_token
        self._connection = None

    def connect(self) -> bool:
        """Open a Databricks SQL Warehouse connection and ping with SELECT 1.

        Returns:
            True on successful ping, False on any error.
        """
        if not is_driver_installed("databricks"):
            _log.warning(
                "Databricks driver not installed. Run: "
                "from core.driver_manager import install_driver; "
                "install_driver('databricks')"
            )
            return False

        try:
            import databricks.sql as dbsql  # noqa: PLC0415

            conn = dbsql.connect(
                server_hostname=self._server_hostname,
                http_path=self._http_path,
                access_token=self.__access_token,
            )
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            self._connection = conn
            _log.info(
                "DatabricksSource connected: host=%s path=%s",
                self._server_hostname,
                self._http_path,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            # Never log the access_token — log host+path only
            _log.error(
                "DatabricksSource connect failed: host=%s path=%s — %s",
                self._server_hostname,
                self._http_path,
                exc,
            )
            self._connection = None
            return False

    def extract(
        self,
        query: str = "",
        max_rows: int = 100_000,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute a SQL query and return the results as a DataFrame.

        Fetch strategy:
          1. cursor.fetchall_arrow().to_pandas() — zero-copy via Apache Arrow;
             available in databricks-sql-connector ≥ 2.x.
          2. Fallback: cursor.fetchall() + column names from cursor.description;
             works with any DBAPI2-compliant cursor.

        Args:
            query:    SQL SELECT statement.
            max_rows: Hard cap on returned rows.
            **kwargs: Ignored.

        Returns:
            DataFrame (empty on error).
        """
        if self._connection is None:
            _log.error("DatabricksSource.extract: not connected — call connect() first")
            return pd.DataFrame()

        if not query:
            _log.error("DatabricksSource.extract: query must not be empty")
            return pd.DataFrame()

        try:
            with self._connection.cursor() as cur:
                cur.execute(query)

                # Path 1: Arrow fetch (zero-copy, preferred for large results)
                if hasattr(cur, "fetchall_arrow"):
                    df = cur.fetchall_arrow().to_pandas()
                else:
                    # Path 2: DBAPI2 fallback — reconstruct column names from description
                    rows = cur.fetchall()
                    cols = [d[0] for d in (cur.description or [])]
                    df = pd.DataFrame(rows, columns=cols)

            df = df.head(max_rows)
            self._record_extract(df)
            _log.info("DatabricksSource extract: %d rows", len(df))
            return df

        except Exception as exc:  # noqa: BLE001
            _log.error("DatabricksSource.extract failed: %s", exc)
            return pd.DataFrame()

    def get_metadata(self) -> dict:
        """Return metadata — access_token is deliberately excluded.

        Returns:
            Dict with server_hostname, http_path, and base metadata.
        """
        meta = super().get_metadata()
        meta["server_hostname"] = self._server_hostname
        meta["http_path"] = self._http_path
        # access_token intentionally absent — never logged or serialised
        return meta
