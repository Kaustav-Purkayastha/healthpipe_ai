"""
ingestion/database_source.py — Universal database connector via SQLAlchemy.

Supports SQLite, SQL Server, PostgreSQL, Redshift, MySQL, Oracle, Snowflake
through a single class.  The driver manager handles lazy installation of the
per-engine pip packages — this file never imports them at module level.

Security note: the password is stored only in instance memory and is NEVER
written to logs, metadata dicts, or any persistent storage.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote_plus

import pandas as pd

from core.driver_manager import DRIVER_SPECS, is_driver_installed, pick_sqlserver_odbc_driver
from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)

# Ping queries — Oracle requires FROM dual; all others accept SELECT 1.
_PING_SQL: dict[str, str] = {
    "oracle": "SELECT 1 FROM dual",
}
_DEFAULT_PING = "SELECT 1"


class DatabaseSource(BaseSource):
    """SQLAlchemy-backed connector for seven database engines.

    Usage (SQLite example)::

        src = DatabaseSource()
        src.configure("sqlite", database="data/sample/demo_clinic.db")
        if src.connect():
            print(src.list_tables())
            df = src.extract(table="claims")
    """

    source_type: str = "database"

    def __init__(self) -> None:
        """Initialise with default registry name; call configure() before use."""
        super().__init__(name="database", description="SQL database via SQLAlchemy")
        self._engine_id: Optional[str] = None
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._database: Optional[str] = None
        self._user: Optional[str] = None
        # WHY password is kept private: connection URLs and metadata dicts are
        # sometimes logged or displayed in the UI.  Keeping the password in a
        # name-mangled attribute ensures it never leaks into those paths.
        self.__password: Optional[str] = None
        self._account: Optional[str] = None    # Snowflake only
        self._schema: Optional[str] = None     # Snowflake only
        self._warehouse: Optional[str] = None  # Snowflake only
        self._engine = None  # sqlalchemy Engine, created lazily by connect()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        engine_id: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        account: Optional[str] = None,
        schema: Optional[str] = None,
        warehouse: Optional[str] = None,
    ) -> None:
        """Store connection parameters.  Password is kept only in memory.

        Args:
            engine_id:  One of the keys in DRIVER_SPECS ("sqlite", "postgres", …).
            host:       Server hostname or IP.
            port:       TCP port (defaults to engine's default_port if None).
            database:   Database/schema name; for SQLite, the file path.
            user:       Login username.
            password:   Login password — never logged or serialised.
            account:    Snowflake account identifier.
            schema:     Snowflake schema.
            warehouse:  Snowflake virtual warehouse.
        """
        self._engine_id = engine_id
        self._host = host
        self._port = port or DRIVER_SPECS.get(engine_id, {}).get("default_port")
        self._database = database
        self._user = user
        self.__password = password
        self._account = account
        self._schema = schema
        self._warehouse = warehouse
        self._engine = None  # invalidate any previous connection

    # ------------------------------------------------------------------
    # URL builder
    # ------------------------------------------------------------------

    def build_url(self) -> str:
        """Construct the SQLAlchemy connection URL from stored config.

        Returns:
            Connection URL string.

        Raises:
            ValueError: If engine_id is not configured or unknown.
        """
        if not self._engine_id:
            raise ValueError("DatabaseSource: call configure() before build_url()")

        spec = DRIVER_SPECS.get(self._engine_id)
        if spec is None:
            raise ValueError(f"Unknown engine_id: {self._engine_id!r}")

        template = spec["url_template"]

        # Encode password for safe URL embedding (special chars like @, : etc.)
        encoded_password = quote_plus(self.__password or "")
        encoded_user = quote_plus(self._user or "")

        if self._engine_id == "sqlite":
            # SQLite: file path only, no host/user/password
            return template.format(database=self._database or "")

        if self._engine_id == "sqlserver":
            odbc_driver = pick_sqlserver_odbc_driver() or "ODBC+Driver+18+for+SQL+Server"
            return template.format(
                user=encoded_user,
                password=encoded_password,
                host=self._host or "localhost",
                port=self._port or 1433,
                database=self._database or "",
                odbc_driver=odbc_driver,
            )

        if self._engine_id == "snowflake":
            return template.format(
                user=encoded_user,
                password=encoded_password,
                account=self._account or "",
                database=self._database or "",
                schema=self._schema or "PUBLIC",
                warehouse=self._warehouse or "",
            )

        # Generic pattern (postgres, redshift, mysql, oracle)
        return template.format(
            user=encoded_user,
            password=encoded_password,
            host=self._host or "localhost",
            port=self._port or spec.get("default_port") or 0,
            database=self._database or "",
        )

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open a database connection and run a dialect-aware ping query.

        Connection flow:
          1. Check driver is installed (importlib.util.find_spec).
          2. build_url() and create_engine() — engine creation is lazy in
             SQLAlchemy 2 and does not open a socket until engine.connect().
          3. engine.connect() opens the actual socket + dialect ping.

        Returns:
            True on successful ping, False on any error.
        """
        if not self._engine_id:
            _log.error("DatabaseSource.connect: call configure() first")
            return False

        # Guard: friendly message when driver package is missing
        if not is_driver_installed(self._engine_id):
            spec = DRIVER_SPECS.get(self._engine_id, {})
            label = spec.get("label", self._engine_id)
            pins = spec.get("pip", [])
            _log.warning(
                "Driver not installed for %s. Install with: pip install %s",
                label,
                " ".join(pins) if pins else "(stdlib)",
            )
            return False

        try:
            # Lazy import — SQLAlchemy is required but engine drivers are not
            from sqlalchemy import create_engine, text  # noqa: PLC0415
            import sqlalchemy.exc  # noqa: PLC0415

            url = self.build_url()
            # create_engine is lazy — no socket opened here
            self._engine = create_engine(url)

            ping_sql = _PING_SQL.get(self._engine_id, _DEFAULT_PING)
            with self._engine.connect() as conn:
                conn.execute(text(ping_sql))

            _log.info(
                "DatabaseSource connected: engine=%s host=%s db=%s",
                self._engine_id,
                self._host,
                self._database,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            # Log engine+host+db only — never the URL (which contains the password)
            _log.error(
                "DatabaseSource connect failed: engine=%s host=%s db=%s — %s",
                self._engine_id,
                self._host,
                self._database,
                exc,
            )
            self._engine = None
            return False

    def extract(
        self,
        table: Optional[str] = None,
        query: Optional[str] = None,
        max_rows: int = 100_000,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch rows from a table or arbitrary SQL query.

        Exactly one of *table* or *query* must be supplied.

        For the table path, SQLAlchemy builds a server-side LIMIT so the
        database truncates the result before sending it over the wire.
        Tradeoff: pd.read_sql with a plain string works for ad-hoc queries
        but cannot push LIMIT to the server portably across dialects.

        Args:
            table:    Table name for a full-table SELECT.
            query:    Arbitrary SELECT statement.
            max_rows: Hard cap on returned rows.
            **kwargs: Ignored.

        Returns:
            DataFrame (empty on error, as per pipeline convention).
        """
        if self._engine is None:
            _log.error("DatabaseSource.extract: not connected — call connect() first")
            return pd.DataFrame()

        if table is None and query is None:
            _log.error("DatabaseSource.extract: supply table= or query=")
            return pd.DataFrame()

        try:
            from sqlalchemy import text, Table, MetaData, select  # noqa: PLC0415

            if table is not None:
                # Server-side limit using SQLAlchemy core — avoids pulling all rows
                meta = MetaData()
                tbl = Table(table, meta, autoload_with=self._engine)
                stmt = select(tbl).limit(max_rows)
                with self._engine.connect() as conn:
                    result = conn.execute(stmt)
                    df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
            else:
                # Ad-hoc query — use pd.read_sql for convenience
                df = pd.read_sql(text(query), self._engine)
                df = df.head(max_rows)

            self._record_extract(df)
            _log.info(
                "DatabaseSource extract: %d rows from %s",
                len(df),
                table or "query",
            )
            return df

        except Exception as exc:  # noqa: BLE001
            _log.error("DatabaseSource.extract failed: %s", exc)
            return pd.DataFrame()

    def list_tables(self) -> list[str]:
        """Return all table names in the connected database/schema.

        Returns:
            Sorted list of table name strings, or empty list on error.
        """
        if self._engine is None:
            _log.error("DatabaseSource.list_tables: not connected")
            return []

        try:
            from sqlalchemy import inspect  # noqa: PLC0415
            inspector = inspect(self._engine)
            return sorted(inspector.get_table_names())
        except Exception as exc:  # noqa: BLE001
            _log.error("DatabaseSource.list_tables failed: %s", exc)
            return []

    def get_metadata(self) -> dict:
        """Return source metadata — password is never included.

        Returns:
            Dict with engine, host, database, and base metadata fields.
            Password is deliberately omitted (security — see configure() docstring).
        """
        meta = super().get_metadata()
        meta["engine_id"] = self._engine_id
        meta["host"] = self._host
        meta["database"] = self._database
        # password intentionally absent
        return meta
