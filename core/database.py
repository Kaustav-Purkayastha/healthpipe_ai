"""
database.py — DuckDB database manager.

DuckDB is a serverless analytics database — it stores everything in a single
file (like SQLite) but is optimized for analytical queries (columnar storage,
vectorized execution).

Key feature: CREATE OR REPLACE TABLE makes operations idempotent — you can
run the pipeline 10 times and get the same result without errors.
"""

from pathlib import Path

import duckdb
import pandas as pd

from core.config import DATABASE_PATH
from core.utils import get_logger

logger = get_logger(__name__)


class DuckDBManager:
    """
    Manages a DuckDB database connection and provides helper methods.

    Usage:
        db = DuckDBManager()
        db.load_dataframe(df, "who_life_expectancy")
        result = db.query("SELECT * FROM who_life_expectancy LIMIT 10")
        db.close()

    Or as a context manager (auto-closes):
        with DuckDBManager() as db:
            db.load_dataframe(df, "my_table")
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """
        Connect to a DuckDB database file.

        If the file doesn't exist, DuckDB creates it automatically.

        Args:
            db_path: Path to the .duckdb file. Defaults to DATABASE_PATH
                     from config.py.
        """
        self._db_path = db_path or DATABASE_PATH
        # duckdb.connect() opens (or creates) the database file
        self._conn = duckdb.connect(str(self._db_path))
        logger.info(f"Connected to DuckDB: {self._db_path}")

    def load_dataframe(
        self, df: pd.DataFrame, table_name: str
    ) -> None:
        """
        Load a pandas DataFrame into a DuckDB table.

        Uses CREATE OR REPLACE TABLE so this is idempotent — running it
        multiple times just overwrites the table, no error.

        Args:
            df:         The DataFrame to load.
            table_name: Name for the DuckDB table.
        """
        # DuckDB can directly register a DataFrame and create a table from it.
        # We register it as a temporary view, then use CREATE OR REPLACE
        # to persist it as a real table.
        self._conn.register("_temp_df", df)
        self._conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _temp_df"
        )
        # Unregister the temporary view to free memory
        self._conn.unregister("_temp_df")

        row_count = self._conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]

        logger.info(
            f"Loaded {row_count} rows into table '{table_name}'"
        )

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.

        Args:
            sql: The SQL query string to execute.

        Returns:
            DataFrame containing the query results.
        """
        logger.info(f"Executing query: {sql[:100]}...")
        result = self._conn.execute(sql).fetchdf()
        logger.info(f"Query returned {len(result)} rows")
        return result

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """
        Get column-level information for a table (like DESCRIBE in SQL).

        Args:
            table_name: Name of the table to inspect.

        Returns:
            DataFrame with column names, types, and nullability.
        """
        # PRAGMA table_info is DuckDB's equivalent of DESCRIBE TABLE
        return self._conn.execute(
            f"PRAGMA table_info('{table_name}')"
        ).fetchdf()

    def execute(self, sql: str, params=None) -> None:
        """
        Run a SQL statement that doesn't return rows (DROP, CREATE, ALTER, etc).

        Use query() instead when you need the result back as a DataFrame.

        Args:
            sql:    The SQL statement to execute.
            params: Optional scalar or list of bind parameters for ? placeholders.
        """
        if params is not None:
            p = params if isinstance(params, (list, tuple)) else [params]
            self._conn.execute(sql, p)
        else:
            self._conn.execute(sql)

    def list_tables(self) -> list[str]:
        """
        List all tables in the database.

        Returns:
            List of table name strings.
        """
        # information_schema.tables is the SQL standard way to list tables
        result = self._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchdf()
        # .tolist() converts a single DataFrame column to a Python list
        return result["table_name"].tolist()

    def get_schema(self, table_name: str) -> list[dict]:
        """
        Return the schema of a table as a list of column dicts.

        Used by the NL\u2192SQL engine (Step 6) to understand table structure
        before building a query.

        Args:
            table_name: Name of the table to inspect.

        Returns:
            List of dicts with keys 'column_name' and 'column_type'.
        """
        # DESCRIBE returns column_name, column_type, null, key, default, extra
        result = self._conn.execute(f"DESCRIBE {table_name}").fetchdf()
        return result[["column_name", "column_type"]].to_dict(orient="records")

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info(f"DuckDB connection closed: {self._db_path}")

    def __enter__(self) -> "DuckDBManager":
        """Support 'with DuckDBManager() as db:' syntax."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Auto-close the connection when exiting a 'with' block."""
        self.close()

    def __repr__(self) -> str:
        """Developer-friendly string for debugging."""
        return f"<DuckDBManager(path='{self._db_path}')>"
