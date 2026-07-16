"""
core/driver_manager.py — Database driver registry and on-demand installer.

Each engine has exact pinned versions verified on Python 3.14.3 / Windows 11.
NEVER substitute or upgrade these versions without re-testing on this machine.

Design principles:
  - All driver imports are LAZY (inside functions) — importing this module must
    not crash on machines that don't have pyodbc, oracledb, etc. installed.
  - Pinned-only install policy — `pip install` always uses exact pins from this
    file, never "latest", to guarantee reproducibility.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Optional
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Driver specification table
# ---------------------------------------------------------------------------

DRIVER_SPECS: dict[str, dict] = {
    "sqlite": {
        "label": "SQLite",
        "pip": [],              # sqlite3 is part of the Python standard library
        "import_names": ["sqlite3"],
        "url_template": "sqlite:///{database}",
        "default_port": None,
    },
    "sqlserver": {
        "label": "SQL Server",
        "pip": ["pyodbc==5.3.0"],
        "import_names": ["pyodbc"],
        # {odbc_driver} is filled by pick_sqlserver_odbc_driver() at connect-time
        "url_template": (
            "mssql+pyodbc://{user}:{password}@{host},{port}/{database}"
            "?driver={odbc_driver}"
        ),
        "default_port": 1433,
    },
    "postgres": {
        "label": "PostgreSQL",
        "pip": ["pg8000==1.31.5"],
        "import_names": ["pg8000"],
        "url_template": (
            "postgresql+pg8000://{user}:{password}@{host}:{port}/{database}"
        ),
        "default_port": 5432,
    },
    "redshift": {
        # Redshift speaks the PostgreSQL wire protocol — we reuse pg8000.
        # Port is 5439 (Redshift default) instead of the standard 5432.
        "label": "Amazon Redshift",
        "pip": ["pg8000==1.31.5"],
        "import_names": ["pg8000"],
        "url_template": (
            "postgresql+pg8000://{user}:{password}@{host}:{port}/{database}"
        ),
        "default_port": 5439,
    },
    "mysql": {
        "label": "MySQL / MariaDB",
        # cryptography is REQUIRED for MySQL 8's caching_sha2_password authentication.
        # Without it, the handshake fails with an opaque SSL error.
        "pip": ["pymysql==1.2.0", "cryptography==49.0.0"],
        "import_names": ["pymysql"],
        "url_template": (
            "mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        ),
        "default_port": 3306,
    },
    "oracle": {
        "label": "Oracle Database",
        # oracledb thin mode — no Oracle Instant Client installation required.
        # The driver connects natively using a pure-Python implementation.
        "pip": ["oracledb==4.0.1"],
        "import_names": ["oracledb"],
        "url_template": (
            "oracle+oracledb://{user}:{password}@{host}:{port}"
            "/?service_name={database}"
        ),
        "default_port": 1521,
    },
    "snowflake": {
        "label": "Snowflake",
        "pip": [
            "snowflake-connector-python==4.6.0",
            "snowflake-sqlalchemy==1.11.0",
        ],
        "import_names": ["snowflake.connector"],
        # account/schema/warehouse are Snowflake-specific — substituted by build_url()
        "url_template": (
            "snowflake://{user}:{password}@{account}/{database}/{schema}"
            "?warehouse={warehouse}"
        ),
        "default_port": None,
    },
    # ------------------------------------------------------------------
    # Cloud / big-data tier (url_template=None — not SQLAlchemy engines)
    # ------------------------------------------------------------------
    "databricks": {
        "label": "Databricks SQL Warehouse",
        "pip": ["databricks-sql-connector==4.3.0"],
        "import_names": ["databricks.sql"],
        "url_template": None,   # uses native DBAPI connector, not SQLAlchemy
        "default_port": 443,
    },
    "bigquery": {
        # BigQuery is its own heavyweight tier (~13 Google packages incl. grpcio).
        # It is deliberately separate from the connectors tier so users can opt in
        # independently without pulling in all of Google's dependencies.
        "label": "Google BigQuery",
        "pip": ["google-cloud-bigquery==3.42.2", "db-dtypes==1.7.1"],
        "import_names": ["google.cloud.bigquery", "db_dtypes"],
        "url_template": None,   # uses google-cloud-bigquery client, not SQLAlchemy
        "default_port": None,
    },
    "object_storage": {
        "label": "Object Storage (S3 / Azure Blob / GCS)",
        "pip": [
            "boto3==1.43.45",
            "azure-storage-blob==12.30.0",
            "google-cloud-storage==3.12.1",
        ],
        "import_names": ["boto3", "azure.storage.blob", "google.cloud.storage"],
        "url_template": None,   # URI-based fetch, not a database connection
        "default_port": None,
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_driver_installed(engine_id: str) -> bool:
    """Return True if every required import for *engine_id* is available.

    Uses importlib.util.find_spec so no actual import side-effects occur.
    Handles dotted module paths (e.g. "snowflake.connector", "google.cloud.bigquery")
    where find_spec raises ModuleNotFoundError if the top-level package is absent.

    Args:
        engine_id: Key from DRIVER_SPECS (e.g. "postgres", "mysql").

    Returns:
        True if all import_names can be found in the current environment.
    """
    spec = DRIVER_SPECS.get(engine_id)
    if spec is None:
        return False
    for name in spec["import_names"]:
        try:
            found = importlib.util.find_spec(name)
        except (ModuleNotFoundError, ValueError):
            # find_spec raises ModuleNotFoundError when a dotted submodule's
            # parent package is missing (e.g. "snowflake.connector" without
            # snowflake installed).
            return False
        if found is None:
            return False
    return True


def install_driver(engine_id: str) -> tuple[bool, str]:
    """Install the pinned pip packages for *engine_id* into the current venv.

    Pinned-only policy: version strings come from DRIVER_SPECS["pip"] and are
    never overridden with "latest" or relaxed constraints.  This guarantees that
    the exact combination tested on Python 3.14.3 / Windows 11 is installed.

    Args:
        engine_id: Key from DRIVER_SPECS.

    Returns:
        Tuple of (success: bool, output: str).
        output contains the last 30 lines of pip's stdout+stderr.
    """
    spec = DRIVER_SPECS.get(engine_id)
    if spec is None:
        return False, f"Unknown engine: {engine_id!r}"

    pins = spec["pip"]
    if not pins:
        # sqlite3 is stdlib — nothing to install
        return True, "No packages to install (stdlib driver)."

    cmd = [sys.executable, "-m", "pip", "install", *pins]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        combined = (result.stdout + result.stderr).strip()
        last_30 = "\n".join(combined.splitlines()[-30:])
        success = result.returncode == 0
        return success, last_30
    except subprocess.TimeoutExpired:
        return False, "pip install timed out after 300 seconds"
    except Exception as exc:  # noqa: BLE001
        return False, f"install_driver failed: {exc}"


def pick_sqlserver_odbc_driver() -> Optional[str]:
    """Return the best available SQL Server ODBC driver name, URL-encoded.

    Preference order:
        1. ODBC Driver 18 for SQL Server  (modern TLS, recommended for Azure)
        2. ODBC Driver 17 for SQL Server
        3. SQL Server                      (legacy — lacks modern TLS; fine for
                                            local/dev demos only)

    Returns:
        URL-encoded driver name (spaces → +) or None if pyodbc is not available
        or no matching driver is found.
    """
    try:
        import pyodbc  # lazy — pyodbc may not be installed  # noqa: PLC0415
    except ImportError:
        return None

    available = pyodbc.drivers()
    for preferred in [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server",
    ]:
        if preferred in available:
            return quote_plus(preferred)  # spaces → +
    return None
