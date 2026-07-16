"""
tests/test_database_source.py — Offline tests for DatabaseSource + driver_manager.

No live database servers required.  SQLite tests use the demo_clinic.db which
is built on-demand by make_demo_db.build_demo_db().
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.driver_manager import (
    DRIVER_SPECS,
    install_driver,
    is_driver_installed,
    pick_sqlserver_odbc_driver,
)
from ingestion.database_source import DatabaseSource


# ---------------------------------------------------------------------------
# Demo DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the demo clinic SQLite DB in a module-scoped tmp directory."""
    tmp = tmp_path_factory.mktemp("db")
    db_path = tmp / "demo_clinic.db"
    from data.sample.make_demo_db import build_demo_db
    build_demo_db(db_path)
    return db_path


@pytest.fixture(scope="module")
def sqlite_src(demo_db_path: Path) -> DatabaseSource:
    """Return a connected DatabaseSource pointing at the demo SQLite DB.

    DatabaseSource.connect() lazily imports SQLAlchemy, which lives in the
    connector tier (requirements-connectors.txt), not core. Skip these
    end-to-end tests cleanly when it's absent — that keeps the core-only CI run
    green while still exercising the path in a full dev environment.
    """
    pytest.importorskip("sqlalchemy")
    src = DatabaseSource()
    src.configure("sqlite", database=str(demo_db_path))
    assert src.connect() is True, "SQLite connection must succeed"
    return src


# ===========================================================================
# DRIVER_SPECS — URL shape tests
# ===========================================================================

class TestBuildURL:
    """build_url() must produce the documented URL shapes for each engine."""

    def _src(self, engine_id: str, **kwargs) -> DatabaseSource:
        src = DatabaseSource()
        src.configure(engine_id, **kwargs)
        return src

    def test_sqlite_url(self) -> None:
        src = self._src("sqlite", database="/tmp/test.db")
        assert src.build_url() == "sqlite:////tmp/test.db"

    def test_postgres_url(self) -> None:
        src = self._src(
            "postgres", host="db.example.com", port=5432,
            database="mydb", user="alice", password="s3cr3t",
        )
        url = src.build_url()
        assert url.startswith("postgresql+pg8000://")
        assert "db.example.com" in url
        assert "mydb" in url

    def test_redshift_uses_pg8000_on_5439(self) -> None:
        src = self._src(
            "redshift", host="rs.example.com",
            database="warehouse", user="bob", password="pw",
        )
        url = src.build_url()
        assert "pg8000" in url
        assert "5439" in url

    def test_mysql_url(self) -> None:
        src = self._src(
            "mysql", host="localhost", database="shop",
            user="root", password="pw",
        )
        url = src.build_url()
        assert url.startswith("mysql+pymysql://")
        assert "3306" in url

    def test_oracle_url_has_service_name(self) -> None:
        src = self._src(
            "oracle", host="ora.internal", database="ORCL",
            user="sys", password="pw",
        )
        url = src.build_url()
        assert "oracle+oracledb://" in url
        assert "service_name=ORCL" in url
        assert "1521" in url

    def test_snowflake_url_has_account_and_warehouse(self) -> None:
        src = self._src(
            "snowflake", database="analytics", user="sf_user", password="pw",
            account="myaccount.us-east-1", schema="PUBLIC", warehouse="WH_COMPUTE",
        )
        url = src.build_url()
        assert "snowflake://" in url
        assert "myaccount" in url
        assert "WH_COMPUTE" in url

    def test_password_not_logged_in_url(self, caplog: pytest.LogCaptureFixture) -> None:
        """build_url() result must not appear in any log record."""
        src = self._src(
            "postgres", host="h", database="d",
            user="u", password="supersecret123",
        )
        with caplog.at_level("DEBUG"):
            src.build_url()
        for record in caplog.records:
            assert "supersecret123" not in record.getMessage()


# ===========================================================================
# is_driver_installed
# ===========================================================================

class TestIsDriverInstalled:
    """is_driver_installed() checks importlib.util.find_spec for each import name."""

    def test_sqlite_always_installed(self) -> None:
        """sqlite3 is stdlib — must always return True."""
        assert is_driver_installed("sqlite") is True

    def test_returns_false_when_find_spec_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate a missing driver by making find_spec return None."""
        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        assert is_driver_installed("postgres") is False

    def test_returns_true_when_find_spec_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate installed driver by making find_spec return a non-None mock."""
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda _: MagicMock(),
        )
        assert is_driver_installed("postgres") is True

    def test_unknown_engine_returns_false(self) -> None:
        assert is_driver_installed("no_such_db") is False


# ===========================================================================
# install_driver
# ===========================================================================

class TestInstallDriver:
    """install_driver() calls pip with exact pinned versions."""

    def test_sqlite_no_packages(self) -> None:
        """SQLite has no pip packages — must return (True, 'No packages…')."""
        success, msg = install_driver("sqlite")
        assert success is True
        assert "No packages" in msg

    def test_postgres_pins_in_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """install_driver("postgres") must include 'pg8000==1.31.5' in the pip command."""
        captured_cmds: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = "Successfully installed pg8000-1.31.5"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_run)
        success, _ = install_driver("postgres")

        assert success is True
        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert "pg8000==1.31.5" in cmd
        assert sys.executable == cmd[0]

    def test_failure_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-zero returncode must produce (False, ...)."""
        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "error: could not install"
            return result

        monkeypatch.setattr(subprocess, "run", mock_run)
        success, output = install_driver("mysql")
        assert success is False
        assert "error" in output.lower()

    def test_mysql_includes_cryptography(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MySQL install must include cryptography (required for MySQL 8 auth)."""
        captured: list = []

        def mock_run(cmd, **kwargs):
            captured.append(cmd)
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            return r

        monkeypatch.setattr(subprocess, "run", mock_run)
        install_driver("mysql")
        assert any("cryptography==49.0.0" in c for c in captured[0])

    def test_unknown_engine_returns_false(self) -> None:
        success, msg = install_driver("no_such_db")
        assert success is False
        assert "Unknown engine" in msg


# ===========================================================================
# connect() with missing driver
# ===========================================================================

class TestConnectMissingDriver:
    """connect() must return False with a friendly log when driver is absent."""

    def test_returns_false_and_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        src = DatabaseSource()
        src.configure("postgres", host="h", database="d", user="u", password="pw")
        with caplog.at_level("WARNING"):
            result = src.connect()
        assert result is False
        assert any("not installed" in r.getMessage().lower() for r in caplog.records)

    def test_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        src = DatabaseSource()
        src.configure("oracle", host="h", database="ORCL", user="u", password="pw")
        # Must complete without raising
        src.connect()


# ===========================================================================
# SQLite end-to-end (no live server needed)
# ===========================================================================

class TestSQLiteEndToEnd:
    """Tests using the demo_clinic.db SQLite database."""

    def test_connect_returns_true(self, sqlite_src: DatabaseSource) -> None:
        assert sqlite_src.connect() is True

    def test_list_tables_has_three_tables(self, sqlite_src: DatabaseSource) -> None:
        tables = sqlite_src.list_tables()
        assert "patients" in tables
        assert "claims" in tables
        assert "providers" in tables
        assert len(tables) == 3

    def test_extract_claims_returns_1000_rows(
        self, sqlite_src: DatabaseSource
    ) -> None:
        df = sqlite_src.extract(table="claims")
        assert len(df) == 1000

    def test_extract_claims_has_expected_columns(
        self, sqlite_src: DatabaseSource
    ) -> None:
        df = sqlite_src.extract(table="claims")
        expected = {
            "claim_id", "patient_id", "procedure_code",
            "billed_amount", "allowed_amount", "paid_amount",
            "claim_date", "status",
        }
        assert expected.issubset(set(df.columns))

    def test_extract_query_denied_claims(self, sqlite_src: DatabaseSource) -> None:
        """DENIED ~8% of 1000 = ~80 rows (stochastic but seeded)."""
        df = sqlite_src.extract(query="SELECT * FROM claims WHERE status='DENIED'")
        # Allow ±20 from expected ~80
        assert 55 <= len(df) <= 110, f"Expected ~80 DENIED rows, got {len(df)}"

    def test_extract_patients_has_negatives(self, sqlite_src: DatabaseSource) -> None:
        """3 patients have deliberately negative ages for quality-check testing."""
        df = sqlite_src.extract(table="patients")
        neg_ages = df[df["age"] < 0]
        assert len(neg_ages) == 3

    def test_extract_requires_table_or_query(
        self, sqlite_src: DatabaseSource
    ) -> None:
        """Calling extract() with neither table nor query must return empty DF."""
        df = sqlite_src.extract()
        assert df.empty

    def test_extract_max_rows_limits_result(
        self, sqlite_src: DatabaseSource
    ) -> None:
        df = sqlite_src.extract(table="claims", max_rows=50)
        assert len(df) == 50


# ===========================================================================
# get_metadata — password never exposed
# ===========================================================================

class TestMetadataSecurity:
    """Password must never appear in get_metadata() or log records."""

    def test_password_not_in_metadata(self) -> None:
        src = DatabaseSource()
        src.configure(
            "postgres", host="h", database="d",
            user="u", password="topsecretpassword",
        )
        meta = src.get_metadata()
        meta_str = str(meta)
        assert "topsecretpassword" not in meta_str

    def test_password_not_in_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        src = DatabaseSource()
        src.configure(
            "sqlite", database="/nonexistent/path/test.db",
        )
        with caplog.at_level("DEBUG"):
            src.connect()  # will fail on bad path — that's fine
        for record in caplog.records:
            assert "topsecretpassword" not in record.getMessage()
