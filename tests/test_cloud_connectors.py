"""
tests/test_cloud_connectors.py — Offline tests for Step 9 cloud connectors.

All network calls and SDK imports are mocked — no cloud accounts required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ingestion.databricks_source import DatabricksSource
from ingestion.bigquery_source import BigQuerySource
from ingestion.object_storage import (
    DriverMissingError,
    extract_from_uri,
    fetch_to_cache,
    parse_uri,
)


# ===========================================================================
# parse_uri
# ===========================================================================

class TestParseURI:
    """parse_uri() must handle all supported schemes and reject unknowns."""

    def test_s3_uri(self) -> None:
        result = parse_uri("s3://my-bucket/path/to/file.parquet")
        assert result["scheme"] == "s3"
        assert result["bucket"] == "my-bucket"
        assert result["key"] == "path/to/file.parquet"

    def test_gs_uri(self) -> None:
        result = parse_uri("gs://gcs-bucket/data/report.csv")
        assert result["scheme"] == "gs"
        assert result["bucket"] == "gcs-bucket"
        assert result["key"] == "data/report.csv"

    def test_az_uri(self) -> None:
        result = parse_uri("az://mycontainer/blobs/file.xlsx")
        assert result["scheme"] == "az"
        assert result["bucket"] == "mycontainer"
        assert result["key"] == "blobs/file.xlsx"

    def test_uppercase_scheme_normalised(self) -> None:
        result = parse_uri("S3://bucket/key.csv")
        assert result["scheme"] == "s3"

    def test_unknown_scheme_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised"):
            parse_uri("ftp://server/file.csv")

    def test_missing_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            parse_uri("s3://bucket-only")

    def test_garbage_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            parse_uri("not-a-uri-at-all")


# ===========================================================================
# DriverMissingError
# ===========================================================================

class TestDriverMissingError:
    """DriverMissingError must carry the engine_id the UI needs."""

    def test_carries_engine_id(self) -> None:
        exc = DriverMissingError("object_storage")
        assert exc.engine_id == "object_storage"

    def test_is_exception(self) -> None:
        assert issubclass(DriverMissingError, Exception)

    def test_message_contains_engine_id(self) -> None:
        exc = DriverMissingError("bigquery")
        assert "bigquery" in str(exc)


# ===========================================================================
# fetch_to_cache (S3)
# ===========================================================================

class TestFetchToCache:
    """fetch_to_cache() with a monkeypatched fake client writes a temp file."""

    def test_s3_writes_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Monkeypatching boto3 to write a small file, then verify path returned."""
        dest_file: list[Path] = []

        def fake_download_file(bucket, key, dest, **kwargs):
            Path(dest).write_bytes(b"fake,data\n1,2\n")
            dest_file.append(Path(dest))

        mock_client = MagicMock()
        mock_client.download_file.side_effect = fake_download_file
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Redirect cache dir to tmp_path for this test
        monkeypatch.setattr("ingestion.object_storage.CACHE_DIR", tmp_path)

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = fetch_to_cache("s3://test-bucket/folder/data.csv")

        assert result.exists()
        assert result.name.startswith("s3__")
        assert result.read_bytes() == b"fake,data\n1,2\n"

    def test_s3_missing_driver_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When boto3 is not installed, DriverMissingError must be raised."""
        monkeypatch.setattr("ingestion.object_storage.CACHE_DIR", tmp_path)
        # Remove boto3 from sys.modules to simulate missing driver
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises((DriverMissingError, ImportError)):
                fetch_to_cache("s3://bucket/file.csv")


# ===========================================================================
# extract_from_uri round-trip (parquet fixture)
# ===========================================================================

class TestExtractFromURI:
    """extract_from_uri fetches then delegates to FileSource."""

    def test_round_trip_with_parquet(
        self, fixture_parquet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Monkeypatch fetch_to_cache to return the local parquet fixture,
        then verify extract_from_uri returns a non-empty DataFrame."""
        monkeypatch.setattr(
            "ingestion.object_storage.fetch_to_cache",
            lambda uri, **auth: fixture_parquet,
        )
        from ingestion.file_source import FileSource
        df = extract_from_uri("s3://bucket/fixture.parquet", FileSource())
        assert len(df) == 20  # fixture has 20 rows
        assert not df.empty


# ===========================================================================
# DatabricksSource
# ===========================================================================

class TestDatabricksSource:
    """DatabricksSource tests using stub cursor/connection."""

    def test_connect_false_when_driver_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        src = DatabricksSource()
        src.configure("host.databricks.com", "/sql/1.0/warehouses/abc", "dapi_token")
        with caplog.at_level("WARNING"):
            assert src.connect() is False
        assert any("not installed" in r.getMessage().lower() for r in caplog.records)

    def test_connect_does_not_raise_when_driver_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        src = DatabricksSource()
        src.configure("h", "/p", "token")
        src.connect()  # must not raise

    def test_extract_with_stub_cursor_arrow_path(self) -> None:
        """Test extract() using a duck-typed stub with fetchall_arrow."""
        stub_df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})

        mock_arrow_table = MagicMock()
        mock_arrow_table.to_pandas.return_value = stub_df

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall_arrow.return_value = mock_arrow_table

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        src = DatabricksSource()
        src._DatabricksSource__access_token = "token"  # bypass configure
        src._connection = mock_conn

        df = src.extract("SELECT * FROM t")
        assert len(df) == 3
        assert "col_a" in df.columns

    def test_extract_with_stub_cursor_fallback_path(self) -> None:
        """Test extract() using a duck-typed stub WITHOUT fetchall_arrow."""
        mock_cursor = MagicMock(spec=["execute", "fetchall", "description",
                                      "__enter__", "__exit__"])
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [(1, "a"), (2, "b")]
        mock_cursor.description = [("id", None), ("name", None)]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        src = DatabricksSource()
        src._DatabricksSource__access_token = "token"
        src._connection = mock_conn

        df = src.extract("SELECT id, name FROM t")
        assert list(df.columns) == ["id", "name"]
        assert len(df) == 2

    def test_access_token_not_in_metadata(self) -> None:
        src = DatabricksSource()
        src.configure("host", "/path", "supersecret_dapi_token")
        meta = src.get_metadata()
        assert "supersecret_dapi_token" not in str(meta)

    def test_access_token_not_in_logs(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        src = DatabricksSource()
        src.configure("host", "/path", "supersecret_dapi_token")
        with caplog.at_level("DEBUG"):
            src.connect()
        for record in caplog.records:
            assert "supersecret_dapi_token" not in record.getMessage()


# ===========================================================================
# BigQuerySource
# ===========================================================================

class TestBigQuerySource:
    """BigQuerySource tests using a stub client."""

    def test_connect_false_when_driver_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        src = BigQuerySource()
        src.configure("my-project")
        with caplog.at_level("WARNING"):
            assert src.connect() is False
        assert any("not installed" in r.getMessage().lower() for r in caplog.records)

    def test_extract_with_stub_client(self) -> None:
        """Test extract() using a duck-typed BigQuery client stub."""
        stub_df = pd.DataFrame({"region": ["US", "EU"], "revenue": [100.0, 200.0]})

        mock_job = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dataframe.return_value = stub_df
        mock_job.result.return_value = mock_result

        mock_client = MagicMock()
        mock_client.query.return_value = mock_job

        src = BigQuerySource()
        src._project_id = "my-project"
        src._client = mock_client

        df = src.extract("SELECT region, revenue FROM dataset.table")
        assert len(df) == 2
        assert "region" in df.columns

    def test_extract_empty_query_returns_empty(self) -> None:
        src = BigQuerySource()
        src._project_id = "proj"
        src._client = MagicMock()
        df = src.extract(query="")
        assert df.empty

    def test_extract_not_connected_returns_empty(self) -> None:
        src = BigQuerySource()
        df = src.extract(query="SELECT 1")
        assert df.empty

    def test_credentials_path_not_in_metadata(self) -> None:
        src = BigQuerySource()
        src.configure("my-project", credentials_path="/secret/creds.json")
        meta = src.get_metadata()
        assert "/secret/creds.json" not in str(meta)
        assert "credentials_path" not in meta
