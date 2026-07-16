"""
ingestion/object_storage.py — Cloud object storage fetch layer for HealthPipe AI v2.

Provides three public functions:
  parse_uri(uri)                  — parse s3://, gs://, az:// URIs into components.
  fetch_to_cache(uri, **auth)     — download to data/cache/ and return the local Path.
  extract_from_uri(uri, fs, **auth) — fetch then delegate to FileSource for format handling.

NOT a BaseSource subclass — this is a fetch layer that feeds the existing FileSource,
reusing all Step-2 format handling (CSV encoding fallback, Parquet, XLSX, etc.) for free.

All cloud SDK imports are LAZY (inside functions) so this module is importable even when
boto3, azure-storage-blob, and google-cloud-storage are not installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from core.config import CACHE_DIR
from core.utils import get_logger

if TYPE_CHECKING:
    from ingestion.file_source import FileSource

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DriverMissingError(Exception):
    """Raised when a required cloud storage driver is not installed.

    The UI maps this exception to the driver-install modal so the user can
    install the missing package with one click.

    Attributes:
        engine_id: The DRIVER_SPECS key that identifies the missing driver.
    """

    def __init__(self, engine_id: str) -> None:
        """Initialise with the missing engine identifier.

        Args:
            engine_id: e.g. ``"object_storage"``.
        """
        self.engine_id = engine_id
        super().__init__(
            f"Driver not installed for '{engine_id}'. "
            f"Install via: from core.driver_manager import install_driver; "
            f"install_driver('{engine_id}')"
        )


# ---------------------------------------------------------------------------
# URI parser
# ---------------------------------------------------------------------------

# Scheme → canonical name mapping used by fetch_to_cache dispatch
_SCHEME_MAP = {"s3": "s3", "gs": "gs", "az": "az"}


def parse_uri(uri: str) -> dict[str, str]:
    """Parse a cloud object storage URI into its components.

    Supported schemes:
      ``s3://bucket/key``             → S3 (any region / endpoint)
      ``gs://bucket/key``             → Google Cloud Storage
      ``az://container/blob_path``    → Azure Blob Storage

    Args:
        uri: The storage URI string.

    Returns:
        Dict with keys ``scheme``, ``bucket`` (or container), ``key`` (or blob).

    Raises:
        ValueError: If the URI does not match any supported scheme or is malformed.
    """
    match = re.match(r"^(s3|gs|az)://([^/]+)/(.+)$", uri, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Unrecognised object storage URI: {uri!r}. "
            "Expected s3://<bucket>/<key>, gs://<bucket>/<key>, "
            "or az://<container>/<blob>."
        )
    scheme = match.group(1).lower()
    bucket = match.group(2)
    key = match.group(3)
    return {"scheme": scheme, "bucket": bucket, "key": key}


# ---------------------------------------------------------------------------
# Fetch to local cache
# ---------------------------------------------------------------------------

def fetch_to_cache(uri: str, **auth: Any) -> Path:
    """Download a cloud object to ``data/cache/`` and return the local Path.

    The cached filename is derived from the object key (last path component),
    prefixed with the scheme for disambiguation.

    Auth keyword arguments:
      S3:  passed to ``boto3.client("s3", **auth)`` — standard boto3 kwargs
           (region_name, aws_access_key_id, aws_secret_access_key, etc.).
           When empty, boto3 uses the standard credential chain (env vars,
           ~/.aws/credentials, instance metadata).
      GCS: ``credentials_path`` optional; sets GOOGLE_APPLICATION_CREDENTIALS.
      Azure: ``connection_string`` OR ``account_url`` + ``sas_token``.

    Args:
        uri:  Cloud URI (s3://, gs://, az://).
        **auth: Provider-specific auth kwargs.

    Returns:
        Path to the downloaded local file.

    Raises:
        DriverMissingError: When the required cloud SDK is not installed.
        ValueError:         When the URI is malformed.
        Exception:          On download failure.
    """
    parsed = parse_uri(uri)
    scheme = parsed["scheme"]
    bucket = parsed["bucket"]
    key = parsed["key"]

    # Safe local filename: scheme__last_key_component
    safe_name = f"{scheme}__{Path(key).name}"
    dest = CACHE_DIR / safe_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if scheme == "s3":
        _fetch_s3(bucket, key, dest, **auth)
    elif scheme == "gs":
        _fetch_gcs(bucket, key, dest, **auth)
    elif scheme == "az":
        _fetch_azure(bucket, key, dest, **auth)

    _log.info("Object fetched from %s → %s", uri, dest)
    return dest


def _fetch_s3(bucket: str, key: str, dest: Path, **auth: Any) -> None:
    """Download from S3.  Auth via standard boto3 credential chain."""
    try:
        import boto3  # noqa: PLC0415
    except ImportError as exc:
        raise DriverMissingError("object_storage") from exc

    # boto3 uses the standard credential chain when auth kwargs are empty:
    # env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), ~/.aws/credentials,
    # IAM role, instance metadata.  Pass kwargs to override.
    client = boto3.client("s3", **auth)
    client.download_file(bucket, key, str(dest))


def _fetch_gcs(bucket: str, key: str, dest: Path, **auth: Any) -> None:
    """Download from Google Cloud Storage."""
    try:
        from google.cloud import storage as gcs  # noqa: PLC0415
    except ImportError as exc:
        raise DriverMissingError("object_storage") from exc

    import os  # noqa: PLC0415
    creds_path = auth.pop("credentials_path", None)
    if creds_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    client = gcs.Client(**auth)
    client.bucket(bucket).blob(key).download_to_filename(str(dest))


def _fetch_azure(container: str, blob_path: str, dest: Path, **auth: Any) -> None:
    """Download from Azure Blob Storage.

    Pass ``connection_string=...`` OR ``account_url=...`` + ``sas_token=...``.
    """
    try:
        from azure.storage.blob import BlobClient  # noqa: PLC0415
    except ImportError as exc:
        raise DriverMissingError("object_storage") from exc

    conn_str = auth.pop("connection_string", None)
    if conn_str:
        blob_client = BlobClient.from_connection_string(
            conn_str, container_name=container, blob_name=blob_path
        )
    else:
        account_url = auth.pop("account_url", "")
        sas = auth.pop("sas_token", "")
        url = f"{account_url.rstrip('/')}/{container}/{blob_path}?{sas}"
        blob_client = BlobClient.from_blob_url(url)

    with dest.open("wb") as fh:
        data = blob_client.download_blob()
        data.readinto(fh)


# ---------------------------------------------------------------------------
# High-level extractor
# ---------------------------------------------------------------------------

def extract_from_uri(
    uri: str,
    file_source: "FileSource",
    **auth: Any,
) -> pd.DataFrame:
    """Fetch a cloud object and load it via FileSource.

    All Step-2 format handling (CSV encoding fallback, Parquet pyarrow, XLSX
    openpyxl, JSON records) is reused automatically by delegating to FileSource.

    Args:
        uri:         Cloud URI (s3://, gs://, az://).
        file_source: An initialised FileSource instance.
        **auth:      Auth kwargs forwarded to fetch_to_cache().

    Returns:
        DataFrame as returned by FileSource.extract(), or empty DataFrame on error.
    """
    try:
        local_path = fetch_to_cache(uri, **auth)
        return file_source.extract(filepath=str(local_path))
    except DriverMissingError:
        raise  # let the UI handle the install modal
    except Exception as exc:  # noqa: BLE001
        _log.error("extract_from_uri failed for %s: %s", uri, exc)
        return pd.DataFrame()
