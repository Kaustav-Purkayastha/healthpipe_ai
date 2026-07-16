"""
ingestion/file_source.py — Local-file connector for HealthPipe AI v2.

Supports CSV, TSV, JSON (records-orient), Parquet, and XLSX.
Returns an empty DataFrame on any failure — never raises — so the
pipeline can treat "nothing ingested" uniformly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from core.utils import get_logger
from ingestion.base_source import BaseSource

_log = get_logger(__name__)

# Encoding fallback chain for text-based formats (CSV/TSV).
# WHY: government/health data files are frequently published in legacy
# encodings (latin-1, cp1252) rather than UTF-8.  Trying each in order
# is more reliable than guessing or using errors="replace" which silently
# corrupts values.
_TEXT_ENCODINGS: list[str] = ["utf-8", "latin-1", "cp1252"]


class FileSource(BaseSource):
    """Reads local files (CSV/TSV/JSON/Parquet/XLSX) into a DataFrame.

    ``connect()`` always returns True because local files need no
    external connection.  ``extract()`` dispatches on the file's suffix
    and logs a clear error on failure, returning an empty DataFrame.
    """

    source_type: str = "file"

    def __init__(self) -> None:
        """Initialise with the fixed registry name and a short description."""
        super().__init__(
            name="file",
            description="Local file reader — CSV, TSV, JSON, Parquet, XLSX",
        )

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Local files require no connection — always ready.

        Returns:
            True unconditionally.
        """
        return True

    def extract(self, filepath: str = "", **kwargs) -> pd.DataFrame:
        """Load a local file into a DataFrame.

        Args:
            filepath:   Absolute or relative path to the file.
            **kwargs:   Forwarded to the underlying pandas reader.
                        Notable extra keys:
                          chunk_size (int) — for CSV/TSV: read in chunks to
                              keep memory bounded on large files.
                          sheet_name (str|int) — for XLSX: which sheet to read.

        Returns:
            Loaded DataFrame, or empty DataFrame on any error (file not found,
            unsupported format, parse error, etc.).
        """
        path = Path(filepath)

        if not path.exists():
            _log.error("FileSource: file not found — %s", path)
            return pd.DataFrame()

        suffix = path.suffix.lower()

        if suffix in {".csv", ".tsv"}:
            return self._read_delimited(path, **kwargs)
        elif suffix == ".json":
            return self._read_json(path, **kwargs)
        elif suffix == ".parquet":
            return self._read_parquet(path, **kwargs)
        elif suffix == ".xlsx":
            return self._read_excel(path, **kwargs)
        else:
            _log.error(
                "FileSource: unsupported file suffix '%s' for file %s",
                suffix,
                path,
            )
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Private readers
    # ------------------------------------------------------------------

    def _read_delimited(self, path: Path, **kwargs) -> pd.DataFrame:
        """Read a CSV or TSV file with a multi-encoding fallback chain.

        The separator is inferred automatically by pandas (sep=None +
        engine="python") so both CSV and TSV are handled transparently.

        Args:
            path:     Path to the delimited text file.
            **kwargs: Passed to pd.read_csv.  ``chunk_size`` is extracted
                      before forwarding; all other kwargs are forwarded.

        Returns:
            Loaded DataFrame, or empty DataFrame on failure.
        """
        # Extract our custom kwarg before forwarding the rest to pandas.
        chunk_size: Optional[int] = kwargs.pop("chunk_size", None)

        for encoding in _TEXT_ENCODINGS:
            try:
                if chunk_size is not None:
                    # WHY chunk_size: keeps memory bounded on large government
                    # datasets (e.g. CMS 9.78M-row file saved locally).
                    chunks = pd.read_csv(
                        path,
                        sep=None,
                        engine="python",
                        encoding=encoding,
                        chunksize=chunk_size,
                        **kwargs,
                    )
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(
                        path,
                        sep=None,
                        engine="python",
                        encoding=encoding,
                        **kwargs,
                    )
                _log.debug(
                    "FileSource: loaded %s rows from %s (encoding=%s)",
                    len(df),
                    path.name,
                    encoding,
                )
                self._record_extract(df)
                return df
            except UnicodeDecodeError:
                # This encoding didn't work; try the next one in the chain.
                _log.debug(
                    "FileSource: encoding '%s' failed for %s, trying next",
                    encoding,
                    path.name,
                )
                continue
            except Exception as exc:
                _log.error("FileSource: error reading %s — %s", path, exc)
                return pd.DataFrame()

        # All encodings exhausted without success.
        _log.error("FileSource: all encodings failed for %s", path)
        return pd.DataFrame()

    def _read_json(self, path: Path, **kwargs) -> pd.DataFrame:
        """Read a JSON file, trying records orient first then pandas default.

        Args:
            path:     Path to the JSON file.
            **kwargs: Forwarded to pd.read_json.

        Returns:
            Loaded DataFrame, or empty DataFrame on failure.
        """
        try:
            df = pd.read_json(path, orient="records", **kwargs)
        except ValueError:
            # orient="records" failed (e.g. dict-of-lists or other shapes);
            # fall back to pandas' own orient detection.
            try:
                df = pd.read_json(path, **kwargs)
            except Exception as exc:
                _log.error("FileSource: error reading JSON %s — %s", path, exc)
                return pd.DataFrame()
        except Exception as exc:
            _log.error("FileSource: error reading JSON %s — %s", path, exc)
            return pd.DataFrame()

        _log.debug("FileSource: loaded %s rows from %s (JSON)", len(df), path.name)
        self._record_extract(df)
        return df

    def _read_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        """Read a Parquet file using the pyarrow engine.

        Args:
            path:     Path to the Parquet file.
            **kwargs: Forwarded to pd.read_parquet.

        Returns:
            Loaded DataFrame, or empty DataFrame on failure.
        """
        try:
            df = pd.read_parquet(path, engine="pyarrow", **kwargs)
        except Exception as exc:
            _log.error("FileSource: error reading Parquet %s — %s", path, exc)
            return pd.DataFrame()

        _log.debug("FileSource: loaded %s rows from %s (Parquet)", len(df), path.name)
        self._record_extract(df)
        return df

    def _read_excel(self, path: Path, **kwargs) -> pd.DataFrame:
        """Read an Excel workbook using the openpyxl engine.

        Args:
            path:       Path to the .xlsx file.
            **kwargs:   Forwarded to pd.read_excel.
                        Pass ``sheet_name`` to target a specific sheet.

        Returns:
            Loaded DataFrame, or empty DataFrame on failure.
        """
        try:
            df = pd.read_excel(path, engine="openpyxl", **kwargs)
        except Exception as exc:
            _log.error("FileSource: error reading Excel %s — %s", path, exc)
            return pd.DataFrame()

        _log.debug("FileSource: loaded %s rows from %s (XLSX)", len(df), path.name)
        self._record_extract(df)
        return df
