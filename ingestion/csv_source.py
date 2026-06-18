"""
csv_source.py — Data source for reading local files (CSV, JSON, TSV).

Handles common file-reading headaches:
    - Auto-detects file type from extension
    - Tries multiple encodings if the first one fails (utf-8 → latin-1 → cp1252)
    - Supports chunked reading for large files (the U.S. Chronic Disease
      Indicators dataset is ~88 MB and may not fit in memory on smaller machines)
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from core.config import SAMPLE_DATA_DIR
from core.utils import get_logger
from ingestion.base_source import BaseSource

logger = get_logger(__name__)

# Encoding fallback order — try the most common encodings in sequence
# utf-8:    standard for modern files
# latin-1:  common in European/government datasets, never fails to decode
# cp1252:   Windows default, handles smart quotes and special characters
ENCODING_FALLBACKS: list[str] = ["utf-8", "latin-1", "cp1252"]

# Default path to the sample dataset shipped with the project.
# This is a 500-row sample of the U.S. Chronic Disease Indicators dataset.
# To run on the full ~309K-row dataset, download it from data.gov (see README)
# and pass it explicitly with --filepath.
DEFAULT_CSV_PATH: Path = SAMPLE_DATA_DIR / "chronic_disease_sample.csv"


class CSVSource(BaseSource):
    """
    Reads local data files (CSV, JSON, TSV) into DataFrames.

    Supports chunked reading for large files: instead of loading the entire
    file into memory at once, it reads N rows at a time and combines them.

    Usage:
        source = CSVSource()
        source.connect()   # checks that the default file exists
        df = source.extract(filepath="data/sample/myfile.csv")
    """

    def __init__(self) -> None:
        """Initialize with a fixed name and description."""
        super().__init__(
            name="csv",
            description="Local file reader (CSV, JSON, TSV)"
        )
        self._last_extract_time: str | None = None
        self._last_record_count: int = 0
        self._last_filepath: str | None = None

    def connect(self) -> bool:
        """
        Check that the default sample data directory exists.

        For file-based sources, "connecting" means verifying the directory
        is accessible — actual file existence is checked during extract().

        Returns:
            True if the sample data directory exists, False otherwise.
        """
        if SAMPLE_DATA_DIR.exists():
            logger.info(
                f"CSV source ready — sample directory exists: "
                f"{SAMPLE_DATA_DIR}"
            )
            return True
        else:
            logger.error(
                f"Sample data directory not found: {SAMPLE_DATA_DIR}"
            )
            return False

    def extract(
        self,
        filepath: str | None = None,
        encoding: str | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Read a local file into a DataFrame.

        Auto-detects file type from extension. If encoding is not specified,
        tries utf-8 first, then falls back to latin-1 and cp1252.

        Args:
            filepath:   Path to the file. Defaults to the U.S. Chronic
                        Disease Indicators CSV if not specified.
            encoding:   Force a specific encoding (e.g., "latin-1").
                        If None, tries encodings in fallback order.
            chunk_size: If set, reads the file in chunks of this many rows.
                        Useful for files too large to fit in memory at once.
                        None means read the entire file at once.

        Returns:
            DataFrame containing the file's data.
        """
        # Resolve the file path
        file_path = Path(filepath) if filepath else DEFAULT_CSV_PATH
        logger.info(
            f"Extracting from local file: {file_path} "
            f"(encoding={encoding}, chunk_size={chunk_size})"
        )

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()

        # Determine file type from extension: ".csv" → "csv", ".json" → "json"
        file_extension = file_path.suffix.lower().lstrip(".")

        # Route to the appropriate reader based on file type
        if file_extension == "json":
            df = self._read_json(file_path, encoding)
        elif file_extension in ("csv", "tsv"):
            df = self._read_delimited(
                file_path, file_extension, encoding, chunk_size
            )
        else:
            logger.error(
                f"Unsupported file extension: '.{file_extension}'. "
                f"Supported: .csv, .tsv, .json"
            )
            return pd.DataFrame()

        # Update metadata
        self._last_extract_time = datetime.now().isoformat()
        self._last_record_count = len(df)
        self._last_filepath = str(file_path)

        logger.info(
            f"CSV extraction complete: {len(df)} records "
            f"from '{file_path.name}'"
        )
        return df

    def get_metadata(self) -> dict:
        """
        Return a summary of this source and the most recent extraction.

        Returns:
            Dict with source info and last extraction stats.
        """
        return {
            "name": self.name,
            "description": self.description,
            "source_type": "file",
            "supported_formats": ["csv", "tsv", "json"],
            "default_path": str(DEFAULT_CSV_PATH),
            "last_extract_time": self._last_extract_time,
            "last_record_count": self._last_record_count,
            "last_filepath": self._last_filepath,
        }

    def _read_delimited(
        self,
        file_path: Path,
        file_type: str,
        encoding: str | None,
        chunk_size: int | None,
    ) -> pd.DataFrame:
        """
        Read a CSV or TSV file, handling encoding fallbacks and chunked reading.

        Args:
            file_path:  Path to the file.
            file_type:  "csv" or "tsv" — determines the delimiter.
            encoding:   Specific encoding to use, or None for auto-detection.
            chunk_size: Number of rows per chunk, or None to read all at once.

        Returns:
            DataFrame with the file's data.
        """
        # TSV uses tab as delimiter, CSV uses comma
        separator = "\t" if file_type == "tsv" else ","

        # Determine which encodings to try
        encodings_to_try = [encoding] if encoding else ENCODING_FALLBACKS

        for enc in encodings_to_try:
            try:
                if chunk_size:
                    # Chunked reading — for large files like the 88 MB dataset
                    df = self._read_in_chunks(
                        file_path, separator, enc, chunk_size
                    )
                else:
                    # Read entire file at once — fine for smaller files
                    # low_memory=False forces pandas to scan the entire column
                    # before guessing its type, which avoids mixed-type warnings
                    df = pd.read_csv(
                        file_path,
                        sep=separator,
                        encoding=enc,
                        low_memory=False,
                    )
                logger.info(
                    f"Successfully read '{file_path.name}' "
                    f"with encoding '{enc}'"
                )
                return df

            except UnicodeDecodeError:
                # This encoding can't handle the file's characters — try next
                logger.warning(
                    f"Encoding '{enc}' failed for '{file_path.name}' "
                    f"— trying next fallback"
                )
                continue
            except pd.errors.EmptyDataError:
                logger.warning(f"File is empty: {file_path}")
                return pd.DataFrame()
            except pd.errors.ParserError as e:
                logger.error(
                    f"Failed to parse '{file_path.name}': {e}"
                )
                return pd.DataFrame()

        # All encodings failed
        logger.error(
            f"Could not read '{file_path.name}' with any encoding: "
            f"{encodings_to_try}"
        )
        return pd.DataFrame()

    def _read_in_chunks(
        self,
        file_path: Path,
        separator: str,
        encoding: str,
        chunk_size: int,
    ) -> pd.DataFrame:
        """
        Read a large file in chunks and concatenate them into one DataFrame.

        This avoids loading the entire file into memory at once.
        For example, reading an 88 MB file in 50,000-row chunks means
        only ~5 MB is in memory per chunk.

        Args:
            file_path:  Path to the file.
            separator:  Column delimiter ("," for CSV, "\\t" for TSV).
            encoding:   Character encoding to use.
            chunk_size: Number of rows per chunk.

        Returns:
            Combined DataFrame from all chunks.
        """
        chunks: list[pd.DataFrame] = []
        chunk_count = 0

        # pd.read_csv with chunksize returns an iterator of DataFrames,
        # each containing chunk_size rows
        reader = pd.read_csv(
            file_path,
            sep=separator,
            encoding=encoding,
            chunksize=chunk_size,
            low_memory=False,
        )

        for chunk in reader:
            chunks.append(chunk)
            chunk_count += 1
            logger.info(
                f"Read chunk {chunk_count}: {len(chunk)} rows"
            )

        # pd.concat() stacks all chunks vertically into one DataFrame
        # ignore_index=True resets the row numbers to 0, 1, 2, ...
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def _read_json(
        self, file_path: Path, encoding: str | None
    ) -> pd.DataFrame:
        """
        Read a JSON file into a DataFrame.

        Handles both JSON arrays ([{...}, {...}]) and line-delimited JSON
        (one JSON object per line).

        Args:
            file_path: Path to the JSON file.
            encoding:  Specific encoding, or None for pandas default (utf-8).

        Returns:
            DataFrame with the parsed data.
        """
        try:
            # Try standard JSON array format first
            df = pd.read_json(file_path, encoding=encoding)
            return df
        except ValueError:
            try:
                # Fall back to line-delimited JSON (JSONL / NDJSON format)
                df = pd.read_json(
                    file_path, lines=True, encoding=encoding
                )
                return df
            except ValueError as e:
                logger.error(
                    f"Failed to parse JSON file '{file_path.name}': {e}"
                )
                return pd.DataFrame()
