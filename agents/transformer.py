"""
transformer.py — Data Transformer Agent.

Applies a series of cleaning and standardization steps to a DataFrame:
    1. Standardize column names to snake_case
    2. Remove duplicate rows
    3. Auto-detect and convert types (numeric, datetime)
    4. Handle null values (median for numbers, "Unknown" for strings)
    5. Clean text columns (strip whitespace, lowercase low-cardinality)
    6. Add metadata columns (_loaded_at, _source)

Every step is logged in an audit trail (transform_log) so you can see
exactly what happened and when.
"""

import re
from datetime import datetime

import pandas as pd

from core.utils import get_logger

logger = get_logger(__name__)


class TransformerAgent:
    """
    Cleans and standardizes a DataFrame through a series of chained steps.

    Each step modifies the DataFrame in place and logs what it did.
    The full log is available via get_transform_summary().

    Usage:
        transformer = TransformerAgent()
        clean_df = transformer.run(df, "who_life_expectancy")
        log = transformer.get_transform_summary()
    """

    def __init__(self) -> None:
        """Initialize with an empty transform log."""
        # Each entry: {"step": 1, "action": "...", "detail": "...", "timestamp": "..."}
        self._transform_log: list[dict] = []
        self._step_counter: int = 0

    def run(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Run the full transformation pipeline.

        Steps execute in a fixed order — each one builds on the previous.
        A copy of the DataFrame is made first so the original is never modified.

        Args:
            df:           The raw DataFrame to transform.
            dataset_name: Used for logging and the _source metadata column.

        Returns:
            Cleaned and standardized DataFrame.
        """
        logger.info(
            f"Transforming dataset: '{dataset_name}' "
            f"({len(df)} rows, {len(df.columns)} cols)"
        )

        # Reset log for this run
        self._transform_log = []
        self._step_counter = 0

        # Work on a copy so the caller's original DataFrame is untouched
        df = df.copy()

        # Chain all transformation steps in order
        df = self.standardize_columns(df)
        df = self.remove_duplicates(df)
        df = self.convert_types(df)
        df = self.handle_nulls(df)
        df = self.clean_text_columns(df)
        df = self.add_metadata_columns(df, dataset_name)

        logger.info(
            f"Transformation complete: {len(df)} rows, "
            f"{len(df.columns)} cols, {len(self._transform_log)} steps"
        )
        return df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all column names to snake_case.

        "YearStart" → "year_start"
        "DataValueUnit" → "data_value_unit"
        "LocationAbbr" → "location_abbr"

        Args:
            df: DataFrame with original column names.

        Returns:
            DataFrame with snake_case column names.
        """
        original_names = list(df.columns)

        def to_snake_case(name: str) -> str:
            """Convert a single string to snake_case."""
            # Insert underscore before uppercase letters: "YearStart" → "Year_Start"
            s = re.sub(r"([A-Z])", r"_\1", str(name))
            # Replace non-alphanumeric characters with underscore
            s = re.sub(r"[^a-zA-Z0-9]", "_", s)
            # Collapse multiple underscores into one
            s = re.sub(r"_+", "_", s)
            # Strip leading/trailing underscores and lowercase everything
            return s.strip("_").lower()

        new_names = [to_snake_case(col) for col in df.columns]

        # Handle duplicate column names by appending _2, _3, etc.
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for name in new_names:
            if name in seen:
                seen[name] += 1
                deduped.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 1
                deduped.append(name)

        df.columns = deduped

        # Count how many columns actually changed
        changed = sum(
            1 for old, new in zip(original_names, df.columns) if old != new
        )

        self._log_step(
            "standardize_columns",
            f"Renamed {changed}/{len(df.columns)} columns to snake_case",
        )
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate rows.

        Args:
            df: DataFrame that may contain duplicate rows.

        Returns:
            DataFrame with duplicates removed.
        """
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)

        self._log_step(
            "remove_duplicates",
            f"Removed {removed} duplicate rows (before={before}, after={len(df)})",
        )
        return df

    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-detect and convert columns from object dtype to numeric or datetime.

        Uses pd.to_numeric() and pd.to_datetime() with errors='coerce' —
        values that can't be converted become NaN/NaT instead of raising errors.

        Args:
            df: DataFrame with potentially mistyped object columns.

        Returns:
            DataFrame with corrected dtypes.
        """
        numeric_converted = 0
        datetime_converted = 0

        for col in df.columns:
            if df[col].dtype != "object":
                # Already typed — skip
                continue

            # Try numeric conversion first
            numeric_result = pd.to_numeric(df[col], errors="coerce")
            # If >50% of non-null values converted successfully, keep it
            non_null = df[col].notna().sum()
            if non_null > 0:
                converted_count = numeric_result.notna().sum()
                # Compare against original non-null count
                original_non_null = df[col].notna().sum()
                if converted_count / original_non_null > 0.5:
                    df[col] = numeric_result
                    numeric_converted += 1
                    continue

            # Try datetime conversion — only if column name hints at dates
            date_hints = ["date", "time", "year", "month", "day", "created", "updated"]
            col_lower = col.lower()
            if any(hint in col_lower for hint in date_hints):
                try:
                    datetime_result = pd.to_datetime(df[col], errors="coerce")
                    # If >50% converted, keep it
                    if datetime_result.notna().sum() / max(non_null, 1) > 0.5:
                        df[col] = datetime_result
                        datetime_converted += 1
                except (ValueError, TypeError):
                    pass

        self._log_step(
            "convert_types",
            f"Converted {numeric_converted} columns to numeric, "
            f"{datetime_converted} to datetime",
        )
        return df

    def handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill null values based on column data type.

        Strategy:
            - Numeric columns: fill with the column's median
            - String/object columns: fill with "Unknown"
            - Datetime columns: left as NaT (no sensible default)

        Args:
            df: DataFrame with null values.

        Returns:
            DataFrame with nulls handled.
        """
        fills_applied = 0

        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue

            dtype_kind = df[col].dtype.kind

            if dtype_kind in ("i", "f"):
                # Numeric — fill with median (less sensitive to outliers than mean)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                fills_applied += 1
                logger.info(
                    f"  Filled {null_count} nulls in '{col}' with median={median_val:.2f}"
                )
            elif dtype_kind == "O":
                # Object/string — fill with "Unknown"
                df[col] = df[col].fillna("Unknown")
                fills_applied += 1
            # Datetime (kind='M') is left as NaT intentionally

        self._log_step(
            "handle_nulls",
            f"Applied null-fill strategy to {fills_applied} columns",
        )
        return df

    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean string columns by stripping whitespace and optionally lowercasing.

        Only lowercases columns with fewer than 100 unique values (categorical data).
        High-cardinality text (like names or addresses) is left as-is to preserve
        original casing.

        Args:
            df: DataFrame with string columns.

        Returns:
            DataFrame with cleaned text.
        """
        cleaned = 0

        for col in df.columns:
            if df[col].dtype != "object":
                continue

            # .str.strip() removes leading/trailing whitespace
            df[col] = df[col].astype(str).str.strip()
            cleaned += 1

            # Only lowercase low-cardinality columns (likely categorical)
            if df[col].nunique() < 100:
                df[col] = df[col].str.lower()

        self._log_step(
            "clean_text_columns",
            f"Stripped whitespace on {cleaned} text columns, "
            f"lowercased columns with < 100 unique values",
        )
        return df

    def add_metadata_columns(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """
        Add pipeline metadata columns to track data provenance.

        Adds:
            _loaded_at: timestamp when this transformation ran
            _source:    name of the source dataset

        Args:
            df:           DataFrame to annotate.
            dataset_name: Value for the _source column.

        Returns:
            DataFrame with two new columns added.
        """
        df["_loaded_at"] = datetime.now()
        df["_source"] = dataset_name

        self._log_step(
            "add_metadata_columns",
            f"Added _loaded_at and _source='{dataset_name}'",
        )
        return df

    def get_transform_summary(self) -> list[dict]:
        """
        Return the audit log of all transformation steps.

        Returns:
            List of dicts, each with step number, action, detail, and timestamp.
        """
        return self._transform_log

    def _log_step(self, action: str, detail: str) -> None:
        """
        Record a transformation step in the audit log.

        Args:
            action: Short name for the step (e.g., "remove_duplicates").
            detail: Description of what happened (e.g., "Removed 5 rows").
        """
        self._step_counter += 1
        entry = {
            "step": self._step_counter,
            "action": action,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        }
        self._transform_log.append(entry)
        logger.info(f"Step {self._step_counter}: {action} — {detail}")
