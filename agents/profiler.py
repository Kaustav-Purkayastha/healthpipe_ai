"""
profiler.py — Data Profiler Agent.

Analyzes a DataFrame and produces a comprehensive profile report:
    - Overview stats (rows, columns, memory, duplicates, completeness)
    - Per-column profiling (numeric stats, string patterns, datetime ranges)
    - Outlier detection using the IQR method
    - Correlation analysis (only strong correlations > 0.7)
    - Quality issue flags (high nulls, constant columns, possible IDs)

No AI involved — this is pure pandas/numpy analysis.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import (
    MAX_NULL_PERCENTAGE,
    OUTLIER_IQR_MULTIPLIER,
    REPORTS_DIR,
)
from core.utils import get_logger, save_json

logger = get_logger(__name__)


class ProfilerAgent:
    """
    Analyzes a DataFrame and produces a detailed statistical profile.

    Usage:
        profiler = ProfilerAgent()
        profile = profiler.run(df, "who_life_expectancy")
    """

    def run(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Run the full profiling pipeline on a DataFrame.

        Args:
            df:           The DataFrame to profile.
            dataset_name: Human-readable name for labeling the report.

        Returns:
            Dict containing overview, column profiles, quality issues,
            and correlations.
        """
        logger.info(f"Profiling dataset: '{dataset_name}' ({len(df)} rows)")

        profile = {
            "dataset_name": dataset_name,
            "profiled_at": datetime.now().isoformat(),
            "overview": self._get_overview(df),
            "columns": self._profile_columns(df),
            "quality_issues": self._detect_quality_issues(df),
            "correlations": self._get_correlations(df),
        }

        # Save the profile report as JSON
        output_path = REPORTS_DIR / f"profile_{dataset_name}.json"
        save_json(profile, output_path)
        logger.info(f"Profile saved to: {output_path}")

        return profile

    def _get_overview(self, df: pd.DataFrame) -> dict:
        """
        Compute high-level dataset statistics.

        Args:
            df: The DataFrame to summarize.

        Returns:
            Dict with row count, column count, memory usage, duplicate
            count/percentage, and overall completeness score.
        """
        total_cells = df.shape[0] * df.shape[1]
        # Count non-null cells across the entire DataFrame
        non_null_cells = df.notna().sum().sum()
        # Completeness = percentage of cells that have values
        completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0.0

        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = (duplicate_count / len(df) * 100) if len(df) > 0 else 0.0

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            # .memory_usage(deep=True) measures actual string content size,
            # not just the pointer size — gives accurate memory numbers
            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum() / (1024 * 1024), 2
            ),
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(duplicate_pct, 2),
            "completeness_score": round(completeness, 2),
        }

    def _profile_columns(self, df: pd.DataFrame) -> dict:
        """
        Generate per-column statistics based on each column's data type.

        Numeric columns get mean/median/std/quartiles/outliers.
        String columns get length stats and top values.
        Datetime columns get min/max date and range.

        Args:
            df: The DataFrame to profile.

        Returns:
            Dict mapping column name → column profile dict.
        """
        profiles = {}

        for col in df.columns:
            col_data = df[col]
            # .dtype.kind is a single character code:
            #   'i' = integer, 'f' = float, 'O' = object (string),
            #   'M' = datetime, 'b' = boolean
            dtype_kind = col_data.dtype.kind

            base_stats = {
                "dtype": str(col_data.dtype),
                "null_count": int(col_data.isna().sum()),
                "null_percentage": round(
                    col_data.isna().sum() / len(df) * 100, 2
                ) if len(df) > 0 else 0.0,
                "unique_count": int(col_data.nunique()),
            }

            if dtype_kind in ("i", "f"):
                # Integer or float — compute numeric statistics
                base_stats.update(self._profile_numeric(col_data))
            elif dtype_kind == "O":
                # Object dtype — usually strings
                base_stats.update(self._profile_string(col_data))
            elif dtype_kind == "M":
                # Datetime
                base_stats.update(self._profile_datetime(col_data))

            profiles[col] = base_stats

        return profiles

    def _profile_numeric(self, series: pd.Series) -> dict:
        """
        Compute statistics for a numeric column.

        Includes quartiles and IQR-based outlier detection.
        A value is an outlier if it falls below Q1 - 1.5*IQR
        or above Q3 + 1.5*IQR.

        Args:
            series: A numeric pandas Series.

        Returns:
            Dict with mean, median, std, min, max, quartiles, and outlier count.
        """
        # .dropna() removes NaN before computing stats — NaN would
        # make most numpy functions return NaN
        clean = series.dropna()

        if len(clean) == 0:
            return {"profile_type": "numeric", "note": "all values are null"}

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1  # Interquartile Range

        # IQR outlier boundaries
        lower_bound = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper_bound = q3 + OUTLIER_IQR_MULTIPLIER * iqr

        # Count values outside the IQR fences
        outlier_count = int(
            ((clean < lower_bound) | (clean > upper_bound)).sum()
        )

        return {
            "profile_type": "numeric",
            "mean": round(float(clean.mean()), 4),
            "median": round(float(clean.median()), 4),
            "std": round(float(clean.std()), 4),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "outlier_count": outlier_count,
            # Skewness: 0 = symmetric, >0 = right-skewed, <0 = left-skewed
            "skewness": round(float(clean.skew()), 4),
        }

    def _profile_string(self, series: pd.Series) -> dict:
        """
        Compute statistics for a string/object column.

        Args:
            series: A string-type pandas Series.

        Returns:
            Dict with length stats, top values, empty count, and ID detection.
        """
        # .dropna().astype(str) converts everything to string safely
        clean = series.dropna().astype(str)

        if len(clean) == 0:
            return {"profile_type": "string", "note": "all values are null"}

        lengths = clean.str.len()

        # .value_counts().head(5) gives the 5 most frequent values
        top_values = clean.value_counts().head(5)

        # Heuristic to detect ID-like columns:
        # if almost every value is unique, it's probably an identifier
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        looks_like_id = unique_ratio > 0.95

        return {
            "profile_type": "string",
            "avg_length": round(float(lengths.mean()), 2),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "empty_string_count": int((clean == "").sum()),
            "top_5_values": {
                str(k): int(v) for k, v in top_values.items()
            },
            "looks_like_id": looks_like_id,
        }

    def _profile_datetime(self, series: pd.Series) -> dict:
        """
        Compute statistics for a datetime column.

        Args:
            series: A datetime pandas Series.

        Returns:
            Dict with min/max dates and the range in days.
        """
        clean = series.dropna()

        if len(clean) == 0:
            return {"profile_type": "datetime", "note": "all values are null"}

        min_date = clean.min()
        max_date = clean.max()
        # .days extracts the integer number of days from a timedelta
        range_days = (max_date - min_date).days

        return {
            "profile_type": "datetime",
            "min_date": str(min_date),
            "max_date": str(max_date),
            "range_days": int(range_days),
        }

    def _detect_quality_issues(self, df: pd.DataFrame) -> list[dict]:
        """
        Scan for common data quality problems.

        Checks for:
            - High null percentages (>50% = critical, >20% = warning)
            - Constant columns (only 1 unique value — useless for analysis)
            - Possible ID columns (>95% unique values)

        Args:
            df: The DataFrame to check.

        Returns:
            List of issue dicts, each with column, issue type, and severity.
        """
        issues: list[dict] = []

        for col in df.columns:
            null_pct = df[col].isna().sum() / len(df) * 100 if len(df) > 0 else 0

            # High null percentage
            if null_pct > 50:
                issues.append({
                    "column": col,
                    "issue": "high_null_rate",
                    "severity": "critical",
                    "detail": f"{null_pct:.1f}% null",
                })
            elif null_pct > MAX_NULL_PERCENTAGE:
                issues.append({
                    "column": col,
                    "issue": "high_null_rate",
                    "severity": "warning",
                    "detail": f"{null_pct:.1f}% null",
                })

            # Constant column (only 1 unique non-null value)
            if df[col].nunique() == 1:
                issues.append({
                    "column": col,
                    "issue": "constant_column",
                    "severity": "info",
                    "detail": f"Only value: {df[col].dropna().iloc[0]}"
                    if not df[col].dropna().empty
                    else "Only nulls",
                })

            # Possible ID column
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if unique_ratio > 0.95 and df[col].dtype == "object":
                issues.append({
                    "column": col,
                    "issue": "possible_id_column",
                    "severity": "info",
                    "detail": f"{df[col].nunique()} unique out of {len(df)} rows",
                })

        logger.info(f"Found {len(issues)} quality issues")
        return issues

    def _get_correlations(self, df: pd.DataFrame) -> list[dict]:
        """
        Find strong correlations (|r| > 0.7) between numeric columns.

        Only reports each pair once (A↔B, not both A→B and B→A).

        Args:
            df: The DataFrame to analyze.

        Returns:
            List of dicts with column_1, column_2, and correlation value.
        """
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.shape[1] < 2:
            # Need at least 2 numeric columns to compute correlation
            return []

        # .corr() computes the pairwise Pearson correlation matrix
        corr_matrix = numeric_df.corr()

        strong_correlations: list[dict] = []
        # np.triu_indices_from gets indices for the upper triangle of the matrix,
        # k=1 skips the diagonal (a column always correlates 1.0 with itself)
        rows, cols = np.triu_indices_from(corr_matrix, k=1)

        for r, c in zip(rows, cols):
            corr_value = corr_matrix.iloc[r, c]
            # Only report correlations where |r| > 0.7
            if abs(corr_value) > 0.7 and not np.isnan(corr_value):
                strong_correlations.append({
                    "column_1": corr_matrix.columns[r],
                    "column_2": corr_matrix.columns[c],
                    "correlation": round(float(corr_value), 4),
                })

        logger.info(
            f"Found {len(strong_correlations)} strong correlations (|r| > 0.7)"
        )
        return strong_correlations
