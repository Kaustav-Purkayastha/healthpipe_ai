"""
quality_checker.py — Data Quality Checker Agent.

Runs a suite of quality checks against a DataFrame and produces a scorecard:
    - Completeness (overall non-null percentage)
    - Duplicates (duplicate row percentage)
    - Null rates (per-column, flagged against threshold)
    - Type consistency (mixed types within object columns)
    - Value ranges (negatives in positive columns, extreme outliers)
    - Uniqueness (ID-like columns verified as actually unique)

Final score = (checks_passed / total_checks) * 100
Grades: A >= 90, B >= 75, C >= 60, F < 60
"""

from datetime import datetime

import numpy as np
import pandas as pd

from core.config import (
    MAX_NULL_PERCENTAGE,
    MAX_DUPLICATE_PERCENTAGE,
    MIN_COMPLETENESS_SCORE,
    EXTREME_OUTLIER_STD,
    QUALITY_GRADES,
    REPORTS_DIR,
)
from core.utils import get_logger, save_json

logger = get_logger(__name__)


class QualityCheckerAgent:
    """
    Runs quality checks on a DataFrame and produces a graded scorecard.

    Usage:
        checker = QualityCheckerAgent()
        scorecard = checker.run(df, "who_life_expectancy")
        print(scorecard["grade"])  # "A", "B", "C", or "F"
    """

    def run(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Run all quality checks and compute the final score.

        Args:
            df:           The DataFrame to check.
            dataset_name: Name for labeling the scorecard.

        Returns:
            Scorecard dict with individual check results, total score, and grade.
        """
        logger.info(
            f"Running quality checks on '{dataset_name}' ({len(df)} rows)"
        )

        checks: list[dict] = []

        # Run each check — each returns a list of individual check results
        checks.extend(self._check_completeness(df))
        checks.extend(self._check_duplicates(df))
        checks.extend(self._check_null_rates(df))
        checks.extend(self._check_type_consistency(df))
        checks.extend(self._check_value_ranges(df))
        checks.extend(self._check_uniqueness(df))

        # Calculate score: percentage of checks that passed
        total = len(checks)
        passed = sum(1 for c in checks if c["passed"])
        score = (passed / total * 100) if total > 0 else 0.0
        grade = self._compute_grade(score)

        scorecard = {
            "dataset_name": dataset_name,
            "checked_at": datetime.now().isoformat(),
            "total_checks": total,
            "checks_passed": passed,
            "checks_failed": total - passed,
            "score": round(score, 2),
            "grade": grade,
            "checks": checks,
        }

        # Save scorecard as JSON
        output_path = REPORTS_DIR / f"quality_{dataset_name}.json"
        save_json(scorecard, output_path)
        logger.info(
            f"Quality scorecard: {score:.1f}% ({grade}) — "
            f"{passed}/{total} checks passed. Saved to: {output_path}"
        )

        return scorecard

    def _check_completeness(self, df: pd.DataFrame) -> list[dict]:
        """
        Check overall dataset completeness (percentage of non-null cells).

        Passes if completeness >= MIN_COMPLETENESS_SCORE (default 70%).

        Args:
            df: The DataFrame to check.

        Returns:
            List with one check result dict.
        """
        total_cells = df.shape[0] * df.shape[1]
        non_null = df.notna().sum().sum()
        completeness = (non_null / total_cells * 100) if total_cells > 0 else 0.0

        return [{
            "check": "overall_completeness",
            "passed": completeness >= MIN_COMPLETENESS_SCORE,
            "value": round(completeness, 2),
            "threshold": MIN_COMPLETENESS_SCORE,
            "detail": f"{completeness:.1f}% complete "
                      f"(threshold: {MIN_COMPLETENESS_SCORE}%)",
        }]

    def _check_duplicates(self, df: pd.DataFrame) -> list[dict]:
        """
        Check that duplicate row percentage is below threshold.

        Passes if duplicate % <= MAX_DUPLICATE_PERCENTAGE (default 5%).

        Args:
            df: The DataFrame to check.

        Returns:
            List with one check result dict.
        """
        dup_count = int(df.duplicated().sum())
        dup_pct = (dup_count / len(df) * 100) if len(df) > 0 else 0.0

        return [{
            "check": "duplicate_rows",
            "passed": dup_pct <= MAX_DUPLICATE_PERCENTAGE,
            "value": round(dup_pct, 2),
            "threshold": MAX_DUPLICATE_PERCENTAGE,
            "detail": f"{dup_count} duplicates ({dup_pct:.1f}%)",
        }]

    def _check_null_rates(self, df: pd.DataFrame) -> list[dict]:
        """
        Check each column's null percentage against the threshold.

        Each column is a separate check. Passes if null % <= MAX_NULL_PERCENTAGE.

        Args:
            df: The DataFrame to check.

        Returns:
            List of check result dicts (one per column).
        """
        results: list[dict] = []

        for col in df.columns:
            null_pct = (
                df[col].isna().sum() / len(df) * 100
            ) if len(df) > 0 else 0.0

            results.append({
                "check": f"null_rate_{col}",
                "passed": null_pct <= MAX_NULL_PERCENTAGE,
                "value": round(null_pct, 2),
                "threshold": MAX_NULL_PERCENTAGE,
                "detail": f"Column '{col}': {null_pct:.1f}% null",
            })

        return results

    def _check_type_consistency(self, df: pd.DataFrame) -> list[dict]:
        """
        Detect mixed types within object columns.

        An object column is "mixed" if it contains both numeric-looking
        and non-numeric values. This usually indicates a data quality problem
        (e.g., "42" and "N/A" in the same column).

        Args:
            df: The DataFrame to check.

        Returns:
            List of check result dicts (one per object column).
        """
        results: list[dict] = []

        for col in df.select_dtypes(include=["object", "string"]).columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            # Try converting each value to a number — count how many succeed
            numeric_test = pd.to_numeric(non_null, errors="coerce")
            numeric_count = int(numeric_test.notna().sum())
            total = len(non_null)

            # "Mixed" = some values convert and some don't (between 10% and 90%)
            numeric_ratio = numeric_count / total if total > 0 else 0
            is_mixed = 0.1 < numeric_ratio < 0.9

            results.append({
                "check": f"type_consistency_{col}",
                "passed": not is_mixed,
                "value": f"{numeric_count}/{total} numeric",
                "threshold": "< 10% or > 90% numeric",
                "detail": f"Column '{col}': {numeric_ratio:.0%} of values "
                          f"are numeric — {'MIXED' if is_mixed else 'consistent'}",
            })

        return results

    def _check_value_ranges(self, df: pd.DataFrame) -> list[dict]:
        """
        Check numeric columns for suspicious values.

        Two checks per numeric column:
            1. Negative values in columns that should be positive
               (name contains "count", "age", "rate", "percentage")
            2. Extreme outliers (> 5 standard deviations from the mean)

        Args:
            df: The DataFrame to check.

        Returns:
            List of check result dicts.
        """
        results: list[dict] = []
        positive_hints = ["count", "age", "rate", "percentage", "pct", "num"]

        for col in df.select_dtypes(include=["number"]).columns:
            clean = df[col].dropna()
            if len(clean) == 0:
                continue

            # Check for negative values in "should be positive" columns
            col_lower = col.lower()
            if any(hint in col_lower for hint in positive_hints):
                neg_count = int((clean < 0).sum())
                results.append({
                    "check": f"no_negatives_{col}",
                    "passed": neg_count == 0,
                    "value": neg_count,
                    "threshold": 0,
                    "detail": f"Column '{col}': "
                              f"{neg_count} negative values found",
                })

            # Check for extreme outliers (> 5 standard deviations)
            mean = clean.mean()
            std = clean.std()
            if std > 0:
                # Calculate how many standard deviations each value is from mean
                z_scores = ((clean - mean) / std).abs()
                extreme_count = int((z_scores > EXTREME_OUTLIER_STD).sum())
                results.append({
                    "check": f"extreme_outliers_{col}",
                    "passed": extreme_count == 0,
                    "value": extreme_count,
                    "threshold": f"> {EXTREME_OUTLIER_STD} std deviations",
                    "detail": f"Column '{col}': "
                              f"{extreme_count} extreme outliers",
                })

        return results

    def _check_uniqueness(self, df: pd.DataFrame) -> list[dict]:
        """
        Verify that columns that look like IDs are actually unique.

        A column "looks like an ID" if its name contains "id", "key", or "code"
        and it has >90% unique values. If it looks like an ID, it should be
        100% unique — duplicates indicate a data problem.

        Args:
            df: The DataFrame to check.

        Returns:
            List of check result dicts.
        """
        results: list[dict] = []
        id_hints = ["_id", "_key", "_code", "report_id"]

        for col in df.columns:
            col_lower = col.lower()
            # Check if column name suggests it's an identifier
            if not any(hint in col_lower for hint in id_hints):
                continue

            unique_count = df[col].nunique()
            total = len(df)
            unique_ratio = unique_count / total if total > 0 else 0

            # Only test columns that are mostly unique (>90%) — otherwise
            # it's probably not a real ID column
            if unique_ratio > 0.9:
                is_unique = unique_count == total
                results.append({
                    "check": f"uniqueness_{col}",
                    "passed": is_unique,
                    "value": unique_count,
                    "threshold": total,
                    "detail": f"Column '{col}': {unique_count}/{total} unique "
                              f"({'all unique' if is_unique else 'HAS DUPLICATES'})",
                })

        return results

    def _compute_grade(self, score: float) -> str:
        """
        Convert a numeric score (0-100) to a letter grade.

        Uses thresholds from config: A >= 90, B >= 75, C >= 60, F < 60.

        Args:
            score: Quality score as a percentage.

        Returns:
            Letter grade string ("A", "B", "C", or "F").
        """
        # QUALITY_GRADES is {"A": 90.0, "B": 75.0, "C": 60.0}
        # Check from highest to lowest
        for grade, threshold in sorted(
            QUALITY_GRADES.items(), key=lambda x: x[1], reverse=True
        ):
            if score >= threshold:
                return grade
        return "F"
