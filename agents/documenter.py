"""
documenter.py — Auto Documenter Agent.

Generates comprehensive documentation for a dataset:
    - Data dictionary (column name, type, nullability, sample values, description)
    - Schema (CREATE TABLE-equivalent info)
    - Lineage (where the data came from + transformation history)
    - Quality summary (from the quality scorecard)
    - Usage notes (practical tips for data consumers)

Outputs both JSON (for programmatic use) and Markdown (for human reading).
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import DOCS_DIR
from core.llm import query_ollama, is_ollama_available
from core.utils import get_logger, save_json

logger = get_logger(__name__)

# Track whether Ollama is reachable — checked once at import time
# to avoid repeated connection attempts for every column
_OLLAMA_READY: bool = is_ollama_available()

# Rule-based description patterns — maps column name keywords to
# human-readable descriptions. Used when no explicit description is available.
DESCRIPTION_PATTERNS: dict[str, str] = {
    "patient_id": "Unique identifier for the patient",
    "patient_age": "Age of the patient at time of event",
    "patient_sex": "Biological sex of the patient (1=Male, 2=Female)",
    "patient_weight": "Patient weight in kilograms",
    "safety_report_id": "Unique FDA safety report identifier",
    "receive_date": "Date the report was received by FDA",
    "drug_name": "Medicinal product name as reported",
    "brand_name": "Commercial brand name of the drug",
    "generic_name": "Generic (non-proprietary) drug name",
    "drug_indication": "Medical condition the drug was prescribed for",
    "reactions": "Adverse reactions experienced (semicolon-separated)",
    "serious": "Whether the event was classified as serious",
    "country_code": "ISO 3-letter country code (e.g., IND, USA)",
    "year": "Year the measurement was taken",
    "value": "Measured numeric value for the indicator",
    "indicator_name": "Name of the health indicator being measured",
    "indicator_code": "WHO code for the health indicator",
    "id": "Unique record identifier",
    "age": "Age in years",
    "date": "Date of the event or measurement",
    "name": "Name field",
    "code": "Categorical code or identifier",
    "type": "Category or classification type",
    "source": "Data source or origin",
    "status": "Current status of the record",
    "count": "Count or frequency of occurrences",
    "rate": "Rate per unit (e.g., per 1000 population)",
    "percentage": "Percentage value (0-100)",
    "_loaded_at": "Timestamp when this record was loaded into the pipeline",
    "_source": "Name of the source dataset this record came from",
}


class DocumenterAgent:
    """
    Generates documentation for a dataset in both JSON and Markdown formats.

    Usage:
        documenter = DocumenterAgent()
        docs = documenter.run(
            df, "who_life_expectancy",
            source_metadata={...},
            profile_data={...},
            transform_log=[...],
            quality_scorecard={...},
        )
    """

    def run(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source_metadata: dict | None = None,
        profile_data: dict | None = None,
        transform_log: list[dict] | None = None,
        quality_scorecard: dict | None = None,
    ) -> dict:
        """
        Generate full documentation for a dataset.

        Args:
            df:                 The DataFrame to document.
            dataset_name:       Name for labeling the docs.
            source_metadata:    Output from source.get_metadata() (optional).
            profile_data:       Output from ProfilerAgent.run() (optional).
            transform_log:      Output from TransformerAgent.get_transform_summary()
                                (optional).
            quality_scorecard:  Output from QualityCheckerAgent.run() (optional).

        Returns:
            Dict containing all documentation sections.
        """
        logger.info(f"Generating documentation for '{dataset_name}'")

        docs = {
            "dataset_name": dataset_name,
            "generated_at": datetime.now().isoformat(),
            "data_dictionary": self._generate_data_dictionary(df),
            "schema": self._generate_schema(df, dataset_name),
            "lineage": self._generate_lineage(
                dataset_name, source_metadata, transform_log
            ),
            "quality_summary": self._generate_quality_summary(
                quality_scorecard
            ),
            "usage_notes": self._generate_usage_notes(
                df, profile_data, quality_scorecard
            ),
        }

        # Save as JSON
        json_path = DOCS_DIR / f"docs_{dataset_name}.json"
        save_json(docs, json_path)
        logger.info(f"Documentation JSON saved to: {json_path}")

        # Save as Markdown (human-readable)
        md_path = DOCS_DIR / f"docs_{dataset_name}.md"
        self._save_as_markdown(docs, md_path)
        logger.info(f"Documentation Markdown saved to: {md_path}")

        return docs

    def _generate_data_dictionary(self, df: pd.DataFrame) -> list[dict]:
        """
        Create a data dictionary: one entry per column with type, stats, and description.

        Args:
            df: The DataFrame to document.

        Returns:
            List of dicts, each describing one column.
        """
        dictionary: list[dict] = []

        for col in df.columns:
            col_data = df[col]

            # Get up to 3 sample values (non-null, unique)
            samples = col_data.dropna().unique()[:3]
            sample_values = [str(s) for s in samples]

            entry = {
                "column_name": col,
                "data_type": str(col_data.dtype),
                "nullable": bool(col_data.isna().any()),
                "null_count": int(col_data.isna().sum()),
                "unique_count": int(col_data.nunique()),
                "sample_values": sample_values,
                # Pass sample values so the AI can use them for context
                "description": self._infer_description(col, sample_values),
            }
            dictionary.append(entry)

        return dictionary

    def _infer_description(
        self, column_name: str, sample_values: list[str] | None = None
    ) -> str:
        """
        Infer a column's description, trying AI first then falling back to rules.

        Strategy:
            1. If Ollama is available, ask the LLM to describe the column
               based on its name and sample values.
            2. If Ollama is unavailable or returns nothing, use rule-based
               pattern matching on the column name.

        Args:
            column_name:   The column name to describe.
            sample_values: Up to 3 sample values from the column (for AI context).

        Returns:
            A human-readable description string.
        """
        # Try AI-generated description first (only if Ollama is running)
        if _OLLAMA_READY:
            ai_description = self._get_ai_description(column_name, sample_values)
            if ai_description:
                logger.info(f"  AI description for '{column_name}': {ai_description}")
                return ai_description

        # Fall back to rule-based pattern matching
        return self._get_rule_based_description(column_name)

    def _get_ai_description(
        self, column_name: str, sample_values: list[str] | None = None
    ) -> str | None:
        """
        Ask the Ollama LLM to describe a data column.

        Args:
            column_name:   The column name.
            sample_values: Example values for context.

        Returns:
            AI-generated description string, or None if it failed.
        """
        samples_str = ", ".join(sample_values) if sample_values else "none"
        prompt = (
            f"In one sentence, describe what a data column named "
            f"'{column_name}' with sample values [{samples_str}] "
            f"likely contains in a healthcare dataset. "
            f"Reply with only the description, no extra text."
        )

        result = query_ollama(prompt)
        if result:
            # Clean up: take only the first sentence, strip quotes
            first_sentence = result.split(".")[0].strip().strip('"').strip("'")
            if len(first_sentence) > 10:
                return first_sentence + "."
        return None

    def _get_rule_based_description(self, column_name: str) -> str:
        """
        Describe a column using rule-based pattern matching on its name.

        Checks exact match first, then partial keyword match.

        Args:
            column_name: The column name to describe.

        Returns:
            A human-readable description string.
        """
        col_lower = column_name.lower()

        # Exact match first — most specific
        if col_lower in DESCRIPTION_PATTERNS:
            return DESCRIPTION_PATTERNS[col_lower]

        # Partial match — check if any known keyword appears in the name
        for pattern, description in DESCRIPTION_PATTERNS.items():
            if pattern in col_lower:
                return description

        return "No description available — review column contents"

    def _generate_schema(
        self, df: pd.DataFrame, dataset_name: str
    ) -> dict:
        """
        Generate CREATE TABLE-equivalent schema information.

        Maps pandas dtypes to SQL-like types for documentation purposes.
        Not actual SQL — just informational.

        Args:
            df:           The DataFrame to document.
            dataset_name: Used as the table name.

        Returns:
            Dict with table name and column definitions.
        """
        # Map pandas dtype kinds to SQL-like type names
        dtype_map = {
            "i": "INTEGER",
            "f": "FLOAT",
            "O": "VARCHAR",
            "M": "TIMESTAMP",
            "b": "BOOLEAN",
        }

        columns = []
        for col in df.columns:
            dtype_kind = df[col].dtype.kind
            sql_type = dtype_map.get(dtype_kind, "VARCHAR")
            nullable = bool(df[col].isna().any())
            columns.append({
                "name": col,
                "type": sql_type,
                "nullable": nullable,
            })

        return {
            "table_name": dataset_name,
            "columns": columns,
            "row_count": len(df),
        }

    def _generate_lineage(
        self,
        dataset_name: str,
        source_metadata: dict | None,
        transform_log: list[dict] | None,
    ) -> dict:
        """
        Document where the data came from and what transformations were applied.

        Data lineage is critical for auditability — it answers "where did
        this data come from and what happened to it?"

        Args:
            dataset_name:    Name of the dataset.
            source_metadata: From source.get_metadata().
            transform_log:   From TransformerAgent.get_transform_summary().

        Returns:
            Dict with source info and transformation history.
        """
        return {
            "dataset_name": dataset_name,
            "source": source_metadata or {"note": "Source metadata not provided"},
            "transformations": transform_log or [],
            "transformation_count": len(transform_log) if transform_log else 0,
        }

    def _generate_quality_summary(
        self, quality_scorecard: dict | None
    ) -> dict:
        """
        Summarize the quality scorecard for documentation.

        Args:
            quality_scorecard: Full scorecard from QualityCheckerAgent.run().

        Returns:
            Dict with score, grade, and counts of passed/failed checks.
        """
        if quality_scorecard is None:
            return {"note": "Quality scorecard not provided"}

        # Extract just the summary fields, not the full check details
        return {
            "score": quality_scorecard.get("score", 0),
            "grade": quality_scorecard.get("grade", "N/A"),
            "total_checks": quality_scorecard.get("total_checks", 0),
            "checks_passed": quality_scorecard.get("checks_passed", 0),
            "checks_failed": quality_scorecard.get("checks_failed", 0),
        }

    def _generate_usage_notes(
        self,
        df: pd.DataFrame,
        profile_data: dict | None,
        quality_scorecard: dict | None,
    ) -> list[str]:
        """
        Generate practical notes for people who will use this data.

        Notes are based on what was found during profiling and quality checking.

        Args:
            df:                The documented DataFrame.
            profile_data:      From ProfilerAgent.run().
            quality_scorecard: From QualityCheckerAgent.run().

        Returns:
            List of human-readable usage notes.
        """
        notes: list[str] = []

        notes.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")

        # Note about data types present
        dtypes = df.dtypes.value_counts()
        dtype_summary = ", ".join(
            f"{count} {dtype}" for dtype, count in dtypes.items()
        )
        notes.append(f"Column types: {dtype_summary}.")

        # Note about high-null columns from profile
        if profile_data:
            quality_issues = profile_data.get("quality_issues", [])
            high_null = [
                i for i in quality_issues if i.get("issue") == "high_null_rate"
            ]
            if high_null:
                col_names = [i["column"] for i in high_null]
                notes.append(
                    f"Columns with high null rates: {', '.join(col_names)}. "
                    f"Consider excluding or imputing before analysis."
                )

        # Note about quality grade
        if quality_scorecard:
            grade = quality_scorecard.get("grade", "N/A")
            score = quality_scorecard.get("score", 0)
            notes.append(f"Quality grade: {grade} ({score}%).")
            if grade in ("C", "F"):
                notes.append(
                    "WARNING: Data quality is below acceptable levels. "
                    "Review the quality scorecard before using this data."
                )

        # Note about metadata columns
        meta_cols = [c for c in df.columns if c.startswith("_")]
        if meta_cols:
            notes.append(
                f"Pipeline metadata columns ({', '.join(meta_cols)}) "
                f"are for internal tracking — exclude from analysis."
            )

        return notes

    def _save_as_markdown(self, docs: dict, file_path: Path) -> None:
        """
        Render the documentation as a Markdown file with tables.

        Uses list + join for string building (not string concatenation)
        to avoid O(n²) performance with large documents.

        Args:
            docs:      The full documentation dict.
            file_path: Where to save the .md file.
        """
        # Build the Markdown as a list of lines, then join at the end
        lines: list[str] = []

        dataset_name = docs.get("dataset_name", "Unknown")
        lines.append(f"# Dataset Documentation: {dataset_name}")
        lines.append("")
        lines.append(f"Generated at: {docs.get('generated_at', 'N/A')}")
        lines.append("")

        # --- Data Dictionary Table ---
        lines.append("## Data Dictionary")
        lines.append("")
        lines.append(
            "| Column | Type | Nullable | Unique Count | Description |"
        )
        lines.append(
            "|--------|------|----------|--------------|-------------|"
        )
        for col in docs.get("data_dictionary", []):
            lines.append(
                f"| {col['column_name']} "
                f"| {col['data_type']} "
                f"| {'Yes' if col['nullable'] else 'No'} "
                f"| {col['unique_count']} "
                f"| {col['description']} |"
            )
        lines.append("")

        # --- Schema ---
        lines.append("## Schema")
        lines.append("")
        schema = docs.get("schema", {})
        lines.append(f"Table: `{schema.get('table_name', 'N/A')}`")
        lines.append(f"Rows: {schema.get('row_count', 'N/A')}")
        lines.append("")
        lines.append("| Column | Type | Nullable |")
        lines.append("|--------|------|----------|")
        for col in schema.get("columns", []):
            lines.append(
                f"| {col['name']} "
                f"| {col['type']} "
                f"| {'Yes' if col['nullable'] else 'No'} |"
            )
        lines.append("")

        # --- Lineage ---
        lines.append("## Data Lineage")
        lines.append("")
        lineage = docs.get("lineage", {})
        source = lineage.get("source", {})
        lines.append(f"**Source:** {source.get('name', 'N/A')} "
                      f"({source.get('description', 'N/A')})")
        lines.append("")
        transforms = lineage.get("transformations", [])
        if transforms:
            lines.append("### Transformation Steps")
            lines.append("")
            for t in transforms:
                lines.append(
                    f"{t.get('step', '?')}. **{t.get('action', 'N/A')}** — "
                    f"{t.get('detail', 'N/A')}"
                )
            lines.append("")

        # --- Quality Summary ---
        lines.append("## Quality Summary")
        lines.append("")
        quality = docs.get("quality_summary", {})
        if "score" in quality:
            lines.append(f"- **Score:** {quality['score']}%")
            lines.append(f"- **Grade:** {quality['grade']}")
            lines.append(f"- **Checks passed:** "
                          f"{quality['checks_passed']}/{quality['total_checks']}")
        else:
            lines.append(quality.get("note", "No quality data available."))
        lines.append("")

        # --- Usage Notes ---
        lines.append("## Usage Notes")
        lines.append("")
        for note in docs.get("usage_notes", []):
            lines.append(f"- {note}")
        lines.append("")

        # Write to file — join all lines with newline
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
