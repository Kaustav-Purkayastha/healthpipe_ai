"""
spark_transformer.py — PySpark version of the data transformer.

This module demonstrates the same transformations as agents/transformer.py
but using PySpark instead of pandas. For datasets under 1M rows, pandas is
faster and simpler. PySpark becomes valuable at 10M+ rows across distributed
clusters, where data doesn't fit in a single machine's memory.

The transformations are intentionally kept parallel to the pandas version
so you can compare the two approaches side by side.

Requirements:
    pip install pyspark

Usage:
    from spark_module.spark_transformer import SparkTransformer

    transformer = SparkTransformer()
    clean_df = transformer.run(spark_df, "my_dataset")
    transformer.stop()  # always stop SparkSession when done
"""

import re
from datetime import datetime

from core.utils import get_logger

logger = get_logger(__name__)

# PySpark is an optional dependency — import is wrapped in try/except
# so the rest of the project works even without PySpark installed
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    logger.warning(
        "PySpark not installed. Install with: pip install pyspark"
    )


class SparkTransformer:
    """
    Cleans and standardizes a PySpark DataFrame.

    Mirrors the pandas TransformerAgent but uses PySpark's distributed
    computation model. Each method returns a new DataFrame (PySpark
    DataFrames are immutable — transforms create new ones).

    Usage:
        transformer = SparkTransformer()
        clean_df = transformer.run(spark_df, "who_life_expectancy")
        transformer.stop()
    """

    def __init__(self, app_name: str = "HealthPipeAI") -> None:
        """
        Initialize a SparkSession.

        SparkSession is the entry point for all PySpark functionality.
        In local mode, it runs on your machine's cores. In a cluster,
        it distributes work across nodes.

        Args:
            app_name: Name shown in the Spark UI (http://localhost:4040).
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError(
                "PySpark is not installed. Run: pip install pyspark"
            )

        # .master("local[*]") uses all available CPU cores on this machine
        # In production, this would be "yarn" or "k8s://..." for a cluster
        self._spark = (
            SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            .getOrCreate()
        )

        self._transform_log: list[dict] = []
        self._step_counter: int = 0
        logger.info(f"SparkSession created: {app_name}")

    def run(self, df: "DataFrame", dataset_name: str) -> "DataFrame":
        """
        Run the full transformation pipeline on a Spark DataFrame.

        Args:
            df:           PySpark DataFrame to transform.
            dataset_name: Name for logging and metadata columns.

        Returns:
            Cleaned PySpark DataFrame.
        """
        logger.info(
            f"[Spark] Transforming '{dataset_name}' "
            f"({df.count()} rows, {len(df.columns)} cols)"
        )

        self._transform_log = []
        self._step_counter = 0

        df = self.standardize_columns(df)
        df = self.remove_duplicates(df)
        df = self.handle_nulls(df)
        df = self.add_metadata_columns(df, dataset_name)

        logger.info(
            f"[Spark] Transformation complete: "
            f"{df.count()} rows, {len(df.columns)} cols"
        )
        return df

    def standardize_columns(self, df: "DataFrame") -> "DataFrame":
        """
        Rename all columns to snake_case.

        PySpark columns are renamed using .withColumnRenamed() or
        .toDF() with a list of new names. We use .toDF() since it
        renames all columns at once.

        Args:
            df: PySpark DataFrame with original column names.

        Returns:
            DataFrame with snake_case column names.
        """
        def to_snake_case(name: str) -> str:
            """Convert a single string to snake_case."""
            s = re.sub(r"([A-Z])", r"_\1", name)
            s = re.sub(r"[^a-zA-Z0-9]", "_", s)
            s = re.sub(r"_+", "_", s)
            return s.strip("_").lower()

        original = df.columns
        new_names = [to_snake_case(col) for col in original]

        # Handle duplicates by appending _2, _3, etc.
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for name in new_names:
            if name in seen:
                seen[name] += 1
                deduped.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 1
                deduped.append(name)

        # .toDF(*names) returns a new DataFrame with renamed columns
        df = df.toDF(*deduped)

        changed = sum(1 for a, b in zip(original, deduped) if a != b)
        self._log_step(
            "standardize_columns",
            f"Renamed {changed}/{len(df.columns)} columns to snake_case",
        )
        return df

    def remove_duplicates(self, df: "DataFrame") -> "DataFrame":
        """
        Remove exact duplicate rows.

        PySpark's .dropDuplicates() works like pandas .drop_duplicates()
        but distributes the dedup across the cluster.

        Args:
            df: PySpark DataFrame that may contain duplicates.

        Returns:
            DataFrame with duplicates removed.
        """
        before = df.count()
        df = df.dropDuplicates()
        after = df.count()
        removed = before - after

        self._log_step(
            "remove_duplicates",
            f"Removed {removed} duplicate rows (before={before}, after={after})",
        )
        return df

    def handle_nulls(self, df: "DataFrame") -> "DataFrame":
        """
        Fill null values based on column data type.

        Strategy (same as pandas version):
            - Numeric columns (int/double/float): fill with median
            - String columns: fill with "Unknown"

        Computing median in Spark uses .approxQuantile() which gives
        an approximate median — faster than exact on large datasets.

        Args:
            df: PySpark DataFrame with null values.

        Returns:
            DataFrame with nulls handled.
        """
        fills_applied = 0

        for field in df.schema.fields:
            col_name = field.name
            # .dataType is Spark's type system: IntegerType, DoubleType,
            # StringType, TimestampType, etc.
            type_name = str(field.dataType)

            null_count = df.where(F.col(col_name).isNull()).count()
            if null_count == 0:
                continue

            if "Int" in type_name or "Double" in type_name or "Float" in type_name:
                # .approxQuantile(col, [0.5], 0.01) returns approximate median
                # The third argument (0.01) is the relative error — lower = more
                # accurate but slower. 0.01 means within 1% of true median.
                quantiles = df.approxQuantile(col_name, [0.5], 0.01)
                if quantiles:
                    median_val = quantiles[0]
                    df = df.fillna({col_name: median_val})
                    fills_applied += 1
                    logger.info(
                        f"  [Spark] Filled {null_count} nulls in "
                        f"'{col_name}' with median={median_val:.2f}"
                    )

            elif "String" in type_name:
                df = df.fillna({col_name: "Unknown"})
                fills_applied += 1

        self._log_step(
            "handle_nulls",
            f"Applied null-fill strategy to {fills_applied} columns",
        )
        return df

    def add_metadata_columns(
        self, df: "DataFrame", dataset_name: str
    ) -> "DataFrame":
        """
        Add pipeline metadata columns.

        Uses F.current_timestamp() for _loaded_at and F.lit() for
        constant values. F.lit() creates a column where every row
        has the same value.

        Args:
            df:           PySpark DataFrame to annotate.
            dataset_name: Value for the _source column.

        Returns:
            DataFrame with _loaded_at and _source columns added.
        """
        df = df.withColumn("_loaded_at", F.current_timestamp())
        # F.lit() creates a "literal" column — same value in every row
        df = df.withColumn("_source", F.lit(dataset_name))

        self._log_step(
            "add_metadata_columns",
            f"Added _loaded_at and _source='{dataset_name}'",
        )
        return df

    def get_transform_summary(self) -> list[dict]:
        """
        Return the audit log of all transformation steps.

        Returns:
            List of dicts with step number, action, detail, and timestamp.
        """
        return self._transform_log

    def from_pandas(self, pandas_df: "object") -> "DataFrame":
        """
        Convert a pandas DataFrame to a PySpark DataFrame.

        Useful for testing: ingest with pandas, then hand off to Spark.

        Args:
            pandas_df: A pandas DataFrame.

        Returns:
            PySpark DataFrame.
        """
        # createDataFrame accepts a pandas DataFrame directly
        return self._spark.createDataFrame(pandas_df)

    def to_pandas(self, spark_df: "DataFrame") -> "object":
        """
        Convert a PySpark DataFrame back to pandas.

        Warning: this collects all data to the driver node. Only use
        for small results or after aggregation.

        Args:
            spark_df: PySpark DataFrame to convert.

        Returns:
            pandas DataFrame.
        """
        return spark_df.toPandas()

    def stop(self) -> None:
        """
        Stop the SparkSession and release cluster resources.

        Always call this when you're done — leaving a SparkSession
        running wastes memory and can block port 4040.
        """
        self._spark.stop()
        logger.info("SparkSession stopped")

    def _log_step(self, action: str, detail: str) -> None:
        """
        Record a transformation step in the audit log.

        Args:
            action: Short name for the step.
            detail: Description of what happened.
        """
        self._step_counter += 1
        entry = {
            "step": self._step_counter,
            "action": action,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
            "engine": "pyspark",
        }
        self._transform_log.append(entry)
        logger.info(f"[Spark] Step {self._step_counter}: {action} — {detail}")

    def __repr__(self) -> str:
        """Developer-friendly string for debugging."""
        return "<SparkTransformer(engine='pyspark')>"
