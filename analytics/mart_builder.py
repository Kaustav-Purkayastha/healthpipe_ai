"""
analytics/mart_builder.py — Reporting mart: reporting_state_health.

Joins CDC chronic disease prevalence + CMS Medicare payments + Census population
into a 51-row state-grain mart with derived per-capita metrics.

Every filter rule and join key in this file was verified by live API calls.
Do not substitute different question IDs, year ranges, or join columns.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from core.config import (
    CACHE_DIR,
    DATABASE_PATH,
    EXCLUDED_LOCATION_CODES,
    FIPS_TO_STATE,
    OLLAMA_MODEL,
    QUALITY_GRADE_BANDS,
    STATE_ABBRS,
)
from core.database import DuckDBManager
from core.utils import get_logger, save_json

_log = get_logger(__name__)

# Mart artifact directory
_MART_DIR = Path("outputs") / "mart"


class MartBuilder:
    """Assembles the reporting_state_health mart from live or cached source data.

    Usage::

        builder = MartBuilder()
        mart_df = builder.build()          # uses cache when available
        scorecard = builder.quality_check(mart_df)
        briefing = builder.insight_briefing(router, mart_df)
        builder.to_duckdb(mart_df)
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialise with optional override for the cache directory.

        Args:
            cache_dir: Override path for cached parquet files.  Defaults to
                       ``data/cache/``.  Tests pass ``tmp_path`` here so they
                       don't pollute the real cache.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Raw per-source row counts captured during the last build() — feeds the
        # build metadata (freshness + "underlying volume" story on the UI).
        self._pull_stats: dict = {}
        # Metadata for the last build() (built_at, row counts, measures, …).
        self.build_metadata: dict = {}

    # ------------------------------------------------------------------
    # 1. Caching layer
    # ------------------------------------------------------------------

    def _cached_pull(
        self,
        cache_name: str,
        pull_fn: Callable[[], pd.DataFrame],
        refresh: bool,
    ) -> pd.DataFrame:
        """Return a cached DataFrame or call pull_fn() to fetch and cache it.

        WHY: API pulls take minutes (CMS ~5 min, CDC ~30 s, Census <5 s).
        After the first pull the demo and tests rebuild the mart in seconds.

        Args:
            cache_name: Base filename (without extension) for the parquet cache.
            pull_fn:    Zero-argument callable that returns a fresh DataFrame.
            refresh:    When True, always re-fetch even if cache exists.

        Returns:
            Cached or freshly pulled DataFrame.
        """
        cache_path = self._cache_dir / f"{cache_name}.parquet"

        if not refresh and cache_path.exists():
            _log.info("Cache hit: loading %s from %s", cache_name, cache_path)
            return pd.read_parquet(cache_path)

        _log.info("Cache miss (%s) — pulling from source...", cache_name)
        t0 = time.monotonic()
        df = pull_fn()
        elapsed = time.monotonic() - t0
        _log.info("Pull complete: %d rows in %.1fs", len(df), elapsed)

        if not df.empty:
            df.to_parquet(cache_path, index=False)
            _log.info("Cached to %s", cache_path)

        return df

    # ------------------------------------------------------------------
    # 2a. Census pull
    # ------------------------------------------------------------------

    def pull_census(self, refresh: bool = False) -> pd.DataFrame:
        """Pull ACS5 state populations from the Census API.

        Returns:
            DataFrame with columns: state_abbr, population (int), population_year.
        """
        def _pull() -> pd.DataFrame:
            from ingestion.census_source import CensusPopulationSource  # lazy import
            src = CensusPopulationSource()
            df = src.extract()
            return df

        raw = self._cached_pull("mart_census", _pull, refresh)
        self._pull_stats["census_rows"] = len(raw)  # for build metadata

        if raw.empty:
            return pd.DataFrame(columns=["state_abbr", "population", "population_year"])

        result = raw[["state_abbr", "population"]].copy()
        result["population"] = pd.to_numeric(result["population"], errors="coerce")
        result["population_year"] = 2023
        return result

    # ------------------------------------------------------------------
    # 2b. CDC CDI measure pull (latest-non-null rule)
    # ------------------------------------------------------------------

    def pull_cdi_measure(
        self,
        question_id: str,
        cache_name: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Pull a CDC Chronic Disease Indicator measure across 2021-2023.

        LATEST-NON-NULL RULE: for each state, use the most recent year (2023 →
        2022 → 2021) that has a non-suppressed, non-null crude prevalence value.
        WHY: FL diabetes is null for some earlier years (verified live); other
        states may have different gaps — pulling 3 years protects against them.

        Args:
            question_id: CDC CDI questionid (e.g. "DIA01").
            cache_name:  Parquet cache name (e.g. "mart_cdi_diabetes").
            refresh:     Force re-pull.

        Returns:
            DataFrame with columns: state_abbr, value (float), vintage_year (int).
        """
        def _pull() -> pd.DataFrame:
            from ingestion.cdc_cdi_source import CDCChronicDiseaseSource  # lazy
            src = CDCChronicDiseaseSource()
            # Pull without year filter — we select 2021-2023 in pandas below.
            # This gives one API call instead of three.
            return src.extract(
                question_id=question_id,
                data_value_type="CRDPREV",
                overall_only=True,
                max_records=20_000,
            )

        raw = self._cached_pull(cache_name, _pull, refresh)
        # Accumulate raw CDI rows across all measures pulled this build.
        self._pull_stats["cdc_rows"] = self._pull_stats.get("cdc_rows", 0) + len(raw)

        if raw.empty or "yearstart" not in raw.columns:
            return pd.DataFrame(columns=["state_abbr", "value", "vintage_year"])

        # Defensive filter: exclude aggregate location codes that the source
        # should already have removed (guards against mock data / source bugs)
        if "locationabbr" in raw.columns:
            raw = raw[~raw["locationabbr"].isin(EXCLUDED_LOCATION_CODES)]

        # Filter to years 2021-2023
        raw = raw[raw["yearstart"].isin({"2021", "2022", "2023"})].copy()

        # Exclude suppressed and null values
        if "is_suppressed" in raw.columns:
            raw = raw[~raw["is_suppressed"]]
        raw = raw[raw["datavalue"].notna()].copy()

        if raw.empty:
            return pd.DataFrame(columns=["state_abbr", "value", "vintage_year"])

        # Convert yearstart to int for correct sort order (descending = newest first)
        raw["yearstart_int"] = pd.to_numeric(raw["yearstart"], errors="coerce")
        raw = raw.sort_values("yearstart_int", ascending=False)

        # For each state take the first (most recent) valid value
        latest = raw.groupby("locationabbr", as_index=False).first()

        return (
            latest[["locationabbr", "datavalue", "yearstart_int"]]
            .rename(columns={
                "locationabbr": "state_abbr",
                "datavalue": "value",
                "yearstart_int": "vintage_year",
            })
        )

    # ------------------------------------------------------------------
    # 2c. CMS Medicare pull + per-state aggregation
    # ------------------------------------------------------------------

    def pull_cms(
        self,
        refresh: bool = False,
        full: bool = False,
    ) -> pd.DataFrame:
        """Pull CMS Medicare provider summary data and aggregate by state.

        Pulls WITHOUT a state filter (faster for 51 states than 51 separate
        requests) then aggregates in pandas.

        NOTE: an unfiltered partial pull is roughly NPI-ordered, which can bias
        state totals toward providers with low NPI numbers.  A WARNING is logged
        when the pull is a sample of the full 1.3 M-row dataset.
        Full pull: pass full=True (or --full-cms on CLI).

        Args:
            refresh: Force re-pull.
            full:    Pull all ~1.3 M providers instead of the default 300 k sample.

        Returns:
            DataFrame with columns: state_abbr, total_medicare_payment (float),
            provider_count (int), payment_year.
        """
        max_records = 1_300_000 if full else 300_000

        def _pull() -> pd.DataFrame:
            from ingestion.cms_source import CMSMedicareSource  # lazy
            src = CMSMedicareSource()
            src.connect()  # populates _total_rows for the sample-mode warning
            meta = src.get_metadata()
            total_in_dataset = meta.get("total_rows_in_dataset", -1)

            df = src.extract(max_records=max_records)

            if total_in_dataset > 0 and len(df) < total_in_dataset:
                _log.warning(
                    "CMS state totals are based on a SAMPLE of %d of %d providers "
                    "— demo mode. Use full=True or --full-cms for the complete dataset.",
                    len(df),
                    total_in_dataset,
                )
            return df

        raw = self._cached_pull("mart_cms_raw", _pull, refresh)
        self._pull_stats["cms_rows"] = len(raw)  # raw provider records, for metadata

        if raw.empty or "state" not in raw.columns:
            return pd.DataFrame(columns=[
                "state_abbr", "total_medicare_payment", "provider_count", "payment_year"
            ])

        # Defensive filter: exclude foreign/aggregate state codes (ZZ, US, etc.)
        cms_df = raw[raw["state"].isin(STATE_ABBRS)].copy()

        if cms_df.empty:
            return pd.DataFrame(columns=[
                "state_abbr", "total_medicare_payment", "provider_count", "payment_year"
            ])

        # Aggregate per state — SUM payments (these are per-provider totals),
        # COUNT distinct NPIs as provider_count
        agg = cms_df.groupby("state", as_index=False).agg(
            total_medicare_payment=("total_medicare_payment", "sum"),
            provider_count=("npi", "nunique"),
        )
        agg["payment_year"] = 2024
        agg = agg.rename(columns={"state": "state_abbr"})
        return agg

    # ------------------------------------------------------------------
    # 3. Assemble the mart
    # ------------------------------------------------------------------

    def build(
        self,
        refresh: bool = False,
        full_cms: bool = False,
        measures: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Assemble the 51-row state-grain reporting mart.

        WHY start from STATE_ABBRS: starting from the canonical 51-state list
        (not from a source frame) guarantees the grain and surfaces missing joins
        as NaN rather than silently dropping states.

        Args:
            refresh:   Re-pull all source data even if cached.
            full_cms:  Use the full 1.3 M-row CMS pull.
            measures:  CDI questionids to include as prevalence columns.  None →
                       the three verified defaults (DIA01/NPW14/TOB04).  The three
                       defaults keep friendly column slugs (diabetes/obesity/
                       smoking) for backward compatibility; any other measure uses
                       its lowercased questionid (e.g. AST01 → ast01_prevalence_pct).

        Returns:
            DataFrame with one row per US state/DC (51 rows).  Also populates
            ``self.build_metadata`` with freshness + volume info (see 5a).
        """
        from analytics.measure_catalog import DEFAULT_MEASURE_IDS, measure_slug

        # Reset per-build stats (raw pull counts are accumulated inside the pulls).
        self._pull_stats = {"cms_rows": 0, "cdc_rows": 0, "census_rows": 0}
        measure_ids = list(measures) if measures else list(DEFAULT_MEASURE_IDS)

        # Canonical 51-state spine — grain guarantee
        mart = pd.DataFrame({"state_abbr": STATE_ABBRS})

        # --- Census ---
        census = self.pull_census(refresh)
        mart = mart.merge(
            census[["state_abbr", "population"]],
            on="state_abbr",
            how="left",
        )

        # --- CDI measures (generalized: one prevalence + vintage column each) ---
        # Track measures whose coverage is thin — not every indicator reports all
        # 51 states, and a sparse column would mislead the charts/facts.
        low_coverage: list[dict] = []
        for qid in measure_ids:
            slug = measure_slug(qid)
            cdi = self.pull_cdi_measure(qid, f"mart_cdi_{slug}", refresh)
            prevalence_col = f"{slug}_prevalence_pct"
            mart = mart.merge(
                cdi.rename(columns={
                    "value": prevalence_col,
                    "vintage_year": f"{slug}_vintage",
                }),
                on="state_abbr",
                how="left",
            )
            covered = int(mart[prevalence_col].notna().sum()) if prevalence_col in mart else 0
            if covered < 40:
                low_coverage.append({"questionid": qid, "states_covered": covered})

        # --- CMS aggregates ---
        cms = self.pull_cms(refresh, full=full_cms)
        mart = mart.merge(
            cms[["state_abbr", "total_medicare_payment", "provider_count"]],
            on="state_abbr",
            how="left",
        )
        mart["payment_year"] = 2024

        # --- state_fips (invert FIPS_TO_STATE: abbr → fips) ---
        abbr_to_fips = {v: k for k, v in FIPS_TO_STATE.items()}
        mart["state_fips"] = mart["state_abbr"].map(abbr_to_fips)

        # --- Derived metrics ---
        mart["medicare_spend_per_capita"] = (
            mart["total_medicare_payment"] / mart["population"]
        ).round(2)

        mart["providers_per_100k"] = (
            mart["provider_count"] / mart["population"] * 100_000
        ).round(1)

        # --- Assertions ---
        assert len(mart) == 51, f"Expected 51 rows, got {len(mart)}"
        assert mart["state_abbr"].nunique() == 51, "state_abbr must be unique in mart"
        bad = set(EXCLUDED_LOCATION_CODES) | {"PR", "VI", "GU", "AS", "MP"}
        assert mart["state_abbr"].isin(bad).sum() == 0, (
            f"Aggregate/territory codes in mart: {mart[mart['state_abbr'].isin(bad)]['state_abbr'].tolist()}"
        )

        # --- Build metadata (freshness + volume story; persisted by to_duckdb) ---
        self.build_metadata = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "measures": measure_ids,
            "sample_mode": not full_cms,
            "cms_rows": self._pull_stats.get("cms_rows", 0),
            "cdc_rows": self._pull_stats.get("cdc_rows", 0),
            "census_rows": self._pull_stats.get("census_rows", 0),
            "low_coverage_measures": low_coverage,
        }

        _log.info(
            "Mart built: %d rows × %d columns (measures=%s)",
            len(mart),
            len(mart.columns),
            measure_ids,
        )
        return mart

    # ------------------------------------------------------------------
    # 4. Quality check
    # ------------------------------------------------------------------

    def quality_check(self, mart_df: pd.DataFrame) -> dict:
        """Run standard + mart-specific quality checks on the mart.

        Merges checks from QualityCheckerAgent with custom mart invariants:
          - row_count == 51
          - population all positive
          - prevalence values within 0-60 (sanity band)
          - medicare_spend_per_capita non-negative
          - at most 5 states with any NaN measure column

        Args:
            mart_df: The assembled mart DataFrame.

        Returns:
            Scorecard dict (same schema as QualityCheckerAgent output).
        """
        from agents.quality_checker import QualityCheckerAgent  # lazy

        agent_sc = QualityCheckerAgent().run(mart_df, "reporting_state_health")

        custom: list[dict] = []

        # Row count
        custom.append({
            "check": "mart_row_count",
            "passed": len(mart_df) == 51,
            "value": len(mart_df),
            "threshold": 51,
            "detail": f"Mart must have exactly 51 rows; found {len(mart_df)}",
        })

        # Population positive
        pop = pd.to_numeric(mart_df.get("population", pd.Series(dtype=float)), errors="coerce")
        neg_pop = int((pop.fillna(0) <= 0).sum())
        custom.append({
            "check": "mart_population_positive",
            "passed": neg_pop == 0,
            "value": neg_pop,
            "threshold": 0,
            "detail": f"{neg_pop} states with non-positive population",
        })

        # Prevalence sanity band 0–60 %
        # Comment: 60 % is well above the highest observed US state prevalence.
        # Values outside this range almost certainly indicate a data quality issue.
        for col in ["diabetes_prevalence_pct", "obesity_prevalence_pct", "smoking_prevalence_pct"]:
            if col not in mart_df.columns:
                continue
            clean = pd.to_numeric(mart_df[col], errors="coerce").dropna()
            out_of_band = int(((clean < 0) | (clean > 60)).sum())
            custom.append({
                "check": f"mart_{col}_band_0_60",
                "passed": out_of_band == 0,
                "value": out_of_band,
                "threshold": 0,
                "detail": f"{out_of_band} states with {col} outside 0–60 sanity band",
            })

        # Spend per capita non-negative
        if "medicare_spend_per_capita" in mart_df.columns:
            spcc = pd.to_numeric(mart_df["medicare_spend_per_capita"], errors="coerce")
            neg_sp = int((spcc.fillna(0) < 0).sum())
            custom.append({
                "check": "mart_spend_per_capita_nonneg",
                "passed": neg_sp == 0,
                "value": neg_sp,
                "threshold": 0,
                "detail": f"{neg_sp} states with negative medicare_spend_per_capita",
            })

        # At most 5 states with any NaN measure
        measure_cols = [c for c in [
            "diabetes_prevalence_pct", "obesity_prevalence_pct",
            "smoking_prevalence_pct", "medicare_spend_per_capita",
            "population",
        ] if c in mart_df.columns]
        if measure_cols:
            states_with_nan = int(mart_df[measure_cols].isna().any(axis=1).sum())
        else:
            states_with_nan = 0
        custom.append({
            "check": "mart_missing_measures_le5",
            "passed": states_with_nan <= 5,
            "value": states_with_nan,
            "threshold": 5,
            "detail": f"{states_with_nan} states with ≥1 NaN measure (threshold: 5)",
        })

        # Merge and recompute grade
        all_checks = agent_sc["checks"] + custom
        total = len(all_checks)
        passed_n = sum(1 for c in all_checks if c["passed"])
        score = round((passed_n / total * 100), 2) if total > 0 else 0.0

        grade = "F"
        for g, thr in sorted(QUALITY_GRADE_BANDS.items(), key=lambda x: x[1], reverse=True):
            if score >= thr:
                grade = g
                break

        return {
            **agent_sc,
            "checks": all_checks,
            "total_checks": total,
            "checks_passed": passed_n,
            "checks_failed": total - passed_n,
            "score": score,
            "grade": grade,
        }

    # ------------------------------------------------------------------
    # 5. Insight briefing
    # ------------------------------------------------------------------

    def insight_briefing(self, router, mart_df: pd.DataFrame) -> dict:
        """Compute deterministic facts and (optionally) narrate them with AI.

        Facts are computed by pandas — the LLM only narrates, never computes.
        WHY: hallucinated analytics are worse than template text; the fact sheet
        is the source of truth regardless of whether AI is available.

        Args:
            router:   AIRouter or None (None → template fallback).
            mart_df:  Assembled mart DataFrame.

        Returns:
            Dict: text, facts (cards), generated_by, latency_s.
        """
        spend_col = "medicare_spend_per_capita"
        diab_col = "diabetes_prevalence_pct"

        # The briefing's diabetes↔spend story needs the default diabetes column.
        # A custom measure set may omit it (5b) — degrade gracefully, never crash.
        if diab_col not in mart_df.columns or spend_col not in mart_df.columns:
            return {
                "text": (
                    "This mart was built with a custom measure set that does not "
                    "include diabetes prevalence, so the standard burden-vs-spend "
                    "briefing is not available. See the per-measure columns in the "
                    "mart table."
                ),
                "facts": {
                    "top3_spend_per_capita": [],
                    "bottom3_spend_per_capita": [],
                    "top3_diabetes_prevalence": [],
                    "corr_diabetes_spend": None,
                    "corr_obesity_spend": None,
                    "high_burden_low_spend_states": [],
                    "high_burden_low_spend_count": 0,
                },
                "generated_by": "rule-based fallback",
                "latency_s": 0.0,
            }

        # Work with rows that have both spend and diabetes values
        mart_clean = mart_df.dropna(subset=[spend_col, diab_col]).copy()

        def _top3(col: str) -> list[dict]:
            return mart_clean.nlargest(3, col)[["state_abbr", col]].to_dict("records")

        def _bot3(col: str) -> list[dict]:
            return mart_clean.nsmallest(3, col)[["state_abbr", col]].to_dict("records")

        top3_spend = _top3(spend_col)
        bot3_spend = _bot3(spend_col)
        top3_diab = _top3(diab_col)

        corr_diab_spend: Optional[float] = None
        if len(mart_clean) > 2:
            r = mart_clean[[diab_col, spend_col]].corr().iloc[0, 1]
            corr_diab_spend = round(float(r), 3) if pd.notna(r) else None

        corr_obesity_spend: Optional[float] = None
        if "obesity_prevalence_pct" in mart_df.columns:
            oc = mart_df.dropna(subset=["obesity_prevalence_pct", spend_col])
            if len(oc) > 2:
                r2 = oc[["obesity_prevalence_pct", spend_col]].corr().iloc[0, 1]
                corr_obesity_spend = round(float(r2), 3) if pd.notna(r2) else None

        # Quadrant: above-median disease burden AND below-median Medicare spend
        med_diab = mart_clean[diab_col].median()
        med_spend = mart_clean[spend_col].median()
        hbls = mart_clean[
            (mart_clean[diab_col] > med_diab) & (mart_clean[spend_col] < med_spend)
        ]["state_abbr"].tolist()

        facts = {
            "top3_spend_per_capita": top3_spend,
            "bottom3_spend_per_capita": bot3_spend,
            "top3_diabetes_prevalence": top3_diab,
            "corr_diabetes_spend": corr_diab_spend,
            "corr_obesity_spend": corr_obesity_spend,
            "high_burden_low_spend_states": hbls,
            "high_burden_low_spend_count": len(hbls),
        }

        def _fmt_states(rows: list[dict], val_col: str, fmt: str = ".0f") -> str:
            return ", ".join(
                f"{r['state_abbr']} ({r[val_col]:{fmt}})"
                for r in rows
            )

        fact_sheet = (
            f"Top 3 Medicare spend/capita: {_fmt_states(top3_spend, spend_col, ',.0f')}\n"
            f"Bottom 3 Medicare spend/capita: {_fmt_states(bot3_spend, spend_col, ',.0f')}\n"
            f"Top 3 diabetes prevalence: {_fmt_states(top3_diab, diab_col, '.1f')}%\n"
            f"Correlation diabetes ↔ spend: {corr_diab_spend}\n"
            f"High-burden / low-spend states ({len(hbls)}): "
            f"{', '.join(hbls[:8])}{' ...' if len(hbls) > 8 else ''}\n"
        )

        text: Optional[str] = None
        latency = 0.0
        provider_used = "none"

        if router is not None:
            from core.router import TaskType  # lazy
            from core.audit import log_ai_call  # lazy

            prompt = (
                "You are a healthcare data analyst writing for a payer audience. "
                "Using ONLY the facts below — do not invent numbers — "
                "write a ~180-word briefing. Cover: which states have the highest "
                "Medicare spend per capita, the burden-spend relationship, "
                "and one actionable policy implication.\n\n"
                f"{fact_sheet}"
            )
            t0 = time.monotonic()
            text, provider_used = router.generate(
                TaskType.BRIEFING, prompt, max_tokens=350
            )
            latency = time.monotonic() - t0

            log_ai_call(
                task=TaskType.BRIEFING,
                provider=provider_used,
                model=OLLAMA_MODEL if provider_used == "ollama" else provider_used,
                latency_s=latency,
                prompt_chars=len(prompt),
                redaction_count=0,
                success=text is not None,
            )

        if text is None:
            # Template fallback: assemble the same facts into readable prose
            top_states = ", ".join(r["state_abbr"] for r in top3_spend)
            bot_states = ", ".join(r["state_abbr"] for r in bot3_spend)
            hi_diab = ", ".join(r["state_abbr"] for r in top3_diab)
            text = (
                f"Analysis of {len(mart_df)} US states reveals wide variation in "
                f"Medicare spending per capita. Highest-spend states: {top_states}. "
                f"Lowest-spend states: {bot_states}. "
                f"States with highest diabetes prevalence: {hi_diab}. "
                f"Correlation between diabetes prevalence and Medicare spending: "
                f"{corr_diab_spend}. "
                f"{len(hbls)} states fall into the high-burden / low-spend quadrant "
                f"(above-median disease prevalence, below-median spend): "
                f"{', '.join(hbls[:5])}{'...' if len(hbls) > 5 else ''}. "
                f"These states may represent underserved populations with unmet care needs."
            )
            generated_by = "rule-based fallback"
        else:
            generated_by = (
                f"{OLLAMA_MODEL} (local)" if provider_used == "ollama"
                else f"{provider_used} (cloud)"
            )

        return {
            "text": text,
            "facts": facts,
            "generated_by": generated_by,
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # 6. Persist to DuckDB + outputs/mart/
    # ------------------------------------------------------------------

    def to_duckdb(self, mart_df: pd.DataFrame, table_name: str = "reporting_state_health") -> None:
        """Load mart into DuckDB and save parquet + JSON artifacts.

        Args:
            mart_df:    Assembled mart DataFrame.
            table_name: DuckDB table (and parquet artifact) name. Defaults to the
                        canonical ``reporting_state_health`` so existing callers
                        and tests are unaffected; the mart screen passes a
                        per-mart slug (e.g. ``mart_diabetes_vs_spend``) so each
                        built mart becomes its own queryable table.
        """
        mart_out = Path(_MART_DIR)
        mart_out.mkdir(parents=True, exist_ok=True)

        # DuckDB
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DuckDBManager(db_path=DATABASE_PATH) as db:
            db.load_dataframe(mart_df, table_name)

        # Parquet artifact
        mart_df.to_parquet(mart_out / f"{table_name}.parquet", index=False)

        # Build metadata (freshness + volume) — written when build() populated it.
        # The mart screen reads this for the "last built" badge, the underlying
        # volume line, and the low-coverage warning.
        if self.build_metadata:
            save_json(self.build_metadata, mart_out / "build_meta.json")

        _log.info("Mart saved: DuckDB table '%s' + %s", table_name, mart_out)
