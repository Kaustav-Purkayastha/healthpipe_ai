# HealthPipe AI — Build Log

A running journal of what was built each phase, decisions made, and lessons learned.

---

## Phase Checklist

| Phase | Description                          | Status      |
|-------|--------------------------------------|-------------|
| 0     | Scaffold: folder structure, config   | ✅ Complete |
| 1     | Ingestion: WHO + OpenFDA + CSV       | ✅ Complete |
| 2     | Agents + DuckDB                      | ✅ Complete |
| 3     | (merged into Phase 2)                | ✅ Complete |
| 4     | AI integration (Ollama + Crew)       | ✅ Complete |
| 5     | Tests + GitHub Actions CI            | ✅ Complete |
| 6     | PySpark + README + Polish            | ✅ Complete |
| 7     | HTML Report Dashboard                | ✅ Complete |

---

## Phase 0 — Scaffold (2026-03-15)

### What was built
- Full folder structure created: `core/`, `ingestion/`, `agents/`, `spark_module/`, `outputs/`, `tests/`, `data/sample/`, `.github/workflows/`
- All `__init__.py` placeholder files added to make each folder a Python package
- `requirements.txt` with all dependencies pinned to minimum compatible versions
- `.gitignore` covering venv, `__pycache__`, outputs, DuckDB files, `.env`
- `core/config.py` — single source of truth for all paths, API URLs, quality thresholds, and logging settings
- `core/utils.py` — `get_logger()`, `save_json()`, `load_json()`, `timestamp_string()`
- `data/sample/sample_health_data.csv` — 10 rows of fake WHO-style health data for testing

### Design decisions
- **`pathlib.Path` everywhere**: avoids OS-specific path separator bugs. `ROOT_DIR = Path(__file__).parent.parent` anchors all paths to the project root regardless of where the script is called from.
- **Centralized config**: every threshold, URL, and constant lives in `config.py`. Changing a value in one place propagates everywhere.
- **Logger deduplication guard**: `if logger.handlers: return logger` prevents duplicate log lines when a module is imported multiple times (common in testing).
- **`default=str` in `json.dump`**: silently converts `datetime` and `Path` objects to strings instead of crashing with `TypeError`.

### Next steps
- Phase 1: build `ingestion/base_source.py`, `who_source.py`, `openfda_source.py`, `csv_source.py`, `registry.py`

---

## Phase 1 — Ingestion Layer (2026-03-22)

### What was built
- `ingestion/base_source.py` — Abstract base class (ABC) defining the `connect()`, `extract()`, `get_metadata()` interface that all sources must implement
- `ingestion/who_source.py` — WHO GHO API source with OData pagination (`$top`/`$skip`), rate limiting (1s between requests), column renaming (`SpatialDim` → `country_code`, etc.), friendly indicator name resolution
- `ingestion/openfda_source.py` — OpenFDA Drug Adverse Events API source with `limit`/`skip` pagination, exponential backoff retry on 429 (1s → 2s → 4s), `_flatten_event()` to convert deeply nested JSON into flat rows
- `ingestion/csv_source.py` — Local file reader supporting CSV/TSV/JSON, encoding fallback chain (utf-8 → latin-1 → cp1252), chunked reading for large files via `pd.read_csv(chunksize=...)`
- `ingestion/registry.py` — Registry pattern: `register()`, `get()`, `list_sources()`, `check_all_connections()`. Auto-registers WHO, OpenFDA, CSV on init.
- `test_phase1.py` — Verification script (temporary, delete after confirming)

### Test results
- **Registry**: 3/3 sources registered ✅
- **CSV**: 309,215 rows × 34 columns from U.S. Chronic Disease Indicators dataset ✅
- **WHO API**: Connected, fetched life expectancy data for IND/USA/BRA ✅
- **OpenFDA API**: Connected, fetched 10 aspirin adverse event records with flattened JSON ✅
- **check_all_connections()**: 3/3 sources OK ✅

### Design decisions
- **Abstract Base Class pattern**: guarantees every source has the same interface. Code that works with `BaseSource` works with any source — new sources (e.g., CDC, FHIR) just need to inherit and implement 3 methods.
- **Registry pattern**: decouples pipeline code from concrete source classes. `registry.get("who")` instead of `from ingestion.who_source import WHOSource`.
- **Encoding fallback chain**: government datasets often use non-UTF-8 encodings. Trying multiple encodings prevents `UnicodeDecodeError` crashes without user intervention.
- **Chunked CSV reading**: the 88 MB dataset is read in 50,000-row chunks (7 chunks total), keeping memory usage manageable.
- **`dict.get("key", default)` everywhere**: safe access pattern — never crashes on missing keys in API responses.

### Next steps
- Phase 2: build `core/database.py` for DuckDB (connect, load_dataframe, query, list_tables, close)

---

## Phase 2 — Agents + DuckDB (2026-03-29)

### What was built
- `agents/profiler.py` — Overview stats, per-column profiling (numeric/string/datetime), IQR outlier detection, correlation analysis (|r| > 0.7), quality issue flags
- `agents/transformer.py` — 6-step pipeline: standardize_columns (snake_case), remove_duplicates, convert_types, handle_nulls, clean_text, add_metadata. Full audit log.
- `agents/quality_checker.py` — 6 check categories (completeness, duplicates, null rates, type consistency, value ranges, uniqueness). Outputs scored/graded scorecard.
- `agents/documenter.py` — Data dictionary, schema, lineage, quality summary, usage notes. Outputs JSON + Markdown.
- `core/database.py` — DuckDB manager with connect, load_dataframe (CREATE OR REPLACE TABLE), query, get_table_info, list_tables, close. Context manager support.
- `test_phase2.py` — End-to-end pipeline test (temporary)

### Test results (WHO life_expectancy → full pipeline)
- **Ingest**: 4 records from WHO API for 5 countries ✅
- **Profile**: 13 quality issues flagged, 3 strong correlations found ✅
- **Transform**: 6 steps completed, 20 columns renamed, 7 null columns filled ✅
- **Quality**: Score 100.0%, Grade A, 59/59 checks passed ✅
- **Document**: JSON + Markdown generated with 28-column data dictionary ✅
- **DuckDB**: Table loaded, SQL queries work, schema visible ✅

### Bug fixed during build
- **Duplicate column names after snake_case**: WHO API returns both `Value` (string like "72.3 [72.0-72.6]") and `NumericValue` → both became `value` after renaming. Fixed by adding deduplication: second occurrence becomes `value_2`.

### Design decisions
- **No AI**: All agents are pure pandas/numpy — no LLM needed for profiling, transforming, or quality checking. LLM integration comes in Phase 4.
- **Audit trail**: TransformerAgent logs every step with step number, action, detail, and timestamp. The DocumenterAgent includes this log in the lineage section.
- **Idempotent DuckDB**: `CREATE OR REPLACE TABLE` means running the pipeline 10 times produces the same result — no duplicate table errors.
- **Context manager**: `with DuckDBManager() as db:` auto-closes the connection even if an error occurs mid-pipeline.

### Next steps
- Phase 4: CrewAI + Ollama integration

---

## Phase 3 — Pipeline Orchestration (2026-04-05)

### What was built
- `main.py` — CLI entry point with argparse, `run_pipeline()` orchestrator, and `sanitize_table_name()` helper
- 6-step pipeline: ingest → profile → transform → quality check → document → DuckDB load
- Full help text with usage examples (`python main.py --help`)

### Test results (all 3 sources)
| Command | Rows | Quality | Grade |
|---------|------|---------|-------|
| `--source who --indicator life_expectancy --countries IND USA BRA` | 2 | 100.0% | A |
| `--source openfda --search aspirin --max-records 200` | 200 | 93.3% | A |
| `--source csv_local --filepath data/sample/U.S._Chronic_Disease_Indicators.csv` | 309,215 | 77.8% | B |

### Design decisions
- **CLI maps "csv_local" → "csv"**: the registry uses "csv" internally, but the CLI uses "csv_local" for clarity (avoids confusion with the `csv` stdlib module).
- **Auto-detect large files**: if a CSV file is >10 MB, `chunk_size=50000` is set automatically — the user doesn't need to know about chunked reading.
- **`sanitize_table_name()`**: converts any dataset name to a valid DuckDB table name (lowercase, underscores only). "WHO Life-Expectancy (2020)" becomes "who_life_expectancy_2020".
- **DuckDB accumulates tables**: each pipeline run adds/replaces its table. After all 3 tests: `['chronic_disease', 'openfda', 'who', 'who_life_expectancy']`.

### Next steps
- Phase 4: AI integration (Ollama + Crew orchestration)

---

## Phase 4 — AI Integration: Ollama + Crew (2026-04-13)

### What was built
- `core/llm.py` — Ollama helper: `query_ollama(prompt)` sends prompts to local gemma3:4b via HTTP, `is_ollama_available()` health check. Graceful fallback to None if Ollama is down.
- `agents/crew.py` — Custom multi-agent orchestrator inspired by CrewAI. Defines `Agent` (role, goal, backstory), `Task`, `Crew`, and `CrewResult` dataclasses. `run_crew()` executes the 4 agents in sequence with AI-generated summaries.
- Updated `agents/documenter.py` — `_infer_description()` now tries AI first (Ollama generates descriptions from column name + sample values), falls back to rule-based patterns.
- Updated `main.py` — `--use-crew` flag delegates to Crew orchestration mode.
- Updated `requirements.txt` — removed crewai/ollama Python packages (not needed).

### Why not CrewAI?
CrewAI requires Python <3.14 but this project uses Python 3.14.3. Instead of downgrading, we built a lightweight custom orchestrator that demonstrates the same multi-agent design patterns (Agent personas, Task chaining, sequential Crew execution) without the dependency.

### Test results
| Mode | AI Descriptions | AI Summaries | Quality | Grade |
|------|----------------|--------------|---------|-------|
| Direct (`--source who`) | ✅ 28 columns via Ollama | N/A | 100.0% | A |
| Crew (`--source who --use-crew`) | ✅ 28 columns via Ollama | ✅ 4 agent summaries | 100.0% | A |

### AI-generated agent summaries (Crew mode)
- **Profiler (Senior Data Analyst)**: "The initial dataset profile revealed 13 data quality issues across two rows, necessitating further investigation..."
- **Transformer (Data Engineer)**: "I successfully cleaned and standardized healthcare data...accompanied by a comprehensive, reversible audit trail..."
- **Quality Checker (QA Specialist)**: "The dataset demonstrated excellent data quality...achieving a Grade A performance."
- **Documenter (Tech Doc Lead)**: "I successfully delivered comprehensive data documentation encompassing 28 columns and five usage notes..."

### Design decisions
- **AI-first with rule fallback**: documenter tries Ollama for every column, falls back silently to pattern matching. No crash if Ollama is down.
- **One-time availability check**: `_OLLAMA_READY` is set at module import time to avoid 28+ connection checks (one per column).
- **Dataclass-based framework**: `Agent`, `Task`, `Crew`, `CrewResult` are Python dataclasses — lightweight, typed, and easy to understand.
- **Closures for task state**: each task function is a closure that reads/writes a shared `pipeline_state` dict, so data flows naturally between sequential tasks.

### Next steps
- Phase 5: pytest + GitHub Actions CI

---

## Phase 5 — Tests + CI/CD (2026-04-26)

### What was built
- `data/sample/test_fixture.csv` — 20-row fixture with intentional quality issues (3 nulls, 1 duplicate, negative age, mixed casing, empty string)
- `tests/conftest.py` — shared pytest fixtures (`test_fixture_path`, `sample_dataframe`)
- `tests/test_ingestion.py` — 5 tests: CSV read fixture, read real dataset, missing file, registry list, registry unknown
- `tests/test_agents.py` — 9 tests: profiler (structure, nulls), transformer (dupes, columns, nulls), quality checker (scoring, negatives), documenter (dictionary, markdown)
- `tests/test_pipeline.py` — 2 integration tests: fixture end-to-end, real data end-to-end
- `.github/workflows/ci.yml` — GitHub Actions CI (Python 3.11, skips `@pytest.mark.network` tests)

### Bug fixed
- **Pandas 3.x deprecation warning**: `select_dtypes(include=["object"])` no longer implicitly includes `str` dtype columns in pandas 3. Fixed by using `include=["object", "string"]` in `quality_checker.py`.

### Test results
```
16 passed, 0 warnings in 176s
```
| Test File | Tests | Status |
|-----------|-------|--------|
| `test_ingestion.py` | 5 | ✅ All pass |
| `test_agents.py` | 9 | ✅ All pass |
| `test_pipeline.py` | 2 | ✅ All pass |

### Next steps
- Phase 6: PySpark transformer (`spark_module/`)

---

## Phase 6 — PySpark + README + Polish (2026-05-04)

### What was built
- `spark_module/spark_transformer.py` — PySpark version of the pandas TransformerAgent. Same 4-step pipeline (standardize_columns, remove_duplicates, handle_nulls, add_metadata_columns) but using Spark's distributed engine. Includes `from_pandas()` / `to_pandas()` conversion helpers.
- `README.md` — Full project documentation: ASCII architecture diagram, tech stack table, data source details, quick start guide, usage examples for all 3 sources + crew mode, sample output (quality scorecard, data dictionary, quality grades), project structure tree, extensibility guide ("Adding a New Data Source"), design patterns table, testing instructions.
- `.gitignore` — Added Spark working directories (`spark-warehouse/`, `metastore_db/`, `derby.log`), `pipeline.log`, `test_phase*.py` (temporary test scripts from earlier phases), and distribution/packaging dirs.
- **Code review**: fixed missing type hints on `DuckDBManager.__exit__()` parameters (`exc_type: type[BaseException] | None`, etc.)

### Test results
```
16 passed, 0 warnings in 184.52s
```
All existing tests continue to pass after Phase 6 changes. No regressions.

### Design decisions
- **Optional PySpark import**: `try: from pyspark.sql import ...` with `PYSPARK_AVAILABLE` flag. The rest of the project works without PySpark installed — it's only needed if you want distributed transforms.
- **Parallel API to pandas transformer**: same method names (`standardize_columns`, `remove_duplicates`, `handle_nulls`, `add_metadata_columns`), same audit log, same snake_case + deduplication logic. Makes it easy to compare the two approaches side by side.
- **Approximate median**: `df.approxQuantile(col, [0.5], 0.01)` instead of exact median. The 0.01 relative error is within 1% of the true median but much faster on large distributed datasets.
- **`F.lit()` for constant columns**: PySpark's way of adding a column where every row has the same value (used for `_source` metadata column).
- **`.toDF(*names)` for bulk rename**: more efficient than calling `.withColumnRenamed()` N times in a loop.

### What's different from the pandas version
| Aspect | pandas (transformer.py) | PySpark (spark_transformer.py) |
|--------|------------------------|-------------------------------|
| Engine | Single machine, in-memory | Distributed across cores/nodes |
| Sweet spot | < 1M rows | 10M+ rows |
| Median | `df[col].median()` (exact) | `.approxQuantile()` (approximate) |
| Immutability | Modifies in place | Returns new DataFrame each step |
| Session | None needed | SparkSession (must `.stop()`) |

## Phase 7 — HTML Report Dashboard (2026-05-10)

### What was built
- `core/report.py` — `ReportGenerator` class that produces a single self-contained HTML file (all CSS inline, no external dependencies) from pipeline outputs. The report is a styled dashboard with 8 sections:
  1. **Header** — dataset name, row/column count, memory footprint, generation timestamp
  2. **Summary cards** — 6 metric cards (rows, columns, quality grade, completeness, duplicates, memory) with color-coded borders per grade
  3. **Quality Scorecard** — score bar + full checks table with PASS/FAIL badges, failed checks sorted first
  4. **Data Dictionary** — column name, type, nullable, null count, unique count, sample values, description
  5. **Column Profiles** — per-column statistics (numeric: mean/median/std/range, string: length stats + top values, datetime: range), with badges for outliers, high nulls, and ID columns
  6. **Transformation Audit Trail** — numbered step timeline with action, detail, and timestamp
  7. **Quality Issues** — severity-coded table (CRITICAL/WARNING/INFO badges)
  8. **Strong Correlations** — column pairs with correlation value and visual bar
  9. **Usage Notes** — practical tips for data consumers
  10. **Footer** — generation info
- Updated `main.py` — pipeline now runs 7 steps (added Step 6: HTML report between documentation and DuckDB load). Summary log includes HTML report path.
- Updated `agents/crew.py` — Crew mode also generates the HTML report after all agents complete, before DuckDB load.

### Test results
```
16 passed in 174.30s
```
All existing tests pass. Pipeline tested end-to-end with WHO data — HTML report generated successfully at `outputs/reports/report_who.html`.

### Design decisions
- **Self-contained HTML**: all CSS is inline inside a `<style>` tag. No external stylesheets, fonts, or JavaScript. The file opens in any browser, works offline, and can be emailed or shared as a single attachment.
- **HTML escaping via `_esc()`**: all user-facing text is escaped (`&`, `<`, `>`, `"`) to prevent rendering issues from data values that contain HTML characters.
- **Graceful degradation**: every section handles missing data — if profile, scorecard, or dictionary data is `None`, the section shows a clean "No data available" placeholder instead of crashing.
- **Failed checks first**: the quality scorecard sorts FAIL results above PASS results so issues are immediately visible.
- **Color-coded grades**: Grade A = green, B = blue, C = yellow, F = red — applied to the card border, score bar, and value text.
- **Print-friendly**: `@media print` styles remove shadows and add borders so the report prints cleanly to PDF.
- **No JavaScript**: the report is pure HTML + CSS. No interactive elements, no dependencies, no security concerns from inline scripts.

<!-- Add new phase entries above this line -->
