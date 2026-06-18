# HealthPipe AI

**A multi-agent data pipeline that ingests healthcare data from public APIs and local files, runs it through four specialized AI agents, and produces clean, documented, analytics-ready output — all from a single CLI command.**

---

## The Problem

Healthcare data is messy. Public health APIs return inconsistent column names, nested JSON, and undocumented fields. CSV exports from government portals arrive in mixed encodings with duplicate rows and missing values. Before any analysis can begin, an engineer needs to manually profile the data, clean it, verify quality, and write documentation — a process that's tedious, error-prone, and rarely done well.

## What This Project Does

HealthPipe AI automates that entire workflow. It connects to real-world health data sources (WHO, FDA, local files), then passes the raw data through a pipeline of four agents — each responsible for one stage of the data engineering process:

1. **Profiler** — statistical analysis, outlier detection, correlation discovery, quality flags
2. **Transformer** — column standardization, deduplication, type conversion, null handling, with a full audit trail
3. **Quality Checker** — scored and graded quality assessment against configurable thresholds
4. **Documenter** — auto-generated data dictionary, schema, lineage, and usage notes (AI-enhanced when available)

The result: clean data loaded into DuckDB (a serverless analytics database), a styled **HTML dashboard** you can open in any browser, quality scorecards in JSON, and human-readable documentation in Markdown — all produced in one reproducible, idempotent run.

### What It Demonstrates

This is a portfolio project built to showcase data engineering patterns in practice:

| Pattern | Implementation | Why It Matters |
|---------|---------------|----------------|
| Abstract Base Class | All data sources share one interface | New sources require only 3 methods |
| Registry | Sources accessed by name, not import | Pipeline code never touches concrete classes |
| Idempotent Operations | `CREATE OR REPLACE TABLE` | Run 10 times, get the same result |
| Audit Trail | Every transform step logged with timestamp | Any change can be traced and reversed |
| Graceful Degradation | AI-first, rule-based fallback | Pipeline works without any AI model installed |
| Context Manager | `with DuckDBManager() as db:` | Connections auto-close even on errors |

---

## Architecture

```
 DATA SOURCES
 ============
 WHO GHO API --> OData pagination, rate limiting
 OpenFDA API --> Retry with exponential backoff
 Local CSV   --> Encoding fallbacks, chunked reading
       |
       v
 INGESTION LAYER
 ===============
 BaseSource (ABC) + Registry pattern
 connect() --> extract() --> pandas DataFrame
       |
       v
 AGENT PIPELINE (sequential)
 ===========================
 [1] Profiler Agent ......... stats, outliers, correlations, quality flags
       |
 [2] Transformer Agent ...... snake_case, dedup, type convert, null fill, audit log
       |
 [3] Quality Checker Agent .. 6 check categories, scored + graded (A/B/C/F)
       |
 [4] Documenter Agent ....... data dictionary, schema, lineage, AI descriptions
       |
       v
 OUTPUTS
 =======
 * HTML dashboard ......... styled report, open in any browser
 * DuckDB table ........... analytics-ready, queryable via SQL
 * Quality scorecard ...... JSON with pass/fail checks
 * Data dictionary ........ JSON + Markdown with column docs
 * Crew report ............ AI agent summaries (if --use-crew)
```

**Without AI installed**: The pipeline runs fully — profiling, transformation, quality checks, and documentation all work using pure pandas/numpy logic. Column descriptions fall back to 28 built-in pattern-matching rules.

**With AI installed**: The Documenter Agent sends each column name and sample values to a local LLM, which writes natural-language descriptions. In Crew mode, each agent also receives an AI-generated summary of its work for the final report.

---

## AI Integration (Optional)

HealthPipe AI supports an **optional local AI model** for enhanced documentation and agent summaries. This is not a cloud API — the model runs entirely on your machine. No data leaves your network.

### How It Works

| Component | Details |
|-----------|---------|
| Runtime | [Ollama](https://ollama.com) — a local LLM server (runs on Mac, Linux, Windows) |
| Model | [Gemma 3 4B](https://ollama.com/library/gemma3:4b) — Google's 4-billion parameter model, ~3 GB download |
| Interface | REST API at `localhost:11434` — no Python SDK needed |
| Privacy | 100% local. No API keys. No data sent to any external service |

### What AI Adds

| Feature | Without AI | With AI |
|---------|-----------|---------|
| Column descriptions | Rule-based pattern matching (28 built-in patterns) | LLM generates natural-language descriptions from column names + sample data |
| Agent summaries | Simple template strings | LLM writes professional summaries of each agent's findings |
| Pipeline behavior | Fully functional | Enhanced documentation quality |

### Setup (5 minutes)

```bash
# 1. Install Ollama (one-time)
# Download from https://ollama.com/download

# 2. Pull the model (one-time, ~3 GB download)
ollama pull gemma3:4b

# 3. Run the pipeline with AI — Ollama auto-detects
python main.py --source who --indicator life_expectancy --countries IND USA BRA

# 4. Or use Crew mode for full multi-agent orchestration with AI summaries
python main.py --source who --use-crew --indicator life_expectancy --countries IND USA BRA
```

If Ollama is not installed or not running, the pipeline silently falls back to rule-based logic. No errors, no configuration needed.

### Crew Mode: Multi-Agent Orchestration

The `--use-crew` flag activates a multi-agent orchestration layer inspired by frameworks like CrewAI. Each agent is defined with a persona (role, goal, backstory) and the Crew coordinates them in sequence:

| Agent | Persona | What It Does |
|-------|---------|-------------|
| Profiler | Senior Data Analyst | Profiles the raw dataset, flags quality issues |
| Transformer | Data Engineer | Cleans and standardizes with an audit trail |
| Quality Checker | QA Specialist | Scores and grades data against thresholds |
| Documenter | Tech Doc Lead | Generates dictionary, schema, lineage, usage notes |

After each agent completes its task, the Crew asks the local LLM to write a professional summary of the agent's work. The summaries are saved in a crew report JSON alongside the standard outputs.

> **Note**: The Crew framework is a custom lightweight implementation. CrewAI requires Python <3.14, so we built a compatible orchestrator using Python dataclasses that demonstrates the same multi-agent patterns without the dependency.

---

## Data Sources

### WHO Global Health Observatory API
- **Endpoint**: `https://ghoapi.azureedge.net/api`
- **Auth**: None required (open public API)
- **Pagination**: OData style (`$top`/`$skip`), 1-second rate limit between pages
- **Indicators supported**:

  | Friendly Name | WHO Code |
  |--------------|----------|
  | `life_expectancy` | WHOSIS_000001 |
  | `neonatal_mortality` | MDG_0000000001 |
  | `tuberculosis_incidence` | MDG_0000000020 |
  | `measles_immunization` | WHS4_100 |

### OpenFDA Drug Adverse Events API
- **Endpoint**: `https://api.fda.gov/drug/event.json`
- **Auth**: None required
- **Retry**: Exponential backoff on 429 errors (1s → 2s → 4s)
- **Processing**: Deeply nested JSON automatically flattened to tabular rows (patient info, drug details, reactions)

### Local Files (CSV, TSV, JSON)
- Encoding fallback chain: UTF-8 → Latin-1 → CP1252
- Files >10 MB automatically use chunked reading (50,000 rows per chunk)
- **Ships with**: a 500-row sample of the U.S. Chronic Disease Indicators dataset
  (`data/sample/chronic_disease_sample.csv`), enough to run the full pipeline out of the box.
- **Full dataset** (~309K rows, 84 MB): download "U.S. Chronic Disease Indicators"
  from [data.gov](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators), place the
  CSV anywhere, and pass it with `--filepath path/to/your_file.csv`. The full file is not
  committed to keep the repository lightweight.

---

## Development Environment

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.11+ | Tested on 3.11, 3.12, 3.14. Uses `type[X] \| None` syntax (3.10+) |
| pip | any | For installing dependencies from `requirements.txt` |
| Git | any | Version control |
| Ollama | latest (optional) | Only needed for AI-enhanced documentation |

### Dependencies

All dependencies are listed in `requirements.txt` with minimum compatible versions:

```
pandas>=2.1.0          # DataFrames, statistics, profiling
numpy>=1.26.0          # Numeric operations, correlation matrices
requests>=2.31.0       # HTTP client for API calls
duckdb>=0.10.0         # Serverless analytics database
pytest>=8.0.0          # Test framework
pytest-cov>=5.0.0      # Coverage reporting
```

**Optional** (not in `requirements.txt`):
- `pyspark` — only needed for the Spark transformer module (10M+ row datasets)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/healthpipe-ai.git
cd healthpipe-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v

# Run the pipeline
python main.py --source who --indicator life_expectancy --countries IND USA BRA
```

### Platform Notes

- Developed and tested on **Windows 11**. Runs on Linux and macOS without modification.
- All file paths use `pathlib.Path` — no OS-specific path separators anywhere in the code.
- The pipeline writes to `outputs/` and `healthpipe.duckdb` in the project root. Both are `.gitignore`d and regenerated each run.
- Logging goes to both console and `pipeline.log` (also `.gitignore`d).

---

## Quick Start

```bash
python main.py --source who --indicator life_expectancy --countries IND USA BRA
```

### Usage Examples

```bash
# WHO API — Life expectancy for India, USA, Brazil
python main.py --source who --indicator life_expectancy --countries IND USA BRA

# WHO API — Neonatal mortality, all countries, limit 500 records
python main.py --source who --indicator neonatal_mortality --max-records 500

# OpenFDA — Drug adverse events for aspirin
python main.py --source openfda --search aspirin --max-records 200

# Local CSV — 500-row sample of U.S. Chronic Disease Indicators (ships with the repo)
python main.py --source csv_local --filepath data/sample/chronic_disease_sample.csv --name chronic_disease

# Crew mode — Multi-agent orchestration with AI summaries (requires Ollama)
python main.py --source who --use-crew --indicator life_expectancy --countries IND USA BRA

# Custom dataset name
python main.py --source who --name my_who_data --indicator measles_immunization
```

---

## Pipeline Output

Each run produces:

| Output | Location | Format |
|--------|----------|--------|
| **HTML dashboard** | `outputs/reports/report_<name>.html` | Self-contained HTML (open in any browser) |
| Clean data | DuckDB table | Queryable via SQL |
| Profile report | `outputs/reports/` | JSON |
| Quality scorecard | `outputs/reports/` | JSON |
| Data dictionary | `outputs/docs/` | JSON + Markdown |
| Crew report (if `--use-crew`) | `outputs/reports/` | JSON |

### HTML Report

The HTML report is a single self-contained file — no external dependencies, no JavaScript, no server required. Open it in any browser, print it to PDF, or share it as an email attachment.

The dashboard includes:
- **Summary cards** — rows, columns, quality grade, completeness, duplicates, memory usage
- **Quality scorecard** — score bar with pass/fail breakdown, failed checks highlighted first
- **Data dictionary** — column names, types, nullability, sample values, AI-generated descriptions
- **Column profiles** — per-column statistics with badges for outliers, high null rates, and ID columns
- **Transformation audit trail** — numbered timeline of every cleaning step applied
- **Quality issues** — severity-coded table (critical, warning, info)
- **Strong correlations** — column pairs with visual correlation bars
- **Usage notes** — practical tips for analysts consuming the data

### Sample Quality Scorecard

```
Quality: 93.33% (Grade A) — 28/30 checks passed

Failed checks:
  - null_rate_patient_weight: 15.5% null (threshold: 20%)
  - extreme_outliers_age: 2 extreme outliers detected
```

### Quality Grades

| Grade | Score | Meaning |
|-------|-------|---------|
| A | >= 90% | Excellent — ready for analysis |
| B | >= 75% | Good — minor issues to note |
| C | >= 60% | Acceptable — review flagged items |
| F | < 60% | Poor — significant issues to address |

### Sample Data Dictionary (auto-generated)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| safety_report_id | object | No | Unique FDA safety report identifier |
| patient_age | float64 | Yes | Age of the patient at time of event |
| brand_name | object | No | Commercial brand name of the drug |
| reactions | object | No | Adverse reactions experienced |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11+ (developed on 3.14) | Core runtime |
| Data Processing | pandas, numpy | DataFrames, statistics, profiling |
| HTTP Client | requests | API calls with retry and backoff |
| Database | DuckDB | Serverless columnar analytics database |
| AI (optional) | Ollama + Gemma 3 4B | Local LLM for descriptions and summaries |
| Agent Framework | Custom (`crew.py`) | Multi-agent orchestration with dataclasses |
| Big Data (optional) | PySpark | Distributed transforms for 10M+ row datasets |
| Testing | pytest | 16 tests — unit, integration, pipeline |
| CI/CD | GitHub Actions | Automated test runs on push |

---

## Project Structure

```
healthpipe-ai/
├── core/
│   ├── config.py              # All paths, API URLs, thresholds, quality grades
│   ├── utils.py               # get_logger(), save_json(), load_json()
│   ├── database.py            # DuckDB manager with context manager support
│   ├── llm.py                 # Ollama REST client with graceful fallback
│   └── report.py              # HTML dashboard generator (self-contained, no deps)
│
├── ingestion/
│   ├── base_source.py         # Abstract base class: connect(), extract(), get_metadata()
│   ├── who_source.py          # WHO GHO API with OData pagination
│   ├── openfda_source.py      # OpenFDA API with retry + JSON flattening
│   ├── csv_source.py          # Local files with encoding fallbacks + chunking
│   └── registry.py            # Source registry: register/lookup by name
│
├── agents/
│   ├── profiler.py            # Stats, outliers (IQR), correlations, quality flags
│   ├── transformer.py         # 6-step pipeline with audit trail
│   ├── quality_checker.py     # 6 check categories, scored + graded scorecard
│   ├── documenter.py          # Data dictionary, lineage — AI-enhanced or rule-based
│   └── crew.py                # Multi-agent orchestration with AI summaries
│
├── spark_module/
│   └── spark_transformer.py   # PySpark transformer for 10M+ row datasets
│
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── test_ingestion.py      # 5 tests — CSV, registry, error handling
│   ├── test_agents.py         # 9 tests — all 4 agents
│   └── test_pipeline.py       # 2 integration tests — end-to-end
│
├── data/sample/               # Sample datasets + test fixtures
├── outputs/reports/           # Quality reports, crew reports (JSON)
├── outputs/docs/              # Data dictionary + docs (JSON + Markdown)
├── main.py                    # CLI entry point
├── requirements.txt           # Pinned dependencies
└── BUILD_LOG.md               # Phase-by-phase development journal
```

---

## PySpark Support (Optional)

For datasets beyond what pandas handles comfortably (10M+ rows), the project includes a PySpark transformer that mirrors the pandas version:

```python
from spark_module.spark_transformer import SparkTransformer

transformer = SparkTransformer()
spark_df = transformer.from_pandas(large_pandas_df)
clean_df = transformer.run(spark_df, "my_dataset")
transformer.stop()
```

| Aspect | pandas | PySpark |
|--------|--------|---------|
| Best for | < 1M rows | 10M+ rows |
| Runs on | Single machine | Distributed across cores/nodes |
| Median calculation | Exact | Approximate (within 1%) |
| Install | Included | `pip install pyspark` |

PySpark is an optional dependency — the rest of the project works without it.

---

## Adding a New Data Source

The Abstract Base Class + Registry pattern makes this straightforward:

```python
# 1. Create ingestion/my_source.py
from ingestion.base_source import BaseSource

class MySource(BaseSource):
    def __init__(self):
        super().__init__(name="my_source", description="My custom source")

    def connect(self) -> bool:
        return True  # test connectivity

    def extract(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(...)  # fetch and return data

    def get_metadata(self) -> dict:
        return {"name": self.name, "source_type": "api"}

# 2. Register in ingestion/registry.py
self.register(MySource())

# 3. Add CLI option in main.py
choices=["who", "openfda", "csv_local", "my_source"]
```

The new source will automatically work with all four agents, DuckDB loading, and the Crew orchestrator.

---

## Testing

```bash
# Run all tests (16 total)
pytest tests/ -v

# Skip network-dependent tests (for CI or offline work)
pytest tests/ -v -m "not network"

# Run a specific test file
pytest tests/test_agents.py -v
```

| Test File | Count | Covers |
|-----------|-------|--------|
| `test_ingestion.py` | 5 | CSV reading, registry, missing file handling |
| `test_agents.py` | 9 | Profiler, transformer, quality checker, documenter |
| `test_pipeline.py` | 2 | Full end-to-end pipeline integration |

CI runs automatically on push via GitHub Actions (Python 3.11, network tests excluded).

---

## Development Process

This project was designed, architected, and directed by me as a portfolio piece to demonstrate real-world data engineering skills. Here's how the work broke down:

**What I did:**
- Conceived the project idea and defined the scope — a multi-agent healthcare data pipeline targeting real public health APIs
- Designed the overall architecture: the ingestion layer with ABC + Registry, the 4-agent pipeline flow, the DuckDB persistence strategy, and the Crew orchestration model
- Made every technical decision: which APIs to use, which design patterns to apply, what quality thresholds make sense for healthcare data, how agents should pass data between stages
- Chose the tech stack (DuckDB over SQLite for analytics workloads, Ollama for local-only AI, custom Crew framework over CrewAI due to Python 3.14 incompatibility)
- Debugged all integration issues: duplicate column name collisions after snake_case conversion, pandas 3.x deprecation warnings, encoding failures on government CSV files, CrewAI version conflicts
- Defined test scenarios and quality criteria, reviewed all generated code for correctness
- Directed the phased build order (Phases 0–7) and decided what belonged in each phase

**Where I used AI assistance (Claude Code):**
- Generating boilerplate and repetitive code — docstrings, type hints, argparse setup, HTML/CSS template strings
- Writing the initial implementation of modules once I specified the interface and behavior
- Scaffolding test files and CI configuration from my test plan
- Formatting documentation (README, BUILD_LOG) from my notes and bullet points
- Code review passes for consistency (import ordering, unused variables, missing type annotations)

In short: the architecture, design decisions, debugging, and technical direction are mine. The AI helped me move faster on the parts that are tedious but straightforward — the kind of work where knowing *what* to build matters more than typing it out.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
