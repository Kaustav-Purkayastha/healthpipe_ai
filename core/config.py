"""
config.py — Centralized settings for the entire pipeline.

All paths, API endpoints, thresholds, and logging configuration live here.
Importing this module anywhere gives you a single source of truth.
"""

import logging
from pathlib import Path  # Path is safer than os.path — handles slashes on any OS

# ---------------------------------------------------------------------------
# ROOT PATHS
# ---------------------------------------------------------------------------

# Path(__file__) is the absolute path to this file (config.py).
# .parent goes up one folder (core/), .parent again goes up to the project root.
ROOT_DIR: Path = Path(__file__).parent.parent

DATA_DIR: Path = ROOT_DIR / "data"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
DOCS_DIR: Path = OUTPUTS_DIR / "docs"

# DuckDB stores everything in one file — no database server needed
DATABASE_PATH: Path = ROOT_DIR / "healthpipe.duckdb"

# ---------------------------------------------------------------------------
# API CONFIGURATION
# ---------------------------------------------------------------------------

WHO_BASE_URL: str = "https://ghoapi.azureedge.net/api"

# WHO indicators we care about, mapped to human-readable names
WHO_INDICATORS: dict = {
    "WHOSIS_000001": "life_expectancy",
    "MDG_0000000001": "neonatal_mortality",
    "MDG_0000000020": "tuberculosis_incidence",
    "WHS4_100": "measles_immunization",
}

# Seconds to wait between consecutive WHO API requests (respect rate limits)
WHO_RATE_LIMIT_SECONDS: float = 1.0

# Number of records to request per WHO API page
WHO_PAGE_SIZE: int = 1000

OPENFDA_BASE_URL: str = "https://api.fda.gov"
OPENFDA_DRUG_EVENT_ENDPOINT: str = "/drug/event.json"

# Max records per OpenFDA request (API hard cap is 100)
OPENFDA_PAGE_SIZE: int = 100

# Seconds to wait between consecutive OpenFDA requests
OPENFDA_RATE_LIMIT_SECONDS: float = 0.5

# Retry settings for OpenFDA (handles 429 Too Many Requests)
OPENFDA_MAX_RETRIES: int = 3
OPENFDA_RETRY_BACKOFF_BASE: float = 1.0  # doubles each attempt: 1s, 2s, 4s

# ---------------------------------------------------------------------------
# DATA QUALITY THRESHOLDS
# ---------------------------------------------------------------------------

# Flag any column whose null% exceeds this value
MAX_NULL_PERCENTAGE: float = 20.0

# Flag any dataset whose duplicate row % exceeds this value
MAX_DUPLICATE_PERCENTAGE: float = 5.0

# Overall completeness must meet this minimum to pass quality check
MIN_COMPLETENESS_SCORE: float = 70.0

# IQR multiplier for outlier detection (standard = 1.5)
OUTLIER_IQR_MULTIPLIER: float = 1.5

# Number of standard deviations beyond which a value is an "extreme outlier"
EXTREME_OUTLIER_STD: float = 5.0

# Quality grade thresholds (score is 0-100)
QUALITY_GRADES: dict = {
    "A": 90.0,  # >= 90 → A
    "B": 75.0,  # >= 75 → B
    "C": 60.0,  # >= 60 → C
    # anything below 60 → F
}

# ---------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------------------------

# Log level applied to all loggers created by get_logger() in utils.py
LOG_LEVEL: int = logging.INFO

# Format: timestamp | severity | module name | message
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Timestamp format inside log messages
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Optional: write logs to a file in addition to the console (set to None to disable)
LOG_FILE: Path | None = ROOT_DIR / "pipeline.log"
