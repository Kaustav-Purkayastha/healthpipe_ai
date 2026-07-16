"""
core/config.py — Single source of truth for all HealthPipe AI v2 constants.

Loads .env at import time so every module that imports from here gets secrets
automatically. Never import secrets directly from os.environ elsewhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env (silently ignored if the file doesn't exist).
load_dotenv()

# ---------------------------------------------------------------------------
# Root & derived paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
SAMPLE_DIR: Path = DATA_DIR / "sample"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
DOCS_DIR: Path = OUTPUTS_DIR / "docs"

# ---------------------------------------------------------------------------
# Secrets from .env
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
MASTER_USERNAME: str = os.environ.get("MASTER_USERNAME", "")
MASTER_PASSWORD: str = os.environ.get("MASTER_PASSWORD", "")
# Allow override of the default model via .env (e.g. for newer flash versions)
_gemini_model_env: str = os.environ.get("GEMINI_MODEL", "")
CENSUS_API_KEY: str = os.environ.get("CENSUS_API_KEY", "")

# ---------------------------------------------------------------------------
# Ollama (local LLM)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "gemma3:4b"
OLLAMA_TIMEOUT_SECONDS: int = 120          # CPU inference; first call pages model into RAM

# ---------------------------------------------------------------------------
# Gemini (cloud LLM, chat only; key from .env)
# ---------------------------------------------------------------------------
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL: str = _gemini_model_env if _gemini_model_env else "gemini-3.1-flash-lite"  # overridable via GEMINI_MODEL in .env

# Client-side rate limits that PROTECT the free-tier quota: when exceeded, chat
# auto-falls back to local gemma instead of provoking a server-side 429.  Both
# are overridable via .env.  The per-minute window is in-memory; the daily count
# is persisted (date-keyed) so it survives restarts and resets at midnight.
GEMINI_RPM_LIMIT: int = int(os.environ.get("GEMINI_RPM_LIMIT", "5"))       # requests/minute
GEMINI_DAILY_LIMIT: int = int(os.environ.get("GEMINI_DAILY_LIMIT", "200"))  # requests/day
GEMINI_USAGE_FILE: Path = OUTPUTS_DIR / "gemini_usage.json"

# ---------------------------------------------------------------------------
# WHO GHO OData API
# ---------------------------------------------------------------------------
WHO_BASE_URL: str = "https://ghoapi.azureedge.net/api"

# ---------------------------------------------------------------------------
# OpenFDA
# ---------------------------------------------------------------------------
OPENFDA_BASE_URL: str = "https://api.fda.gov/drug/event.json"

# ---------------------------------------------------------------------------
# CMS Medicare (data.cms.gov Data API v1; keyless; values arrive as strings)
# ---------------------------------------------------------------------------
CMS_BASE_URL: str = "https://data.cms.gov/data-api/v1/dataset"
CMS_PROVIDER_SUMMARY_ID: str = "8889d81e-2ee7-448f-8713-f071038289b5"  # by Provider, CY2024, ~1.3M rows
CMS_PROVIDER_SERVICE_ID: str = "92396110-2aed-4d63-a6a2-5d6207d46a29"  # by Provider & Service, 9.78M rows
CMS_PAGE_SIZE: int = 1000                  # API hard max is 5000
CMS_RATE_LIMIT_SECONDS: float = 0.5

# ---------------------------------------------------------------------------
# CDC Socrata APIs (keyless; JSON field names are LOWERCASE)
# ---------------------------------------------------------------------------
CDC_CDI_URL: str = "https://data.cdc.gov/resource/hksd-2xuw.json"     # Chronic Disease Indicators
CDC_BRFSS_URL: str = "https://data.cdc.gov/resource/dttw-5yxu.json"   # BRFSS Prevalence 2011+
# CDC PLACES — county-level model-based estimates (2025 release carries 2023 data).
# Live-verified: lowercase fields; locationid is a 5-digit county FIPS STRING;
# datavaluetypeid is MIXED case ('CrdPrv' crude / 'AgeAdjPrv' age-adjusted).
CDC_PLACES_URL: str = "https://data.cdc.gov/resource/swc5-untb.json"
SOCRATA_PAGE_SIZE: int = 1000

# ---------------------------------------------------------------------------
# US Census (REQUIRES free key in .env as CENSUS_API_KEY)
# ---------------------------------------------------------------------------
CENSUS_ACS5_URL: str = "https://api.census.gov/data/2023/acs/acs5"
CENSUS_POPULATION_VAR: str = "B01003_001E"

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
# Grades: >=90 → A, >=75 → B, >=60 → C, else F
QUALITY_GRADE_BANDS: dict[str, float] = {"A": 90.0, "B": 75.0, "C": 60.0}
QUALITY_GATE_DEFAULT_MIN_GRADE: str = "B"

# Alias used by the v1-ported agent code (quality_checker imports QUALITY_GRADES).
QUALITY_GRADES: dict[str, float] = QUALITY_GRADE_BANDS

# --- Agent analysis thresholds ---
MAX_NULL_PERCENTAGE: float = 20.0        # warn when a column exceeds this % nulls
OUTLIER_IQR_MULTIPLIER: float = 1.5     # IQR fence multiplier for outlier detection
MAX_DUPLICATE_PERCENTAGE: float = 5.0   # warn when dup row rate exceeds this %
MIN_COMPLETENESS_SCORE: float = 70.0    # minimum acceptable overall completeness %
EXTREME_OUTLIER_STD: float = 5.0        # z-score threshold for extreme outliers

# --- DuckDB default path ---
DATABASE_PATH: Path = OUTPUTS_DIR / "healthpipe.duckdb"

# ---------------------------------------------------------------------------
# Canonical US state list (50 states + DC) — 51 entries
# ---------------------------------------------------------------------------
STATE_ABBRS: list[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",  # District of Columbia
]

# FIPS numeric code (zero-padded 2-digit string) → USPS 2-letter abbreviation.
# Source: US Census Bureau FIPS state codes.  Includes DC (11).
FIPS_TO_STATE: dict[str, str] = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

# Location codes to exclude from state-level analysis:
#   US  — national aggregate rows (CMS, CDC)
#   UW  — BRFSS national median rows
#   GU  — Guam (territory)
#   PR  — Puerto Rico (territory)
#   VI  — U.S. Virgin Islands (territory)
#   ZZ  — CMS placeholder for foreign/unknown provider addresses
EXCLUDED_LOCATION_CODES: set[str] = {"US", "UW", "GU", "PR", "VI", "ZZ"}
