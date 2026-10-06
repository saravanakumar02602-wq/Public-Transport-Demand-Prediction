"""
Shared configuration for the whole pipeline.

Every Python script (data_prep, features, train_ml, train_dl, evaluate, the
Streamlit app) imports its paths and split settings from here so there is a
single source of truth. The R notebooks can't import this directly, but they
replicate the same TEST_WEEKS / VAL_WEEKS logic against the same clean CSV,
which keeps the two sides comparable (see README "Keeping R and Python in
sync").
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_CSV_PATH = DATA_RAW_DIR / "cta_daily_raw.csv"
CLEAN_CSV_PATH = DATA_PROCESSED_DIR / "clean_v1.csv"
FEATURES_CSV_PATH = DATA_PROCESSED_DIR / "features_v1.csv"

BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.csv"
ML_METRICS_PATH = REPORTS_DIR / "ml_metrics.csv"
DL_METRICS_PATH = REPORTS_DIR / "dl_metrics.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Data source (Chicago Transit Authority open data, Socrata / SODA API)
# ---------------------------------------------------------------------------
# Dataset: "CTA - Ridership - Daily Boarding Totals" (system-wide, 2001-present)
# https://data.cityofchicago.org/Transportation/CTA-Ridership-Daily-Boarding-Totals/6iiy-9s97
SOCRATA_DATASET_ID = "6iiy-9s97"
SOCRATA_BASE_URL = f"https://data.cityofchicago.org/resource/{SOCRATA_DATASET_ID}.json"

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
DATE_COL = "service_date"
DAY_TYPE_COL = "day_type"          # W = Weekday, A = Saturday, U = Sunday/Holiday
MODE_COL = "mode"                  # bus | rail | total  (long-format "series id")
TARGET_COL = "rides"               # ridership count for that (date, mode)

MODES = ["bus", "rail", "total"]

# ---------------------------------------------------------------------------
# Train / validation / test split (time-based, never random — see spec)
# ---------------------------------------------------------------------------
TEST_WEEKS = 10
VAL_WEEKS = 6

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
LAGS = [1, 2, 7]
ROLLING_WINDOWS = [7, 14, 28]

RANDOM_SEED = 42
