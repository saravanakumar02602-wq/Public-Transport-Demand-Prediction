"""
Phase 3 — Feature engineering.

Builds, per (mode) series: lag features, rolling mean/std, cyclical
day-of-week encodings, calendar/holiday flags, and a leakage-safe,
TIME-based train/val/test split.

Run directly:
    python -m src.features
or:
    python src/features.py

No-leakage rule: every feature at row t is computed only from data at or
before t (lags, rolling windows use `.shift(1)` before rolling so the
current day's own value is never included in its own rolling mean). The
split is by date, not random — the test window is always the most recent
TEST_WEEKS of the series, matching notebooks/03_baseline_models.Rmd so the
R and Python sides are scored on an identical held-out period.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holidays
import numpy as np
import pandas as pd

from src import config


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = df[config.DATE_COL]
    df["day_of_week"] = dates.dt.dayofweek  # Monday=0
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = dates.dt.month
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)

    us_holidays = holidays.UnitedStates(state="IL", years=range(dates.dt.year.min(), dates.dt.year.max() + 1))
    df["is_holiday"] = dates.dt.date.astype("O").isin(us_holidays).astype(int)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dow = df["day_of_week"]
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    month = df["month"]
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def add_lag_features(df: pd.DataFrame, group_col: str, target_col: str, lags=config.LAGS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, group_col: str, target_col: str,
                          windows=config.ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    # shift(1) first so the rolling window never includes the current row's
    # own (future-relative-to-the-lag-features-below) target value.
    shifted = df.groupby(group_col)[target_col].shift(1)
    for w in windows:
        df[f"roll_mean_{w}"] = shifted.groupby(df[group_col]).rolling(w, min_periods=max(2, w // 2)) \
            .mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = shifted.groupby(df[group_col]).rolling(w, min_periods=max(2, w // 2)) \
            .std().reset_index(level=0, drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates Phase 3 end to end on the tidy long-format clean data."""
    df = df.sort_values([config.MODE_COL, config.DATE_COL]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df, config.MODE_COL, config.TARGET_COL)
    df = add_rolling_features(df, config.MODE_COL, config.TARGET_COL)

    feature_cols = [c for c in df.columns if c.startswith(("lag_", "roll_mean_", "roll_std_"))]
    # Drop rows missing lag/rolling history AND rows whose own target is
    # missing (e.g. a calendar day the source simply didn't report). Checking
    # feature_cols alone isn't enough: a day can have a perfectly complete
    # lookback (its neighbors are fine) while its own `rides` value is NaN,
    # which would otherwise reach model training as an unusable label.
    n_before = len(df)
    df = df.dropna(subset=feature_cols + [config.TARGET_COL])
    print(f"[features] Dropped {n_before - len(df)} rows with incomplete "
          f"lag/rolling history (needs {max(config.LAGS + config.ROLLING_WINDOWS)} "
          f"prior days per mode) or a missing target value.")
    return df.reset_index(drop=True)


def time_based_split(df: pd.DataFrame, date_col: str = config.DATE_COL,
                      test_weeks: int = config.TEST_WEEKS,
                      val_weeks: int = config.VAL_WEEKS):
    """
    Splits by TIME, not randomly. Test = last `test_weeks` of the series.
    Val = the `val_weeks` immediately before that (used for hyperparameter
    tuning in Phase 5). Everything earlier is train.

    Returns (train_df, val_df, test_df).
    """
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(weeks=test_weeks) + pd.Timedelta(days=1)
    val_start = test_start - pd.Timedelta(weeks=val_weeks)

    train_df = df[df[date_col] < val_start]
    val_df = df[(df[date_col] >= val_start) & (df[date_col] < test_start)]
    test_df = df[df[date_col] >= test_start]
    return train_df, val_df, test_df


def main(argv=None) -> None:
    clean = pd.read_csv(config.CLEAN_CSV_PATH, parse_dates=[config.DATE_COL])
    features = build_feature_matrix(clean)

    config.FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.FEATURES_CSV_PATH, index=False)
    print(f"[features] Saved {len(features):,} rows x {features.shape[1]} cols "
          f"to {config.FEATURES_CSV_PATH}")

    train_df, val_df, test_df = time_based_split(features)
    print(f"[features] Split sizes -> train: {len(train_df)}, val: {len(val_df)}, "
          f"test: {len(test_df)} (per mode; totals across {config.MODES}).")
    print(f"[features] Test window: {test_df[config.DATE_COL].min().date()} "
          f"to {test_df[config.DATE_COL].max().date()}")


if __name__ == "__main__":
    main(sys.argv[1:])
