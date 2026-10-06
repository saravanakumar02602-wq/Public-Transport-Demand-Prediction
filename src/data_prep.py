"""
Phase 1 — Data acquisition & cleaning.

Downloads the CTA "Ridership - Daily Boarding Totals" dataset (system-wide
bus + rail boardings, daily, 2001-present) from Chicago's open data portal,
cleans it, and reshapes it into a tidy long-format table:

    service_date | mode (bus/rail/total) | rides | day_type

Run directly:
    python -m src.data_prep
or:
    python src/data_prep.py

Why this dataset: the project brief left [CITY / AGENCY NAME] and
[DATASET NAME + SOURCE URL] as placeholders. This dataset was chosen because
it is free, requires no API key, is small enough for a laptop-only / no-GPU
timeline, and is a real, actively-maintained open dataset. It is *system-wide*
rather than per-route/per-station — the per-route CTA dataset is ~6M+ rows,
which is a much bigger scope than a solo 3-week project supports. See the
README "Scope decisions" section and reports/final_report.qmd "Limitations"
for the honest trade-off this implies (Phase 9 asks you to state it there).

Outlier-handling rule (documented per Phase 1 spec, not silently dropped):
we FLAG statistically extreme days (robust z-score on a rolling median/MAD,
see `flag_outliers`) but we do NOT remove them. Ridership has real,
event-driven shocks (COVID-19 onset, blizzards, holidays, service changes)
that are exactly the kind of structural break Phase 2's EDA is supposed to
surface — deleting them would hide the thing the assignment asks you to find.
The `is_outlier` flag is carried through to the cleaned file so later phases
can choose to use it (e.g. as a feature, or to exclude only from training
windows where it would leak an unrepeatable shock into "normal" patterns).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Makes `python src/data_prep.py` work the same as `python -m src.data_prep`
# by putting the project root on sys.path before the package-style import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import requests

from src import config


def download_raw(out_path=config.RAW_CSV_PATH, limit: int = 20000, force: bool = False) -> pd.DataFrame:
    """Pull the full dataset from the Socrata API and save it untouched."""
    if out_path.exists() and not force:
        print(f"[data_prep] {out_path} already exists, skipping download "
              f"(pass --force to re-download).")
        return pd.read_csv(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = {"$limit": limit, "$order": config.DATE_COL}
    print(f"[data_prep] GET {config.SOCRATA_BASE_URL} (limit={limit}) ...")
    resp = requests.get(config.SOCRATA_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    records = resp.json()
    if not records:
        raise RuntimeError(
            "No records returned from the Socrata API. Check "
            "config.SOCRATA_DATASET_ID and your network connection."
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)
    print(f"[data_prep] Saved {len(df):,} raw rows to {out_path}")
    return df


def load_raw(path=config.RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {config.DATE_COL, config.DAY_TYPE_COL, "bus", "rail_boardings", "total_rides"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")
    return df


def flag_outliers(series: pd.Series, window: int = 12, z_thresh: float = 4.0) -> pd.Series:
    """
    Robust rolling z-score outlier flag (median + MAD instead of mean/std,
    since mean/std are themselves distorted by the same outliers we're
    trying to find).

    IMPORTANT: call this on a series that's already restricted to one
    `day_type` (W/A/U) within one `mode`, not the raw daily series. Weekday
    vs. Saturday vs. Sunday/holiday ridership differs by ~40% as a matter of
    routine, not anomaly — a rolling window that mixes day types will flag
    most weekends as "outliers" simply because they sit next to weekdays in
    calendar order. `clean_and_reshape` below groups by (mode, day_type)
    before calling this, which is what keeps the flag meaningful.
    """
    rolling_median = series.rolling(window, center=True, min_periods=window // 2).median()
    abs_dev = (series - rolling_median).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=window // 2).median()
    # 1.4826 scales MAD to be comparable to a standard deviation for normal data
    robust_z = 0.6745 * (series - rolling_median) / mad.replace(0, np.nan)
    return robust_z.abs() > z_thresh


def clean_and_reshape(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # --- parse & de-duplicate -------------------------------------------------
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL]).dt.normalize()
    before = len(df)
    df = df.drop_duplicates(subset=[config.DATE_COL], keep="first")
    dupes_dropped = before - len(df)
    if dupes_dropped:
        print(f"[data_prep] Dropped {dupes_dropped} duplicate service_date rows.")

    df = df.sort_values(config.DATE_COL)

    # --- missing dates: reindex to a full daily calendar ----------------------
    full_range = pd.date_range(df[config.DATE_COL].min(), df[config.DATE_COL].max(), freq="D")
    missing_dates = full_range.difference(df[config.DATE_COL])
    if len(missing_dates):
        print(f"[data_prep] {len(missing_dates)} calendar day(s) missing from the "
              f"source; these will appear as NaN rides after reindexing rather "
              f"than being silently skipped.")
    df = df.set_index(config.DATE_COL).reindex(full_range)
    df.index.name = config.DATE_COL
    df = df.reset_index()

    # day_type wasn't in the source for reindexed rows -> forward-fill is wrong
    # for a day-of-week attribute, so we don't touch it here; features.py
    # recomputes day-of-week / holiday flags from the date itself instead of
    # trusting this column for anything the code can derive deterministically.

    # --- reshape wide (bus, rail_boardings, total_rides) -> long -------------
    long_df = df.melt(
        id_vars=[config.DATE_COL, config.DAY_TYPE_COL],
        value_vars=["bus", "rail_boardings", "total_rides"],
        var_name=config.MODE_COL,
        value_name=config.TARGET_COL,
    )
    long_df[config.MODE_COL] = long_df[config.MODE_COL].replace(
        {"rail_boardings": "rail", "total_rides": "total"}
    )

    # --- outlier flag, per (mode, day_type) -----------------------------------
    # Grouping by day_type too (not just mode) is what keeps this from
    # flagging routine weekday/weekend seasonality as "extreme" — see the
    # warning in flag_outliers()'s docstring.
    long_df["is_outlier"] = (
        long_df.sort_values(config.DATE_COL)
        .groupby([config.MODE_COL, config.DAY_TYPE_COL])[config.TARGET_COL]
        .transform(lambda s: flag_outliers(s))
    )

    long_df = long_df.sort_values([config.MODE_COL, config.DATE_COL]).reset_index(drop=True)
    return long_df


def save_processed(df: pd.DataFrame, out_path=config.CLEAN_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[data_prep] Saved {len(df):,} tidy rows to {out_path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20000,
                         help="Row cap for the Socrata API pull (dataset has ~9.3k rows).")
    parser.add_argument("--force", action="store_true",
                         help="Re-download even if the raw file already exists.")
    args = parser.parse_args(argv)

    raw = download_raw(limit=args.limit, force=args.force)
    raw = load_raw(config.RAW_CSV_PATH)  # re-read from disk so this step is reproducible standalone
    clean = clean_and_reshape(raw)
    save_processed(clean)

    n_outliers = int(clean["is_outlier"].sum())
    print(f"[data_prep] Done. {n_outliers} rows flagged as outliers "
          f"({n_outliers / len(clean):.2%} of rows) — inspect these in "
          f"notebooks/01_eda.Rmd rather than assuming they're errors.")


if __name__ == "__main__":
    main(sys.argv[1:])
