"""
Shared scoring functions (used by train_ml.py, train_dl.py, and referenced
by notebooks/03_baseline_models.Rmd for parity) plus the Phase 7
consolidation step that merges the R baseline metrics with the Python ML/DL
metrics into one comparison table.

Run directly (after Phases 4-6 have each written their own metrics CSV):
    python -m src.evaluate
or:
    python src/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE — use this instead of MAPE if any y_true could be 0."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def score_all(y_true, y_pred) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def append_metrics_row(path, row: dict) -> None:
    """Append one model's metrics to a phase-level CSV (creates it if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row_df], ignore_index=True)
        # keep the latest row per (model, mode) pair if a script is re-run
        combined = combined.drop_duplicates(subset=["model", "mode"], keep="last")
    else:
        combined = row_df
    combined.to_csv(path, index=False)


def consolidate(out_path=config.MODEL_COMPARISON_PATH) -> pd.DataFrame:
    """
    Phase 7: merge baseline (R/fable), ML (Python), and optional DL metrics
    into one table: model | mode | MAE | RMSE | MAPE | training_time_s |
    inference_time_s | interpretability.
    """
    frames = []
    for path, family in [
        (config.BASELINE_METRICS_PATH, "baseline"),
        (config.ML_METRICS_PATH, "ml"),
        (config.DL_METRICS_PATH, "dl"),
    ]:
        if path.exists():
            f = pd.read_csv(path)
            f["family"] = family
            frames.append(f)
        else:
            reason = (
                "the optional DL phase was not run"
                if family == "dl"
                else f"the {family} phase has not been run"
            )
            print(f"[evaluate] {path} not found — skipping ({reason}).")

    if not frames:
        raise FileNotFoundError(
            "No metrics files found. Run Phase 4 (R baselines) and Phase 5 "
            "(train_ml.py) at minimum before consolidating."
        )

    comparison = pd.concat(frames, ignore_index=True, sort=False)
    comparison = comparison.sort_values(["mode", "RMSE"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_path, index=False)
    print(f"[evaluate] Wrote consolidated comparison table to {out_path}")
    print(comparison.to_string(index=False))
    return comparison


def main(argv=None) -> None:
    consolidate()


if __name__ == "__main__":
    main(sys.argv[1:])
