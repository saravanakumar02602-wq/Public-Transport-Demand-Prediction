"""
Phase 6 — Deep learning (OPTIONAL).

Trains a small LSTM per mode on a sliding window of the past WINDOW_SIZE
days to predict the next day's ridership, scored on the identical test
window as Phases 4-5. Explicitly prints whether the accuracy gain over
Phase 5's XGBoost justifies the extra complexity — per the brief, "if the
gain is small, say so" is itself a legitimate result, not a failure.

This phase is optional because a laptop-only, no-GPU, solo timeline may not
have room for it. Skip it (don't run this file) and the rest of the
pipeline (Phases 1-5, 7-9) works fine without it — evaluate.py's
consolidate() just omits the DL row if dl_metrics.csv doesn't exist.

Run directly (requires `pip install -r requirements-dl.txt`):
    python -m src.train_dl
or:
    python src/train_dl.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, evaluate

WINDOW_SIZE = 28


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
        return tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow isn't installed. Phase 6 is optional — install it with "
            "`pip install -r requirements-dl.txt` if you want to run it, or "
            "skip this file entirely and move on to Phase 7."
        ) from exc


def build_windows(series_df: pd.DataFrame, window: int = WINDOW_SIZE):
    """series_df: columns [DATE_COL, TARGET_COL] for ONE mode, sorted ascending."""
    values = series_df[config.TARGET_COL].to_numpy(dtype="float32")
    dates = series_df[config.DATE_COL].to_numpy()

    X, y, target_dates = [], [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
        target_dates.append(dates[i])

    X = np.asarray(X)[..., np.newaxis]  # (n_samples, window, 1) for LSTM input
    y = np.asarray(y)
    target_dates = pd.to_datetime(pd.Series(target_dates))
    return X, y, target_dates


def split_by_target_date(X, y, target_dates: pd.Series,
                          test_weeks=config.TEST_WEEKS, val_weeks=config.VAL_WEEKS):
    max_date = target_dates.max()
    test_start = max_date - pd.Timedelta(weeks=test_weeks) + pd.Timedelta(days=1)
    val_start = test_start - pd.Timedelta(weeks=val_weeks)

    train_mask = (target_dates < val_start).to_numpy()
    val_mask = ((target_dates >= val_start) & (target_dates < test_start)).to_numpy()
    test_mask = (target_dates >= test_start).to_numpy()

    return (X[train_mask], y[train_mask]), (X[val_mask], y[val_mask]), (X[test_mask], y[test_mask])


def run_for_mode(clean_df: pd.DataFrame, mode: str) -> None:
    tf = _require_tensorflow()
    from tensorflow import keras

    print(f"\n[train_dl] === mode: {mode} ===")
    series_df = (
        clean_df[clean_df[config.MODE_COL] == mode]
        .sort_values(config.DATE_COL)[[config.DATE_COL, config.TARGET_COL]]
        .dropna()
        .reset_index(drop=True)
    )

    X, y, target_dates = build_windows(series_df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_target_date(X, y, target_dates)

    # Standardize using TRAIN stats only (no leakage from val/test into scaling).
    mean, std = y_train.mean(), y_train.std()
    X_train_s, X_val_s, X_test_s = (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std
    y_train_s, y_val_s = (y_train - mean) / std, (y_val - mean) / std

    keras.utils.set_random_seed(config.RANDOM_SEED)
    model = keras.Sequential([
        keras.layers.Input(shape=(WINDOW_SIZE, 1)),
        keras.layers.LSTM(64),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    t0 = time.perf_counter()
    model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=100, batch_size=32, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )
    training_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_s = model.predict(X_test_s, verbose=0).flatten()
    inference_time = time.perf_counter() - t0
    y_pred = y_pred_s * std + mean  # back to raw ridership units for fair comparison

    metrics = evaluate.score_all(y_test, y_pred)
    row = {
        "model": "lstm", "mode": mode, **metrics,
        "training_time_s": round(training_time, 3),
        "inference_time_s": round(inference_time, 4),
        "interpretability": "low (opaque weights; use permutation importance if needed)",
    }
    evaluate.append_metrics_row(config.DL_METRICS_PATH, row)
    print(f"[train_dl] lstm/{mode} -> MAE={metrics['MAE']:.1f} RMSE={metrics['RMSE']:.1f} "
          f"MAPE={metrics['MAPE']:.2f}%")

    model_path = config.MODELS_DIR / f"lstm_{mode}.keras"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"[train_dl] Saved model -> {model_path}")

    _print_verdict_vs_xgboost(mode, metrics["RMSE"])


def _print_verdict_vs_xgboost(mode: str, lstm_rmse: float) -> None:
    if not config.ML_METRICS_PATH.exists():
        print("[train_dl] (Run train_ml.py first to get a Phase 5 comparison verdict.)")
        return
    ml = pd.read_csv(config.ML_METRICS_PATH)
    xgb_row = ml[(ml["model"] == "xgboost") & (ml["mode"] == mode)]
    if xgb_row.empty:
        return
    xgb_rmse = float(xgb_row["RMSE"].iloc[0])
    improvement_pct = (xgb_rmse - lstm_rmse) / xgb_rmse * 100

    print(f"[train_dl] Verdict ({mode}): XGBoost RMSE={xgb_rmse:.1f}, LSTM RMSE={lstm_rmse:.1f} "
          f"({improvement_pct:+.1f}% change).")
    if improvement_pct < 3:
        print("[train_dl]   -> Gain is small (<3%) relative to a much heavier, less "
              "interpretable model. Per the brief's Phase 6 guidance, this is a "
              "legitimate reason to SKIP the LSTM in the final product and ship "
              "XGBoost instead — write this trade-off up explicitly in "
              "reports/final_report.qmd rather than defaulting to 'the fancier "
              "model wins'.")
    else:
        print("[train_dl]   -> LSTM meaningfully beats XGBoost here. Worth keeping, "
              "but still note the added training time / serving complexity cost "
              "in the final report's justification.")


def main(argv=None) -> None:
    clean_df = pd.read_csv(config.CLEAN_CSV_PATH, parse_dates=[config.DATE_COL])
    for mode in config.MODES:
        run_for_mode(clean_df, mode)
    print("\n[train_dl] Done. Metrics in", config.DL_METRICS_PATH)


if __name__ == "__main__":
    main(sys.argv[1:])
