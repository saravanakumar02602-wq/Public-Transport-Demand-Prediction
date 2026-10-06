"""
Phase 5 — Machine learning models.

Trains a Random Forest and an XGBoost regressor per mode (bus / rail /
total), tunes hyperparameters with rolling-origin (TimeSeriesSplit) CV —
never shuffled k-fold — scores on the identical held-out test window used
by the Phase 4 baselines, saves a SHAP feature-importance plot per mode,
and serializes the fitted models.

Run directly:
    python -m src.train_ml
or:
    python src/train_ml.py

Note on `is_outlier`: this column is deliberately EXCLUDED from the feature
set. It was computed in data_prep.py with a *centered* rolling window
(uses +/- `window/2` days around each point), which is fine for flagging
days worth inspecting in EDA but would leak future information into a
forecasting feature — a day can only be flagged "extreme" once you've also
seen what came after it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from src import config, evaluate
from src.features import time_based_split

NON_FEATURE_COLS = {config.DATE_COL, config.MODE_COL, config.TARGET_COL,
                     config.DAY_TYPE_COL, "is_outlier"}

RF_PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 8, 16, 24],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.5, 1.0],
}

XGB_PARAM_GRID = {
    "n_estimators": [200, 400, 800],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def tune_and_fit(estimator, param_grid, X_trainval, y_trainval, n_splits: int = 5, n_iter: int = 15):
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_trainval, y_trainval)
    return search.best_estimator_, search.best_params_


def save_shap_plot(model, X_sample: pd.DataFrame, mode: str, model_name: str) -> None:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        out_path = config.FIGURES_DIR / f"shap_{model_name}_{mode}.png"
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[train_ml] Saved SHAP plot -> {out_path}")
    except Exception as exc:  # SHAP/plotting is diagnostic, not critical-path
        print(f"[train_ml] Skipped SHAP plot for {model_name}/{mode}: {exc}")


def run_for_mode(features_df: pd.DataFrame, mode: str) -> None:
    print(f"\n[train_ml] === mode: {mode} ===")
    mode_df = features_df[features_df[config.MODE_COL] == mode].sort_values(config.DATE_COL)
    train_df, val_df, test_df = time_based_split(mode_df)
    trainval_df = pd.concat([train_df, val_df]).sort_values(config.DATE_COL)

    feature_cols = get_feature_columns(mode_df)
    X_trainval, y_trainval = trainval_df[feature_cols], trainval_df[config.TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[config.TARGET_COL]

    models = {
        "random_forest": (RandomForestRegressor(random_state=config.RANDOM_SEED), RF_PARAM_GRID,
                           "medium (impurity-based + SHAP feature importance)"),
        "xgboost": (XGBRegressor(random_state=config.RANDOM_SEED, objective="reg:squarederror",
                                  tree_method="hist"), XGB_PARAM_GRID,
                    "medium (gain-based + SHAP feature importance)"),
    }

    for model_name, (estimator, grid, interpretability) in models.items():
        t0 = time.perf_counter()
        best_model, best_params = tune_and_fit(estimator, grid, X_trainval, y_trainval)
        training_time = time.perf_counter() - t0
        print(f"[train_ml] {model_name}/{mode} best params: {best_params}")

        t0 = time.perf_counter()
        y_pred = best_model.predict(X_test)
        inference_time = time.perf_counter() - t0

        metrics = evaluate.score_all(y_test, y_pred)
        row = {
            "model": model_name,
            "mode": mode,
            **metrics,
            "training_time_s": round(training_time, 3),
            "inference_time_s": round(inference_time, 4),
            "interpretability": interpretability,
        }
        evaluate.append_metrics_row(config.ML_METRICS_PATH, row)
        print(f"[train_ml] {model_name}/{mode} -> "
              f"MAE={metrics['MAE']:.1f} RMSE={metrics['RMSE']:.1f} MAPE={metrics['MAPE']:.2f}%")

        model_path = config.MODELS_DIR / f"{model_name}_{mode}.joblib"
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": best_model, "feature_cols": feature_cols}, model_path)
        print(f"[train_ml] Saved model -> {model_path}")

        # SHAP is most meaningful (and fast) on the tree models; sample the
        # test set if it's large so the plot doesn't take forever.
        sample = X_test.sample(min(len(X_test), 300), random_state=config.RANDOM_SEED)
        save_shap_plot(best_model, sample, mode, model_name)


def main(argv=None) -> None:
    features_df = pd.read_csv(config.FEATURES_CSV_PATH, parse_dates=[config.DATE_COL])
    for mode in config.MODES:
        run_for_mode(features_df, mode)
    print("\n[train_ml] Done. Metrics in", config.ML_METRICS_PATH)


if __name__ == "__main__":
    main(sys.argv[1:])
