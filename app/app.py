"""
Phase 8 — Product / deployment.

A Streamlit dashboard over the trained models:
  - pick a mode (bus / rail / total — this project's system-wide stand-in
    for "route/station", see README "Scope decisions") and a date range
  - historical actual vs. predicted ridership over that range
  - a short forecast beyond the last known date, with a simple
    (heuristic, not a real prediction interval) confidence band
  - the Phase 7 model comparison table, for transparency

Run:
    streamlit run app/app.py

Requires Phases 1, 3, 5 (and optionally 4, 6) to have already been run —
this app only reads files they produce, it doesn't compute anything from
scratch.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.features import add_calendar_features, add_cyclical_features  # noqa: E402

st.set_page_config(page_title="CTA Ridership Demand", page_icon="🚌", layout="wide")


@st.cache_data
def load_clean():
    return pd.read_csv(config.CLEAN_CSV_PATH, parse_dates=[config.DATE_COL])


@st.cache_data
def load_features():
    return pd.read_csv(config.FEATURES_CSV_PATH, parse_dates=[config.DATE_COL])


@st.cache_data
def load_comparison():
    if config.MODEL_COMPARISON_PATH.exists():
        return pd.read_csv(config.MODEL_COMPARISON_PATH)
    return None


def available_models_for_mode(mode: str) -> list[str]:
    """
    Only joblib-serialized models (Random Forest / XGBoost from train_ml.py)
    are listed here. An LSTM from train_dl.py is saved as a *.keras file
    with a completely different input shape (a raw sliding window of past
    values, not this app's tabular lag/rolling feature set) and would need
    its own prediction + iterative-forecast path to serve safely. If Phase
    7 picks the LSTM as your production model, that's a real (documented)
    follow-up for app.py, not something this dashboard silently pretends to
    support.
    """
    if not config.MODELS_DIR.exists():
        return []
    return sorted(
        p.stem.rsplit(f"_{mode}", 1)[0]
        for p in config.MODELS_DIR.glob(f"*_{mode}.joblib")
    )


def load_model_bundle(model_name: str, mode: str):
    path = config.MODELS_DIR / f"{model_name}_{mode}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)  # {"model": ..., "feature_cols": [...]}


def make_feature_row(history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build one row of features for `target_date` from a history buffer of
    (date, rides) for a single mode, matching train-time feature logic:
    lags / rolling stats only ever look at days strictly before target_date.
    """
    row = pd.DataFrame({config.DATE_COL: [target_date], config.MODE_COL: ["_tmp"]})
    row = add_calendar_features(row)
    row = add_cyclical_features(row)

    hist_sorted = history.sort_values(config.DATE_COL)
    values = hist_sorted[config.TARGET_COL].to_numpy()

    for lag in config.LAGS:
        row[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
    for w in config.ROLLING_WINDOWS:
        window_vals = values[-w:] if len(values) >= 2 else values
        row[f"roll_mean_{w}"] = np.mean(window_vals)
        row[f"roll_std_{w}"] = np.std(window_vals, ddof=1) if len(window_vals) > 1 else 0.0
    return row


def iterative_forecast(model_bundle, history: pd.DataFrame, steps: int, band_rmse: float):
    """Recursive one-step-ahead forecast: each prediction feeds the next step's lags."""
    model, feature_cols = model_bundle["model"], model_bundle["feature_cols"]
    hist = history[[config.DATE_COL, config.TARGET_COL]].copy()
    last_date = hist[config.DATE_COL].max()

    forecasts = []
    for step in range(1, steps + 1):
        target_date = last_date + pd.Timedelta(days=step)
        row = make_feature_row(hist, target_date)
        X = row[feature_cols]
        pred = float(model.predict(X)[0])

        # Growing heuristic band: random-walk-style sqrt(horizon) widening,
        # anchored to the model's own held-out RMSE. This is intentionally
        # simple (per the brief) — not a calibrated prediction interval.
        band = band_rmse * np.sqrt(step)
        forecasts.append({
            config.DATE_COL: target_date,
            "forecast": pred,
            "lower": pred - band,
            "upper": pred + band,
        })
        hist = pd.concat([hist, pd.DataFrame({config.DATE_COL: [target_date],
                                               config.TARGET_COL: [pred]})], ignore_index=True)
    return pd.DataFrame(forecasts)


# --- page styling -------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; }
    section[data-testid="stMain"] { color: #25364d; }
    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3 { color: #14233a !important; }
    section[data-testid="stMain"] p { color: #53657a; }
    [data-testid="stSidebar"] { background: #101b32; }
    [data-testid="stSidebar"] * { color: #eef4ff; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #17243b; }
    .hero {
        padding: 2rem 2.2rem; border-radius: 22px; margin: .3rem 0 1.2rem;
        color: white; background: linear-gradient(120deg, #101b32, #193d63 68%, #126c71);
        box-shadow: 0 12px 32px rgba(16, 35, 60, .15);
    }
    .hero-kicker {
        color: #76e2d0; text-transform: uppercase; letter-spacing: .15em;
        font-size: .76rem; font-weight: 750;
    }
    section[data-testid="stMain"] .hero h1 {
        color: white !important; font-size: clamp(2rem, 4vw, 3.15rem); margin: .4rem 0;
    }
    .hero p { color: #d6e3f3 !important; font-size: 1.04rem; margin: 0; }
    div[data-testid="stMetric"] {
        background: white; padding: 1rem 1.1rem; border-radius: 16px;
        border: 1px solid #e4ebf3; box-shadow: 0 4px 16px rgba(27, 49, 78, .05);
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {
        color: #52647b !important; opacity: 1 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #14233a !important; font-size: clamp(1.25rem, 2vw, 1.9rem) !important;
    }
    .metric-grid {
        display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: .8rem; margin: .4rem 0 .65rem;
    }
    .metric-card {
        background: white; padding: 1rem 1.1rem; border-radius: 16px;
        border: 1px solid #e4ebf3; box-shadow: 0 4px 16px rgba(27, 49, 78, .05);
        min-width: 0;
    }
    .metric-label {
        color: #52647b; text-transform: uppercase; letter-spacing: .08em;
        font-size: .69rem; font-weight: 750;
    }
    .metric-value {
        color: #14233a; font-size: clamp(1.25rem, 2vw, 1.9rem);
        font-weight: 700; line-height: 1.25; margin: .45rem 0 .2rem;
        white-space: nowrap;
    }
    .metric-help { color: #728198; font-size: .78rem; }
    @media (max-width: 850px) {
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .section-note { color: #62738a; margin-top: -.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- load available output files ----------------------------------------------
clean_df = load_clean() if config.CLEAN_CSV_PATH.exists() else None
features_df = load_features() if config.FEATURES_CSV_PATH.exists() else None
comparison_df = load_comparison()

# --- sidebar -----------------------------------------------------------------
st.sidebar.markdown("## CTA RIDERSHIP")
st.sidebar.caption("Demand forecasting · Chicago")
mode = st.sidebar.selectbox(
    "Transit mode",
    config.MODES,
    index=config.MODES.index("total"),
    format_func=str.title,
)

available_models = available_models_for_mode(mode)
comparison_for_mode = (
    comparison_df[comparison_df[config.MODE_COL] == mode]
    if comparison_df is not None and config.MODE_COL in comparison_df
    else pd.DataFrame()
)
metric_models = comparison_for_mode["model"].dropna().astype(str).tolist() if not comparison_for_mode.empty else []
model_options = available_models or metric_models
model_name = (
    st.sidebar.selectbox(
        "Model",
        model_options,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    if model_options
    else "random_forest"
)
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=30, value=14)

mode_history = (
    clean_df[clean_df[config.MODE_COL] == mode].sort_values(config.DATE_COL)
    if clean_df is not None
    else pd.DataFrame()
)
start, end = None, None
if not mode_history.empty:
    min_date, max_date = mode_history[config.DATE_COL].min(), mode_history[config.DATE_COL].max()
    date_range = st.sidebar.date_input(
        "Historical range",
        value=(max(min_date, max_date - pd.Timedelta(days=180)), max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (tuple, list, np.ndarray)):
        if len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        elif len(date_range) == 1:
            start = end = pd.Timestamp(date_range[0])
    elif date_range is not None:
        start = end = pd.Timestamp(date_range)

st.sidebar.divider()
st.sidebar.caption("Source · Chicago Data Portal")
st.sidebar.caption("Forecasts are system-wide, not route-level.")

# --- hero ---------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">Transit intelligence · Chicago</div>
      <h1>Ridership, in focus.</h1>
      <p>Explore {mode} demand, compare model accuracy, and review the latest
         available performance snapshot.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- headline results ----------------------------------------------------------
selected_result = comparison_for_mode[
    comparison_for_mode["model"].astype(str) == model_name
] if not comparison_for_mode.empty else pd.DataFrame()
best_result = (
    comparison_for_mode.sort_values("RMSE").iloc[0]
    if not comparison_for_mode.empty and "RMSE" in comparison_for_mode
    else None
)

if not selected_result.empty:
    result = selected_result.iloc[0]
    headline_metrics = [
        ("Test RMSE", f"{result['RMSE']:,.0f}", "rides · lower is better"),
        ("Test MAE", f"{result['MAE']:,.0f}", "rides · lower is better"),
        ("MAPE", f"{result['MAPE']:.2f}%", "percentage error"),
        ("Holdout window", "10 weeks", "chronological test period"),
    ]
elif best_result is not None:
    headline_metrics = [
        ("Best test RMSE", f"{best_result['RMSE']:,.0f}", "rides · lower is better"),
        ("Test MAE", f"{best_result['MAE']:,.0f}", "rides · lower is better"),
        ("MAPE", f"{best_result['MAPE']:.2f}%", "percentage error"),
        ("Holdout window", "10 weeks", "chronological test period"),
    ]
else:
    headline_metrics = []
    st.info("No saved model metrics yet. Run `python -m src.train_ml` to create the first evaluation.")

if headline_metrics:
    cards = "".join(
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-help">{escape(help_text)}</div>'
        '</div>'
        for label, value, help_text in headline_metrics
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)

st.caption(
    f"Selected model: **{model_name.replace('_', ' ').title()}** · "
    "Metrics are from the saved, chronological 10-week test window; lower error is better."
)

# --- historical demand ---------------------------------------------------------
left, right = st.columns([1.55, 1])
with left:
    st.subheader("Ridership trend")
    if not mode_history.empty:
        chart_history = mode_history
        if start is not None and end is not None:
            chart_history = chart_history[
                (chart_history[config.DATE_COL] >= start)
                & (chart_history[config.DATE_COL] <= end)
            ]
        trend = go.Figure()
        trend.add_trace(go.Scatter(
            x=chart_history[config.DATE_COL],
            y=chart_history[config.TARGET_COL],
            name="Daily rides",
            mode="lines",
            line=dict(color="#168c91", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(22, 140, 145, .09)",
            hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f} rides<extra></extra>",
        ))
        trend.update_layout(
            template="plotly_white",
            height=370,
            margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font=dict(color="#26364b"),
            xaxis_title=None,
            yaxis_title="Daily boardings",
            hovermode="x unified",
            showlegend=False,
        )
        trend.update_xaxes(showgrid=False)
        trend.update_yaxes(gridcolor="#edf1f6", separatethousands=True)
        st.plotly_chart(trend, width="stretch", theme=None)
    else:
        st.info("Historical data is not generated yet. Run the data preparation steps in the README.")

with right:
    st.subheader("Model scorecard")
    if not comparison_for_mode.empty:
        scorecard = comparison_for_mode.sort_values("RMSE").copy()
        scorecard["Model"] = scorecard["model"].str.replace("_", " ").str.title()
        score = go.Figure(go.Bar(
            x=scorecard["Model"],
            y=scorecard["RMSE"],
            marker=dict(
                color=["#168c91" if name == model_name else "#b9c8d9" for name in scorecard["model"]],
                line=dict(width=0),
            ),
            text=scorecard["RMSE"].map(lambda value: f"{value:,.0f}"),
            textposition="outside",
            hovertemplate="%{x}<br>RMSE %{y:,.0f} rides<extra></extra>",
        ))
        score.update_layout(
            template="plotly_white",
            height=370,
            margin=dict(l=8, r=8, t=28, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font=dict(color="#26364b"),
            xaxis_title=None,
            yaxis_title="RMSE · rides (lower is better)",
            showlegend=False,
        )
        score.update_xaxes(showgrid=False)
        score.update_yaxes(gridcolor="#edf1f6", rangemode="tozero")
        st.plotly_chart(score, width="stretch", theme=None)
    else:
        st.info("Run model evaluation to populate this comparison.")

# --- detailed results -----------------------------------------------------------
st.subheader(f"{mode.capitalize()} evaluation results")
if not comparison_for_mode.empty:
    display_columns = [
        column for column in
        ["model", "family", "MAE", "RMSE", "MAPE", "training_time_s", "inference_time_s"]
        if column in comparison_for_mode.columns
    ]
    st.dataframe(
        comparison_for_mode.sort_values("RMSE")[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "family": st.column_config.TextColumn("Family"),
            "MAE": st.column_config.NumberColumn("MAE · rides", format="%,.0f"),
            "RMSE": st.column_config.NumberColumn("RMSE · rides", format="%,.0f"),
            "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
            "training_time_s": st.column_config.NumberColumn("Training · sec", format="%.1f"),
            "inference_time_s": st.column_config.NumberColumn("Inference · sec", format="%.4f"),
        },
    )
else:
    st.info("No evaluation results are available yet.")

# --- optional live predictions --------------------------------------------------
if clean_df is None or features_df is None:
    st.warning(
        "Live predictions are not ready because the generated data or feature file is missing. "
        "Run `python -m src.data_prep`, `python -m src.features`, and `python -m src.train_ml`."
    )
elif not available_models:
    st.warning(
        "Metrics and historical demand are shown above. A trained model file is not available "
        "for live predictions; run `python -m src.train_ml`."
    )
else:
    try:
        bundle = load_model_bundle(model_name, mode)
        mode_features = features_df[
            features_df[config.MODE_COL] == mode
        ].sort_values(config.DATE_COL)
        X_all = mode_features[bundle["feature_cols"]]
        mode_features = mode_features.assign(predicted=bundle["model"].predict(X_all))

        if start is not None and end is not None:
            prediction_view = mode_features[
                (mode_features[config.DATE_COL] >= start)
                & (mode_features[config.DATE_COL] <= end)
            ]
        else:
            prediction_view = mode_features

        st.subheader("Actual vs. predicted")
        prediction_chart = go.Figure()
        prediction_chart.add_trace(go.Scatter(
            x=prediction_view[config.DATE_COL],
            y=prediction_view[config.TARGET_COL],
            name="Actual",
            mode="lines",
            line=dict(color="#14233a", width=2),
        ))
        prediction_chart.add_trace(go.Scatter(
            x=prediction_view[config.DATE_COL],
            y=prediction_view["predicted"],
            name=f"Predicted · {model_name.replace('_', ' ').title()}",
            mode="lines",
            line=dict(color="#168c91", width=2, dash="dash"),
        ))
        prediction_chart.update_layout(
            template="plotly_white",
            height=370,
            margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font=dict(color="#26364b"),
            xaxis_title=None,
            yaxis_title="Daily boardings",
            hovermode="x unified",
        )
        prediction_chart.update_xaxes(showgrid=False)
        prediction_chart.update_yaxes(gridcolor="#edf1f6", separatethousands=True)
        st.plotly_chart(prediction_chart, width="stretch", theme=None)

        band_rmse = float(result["RMSE"]) if not selected_result.empty else 0.0
        forecast_df = iterative_forecast(bundle, mode_history, horizon, band_rmse)
        st.subheader(f"Next {horizon} days · experimental forecast")
        forecast_chart = go.Figure()
        forecast_chart.add_trace(go.Scatter(
            x=forecast_df[config.DATE_COL],
            y=forecast_df["forecast"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#168c91", width=2.5),
        ))
        forecast_chart.add_trace(go.Scatter(
            x=pd.concat([forecast_df[config.DATE_COL], forecast_df[config.DATE_COL][::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(22, 140, 145, .12)",
            line=dict(width=0),
            name="RMSE-based range",
            hoverinfo="skip",
        ))
        forecast_chart.update_layout(
            template="plotly_white",
            height=330,
            margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font=dict(color="#26364b"),
            xaxis_title=None,
            yaxis_title="Daily boardings",
        )
        forecast_chart.update_xaxes(showgrid=False)
        forecast_chart.update_yaxes(gridcolor="#edf1f6", separatethousands=True)
        st.plotly_chart(forecast_chart, width="stretch", theme=None)
        st.caption(
            "The shaded RMSE × √horizon band is a simple visual guide, not a calibrated "
            "statistical prediction interval."
        )
    except (ImportError, OSError, ValueError) as exc:
        st.warning(
            "Live model predictions are unavailable in this Python environment. "
            "The saved scorecard and historical ridership above are still available. "
            "Reinstall the project requirements in an environment permitted to load "
            f"scikit-learn's native libraries. Details: {exc}"
        )

st.divider()
st.caption(
    "CTA public ridership data · Daily system-wide totals · "
    "Forecast evaluation is historical and does not guarantee future performance."
)
