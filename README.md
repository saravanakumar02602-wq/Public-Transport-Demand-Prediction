# Public Transport Demand Prediction

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

**A time-series forecasting project for daily Chicago Transit Authority (CTA) ridership**, with separate system-wide predictions for bus, rail, and combined boardings. It includes a reproducible data and machine-learning pipeline, model evaluation, and an interactive Streamlit dashboard.

> **Scope:** This project forecasts daily, system-wide ridership by mode. It does not forecast ridership by route or station.

## Project overview

The pipeline downloads and prepares public CTA ridership data, creates time-series and calendar features, trains Random Forest and XGBoost models, evaluates forecasts on a time-ordered holdout period, and serves the trained models in an interactive dashboard.

### Highlights

- Public data from the [Chicago Data Portal](https://data.cityofchicago.org/Transportation/CTA-Ridership-Daily-Boarding-Totals/6iiy-9s97); no API key is required.
- Three daily series: bus, rail, and total boardings.
- Lag, rolling-statistic, calendar, and holiday features.
- Time-based train, validation, and test splits; no random shuffling of time-series observations.
- Random Forest and XGBoost training with time-series cross-validation.
- Interactive historical predictions and short-horizon forecasts in Streamlit.
- Optional R baseline notebooks and an optional LSTM experiment.

## Current results

The table below summarizes the checked-in results from [`reports/ml_metrics.csv`](reports/ml_metrics.csv). The test period is the final 10 weeks in the processed data snapshot, **2026-04-22 to 2026-06-30**. Random Forest had the lowest RMSE of the two machine-learning models tested for each mode.

| Mode | Best of evaluated ML models | MAE (rides) | RMSE (rides) | MAPE |
|---|---|---:|---:|---:|
| Bus | Random Forest | 26,705.9 | 34,375.5 | 4.81% |
| Rail | Random Forest | 20,009.8 | 27,254.8 | 5.18% |
| Total | Random Forest | 46,210.5 | 59,673.3 | 4.87% |

These results cover the two Python ML models in the metrics file; they are **not** a comparison against the R baselines or the optional LSTM. The scores are specific to this data snapshot and test window and should not be interpreted as a guarantee of future performance. The underlying evaluation table is also available in [`reports/model_comparison.csv`](reports/model_comparison.csv).

## Data

The project uses the CTA's **Ridership – Daily Boarding Totals** dataset, available through the Chicago Data Portal's Socrata API. The current processed snapshot covers **2001-01-01 through 2026-06-30**. The source provides system-wide daily totals, not route- or station-level ridership.

Raw downloads and generated processed datasets are excluded from Git by [`.gitignore`](.gitignore). On a fresh checkout, the first pipeline command downloads the source data:

```bash
python -m src.data_prep
```

The downloader uses the public endpoint configured in [`src/config.py`](src/config.py). To refresh an existing raw download, run:

```bash
python -m src.data_prep --force
```

## Getting started

Use Python 3.10 or later and run these commands from the repository root.

### 1. Create an environment and install dependencies

```bash
python -m venv .venv
```

Activate the environment, then install the core requirements:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare data and train the models

```bash
python -m src.data_prep
python -m src.features
python -m src.train_ml
python -m src.evaluate
```

The scripts create their output folders as needed. Running the full ML training step can take several minutes or longer, depending on your machine.

### 3. Launch the dashboard

```bash
streamlit run app/app.py
```

The dashboard needs the processed data, generated features, and trained model files from the previous steps. It serves the Random Forest and XGBoost models; the optional LSTM model is not supported by the dashboard.

## Optional experiments

### R baseline models

The R notebooks provide exploratory analysis and statistical baselines:

- [`notebooks/01_eda.Rmd`](notebooks/01_eda.Rmd)
- [`notebooks/03_baseline_models.Rmd`](notebooks/03_baseline_models.Rmd)

Open and run them in RStudio or another R Markdown-capable environment after preparing the data. The checked-in [`renv.lock`](renv.lock) identifies itself as a **template**, not a machine-generated lockfile; create a real lockfile with `renv::snapshot()` before relying on it for reproducible R environments.

### Deep-learning experiment

The LSTM phase is optional and may require additional setup or hardware. Install its separate dependency and run the training module only if you want to try it:

```bash
pip install -r requirements-dl.txt
python -m src.train_dl
```

## Repository layout

```text
.
├── app/
│   └── app.py                  # Streamlit dashboard
├── data/
│   ├── raw/                    # Downloaded source data (not tracked)
│   └── processed/              # Cleaned data and features (not tracked)
├── figures/                    # Generated plots (not tracked)
├── models/                     # Trained model files (not tracked)
├── notebooks/                  # EDA, feature, and model notebooks
├── reports/
│   ├── final_report.qmd        # Project report
│   ├── ml_metrics.csv          # Checked-in Python ML results
│   └── model_comparison.csv    # Checked-in model comparison
├── src/
│   ├── config.py               # Shared paths, columns, and split settings
│   ├── data_prep.py            # Downloading and data cleaning
│   ├── features.py             # Feature generation and time-based splits
│   ├── train_ml.py             # Random Forest and XGBoost training
│   ├── train_dl.py             # Optional LSTM training
│   └── evaluate.py             # Metrics and model comparison
├── requirements.txt            # Core Python dependencies
└── requirements-dl.txt         # Optional TensorFlow dependency
```

## Methodology

Features include prior-day and prior-week lags, rolling means and standard deviations, calendar encodings, and a holiday indicator. Lag and rolling features use only earlier observations. Data is split chronologically, and the ML models use `TimeSeriesSplit` for hyperparameter search.

The test period is the most recent 10 weeks; a 6-week validation period immediately before it is used in the Python ML workflow. Shared settings are maintained in [`src/config.py`](src/config.py).

## Limitations

- **System-wide grain:** the source does not support route- or station-level predictions. That would require a different dataset and changes to the feature and modeling workflow.
- **External events:** the current feature set does not include weather, service disruptions, special events, or other external drivers.
- **Structural changes:** long-term changes in transit use, including the COVID-19 disruption, can affect model behavior and the relevance of historical patterns.
- **Forecast uncertainty:** the dashboard's widening forecast band is a simple heuristic, not a calibrated prediction interval.
- **R and deep-learning results:** the report documents Python ML results only. Run the R baseline notebook and optional LSTM experiment before making comparisons that include those models.

## Before publishing

- The checked-in results are a snapshot; rerunning the pipeline may update them.
- Raw and processed data plus trained model binaries remain excluded from Git. A fresh checkout must regenerate them by following the setup steps above before launching the dashboard.
- Add a `LICENSE` file if you want to specify how others may use, modify, and distribute this project.
- Initialize Git in this project folder, review the staged files, and confirm that no private or unrelated files are included before pushing to GitHub. In particular, do not force-add the large generated model files.
