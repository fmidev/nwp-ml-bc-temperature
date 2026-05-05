# nwp-ml-bc-temperature

Machine learning workflows for bias-correcting near-surface temperature forecasts from numerical weather prediction (NWP) data.

This repository contains research scripts for training, running inference, evaluating, and visualizing machine-learning-based temperature bias correction models. The data used by the project is not public, so this README focuses on the model workflow, expected prepared inputs, script roles, and generated outputs rather than documenting raw data access.

## Overview

Numerical weather prediction models can contain systematic near-surface temperature biases. This project uses machine learning to estimate the forecast error and apply that estimate as a correction to the raw forecast.

The core target is the forecast temperature bias:

```text
bias = observed_temperature - forecast_temperature
```

The corrected forecast is then computed as:

```text
corrected_temperature = raw_forecast_temperature + predicted_bias
```

The repository is organized as a collection of standalone experiment and analysis scripts rather than an installable Python package.

## Current repository structure

```text
.
├── scripts/
│   ├── GNN_train_tuning.py
│   ├── LSTM_train.py
│   ├── LSTM_inference.py
│   ├── xgboost_train_tuning.py
│   ├── xgboost_inference.py
│   ├── gnn_inference.py
│   ├── shap_analysis.py
│   ├── QQ-plot.py
│   ├── correlation_plot.py
│   ├── hit_rate.py
│   ├── hit_rate_plot.py
│   ├── metrics_station_plot.py
│   ├── mos_plot_multi.py
│   ├── observation_checks.py
│   ├── observation_data.py
│   ├── scatterplot_multi.py
│   ├── single_analysistime_plot.py
│   ├── station_plots.py
│   └── gnn_station_plot.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Data

The scripts expect prepared meteorological forecast, observation, and station datasets to already be available in the local execution environment. Some scripts may still contain local path assumptions from the research environment and should be reviewed before running elsewhere.

This README intentionally avoids documenting private source-data locations or raw data acquisition.

## Expected input data

The model scripts expect prepared tabular forecast-observation data, typically in Parquet or CSV format depending on the script.

Common data concepts used across the workflow include:

- station identifier
- forecast analysis time
- forecast valid time
- forecast lead time
- observed near-surface air temperature
- raw forecast near-surface temperature
- additional NWP predictor variables
- station metadata such as longitude, latitude, and elevation
- derived time features such as hour-of-day and day-of-year encodings

Common column names referenced by the modeling scripts include:

```text
SID
analysistime
validtime
leadtime
obs_TA
T2
D2
U10
V10
MSL
SKT
MX2T
MN2T
T_925
T2_ENSMEAN_MA1
T2_M1
T_925_M1
lon
lat
elev
sin_hod
cos_hod
sin_doy
cos_doy
analysishour
```

The exact required columns depend on the model and script being run.

## Installation

Create and activate a Python environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The current dependency file includes packages for data processing, plotting, classical machine learning, deep learning, and graph neural network workflows, including `numpy`, `pandas`, `polars`, `pyarrow`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `optuna`, `torch`, `torch-geometric`, `geopandas`, `shapely`, and `shap`.

For GPU or HPC environments, install PyTorch, XGBoost, and PyTorch Geometric versions that are compatible with your CUDA and driver setup. The plain `requirements.txt` installation may not be sufficient for all GPU environments.

## Recommended workflow

A typical workflow is:

1. Prepare or obtain compatible internal datasets.
2. Check paths and configuration values inside the script you want to run.
3. Train a model.
4. Run inference using the trained model.
5. Generate evaluation plots and diagnostics.

Example:

```bash
# Activate environment
source venv/bin/activate

# Train a model
python scripts/xgboost_train_tuning.py

# Run inference
python scripts/xgboost_inference.py

# Generate plots or diagnostics
python scripts/QQ-plot.py
```

Because this repository contains standalone research scripts, review each script before running it in a new environment.

## Main model workflows

### XGBoost workflow

#### `scripts/xgboost_train_tuning.py`

Trains an XGBoost model for temperature bias correction.

The training workflow is intended to:

- read prepared ML-ready input data
- define predictor variables
- split data into training, validation, and test periods
- tune hyperparameters with Optuna
- train a final XGBoost model
- save trained model artifacts and tuning outputs

Run:

```bash
python scripts/xgboost_train_tuning.py
```

Before running, check:

- input data path
- output path
- feature list
- train/validation/test date logic
- Optuna trial settings
- available CPU/GPU resources

#### `scripts/xgboost_inference.py`

Runs inference with a trained XGBoost model.

The inference workflow is intended to:

- load a trained XGBoost model
- read prepared input data
- predict the forecast bias
- apply the correction to raw temperature forecasts
- write outputs for later evaluation or plotting

Run:

```bash
python scripts/xgboost_inference.py
```

Before running, verify that the feature list and preprocessing match the training script.

### LSTM workflow

#### `scripts/LSTM_train.py`

Trains an LSTM-based bias-correction model with PyTorch.

The LSTM workflow uses prepared time-series-style input data and learns to predict temperature forecast error from sequential station and forecast features.

Run:

```bash
python scripts/LSTM_train.py
```

Before running, check:

- prepared input path
- output model path
- sequence construction logic
- feature list
- batch size
- number of epochs
- train/validation/test split
- GPU configuration

#### `scripts/LSTM_inference.py`

Runs inference using a trained LSTM model.

Run:

```bash
python scripts/LSTM_inference.py
```

Before running, confirm that:

- the correct model checkpoint is loaded
- normalization/statistics files match the trained model
- the inference feature order matches training
- output directories exist or can be created

### Graph neural network workflow

#### `scripts/GNN_train_tuning.py`

Trains and tunes a graph-based temperature bias-correction model.

The GNN workflow is intended to use station metadata and meteorological predictors in a spatial learning setup. It relies on PyTorch and PyTorch Geometric.

Run:

```bash
python scripts/GNN_train_tuning.py
```

Before running, check:

- graph construction logic
- station metadata requirements
- input feature columns
- train/validation/test periods
- Optuna tuning settings
- PyTorch Geometric installation

#### `scripts/gnn_inference.py`

Runs inference using a trained graph-based model.

Run:

```bash
python scripts/gnn_inference.py
```

Before running, verify that:

- the graph structure matches training
- station ordering is consistent
- feature preprocessing matches the training workflow
- the correct model run directory is used

## Analysis and visualization scripts

The repository includes plotting and diagnostic scripts for model evaluation.

### Forecast and station diagnostics

- `station_plots.py` — station-level visualizations
- `gnn_station_plot.py` — station plots for GNN outputs
- `observation_checks.py` — observation-data checks and diagnostics
- `observation_data.py` — observation-related processing/checking
- `correlation_plot.py` — correlation visualizations

### Model evaluation plots

- `QQ-plot.py` — Q-Q plots for forecast or model error distributions
- `metrics_station_plot.py` — station-level metric plots
- `hit_rate.py` — hit-rate metric calculations
- `hit_rate_plot.py` — hit-rate visualizations
- `scatterplot_multi.py` — multi-panel scatter plots
- `single_analysistime_plot.py` — plots for selected forecast analysis times
- `mos_plot_multi.py` — MOS-vs-ML comparison visualizations
- `shap_analysis.py` — SHAP-based feature-effect analysis

Most plotting scripts expect inference or metrics outputs to already exist.

## Outputs

Depending on the script, generated outputs may include:

```text
models/
metrics/
figures/
```

Generated data, models, metrics, figures, CSV files, caches, virtual environments, and IDE files are excluded by `.gitignore`.

## Troubleshooting

### `FileNotFoundError`

Check the input path expected by the script. The data is not public, so local paths must point to internally available prepared datasets.

### Missing columns

Check that the prepared input file contains the features required by the selected script. Training and inference scripts must use the same feature names and feature ordering.

### GPU or CUDA errors

Check your PyTorch, CUDA, driver, and PyTorch Geometric versions. On shared GPU/HPC systems, verify that the selected CUDA device is available.

### Different results between training and inference

Verify that:

- the same feature list is used
- the same preprocessing and normalization are used
- station ordering is consistent
- graph construction is identical for GNN inference
- the correct checkpoint/model artifacts are loaded

## Development notes

This repository is currently a research-script repository, not a packaged Python library.

Recommended future improvements:

- Add a central path/configuration utility.
- Add an example config file for local paths.
- Replace remaining hard-coded paths with command-line arguments.
- Add a small synthetic dataset for testing.
- Add input schema checks before training and inference.
- Move shared feature definitions into a common module.
- Add tests for metrics and preprocessing utilities.
- Add CI checks for formatting and syntax.
- Document model-specific workflows under a `docs/` directory.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
