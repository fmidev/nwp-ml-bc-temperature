# nwp-ml-bc-temperature

Machine learning workflows for bias-correcting near-surface temperature forecasts from numerical weather prediction (NWP) data.

This repository contains research scripts for training, running inference, evaluating, and visualizing machine-learning-based temperature bias correction models.

## Branch note: configurable scripts

The branch, `configurable_scipts`, contains updated versions of the scripts where hardcoded local paths have been replaced with command-line arguments.

Use this branch instead of `main` to run the scripts on:

- your own user account
- a different workstation
- an HPC environment
- your own data

The main change is that scripts now accept paths and runtime settings through arguments such as:

```bash
--input
--output-dir
--model-dir
--run-dir
--prep-dir
--model-tag
--sample-size
--device
--cuda-visible-devices
```

This means that instead of editing paths inside the Python scripts, you should pass the paths when running the script.

Example:

```bash
python scripts/correlation_plot.py \
  --input "$HOME/path/to/your_data" \
  --output-dir "$HOME/output/path" \
  --output-name output_name.svg \
  --sample-size 200000
```

> Note: The `main` branch may still contain older versions of some scripts with hardcoded local paths. For reusable command-line usage, use this `configurable_scipts` branch.

## Overview

Numerical weather prediction models can contain systematic near-surface temperature biases. This project uses machine learning to estimate the forecast error and apply that estimate as a correction to the raw forecast.

The core target is the forecast temperature bias:

```text
bias = observed_temperature - forecast_temperature
```

The corrected forecast is computed as:

```text
corrected_temperature = raw_forecast_temperature + predicted_bias
```

The repository is organized as a collection of standalone experiment and analysis scripts rather than an installable Python package.

## Repository structure

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

The scripts expect prepared meteorological forecast, observation, and station datasets to already be available in the local execution environment.

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
LCC
MCC
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


Most plotting and analysis scripts expect a comparator model called MOS.

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

The dependency file includes packages for data processing, plotting, classical machine learning, deep learning, and graph neural network workflows, including:

```text
numpy
pandas
polars
pyarrow
matplotlib
scikit-learn
xgboost
optuna
torch
torch-geometric
geopandas
shapely
shap
joblib
```

For GPU or HPC environments, install PyTorch, XGBoost, and PyTorch Geometric versions that are compatible with your CUDA and driver setup. The plain `requirements.txt` installation may not be sufficient for all GPU environments.

## Recommended workflow

A typical workflow is:

1. Clone the repository (preferably the configurable_scripts branch for easier implementation).
2. Prepare or obtain compatible input datasets.
3. Install the Python dependencies.
4. Train a model or use an existing trained model.
5. Run inference to create corrected forecast outputs.
6. Generate metrics, plots, and diagnostics.

Example:

```bash
git clone https://github.com/fmidev/nwp-ml-bc-temperature.git
cd nwp-ml-bc-temperature
git checkout configurable_scipts

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run scripts by passing paths as command-line arguments.

## XGBoost workflow

### Train XGBoost model

Script:

```text
scripts/xgboost_train_tuning.py
```

Purpose:

- read ML-ready Parquet data
- split into train/validation/test periods
- tune XGBoost hyperparameters with Optuna
- train final model
- save model and metadata

### Run XGBoost inference

Script:

```text
scripts/xgboost_inference.py
```

Purpose:

- load a trained XGBoost model
- read ML-ready input data
- predict forecast bias
- add predicted bias to raw forecast
- write evaluation rows for later metrics and plots

Expected output files are typically named like:

```text
eval_rows_validtime_tuned_full_2024.parquet
eval_rows_validtime_tuned_full_2025.parquet
```

## GNN workflow

### Train GNN model

Script:

```text
scripts/GNN_train_tuning.py
```

Purpose:

- build a graph between stations
- train a GATv2-based graph neural network
- optionally tune graph/model parameters with Optuna
- save all artifacts required for inference

The training script saves artifacts such as:

```text
gnn_model.pt
preproc.joblib
stations.parquet
sid_to_idx.json
edge_index.pkl
edge_attr.pkl
graph_params.json
```

These artifacts must be kept together for GNN inference.

### Run GNN inference

Script:

```text
scripts/gnn_inference.py
```

Purpose:

- load a trained GNN run directory
- load graph and preprocessing artifacts
- predict station-level bias
- write corrected forecast evaluation rows

### Plot GNN station graph

Script:

```text
scripts/gnn_station_plot.py
```

Purpose:

- load GNN graph artifacts
- plot station graph edges and stations
- optionally subsample edges for readability


## LSTM workflow

### Train LSTM model

Script:

```text
scripts/LSTM_train.py
```

Purpose:

- optionally prepare raw ML data into sequence-ready Parquet files
- fit normalization statistics on training data
- optionally tune LSTM parameters with Optuna
- train final LSTM model
- save model, statistics, config, Optuna study, and results

First prepare sequence-ready files:

```bash
python scripts/LSTM_train.py --prepare
```

Model can also be trained using fixed parameters:

Example fixed parameters:
```
  "hidden": 192,
  "num_layers": 1,
  "dropout": 0.2447411578889518,
  "lr": 0.000727288685484562,
  "wd": 0.000628977181948703,
  "grad_clip": 1.4159046082342293,
  "batch_size": 16384,
  "seq_len": 24
```

### Run LSTM inference

Script:

```text
scripts/LSTM_inference.py
```

Purpose:

- load a trained LSTM run directory
- load training config and normalization statistics
- generate corrected forecasts for the test period
- write yearly evaluation Parquet files


## Analysis and visualization scripts

The repository includes plotting and diagnostic scripts for model evaluation. Most plotting scripts expect inference or metrics outputs to already exist.

### Correlation heatmap

Script:

```text
scripts/correlation_plot.py
```

Purpose:

- read ML-ready Parquet data
- sample rows
- compute Spearman correlations
- save a heatmap

### SHAP analysis

Script:

```text
scripts/shap_analysis.py
```

Purpose:

- load an XGBoost model
- sample rows from ML-ready input data
- compute SHAP values
- save SHAP summary and dependence plots

### Station map and elevation histogram

Script:

```text
scripts/station_plots.py
```

Purpose:

- read station metadata
- plot station locations
- plot elevation distribution

### Q-Q plots

Script:

```text
scripts/QQ-plot.py
```

Purpose:

- compare observed and predicted distributions
- generate combined and station-specific Q-Q plots

### Hit-rate calculation

Script:

```text
scripts/hit_rate.py
```

Purpose:

- align MOS and ML evaluation rows
- calculate hit rates overall and by lead time
- save CSV files
- can be run on full station set or specified station subset


### Hit-rate plot

Script:

```text
scripts/hit_rate_plot.py
```

Purpose:

- read hit-rate CSV
- plot MOS and ML hit rates by lead time
- plot model difference on secondary axis
- can be run on full station set or specified station subset based on what was used in the hit rate calculation


### Single analysis-time time series

Script:

```text
scripts/single_analysistime_plot.py
```

Purpose:

- compare raw forecast, MOS, XGBoost, GNN, LSTM, and observations for selected station/init times
- can also be used in anomaly mode to plot forecasts initalizations with anomalous observed temperatures

### Scatter density plots

Script:

```text
scripts/scatterplot_multi.py
```

Purpose:

- compare observations and model predictions with density plots
- optionally group stations and plot by monthly windows or seasons


### Station-level metric maps

Script:

```text
scripts/metrics_station_plot.py
```

Purpose:

- compute station-level RMSE, MAE, bias, skill scores, and model differences
- plot station maps and histograms
- can be used in single mode or paired comparison mode

## Outputs

Depending on the script, generated outputs may include:

```text
models/
metrics/
figures/
```

Common generated files include:

```text
*.pt
*.json
*.pkl
*.parquet
*.csv
*.svg
*.png
*.pdf
```

Generated data, models, metrics, figures, CSV files, caches, virtual environments, and IDE files are excluded by `.gitignore`.

## Troubleshooting

### Missing columns

Check that the prepared input file contains the features required by the selected script. Training and inference scripts must use the same feature names and feature ordering.

For Parquet files, you can inspect columns with Python:

```python
import polars as pl

schema = pl.scan_parquet("example.parquet").collect_schema()
print(schema.names())
```

### GPU or CUDA errors

Check your PyTorch, CUDA, driver, and PyTorch Geometric versions. On shared GPU/HPC systems, verify that the selected CUDA device is available.

For scripts with CUDA support, use:

```bash
--cuda-visible-devices 0
```

or force CPU:

```bash
--device cpu
```

### Different results between training and inference

Verify that:

- the same feature list is used
- the same preprocessing and normalization are used
- station ordering is consistent
- graph construction is identical for GNN inference
- the correct checkpoint/model artifacts are loaded
- the model tag matches the corrected output column


## Development notes

This repository is currently a research-script repository, not a packaged Python library.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
