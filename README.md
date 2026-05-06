# nwp-ml-bc-temperature

Machine learning workflows for bias-correcting near-surface temperature forecasts from numerical weather prediction (NWP) data.

This repository contains research scripts for training, running inference, evaluating, and visualizing machine-learning-based temperature bias correction models.

## Branch note: configurable scripts

This branch, `configurable_scipts`, contains updated versions of the scripts where hardcoded local paths have been replaced with command-line arguments.

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


Most examples below use placeholder paths such as:

```bash
$HOME/project/data/ml_data_full
$HOME/project/models
$HOME/project/metrics
$HOME/project/figures
```

Replace these with the paths used in your own environment.

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

Most plotting and analysis scripts require a comparator model called MOS.

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

## Checking script options

Most updated scripts can show their available command-line arguments with:

```bash
python scripts/correlation_plot.py --help
python scripts/xgboost_train_tuning.py --help
python scripts/xgboost_inference.py --help
python scripts/GNN_train_tuning.py --help
python scripts/gnn_inference.py --help
python scripts/LSTM_train.py --help
python scripts/LSTM_inference.py --help
```

Use `--help` before running a script in a new environment.

## Recommended workflow

A typical workflow is:

1. Clone this branch.
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

Example:

```bash
python scripts/xgboost_train_tuning.py \
  --input "$HOME/project/data/ml_data_full/ml_data_full_*.parquet" \
  --output-dir "$HOME/project/models/xgboost_tuned_full" \
  --figures-dir "$HOME/project/figures/xgboost_tuned_full" \
  --model-name "bias_model_tuned_full_new.json" \
  --n-trials 30 \
  --cuda-visible-devices 1
```

If the script expects a directory instead of a glob, use:

```bash
python scripts/xgboost_train_tuning.py \
  --input "$HOME/project/data/ml_data_full" \
  --output-dir "$HOME/project/models/xgboost_tuned_full" \
  --figures-dir "$HOME/project/figures/xgboost_tuned_full"
```

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

Example:

```bash
python scripts/xgboost_inference.py \
  --input-dir "$HOME/project/data/ml_data_full" \
  --model-path "$HOME/project/models/xgboost_tuned_full/bias_model_tuned_full_new.json" \
  --output-dir "$HOME/project/metrics/tuned_full_new" \
  --model-tag "tuned_full" \
  --start-year 2024 \
  --end-year 2025
```

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

Example without Optuna:

```bash
python scripts/GNN_train_tuning.py \
  --input "$HOME/project/data/ml_data_full/ml_data_full_*.parquet" \
  --coast-shp "$HOME/project/data/maps/coastline/ne_10m_coastline.shp" \
  --model-base-dir "$HOME/project/models/gnn_bias_correction" \
  --out-name "full_gnn_gat_lsm" \
  --cuda-visible-devices 1 \
  --device auto \
  --threads 16
```

Example with Optuna:

```bash
python scripts/GNN_train_tuning.py \
  --input "$HOME/project/data/ml_data_full" \
  --coast-shp "$HOME/project/data/maps/coastline/ne_10m_coastline.shp" \
  --model-base-dir "$HOME/project/models/gnn_bias_correction" \
  --run-optuna \
  --n-trials 30 \
  --cuda-visible-devices 1
```

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

Example:

```bash
python scripts/gnn_inference.py \
  --data-dir "$HOME/project/data/ml_data_full" \
  --run-dir "$HOME/project/models/gnn_bias_correction/full_gnn_gat_lsm" \
  --output-dir "$HOME/project/metrics/full_gnn_gat_lsm" \
  --model-tag "full_gnn_gat_lsm" \
  --start-year 2024 \
  --end-year 2025 \
  --eval-start "2024-09-01" \
  --eval-end "2025-09-01" \
  --threads 16
```

To force CPU inference:

```bash
python scripts/gnn_inference.py \
  --data-dir "$HOME/project/data/ml_data_full" \
  --run-dir "$HOME/project/models/gnn_bias_correction/full_gnn_gat_lsm" \
  --output-dir "$HOME/project/metrics/full_gnn_gat_lsm" \
  --device cpu
```

### Plot GNN station graph

Script:

```text
scripts/gnn_station_plot.py
```

Purpose:

- load GNN graph artifacts
- plot station graph edges and stations
- optionally subsample edges for readability

Example:

```bash
python scripts/gnn_station_plot.py \
  --graph-dir "$HOME/project/models/gnn_bias_correction/full_gnn_gat_lsm" \
  --world-shp "$HOME/project/data/maps/ne_110m_admin_0_countries.shp" \
  --output-dir "$HOME/project/figures/station_plots" \
  --output-name "station_graph_edges_lsm.svg" \
  --coast-threshold-km 20 \
  --max-edges-to-plot 2000
```

To plot all edges:

```bash
python scripts/gnn_station_plot.py \
  --graph-dir "$HOME/project/models/gnn_bias_correction/full_gnn_gat_lsm" \
  --world-shp "$HOME/project/data/maps/ne_110m_admin_0_countries.shp" \
  --output-dir "$HOME/project/figures/station_plots" \
  --max-edges-to-plot 0
```

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
python scripts/LSTM_train.py \
  --raw-input "$HOME/project/data/ml_data_full/ml_data_full_*.parquet" \
  --prep-dir "$HOME/project/data/ml_data_prepared" \
  --model-dir "$HOME/project/models" \
  --prepare
```

Train without Optuna:

```bash
python scripts/LSTM_train.py \
  --raw-input "$HOME/project/data/ml_data_full/ml_data_full_*.parquet" \
  --prep-dir "$HOME/project/data/ml_data_prepared" \
  --model-dir "$HOME/project/models" \
  --model-tag "bias_lstm_stream" \
  --cuda-visible-devices 1 \
  --threads 4
```

Train with Optuna:

```bash
python scripts/LSTM_train.py \
  --raw-input "$HOME/project/data/ml_data_full/ml_data_full_*.parquet" \
  --prep-dir "$HOME/project/data/ml_data_prepared" \
  --model-dir "$HOME/project/models" \
  --model-tag "bias_lstm_stream" \
  --run-optuna \
  --n-trials 30 \
  --cuda-visible-devices 1 \
  --threads 4
```

Train using fixed parameters from JSON:

```bash
python scripts/LSTM_train.py \
  --raw-input "$HOME/project/data/ml_data_full" \
  --prep-dir "$HOME/project/data/ml_data_prepared" \
  --model-dir "$HOME/project/models" \
  --model-tag "bias_lstm_stream" \
  --fixed-params-json "$HOME/project/models/best_lstm_params.json"
```

Example fixed parameter file:

```json
{
  "hidden": 192,
  "num_layers": 1,
  "dropout": 0.2447411578889518,
  "lr": 0.000727288685484562,
  "wd": 0.000628977181948703,
  "grad_clip": 1.4159046082342293,
  "batch_size": 16384,
  "seq_len": 24
}
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

Example using newest LSTM run:

```bash
python scripts/LSTM_inference.py \
  --prep-input "$HOME/project/data/ml_data_prepared/ml_data_prep_*.parquet" \
  --model-dir "$HOME/project/models" \
  --output-dir "$HOME/project/metrics/bias_lstm_stream" \
  --model-tag "bias_lstm_stream" \
  --cuda-visible-devices 1 \
  --threads 4
```

Example using a specific run directory:

```bash
python scripts/LSTM_inference.py \
  --prep-input "$HOME/project/data/ml_data_prepared" \
  --model-dir "$HOME/project/models" \
  --run-dir "$HOME/project/models/bias_lstm_20250831_120000" \
  --output-dir "$HOME/project/metrics/bias_lstm_stream" \
  --model-tag "bias_lstm_stream"
```

Run on CPU:

```bash
python scripts/LSTM_inference.py \
  --prep-input "$HOME/project/data/ml_data_prepared" \
  --model-dir "$HOME/project/models" \
  --output-dir "$HOME/project/metrics/bias_lstm_stream" \
  --device cpu
```

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

Example with input directory:

```bash
python scripts/correlation_plot.py \
  --input "$HOME/project/data/ml_data_full" \
  --output-dir "$HOME/project/figures" \
  --output-name correlation_map.svg \
  --sample-size 200000
```

Example with one file for testing:

```bash
python scripts/correlation_plot.py \
  --input "$HOME/project/data/ml_data_full/ml_data_full_2024-09.parquet" \
  --output-dir "$HOME/project/figures/test" \
  --output-name correlation_map_test.svg \
  --sample-size 50000
```

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

Example:

```bash
python scripts/shap_analysis.py \
  --input "$HOME/project/data/ml_data_full/ml_data_full_*.parquet" \
  --model-path "$HOME/project/models/xgboost_tuned_full/bias_model_tuned_full_new.json" \
  --output-dir "$HOME/project/figures/SHAP" \
  --n-explain 30000 \
  --holdout-days 365
```

### Station map and elevation histogram

Script:

```text
scripts/station_plots.py
```

Purpose:

- read station metadata
- plot station locations
- plot elevation distribution

Example:

```bash
python scripts/station_plots.py \
  --stations-csv "$HOME/project/data/stations.csv" \
  --world-shp "$HOME/project/data/maps/ne_110m_admin_0_countries.shp" \
  --output-dir "$HOME/project/figures/station_plots"
```

### Q-Q plots

Script:

```text
scripts/QQ-plot.py
```

Purpose:

- compare observed and predicted distributions
- generate combined and station-specific Q-Q plots

Example:

```bash
python scripts/QQ-plot.py \
  --mos-dir "$HOME/project/metrics/mos" \
  --ml-dir "$HOME/project/metrics/bias_lstm_stream" \
  --output-dir "$HOME/project/figures/QQ-plots/FULL/all" \
  --ml-tag "bias_lstm_stream" \
  --ml-name "EC_ML_LSTM" \
  --stations "100932" "101118" "101932" \
  --leadtime 240
```

### Hit-rate calculation

Script:

```text
scripts/hit_rate.py
```

Purpose:

- align MOS and ML evaluation rows
- calculate hit rates overall and by lead time
- save CSV files

Example:

```bash
python scripts/hit_rate.py \
  --mos-dir "$HOME/project/metrics/mos" \
  --ml-dir "$HOME/project/metrics/tuned_full_new" \
  --output-dir "$HOME/project/metrics/hitrate/XGBoost" \
  --ml-tag "tuned_full" \
  --ml-name "XGBoost" \
  --seasonal
```

Example with a station subset:

```bash
python scripts/hit_rate.py \
  --mos-dir "$HOME/project/metrics/mos" \
  --ml-dir "$HOME/project/metrics/tuned_full_new" \
  --output-dir "$HOME/project/metrics/hitrate/XGBoost" \
  --ml-tag "tuned_full" \
  --ml-name "XGBoost" \
  --stations "100917" "100932" "101932"
```

### Hit-rate plot

Script:

```text
scripts/hit_rate_plot.py
```

Purpose:

- read hit-rate CSV
- plot MOS and ML hit rates by lead time
- plot model difference on secondary axis

Example:

```bash
python scripts/hit_rate_plot.py \
  --input "$HOME/project/metrics/hitrate/XGBoost/hit_rate_XGBoost_summer.csv" \
  --output-dir "$HOME/project/figures/hit_rate" \
  --output-name "hit_rate_by_leadtime_summer.png" \
  --ml-model "EC_ML_XGBoost" \
  --title "Hit Rate Comparison by Leadtime Summer"
```

### Single analysis-time time series

Script:

```text
scripts/single_analysistime_plot.py
```

Purpose:

- compare raw forecast, MOS, XGBoost, GNN, LSTM, and observations for selected station/init times

Example:

```bash
python scripts/single_analysistime_plot.py \
  --station-file "$HOME/project/data/stations.csv" \
  --mos-dir "$HOME/project/metrics/mos" \
  --ml-dir "$HOME/project/metrics/tuned_full_new" \
  --gnn-dir "$HOME/project/metrics/full_gnn_gat" \
  --lstm-dir "$HOME/project/metrics/bias_lstm_stream" \
  --output-dir "$HOME/project/figures/MOSvsML_timeseries/single_analysistime" \
  --target-station "101932" \
  --inits "2024-12-13 12:00:00" "2024-12-13 00:00:00"
```

Example anomaly mode:

```bash
python scripts/single_analysistime_plot.py \
  --station-file "$HOME/project/data/stations.csv" \
  --mos-dir "$HOME/project/metrics/mos" \
  --ml-dir "$HOME/project/metrics/tuned_full_new" \
  --gnn-dir "$HOME/project/metrics/full_gnn_gat" \
  --lstm-dir "$HOME/project/metrics/bias_lstm_stream" \
  --output-dir "$HOME/project/figures/MOSvsML_timeseries/anomalies" \
  --anom \
  --anom-csv "$HOME/project/metrics/all_anomalous_analysistimes_by_station.csv"
```

### Scatter density plots

Script:

```text
scripts/scatterplot_multi.py
```

Purpose:

- compare observations and model predictions with density plots
- optionally group stations and plot by monthly windows or seasons

Example:

```bash
python scripts/scatterplot_multi.py \
  --station-file "$HOME/project/data/stations.csv" \
  --mos-dir "$HOME/project/metrics/mos" \
  --ml-dir "$HOME/project/metrics/2019_tuned_ah" \
  --output-dir "$HOME/project/figures/MOSvsML_timeseries/north-Finland/ldt48" \
  --ml-tag "tuned_ah_2019" \
  --ml-name "EC_ML_XGBoost_2019" \
  --stations "101886" "101928" "101932" "102016" "102033" "102035" \
  --leadtime 48 \
  --month-start "2024-09-01 00:00:00" \
  --month-end "2025-08-31 12:00:00"
```

### Station-level metric maps

Script:

```text
scripts/metrics_station_plot.py
```

Purpose:

- compute station-level RMSE, MAE, bias, skill scores, and model differences
- plot station maps and histograms

Example single-model comparison:

```bash
python scripts/metrics_station_plot.py \
  --stations-csv "$HOME/project/data/stations.csv" \
  --world-shp "$HOME/project/data/maps/ne_110m_admin_0_countries.shp" \
  --mos-dir "$HOME/project/metrics/mos" \
  --ml1-dir "$HOME/project/metrics/bias_lstm_stream" \
  --output-base-dir "$HOME/project/figures/station_plots/evaluation" \
  --ml1-tag "bias_lstm_stream" \
  --ml1-name "EC_ML_LSTM" \
  --compare-mode "single" \
  --seasonal
```

Example pairwise ML comparison:

```bash
python scripts/metrics_station_plot.py \
  --stations-csv "$HOME/project/data/stations.csv" \
  --world-shp "$HOME/project/data/maps/ne_110m_admin_0_countries.shp" \
  --mos-dir "$HOME/project/metrics/mos" \
  --ml1-dir "$HOME/project/metrics/bias_lstm_stream" \
  --ml2-dir "$HOME/project/metrics/2019_tuned_new" \
  --output-base-dir "$HOME/project/figures/station_plots/evaluation" \
  --ml1-tag "bias_lstm_stream" \
  --ml1-name "EC_ML_LSTM" \
  --ml2-tag "tuned_ah_2019" \
  --ml2-name "EC_ML_XGB" \
  --compare-mode "pair" \
  --seasonal
```

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

### `FileNotFoundError`

Check the path passed to the script. The data is not public, so local paths must point to internally available prepared datasets.

Example:

```bash
ls "$HOME/project/data/ml_data_full"
```

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

### Script argument mismatch

If an example does not match your local script version, check the current script arguments:

```bash
python scripts/name_of_script.py --help
```

This branch is actively being adapted from standalone research scripts, so some argument names may differ between scripts.

## Development notes

This repository is currently a research-script repository, not a packaged Python library.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
