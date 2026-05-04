# nwp-ml-bc-temperature

Machine learning workflows for bias-correcting near-surface temperature forecasts from numerical weather prediction (NWP) data.

This repository contains Python scripts for training, running, and evaluating machine learning models that estimate temperature forecast bias. The data used by this project is not public, so this README focuses on the modeling workflow, expected prepared inputs, outputs, and script roles rather than documenting internal data extraction or raw data preparation steps.

## Overview

Numerical weather prediction models can contain systematic biases in 2-metre temperature forecasts. This project explores machine learning approaches for estimating the forecast error and applying a correction to the raw forecast.

The main idea is to learn the temperature bias:

```text
bias = observed_temperature - forecast_temperature
```

and then apply the predicted bias to the raw forecast:

```text
corrected_temperature = raw_forecast_temperature + predicted_bias
```

The repository is organized as a collection of research and experiment scripts rather than a packaged command-line application.

## Repository structure

```text
.
├── scripts/
│   ├── data_parsing.py
│   ├── ml_data.py
│   ├── xgboost_train_tuning.py
│   ├── xgboost_inference.py
│   ├── LSTM_train.py
│   ├── LSTM_inference.py
│   ├── GNN_train_tuning.py
│   ├── gnn_inference.py
│   ├── mos_inference.py
│   ├── forecast_summary.py
│   ├── metrics_plot.py
│   ├── station_plots.py
│   └── ...
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Data availability

The datasets used in this project are not public.

Several scripts were developed for an internal/local research environment and may contain hard-coded local paths. Users outside that environment will need access to equivalent prepared datasets and may need to update paths before running the scripts.

This README therefore does not describe the full raw-data acquisition or internal preprocessing process. Instead, it documents the expected prepared data structure at a high level.

## Expected prepared input data

The training and inference scripts expect prepared tabular forecast-observation datasets, typically stored as Parquet files.

At a high level, the prepared data should contain:

- station identifier
- forecast analysis time
- forecast valid time
- forecast lead time
- observed 2-metre air temperature
- raw forecast 2-metre temperature
- additional NWP predictor variables
- station metadata such as longitude, latitude, and elevation
- engineered temporal features such as hour-of-day and day-of-year encodings

Common columns referenced by the scripts include:

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

The exact required columns depend on the model and script being used.

## Installation

Create a Python environment and install the project dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For GPU or HPC environments, install PyTorch, XGBoost, and PyTorch Geometric according to the target system and CUDA version. Some deep learning dependencies may require environment-specific installation commands.

## Recommended workflow

Because the repository consists of standalone scripts, the usual workflow is:

1. Prepare or obtain compatible Parquet datasets.
2. Check and adjust paths inside the relevant scripts.
3. Train a model.
4. Run inference using the trained model.
5. Generate metrics and plots.

A typical local directory layout used by the scripts is:

```text
~/thesis_project/
├── data/
│   ├── ml_data/
│   ├── ml_data_full/
│   └── ml_data_prepared/
├── models/
├── metrics/
└── figures/
```

You may use a different layout, but paths in the scripts should be updated accordingly.

## Script guide

### Data interface and feature preparation

These scripts are related to creating or transforming prepared machine learning datasets. They are included for reproducibility of the internal research workflow, but they may not run without access to the original non-public data.

#### `scripts/data_parsing.py`

Combines forecast data, station metadata, and observations into monthly Parquet files.

This script is specific to the internal/local data layout and should be treated as an example of how the merged modeling dataset was produced. It may require non-public source files.

#### `scripts/ml_data.py`

Creates machine-learning-ready datasets from combined Parquet files.

Typical operations include:

- filtering stations
- removing duplicate records
- selecting model variables
- creating time-based features
- writing monthly Parquet files for model training

This script is useful as a reference for expected model input features, but it may need path and column adjustments for other datasets.

#### `scripts/ml_data_full.py` / related prepared-data scripts

Additional dataset preparation variants used in experiments.

These scripts should be reviewed before use because they may reflect a particular experiment, data period, station subset, or local file layout.

### XGBoost workflow

#### `scripts/xgboost_train_tuning.py`

Trains an XGBoost model for temperature bias correction.

The script is designed to:

- read prepared ML Parquet files
- define predictor variables
- create time-based training, validation, and test splits
- tune hyperparameters with Optuna
- train a final XGBoost model
- save the trained model and tuning outputs

The model target is the temperature bias, typically computed from observed and forecast temperatures.

Run:

```bash
python scripts/xgboost_train_tuning.py
```

Before running, check:

- input data paths
- output model path
- selected feature columns
- date ranges used for train/validation/test splitting
- Optuna trial count and training settings

#### `scripts/xgboost_inference.py`

Runs inference using a trained XGBoost model.

The script typically:

- loads a saved XGBoost model
- reads prepared input data
- predicts the forecast bias
- applies the predicted correction to the raw temperature forecast
- writes evaluation output for later metrics and plotting

Run:

```bash
python scripts/xgboost_inference.py
```

Before running, check:

- model file path
- input data path
- output metrics path
- feature list consistency with training

### LSTM workflow

#### `scripts/LSTM_train.py`

Trains an LSTM-based model for temperature bias correction.

The script uses prepared time-series-style input data and trains a neural network model with PyTorch.

Run:

```bash
python scripts/LSTM_train.py
```

Before running, check:

- sequence construction logic
- selected features
- training/validation/test periods
- batch size and number of epochs
- model output path
- CPU/GPU settings

#### `scripts/LSTM_inference.py`

Runs inference for a trained LSTM model.

The script loads a saved model and applies it to prepared forecast data.

Run:

```bash
python scripts/LSTM_inference.py
```

Before running, verify that the inference feature order and preprocessing match the training script.

### Graph neural network workflow

#### `scripts/GNN_train_tuning.py`

Trains and tunes a graph-based model workflow.

The graph workflow is intended to use station metadata and meteorological features in a spatial learning setup. It uses PyTorch and PyTorch Geometric.

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

Before running, confirm that the graph structure, station ordering, and feature preprocessing match training.

### MOS baseline / comparison workflow

#### `scripts/mos_inference.py`

Runs inference or evaluation for a MOS-style baseline/comparison workflow.

Use this script to compare machine learning corrections against an existing statistical correction or baseline forecast, where available.

Before running, check the expected input files and output metric paths.

### Evaluation and plotting scripts

The repository includes scripts for summarizing results and producing diagnostic plots.

Common analysis scripts include:

- `forecast_summary.py`
- `forecast_summary_plots.py`
- `metrics_plot.py`
- `metrics_station_plot.py`
- `QQ-plot.py`
- `correlation_plot.py`
- `hit_rate.py`
- `hit_rate_plot.py`
- `station_plots.py`
- `station_group_plot.py`
- `scatterplot_multi.py`
- `pred_obs_plot.py`
- `shap_analysis.py`

These scripts are generally used after inference outputs have been written.

Typical outputs include:

```text
metrics/
figures/
```

Before running plotting scripts, check:

- expected metrics file names
- expected column names
- output figure directory
- station grouping or filtering assumptions

## Example workflow

The exact commands depend on the available prepared data and local paths, but a typical workflow is:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Optional: create or transform ML-ready data
# Requires access to compatible prepared/internal data.
python scripts/ml_data.py

# 3. Train a model
python scripts/xgboost_train_tuning.py

# 4. Run inference
python scripts/xgboost_inference.py

# 5. Generate plots or metrics summaries
python scripts/metrics_plot.py
```

## Important notes

- The raw data used by this project is not public.
- Some scripts contain hard-coded local paths and may require editing before use.
- The repository is not currently structured as an installable Python package.
- The scripts should be reviewed before running in a new environment.
- Feature lists must remain consistent between training and inference.
- Deep learning workflows may require environment-specific GPU/CUDA setup.
- Generated data, models, metrics, and figures should not be committed unless intentionally added.

## Outputs

Depending on the workflow, generated outputs may include:

```text
data/ml_data/
data/ml_data_full/
data/ml_data_prepared/
models/
metrics/
figures/
```

The repository `.gitignore` excludes many generated files and directories, including data, figures, metrics, models, JSON outputs, virtual environments, caches, and CSV files.

## Troubleshooting

### `FileNotFoundError`

Check whether the script contains a local hard-coded path. Update the input/output path to match your environment.

### Missing columns

Check that the prepared Parquet files contain the required feature columns for the selected model. Training and inference scripts must use the same feature order and preprocessing.

### PyTorch Geometric installation errors

Install PyTorch and PyTorch Geometric using versions compatible with your Python, CUDA, and system environment.

### Different results between training and inference

Verify that:

- the same feature list is used
- station ordering is consistent
- temporal features are generated in the same way
- scaling or preprocessing steps are identical
- the correct model checkpoint is loaded

## Suggested improvements

Future improvements that would make the repository easier to reuse include:

- Add a central configuration file for paths and model settings.
- Replace hard-coded paths with command-line arguments.
- Add a small synthetic or mock dataset for testing.
- Add scripts that validate the input schema before training.
- Move shared feature lists and utilities into common modules.
- Add automated tests for feature engineering and metrics.
- Add CI checks for formatting and linting.
- Document each model workflow in separate files under `docs/`.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
