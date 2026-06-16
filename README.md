# nwp-ml-bc-temperature

Minimal XGBoost workflow branch for bias-correcting near-surface temperature forecasts from numerical weather prediction (NWP) data.

This branch contains only the scripts needed to build clean training data, train a CPU XGBoost model, run inference, and generate evaluation plots. It is intended for users who want the XGBoost temperature workflow without the additional model families and research utilities from the main branch.

## Overview

Numerical weather prediction models can contain systematic near-surface temperature biases. This workflow uses machine learning to estimate the forecast error and apply that estimate as a correction to the raw forecast.

The core target is the forecast bias:

```text
bias = observation - forecast
```

The corrected forecast is then computed as:

```text
corrected_forecast = raw_forecast + predicted_bias
```

This minimal branch supports multiple temperature-related targets:

- hourly air temperature bias (`target_bias_TA`)
- hourly dewpoint bias (`target_bias_TD`)
- daily maximum temperature bias (`target_bias_TMAX`)
- daily minimum temperature bias (`target_bias_TMIN`)

## Current branch structure

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── ml_data.py
    ├── results_plot.py
    ├── results_plot_extremes.py
    ├── xgboost_inference_clean.py
    └── xgboost_train_cpu.py
```

## Data

The scripts expect prepared forecast, observation, and station datasets to already be available in the local execution environment. The source data used by the project is not public, so this README focuses on the workflow, expected prepared inputs, and outputs rather than documenting raw data access.

## Expected input data

The workflow uses three prepared input sources before training:

- monthly forecast files in CSV or Parquet format
- yearly observation CSV files
- a station metadata CSV containing station identifiers and location metadata

`scripts/ml_data.py` merges those inputs and writes clean monthly Parquet files for XGBoost. Those clean files contain shared feature columns once and expose multiple target columns for model training.

Common columns used across the workflow include:

```text
SID
analysistime
validtime
leadtime
obs_TA
obs_TD
T2
D2
MX2T
MN2T
U10
V10
MSL
SKT
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
target_bias_TA
target_bias_TD
target_bias_TMAX
target_bias_TMIN
```

The exact required columns depend on which step of the workflow you are running.

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

The branch dependency file is intentionally minimal and only includes packages required by the five scripts in this branch.

If you want to use Optuna hyperparameter tuning with `scripts/xgboost_train_cpu.py --tune`, install it separately:

```bash
pip install optuna
```

The plotting scripts also require a station CSV and a world shapefile such as Natural Earth administrative boundaries.

## Recommended workflow

A typical workflow is:

1. Create clean monthly training files with `scripts/ml_data.py`.
2. Train a CPU XGBoost model with `scripts/xgboost_train_cpu.py`.
3. Run inference with `scripts/xgboost_inference_clean.py`.
4. Generate hourly TA/TD evaluation plots with `scripts/results_plot.py`.
5. Generate event-level TMAX/TMIN plots with `scripts/results_plot_extremes.py`.

Example:

```bash
# Activate environment
source venv/bin/activate

# Build clean monthly XGBoost input files
python scripts/ml_data.py \
  --forecast-input /path/to/forecasts \
  --observations /path/to/observations \
  --stations /path/to/stations.csv \
  --output-dir /path/to/full_output \
  --clean-xgb-output-dir /path/to/clean_output \
  --clean-only

# Train hourly temperature-bias model
python scripts/xgboost_train_cpu.py \
  --input "/path/to/clean_output/ml_data_clean_*.parquet" \
  --output /path/to/model_output \
  --target-col target_bias_TA \
  --external-memory \
  --cpu-clean-input

# Run inference for selected months
python scripts/xgboost_inference_clean.py \
  --input-dir /path/to/clean_output \
  --model-path /path/to/model_output/bias_model_cpu_external_memory_target_bias_TA.json \
  --output-dir /path/to/eval_output \
  --model-tag ta_cpu \
  --target-col target_bias_TA \
  --clean-input \
  --months 2025-03,2025-04 \
  --single-output
```

Review script arguments before running in a new environment, especially local paths, target-column choices, and output directories.

## Main workflow scripts

### `scripts/ml_data.py`

Creates monthly clean Parquet files for XGBoost from prepared forecast, observation, and station data.

The preprocessing workflow is intended to:

- read monthly forecast files
- read yearly observation CSV files
- merge hourly observations and aligned daily extrema
- compute clean multi-target training columns
- write one clean Parquet file per month

Typical outputs include:

```text
<output-prefix>_YYYY-MM.parquet
<clean-output-prefix>_YYYY-MM.parquet
```

### `scripts/xgboost_train_cpu.py`

Trains an XGBoost model using CPU external-memory training from cleaned monthly Parquet files.

The training workflow is intended to:

- read one or more cleaned monthly Parquet files
- select the requested target column
- split files into train, validation, and test periods
- optionally tune hyperparameters with Optuna
- train the final model
- save model, parameter, and test-report artifacts

Typical outputs include:

```text
bias_model_cpu_external_memory_<target>.json
fixed_params_cpu_external_memory_<target>.json
best_params_cpu_external_memory_<target>.json
test_report_cpu_external_memory_<target>.json
```

### `scripts/xgboost_inference_clean.py`

Runs inference on cleaned Parquet files using a trained XGBoost model.

The inference workflow is intended to:

- load a trained XGBoost model
- select files by year or exact month list
- predict bias for the requested target
- reconstruct corrected forecasts and observation columns
- write evaluation-ready Parquet outputs

Typical output pattern:

```text
eval_rows_validtime_<model-tag>_<period>.parquet
```

### `scripts/results_plot.py`

Creates station-level evaluation maps and plots for hourly variables such as air temperature and dewpoint.

The plotting workflow can:

- evaluate one ML model against raw forecasts
- compare two ML models
- optionally compare ML outputs against precomputed MOS evaluation files
- save station-metric CSV files and map figures

Typical outputs include station metric tables and figure files under the selected output directory.

### `scripts/results_plot_extremes.py`

Evaluates TMAX or TMIN inference outputs at event level instead of hourly row level.

The workflow is intended to:

- read one inference Parquet file
- aggregate copied hourly rows into one event row per station and analysistime
- compute station-level metrics
- save event rows and station metrics
- generate Europe map plots for raw and corrected performance

Typical outputs include:

```text
event_rows_TMAX_<model-tag>_<aggregation>.parquet
event_rows_TMIN_<model-tag>_<aggregation>.parquet
station_metrics_TMAX_<model-tag>_<aggregation>.csv
station_metrics_TMIN_<model-tag>_<aggregation>.csv
```

## Outputs

Depending on the stage of the workflow, generated outputs may include:

```text
clean monthly parquet files
trained XGBoost model JSON files
parameter JSON files
evaluation parquet files
station metric CSV files
map and summary figures
```

## Troubleshooting

### `FileNotFoundError`

Check that all input paths, model paths, and output directories match your local environment.

### Missing columns

Check that the prepared data contains the features and target columns expected by the selected script. Training and inference must use matching feature names and compatible target definitions.

### `optuna` import warning

`optuna` is optional. Install it only if you plan to run `scripts/xgboost_train_cpu.py --tune`.

### Plotting or shapefile errors

Check that the stations CSV and world shapefile exist and that `geopandas` is installed correctly in your environment.

### Different results between training and inference

Verify that:

- the same feature set is used
- the selected target column matches the trained model
- the correct model file is loaded
- month or year filtering matches the intended evaluation period

## License

This branch belongs to the same repository as the upstream project and follows the same MIT licensing terms.
