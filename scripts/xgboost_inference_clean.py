import argparse
import os
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb


# -----------------------------------------------------------------------------
# Global model state set in main()
# -----------------------------------------------------------------------------

MODEL_TAG = None
CORR_COL = None
model = None

TARGET_COL = None
BASE_FORECAST_COL = None
OBS_OUTPUT_COL = None


# -----------------------------------------------------------------------------
# Column names
# -----------------------------------------------------------------------------

SPLIT_COLUMN = "validtime"

TEMP_FC = "T2"
DEW_FC = "D2"
TMAX_FC = "MX2T"
TMIN_FC = "MN2T"

LABEL_OBS = "obs_TA"
TARGET_BIAS = "target_bias"

DEFAULT_TARGET_TO_BASE_FC = {
    "target_bias": TEMP_FC,
    "target_bias_TA": TEMP_FC,
    "target_bias_TD": DEW_FC,
    "target_bias_TMAX": TMAX_FC,
    "target_bias_TMIN": TMIN_FC,
}

DEFAULT_TARGET_TO_OBS_OUTPUT = {
    "target_bias": "obs_TA_reconstructed",
    "target_bias_TA": "obs_TA_reconstructed",
    "target_bias_TD": "obs_TD_reconstructed",
    "target_bias_TMAX": "obs_TMAX_reconstructed",
    "target_bias_TMIN": "obs_TMIN_reconstructed",
}

weather = [
    "MSL", TEMP_FC, DEW_FC, "U10", "V10", "LCC", "MCC", "SKT",
    TMAX_FC, TMIN_FC, "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1",
]

meta = [
    "leadtime", "lon", "lat", "elev",
    "sin_hod", "cos_hod", "sin_doy", "cos_doy", "analysishour",
]

FEATS = weather + meta

# leadtime is already in FEATS, but it is useful in output tables too.
# Keep this order for plotting.
ID = ["SID", "analysistime", "validtime", "leadtime"]


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained XGBoost bias-correction model on parquet data. "
            "Supports full combined data and clean XGBoost data with ID columns. "
            "Can process whole years or exact YYYY-MM month files. "
            "For clean multi-target data, use --target-col and --base-forecast-col."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing input parquet files.",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to trained XGBoost model JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where evaluation parquet files will be saved.",
    )

    parser.add_argument(
        "--model-tag",
        default="xgb_model",
        type=str,
        help="Tag used in corrected column name and output filenames.",
    )

    parser.add_argument(
        "--target-col",
        default="target_bias",
        type=str,
        help=(
            "Target bias column in clean input. Examples: target_bias_TA, "
            "target_bias_TD, target_bias_TMAX, target_bias_TMIN. "
            "Default: target_bias."
        ),
    )

    parser.add_argument(
        "--base-forecast-col",
        default=None,
        type=str,
        help=(
            "Forecast column to add predicted bias to. Examples: T2, D2, MX2T, MN2T. "
            "If omitted, inferred from --target-col when possible."
        ),
    )

    parser.add_argument(
        "--obs-output-col",
        default=None,
        type=str,
        help=(
            "Name of reconstructed observation output column. Examples: "
            "obs_TA_reconstructed, obs_TD_reconstructed, obs_TMAX_reconstructed. "
            "If omitted, inferred from --target-col when possible."
        ),
    )

    parser.add_argument(
        "--start-year",
        default=2024,
        type=int,
        help="First year to process, inclusive. Ignored if --months is set.",
    )

    parser.add_argument(
        "--end-year",
        default=2025,
        type=int,
        help="Last year to process, inclusive. Ignored if --months is set.",
    )

    parser.add_argument(
        "--months",
        default=None,
        type=str,
        help=(
            "Optional comma-separated YYYY-MM months to process exactly. "
            "Example: 2025-03,2025-04,2026-01. "
            "If set, this overrides --start-year and --end-year file selection."
        ),
    )

    parser.add_argument(
        "--threads",
        default="16",
        type=str,
        help="Number of CPU threads for OMP and MKL.",
    )

    parser.add_argument(
        "--clean-input",
        action="store_true",
        help=(
            "Use cleaned XGBoost parquet files containing ID columns + FEATS + target. "
            "In this mode observations are reconstructed as base_forecast + target_col."
        ),
    )

    parser.add_argument(
        "--file-pattern",
        default=None,
        type=str,
        help=(
            "Optional filename glob pattern. Can use {year} and/or {month}. "
            "Examples: ml_data_clean_{year}-*.parquet, ml_data_clean_{month}.parquet, "
            "ml_data_full_{month}.parquet. If omitted, defaults are based on "
            "--clean-input and whether --months is set."
        ),
    )

    parser.add_argument(
        "--output-row-errors",
        action="store_true",
        help=(
            "Include row-level raw/corrected errors and predicted bias columns in "
            "the output parquet. Useful for diagnostics but increases output size."
        ),
    )

    parser.add_argument(
        "--keep-features",
        action="store_true",
        help=(
            "Keep model feature columns in output parquet. Useful for debugging; "
            "usually not needed for plotting."
        ),
    )

    parser.add_argument(
        "--single-output",
        action="store_true",
        help=(
            "When --months is set, write one combined output parquet for all requested "
            "months instead of one file per month."
        ),
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_months(months_arg):
    if months_arg is None or months_arg.strip() == "":
        return None

    months = []
    for month in months_arg.split(","):
        month = month.strip()
        if not month:
            continue

        if len(month) != 7 or month[4] != "-":
            raise ValueError(f"Invalid month '{month}'. Expected YYYY-MM.")

        yyyy, mm = month.split("-")
        if not (yyyy.isdigit() and mm.isdigit()):
            raise ValueError(f"Invalid month '{month}'. Expected YYYY-MM.")

        if int(mm) < 1 or int(mm) > 12:
            raise ValueError(f"Invalid month '{month}'. Month must be 01..12.")

        months.append(month)

    if not months:
        return None

    return months


def safe_month(col):
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False, exact=False)
        .dt.month()
    )


def safe_year_filter_expr(col, year):
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False, exact=False)
        .dt.year()
        == int(year)
    )


def safe_month_filter_expr(col, month):
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False, exact=False)
        .dt.strftime("%Y-%m")
        == str(month)
    )


def infer_base_forecast_col(target_col, base_forecast_col):
    if base_forecast_col is not None:
        return base_forecast_col

    if target_col in DEFAULT_TARGET_TO_BASE_FC:
        return DEFAULT_TARGET_TO_BASE_FC[target_col]

    raise ValueError(
        f"Could not infer --base-forecast-col for target_col={target_col}. "
        "Pass --base-forecast-col explicitly."
    )


def infer_obs_output_col(target_col, obs_output_col):
    if obs_output_col is not None:
        return obs_output_col

    if target_col in DEFAULT_TARGET_TO_OBS_OUTPUT:
        return DEFAULT_TARGET_TO_OBS_OUTPUT[target_col]

    return f"obs_reconstructed_{target_col}"


def predict_bias(df):
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features at inference: {missing}")

    X = df.select(FEATS).to_numpy().astype(np.float32, copy=False)
    dmat = xgb.DMatrix(X, feature_names=FEATS)
    return model.predict(dmat)


def get_files_for_period(data_dir, args, year=None, month=None):
    if month is not None:
        year = int(month[:4])

        if args.file_pattern is not None:
            pattern = args.file_pattern.format(year=year, month=month)
        elif args.clean_input:
            pattern = f"ml_data_clean_{month}.parquet"
        else:
            pattern = f"ml_data_full_{month}.parquet"
    else:
        if args.file_pattern is not None:
            pattern = args.file_pattern.format(year=year)
        elif args.clean_input:
            pattern = f"ml_data_clean_{year}-*.parquet"
        else:
            pattern = f"ml_data_full_{year}-*.parquet"

    return sorted(data_dir.glob(pattern)), pattern


def select_existing(df, columns):
    return [c for c in columns if c in df.columns]


def needed_columns(clean_input):
    if clean_input:
        return list(dict.fromkeys(ID + FEATS + [TARGET_COL, BASE_FORECAST_COL]))
    return list(dict.fromkeys(ID + FEATS + [LABEL_OBS, BASE_FORECAST_COL]))


# -----------------------------------------------------------------------------
# Prediction functions
# -----------------------------------------------------------------------------

def add_predictions_full(df):
    """
    Add corrected values for full combined/evaluation data.

    Expects:
        FEATS + base forecast column + LABEL_OBS

    This branch is mainly for the old full data format. For multi-target clean data,
    use --clean-input.
    """
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features at inference: {missing}")

    if LABEL_OBS not in df.columns:
        raise ValueError(f"Full input is missing required observation column: {LABEL_OBS}")

    if BASE_FORECAST_COL not in df.columns:
        raise ValueError(f"Full input is missing base forecast column: {BASE_FORECAST_COL}")

    df = df.filter(
        pl.col(BASE_FORECAST_COL).is_not_null()
        & pl.col(LABEL_OBS).is_not_null()
    )

    if df.height == 0:
        return df

    bias_hat = predict_bias(df)
    raw_fc = df[BASE_FORECAST_COL].to_numpy()
    obs = df[LABEL_OBS].to_numpy()
    corrected = raw_fc + bias_hat

    return df.with_columns([
        pl.Series("raw_fc", raw_fc),
        pl.Series(f"bias_pred_{MODEL_TAG}", bias_hat),
        pl.Series(CORR_COL, corrected),
        pl.Series(f"error_raw_{MODEL_TAG}", obs - raw_fc),
        pl.Series(f"error_corrected_{MODEL_TAG}", obs - corrected),
    ])


def add_predictions_clean(df):
    """
    Add corrected values for clean XGBoost data with IDs.

    Clean multi-target input contains:
        ID columns + FEATS + TARGET_COL

    It may not contain an observation value directly, so reconstruct:
        OBS_OUTPUT_COL = BASE_FORECAST_COL + TARGET_COL

    Corrected forecast:
        corrected_<MODEL_TAG> = BASE_FORECAST_COL + predicted_bias
    """
    required = ID + FEATS + [TARGET_COL, BASE_FORECAST_COL]
    required = list(dict.fromkeys(required))

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in clean input: {missing}")

    df = df.filter(
        pl.col(BASE_FORECAST_COL).is_not_null()
        & pl.col(TARGET_COL).is_not_null()
    )

    if df.height == 0:
        return df

    bias_hat = predict_bias(df)
    raw_fc = df[BASE_FORECAST_COL].to_numpy()
    target_bias = df[TARGET_COL].to_numpy()
    obs_reconstructed = raw_fc + target_bias
    corrected = raw_fc + bias_hat

    return df.with_columns([
        pl.Series("raw_fc", raw_fc),
        pl.Series(OBS_OUTPUT_COL, obs_reconstructed),
        pl.Series(f"bias_pred_{MODEL_TAG}", bias_hat),
        pl.Series(CORR_COL, corrected),
        pl.Series(f"error_raw_{MODEL_TAG}", obs_reconstructed - raw_fc),
        pl.Series(f"error_corrected_{MODEL_TAG}", obs_reconstructed - corrected),
    ])


def output_columns(args, df):
    if args.clean_input:
        base_cols = ID + ["raw_fc", OBS_OUTPUT_COL, TARGET_COL, CORR_COL]
    else:
        base_cols = ID + ["raw_fc", LABEL_OBS, CORR_COL]

    extra_cols = []
    if args.output_row_errors:
        extra_cols.extend([
            f"bias_pred_{MODEL_TAG}",
            f"error_raw_{MODEL_TAG}",
            f"error_corrected_{MODEL_TAG}",
        ])

    if args.keep_features:
        extra_cols.extend(FEATS)

    out_cols = list(dict.fromkeys(base_cols + extra_cols))
    return select_existing(df, out_cols)


# -----------------------------------------------------------------------------
# Period processing
# -----------------------------------------------------------------------------

def process_period(data_dir, outdir, args, year=None, month=None):
    if month is not None:
        period_label = month
        print("=" * 80, flush=True)
        print(f"Processing month {month}...", flush=True)
    else:
        period_label = str(year)
        print("=" * 80, flush=True)
        print(f"Processing year {year}...", flush=True)

    files, pattern = get_files_for_period(data_dir, args, year=year, month=month)
    print(f"File pattern: {pattern}", flush=True)
    print(f"Files found: {len(files)}", flush=True)

    if not files:
        print(f"No files found for period {period_label}; skipping.", flush=True)
        return None

    accumulated = []
    needed = needed_columns(args.clean_input)

    for i, f in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Reading {f.name}", flush=True)

        lf = pl.scan_parquet(str(f)).select(needed)
        lf = lf.with_columns(pl.col(SPLIT_COLUMN).cast(pl.Utf8))

        # Filter by requested validtime period.
        if month is not None:
            lf = lf.filter(safe_month_filter_expr(SPLIT_COLUMN, month))
        else:
            lf = lf.filter(safe_year_filter_expr(SPLIT_COLUMN, year))

        if lf.head(1).collect(engine="streaming").height == 0:
            print("  no rows after validtime filter; skipping", flush=True)
            continue

        # Sorting is useful for plotting/reproducibility.
        # In clean mode, do NOT deduplicate: different leadtimes/analysistimes can
        # share SID + validtime and must be preserved.
        if args.clean_input:
            lf = lf.sort(["SID", "analysistime", "leadtime", "validtime"])
        else:
            lf = (
                lf.sort(["SID", "analysistime", "leadtime", "validtime"])
                .unique(
                    subset=["SID", "analysistime", "leadtime"],
                    keep="last",
                )
            )

        df = lf.collect(engine="streaming")
        if df.height == 0:
            continue

        if args.clean_input:
            df = add_predictions_clean(df)
        else:
            df = add_predictions_full(df)

        if df.height == 0 or CORR_COL not in df.columns:
            continue

        accumulated.append(df.select(output_columns(args, df)))

    if not accumulated:
        print(f"No rows collected for period {period_label}; skipping.", flush=True)
        return None

    all_df = (
        pl.concat(accumulated, how="vertical_relaxed")
        .with_columns(month=safe_month("validtime"))
    )

    if args.single_output and month is not None:
        # Caller will write combined output.
        return all_df

    tag = f"{SPLIT_COLUMN}_{MODEL_TAG}_{period_label}"
    output_path = outdir / f"eval_rows_{tag}.parquet"
    all_df.write_parquet(output_path)

    print("Saved to:", output_path, flush=True)
    print("Rows:", all_df.height, flush=True)
    print("Columns:", all_df.columns, flush=True)

    return all_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    global model, MODEL_TAG, CORR_COL
    global TARGET_COL, BASE_FORECAST_COL, OBS_OUTPUT_COL

    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = args.threads
    os.environ["MKL_NUM_THREADS"] = args.threads

    TARGET_COL = args.target_col
    BASE_FORECAST_COL = infer_base_forecast_col(args.target_col, args.base_forecast_col)
    OBS_OUTPUT_COL = infer_obs_output_col(args.target_col, args.obs_output_col)

    data_dir = Path(args.input_dir)
    model_path = Path(args.model_path)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    MODEL_TAG = args.model_tag
    CORR_COL = f"corrected_{MODEL_TAG}"

    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {data_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    months = parse_months(args.months)

    print("Loading model:", model_path, flush=True)
    model = xgb.Booster()
    model.load_model(str(model_path))

    print("Input directory:", data_dir, flush=True)
    print("Output directory:", outdir, flush=True)
    print("Clean input:", bool(args.clean_input), flush=True)
    print("Model tag:", MODEL_TAG, flush=True)
    print("Corrected column:", CORR_COL, flush=True)
    print("Target column:", TARGET_COL, flush=True)
    print("Base forecast column:", BASE_FORECAST_COL, flush=True)
    print("Observation output column:", OBS_OUTPUT_COL, flush=True)

    if months is not None:
        print("Month mode enabled. Months:", months, flush=True)

        month_frames = []
        for month in months:
            df_month = process_period(data_dir, outdir, args, month=month)
            if args.single_output and df_month is not None:
                month_frames.append(df_month)

        if args.single_output:
            if month_frames:
                all_months = pl.concat(month_frames, how="vertical_relaxed")
                first_month = months[0]
                last_month = months[-1]
                tag = f"{SPLIT_COLUMN}_{MODEL_TAG}_{first_month}_to_{last_month}"
                output_path = outdir / f"eval_rows_{tag}.parquet"
                all_months.write_parquet(output_path)
                print("=" * 80, flush=True)
                print("Saved combined month output to:", output_path, flush=True)
                print("Rows:", all_months.height, flush=True)
                print("Columns:", all_months.columns, flush=True)
            else:
                print("No month rows collected; no combined output written.", flush=True)

    else:
        for year in range(args.start_year, args.end_year + 1):
            process_period(data_dir, outdir, args, year=year)

    print("=" * 80, flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    main()

