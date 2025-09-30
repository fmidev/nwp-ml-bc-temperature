
import os
from pathlib import Path
import numpy as np
import polars as pl
from xgboost import XGBRegressor

# Paths
HOME = Path.home()
DATA_GLOB = HOME / "thesis_project" / "data" / "ml_data" / "ml_data_*.parquet"
MODEL_PATH_BASE  = "bias_model_tuned_best.json"
MODEL_PATH_TUNED = "bias_model_tuned_weighted_best.json"

OUTDIR = HOME / "thesis_project" / "metrics"
OUTDIR.mkdir(parents=True, exist_ok=True)

# What defines the test split
TEST_YEAR = 2024               # <— change if needed
SPLIT_COLUMN = "analysistime"  # or "validtime"

# Features and label
TEMP_FC   = "T2"
LABEL_OBS = "obs_TA"

weather = [
    "MSL", TEMP_FC, "D2", "U10", "V10", "LCC", "MCC", "SKT",
    "MX2T", "MN2T", "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1"
]
meta   = ["leadtime", "lon", "lat", "elev", "sin_hod", "cos_hod", "sin_doy", "cos_doy"]
FEATS  = weather + meta
ID     = ["SID", "analysistime", "validtime", "leadtime"]


# Load models
base_model = XGBRegressor()
base_model.load_model(MODEL_PATH_BASE)

tuned_model = XGBRegressor()
tuned_model.load_model(MODEL_PATH_TUNED)

def add_predictions(df):
    """Add corrected temps from both models. Expects FEATS + TEMP_FC present.
            Params: 
                df = Dataframe of the data
            Returns: Dataframe with the corrected tempreratures for both models"""
            
    # Guard
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features at inference: {missing}")

    # Check for base forecast
    if df[TEMP_FC].null_count() > 0:
        # cannot add bias back if base forecast missing
        df = df.filter(pl.col(TEMP_FC).is_not_null())

    if df.height == 0:
        return df

    # Get the fetures and cast to float32 to save memory
    X = df.select(FEATS).to_numpy().astype(np.float32, copy=False)

    # Get the predictions (bias) for the models
    bias_base  = base_model.predict(X)
    bias_tuned = tuned_model.predict(X)

    # Get the base forecast
    raw_fc = df[TEMP_FC].to_numpy()

    # Make the corrected temperature based on the forecast and predicted bias
    corrected_base  = raw_fc + bias_base
    corrected_tuned = raw_fc + bias_tuned

    return df.with_columns(
        pl.Series("corrected_base",  corrected_base),
        pl.Series("corrected_tuned", corrected_tuned)
    )

def metrics_frame(df, target, preds):
    """
    Function to compute evaluation metrics for one or more prediction columns
        Params:
            df = DataFrame containing the target column and prediction columns
            target = The name of the target column
            preds  = A list of prediction column names
        Returns:
            DataFrame = A Polars DataFrame containing metrics for each prediction column:
                        - rmse_<pred> : Root Mean Squared Error
                        - mae_<pred>  : Mean Absolute Error
                        - mbe_<pred>  : Mean Bias Error
                        - r2_<pred>   : Coefficient of determination (R²)
                        - r_<pred>    : Pearson correlation coefficient
    """
    y = pl.col(target)
    exprs = []

    # Loop through predictions
    for p in preds:
        yhat  = pl.col(p)
        resid = y - yhat

        # Covariance
        cov   = ((y - y.mean()) * (yhat - yhat.mean())).mean()
        # Denominator
        denom = y.std() * yhat.std()
        # Pearson correlation coefficient
        r_expr = pl.when(denom == 0).then(None).otherwise(cov / denom).alias(f"r_{p}")

        exprs.extend([
            resid.pow(2).mean().sqrt().alias(f"rmse_{p}"), # RMSE
            resid.abs().mean().alias(f"mae_{p}"),          # MAE
            (yhat - y).mean().alias(f"mbe_{p}"),           # MBE
            pl.when(y.var(ddof=0) == 0)
              .then(None)
              .otherwise(1 - (resid.pow(2).mean() / y.var(ddof=0)))
              .alias(f"r2_{p}"),                           # R_squred 
            r_expr,
        ])

    return df.select(exprs)



def metrics_grouped(df, by, target, preds):
    """
    Function to compute grouped evaluation metrics for one or more prediction columns
        Params:
            df = A Polars DataFrame containing the target column, prediction columns,
                 and optional "raw_fc" baseline column
            by = A list of column names to group the metrics by (e.g., ["leadtime", "station"])
            target = The name of the target column (string)
            preds  = A list of prediction column names (list of strings)
        Returns:
            DataFrame = A Polars DataFrame grouped by the specified columns, containing:
                        - n             : Number of samples per group
                        - rmse_raw      : RMSE of the raw forecast (if "raw_fc" exists)
                        - rmse_<pred>   : Root Mean Squared Error per prediction
                        - mae_<pred>    : Mean Absolute Error per prediction
                        - mbe_<pred>    : Mean Bias Error per prediction
                        - r2_<pred>     : Coefficient of determination (R²) per prediction
                        - r_<pred>      : Pearson correlation coefficient per prediction
                        - skill_vs_raw_<pred> : Relative skill compared to raw forecast (if available)
    """
    y = pl.col(target)

    # Add count once
    aggs = [pl.len().alias("n")]

    # Raw baseline RMSE (if available)
    has_raw = "raw_fc" in df.columns
    if has_raw:
        aggs.append((y - pl.col("raw_fc")).pow(2).mean().sqrt().alias("rmse_raw"))

    # Loop through prediction
    for p in preds:
        yhat  = pl.col(p)
        resid = y - yhat

        # Covariance
        cov   = ((y - y.mean()) * (yhat - yhat.mean())).mean()
        # Denominator
        denom = y.std() * yhat.std()
        # Pearson correlation coefficient
        r_expr = pl.when(denom == 0).then(None).otherwise(cov / denom).alias(f"r_{p}")

        aggs.extend([
            resid.pow(2).mean().sqrt().alias(f"rmse_{p}"),  # RMSE
            resid.abs().mean().alias(f"mae_{p}"),           # MAE
            (yhat - y).mean().alias(f"mbe_{p}"),            # MBE
            pl.when(y.var(ddof=0) == 0)
              .then(None)
              .otherwise(1 - (resid.pow(2).mean() / y.var(ddof=0)))
              .alias(f"r2_{p}"),                           # R_squared
            r_expr,
        ])

    out = df.group_by(by).agg(aggs).sort(by)

    # skill vs raw after aggregation (can reference rmse_* now)
    if has_raw:
        out = out.with_columns([
            (1 - (pl.col(f"rmse_{p}") / pl.col("rmse_raw"))).alias(f"skill_vs_raw_{p}")
            for p in preds
        ])
    return out


def safe_month(col):
    """
    Function to safely extract the month as an integer from a string column
        Params:
            col = The name of the column (string) containing datetime-like values
        Returns:
            Expr = A Polars expression that parses the column to datetime 
                   (non-strict, non-exact) and extracts the month component (1–12)
    """
    return pl.col(col).str.strptime(pl.Datetime, strict=False, exact=False).dt.month()


def main():

    # Get the data
    files = sorted(Path(DATA_GLOB.parent).glob(DATA_GLOB.name))
    if not files:
        raise FileNotFoundError(f"No files found for {DATA_GLOB}")

    accumulated = []

    # Needed columns
    needed = list(set(ID + FEATS + [LABEL_OBS]))  

    # Loop through files
    for f in files:
        # Scan lazily, select minimal colums, filter test year, deduplicate
        lf = (
            pl.scan_parquet(str(f))
              .select(needed)
        )

        # Filter by year on chosen split column
        lf = lf.with_columns(pl.col(SPLIT_COLUMN).cast(pl.Utf8))
        lf = lf.filter(pl.col(SPLIT_COLUMN).str.strptime(pl.Datetime, strict=False, exact=False).dt.year() == TEST_YEAR)

        # Check if there is anything in the file
        head = lf.head(1).collect(engine="streaming")
        if head.height == 0:
            continue

        # Deduplicate to one row per (SID, analysistime, leadtime) keeping latest validtime
        lf = (
            lf.sort(["SID", "analysistime", "leadtime", "validtime"])
              .unique(subset=["SID", "analysistime", "leadtime"], keep="last")
        )

        df = lf.collect(engine="streaming")

        if df.height == 0:
            continue

        
       
        # Drop rows without observation or base forecast
        df = df.filter(pl.col(TEMP_FC).is_not_null() & pl.col(LABEL_OBS).is_not_null())

        if df.height == 0:
            continue

        # Add predictions
        df = add_predictions(df)

        # Build convenience columns
        df = df.rename({TEMP_FC: "raw_fc"})

        # Keep only rows where predictions exist
        if ("corrected_base" not in df.columns) or ("corrected_tuned" not in df.columns):
            continue

        accumulated.append(df.select(ID + ["raw_fc", LABEL_OBS, "corrected_tuned", "corrected_tuned_weighted"]))

    if not accumulated:
        raise RuntimeError("No rows collected for the specified TEST_YEAR and filters.")

    all_df = pl.concat(accumulated, how="vertical_relaxed")

    # Add group helpers
    all_df = (
        all_df
        .with_columns(
            month = safe_month("validtime"),
        )
    )

    
    # Get the metrics
    preds = ["raw_fc", "corrected_tuned", "corrected_tuned_weighted"]

    # Overall
    overall = metrics_frame(all_df.select([LABEL_OBS] + preds), LABEL_OBS, preds)

    # By leadtime
    by_lt   = metrics_grouped(all_df, ["leadtime"], LABEL_OBS, ["corrected_tuned", "corrected_tuned_weighted"])
    # By station ID
    by_sid  = metrics_grouped(all_df, ["SID"], LABEL_OBS, ["corrected_tuned", "corrected_tuned_weighted"])
    # By month
    by_mon  = metrics_grouped(all_df, ["month"], LABEL_OBS, ["corrected_tuned", "corrected_tuned_weighted"])

    # Also provide a compact overall table with skill vs raw
    overall_skill = (
        all_df
        .select([
            ((pl.col(LABEL_OBS) - pl.col("raw_fc")).pow(2).mean().sqrt()).alias("rmse_raw"),
            ((pl.col(LABEL_OBS) - pl.col("corrected_tuned")).pow(2).mean().sqrt()).alias("rmse_corrected_tuned"),
            ((pl.col(LABEL_OBS) - pl.col("corrected_tuned_weighted")).pow(2).mean().sqrt()).alias("rmse_corrected_tuned_weighted"),
        ])
        .with_columns(
            (1 - (pl.col("rmse_corrected_tuned")  / pl.col("rmse_raw"))).alias("skill_vs_raw_corrected_tuned"),
            (1 - (pl.col("rmse_corrected_tuned_weighted") / pl.col("rmse_raw"))).alias("skill_vs_raw_corrected_tuned_weighted"),
        )
    )

    # SAVE
    overall.write_csv(OUTDIR / f"metrics_overall_{SPLIT_COLUMN}_{TEST_YEAR}.csv")
    overall_skill.write_csv(OUTDIR / f"metrics_overall_skill_{SPLIT_COLUMN}_{TEST_YEAR}.csv")
    by_lt.write_csv(OUTDIR / f"metrics_by_leadtime_{SPLIT_COLUMN}_{TEST_YEAR}.csv")
    by_sid.write_csv(OUTDIR / f"metrics_by_station_{SPLIT_COLUMN}_{TEST_YEAR}.csv")
    by_mon.write_csv(OUTDIR / f"metrics_by_month_{SPLIT_COLUMN}_{TEST_YEAR}.csv")

    # Tidy “long” file for custom plotting later
    all_df.write_parquet(OUTDIR / f"eval_rows_{SPLIT_COLUMN}_{TEST_YEAR}.parquet")

    print("Saved to:", OUTDIR)

if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    main()
