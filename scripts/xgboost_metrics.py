
import os
from pathlib import Path
import numpy as np
import polars as pl
from xgboost import XGBRegressor

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

# Paths
HOME = Path.home()
DATA_GLOB = HOME / "thesis_project" / "data" / "ml_data_full" 
MODEL_PATH = "bias_model_tuned_w_best_full.json"
MODEL_TAG = "tuned_w_full"

OUTDIR = HOME / "thesis_project" / "metrics" / "full_tuned_w"
OUTDIR.mkdir(parents=True, exist_ok=True)

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
model = XGBRegressor()
model.load_model(MODEL_PATH)
CORR_COL = f"corrected_{MODEL_TAG}" 




def add_predictions(df):
    """Add corrected temps from both models. Expects FEATS + TEMP_FC present.
            Params: 
                df = Dataframe of the data
            Returns: Dataframe with the corrected tempreratures for both models"""
            
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features at inference: {missing}")

    # require base forecast present to add bias back
    if df[TEMP_FC].null_count() > 0:
        df = df.filter(pl.col(TEMP_FC).is_not_null())
    if df.height == 0:
        return df

    X = df.select(FEATS).to_numpy().astype(np.float32, copy=False)
    bias_hat = model.predict(X)

    raw_fc = df[TEMP_FC].to_numpy()
    corrected = raw_fc + bias_hat

    return df.with_columns(pl.Series(CORR_COL, corrected))

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
    for p in preds:
        yhat  = pl.col(p)
        resid = y - yhat
        cov   = ((y - y.mean()) * (yhat - yhat.mean())).mean()
        denom = y.std() * yhat.std()
        r_expr = pl.when(denom == 0).then(None).otherwise(cov / denom).alias(f"r_{p}")
        exprs.extend([
            resid.pow(2).mean().sqrt().alias(f"rmse_{p}"),
            resid.abs().mean().alias(f"mae_{p}"),
            (yhat - y).mean().alias(f"mbe_{p}"),
            pl.when(y.var(ddof=0) == 0).then(None).otherwise(1 - (resid.pow(2).mean() / y.var(ddof=0))).alias(f"r2_{p}"),
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
    aggs = [pl.len().alias("n")]
    has_raw = "raw_fc" in df.columns
    if has_raw:
        aggs.append((y - pl.col("raw_fc")).pow(2).mean().sqrt().alias("rmse_raw"))

    for p in preds:
        yhat  = pl.col(p)
        resid = y - yhat
        cov   = ((y - y.mean()) * (yhat - yhat.mean())).mean()
        denom = y.std() * yhat.std()
        r_expr = pl.when(denom == 0).then(None).otherwise(cov / denom).alias(f"r_{p}")
        aggs.extend([
            resid.pow(2).mean().sqrt().alias(f"rmse_{p}"),
            resid.abs().mean().alias(f"mae_{p}"),
            (yhat - y).mean().alias(f"mbe_{p}"),
            pl.when(y.var(ddof=0) == 0).then(None).otherwise(1 - (resid.pow(2).mean() / y.var(ddof=0))).alias(f"r2_{p}"),
            r_expr,
        ])

    out = df.group_by(by).agg(aggs).sort(by)
    if has_raw:
        out = out.with_columns([(1 - (pl.col(f"rmse_{p}") / pl.col("rmse_raw"))).alias(f"skill_vs_raw_{p}") for p in preds])
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
        
    for year in range(2024, 2026):
        print(f"Processing year {year}...")
        
        files = sorted(DATA_GLOB.glob(f"ml_data_full_{year}-*.parquet"))
        accumulated = []

        # Needed columns
        needed = list(set(ID + FEATS + [LABEL_OBS]))  

        for f in files:
            lf = pl.scan_parquet(str(f)).select(needed)
            lf = lf.with_columns(pl.col(SPLIT_COLUMN).cast(pl.Utf8))
            lf = lf.filter(pl.col(SPLIT_COLUMN).str.strptime(pl.Datetime, strict=False, exact=False).dt.year() == year)

            if lf.head(1).collect(engine="streaming").height == 0:
                continue

            # dedup like your single-init logic
            lf = lf.sort(["SID", "analysistime", "leadtime", "validtime"]).unique(subset=["SID", "analysistime", "leadtime"], keep="last")
            df = lf.collect(engine="streaming")
            if df.height == 0:
                continue

            df = df.filter(pl.col(TEMP_FC).is_not_null() & pl.col(LABEL_OBS).is_not_null())
            if df.height == 0:
                continue

            df = add_predictions(df)
            df = df.rename({TEMP_FC: "raw_fc"})
            if CORR_COL not in df.columns:
                continue

            accumulated.append(df.select(ID + ["raw_fc", LABEL_OBS, CORR_COL]))

            if not accumulated:
                print(f"No rows collected for year {year}; skipping.")
                continue

            all_df = pl.concat(accumulated, how="vertical_relaxed").with_columns(month=safe_month("validtime"))

            """# metrics
            preds = ["raw_fc", CORR_COL]
            overall = metrics_frame(all_df.select([LABEL_OBS] + preds), LABEL_OBS, preds)
            by_lt   = metrics_grouped(all_df, ["leadtime"], LABEL_OBS, [CORR_COL])
            by_sid  = metrics_grouped(all_df, ["SID"], LABEL_OBS, [CORR_COL])
            by_mon  = metrics_grouped(all_df, ["month"], LABEL_OBS, [CORR_COL])

            overall_skill = (
            all_df.select([
                ((pl.col(LABEL_OBS) - pl.col("raw_fc")).pow(2).mean().sqrt()).alias("rmse_raw"),
                ((pl.col(LABEL_OBS) - pl.col(CORR_COL)).pow(2).mean().sqrt()).alias(f"rmse_{CORR_COL}"),
                ]).with_columns((1 - (pl.col(f"rmse_{CORR_COL}") / pl.col("rmse_raw"))).alias(f"skill_vs_raw_{CORR_COL}")))"""

        # save
        tag = f"{SPLIT_COLUMN}_{MODEL_TAG}_{year}"
        #overall.write_csv(      OUTDIR / f"metrics_overall_{tag}.csv")
        #overall_skill.write_csv(OUTDIR / f"metrics_overall_skill_{tag}.csv")
        #by_lt.write_csv(        OUTDIR / f"metrics_by_leadtime_{tag}.csv")
        #by_sid.write_csv(       OUTDIR / f"metrics_by_station_{tag}.csv")
        #by_mon.write_csv(       OUTDIR / f"metrics_by_month_{tag}.csv")
        all_df.write_parquet(   OUTDIR / f"eval_rows_{tag}.parquet")

        print("Saved to:", OUTDIR)

if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    main()