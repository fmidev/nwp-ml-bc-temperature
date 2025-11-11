
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
MODEL_PATH = "bias_model_tuned_ah_best_2019.json"
MODEL_TAG = "tuned_ah_2019"

OUTDIR = HOME / "thesis_project" / "metrics" / "2019_tuned_ah"
OUTDIR.mkdir(parents=True, exist_ok=True)

SPLIT_COLUMN = "analysistime"  # or "validtime"

# Features and label
TEMP_FC   = "T2"
LABEL_OBS = "obs_TA"

weather = [
    "MSL", TEMP_FC, "D2", "U10", "V10", "LCC", "MCC", "SKT",
    "MX2T", "MN2T", "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1"
]
meta   = ["leadtime", "lon", "lat", "elev", "sin_hod", "cos_hod", "sin_doy", "cos_doy", "analysishour"]
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
        
    for year in range(2019, 2026):
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


        # save
        tag = f"{SPLIT_COLUMN}_{MODEL_TAG}_{year}"
        all_df.write_parquet(   OUTDIR / f"eval_rows_{tag}.parquet")

        print("Saved to:", OUTDIR)

if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    main()