import os
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
HOME = Path.home()
DATA_DIR = HOME / "thesis_project" / "data"
METRICS_DIR = HOME / "thesis_project" / "metrics"
MOS_DIR = METRICS_DIR / "mos"
ML1_DIR = METRICS_DIR / "2019_tuned_ah"

OUT_DIR = METRICS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns
SPLIT = "analysistime"        # analysistime defines window
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS_CORR = "corrected_mos"

ML1_TAG = "tuned_ah_2019"
ML1_CORR = f"corrected_{ML1_TAG}"


# ------------------------------------------------
# Data loading and preparation functions
# ------------------------------------------------

def load_eval_rows_evaldir(eval_dir: Path, pattern: str, cols: list[str]) -> pd.DataFrame:
    """Load the data
        Params: 
            eval_dir = Directory of the files
            pattern = Filename pattern
            tag = Name of the column/model data is loaded for
        Return: Loaded data
    """

    # Gather the files
    files = sorted(eval_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")
    
    # Loop through files and collect needed data
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=[c for c in cols if c != "SID"] + (["SID"] if "SID" in cols else []))
        dfs.append(df)

    # Concatenate into one dataframe
    out = pd.concat(dfs, ignore_index=True)
    if "SID" in out.columns:
        out["SID"] = out["SID"].astype(str)
    if SPLIT in out.columns:
        out[SPLIT] = pd.to_datetime(out[SPLIT], errors="coerce").dt.tz_localize(None)
    return out

def mos_coverage_window(mos_eval: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get the time window based on MOS"""

    df = mos_eval.dropna(subset=[MOS_CORR, RAW, OBS]).copy()
    if df.empty:
        raise ValueError("MOS eval rows have no usable values.")
    return df[SPLIT].min(), df[SPLIT].max()

def align_mos_window(mos_eval: pd.DataFrame, ml_eval: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """Use MOS to define the window+samples, inner-join ML.
        Params: 
            mos_eval = Dataframe with MOS predictions
            ml_eval = Dataframe with ML predictions
            t0 = Start time
            t1 = End time
        Returns: Joined dataframe"""

    # Split the data
    mos = mos_eval[(mos_eval[SPLIT] >= t0) & (mos_eval[SPLIT] <= t1)].copy()
    keys = ["SID", SPLIT, "validtime", "leadtime"]
    keep_cols = keys + [RAW, OBS, MOS_CORR, ML1_CORR]

    # Join the dataframes
    joined = (
        mos.merge(ml_eval[keys + [ML1_CORR]], on=keys, how="inner")
           .dropna(subset=[OBS, RAW, MOS_CORR, ML1_CORR])
           [keep_cols].copy()
    )
    if not joined.empty:
        print(f"MOS-driven window (single-ML): {joined[SPLIT].min()} → {joined[SPLIT].max()}")
    else:
        print("Warning: No aligned rows after MOS-driven single-ML join.")
    return joined



def hit_rate(
    df: pd.DataFrame,
    pred_cols: list[str] | tuple[str, ...] = (MOS_CORR, ML1_CORR),
    obs_col: str = OBS,
    by: str | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute hit rates for one or more prediction columns in a single call.

    Returns a dict with:
      - 'overall': rows per model with hits, n, hit_rate
      - 'by' (optional): grouped breakdown with your 'by' columns + model + hits, n, hit_rate
    """
    if df.empty:
        return {"overall": pd.DataFrame(columns=["model", "hits", "n", "hit_rate"])}

    if isinstance(pred_cols, str):
        pred_cols = [pred_cols]

    # Vectorized threshold logic
    obs = df[obs_col]
    cold = obs <= 258.15
    cool = (obs > 258.15) & (obs <= 268.15)
    warm = obs > 268.15

    overall_rows = []
    by_rows = []

    # Prepare grouping
    if by is not None and not isinstance(by, list):
        by = [by]

    for col in pred_cols:
        if col not in df.columns:
            # Skip gracefully if a requested column is missing
            overall_rows.append({"model": col, "hits": 0, "n": 0, "hit_rate": float("nan")})
            continue

        # APlly mask to get the overall hit-rate
        diff = (obs - df[col]).abs()
        mask = (cold & (diff < 5.0)) | (cool & (diff < 3.5)) | (warm & (diff < 2.5))

        n = int(mask.size)
        hits = int(mask.sum())
        overall_rows.append({"model": col, "hits": hits, "n": n, "hit_rate": hits / n if n else float("nan")})

        # If specified column get hit-rate per that column
        if by:
            # Group on original df indices to keep alignment with mask
            g = df.assign(__hit__=mask).groupby(by, observed=True)["__hit__"]
            grp = g.agg(hits=lambda s: int(s.sum()), n=lambda s: int(s.size)).reset_index()
            grp["hit_rate"] = grp["hits"] / grp["n"]
            grp["model"] = col
            by_rows.append(grp)

    # Store and return results
    overall = pd.DataFrame(overall_rows).sort_values("hit_rate", ascending=False, na_position="last").reset_index(drop=True)
    out = {"overall": overall}

    if by_rows:
        by_df = pd.concat(by_rows, ignore_index=True)
        # Put 'model' at front for readability
        cols = ["model"] + [c for c in by_df.columns if c not in ("model")]
        out["by"] = by_df[cols]

    return out


def main():

    # Load
    mos_eval = load_eval_rows_evaldir(
        MOS_DIR,
        pattern=f"eval_rows_{SPLIT}_MOS_2025_summer.parquet",
        cols=["SID", SPLIT, "validtime", "leadtime", RAW, OBS, MOS_CORR],
    )
    ml1_eval = load_eval_rows_evaldir(
        ML1_DIR,
        pattern=f"eval_rows_{SPLIT}_{ML1_TAG}_20*.parquet",
        cols=["SID", SPLIT, "validtime", "leadtime", RAW, OBS, ML1_CORR],
    )

    # Get time frame
    t0, t1 = mos_coverage_window(mos_eval)
    print(f"MOS coverage window: {t0} → {t1}")


    # Join the data 
    joined = align_mos_window(mos_eval, ml1_eval, t0, t1)
    print("Rows after alignment (single):", len(joined))
    print("Stations after alignment (single):", joined["SID"].nunique())

    # Get the overall hit-rate and per-leadtime hit-rate
    results = hit_rate(joined, pred_cols=[MOS_CORR, ML1_CORR], by="leadtime")

    rename_map = {
        MOS_CORR: "MOS",
        ML1_CORR: "EC_ML",  # or "Tuned ML", "ML Model", etc.
    }

    # Apply renames
    results["overall"]["model"] = results["overall"]["model"].replace(rename_map)
    if "by" in results:
        results["by"]["model"] = results["by"]["model"].replace(rename_map)


    print("Overall hit rates:")
    print(results["overall"].to_string(index=False))

    if "by" in results:
        print("\nHit rate by leadtime:")

    
    # Combine and save the results
    overall = results["overall"].copy()
    overall["level"] = "overall"

    if "by" in results:
        by_df = results["by"].copy()
        by_df["level"] = "by_leadtime"
        # Ensure same columns order and compatibility
        combined = pd.concat([overall, by_df], ignore_index=True, sort=False)
    else:
        combined = overall

    out_path = OUT_DIR / f"hit_rate_from2019_summer.csv"

    combined.to_csv(out_path, index=False)
    print(f"\n Combined results saved to: {out_path}")


if __name__ == "__main__":
    main()