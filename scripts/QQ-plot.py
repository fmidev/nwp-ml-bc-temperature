from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your ML model tag used when saving eval rows (=> column "corrected_<ML_TAG>")
ML_TAG = "tuned_ah_2019" 
# Name of the model for the plot  
ML_NAME = "EC_ML_XGBoost_2019"                    

# Plot settings
SHOW_PLOT = False
SAVE_PLOT = True
FIG_DPI   = 150

# Paths
HOME     = Path.home()
METRICS  = HOME / "thesis_project" / "metrics"
MOS_DIR  = METRICS / "mos"
ML_DIR   = METRICS / "2019_tuned_ah"                            
OUT_DIR  = HOME / "thesis_project" / "figures" / "QQ-plots" / "2019" / "all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns
SPLIT = "analysistime"
KEYS  = ["SID", SPLIT, "validtime", "leadtime"]
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS   = "corrected_mos"
MLCOL = f"corrected_{ML_TAG}"

SEASONAL = True

def load_eval_rows_evaldir(eval_dir: Path, pattern: str, tag: str):
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
        cols = KEYS + [RAW, OBS, MLCOL, MOS]
        # tolerate older files lacking RAW (we’ll drop if missing)
        existing = [c for c in cols if c in pl.scan_parquet(str(f)).collect_schema().keys()]
        df = pl.read_parquet(f, columns=existing).with_columns([
            pl.col("SID").cast(pl.Utf8),
            pl.col(SPLIT).cast(pl.Utf8),
        ])
        dfs.append(df)

    # Concatenate the into one dataframe
    out = pl.concat(dfs, how="vertical_relaxed")

    print(f"[INFO] {tag} rows loaded: {out.height:,}")
    return out

# --- Add this helper (anywhere above main) ---
def filter_leadtime_leq(df: pl.DataFrame, max_lt: int = 48) -> pl.DataFrame:
    # Coerce leadtime to Int64 if needed and filter
    return (
        df.with_columns(pl.col("leadtime").cast(pl.Int64))
          .filter(pl.col("leadtime") <= max_lt)
    )

def filter_stations(df: pl.DataFrame, sids: list[str] | list[int]) -> pl.DataFrame:
    return df.filter(pl.col("SID").cast(pl.Utf8).is_in([str(s) for s in sids]))


# =========================
# Q–Q plot helpers
# =========================
def _qq_points(a: np.ndarray, b: np.ndarray, probs: np.ndarray | None = None):
    """Return matched quantiles (qa, qb) for arrays a and b on a shared prob grid."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        raise ValueError("Empty array after NaN removal; cannot make Q–Q plot.")
    if probs is None:
        # Use a shared grid based on the smaller sample size but cap at 1000 points
        n = min(a.size, b.size, 2000)
        # Avoid 0 and 1 exactly to keep quantiles finite with extremes
        probs = (np.arange(1, n + 1) - 0.5) / n
    qa = np.quantile(a, probs, method="linear")
    qb = np.quantile(b, probs, method="linear")
    return qa, qb

def qqplot_arrays(x: np.ndarray, y: np.ndarray, title: str, outpath: Path | None = None):
    qa, qb = _qq_points(x, y)

    # Reference line (45°) spanning combined range
    lo = min(qa.min(), qb.min())
    hi = max(qa.max(), qb.max())

    plt.figure(dpi=FIG_DPI)
    plt.scatter(qa, qb, s=10, alpha=0.8, color="#8080ff", edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="#000d1a")
    plt.xlabel("Observed quantiles")
    plt.ylabel("Predicted quantiles")
    plt.title(title)
    plt.tight_layout()

    if SAVE_PLOT and outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath)
        print(f"[INFO] Saved: {outpath}")
    if SHOW_PLOT:
        plt.show()
    plt.close()

def qqplot_polars(df: pl.DataFrame, obs_col: str, pred_col: str,
                  title: str, fname: str):
    obs = df.select(pl.col(obs_col).cast(pl.Float64)).to_numpy().ravel()
    pred = df.select(pl.col(pred_col).cast(pl.Float64)).to_numpy().ravel()
    qqplot_arrays(
        obs, pred,
        title=title,
        outpath=OUT_DIR / fname if SAVE_PLOT else None
    )



def qqplot_by_station(df: pl.DataFrame, obs_col: str, pred_col: str,
                      model_name: str, prefix: str, season: str,  max_lt: int = 48):
    """Generate and save one Q–Q plot per station (SID) for leadtime <= max_lt."""
    df_filt = (
        df.with_columns(pl.col("leadtime").cast(pl.Int64))
          .filter(pl.col("leadtime") <= max_lt)
          .filter(pl.col(obs_col).is_not_null() & pl.col(pred_col).is_not_null())
    )

    sids = df_filt.select("SID").unique().to_series().to_list()
    print(f"[INFO] Generating per-station Q–Q plots for {len(sids)} stations (leadtime ≤ {max_lt})")

    for sid in sids:
        sub = df_filt.filter(pl.col("SID") == sid)
        if sub.height < 10:  # skip tiny samples
            continue
        title = f"Q–Q: Observed vs {model_name} \n (SID={sid}, leadtime ≤ {max_lt}h {season})"
        fname = f"{prefix}_qq_obs_vs_{model_name}_SID_{sid}_lead_le{max_lt}_{season}.svg"
        qqplot_polars(sub, obs_col, pred_col, title, fname)


# =========================
# Use in your main()
# =========================
def main():

    stations = ["100932", "101118", "101932"]
    leadtime = 240
    if SEASONAL:
        AVAILABLE = {
            "2024": ["autumn"],
            "2025": ["winter", "spring", "summer"],
        }

        for year, seasons in AVAILABLE.items():
            for season in seasons:

                mos_all = load_eval_rows_evaldir(MOS_DIR, f"eval_rows_{SPLIT}_MOS_{year}_{season}.parquet", MOS)
                ml_all = load_eval_rows_evaldir(ML_DIR, f"eval_rows_{SPLIT}_{ML_TAG}_{year}.parquet", MLCOL)

                mos_sub = filter_stations(mos_all, stations)
                ml_sub = filter_stations(ml_all, stations)

                # ---- Combine all leadtimes <= 48 ----
                mos_le48 = filter_leadtime_leq(mos_all, leadtime)
                ml_le48  = filter_leadtime_leq(ml_all, leadtime)

                # --- Combined Q–Q plots (all rows with leadtime <= 48) ---
                if MOS in mos_le48.columns and OBS in mos_le48.columns and mos_le48.height > 0:
                    qqplot_polars(
                        mos_le48, OBS, MOS,
                        title=f"Q–Q: Observed vs MOS \n (leadtime ≤ {leadtime}h, combined {season})",
                        fname=f"qq_obs_vs_MOS_lead_{leadtime}_combined{season}.svg"
                    )
                else:
                    print(f"[WARN] No MOS/OBS data after leadtime ≤ {leadtime} filter.")

                if MLCOL in ml_le48.columns and OBS in ml_le48.columns and ml_le48.height > 0:
                    qqplot_polars(
                        ml_le48, OBS, MLCOL,
                        title=f"Q–Q: Observed vs {ML_NAME} \n (leadtime ≤ {leadtime}h, combined {season})",
                        fname=f"qq_obs_vs_{ML_NAME}_lead_{leadtime}_combined{season}.svg"
                    )
                else:
                    print(f"[WARN] No ML/OBS data after leadtime ≤ {leadtime} filter.")

                if RAW in ml_le48.columns and OBS in ml_le48.columns and ml_le48.height > 0:
                    qqplot_polars(
                        ml_le48, OBS, RAW,
                        title=f"Q–Q: Observed vs ECMWF \n (leadtime ≤ {leadtime}h, combined {season})",
                        fname=f"qq_obs_vs_ECMWF_lead_{leadtime}_combined{season}.svg"
                    )
                else:
                    print(f"[WARN] No ML/OBS data after leadtime ≤ {leadtime} filter.")
                
                # --- Per-station plots (leadtime ≤ 48) ---
                qqplot_by_station(mos_sub, OBS, MOS, "MOS", prefix="MOS", season = season, max_lt=leadtime)
                qqplot_by_station(ml_sub, OBS, MLCOL, ML_NAME, prefix="ML", season = season, max_lt=leadtime)
                qqplot_by_station(ml_sub, OBS, RAW, "ECMWF", prefix="ECMWF", season = season, max_lt=leadtime)
    else:
        mos_all = load_eval_rows_evaldir(MOS_DIR, f"eval_rows_{SPLIT}_MOS_*.parquet", MOS)
        ml_all = load_eval_rows_evaldir(ML_DIR, f"eval_rows_{SPLIT}_{ML_TAG}_202*.parquet", MLCOL)

        mos_sub = filter_stations(mos_all, stations)
        ml_sub = filter_stations(ml_all, stations)

        # ---- Combine all leadtimes <= 48 ----
        mos_le48 = filter_leadtime_leq(mos_all, leadtime)
        ml_le48  = filter_leadtime_leq(ml_all, leadtime)

        # --- Combined Q–Q plots (all rows with leadtime <= 48) ---
        if MOS in mos_le48.columns and OBS in mos_le48.columns and mos_le48.height > 0:
            qqplot_polars(
                mos_le48, OBS, MOS,
                title=f"Q–Q: Observed vs MOS \n (leadtime ≤ {leadtime}h, combined)",
                fname=f"qq_obs_vs_MOS_lead_{leadtime}_combined.svg"
            )
        else:
            print(f"[WARN] No MOS/OBS data after leadtime ≤ {leadtime} filter.")

        if MLCOL in ml_le48.columns and OBS in ml_le48.columns and ml_le48.height > 0:
            qqplot_polars(
                ml_le48, OBS, MLCOL,
                title=f"Q–Q: Observed vs {ML_NAME} \n (leadtime ≤ {leadtime}h, combined)",
                fname=f"qq_obs_vs_{ML_NAME}_lead_{leadtime}_combinened.svg"
            )
        else:
            print(f"[WARN] No ML/OBS data after leadtime ≤ {leadtime} filter.")
        if RAW in ml_le48.columns and OBS in ml_le48.columns and ml_le48.height > 0:
            qqplot_polars(
                ml_le48, OBS, RAW,
                title=f"Q–Q: Observed vs ECMWF \n (leadtime ≤ {leadtime}h, combined)",
                fname=f"qq_obs_vs_ECMWF_lead_{leadtime}_combined.svg"
            )
        else:
            print(f"[WARN] No ML/OBS data after leadtime ≤ {leadtime} filter.")
        
        # --- Per-station plots (leadtime ≤ 48) ---
        qqplot_by_station(mos_sub, OBS, MOS, "MOS", prefix="MOS", season = "Full year", max_lt=leadtime)
        qqplot_by_station(ml_sub, OBS, MLCOL, ML_NAME, prefix="ML", season = "Full year", max_lt=leadtime)
        qqplot_by_station(ml_sub, OBS, RAW, "ECMWF", prefix="ECMWF", season = "Full year", max_lt=leadtime)


if __name__ == "__main__":
    main()
