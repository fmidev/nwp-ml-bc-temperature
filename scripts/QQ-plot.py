from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your ML model tag used when saving eval rows (=> column "corrected_<ML_TAG>")
ML_TAG = "bias_lstm_stream"
# Name of the model for the plot
ML_NAME = "EC_ML_LSTM"

# Plot settings
SHOW_PLOT = False
SAVE_PLOT = True
FIG_DPI   = 150

# Paths
HOME     = Path.home()
METRICS  = HOME / "thesis_project" / "metrics"
MOS_DIR  = METRICS / "mos"
ML_DIR   = METRICS / ML_TAG
OUT_DIR  = HOME / "thesis_project" / "figures" / "QQ-plots" / "FULL" / "all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns
SPLIT = "validtime"
KEYS  = ["SID", SPLIT, "analysistime", "leadtime"]
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS   = "corrected_mos"
MLCOL = f"corrected_{ML_TAG}"

SEASONAL = False

def save_figure_multi(outpath: Path, dpi: int = FIG_DPI):
    """
    Save current matplotlib figure as both SVG and PDF.
    If outpath already has a suffix, that suffix is ignored and both formats are written.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    stem = outpath.with_suffix("")
    plt.savefig(stem.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")
    plt.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    print(f"[INFO] Saved: {stem.with_suffix('.svg')}")
    print(f"[INFO] Saved: {stem.with_suffix('.pdf')}")

def _read_parquet_subset(f: Path, wanted_cols: list[str]) -> pl.DataFrame:
    """Read only existing columns from a parquet file."""
    schema = pl.scan_parquet(str(f)).collect_schema().names()
    existing = [c for c in wanted_cols if c in schema]
    return pl.read_parquet(f, columns=existing)


def _normalize_eval_df(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize key dtypes across MOS/ML files."""
    exprs = []

    if "SID" in df.columns:
        exprs.append(pl.col("SID").cast(pl.Utf8))

    if SPLIT in df.columns:
        exprs.append(
            pl.when(pl.col(SPLIT).is_not_null())
            .then(pl.col(SPLIT).cast(pl.Utf8).str.to_datetime(strict=False))
            .otherwise(None)
            .alias(SPLIT)
        )

    if "analysistime" in df.columns:
        exprs.append(
            pl.when(pl.col("analysistime").is_not_null())
            .then(pl.col("analysistime").cast(pl.Utf8).str.to_datetime(strict=False))
            .otherwise(None)
            .alias("analysistime")
        )

    if "leadtime" in df.columns:
        exprs.append(pl.col("leadtime").cast(pl.Int64))

    if exprs:
        df = df.with_columns(exprs)

    return df


def load_eval_rows_evaldir(eval_dir: Path, pattern: str, tag: str) -> pl.DataFrame:
    """
    Generic loader for MOS / regular validtime-split models.
    """
    files = sorted(eval_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")

    dfs = []
    wanted_cols = KEYS + [RAW, OBS, MOS, MLCOL]

    for f in files:
        df = _read_parquet_subset(f, wanted_cols)
        df = _normalize_eval_df(df)
        dfs.append(df)

    out = pl.concat(dfs, how="vertical_relaxed")
    print(f"[INFO] {tag} rows loaded: {out.height:,}")
    return out


def load_model_eval_rows(model_dir: Path, ml_tag: str) -> pl.DataFrame:
    """
    Load ML eval rows for either:
      1) LSTM-style analysistime-split parquet files with split_set column
      2) regular validtime-split parquet files

    Auto-detects layout by filename pattern.
    """
    model_col = f"corrected_{ml_tag}"

    lstm_pattern = f"eval_rows_analysistime_{ml_tag}_20??_fin.parquet"
    validtime_pattern = f"eval_rows_{SPLIT}_{ml_tag}_20*.parquet"

    lstm_files = sorted(model_dir.glob(lstm_pattern))
    validtime_files = sorted(model_dir.glob(validtime_pattern))

    if lstm_files:
        files = lstm_files
        is_lstm_layout = True
        print(f"[INFO] Detected LSTM analysistime-split layout for {ml_tag}")
    elif validtime_files:
        files = validtime_files
        is_lstm_layout = False
        print(f"[INFO] Detected validtime-split layout for {ml_tag}")
    else:
        raise FileNotFoundError(
            f"No files matched either:\n"
            f"  {model_dir}/{lstm_pattern}\n"
            f"  {model_dir}/{validtime_pattern}"
        )

    dfs = []
    wanted_cols = ["SID", "validtime", "analysistime", "leadtime", RAW, OBS, model_col, "split_set"]

    for f in files:
        df = _read_parquet_subset(f, wanted_cols)
        df = _normalize_eval_df(df)

        if is_lstm_layout and "split_set" in df.columns:
            df = df.filter(pl.col("split_set") == "test").drop("split_set")

        dfs.append(df)

    out = pl.concat(dfs, how="vertical_relaxed")
    print(f"[INFO] corrected_{ml_tag} rows loaded: {out.height:,}")
    return out


def filter_leadtime_leq(df: pl.DataFrame, max_lt: int = 48) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("leadtime").cast(pl.Int64))
          .filter(pl.col("leadtime") <= max_lt)
    )


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
        n = min(a.size, b.size, 5000)
        probs = (np.arange(1, n + 1) - 0.5) / n
    qa = np.quantile(a, probs, method="linear")
    qb = np.quantile(b, probs, method="linear")
    return qa, qb


def qqplot_arrays(x: np.ndarray, y: np.ndarray, title: str, outpath: Path | None = None):
    qa, qb = _qq_points(x, y)

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
        save_figure_multi(outpath, dpi=FIG_DPI)
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
                      model_name: str, prefix: str, season: str, max_lt: int = 48):
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
        if sub.height < 10:
            continue
        title = f"Q–Q: Observed vs {model_name}\n(SID={sid}, leadtime ≤ {max_lt}h {season})"
        fname = f"{prefix}_qq_obs_vs_{model_name}_SID_{sid}_lead_le{max_lt}_{season}.svg"
        qqplot_polars(sub, obs_col, pred_col, title, fname)


def mos_coverage_window(mos_eval: pl.DataFrame):
    usable = mos_eval.filter(
        pl.col(MOS).is_not_null() &
        pl.col(RAW).is_not_null() &
        pl.col(OBS).is_not_null()
    )
    if usable.height == 0:
        raise ValueError("MOS eval rows have no usable values.")

    return usable.select(
        pl.col(SPLIT).min().alias("t0"),
        pl.col(SPLIT).max().alias("t1"),
    ).row(0)


def filter_for_station_init(df: pl.DataFrame, sids: list[str], start_init, end_init, tag: str):
    """Filter dataframe to wanted stations and validtime window."""
    plot_df = df.filter(
        pl.col("SID").is_in([str(s) for s in sids]) &
        (pl.col(SPLIT) >= start_init) &
        (pl.col(SPLIT) <= end_init)
    )

    if plot_df.height == 0:
        raise ValueError(f"No {tag} rows for SID={sids}, window={start_init} -> {end_init}")

    return plot_df


# =========================
# Main
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
                mos_all = load_eval_rows_evaldir(
                    MOS_DIR,
                    f"eval_rows_{SPLIT}_MOS_{year}_{season}.parquet",
                    MOS
                )
                ml_all = load_model_eval_rows(ML_DIR, ML_TAG)

                t0, t1 = mos_coverage_window(mos_all)

                mos_sub = filter_for_station_init(mos_all, stations, t0, t1, MOS)
                ml_sub = filter_for_station_init(ml_all, stations, t0, t1, ML_NAME)

                mos_le = filter_leadtime_leq(mos_sub, leadtime)
                ml_le  = filter_leadtime_leq(ml_sub, leadtime)

                # Combined Q–Q plots
                if MOS in mos_le.columns and OBS in mos_le.columns and mos_le.height > 0:
                    qqplot_polars(
                        mos_le, OBS, MOS,
                        title=f"Q–Q: Observed vs MOS\n(leadtime ≤ {leadtime}h combined {season})",
                        fname=f"qq_obs_vs_MOS_lead_{leadtime}_combined_{season}.svg"
                    )
                else:
                    print(f"[WARN] No MOS/OBS data after leadtime ≤ {leadtime} filter.")

                if MLCOL in ml_le.columns and OBS in ml_le.columns and ml_le.height > 0:
                    qqplot_polars(
                        ml_le, OBS, MLCOL,
                        title=f"Q–Q: Observed vs {ML_NAME}\n(leadtime ≤ {leadtime}h combined {season})",
                        fname=f"qq_obs_vs_{ML_NAME}_lead_{leadtime}_combined_{season}.svg"
                    )
                else:
                    print(f"[WARN] No ML/OBS data after leadtime ≤ {leadtime} filter.")

                if RAW in ml_le.columns and OBS in ml_le.columns and ml_le.height > 0:
                    qqplot_polars(
                        ml_le, OBS, RAW,
                        title=f"Q–Q: Observed vs ECMWF\n(leadtime ≤ {leadtime}h combined {season})",
                        fname=f"qq_obs_vs_ECMWF_lead_{leadtime}_combined_{season}.svg"
                    )
                else:
                    print(f"[WARN] No RAW/OBS data after leadtime ≤ {leadtime} filter.")

                # Per-station plots
                qqplot_by_station(mos_sub, OBS, MOS, "MOS", prefix="MOS", season=season, max_lt=leadtime)
                qqplot_by_station(ml_sub, OBS, MLCOL, ML_NAME, prefix="ML", season=season, max_lt=leadtime)
                qqplot_by_station(ml_sub, OBS, RAW, "ECMWF", prefix="ECMWF", season=season, max_lt=leadtime)

    else:
        mos_all = load_eval_rows_evaldir(
            MOS_DIR,
            f"eval_rows_{SPLIT}_MOS_*.parquet",
            MOS
        )
        ml_all = load_model_eval_rows(ML_DIR, ML_TAG)

        t0, t1 = mos_coverage_window(mos_all)

        mos_sub = filter_for_station_init(mos_all, stations, t0, t1, MOS)
        ml_sub = filter_for_station_init(ml_all, stations, t0, t1, ML_NAME)

        mos_le = filter_leadtime_leq(mos_sub, leadtime)
        ml_le  = filter_leadtime_leq(ml_sub, leadtime)

        # Combined Q–Q plots
        if MOS in mos_le.columns and OBS in mos_le.columns and mos_le.height > 0:
            qqplot_polars(
                mos_le, OBS, MOS,
                title=f"Q–Q: Observed vs MOS\n(leadtime ≤ {leadtime}h combined)",
                fname=f"qq_obs_vs_MOS_lead_{leadtime}_combined.svg"
            )
        else:
            print(f"[WARN] No MOS/OBS data after leadtime ≤ {leadtime} filter.")

        if MLCOL in ml_le.columns and OBS in ml_le.columns and ml_le.height > 0:
            qqplot_polars(
                ml_le, OBS, MLCOL,
                title=f"Q–Q: Observed vs {ML_NAME}\n(leadtime ≤ {leadtime}h combined)",
                fname=f"qq_obs_vs_{ML_NAME}_lead_{leadtime}_combined.svg"
            )
        else:
            print(f"[WARN] No ML/OBS data after leadtime ≤ {leadtime} filter.")

        if RAW in ml_le.columns and OBS in ml_le.columns and ml_le.height > 0:
            qqplot_polars(
                ml_le, OBS, RAW,
                title=f"Q–Q: Observed vs ECMWF\n(leadtime ≤ {leadtime}h combined)",
                fname=f"qq_obs_vs_ECMWF_lead_{leadtime}_combined.svg"
            )
        else:
            print(f"[WARN] No RAW/OBS data after leadtime ≤ {leadtime} filter.")

        # Per-station plots
        qqplot_by_station(mos_sub, OBS, MOS, "MOS", prefix="MOS", season="Full year", max_lt=leadtime)
        qqplot_by_station(ml_sub, OBS, MLCOL, ML_NAME, prefix="ML", season="Full year", max_lt=leadtime)
        qqplot_by_station(ml_sub, OBS, RAW, "ECMWF", prefix="ECMWF", season="Full year", max_lt=leadtime)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()