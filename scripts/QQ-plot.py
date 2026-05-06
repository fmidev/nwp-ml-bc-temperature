from pathlib import Path
import argparse

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Columns
# =========================

SPLIT = "validtime"
KEYS = ["SID", SPLIT, "analysistime", "leadtime"]

OBS = "obs_TA"
RAW = "raw_fc"
MOS = "corrected_mos"


# =========================
# Argument parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Q-Q plots comparing observations, MOS, raw forecast, and an ML model."
    )

    parser.add_argument(
        "--mos-dir",
        required=True,
        type=str,
        help="Directory containing MOS evaluation parquet files.",
    )

    parser.add_argument(
        "--ml-dir",
        required=True,
        type=str,
        help="Directory containing ML evaluation parquet files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where Q-Q plots will be saved.",
    )

    parser.add_argument(
        "--ml-tag",
        default="bias_lstm_stream",
        type=str,
        help=(
            "ML model tag used in filenames and corrected_<ml-tag> column. "
            "Default: bias_lstm_stream."
        ),
    )

    parser.add_argument(
        "--ml-name",
        default="EC_ML_LSTM",
        type=str,
        help="Readable model name used in plot titles and output filenames.",
    )

    parser.add_argument(
        "--leadtime",
        default=240,
        type=int,
        help="Maximum leadtime to include in Q-Q plots. Default: 240.",
    )

    parser.add_argument(
        "--seasonal",
        action="store_true",
        help="Create separate plots for each configured season.",
    )

    parser.add_argument(
        "--non-seasonal",
        action="store_true",
        help="Create combined full-period plots.",
    )

    parser.add_argument(
        "--stations-file",
        default=None,
        type=str,
        help=(
            "Optional text file containing station IDs to include, one per line. "
            "If omitted, all stations in the aligned data are used."
        ),
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show plots interactively.",
    )

    parser.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Do not save plots to disk.",
    )

    parser.add_argument(
        "--fig-dpi",
        default=150,
        type=int,
        help="Figure DPI. Default: 150.",
    )

    return parser.parse_args()


def load_station_subset(stations_file: str | None) -> list[str] | None:
    """
    Load optional station subset.

    Returns:
        None if no station file is given, meaning use all stations.
        Otherwise returns a list of station IDs as strings.
    """
    if stations_file is None:
        return None

    path = Path(stations_file)

    if not path.exists():
        raise FileNotFoundError(f"Stations file not found: {path}")

    with open(path, "r") as f:
        stations = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not stations:
        raise ValueError(f"No station IDs found in station file: {path}")

    return stations


# =========================
# Data loading helpers
# =========================

def save_figure_multi(outpath: Path, dpi: int):
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

    if not existing:
        return pl.DataFrame()

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
        exprs.append(pl.col("leadtime").cast(pl.Int64, strict=False))

    if exprs:
        df = df.with_columns(exprs)

    return df


def load_eval_rows_evaldir(
    eval_dir: Path,
    pattern: str,
    tag: str,
    ml_col: str,
) -> pl.DataFrame:
    """
    Generic loader for MOS / regular validtime-split model files.
    """
    files = sorted(eval_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")

    dfs = []
    wanted_cols = KEYS + [RAW, OBS, MOS, ml_col]

    for f in files:
        df = _read_parquet_subset(f, wanted_cols)

        if df.height == 0:
            continue

        df = _normalize_eval_df(df)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No usable columns found in files matching: {eval_dir}/{pattern}")

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
    wanted_cols = [
        "SID",
        "validtime",
        "analysistime",
        "leadtime",
        RAW,
        OBS,
        model_col,
        "split_set",
    ]

    for f in files:
        df = _read_parquet_subset(f, wanted_cols)

        if df.height == 0:
            continue

        df = _normalize_eval_df(df)

        if is_lstm_layout and "split_set" in df.columns:
            df = df.filter(pl.col("split_set") == "test").drop("split_set")

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No usable ML eval rows found in: {model_dir}")

    out = pl.concat(dfs, how="vertical_relaxed")
    print(f"[INFO] corrected_{ml_tag} rows loaded: {out.height:,}")

    return out


def filter_leadtime_leq(df: pl.DataFrame, max_lt: int) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("leadtime").cast(pl.Int64, strict=False))
        .filter(pl.col("leadtime") <= max_lt)
    )


# =========================
# Q-Q plot helpers
# =========================

def _qq_points(
    a: np.ndarray,
    b: np.ndarray,
    probs: np.ndarray | None = None,
):
    """Return matched quantiles for arrays a and b on a shared probability grid."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if a.size == 0 or b.size == 0:
        raise ValueError("Empty array after NaN removal; cannot make Q-Q plot.")

    if probs is None:
        n = min(a.size, b.size, 5000)
        probs = (np.arange(1, n + 1) - 0.5) / n

    qa = np.quantile(a, probs, method="linear")
    qb = np.quantile(b, probs, method="linear")

    return qa, qb


def qqplot_arrays(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    outpath: Path | None,
    fig_dpi: int,
    save_plot: bool,
    show_plot: bool,
):
    qa, qb = _qq_points(x, y)

    lo = min(qa.min(), qb.min())
    hi = max(qa.max(), qb.max())

    plt.figure(dpi=fig_dpi)
    plt.scatter(
        qa,
        qb,
        s=10,
        alpha=0.8,
        color="#8080ff",
        edgecolors="none",
    )
    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color="#000d1a",
    )

    plt.xlabel("Observed quantiles")
    plt.ylabel("Predicted quantiles")
    plt.title(title)
    plt.tight_layout()

    if save_plot and outpath is not None:
        save_figure_multi(outpath, dpi=fig_dpi)

    if show_plot:
        plt.show()

    plt.close()


def qqplot_polars(
    df: pl.DataFrame,
    obs_col: str,
    pred_col: str,
    title: str,
    fname: str,
    out_dir: Path,
    fig_dpi: int,
    save_plot: bool,
    show_plot: bool,
):
    obs = df.select(pl.col(obs_col).cast(pl.Float64)).to_numpy().ravel()
    pred = df.select(pl.col(pred_col).cast(pl.Float64)).to_numpy().ravel()

    qqplot_arrays(
        obs,
        pred,
        title=title,
        outpath=out_dir / fname if save_plot else None,
        fig_dpi=fig_dpi,
        save_plot=save_plot,
        show_plot=show_plot,
    )


def qqplot_by_station(
    df: pl.DataFrame,
    obs_col: str,
    pred_col: str,
    model_name: str,
    prefix: str,
    season: str,
    max_lt: int,
    out_dir: Path,
    fig_dpi: int,
    save_plot: bool,
    show_plot: bool,
):
    """Generate and save one Q-Q plot per station for leadtime <= max_lt."""
    df_filt = (
        df.with_columns(pl.col("leadtime").cast(pl.Int64, strict=False))
        .filter(pl.col("leadtime") <= max_lt)
        .filter(pl.col(obs_col).is_not_null() & pl.col(pred_col).is_not_null())
    )

    sids = df_filt.select("SID").unique().to_series().to_list()

    print(
        f"[INFO] Generating per-station Q-Q plots for "
        f"{len(sids)} stations, leadtime <= {max_lt}"
    )

    safe_season = season.replace(" ", "_")

    for sid in sids:
        sub = df_filt.filter(pl.col("SID") == sid)

        if sub.height < 10:
            continue

        title = (
            f"Q-Q: Observed vs {model_name}\n"
            f"SID={sid}, leadtime <= {max_lt}h, {season}"
        )

        fname = (
            f"{prefix}_qq_obs_vs_{model_name}_SID_{sid}_"
            f"lead_le{max_lt}_{safe_season}.svg"
        )

        qqplot_polars(
            sub,
            obs_col,
            pred_col,
            title,
            fname,
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            save_plot=save_plot,
            show_plot=show_plot,
        )


def mos_coverage_window(mos_eval: pl.DataFrame):
    usable = mos_eval.filter(
        pl.col(MOS).is_not_null()
        & pl.col(RAW).is_not_null()
        & pl.col(OBS).is_not_null()
    )

    if usable.height == 0:
        raise ValueError("MOS eval rows have no usable values.")

    return usable.select(
        pl.col(SPLIT).min().alias("t0"),
        pl.col(SPLIT).max().alias("t1"),
    ).row(0)


def filter_for_station_init(
    df: pl.DataFrame,
    station_subset: list[str] | None,
    start_init,
    end_init,
    tag: str,
):
    """Filter dataframe to optional station subset and validtime window."""
    filter_expr = (pl.col(SPLIT) >= start_init) & (pl.col(SPLIT) <= end_init)

    if station_subset is not None:
        filter_expr = filter_expr & pl.col("SID").is_in([str(s) for s in station_subset])

    plot_df = df.filter(filter_expr)

    if plot_df.height == 0:
        station_msg = (
            "all stations"
            if station_subset is None
            else f"station subset of {len(station_subset)} stations"
        )

        raise ValueError(
            f"No {tag} rows for {station_msg}, "
            f"window={start_init} -> {end_init}"
        )

    return plot_df


def make_combined_plots(
    mos_le: pl.DataFrame,
    ml_le: pl.DataFrame,
    ml_col: str,
    ml_name: str,
    leadtime: int,
    season_label: str,
    fname_suffix: str,
    out_dir: Path,
    fig_dpi: int,
    save_plot: bool,
    show_plot: bool,
):
    """Create combined MOS, ML, and raw forecast Q-Q plots."""
    if MOS in mos_le.columns and OBS in mos_le.columns and mos_le.height > 0:
        qqplot_polars(
            mos_le,
            OBS,
            MOS,
            title=f"Q-Q: Observed vs MOS\nleadtime <= {leadtime}h, {season_label}",
            fname=f"qq_obs_vs_MOS_lead_{leadtime}_combined{fname_suffix}.svg",
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            save_plot=save_plot,
            show_plot=show_plot,
        )
    else:
        print(f"[WARN] No MOS/OBS data after leadtime <= {leadtime} filter.")

    if ml_col in ml_le.columns and OBS in ml_le.columns and ml_le.height > 0:
        qqplot_polars(
            ml_le,
            OBS,
            ml_col,
            title=f"Q-Q: Observed vs {ml_name}\nleadtime <= {leadtime}h, {season_label}",
            fname=f"qq_obs_vs_{ml_name}_lead_{leadtime}_combined{fname_suffix}.svg",
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            save_plot=save_plot,
            show_plot=show_plot,
        )
    else:
        print(f"[WARN] No ML/OBS data after leadtime <= {leadtime} filter.")

    if RAW in ml_le.columns and OBS in ml_le.columns and ml_le.height > 0:
        qqplot_polars(
            ml_le,
            OBS,
            RAW,
            title=f"Q-Q: Observed vs ECMWF\nleadtime <= {leadtime}h, {season_label}",
            fname=f"qq_obs_vs_ECMWF_lead_{leadtime}_combined{fname_suffix}.svg",
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            save_plot=save_plot,
            show_plot=show_plot,
        )
    else:
        print(f"[WARN] No RAW/OBS data after leadtime <= {leadtime} filter.")


# =========================
# Main
# =========================

def main():
    args = parse_args()

    if args.seasonal and args.non_seasonal:
        raise ValueError("Use either --seasonal or --non-seasonal, not both.")

    # Default to non-seasonal, matching your original SEASONAL = False.
    seasonal = True if args.seasonal else False

    mos_dir = Path(args.mos_dir)
    ml_dir = Path(args.ml_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ml_tag = args.ml_tag
    ml_name = args.ml_name
    ml_col = f"corrected_{ml_tag}"

    leadtime = args.leadtime
    fig_dpi = args.fig_dpi
    show_plot = args.show_plot
    save_plot = not args.no_save_plot

    station_subset = load_station_subset(args.stations_file)

    if station_subset is None:
        print("[INFO] Using all stations in the aligned data.")
    else:
        print(f"[INFO] Using station subset from {args.stations_file}: {len(station_subset)} stations")

    print(f"[INFO] MOS directory: {mos_dir}")
    print(f"[INFO] ML directory: {ml_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] ML tag: {ml_tag}")
    print(f"[INFO] ML column: {ml_col}")
    print(f"[INFO] ML name: {ml_name}")
    print(f"[INFO] Leadtime <= {leadtime}")
    print(f"[INFO] Seasonal mode: {seasonal}")

    ml_all = load_model_eval_rows(ml_dir, ml_tag)

    if seasonal:
        available = {
            "2024": ["autumn"],
            "2025": ["winter", "spring", "summer"],
        }

        for year, seasons in available.items():
            for season in seasons:
                print(f"\n[INFO] Processing {year} {season}")

                mos_all = load_eval_rows_evaldir(
                    mos_dir,
                    f"eval_rows_{SPLIT}_MOS_{year}_{season}.parquet",
                    MOS,
                    ml_col=ml_col,
                )

                t0, t1 = mos_coverage_window(mos_all)

                mos_sub = filter_for_station_init(
                    mos_all,
                    station_subset,
                    t0,
                    t1,
                    MOS,
                )

                ml_sub = filter_for_station_init(
                    ml_all,
                    station_subset,
                    t0,
                    t1,
                    ml_name,
                )

                print(f"[INFO] MOS window: {t0} -> {t1}")
                print(f"[INFO] MOS rows after station/window filter: {mos_sub.height:,}")
                print(f"[INFO] ML rows after station/window filter: {ml_sub.height:,}")
                print(f"[INFO] MOS stations: {mos_sub.select('SID').n_unique()}")
                print(f"[INFO] ML stations: {ml_sub.select('SID').n_unique()}")

                mos_le = filter_leadtime_leq(mos_sub, leadtime)
                ml_le = filter_leadtime_leq(ml_sub, leadtime)

                suffix = f"_{season}"

                make_combined_plots(
                    mos_le,
                    ml_le,
                    ml_col=ml_col,
                    ml_name=ml_name,
                    leadtime=leadtime,
                    season_label=season,
                    fname_suffix=suffix,
                    out_dir=out_dir,
                    fig_dpi=fig_dpi,
                    save_plot=save_plot,
                    show_plot=show_plot,
                )

                qqplot_by_station(
                    mos_sub,
                    OBS,
                    MOS,
                    "MOS",
                    prefix="MOS",
                    season=season,
                    max_lt=leadtime,
                    out_dir=out_dir,
                    fig_dpi=fig_dpi,
                    save_plot=save_plot,
                    show_plot=show_plot,
                )

                qqplot_by_station(
                    ml_sub,
                    OBS,
                    ml_col,
                    ml_name,
                    prefix="ML",
                    season=season,
                    max_lt=leadtime,
                    out_dir=out_dir,
                    fig_dpi=fig_dpi,
                    save_plot=save_plot,
                    show_plot=show_plot,
                )

                qqplot_by_station(
                    ml_sub,
                    OBS,
                    RAW,
                    "ECMWF",
                    prefix="ECMWF",
                    season=season,
                    max_lt=leadtime,
                    out_dir=out_dir,
                    fig_dpi=fig_dpi,
                    save_plot=save_plot,
                    show_plot=show_plot,
                )

    else:
        print("\n[INFO] Processing non-seasonal/full-period Q-Q plots")

        mos_all = load_eval_rows_evaldir(
            mos_dir,
            f"eval_rows_{SPLIT}_MOS_*.parquet",
            MOS,
            ml_col=ml_col,
        )

        t0, t1 = mos_coverage_window(mos_all)

        mos_sub = filter_for_station_init(
            mos_all,
            station_subset,
            t0,
            t1,
            MOS,
        )

        ml_sub = filter_for_station_init(
            ml_all,
            station_subset,
            t0,
            t1,
            ml_name,
        )

        print(f"[INFO] MOS window: {t0} -> {t1}")
        print(f"[INFO] MOS rows after station/window filter: {mos_sub.height:,}")
        print(f"[INFO] ML rows after station/window filter: {ml_sub.height:,}")
        print(f"[INFO] MOS stations: {mos_sub.select('SID').n_unique()}")
        print(f"[INFO] ML stations: {ml_sub.select('SID').n_unique()}")

        mos_le = filter_leadtime_leq(mos_sub, leadtime)
        ml_le = filter_leadtime_leq(ml_sub, leadtime)

        make_combined_plots(
            mos_le,
            ml_le,
            ml_col=ml_col,
            ml_name=ml_name,
            leadtime=leadtime,
            season_label="full period",
            fname_suffix="",
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            save_plot=save_plot,
            show_plot=show_plot,
        )

        qqplot_by_station(
            mos_sub,
            OBS,
            MOS,
            "MOS",
            prefix="MOS",
            season="Full_year",
            max_lt=leadtime,
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            save_plot=save_plot,
            show_plot=show_plot,
        )

        qqplot_by_station(
            ml_sub,
            OBS,
            ml_col,
            ml_name,
            prefix="ML",
            season="Full_year",
            max_lt=leadtime,
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            save_plot=save_plot,
            show_plot=show_plot,
        )

        qqplot_by_station(
            ml_sub,
            OBS,
            RAW,
            "ECMWF",
            prefix="ECMWF",
            season="Full_year",
            max_lt=leadtime,
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            save_plot=save_plot,
            show_plot=show_plot,
        )


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()