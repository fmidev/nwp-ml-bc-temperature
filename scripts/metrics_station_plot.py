import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


# =====================
# Default/global settings
# These are overwritten from command-line args in main()
# =====================

SPLIT = "validtime"
OBS = "obs_TA"
RAW = "raw_fc"
MOS_CORR = "corrected_mos"

RAW_NAME = "ECMWF"

ML1_TAG = None
ML2_TAG = None
ML1_NAME = None
ML2_NAME = None
ML1_CORR = None
ML2_CORR = None

COMPARE_MODE = None

STATIONS_CSV = None
WORLD_SHP = None

LON_MIN, LAT_MIN = -25.0, 25.5
LON_MAX, LAT_MAX = 42.0, 72.0

DIV_CMAP = "PuOr"
SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill_cmp",
    ["#570040", "white", "#005717"],
)
BIAS_CMAP = "coolwarm"
RMSE_CMAP = "hot_r"


# =====================
# Argument parsing
# =====================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create station-level evaluation maps and plots comparing MOS, raw forecast, "
            "and one or two ML models."
        )
    )

    parser.add_argument(
        "--mos-dir",
        required=True,
        type=str,
        help="Directory containing MOS evaluation parquet files.",
    )

    parser.add_argument(
        "--ml1-dir",
        required=True,
        type=str,
        help="Directory containing ML1 evaluation parquet files.",
    )

    parser.add_argument(
        "--ml2-dir",
        default=None,
        type=str,
        help="Directory containing ML2 evaluation parquet files. Required for --compare-mode pair.",
    )

    parser.add_argument(
        "--stations-csv",
        required=True,
        type=str,
        help="Path to stations.csv. Must contain SID, lon, lat columns.",
    )

    parser.add_argument(
        "--world-shp",
        required=True,
        type=str,
        help="Path to Natural Earth world shapefile, e.g. ne_110m_admin_0_countries.shp.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Base output directory where figures will be saved.",
    )

    parser.add_argument(
        "--ml1-tag",
        default="bias_lstm_stream",
        type=str,
        help="ML1 tag used in filenames and corrected_<tag> column.",
    )

    parser.add_argument(
        "--ml2-tag",
        default="tuned_ah_2019",
        type=str,
        help="ML2 tag used in filenames and corrected_<tag> column.",
    )

    parser.add_argument(
        "--ml1-name",
        default="EC_ML_LSTM",
        type=str,
        help="Readable name for ML1 used in titles and filenames.",
    )

    parser.add_argument(
        "--ml2-name",
        default="EC_ML_XGB",
        type=str,
        help="Readable name for ML2 used in titles and filenames.",
    )

    parser.add_argument(
        "--raw-name",
        default="ECMWF",
        type=str,
        help="Readable name for the raw forecast. Default: ECMWF.",
    )

    parser.add_argument(
        "--compare-mode",
        choices=["single", "pair"],
        default="single",
        help=(
            "single = compare ML1 vs MOS and raw. "
            "pair = compare ML1 and ML2, choose best ML per station, then compare to MOS/raw."
        ),
    )

    parser.add_argument(
        "--seasonal",
        action="store_true",
        help="Run seasonal evaluation using seasonal MOS files.",
    )

    parser.add_argument(
        "--non-seasonal",
        action="store_true",
        help="Run one combined/non-seasonal evaluation.",
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["autumn", "winter", "spring", "summer"],
        help="Seasons to process in seasonal mode. Default: autumn winter spring summer.",
    )

    parser.add_argument(
        "--season-years",
        nargs="+",
        default=["2024", "2025"],
        help="Years to search for seasonal MOS files. Default: 2024 2025.",
    )

    parser.add_argument(
        "--leadtime-filter",
        action="store_true",
        help="Filter all rows to leadtime <= --max-leadtime.",
    )

    parser.add_argument(
        "--max-leadtime",
        default=48,
        type=int,
        help="Maximum leadtime when --leadtime-filter is used. Default: 48.",
    )

    parser.add_argument(
        "--threads",
        default="16",
        type=str,
        help="Thread count for OMP_NUM_THREADS and MKL_NUM_THREADS. Default: 16.",
    )

    parser.add_argument(
        "--lon-min",
        default=-25.0,
        type=float,
        help="Minimum longitude for map extent.",
    )

    parser.add_argument(
        "--lon-max",
        default=42.0,
        type=float,
        help="Maximum longitude for map extent.",
    )

    parser.add_argument(
        "--lat-min",
        default=25.5,
        type=float,
        help="Minimum latitude for map extent.",
    )

    parser.add_argument(
        "--lat-max",
        default=72.0,
        type=float,
        help="Maximum latitude for map extent.",
    )

    return parser.parse_args()


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


# =====================
# Helper functions
# =====================

def rmse(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    return np.sqrt(np.mean(a * a)) if a.size else np.nan


def mae(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    return np.mean(np.abs(a)) if a.size else np.nan


def wmean(series: pd.Series, weights: pd.Series | None) -> float:
    s = pd.to_numeric(series, errors="coerce")

    if weights is None:
        return float(np.nanmean(s))

    w = pd.to_numeric(weights, errors="coerce")
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)

    return float(np.average(s[mask], weights=w[mask])) if mask.any() else float("nan")


def shared_diverging_norm(series_list, q=0.95, vcenter=0.0):
    vals = pd.concat(
        [pd.to_numeric(s, errors="coerce") for s in series_list],
        ignore_index=True,
    )

    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

    if vals.empty:
        span = 1.0
    else:
        span = float(np.nanquantile(np.abs(vals), q))

        if not np.isfinite(span) or span == 0:
            span = float(np.nanmax(np.abs(vals))) or 1.0

    return TwoSlopeNorm(vmin=-span, vcenter=vcenter, vmax=+span)


def shared_sequential_limits(series_list, lo=5, hi=95):
    vals = pd.concat(
        [pd.to_numeric(s, errors="coerce") for s in series_list],
        ignore_index=True,
    )

    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

    if vals.empty:
        return None, None

    return float(np.nanpercentile(vals, lo)), float(np.nanpercentile(vals, hi))


def auto_diverging_norm(
    series: pd.Series,
    q: float = 0.95,
    vcenter: float = 0.0,
    min_span: float | None = None,
    max_span: float | None = None,
) -> TwoSlopeNorm:
    vals = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if vals.empty:
        span = 1.0
    else:
        span = float(np.nanquantile(np.abs(vals), q))

        if not np.isfinite(span) or span == 0:
            span = float(np.nanmax(np.abs(vals))) or 1.0

    if min_span is not None:
        span = max(span, float(min_span))

    if max_span is not None:
        span = min(span, float(max_span))

    return TwoSlopeNorm(vmin=-span, vcenter=vcenter, vmax=+span)


def robust_symmetric_norm(series: pd.Series, q=0.95):
    vals = (
        pd.to_numeric(series, errors="coerce")
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    span = float(np.nanquantile(vals, q)) if len(vals) else 1.0

    return TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)


def attach_stations_gdf(station_metrics: pd.DataFrame) -> gpd.GeoDataFrame:
    st = pd.read_csv(STATIONS_CSV)
    st["SID"] = st["SID"].astype(str)

    required_cols = {"SID", "lon", "lat"}
    missing = required_cols - set(st.columns)

    if missing:
        raise ValueError(f"Stations CSV is missing required columns: {sorted(missing)}")

    df = st.merge(station_metrics, on="SID", how="left")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )


# =====================
# Data loading / alignment
# =====================

def _read_parquet_subset(file_path: Path, cols: list[str]) -> pd.DataFrame:
    preview = pd.read_parquet(file_path)
    existing = [c for c in cols if c in preview.columns]

    if not existing:
        return pd.DataFrame()

    return pd.read_parquet(file_path, columns=existing)


def load_eval_rows_evaldir(eval_dir: Path, pattern: str, cols: list[str]) -> pd.DataFrame:
    files = sorted(eval_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")

    dfs = []

    for f in files:
        df = _read_parquet_subset(f, cols)

        if df.empty:
            continue

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No usable columns found in files matching: {eval_dir}/{pattern}")

    out = pd.concat(dfs, ignore_index=True)

    if "SID" in out.columns:
        out["SID"] = out["SID"].astype(str)

    if SPLIT in out.columns:
        out[SPLIT] = pd.to_datetime(out[SPLIT], errors="coerce").dt.tz_localize(None)

    if "analysistime" in out.columns:
        out["analysistime"] = pd.to_datetime(out["analysistime"], errors="coerce").dt.tz_localize(None)

    if "validtime" in out.columns:
        out["validtime"] = pd.to_datetime(out["validtime"], errors="coerce").dt.tz_localize(None)

    if "leadtime" in out.columns:
        out["leadtime"] = pd.to_numeric(out["leadtime"], errors="coerce")

    return out


def load_model_eval_rows(model_dir: Path, ml_tag: str, ml_corr: str, ml_name: str) -> pd.DataFrame:
    """
    Load ML eval rows for either:
      1) LSTM-style analysistime-split parquet files with split_set column
      2) regular validtime-split parquet files

    Auto-detects layout by filename pattern.
    """
    lstm_pattern = f"eval_rows_analysistime_{ml_tag}_20??_fin.parquet"
    validtime_pattern = f"eval_rows_{SPLIT}_{ml_tag}_20*.parquet"

    lstm_files = sorted(model_dir.glob(lstm_pattern))
    validtime_files = sorted(model_dir.glob(validtime_pattern))

    if lstm_files:
        pattern = lstm_pattern
        cols = ["SID", "validtime", "analysistime", "leadtime", RAW, OBS, ml_corr, "split_set"]
        print(f"[INFO] Detected LSTM analysistime-split layout for {ml_name}")
    elif validtime_files:
        pattern = validtime_pattern
        cols = ["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, ml_corr]
        print(f"[INFO] Detected validtime-split layout for {ml_name}")
    else:
        raise FileNotFoundError(
            f"No files matched either:\n"
            f"  {model_dir}/{lstm_pattern}\n"
            f"  {model_dir}/{validtime_pattern}"
        )

    df = load_eval_rows_evaldir(model_dir, pattern=pattern, cols=cols)

    if "split_set" in df.columns:
        df = df[df["split_set"] == "test"].copy()
        df = df.drop(columns=["split_set"], errors="ignore")

    print(f"[INFO] {ml_name} rows loaded: {len(df):,}")

    return df


def mos_coverage_window(mos_eval: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = mos_eval.dropna(subset=[MOS_CORR, RAW, OBS]).copy()

    if df.empty:
        raise ValueError("MOS eval rows have no usable values.")

    return df[SPLIT].min(), df[SPLIT].max()


def align_mos_window_join_single(
    mos_eval: pd.DataFrame,
    ml1_eval: pd.DataFrame,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> pd.DataFrame:
    mos = mos_eval[(mos_eval[SPLIT] >= t0) & (mos_eval[SPLIT] <= t1)].copy()

    keys = ["SID", SPLIT, "analysistime", "leadtime"]
    keep_cols = keys + [RAW, OBS, MOS_CORR, ML1_CORR]

    joined = (
        mos.merge(ml1_eval[keys + [ML1_CORR]], on=keys, how="inner")
        .dropna(subset=[OBS, RAW, MOS_CORR, ML1_CORR])[keep_cols]
        .copy()
    )

    if not joined.empty:
        print(f"MOS-driven window single-ML: {joined[SPLIT].min()} -> {joined[SPLIT].max()}")
    else:
        print("Warning: No aligned rows after MOS-driven single-ML join.")

    return joined


def align_mos_window_join(
    mos_eval: pd.DataFrame,
    ml1_eval: pd.DataFrame,
    ml2_eval: pd.DataFrame,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> pd.DataFrame:
    mos = mos_eval[(mos_eval[SPLIT] >= t0) & (mos_eval[SPLIT] <= t1)].copy()

    keys = ["SID", SPLIT, "analysistime", "leadtime"]
    keep_cols = keys + [RAW, OBS, MOS_CORR, ML1_CORR, ML2_CORR]

    joined = (
        mos.merge(ml1_eval[keys + [ML1_CORR]], on=keys, how="inner")
        .merge(ml2_eval[keys + [ML2_CORR]], on=keys, how="inner")
        .dropna(subset=[OBS, RAW, MOS_CORR, ML1_CORR, ML2_CORR])[keep_cols]
        .copy()
    )

    if not joined.empty:
        print(f"MOS-driven window pair: {joined[SPLIT].min()} -> {joined[SPLIT].max()}")
    else:
        print("Warning: No aligned rows after MOS-driven pair join.")

    return joined


# =====================
# Metrics
# =====================

def compute_station_metrics_single_ml_vs_mos(joined: pd.DataFrame) -> pd.DataFrame:
    grp = joined.groupby("SID", as_index=False)
    eps = 1e-12

    st = (
        grp.apply(
            lambda g: pd.Series({
                "rmse_raw": rmse(g[OBS] - g[RAW]),
                "rmse_mos": rmse(g[OBS] - g[MOS_CORR]),
                f"rmse_{ML1_TAG}": rmse(g[OBS] - g[ML1_CORR]),

                "mae_raw": mae(g[OBS] - g[RAW]),
                "mae_mos": mae(g[OBS] - g[MOS_CORR]),
                f"mae_{ML1_TAG}": mae(g[OBS] - g[ML1_CORR]),

                "bias_raw": np.nanmean((g[RAW] - g[OBS]).astype(float)),
                "bias_mos": np.nanmean((g[MOS_CORR] - g[OBS]).astype(float)),
                f"bias_{ML1_TAG}": np.nanmean((g[ML1_CORR] - g[OBS]).astype(float)),

                "n": int(len(g)),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    st["skill_mos_pct"] = 100.0 * (1.0 - st["mae_mos"] / (st["mae_raw"] + eps))
    st[f"skill_{ML1_TAG}_pct"] = 100.0 * (
        1.0 - st[f"mae_{ML1_TAG}"] / (st["mae_raw"] + eps)
    )

    st["delta_skill_ml1_minus_mos_pct"] = (
        st[f"skill_{ML1_TAG}_pct"] - st["skill_mos_pct"]
    )

    st["delta_rmse_mos_minus_ml1"] = st["rmse_mos"] - st[f"rmse_{ML1_TAG}"]
    st["delta_rmse_ml1_minus_raw"] = st[f"rmse_{ML1_TAG}"] - st["rmse_raw"]
    st["delta_rmse_mos_minus_raw"] = st["rmse_mos"] - st["rmse_raw"]

    return st


def compute_station_metrics_two_ml(joined: pd.DataFrame) -> pd.DataFrame:
    grp = joined.groupby("SID", as_index=False)
    eps = 1e-12

    station = (
        grp.apply(
            lambda g: pd.Series({
                "rmse_raw": rmse(g[OBS] - g[RAW]),
                f"rmse_{ML1_TAG}": rmse(g[OBS] - g[ML1_CORR]),
                f"rmse_{ML2_TAG}": rmse(g[OBS] - g[ML2_CORR]),

                "mae_raw": mae(g[OBS] - g[RAW]),
                f"mae_{ML1_TAG}": mae(g[OBS] - g[ML1_CORR]),
                f"mae_{ML2_TAG}": mae(g[OBS] - g[ML2_CORR]),

                f"bias_{ML1_TAG}": np.nanmean((g[ML1_CORR] - g[OBS]).astype(float)),
                f"bias_{ML2_TAG}": np.nanmean((g[ML2_CORR] - g[OBS]).astype(float)),

                "n": int(len(g)),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    station[f"skill_{ML1_TAG}_pct"] = 100.0 * (
        1.0 - station[f"mae_{ML1_TAG}"] / (station["mae_raw"] + eps)
    )

    station[f"skill_{ML2_TAG}_pct"] = 100.0 * (
        1.0 - station[f"mae_{ML2_TAG}"] / (station["mae_raw"] + eps)
    )

    station["delta_skill_ML2_minus_ML1_pct"] = (
        station[f"skill_{ML2_TAG}_pct"] - station[f"skill_{ML1_TAG}_pct"]
    )

    station["delta_rmse_ML1_minus_ML2"] = (
        station[f"rmse_{ML1_TAG}"] - station[f"rmse_{ML2_TAG}"]
    )

    station["best_ml_tag"] = np.where(
        station[f"rmse_{ML2_TAG}"] < station[f"rmse_{ML1_TAG}"],
        ML2_TAG,
        ML1_TAG,
    )

    station["rmse_best_ml"] = station[
        [f"rmse_{ML1_TAG}", f"rmse_{ML2_TAG}"]
    ].min(axis=1)

    station["mae_best_ml"] = station[
        [f"mae_{ML1_TAG}", f"mae_{ML2_TAG}"]
    ].min(axis=1)

    station["bias_best_ml"] = np.where(
        station["best_ml_tag"] == ML2_TAG,
        station[f"bias_{ML2_TAG}"],
        station[f"bias_{ML1_TAG}"],
    )

    return station


def compute_station_metrics_bestml_vs_mos(
    joined: pd.DataFrame,
    station_two_ml: pd.DataFrame,
) -> pd.DataFrame:
    grp = joined.groupby("SID", as_index=False)

    mos_stats = (
        grp.apply(
            lambda g: pd.Series({
                "rmse_mos": rmse(g[OBS] - g[MOS_CORR]),
                "mae_mos": mae(g[OBS] - g[MOS_CORR]),

                "rmse_raw": rmse(g[OBS] - g[RAW]),
                "mae_raw": mae(g[OBS] - g[RAW]),

                "bias_mos": np.nanmean((g[MOS_CORR] - g[OBS]).astype(float)),
                "bias_raw": np.nanmean((g[RAW] - g[OBS]).astype(float)),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    st = station_two_ml.merge(mos_stats, on="SID", how="left", suffixes=("_two", ""))

    eps = 1e-12

    st["skill_mos_pct"] = 100.0 * (1.0 - st["mae_mos"] / (st["mae_raw"] + eps))
    st["skill_best_ml_pct"] = 100.0 * (
        1.0 - st["mae_best_ml"] / (st["mae_raw"] + eps)
    )

    st["delta_rmse_mos_minus_bestml"] = st["rmse_mos"] - st["rmse_best_ml"]
    st["delta_rmse_bestml_minus_raw"] = st["rmse_best_ml"] - st["rmse_raw"]
    st["delta_rmse_mos_minus_raw"] = st["rmse_mos"] - st["rmse_raw"]

    st["delta_skill_bestml_minus_mos_pct"] = (
        st["skill_best_ml_pct"] - st["skill_mos_pct"]
    )

    return st


# =====================
# Plotting
# =====================

def _world_gdf():
    return gpd.read_file(WORLD_SHP)


def save_svg_pdf(out_dir: Path, stem: str):
    svg = out_dir / f"{stem}.svg"
    pdf = out_dir / f"{stem}.pdf"

    plt.savefig(svg, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")

    print(f"[OK] Saved {svg}")
    print(f"[OK] Saved {pdf}")


def plot_delta_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    legend_label: str,
    stem: str,
    out_dir: Path,
    cmap: str | None = None,
    norm=None,
):
    world = _world_gdf()

    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=0.05)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

    gdf.plot(
        ax=ax,
        column=column,
        cmap=(cmap or DIV_CMAP),
        norm=norm,
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        legend_kwds={
            "label": legend_label,
            "orientation": "horizontal",
            "shrink": 0.7,
        },
    )

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def plot_bias_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    stem: str,
    out_dir: Path,
    cmap: str | None = None,
    norm=None,
):
    world = _world_gdf()

    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=0.05)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

    gdf.plot(
        ax=ax,
        column=column,
        cmap=(cmap or BIAS_CMAP),
        norm=norm,
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        legend_kwds={
            "label": "Mean bias (Model - Obs) [K]",
            "orientation": "horizontal",
            "shrink": 0.7,
        },
    )

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def plot_abs_rmse_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    legend_label: str,
    stem: str,
    out_dir: Path,
    cmap: str | None = None,
    vmin=None,
    vmax=None,
):
    world = _world_gdf()

    if vmin is None or vmax is None:
        vals = (
            pd.to_numeric(gdf[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        vmin, vmax = (
            (np.nanpercentile(vals, 5), np.nanpercentile(vals, 95))
            if len(vals)
            else (None, None)
        )

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

    gdf.plot(
        ax=ax,
        column=column,
        cmap=(cmap or RMSE_CMAP),
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={
            "label": legend_label,
            "orientation": "horizontal",
            "shrink": 0.7,
        },
    )

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def plot_hist(
    series: pd.Series,
    title: str,
    xlabel: str,
    stem: str,
    out_dir: Path,
    bins=25,
    fixed_bins: np.ndarray | None = None,
):
    data = pd.to_numeric(series, errors="coerce").dropna()

    plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(
        data,
        bins=(fixed_bins if fixed_bins is not None else bins),
        color="#3B9AB2",
        edgecolor="black",
    )

    plt.axvline(0, color="black", linestyle="--")

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Stations", fontsize=14)
    plt.title(title, fontsize=18)

    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close()


def plot_skill_score_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    stem: str,
    out_dir: Path,
    norm=None,
):
    world = _world_gdf()

    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=1.0, vcenter=0.0)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

    gdf.plot(
        ax=ax,
        column=column,
        cmap=SKILL_CMAP,
        norm=norm,
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        legend_kwds={
            "label": f"Skill vs {RAW_NAME} (MAE) [%]",
            "orientation": "horizontal",
            "shrink": 0.7,
        },
    )

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def plot_skill_diff_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    legend_label: str,
    stem: str,
    out_dir: Path,
):
    world = _world_gdf()
    norm = robust_symmetric_norm(gdf[column])

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

    gdf.plot(
        ax=ax,
        column=column,
        cmap=SKILL_CMAP,
        norm=norm,
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        legend_kwds={
            "label": legend_label,
            "orientation": "horizontal",
            "shrink": 0.7,
        },
    )

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def plot_monthly_maess(joined: pd.DataFrame, stem: str, out_dir: Path, title_suffix: str = ""):
    df = joined.copy()
    df["month"] = df[SPLIT].dt.to_period("M").dt.to_timestamp()

    eps = 1e-12

    monthly = (
        df.groupby("month")
        .apply(
            lambda g: pd.Series({
                "mae_raw": mae(g[OBS] - g[RAW]),
                "mae_mos": mae(g[OBS] - g[MOS_CORR]),
                f"mae_{ML1_TAG}": mae(g[OBS] - g[ML1_CORR]),
                "n": len(g),
            }),
            include_groups=False,
        )
        .reset_index()
        .sort_values("month")
    )

    monthly["maess_mos_pct"] = 100.0 * (
        1.0 - monthly["mae_mos"] / (monthly["mae_raw"] + eps)
    )

    monthly[f"maess_{ML1_TAG}_pct"] = 100.0 * (
        1.0 - monthly[f"mae_{ML1_TAG}"] / (monthly["mae_raw"] + eps)
    )

    plt.figure(figsize=(9, 5), dpi=150)

    plt.plot(
        monthly["month"],
        monthly["maess_mos_pct"],
        marker="o",
        linewidth=2,
        label="MOS",
        color="#637AB9",
    )

    plt.plot(
        monthly["month"],
        monthly[f"maess_{ML1_TAG}_pct"],
        marker="o",
        linewidth=2,
        label=ML1_NAME,
        color="#D18F49",
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Month", fontsize=14)
    plt.ylabel("MAESS [%]", fontsize=14)
    plt.title(f"Monthly overall MAESS{title_suffix}", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close()


def plot_monthly_rmse_difference(
    joined: pd.DataFrame,
    stem: str,
    out_dir: Path,
    title_suffix: str = "",
):
    df = joined.copy()
    df["month"] = df[SPLIT].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month")
        .apply(
            lambda g: pd.Series({
                "rmse_mos": rmse(g[OBS] - g[MOS_CORR]),
                f"rmse_{ML1_TAG}": rmse(g[OBS] - g[ML1_CORR]),
                "n": len(g),
            }),
            include_groups=False,
        )
        .reset_index()
        .sort_values("month")
    )

    monthly["delta_rmse_mos_minus_ml"] = (
        monthly["rmse_mos"] - monthly[f"rmse_{ML1_TAG}"]
    )

    plt.figure(figsize=(9, 5), dpi=150)

    plt.plot(
        monthly["month"],
        monthly["delta_rmse_mos_minus_ml"],
        marker="o",
        linewidth=2,
        label=f"MOS - {ML1_NAME}",
        color="#637AB9",
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Delta RMSE [K]", fontsize=14)
    plt.title(f"Monthly RMSE difference{title_suffix}", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close()


# =====================
# Unified evaluation runner
# =====================

def run_evaluation_case(
    mos_eval: pd.DataFrame,
    ml1_eval: pd.DataFrame,
    ml2_eval: pd.DataFrame | None,
    label: str,
    out_dir: Path,
):
    t0, t1 = mos_coverage_window(mos_eval)

    print(f"MOS coverage window: {t0} -> {t1}")

    season_label = label or f"{t0:%b %Y} - {t1:%b %Y}"

    if COMPARE_MODE == "single":
        joined = align_mos_window_join_single(mos_eval, ml1_eval, t0, t1)

        print("Rows after alignment single:", len(joined))
        print("Stations after alignment single:", joined["SID"].nunique())

        if joined.empty:
            print("[WARN] Joined data is empty; skipping case.")
            return

        station = compute_station_metrics_single_ml_vs_mos(joined)
        gdf = attach_stations_gdf(station)

        rmse_vmin, rmse_vmax = shared_sequential_limits(
            [gdf["rmse_raw"], gdf["rmse_mos"], gdf[f"rmse_{ML1_TAG}"]]
        )

        bias_norm = shared_diverging_norm(
            [gdf["bias_raw"], gdf["bias_mos"], gdf[f"bias_{ML1_TAG}"]]
        )

        ov_rmse_raw = wmean(station["rmse_raw"], station["n"])
        ov_rmse_mos = wmean(station["rmse_mos"], station["n"])
        ov_rmse_ml1 = wmean(station[f"rmse_{ML1_TAG}"], station["n"])

        ov_bias_raw = wmean(station["bias_raw"], station["n"])
        ov_bias_mos = wmean(station["bias_mos"], station["n"])
        ov_bias_ml1 = wmean(station[f"bias_{ML1_TAG}"], station["n"])

        ov_drmse_mos_minus_ml1 = wmean(station["delta_rmse_mos_minus_ml1"], station["n"])
        ov_drmse_ml1_minus_raw = wmean(station["delta_rmse_ml1_minus_raw"], station["n"])
        ov_drmse_mos_minus_raw = wmean(station["delta_rmse_mos_minus_raw"], station["n"])

        ov_skill_mos = wmean(station["skill_mos_pct"], station["n"])
        ov_skill_ml1 = wmean(station[f"skill_{ML1_TAG}_pct"], station["n"])
        ov_dskill_ml1_minus_mos = wmean(station["delta_skill_ml1_minus_mos_pct"], station["n"])

        safe_ml1 = safe_name(ML1_NAME)
        safe_raw = safe_name(RAW_NAME)

        plot_hist(
            station["delta_rmse_mos_minus_ml1"],
            f"Delta RMSE MOS - {ML1_NAME}",
            f"Delta RMSE MOS - {ML1_NAME} [K]",
            f"hist_delta_rmse_mos_minus_{safe_ml1}",
            out_dir,
        )

        plot_hist(
            station["delta_rmse_mos_minus_raw"],
            f"Delta RMSE MOS - {RAW_NAME}",
            f"Delta RMSE MOS - {RAW_NAME} [K]",
            f"hist_delta_rmse_mos_minus_{safe_raw}",
            out_dir,
        )

        plot_hist(
            station["delta_skill_ml1_minus_mos_pct"],
            f"Delta Skill {ML1_NAME} - MOS",
            f"Delta Skill {ML1_NAME} - MOS [%]",
            f"hist_delta_skill_{safe_ml1}_minus_mos",
            out_dir,
        )

        plot_hist(
            station["skill_mos_pct"],
            f"MOS skill vs {RAW_NAME}",
            f"Skill vs {RAW_NAME} [%]",
            "hist_skill_mos_vs_raw",
            out_dir,
        )

        plot_hist(
            station[f"skill_{ML1_TAG}_pct"],
            f"{ML1_NAME} skill vs {RAW_NAME}",
            f"Skill vs {RAW_NAME} [%]",
            f"hist_skill_{safe_ml1}_vs_raw",
            out_dir,
        )

        plot_abs_rmse_map(
            gdf,
            "rmse_raw",
            f"{RAW_NAME} RMSE ({season_label}) - overall {ov_rmse_raw:.2f} K",
            f"RMSE ({RAW_NAME}) [K]",
            f"map_rmse_{safe_raw}",
            out_dir,
            vmin=rmse_vmin,
            vmax=rmse_vmax,
        )

        plot_abs_rmse_map(
            gdf,
            "rmse_mos",
            f"MOS RMSE ({season_label}) - overall {ov_rmse_mos:.2f} K",
            "RMSE (MOS) [K]",
            "map_rmse_mos",
            out_dir,
            vmin=rmse_vmin,
            vmax=rmse_vmax,
        )

        plot_abs_rmse_map(
            gdf,
            f"rmse_{ML1_TAG}",
            f"{ML1_NAME} RMSE ({season_label}) - overall {ov_rmse_ml1:.2f} K",
            f"RMSE ({ML1_NAME}) [K]",
            f"map_rmse_{safe_ml1}",
            out_dir,
            vmin=rmse_vmin,
            vmax=rmse_vmax,
        )

        plot_delta_map(
            gdf,
            "delta_rmse_mos_minus_raw",
            f"MOS vs {RAW_NAME}\nDelta RMSE ({season_label}) - mean {ov_drmse_mos_minus_raw:+.2f} K",
            f"Delta RMSE MOS - {RAW_NAME} [K]",
            f"map_delta_rmse_mos_vs_{safe_raw}",
            out_dir,
            norm=auto_diverging_norm(gdf["delta_rmse_mos_minus_raw"], q=0.95, min_span=0.05),
        )

        plot_delta_map(
            gdf,
            "delta_rmse_mos_minus_ml1",
            f"MOS vs {ML1_NAME}\nDelta RMSE ({season_label}) - mean {ov_drmse_mos_minus_ml1:+.2f} K",
            f"Delta RMSE MOS - {ML1_NAME} [K]",
            f"map_delta_rmse_mos_vs_{safe_ml1}",
            out_dir,
            norm=auto_diverging_norm(gdf["delta_rmse_mos_minus_ml1"], q=0.95, min_span=0.05),
        )

        plot_delta_map(
            gdf,
            "delta_rmse_ml1_minus_raw",
            f"{ML1_NAME} vs {RAW_NAME}\nDelta RMSE ({season_label}) - mean {ov_drmse_ml1_minus_raw:+.2f} K",
            f"Delta RMSE {ML1_NAME} - {RAW_NAME} [K]",
            f"map_delta_rmse_{safe_ml1}_vs_{safe_raw}",
            out_dir,
            norm=auto_diverging_norm(gdf["delta_rmse_ml1_minus_raw"], q=0.95, min_span=0.05),
        )

        plot_bias_map(
            gdf,
            "bias_raw",
            f"{RAW_NAME} mean bias ({season_label}) - overall {ov_bias_raw:+.2f} K",
            f"map_bias_{safe_raw}",
            out_dir,
            norm=bias_norm,
        )

        plot_bias_map(
            gdf,
            "bias_mos",
            f"MOS mean bias ({season_label}) - overall {ov_bias_mos:+.2f} K",
            "map_bias_mos",
            out_dir,
            norm=bias_norm,
        )

        plot_bias_map(
            gdf,
            f"bias_{ML1_TAG}",
            f"{ML1_NAME} mean bias ({season_label}) - overall {ov_bias_ml1:+.2f} K",
            f"map_bias_{safe_ml1}",
            out_dir,
            norm=bias_norm,
        )

        skill_norm = shared_diverging_norm(
            [gdf["skill_mos_pct"], gdf[f"skill_{ML1_TAG}_pct"]],
            q=0.95,
            vcenter=0.0,
        )

        plot_skill_score_map(
            gdf,
            "skill_mos_pct",
            f"MOS skill vs {RAW_NAME} ({season_label}) - overall {ov_skill_mos:+.2f}%",
            "map_skill_mos_vs_raw",
            out_dir,
            norm=skill_norm,
        )

        plot_skill_score_map(
            gdf,
            f"skill_{ML1_TAG}_pct",
            f"{ML1_NAME} skill vs {RAW_NAME} ({season_label}) - overall {ov_skill_ml1:+.2f}%",
            f"map_skill_{safe_ml1}_vs_raw",
            out_dir,
            norm=skill_norm,
        )

        plot_skill_diff_map(
            gdf,
            "delta_skill_ml1_minus_mos_pct",
            f"{ML1_NAME} vs MOS - Skill Difference ({season_label})\nmean delta skill {ov_dskill_ml1_minus_mos:+.2f}%",
            f"Delta Skill {ML1_NAME} - MOS [%]",
            f"map_delta_skill_{safe_ml1}_vs_mos",
            out_dir,
        )

        plot_monthly_maess(
            joined,
            stem="line_monthly_maess_mos_vs_ml",
            out_dir=out_dir,
            title_suffix=f" ({season_label})",
        )

        plot_monthly_rmse_difference(
            joined,
            stem="line_monthly_delta_rmse_mos_vs_ml",
            out_dir=out_dir,
            title_suffix=f" ({season_label})",
        )

    else:
        if ml2_eval is None:
            raise ValueError("compare-mode='pair' but ml2_eval is None.")

        joined = align_mos_window_join(mos_eval, ml1_eval, ml2_eval, t0, t1)

        print("Rows after alignment pair:", len(joined))
        print("Stations after alignment pair:", joined["SID"].nunique())

        if joined.empty:
            print("[WARN] Joined data is empty; skipping case.")
            return

        station_two_ml = compute_station_metrics_two_ml(joined)
        station = compute_station_metrics_bestml_vs_mos(joined, station_two_ml)

        gdf_ml = attach_stations_gdf(station_two_ml)
        gdf_best = attach_stations_gdf(station)

        rmse_vmin, rmse_vmax = shared_sequential_limits([
            gdf_best["rmse_raw"],
            gdf_best["rmse_mos"],
            gdf_best["rmse_best_ml"],
            gdf_ml[f"rmse_{ML1_TAG}"],
            gdf_ml[f"rmse_{ML2_TAG}"],
        ])

        bias_norm = shared_diverging_norm([
            gdf_best["bias_raw"],
            gdf_best["bias_mos"],
            gdf_best["bias_best_ml"],
            gdf_ml[f"bias_{ML1_TAG}"],
            gdf_ml[f"bias_{ML2_TAG}"],
        ])

        ov_rmse_raw = wmean(station["rmse_raw"], station.get("n", None))
        ov_rmse_mos = wmean(station["rmse_mos"], station.get("n", None))
        ov_rmse_best = wmean(station["rmse_best_ml"], station.get("n", None))

        ov_bias_raw = wmean(station["bias_raw"], station.get("n", None))
        ov_bias_mos = wmean(station["bias_mos"], station.get("n", None))
        ov_bias_best = wmean(station["bias_best_ml"], station.get("n", None))

        ov_drmse_mos_minus_best = wmean(station["delta_rmse_mos_minus_bestml"], station.get("n", None))
        ov_drmse_best_minus_raw = wmean(station["delta_rmse_bestml_minus_raw"], station.get("n", None))
        ov_drmse_mos_minus_raw = wmean(station["delta_rmse_mos_minus_raw"], station.get("n", None))

        ov_skill_mos = wmean(station["skill_mos_pct"], station.get("n", None))
        ov_skill_best = wmean(station["skill_best_ml_pct"], station.get("n", None))
        ov_dskill_best_minus_mos = wmean(station["delta_skill_bestml_minus_mos_pct"], station.get("n", None))

        safe_raw = safe_name(RAW_NAME)

        plot_abs_rmse_map(
            gdf_best,
            "rmse_raw",
            f"{RAW_NAME} RMSE ({season_label}) - overall {ov_rmse_raw:.2f} K",
            f"RMSE ({RAW_NAME}) [K]",
            f"map_rmse_{safe_raw}",
            out_dir,
            vmin=rmse_vmin,
            vmax=rmse_vmax,
        )

        plot_abs_rmse_map(
            gdf_best,
            "rmse_mos",
            f"MOS RMSE ({season_label}) - overall {ov_rmse_mos:.2f} K",
            "RMSE (MOS) [K]",
            "map_rmse_mos",
            out_dir,
            vmin=rmse_vmin,
            vmax=rmse_vmax,
        )

        plot_abs_rmse_map(
            gdf_best,
            "rmse_best_ml",
            f"Best-ML RMSE ({season_label}) - overall {ov_rmse_best:.2f} K",
            "RMSE (Best-ML) [K]",
            "map_rmse_best_ml",
            out_dir,
            vmin=rmse_vmin,
            vmax=rmse_vmax,
        )

        plot_delta_map(
            gdf_best,
            "delta_rmse_mos_minus_raw",
            f"MOS vs {RAW_NAME}\nDelta RMSE ({season_label}) - mean {ov_drmse_mos_minus_raw:+.2f} K",
            f"Delta RMSE MOS - {RAW_NAME} [K]",
            f"map_delta_rmse_mos_vs_{safe_raw}",
            out_dir,
            norm=auto_diverging_norm(gdf_best["delta_rmse_mos_minus_raw"], q=0.95, min_span=0.05),
        )

        plot_delta_map(
            gdf_best,
            "delta_rmse_mos_minus_bestml",
            f"MOS vs Best-ML\nDelta RMSE ({season_label}) - mean {ov_drmse_mos_minus_best:+.2f} K",
            "Delta RMSE MOS - Best-ML [K]",
            "map_delta_rmse_mos_vs_bestml",
            out_dir,
            norm=auto_diverging_norm(gdf_best["delta_rmse_mos_minus_bestml"], q=0.95, min_span=0.05),
        )

        plot_delta_map(
            gdf_best,
            "delta_rmse_bestml_minus_raw",
            f"Best-ML vs {RAW_NAME}\nDelta RMSE ({season_label}) - mean {ov_drmse_best_minus_raw:+.2f} K",
            f"Delta RMSE Best-ML - {RAW_NAME} [K]",
            f"map_delta_rmse_bestml_vs_{safe_raw}",
            out_dir,
            norm=auto_diverging_norm(gdf_best["delta_rmse_bestml_minus_raw"], q=0.95, min_span=0.05),
        )

        plot_bias_map(
            gdf_best,
            "bias_raw",
            f"{RAW_NAME} mean bias ({season_label}) - overall {ov_bias_raw:+.2f} K",
            f"map_bias_{safe_raw}",
            out_dir,
            norm=bias_norm,
        )

        plot_bias_map(
            gdf_best,
            "bias_mos",
            f"MOS mean bias ({season_label}) - overall {ov_bias_mos:+.2f} K",
            "map_bias_mos",
            out_dir,
            norm=bias_norm,
        )

        plot_bias_map(
            gdf_best,
            "bias_best_ml",
            f"Best-ML mean bias ({season_label}) - overall {ov_bias_best:+.2f} K",
            "map_bias_best_ml",
            out_dir,
            norm=bias_norm,
        )

        skill_norm = shared_diverging_norm(
            [gdf_best["skill_mos_pct"], gdf_best["skill_best_ml_pct"]],
            q=0.95,
            vcenter=0.0,
        )

        plot_skill_score_map(
            gdf_best,
            "skill_mos_pct",
            f"MOS skill vs {RAW_NAME} ({season_label}) - overall {ov_skill_mos:+.2f}%",
            "map_skill_mos_vs_raw",
            out_dir,
            norm=skill_norm,
        )

        plot_skill_score_map(
            gdf_best,
            "skill_best_ml_pct",
            f"Best-ML skill vs {RAW_NAME} ({season_label}) - overall {ov_skill_best:+.2f}%",
            "map_skill_bestml_vs_raw",
            out_dir,
            norm=skill_norm,
        )

        plot_skill_diff_map(
            gdf_best,
            "delta_skill_bestml_minus_mos_pct",
            f"Best-ML vs MOS - Skill Difference ({season_label})\nmean delta skill {ov_dskill_best_minus_mos:+.2f}%",
            "Delta Skill Best-ML - MOS [%]",
            "map_delta_skill_bestml_vs_mos",
            out_dir,
        )

        plot_hist(
            station["skill_mos_pct"],
            f"MOS skill vs {RAW_NAME}",
            f"Skill vs {RAW_NAME} [%]",
            "hist_skill_mos_vs_raw",
            out_dir,
        )

        plot_hist(
            station["skill_best_ml_pct"],
            f"Best-ML skill vs {RAW_NAME}",
            f"Skill vs {RAW_NAME} [%]",
            "hist_skill_bestml_vs_raw",
            out_dir,
        )

        plot_hist(
            station["delta_rmse_mos_minus_raw"],
            f"Delta RMSE MOS - {RAW_NAME}",
            f"Delta RMSE MOS - {RAW_NAME} [K]",
            f"hist_delta_rmse_mos_minus_{safe_raw}",
            out_dir,
        )

        plot_hist(
            station["delta_rmse_mos_minus_bestml"],
            "Delta RMSE MOS - Best-ML",
            "Delta RMSE MOS - Best-ML [K]",
            "hist_delta_rmse_mos_minus_bestml",
            out_dir,
        )

        plot_hist(
            station["delta_skill_bestml_minus_mos_pct"],
            "Delta Skill Best-ML - MOS",
            "Delta Skill Best-ML - MOS [%]",
            "hist_delta_skill_bestml_minus_mos_pct",
            out_dir,
        )

    print("Saved figures to:", out_dir)


# =====================
# Main
# =====================

def main():
    global ML1_TAG, ML2_TAG, ML1_NAME, ML2_NAME
    global ML1_CORR, ML2_CORR, RAW_NAME, COMPARE_MODE
    global STATIONS_CSV, WORLD_SHP
    global LON_MIN, LON_MAX, LAT_MIN, LAT_MAX

    args = parse_args()

    if args.seasonal and args.non_seasonal:
        raise ValueError("Use either --seasonal or --non-seasonal, not both.")

    os.environ["OMP_NUM_THREADS"] = args.threads
    os.environ["MKL_NUM_THREADS"] = args.threads

    mos_dir = Path(args.mos_dir)
    ml1_dir = Path(args.ml1_dir)
    ml2_dir = Path(args.ml2_dir) if args.ml2_dir is not None else None
    base_out_dir = Path(args.output_dir)

    STATIONS_CSV = Path(args.stations_csv)
    WORLD_SHP = Path(args.world_shp)

    if not STATIONS_CSV.exists():
        raise FileNotFoundError(f"Stations CSV not found: {STATIONS_CSV}")

    if not WORLD_SHP.exists():
        raise FileNotFoundError(f"World shapefile not found: {WORLD_SHP}")

    ML1_TAG = args.ml1_tag
    ML2_TAG = args.ml2_tag

    ML1_NAME = args.ml1_name
    ML2_NAME = args.ml2_name

    ML1_CORR = f"corrected_{ML1_TAG}"
    ML2_CORR = f"corrected_{ML2_TAG}"

    RAW_NAME = args.raw_name
    COMPARE_MODE = args.compare_mode

    LON_MIN = args.lon_min
    LON_MAX = args.lon_max
    LAT_MIN = args.lat_min
    LAT_MAX = args.lat_max

    seasonal = True if args.seasonal else False

    if COMPARE_MODE == "pair" and ml2_dir is None:
        raise ValueError("--ml2-dir is required when --compare-mode pair.")

    print(f"[INFO] MOS directory: {mos_dir}")
    print(f"[INFO] ML1 directory: {ml1_dir}")
    print(f"[INFO] ML2 directory: {ml2_dir}")
    print(f"[INFO] Output base directory: {base_out_dir}")
    print(f"[INFO] Compare mode: {COMPARE_MODE}")
    print(f"[INFO] Seasonal mode: {seasonal}")
    print(f"[INFO] Leadtime filter: {args.leadtime_filter}")
    print(f"[INFO] Max leadtime: {args.max_leadtime}")

    if seasonal:
        for year in args.season_years:
            for season in args.seasons:
                print(f"\n[INFO] Processing {year} {season}")

                mos_eval = load_eval_rows_evaldir(
                    mos_dir,
                    pattern=f"eval_rows_{SPLIT}_MOS_{year}_{season}.parquet",
                    cols=["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, MOS_CORR],
                )

                ml1_eval = load_model_eval_rows(
                    ml1_dir,
                    ml_tag=ML1_TAG,
                    ml_corr=ML1_CORR,
                    ml_name=ML1_NAME,
                )

                ml2_eval = None

                if COMPARE_MODE == "pair":
                    ml2_eval = load_model_eval_rows(
                        ml2_dir,
                        ml_tag=ML2_TAG,
                        ml_corr=ML2_CORR,
                        ml_name=ML2_NAME,
                    )

                if args.leadtime_filter:
                    mos_eval = mos_eval[mos_eval["leadtime"] <= args.max_leadtime].copy()
                    ml1_eval = ml1_eval[ml1_eval["leadtime"] <= args.max_leadtime].copy()

                    if ml2_eval is not None:
                        ml2_eval = ml2_eval[ml2_eval["leadtime"] <= args.max_leadtime].copy()

                    out_dir = base_out_dir / f"{season}_ldt{args.max_leadtime}"
                else:
                    out_dir = base_out_dir / season

                out_dir.mkdir(parents=True, exist_ok=True)

                run_evaluation_case(
                    mos_eval=mos_eval,
                    ml1_eval=ml1_eval,
                    ml2_eval=ml2_eval,
                    label=season,
                    out_dir=out_dir,
                )

    else:
        print("\n[INFO] Processing non-seasonal evaluation")

        mos_eval = load_eval_rows_evaldir(
            mos_dir,
            pattern=f"eval_rows_{SPLIT}_MOS_20*.parquet",
            cols=["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, MOS_CORR],
        )

        ml1_eval = load_model_eval_rows(
            ml1_dir,
            ml_tag=ML1_TAG,
            ml_corr=ML1_CORR,
            ml_name=ML1_NAME,
        )

        ml2_eval = None

        if COMPARE_MODE == "pair":
            ml2_eval = load_model_eval_rows(
                ml2_dir,
                ml_tag=ML2_TAG,
                ml_corr=ML2_CORR,
                ml_name=ML2_NAME,
            )

        if args.leadtime_filter:
            mos_eval = mos_eval[mos_eval["leadtime"] <= args.max_leadtime].copy()
            ml1_eval = ml1_eval[ml1_eval["leadtime"] <= args.max_leadtime].copy()

            if ml2_eval is not None:
                ml2_eval = ml2_eval[ml2_eval["leadtime"] <= args.max_leadtime].copy()

            out_dir = base_out_dir / f"ldt{args.max_leadtime}"
        else:
            out_dir = base_out_dir

        out_dir.mkdir(parents=True, exist_ok=True)

        run_evaluation_case(
            mos_eval=mos_eval,
            ml1_eval=ml1_eval,
            ml2_eval=ml2_eval,
            label="",
            out_dir=out_dir,
        )


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()