import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# =====================
# Config
# =====================
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

HOME = Path.home()
DATA_DIR = HOME / "thesis_project" / "data"
METRICS_DIR = HOME / "thesis_project" / "metrics"
MOS_DIR = METRICS_DIR / "mos"
ML1_DIR = METRICS_DIR / "bias_lstm_stream"
ML2_DIR = METRICS_DIR / "2019_tuned_new"

STATIONS_CSV = DATA_DIR / "stations.csv"
MAPS_DIR = DATA_DIR / "maps"
WORLD_SHP = MAPS_DIR / "ne_110m_admin_0_countries.shp"

LON_MIN, LAT_MIN = -25.0, 25.5
LON_MAX, LAT_MAX = 42.0, 72.0

SPLIT = "validtime"
OBS = "obs_TA"
RAW = "raw_fc"
MOS_CORR = "corrected_mos"

RAW_NAME = "ECMWF"  # <<<<<<<<<<<<<<<<<<<<<<<<

ML1_TAG = "bias_lstm_stream"
ML2_TAG = "tuned_ah_2019"

ML1_NAME = "EC_ML_LSTM"
ML2_NAME = "EC_ML_XGB"

ML1_CORR = f"corrected_{ML1_TAG}"
ML2_CORR = f"corrected_{ML2_TAG}"

COMPARE_MODE = "single"   # "single" or "pair"
SEASONAL = True
LDT_FILTER = False
MAX_LEADTIME = 48

DIV_CMAP = "PuOr"
SKILL_CMAP = LinearSegmentedColormap.from_list("skill_cmp", ["#570040", "white", "#005717"])
BIAS_CMAP = "coolwarm"
RMSE_CMAP = "hot_r"

DO_NRMSE = False  # unchanged toggle (still available)


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
    vals = pd.concat([pd.to_numeric(s, errors="coerce") for s in series_list], ignore_index=True)
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        span = 1.0
    else:
        span = float(np.nanquantile(np.abs(vals), q))
        if not np.isfinite(span) or span == 0:
            span = float(np.nanmax(np.abs(vals))) or 1.0
    return TwoSlopeNorm(vmin=-span, vcenter=vcenter, vmax=+span)

def shared_sequential_limits(series_list, lo=5, hi=95):
    vals = pd.concat([pd.to_numeric(s, errors="coerce") for s in series_list], ignore_index=True)
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None, None
    return float(np.nanpercentile(vals, lo)), float(np.nanpercentile(vals, hi))

def auto_diverging_norm(series: pd.Series, q: float = 0.95, vcenter: float = 0.0,
                        min_span: float | None = None, max_span: float | None = None) -> TwoSlopeNorm:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
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
    vals = pd.to_numeric(series, errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
    span = float(np.nanquantile(vals, q)) if len(vals) else 1.0
    return TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

def attach_stations_gdf(station_metrics: pd.DataFrame) -> gpd.GeoDataFrame:
    st = pd.read_csv(STATIONS_CSV)
    st["SID"] = st["SID"].astype(str)
    df = st.merge(station_metrics, on="SID", how="left")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")


# =====================
# Data loading / alignment
# =====================
def load_eval_rows_evaldir(eval_dir: Path, pattern: str, cols: list[str]) -> pd.DataFrame:
    files = sorted(eval_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=[c for c in cols if c != "SID"] + (["SID"] if "SID" in cols else []))
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    if "SID" in out.columns:
        out["SID"] = out["SID"].astype(str)
    if SPLIT in out.columns:
        out[SPLIT] = pd.to_datetime(out[SPLIT], errors="coerce").dt.tz_localize(None)
    return out

def mos_coverage_window(mos_eval: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = mos_eval.dropna(subset=[MOS_CORR, RAW, OBS]).copy()
    if df.empty:
        raise ValueError("MOS eval rows have no usable values.")
    return df[SPLIT].min(), df[SPLIT].max()

def align_mos_window_join_single(mos_eval: pd.DataFrame, ml1_eval: pd.DataFrame,
                                 t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    mos = mos_eval[(mos_eval[SPLIT] >= t0) & (mos_eval[SPLIT] <= t1)].copy()
    keys = ["SID", SPLIT, "analysistime", "leadtime"]
    keep_cols = keys + [RAW, OBS, MOS_CORR, ML1_CORR]
    joined = (
        mos.merge(ml1_eval[keys + [ML1_CORR]], on=keys, how="inner")
           .dropna(subset=[OBS, RAW, MOS_CORR, ML1_CORR])[keep_cols]
           .copy()
    )
    print(f"MOS-driven window (single-ML): {joined[SPLIT].min()} → {joined[SPLIT].max()}" if not joined.empty
          else "Warning: No aligned rows after MOS-driven single-ML join.")
    return joined

def align_mos_window_join(mos_eval: pd.DataFrame, ml1_eval: pd.DataFrame, ml2_eval: pd.DataFrame,
                          t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    mos = mos_eval[(mos_eval[SPLIT] >= t0) & (mos_eval[SPLIT] <= t1)].copy()
    keys = ["SID", SPLIT, "analysistime", "leadtime"]
    keep_cols = keys + [RAW, OBS, MOS_CORR, ML1_CORR, ML2_CORR]
    joined = (
        mos.merge(ml1_eval[keys + [ML1_CORR]], on=keys, how="inner")
           .merge(ml2_eval[keys + [ML2_CORR]], on=keys, how="inner")
           .dropna(subset=[OBS, RAW, MOS_CORR, ML1_CORR, ML2_CORR])[keep_cols]
           .copy()
    )
    print(f"MOS-driven window: {joined[SPLIT].min()} → {joined[SPLIT].max()}" if not joined.empty
          else "Warning: No aligned rows after MOS-driven join.")
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
            include_groups=False
        ).reset_index(drop=True)
    )

    # Skill vs ECMWF/raw (MAE-based)
    st["skill_mos_pct"] = 100.0 * (1.0 - st["mae_mos"] / (st["mae_raw"] + eps))
    st[f"skill_{ML1_TAG}_pct"] = 100.0 * (1.0 - st[f"mae_{ML1_TAG}"] / (st["mae_raw"] + eps))
    st["delta_skill_ml1_minus_mos_pct"] = st[f"skill_{ML1_TAG}_pct"] - st["skill_mos_pct"]

    # RMSE deltas
    st["delta_rmse_mos_minus_ml1"] = st["rmse_mos"] - st[f"rmse_{ML1_TAG}"]
    st["delta_rmse_ml1_minus_raw"] = st[f"rmse_{ML1_TAG}"] - st["rmse_raw"]
    st["delta_rmse_mos_minus_raw"] = st["rmse_mos"] - st["rmse_raw"]  # NEW

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
        ).reset_index(drop=True)
    )

    station[f"skill_{ML1_TAG}_pct"] = 100.0 * (1.0 - station[f"mae_{ML1_TAG}"] / (station["mae_raw"] + eps))
    station[f"skill_{ML2_TAG}_pct"] = 100.0 * (1.0 - station[f"mae_{ML2_TAG}"] / (station["mae_raw"] + eps))

    station["delta_skill_ML2_minus_ML1_pct"] = station[f"skill_{ML2_TAG}_pct"] - station[f"skill_{ML1_TAG}_pct"]
    station["delta_rmse_ML1_minus_ML2"] = station[f"rmse_{ML1_TAG}"] - station[f"rmse_{ML2_TAG}"]

    station["best_ml_tag"] = np.where(station[f"rmse_{ML2_TAG}"] < station[f"rmse_{ML1_TAG}"], ML2_TAG, ML1_TAG)
    station["rmse_best_ml"] = station[[f"rmse_{ML1_TAG}", f"rmse_{ML2_TAG}"]].min(axis=1)
    station["mae_best_ml"] = station[[f"mae_{ML1_TAG}", f"mae_{ML2_TAG}"]].min(axis=1)
    station["bias_best_ml"] = np.where(
        station["best_ml_tag"] == ML2_TAG, station[f"bias_{ML2_TAG}"], station[f"bias_{ML1_TAG}"]
    )

    return station

def compute_station_metrics_bestml_vs_mos(joined: pd.DataFrame, station_two_ml: pd.DataFrame) -> pd.DataFrame:
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
        ).reset_index(drop=True)
    )

    st = station_two_ml.merge(mos_stats, on="SID", how="left", suffixes=("_two", ""))

    eps = 1e-12
    st["skill_mos_pct"] = 100.0 * (1.0 - st["mae_mos"] / (st["mae_raw"] + eps))
    st["skill_best_ml_pct"] = 100.0 * (1.0 - st["mae_best_ml"] / (st["mae_raw"] + eps))

    st["delta_rmse_mos_minus_bestml"] = st["rmse_mos"] - st["rmse_best_ml"]
    st["delta_rmse_bestml_minus_raw"] = st["rmse_best_ml"] - st["rmse_raw"]
    st["delta_rmse_mos_minus_raw"] = st["rmse_mos"] - st["rmse_raw"]  # NEW

    st["delta_skill_bestml_minus_mos_pct"] = st["skill_best_ml_pct"] - st["skill_mos_pct"]

    return st


# =====================
# Plotting
# =====================
def plot_delta_map(gdf: gpd.GeoDataFrame, column: str, title: str, legend_label: str,
                   out_name: str, out_name_pdf: str,  OUT_DIR: Path, cmap: str | None = None, norm=None):
    world = gpd.read_file(WORLD_SHP)
    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=0.05)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax, column=column, cmap=(cmap or DIV_CMAP), norm=norm,
        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
        legend=True, legend_kwds={"label": legend_label, "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14); ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, bbox_inches="tight"); plt.savefig(OUT_DIR / out_name_pdf, bbox_inches="tight"); plt.close()

def plot_bias_map(gdf: gpd.GeoDataFrame, column: str, title: str,
                  out_name: str, out_name_pdf: str, OUT_DIR: Path, cmap: str | None = None, norm=None):
    world = gpd.read_file(WORLD_SHP)
    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=0.05)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax, column=column, cmap=(cmap or BIAS_CMAP), norm=norm,
        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
        legend=True, legend_kwds={"label": "Mean bias (Model − Obs) [K]", "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14); ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, bbox_inches="tight"); plt.savefig(OUT_DIR / out_name_pdf, bbox_inches="tight"); plt.close()

def plot_abs_rmse_map(gdf: gpd.GeoDataFrame, column: str, title: str, legend_label: str,
                      out_name: str, out_name_pdf: str, OUT_DIR: Path, cmap: str | None = None, vmin=None, vmax=None):
    world = gpd.read_file(WORLD_SHP)
    if vmin is None or vmax is None:
        vals = pd.to_numeric(gdf[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        vmin, vmax = (np.nanpercentile(vals, 5), np.nanpercentile(vals, 95)) if len(vals) else (None, None)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax, column=column, cmap=(cmap or RMSE_CMAP),
        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
        legend=True, vmin=vmin, vmax=vmax,
        legend_kwds={"label": legend_label, "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14); ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, bbox_inches="tight"); plt.savefig(OUT_DIR / out_name_pdf, bbox_inches="tight"); plt.close()

def plot_hist(series: pd.Series, title: str, xlabel: str, out_name: str, OUT_DIR: Path,
              bins=25, fixed_bins: np.ndarray | None = None):
    plt.figure(figsize=(6, 4))
    data = pd.to_numeric(series, errors="coerce").dropna()
    plt.hist(data, bins=(fixed_bins if fixed_bins is not None else bins), color="#3B9AB2", edgecolor="black")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel(xlabel, fontsize=14); plt.ylabel("Stations", fontsize=14); plt.title(title, fontsize=18)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, dpi=150); plt.close()

def plot_skill_score_map(gdf: gpd.GeoDataFrame, column: str, title: str,
                         out_name: str, out_name_pdf: str, OUT_DIR: Path, norm=None):
    world = gpd.read_file(WORLD_SHP)
    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=1.0, vcenter=0.0)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax, column=column, cmap=SKILL_CMAP, norm=norm,
        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
        legend=True,
        legend_kwds={"label": f"Skill vs {RAW_NAME} (MAE) [%]", "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14); ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, bbox_inches="tight"); plt.savefig(OUT_DIR / out_name_pdf, bbox_inches="tight"); plt.close()

def plot_skill_diff_map(gdf: gpd.GeoDataFrame, column: str, title: str, legend_label: str,
                        out_name: str, out_name_pdf: str, OUT_DIR: Path):
    world = gpd.read_file(WORLD_SHP)
    norm = robust_symmetric_norm(gdf[column])
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax, column=column, cmap=SKILL_CMAP, norm=norm,
        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
        legend=True, legend_kwds={"label": legend_label, "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)  # NOTE: no "(positive=...)" text
    ax.set_xlabel("Longitude", fontsize=14); ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, bbox_inches="tight"); plt.savefig(OUT_DIR / out_name_pdf, bbox_inches="tight"); plt.close()


def plot_monthly_maess(joined: pd.DataFrame, out_name: str, OUT_DIR: Path, title_suffix: str = ""):
    """
    Monthly overall MAE skill score (MAESS) against RAW/ECMWF.
    x-axis: month
    y-axis: MAESS [%]
    """

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
              include_groups=False
          )
          .reset_index()
          .sort_values("month")
    )

    monthly["maess_mos_pct"] = 100.0 * (1.0 - monthly["mae_mos"] / (monthly["mae_raw"] + eps))
    monthly[f"maess_{ML1_TAG}_pct"] = 100.0 * (
        1.0 - monthly[f"mae_{ML1_TAG}"] / (monthly["mae_raw"] + eps)
    )

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(monthly["month"], monthly["maess_mos_pct"], marker="o", linewidth=2, label="MOS", color="#637AB9")
    plt.plot(monthly["month"], monthly[f"maess_{ML1_TAG}_pct"], marker="o", linewidth=2, label=ML1_NAME, color="#D18F49")

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("MAESS [%]", fontsize=14)
    plt.title(f"Monthly overall MAESS{title_suffix}", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.savefig(OUT_DIR / "line_monthly_maess_mos_vs_ml.pdf", bbox_inches="tight")
    plt.close()

def plot_monthly_rmse_difference(joined: pd.DataFrame, out_name: str, OUT_DIR: Path, title_suffix: str = ""):
    """
    Monthly overall RMSE difference between MOS and ML.
    Positive = ML better (lower RMSE than MOS).
    Negative = MOS better.
    """
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
            include_groups=False
        )
        .reset_index()
        .sort_values("month")
    )

    monthly["delta_rmse_mos_minus_ml"] = monthly["rmse_mos"] - monthly[f"rmse_{ML1_TAG}"]

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(
        monthly["month"],
        monthly["delta_rmse_mos_minus_ml"],
        marker="o",
        linewidth=2,
        label=f"MOS − {ML1_NAME}",
        color="#637AB9"
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("ΔRMSE [K]", fontsize=14)
    plt.title(f"Monthly RMSE difference{title_suffix}", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.savefig((OUT_DIR / out_name).with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig((OUT_DIR / "line_monthly_delta_rmse_mos_vs_ml.pdf").with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

# =====================
# Unified evaluation runner (removes duplication)
# =====================
def run_evaluation_case(mos_eval: pd.DataFrame, ml1_eval: pd.DataFrame, ml2_eval: pd.DataFrame | None,
                        label: str, OUT_DIR: Path):

    t0, t1 = mos_coverage_window(mos_eval)
    print(f"MOS coverage window: {t0} → {t1}")
    season_label = label or f"{t0:%b %Y} – {t1:%b %Y}"

    if COMPARE_MODE == "single":
        joined = align_mos_window_join_single(mos_eval, ml1_eval, t0, t1)
        print("Rows after alignment (single):", len(joined))
        print("Stations after alignment (single):", joined["SID"].nunique())

        station = compute_station_metrics_single_ml_vs_mos(joined)
        gdf = attach_stations_gdf(station)

        # Shared scales
        rmse_vmin, rmse_vmax = shared_sequential_limits([gdf["rmse_raw"], gdf["rmse_mos"], gdf[f"rmse_{ML1_TAG}"]])
        bias_norm = shared_diverging_norm([gdf["bias_raw"], gdf["bias_mos"], gdf[f"bias_{ML1_TAG}"]])

        # Overall metrics
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

        # Histograms (RMSE deltas)
        plot_hist(station["delta_rmse_mos_minus_ml1"],
                  f"ΔRMSE (MOS−{ML1_NAME})", f"ΔRMSE (MOS − {ML1_NAME}) [K]",
                  f"hist_delta_rmse_mos_minus_{ML1_NAME}.svg", OUT_DIR)

        plot_hist(station["delta_rmse_mos_minus_raw"],
                  f"ΔRMSE (MOS−{RAW_NAME})", f"ΔRMSE (MOS − {RAW_NAME}) [K]",
                  f"hist_delta_rmse_mos_minus_{RAW_NAME}.svg", OUT_DIR)

        # Skill difference hist
        plot_hist(station["delta_skill_ml1_minus_mos_pct"],
                  f"ΔSkill ({ML1_NAME}−MOS)", f"ΔSkill ({ML1_NAME} − MOS) [%]",
                  f"hist_delta_skill_{ML1_NAME}_minus_mos.svg", OUT_DIR)

        # Skill score hists (NEW)
        plot_hist(station["skill_mos_pct"],
                  f"MOS skill vs {RAW_NAME} (per-station)", f"Skill vs {RAW_NAME} [%]",
                  "hist_skill_mos_vs_ecmwf.svg", OUT_DIR)

        plot_hist(station[f"skill_{ML1_TAG}_pct"],
                  f"{ML1_NAME} skill vs {RAW_NAME} (per-station)", f"Skill vs {RAW_NAME} [%]",
                  f"hist_skill_{ML1_NAME}_vs_ecmwf.svg", OUT_DIR)

        # RMSE maps
        plot_abs_rmse_map(gdf, "rmse_raw",
                          f"{RAW_NAME} RMSE ({season_label}) — overall {ov_rmse_raw:.2f} K",
                          f"RMSE ({RAW_NAME}) [K]", f"map_rmse_{RAW_NAME}.svg", f"map_rmse_{RAW_NAME}.pdf",  OUT_DIR,
                          vmin=rmse_vmin, vmax=rmse_vmax)

        plot_abs_rmse_map(gdf, "rmse_mos",
                          f"MOS RMSE ({season_label}) — overall {ov_rmse_mos:.2f} K",
                          "RMSE (MOS) [K]", "map_rmse_mos.svg", f"map_rmse_mos.pdf",  OUT_DIR,
                          vmin=rmse_vmin, vmax=rmse_vmax)

        plot_abs_rmse_map(gdf, f"rmse_{ML1_TAG}",
                          f"{ML1_NAME} RMSE ({season_label}) — overall {ov_rmse_ml1:.2f} K",
                          f"RMSE ({ML1_NAME}) [K]", f"map_rmse_{ML1_NAME}.svg", f"map_rmse_{ML1_NAME}.pdf", OUT_DIR,
                          vmin=rmse_vmin, vmax=rmse_vmax)

        # ΔRMSE maps (NEW: MOS vs ECMWF)
        plot_delta_map(
            gdf, "delta_rmse_mos_minus_raw",
            f"MOS vs {RAW_NAME}\nΔRMSE ({season_label}) — mean {ov_drmse_mos_minus_raw:+.2f} K",
            f"ΔRMSE (MOS − {RAW_NAME}) [K]",
            f"map_delta_rmse_mos_vs_{RAW_NAME}.svg", f"map_delta_rmse_mos_vs_{RAW_NAME}.pdf", OUT_DIR,
            norm=auto_diverging_norm(gdf["delta_rmse_mos_minus_raw"], q=0.95, min_span=0.05)
        )

        plot_delta_map(
            gdf, "delta_rmse_mos_minus_ml1",
            f"MOS vs {ML1_NAME}\nΔRMSE ({season_label}) — mean {ov_drmse_mos_minus_ml1:+.2f} K",
            f"ΔRMSE (MOS − {ML1_NAME}) [K]",
            f"map_delta_rmse_mos_vs_{ML1_NAME}.svg", f"map_delta_rmse_mos_vs_{ML1_NAME}.pdf", OUT_DIR,
            norm=auto_diverging_norm(gdf["delta_rmse_mos_minus_ml1"], q=0.95, min_span=0.05)
        )

        plot_delta_map(
            gdf, "delta_rmse_ml1_minus_raw",
            f"{ML1_NAME} vs {RAW_NAME}\nΔRMSE ({season_label}) — mean {ov_drmse_ml1_minus_raw:+.2f} K",
            f"ΔRMSE ({ML1_NAME} − {RAW_NAME}) [K]",
            f"map_delta_rmse_{ML1_NAME}_vs_{RAW_NAME}.svg", f"map_delta_rmse_{ML1_NAME}_vs_{RAW_NAME}.pdf", OUT_DIR,
            norm=auto_diverging_norm(gdf["delta_rmse_ml1_minus_raw"], q=0.95, min_span=0.05)
        )

        # Bias maps (ensures raw bias is plotted)
        plot_bias_map(gdf, "bias_raw",
                     f"{RAW_NAME} mean bias ({season_label}) — overall {ov_bias_raw:+.2f} K",
                     f"map_bias_{RAW_NAME}.svg", f"map_bias_{RAW_NAME}.pdf", OUT_DIR, norm=bias_norm)

        plot_bias_map(gdf, "bias_mos",
                     f"MOS mean bias ({season_label}) — overall {ov_bias_mos:+.2f} K",
                     "map_bias_mos.svg", "map_bias_mos.pdf", OUT_DIR, norm=bias_norm)

        plot_bias_map(gdf, f"bias_{ML1_TAG}",
                     f"{ML1_NAME} mean bias ({season_label}) — overall {ov_bias_ml1:+.2f} K",
                     f"map_bias_{ML1_NAME}.svg", f"map_bias_{ML1_NAME}.pdf", OUT_DIR, norm=bias_norm)

        # Skill score maps vs ECMWF (NEW)
        skill_norm = shared_diverging_norm([gdf["skill_mos_pct"], gdf[f"skill_{ML1_TAG}_pct"]], q=0.95, vcenter=0.0)

        plot_skill_score_map(gdf, "skill_mos_pct",
                             f"MOS skill vs {RAW_NAME} ({season_label}) — overall {ov_skill_mos:+.2f}%",
                             "map_skill_mos_vs_ecmwf.svg", "map_skill_mos_vs_ecmwf.pdf", OUT_DIR, norm=skill_norm)

        plot_skill_score_map(gdf, f"skill_{ML1_TAG}_pct",
                             f"{ML1_NAME} skill vs {RAW_NAME} ({season_label}) — overall {ov_skill_ml1:+.2f}%",
                             f"map_skill_{ML1_NAME}_vs_ecmwf.svg", f"map_skill_{ML1_NAME}_vs_ecmwf.pdf", OUT_DIR, norm=skill_norm)

        # Skill difference map (title has no "(positive = ...)")
        plot_skill_diff_map(
            gdf,
            "delta_skill_ml1_minus_mos_pct",
            f"{ML1_NAME} vs MOS — Skill Difference ({season_label})\nmean Δskill {ov_dskill_ml1_minus_mos:+.2f}%",
            f"ΔSkill ({ML1_NAME} − MOS) [%]",
            f"map_delta_skill_{ML1_NAME}_vs_mos.svg", f"map_delta_skill_{ML1_NAME}_vs_mos.pdf", OUT_DIR
        )

        plot_monthly_maess(
            joined,
            out_name="line_monthly_maess_mos_vs_ml.svg",
            OUT_DIR=OUT_DIR,
            title_suffix=f" ({season_label})"
        )
        plot_monthly_rmse_difference(
            joined,
            out_name="line_monthly_delta_rmse_mos_vs_ml.svg",
            OUT_DIR=OUT_DIR,
            title_suffix=f" ({season_label})"
        )

    else:
        # PAIR mode
        if ml2_eval is None:
            raise ValueError("COMPARE_MODE='pair' but ml2_eval was None.")

        joined = align_mos_window_join(mos_eval, ml1_eval, ml2_eval, t0, t1)
        print("Rows after alignment (pair):", len(joined))
        print("Stations after alignment (pair):", joined["SID"].nunique())

        station_two_ml = compute_station_metrics_two_ml(joined)
        station = compute_station_metrics_bestml_vs_mos(joined, station_two_ml)

        gdf_ml = attach_stations_gdf(station_two_ml)
        gdf_best = attach_stations_gdf(station)

        rmse_vmin, rmse_vmax = shared_sequential_limits([
            gdf_best["rmse_raw"], gdf_best["rmse_mos"], gdf_best["rmse_best_ml"],
            gdf_ml[f"rmse_{ML1_TAG}"], gdf_ml[f"rmse_{ML2_TAG}"],
        ])

        bias_norm = shared_diverging_norm([
            gdf_best["bias_raw"], gdf_best["bias_mos"], gdf_best["bias_best_ml"],
            gdf_ml[f"bias_{ML1_TAG}"], gdf_ml[f"bias_{ML2_TAG}"],
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

        # Maps
        plot_abs_rmse_map(gdf_best, "rmse_raw",
                          f"{RAW_NAME} RMSE ({season_label}) — overall {ov_rmse_raw:.2f} K",
                          f"RMSE ({RAW_NAME}) [K]", f"map_rmse_{RAW_NAME}.svg", OUT_DIR,
                          vmin=rmse_vmin, vmax=rmse_vmax)

        plot_abs_rmse_map(gdf_best, "rmse_mos",
                          f"MOS RMSE ({season_label}) — overall {ov_rmse_mos:.2f} K",
                          "RMSE (MOS) [K]", "map_rmse_mos.svg", OUT_DIR,
                          vmin=rmse_vmin, vmax=rmse_vmax)

        plot_abs_rmse_map(gdf_best, "rmse_best_ml",
                          f"Best-ML RMSE ({season_label}) — overall {ov_rmse_best:.2f} K",
                          "RMSE (Best-ML) [K]", "map_rmse_best_ml.svg", OUT_DIR,
                          vmin=rmse_vmin, vmax=rmse_vmax)

        # NEW: MOS vs ECMWF ΔRMSE
        plot_delta_map(
            gdf_best, "delta_rmse_mos_minus_raw",
            f"MOS vs {RAW_NAME}\nΔRMSE ({season_label}) — mean {ov_drmse_mos_minus_raw:+.2f} K",
            f"ΔRMSE (MOS − {RAW_NAME}) [K]",
            f"map_delta_rmse_mos_vs_{RAW_NAME}.svg", OUT_DIR,
            norm=auto_diverging_norm(gdf_best["delta_rmse_mos_minus_raw"], q=0.95, min_span=0.05)
        )

        plot_delta_map(
            gdf_best, "delta_rmse_mos_minus_bestml",
            f"MOS vs Best-ML\nΔRMSE ({season_label}) — mean {ov_drmse_mos_minus_best:+.2f} K",
            "ΔRMSE (MOS − Best-ML) [K]",
            "map_delta_rmse_mos_vs_bestml.svg", OUT_DIR,
            norm=auto_diverging_norm(gdf_best["delta_rmse_mos_minus_bestml"], q=0.95, min_span=0.05)
        )

        plot_delta_map(
            gdf_best, "delta_rmse_bestml_minus_raw",
            f"Best-ML vs {RAW_NAME}\nΔRMSE ({season_label}) — mean {ov_drmse_best_minus_raw:+.2f} K",
            f"ΔRMSE (Best-ML − {RAW_NAME}) [K]",
            f"map_delta_rmse_bestml_vs_{RAW_NAME}.svg", OUT_DIR,
            norm=auto_diverging_norm(gdf_best["delta_rmse_bestml_minus_raw"], q=0.95, min_span=0.05)
        )

        # Bias maps (including ECMWF always)
        plot_bias_map(gdf_best, "bias_raw",
                      f"{RAW_NAME} mean bias ({season_label}) — overall {ov_bias_raw:+.2f} K",
                      f"map_bias_{RAW_NAME}.svg", OUT_DIR, norm=bias_norm)

        plot_bias_map(gdf_best, "bias_mos",
                      f"MOS mean bias ({season_label}) — overall {ov_bias_mos:+.2f} K",
                      "map_bias_mos.svg", OUT_DIR, norm=bias_norm)

        plot_bias_map(gdf_best, "bias_best_ml",
                      f"Best-ML mean bias ({season_label}) — overall {ov_bias_best:+.2f} K",
                      "map_bias_best_ml.svg", OUT_DIR, norm=bias_norm)

        # Skill score maps vs ECMWF (NEW)
        skill_norm = shared_diverging_norm([gdf_best["skill_mos_pct"], gdf_best["skill_best_ml_pct"]], q=0.95, vcenter=0.0)

        plot_skill_score_map(gdf_best, "skill_mos_pct",
                             f"MOS skill vs {RAW_NAME} ({season_label}) — overall {ov_skill_mos:+.2f}%",
                             "map_skill_mos_vs_ecmwf.svg", OUT_DIR, norm=skill_norm)

        plot_skill_score_map(gdf_best, "skill_best_ml_pct",
                             f"Best-ML skill vs {RAW_NAME} ({season_label}) — overall {ov_skill_best:+.2f}%",
                             "map_skill_bestml_vs_ecmwf.svg", OUT_DIR, norm=skill_norm)

        # Skill difference map (no positive= text)
        plot_skill_diff_map(
            gdf_best,
            "delta_skill_bestml_minus_mos_pct",
            f"Best-ML vs MOS — Skill Difference ({season_label})\nmean Δskill {ov_dskill_best_minus_mos:+.2f}%",
            "ΔSkill (Best-ML − MOS) [%]",
            "map_delta_skill_bestml_vs_mos.svg", OUT_DIR
        )

        # Skill score histograms (NEW)
        plot_hist(station["skill_mos_pct"], f"MOS skill vs {RAW_NAME} (per-station)",
                  f"Skill vs {RAW_NAME} [%]", "hist_skill_mos_vs_ecmwf.svg", OUT_DIR)

        plot_hist(station["skill_best_ml_pct"], f"Best-ML skill vs {RAW_NAME} (per-station)",
                  f"Skill vs {RAW_NAME} [%]", "hist_skill_bestml_vs_ecmwf.svg", OUT_DIR)

        # ΔRMSE hists (including MOS vs ECMWF)
        plot_hist(station["delta_rmse_mos_minus_raw"], f"ΔRMSE (MOS−{RAW_NAME})",
                  f"ΔRMSE (MOS − {RAW_NAME}) [K]", f"hist_delta_rmse_mos_minus_{RAW_NAME}.svg", OUT_DIR)

        plot_hist(station["delta_rmse_mos_minus_bestml"], "ΔRMSE (MOS−Best-ML)",
                  "ΔRMSE (MOS − Best-ML) [K]", "hist_delta_rmse_mos_minus_bestml.svg", OUT_DIR)

        # ΔSkill hist
        plot_hist(station["delta_skill_bestml_minus_mos_pct"], "ΔSkill (Best-ML−MOS)",
                  "ΔSkill (Best-ML − MOS) [%]", "hist_delta_skill_bestml_minus_mos_pct.svg", OUT_DIR)

    print("Saved figures to:", OUT_DIR)


# =====================
# Main
# =====================
def main():
    if SEASONAL:
        AVAILABLE = {"2024": ["autumn"], "2025": ["winter", "spring", "summer"]}

        for year, seasons in AVAILABLE.items():
            for season in seasons:
                mos_eval = load_eval_rows_evaldir(
                    MOS_DIR,
                    pattern=f"eval_rows_{SPLIT}_MOS_{year}_{season}.parquet",
                    cols=["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, MOS_CORR],
                )
                if ML1_NAME == "EC_ML_LSTM":
                    ml1_eval = load_eval_rows_evaldir(
                        ML1_DIR,
                        pattern=f"eval_rows_analysistime_{ML1_TAG}_20??_fin.parquet",
                        cols=["SID", "validtime", "analysistime", "leadtime", RAW, OBS, ML1_CORR, "split_set"],
                    )

                    ml1_eval = ml1_eval[ml1_eval["split_set"] == "test"].copy()
                else:
                    ml1_eval = load_eval_rows_evaldir(
                        ML1_DIR,
                        pattern=f"eval_rows_{SPLIT}_{ML1_TAG}_20*.parquet",
                        cols=["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, ML1_CORR],
                    )
                ml2_eval = load_eval_rows_evaldir(
                    ML2_DIR,
                    pattern=f"eval_rows_{SPLIT}_{ML2_TAG}_20*.parquet",
                    cols=["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, ML2_CORR],
                )

                if LDT_FILTER:
                    mos_eval = mos_eval[mos_eval["leadtime"] <= MAX_LEADTIME].copy()
                    ml1_eval = ml1_eval[ml1_eval["leadtime"] <= MAX_LEADTIME].copy()
                    ml2_eval = ml2_eval[ml2_eval["leadtime"] <= MAX_LEADTIME].copy()

                    OUT_DIR = HOME / "thesis_project" / "figures" / "station_plots" / "evaluation" / f"{season}_ldt{MAX_LEADTIME}"
                else:
                    OUT_DIR = HOME / "thesis_project" / "figures" / "station_plots" / "evaluation" / f"{season}"

                OUT_DIR.mkdir(parents=True, exist_ok=True)

                run_evaluation_case(
                    mos_eval=mos_eval,
                    ml1_eval=ml1_eval,
                    ml2_eval=ml2_eval,
                    label=season,
                    OUT_DIR=OUT_DIR,
                )

    else:
        mos_eval = load_eval_rows_evaldir(
            MOS_DIR,
            pattern=f"eval_rows_{SPLIT}_MOS_20*.parquet",
            cols=["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, MOS_CORR],
        )
        if ML1_NAME == "EC_ML_LSTM":
            ml1_eval = load_eval_rows_evaldir(
                ML1_DIR,
                pattern=f"eval_rows_analysistime_{ML1_TAG}_20??_fin.parquet",
                cols=["SID", "validtime", "analysistime", "leadtime", RAW, OBS, ML1_CORR, "split_set"],
            )

            ml1_eval = ml1_eval[ml1_eval["split_set"] == "test"].copy()
        else:
            ml1_eval = load_eval_rows_evaldir(
                ML1_DIR,
                pattern=f"eval_rows_{SPLIT}_{ML1_TAG}_20*.parquet",
                cols=["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, ML1_CORR],
            )
        ml2_eval = load_eval_rows_evaldir(
            ML2_DIR,
            pattern=f"eval_rows_{SPLIT}_{ML2_TAG}_20*.parquet",
            cols=["SID", SPLIT, "analysistime", "leadtime", RAW, OBS, ML2_CORR],
        )

        if LDT_FILTER:
            mos_eval = mos_eval[mos_eval["leadtime"] <= MAX_LEADTIME].copy()
            ml1_eval = ml1_eval[ml1_eval["leadtime"] <= MAX_LEADTIME].copy()
            ml2_eval = ml2_eval[ml2_eval["leadtime"] <= MAX_LEADTIME].copy()
            OUT_DIR = HOME / "thesis_project" / "figures" / "station_plots" / "evaluation" / f"ldt{MAX_LEADTIME}"
        else:
            OUT_DIR = HOME / "thesis_project" / "figures" / "station_plots" / "evaluation" 

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        run_evaluation_case(
            mos_eval=mos_eval,
            ml1_eval=ml1_eval,
            ml2_eval=ml2_eval,
            label="",  # will fall back to time window label
            OUT_DIR=OUT_DIR,
        )


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
