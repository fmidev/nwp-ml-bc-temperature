
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm

# ------------------- Config -------------------
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

HOME = Path.home()
DATA_DIR = HOME / "thesis_project" / "data"
METRICS_DIR = HOME / "thesis_project" / "metrics"
MOS_DIR = METRICS_DIR / "mos"
ML1_DIR = METRICS_DIR / "2019_tuned_ah"
ML_2_DIR = METRICS_DIR / "full_tuned_ah"

# Station metadata + basemap
STATIONS_CSV = DATA_DIR / "stations.csv"
MAPS_DIR = DATA_DIR / "maps"
WORLD_SHP = MAPS_DIR / "ne_110m_admin_0_countries.shp"

# Map bounds
LON_MIN, LAT_MIN = -25.0, 25.5
LON_MAX, LAT_MAX =  42.0, 72.0

# Column & file patterns
SPLIT = "analysistime"        # analysistime defines window
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS_CORR = "corrected_mos"

# >>> Compare two ML models first <<<
ML1_TAG = "tuned_ah_2019"
ML2_TAG = "tuned_ah_full"

ML1_NAME = "EC_ML_XGBoost_2019"
ML2_NAME = "EC_ML_XGBoost_full"

ML1_CORR = f"corrected_{ML1_TAG}"
ML2_CORR = f"corrected_{ML2_TAG}"

# Compare mode:
#   "single" -> only MOS vs ML1 (and ML1 vs RAW)
#   "pair"   -> ML1 vs ML2, then Best-ML vs MOS (your current behavior)
COMPARE_MODE = "single"   # change to "pair" when you want both-ML comparison

SEASONAL = True

LDT_FILTER = False



# Plot colormaps
DIV_CMAP = "PuOr"  # diverging for deltas (negative = better)
SKILL_CMAP = LinearSegmentedColormap.from_list("skill_cmp", ["#570040", "white", "#005717"])
BIAS_CMAP = "coolwarm"
RMSE_CMAP = "hot_r"

# Toggle: compute normalized RMSE (nRMSE = RMSE / std(obs))
DO_NRMSE = False
# ---------------------------------------------


def rmse(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    return np.sqrt(np.mean(a * a)) if a.size else np.nan

def mae(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    return np.mean(np.abs(a)) if a.size else np.nan


def shared_diverging_norm(series_list, q=0.95, vcenter=0.0):
    """Pooled robust ±span for bias/delta maps, centered at vcenter."""
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
    """Pooled [vmin, vmax] for absolute-value maps (e.g., RMSE)."""
    vals = pd.concat([pd.to_numeric(s, errors="coerce") for s in series_list], ignore_index=True)
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None, None
    return float(np.nanpercentile(vals, lo)), float(np.nanpercentile(vals, hi))


def auto_diverging_norm(series: pd.Series, q: float = 0.95, vcenter: float = 0.0,
                        min_span: float | None = None, max_span: float | None = None) -> TwoSlopeNorm:
    """
    Build a TwoSlopeNorm from THIS series only (plot-local).
    - q: quantile for robust span (e.g., 0.95 -> ignore top/bottom 5%).
    - min_span: clamp to ensure we don’t end up with an almost flat colormap (e.g., 0.05 K).
    - max_span: clamp to avoid single extreme dominating (optional).
    """
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vals = vals.dropna()
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


def align_mos_window_join(mos_eval: pd.DataFrame,
                          ml1_eval: pd.DataFrame,
                          ml2_eval: pd.DataFrame,
                          t0: pd.Timestamp,
                          t1: pd.Timestamp) -> pd.DataFrame:
    """Use MOS to define the analysis window and samples, then inner-join ML1 & ML2."""
    # 1) Filter MOS to its own (t0, t1) coverage
    mos = mos_eval[(mos_eval[SPLIT] >= t0) & (mos_eval[SPLIT] <= t1)].copy()

    # 2) Prepare ML frames (no window cut needed; MOS keys drive the alignment)
    ml1 = ml1_eval.copy()
    ml2 = ml2_eval.copy()

    # 3) Exact sample alignment on MOS keys
    keys = ["SID", SPLIT, "validtime", "leadtime"]
    keep_cols = keys + [RAW, OBS, MOS_CORR, ML1_CORR, ML2_CORR]
    joined = (
        mos.merge(ml1[keys + [ML1_CORR]], on=keys, how="inner")
           .merge(ml2[keys + [ML2_CORR]], on=keys, how="inner")
           .dropna(subset=[OBS, RAW, MOS_CORR, ML1_CORR, ML2_CORR])
           [keep_cols]
           .copy()
    )

    if not joined.empty:
        print(f"MOS-driven window: {joined[SPLIT].min()} → {joined[SPLIT].max()}")
    else:
        print("Warning: No aligned rows after MOS-driven join.")
    return joined

def align_mos_window_join_single(mos_eval: pd.DataFrame,
                                 ml1_eval: pd.DataFrame,
                                 t0: pd.Timestamp,
                                 t1: pd.Timestamp) -> pd.DataFrame:
    """Use MOS to define the window+samples, inner-join ML1 only."""
    mos = mos_eval[(mos_eval[SPLIT] >= t0) & (mos_eval[SPLIT] <= t1)].copy()
    keys = ["SID", SPLIT, "validtime", "leadtime"]
    keep_cols = keys + [RAW, OBS, MOS_CORR, ML1_CORR]
    joined = (
        mos.merge(ml1_eval[keys + [ML1_CORR]], on=keys, how="inner")
           .dropna(subset=[OBS, RAW, MOS_CORR, ML1_CORR])
           [keep_cols].copy()
    )
    if not joined.empty:
        print(f"MOS-driven window (single-ML): {joined[SPLIT].min()} → {joined[SPLIT].max()}")
    else:
        print("Warning: No aligned rows after MOS-driven single-ML join.")
    return joined

def compute_station_metrics_single_ml_vs_mos(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-station RMSE/skill for ML1 vs MOS (same MOS-driven rows)."""
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
                "bias_raw":    np.nanmean((g[RAW]      - g[OBS]).astype(float)),
                "bias_mos":    np.nanmean((g[MOS_CORR] - g[OBS]).astype(float)),
                f"bias_{ML1_TAG}": np.nanmean((g[ML1_CORR] - g[OBS]).astype(float)),
                "n": int(len(g)),
            }),
            include_groups=False
        ).reset_index(drop=True)
    )

    st["skill_mos_pct"]        = 100.0 * (1.0 - st["mae_mos"] / (st["mae_raw"] + eps))
    st[f"skill_{ML1_TAG}_pct"] = 100.0 * (1.0 - st[f"mae_{ML1_TAG}"] / (st["mae_raw"] + eps))
    st["delta_skill_ml1_minus_mos_pct"] = st[f"skill_{ML1_TAG}_pct"] - st["skill_mos_pct"]

    # RMSE deltas you already use
    st["delta_rmse_mos_minus_ml1"] = st["rmse_mos"] - st[f"rmse_{ML1_TAG}"]
    st["delta_rmse_ml1_minus_raw"] = st[f"rmse_{ML1_TAG}"] - st["rmse_raw"]
    return st



def compute_station_metrics_two_ml(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-station RMSE + skill for ML1 vs ML2 and choose best ML per station."""
    grp = joined.groupby("SID", as_index=False)
    eps = 1e-12

    station = (
            grp.apply(
                lambda g: pd.Series({
                    # RMSE (keep for RMSE maps/hists)
                    "rmse_raw": rmse(g[OBS] - g[RAW]),
                    f"rmse_{ML1_TAG}": rmse(g[OBS] - g[ML1_CORR]),
                    f"rmse_{ML2_TAG}": rmse(g[OBS] - g[ML2_CORR]),
                    # MAE (used for skill)
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

        # === MAE-based skill (%)
    station[f"skill_{ML1_TAG}_pct"] = 100.0 * (1.0 - station[f"mae_{ML1_TAG}"] / (station["mae_raw"] + eps))
    station[f"skill_{ML2_TAG}_pct"] = 100.0 * (1.0 - station[f"mae_{ML2_TAG}"] / (station["mae_raw"] + eps))

    # Deltas in skill (positive => ML2 better than ML1)
    station["delta_skill_ML2_minus_ML1_pct"] = station[f"skill_{ML2_TAG}_pct"] - station[f"skill_{ML1_TAG}_pct"]

    # Keep your RMSE-difference too (positive => ML2 better if ML1−ML2 > 0)
    station["delta_rmse_ML1_minus_ML2"] = station[f"rmse_{ML1_TAG}"] - station[f"rmse_{ML2_TAG}"]

    # Choose best ML by RMSE (unchanged choice logic; if you want by MAE, switch the min axis)
    station["best_ml_tag"] = np.where(
        station[f"rmse_{ML2_TAG}"] < station[f"rmse_{ML1_TAG}"], ML2_TAG, ML1_TAG
    )
    station["rmse_best_ml"] = station[[f"rmse_{ML1_TAG}", f"rmse_{ML2_TAG}"]].min(axis=1)
    station["mae_best_ml"]  = station[[f"mae_{ML1_TAG}",  f"mae_{ML2_TAG}"]].min(axis=1)

    station["bias_best_ml"] = np.where(
        station["best_ml_tag"] == ML2_TAG, station[f"bias_{ML2_TAG}"], station[f"bias_{ML1_TAG}"]
    )

    return station


def compute_station_metrics_bestml_vs_mos(joined: pd.DataFrame, station_two_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Using the same aligned samples:
    1) compute MOS RMSE/MAE per station,
    2) reuse *_best_ml from station_two_ml (per-station winner),
    3) compute deltas/skills for best-ML vs MOS.
    """
    grp = joined.groupby("SID", as_index=False)

    mos_stats = (
        grp.apply(
            lambda g: pd.Series({
                "rmse_mos": rmse(g[OBS] - g[MOS_CORR]),
                "mae_mos":  mae(g[OBS] - g[MOS_CORR]),
                "rmse_raw": rmse(g[OBS] - g[RAW]),
                "mae_raw":  mae(g[OBS] - g[RAW]),
                "bias_mos":  np.nanmean((g[MOS_CORR] - g[OBS]).astype(float)),
                "bias_raw":  np.nanmean((g[RAW]      - g[OBS]).astype(float)),
            }),
            include_groups=False,
        ).reset_index(drop=True)
    )

    # Suffix *left* (station_two_ml) on overlap so MOS/aligned columns keep their names
    st = station_two_ml.merge(mos_stats, on="SID", how="left", suffixes=("_two", ""))

    eps = 1e-12

    # === MAE-based skill (%)
    st["skill_mos_pct"]     = 100.0 * (1.0 - st["mae_mos"]     / (st["mae_raw"] + eps))
    st["skill_best_ml_pct"] = 100.0 * (1.0 - st["mae_best_ml"] / (st["mae_raw"] + eps))

    # === RMSE deltas ===
    st["delta_rmse_mos_minus_bestml"] = st["rmse_mos"]    - st["rmse_best_ml"]
    st["delta_rmse_bestml_minus_raw"] = st["rmse_best_ml"] - st["rmse_raw"]

    # Positive => Best-ML has higher (better) MAE-skill than MOS
    st["delta_skill_bestml_minus_mos_pct"] = st["skill_best_ml_pct"] - st["skill_mos_pct"]

    return st


def compute_obs_std_from_joined(joined: pd.DataFrame) -> pd.DataFrame:
    obs_std = (joined.groupby("SID", as_index=False)[OBS]
                    .agg(obs_std=lambda x: np.nanstd(pd.to_numeric(x, errors="coerce")))
                    .rename(columns={OBS: "obs_std"}))
    return obs_std


def attach_stations_gdf(station_metrics: pd.DataFrame) -> gpd.GeoDataFrame:
    st = pd.read_csv(STATIONS_CSV)
    st["SID"] = st["SID"].astype(str)
    df = st.merge(station_metrics, on="SID", how="left")
    gdf = gpd.GeoDataFrame(df,
                           geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                           crs="EPSG:4326")
    return gdf


def robust_symmetric_norm(series: pd.Series, q=0.95):
    vals = pd.to_numeric(series, errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
    span = float(np.nanquantile(vals, q)) if len(vals) else 1.0
    return TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)


def plot_delta_map(gdf: gpd.GeoDataFrame, column: str, title: str, legend_label: str,
                   out_name: str, OUT_DIR: Path, cmap: str | None = None, norm=None):
    world = gpd.read_file(WORLD_SHP)
    if norm is None:
        # fallback: robust ± symmetric
        vals = pd.to_numeric(gdf[column], errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
        span = float(np.nanquantile(vals, 0.95)) if len(vals) else 1.0
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax, column=column, cmap=(cmap or DIV_CMAP), norm=norm,
        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
        legend=True, legend_kwds={"label": legend_label, "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, bbox_inches="tight"); plt.close()


def plot_bias_map(gdf: gpd.GeoDataFrame, column: str, title: str,
                  out_name: str, OUT_DIR: Path, cmap: str | None = None, norm=None):
    """Mean bias (model − obs). Positive = overestimation."""
    world = gpd.read_file(WORLD_SHP)
    if norm is None:
        # fallback: robust ± symmetric centered at 0
        vals = pd.to_numeric(gdf[column], errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
        span = float(np.nanquantile(vals, 0.95)) if len(vals) else 1.0
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax, column=column, cmap=(cmap or BIAS_CMAP), norm=norm,
        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
        legend=True, legend_kwds={"label": "Mean bias (Model − Obs) [K]", "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, bbox_inches="tight"); plt.close()


def plot_abs_rmse_map(gdf: gpd.GeoDataFrame, column: str, title: str, legend_label: str,
                      out_name: str, OUT_DIR: Path, cmap: str | None = None, vmin=None, vmax=None):
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
    ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, bbox_inches="tight"); plt.close()


def plot_hist(series: pd.Series, title: str, xlabel: str,
              out_name: str, OUT_DIR: Path, bins=25, fixed_bins: np.ndarray | None = None):
    plt.figure(figsize=(6, 4))
    data = pd.to_numeric(series, errors="coerce").dropna()
    plt.hist(data, bins=(fixed_bins if fixed_bins is not None else bins), color="#3B9AB2", edgecolor="black")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel(xlabel); plt.ylabel("Stations"); plt.title(title)
    plt.tight_layout(); plt.savefig(OUT_DIR / out_name, dpi=150); plt.close()


def main():

    if SEASONAL:

        AVAILABLE = {
            "2024": ["autumn"],
            "2025": ["winter", "spring", "summer"],
        }

        for year, seasons in AVAILABLE.items():
            for season in seasons:

                # Load
                mos_eval = load_eval_rows_evaldir(
                    MOS_DIR,
                    pattern=f"eval_rows_{SPLIT}_MOS_{year}_{season}.parquet",
                    cols=["SID", SPLIT, "validtime", "leadtime", RAW, OBS, MOS_CORR],
                )
                ml1_eval = load_eval_rows_evaldir(
                    ML1_DIR,
                    pattern=f"eval_rows_{SPLIT}_{ML1_TAG}_20*.parquet",
                    cols=["SID", SPLIT, "validtime", "leadtime", RAW, OBS, ML1_CORR],
                )
                ml2_eval = load_eval_rows_evaldir(
                    ML_2_DIR,
                    pattern=f"eval_rows_{SPLIT}_{ML2_TAG}_20*.parquet",
                    cols=["SID", SPLIT, "validtime", "leadtime", RAW, OBS, ML2_CORR],
                )

                if LDT_FILTER:
                        # --- Filter to leadtime == 24 only ---
                        mos_eval  = mos_eval[mos_eval["leadtime"] <= 24].copy()
                        ml1_eval  = ml1_eval[ml1_eval["leadtime"] <= 24].copy()
                        ml2_eval  = ml2_eval[ml2_eval["leadtime"] <= 24].copy()
                        print(f"Filtered to leadtime <= 24: MOS={len(mos_eval)}, ML1={len(ml1_eval)}, ML2={len(ml2_eval)}")
                        OUT_DIR = HOME / "thesis_project" / "figures" / "station_plots" / "evaluation" / "2019" / f"{season}_ldt24"
                        OUT_DIR.mkdir(parents=True, exist_ok=True)
                else:
                    OUT_DIR = HOME / "thesis_project" / "figures" / "station_plots" / "evaluation" / "2019" / f"{season}"
                    OUT_DIR.mkdir(parents=True, exist_ok=True)

                
                # Derive window from MOS (non-null MOS/RAW/OBS)
                t0, t1 = mos_coverage_window(mos_eval)
                print(f"MOS coverage window: {t0} → {t1}")
                window = f"{t0:%b %Y} – {t1:%b %Y}"

                if COMPARE_MODE == "single":
                    # ======= MOS vs ML1 only =======
                    joined = align_mos_window_join_single(mos_eval, ml1_eval, t0, t1)
                    print("Rows after alignment (single):", len(joined))
                    print("Stations after alignment (single):", joined["SID"].nunique())

                    station_single = compute_station_metrics_single_ml_vs_mos(joined)

                    """# Optional nRMSE
                    if DO_NRMSE:
                        obs_std = compute_obs_std_from_joined(joined)
                        station_single = station_single.merge(obs_std, on="SID", how="left")
                        station_single["nrmse_raw"] = station_single["rmse_raw"] / station_single["obs_std"]
                        station_single["nrmse_mos"] = station_single["rmse_mos"] / station_single["obs_std"]
                        station_single[f"nrmse_{ML1_TAG}"] = station_single[f"rmse_{ML1_TAG}"] / station_single["obs_std"]
                        station_single["delta_nrmse_mos_minus_ml1"] = station_single["nrmse_mos"] - station_single[f"nrmse_{ML1_TAG}"]"""

                    # Summaries
                    print(f"\n=== Station-level: {ML1_NAME} vs MOS (MOS-driven {season}) ===")
                    print(f"Stations total: {len(station_single)}")
                    ml1_better = (station_single["delta_rmse_mos_minus_ml1"] > 0).sum()
                    mos_better = (station_single["delta_rmse_mos_minus_ml1"] < 0).sum()
                    print(f"{ML1_NAME} better (RMSE): {ml1_better} ({ml1_better/len(station_single)*100:.1f}%)")
                    print(f"MOS better (RMSE):   {mos_better} ({mos_better/len(station_single)*100:.1f}%)")
                    print(f"Mean ΔRMSE (MOS−{ML1_NAME}) [K]: {station_single['delta_rmse_mos_minus_ml1'].mean():+.3f}")
                    print(f"Median ΔRMSE [K]:             {station_single['delta_rmse_mos_minus_ml1'].median():+.3f}")

                    # Plots: use only ML1
                    gdf_single = attach_stations_gdf(station_single)

                    # --- Shared scales for comparability (SINGLE mode) ---
                    # 1) Absolute RMSE: share [vmin, vmax] across RAW / MOS / ML1
                    rmse_vmin, rmse_vmax = shared_sequential_limits([
                        gdf_single["rmse_raw"], gdf_single["rmse_mos"], gdf_single[f"rmse_{ML1_TAG}"]
                    ])

                    # 2) Bias: share across RAW / MOS / ML1
                    bias_norm = shared_diverging_norm([
                        gdf_single.get("bias_raw", pd.Series(dtype=float)),
                        gdf_single.get("bias_mos", pd.Series(dtype=float)),
                        gdf_single.get(f"bias_{ML1_TAG}", pd.Series(dtype=float)),
                    ])

                    # Histograms
                    plot_hist(station_single["delta_rmse_mos_minus_ml1"],
                            f"ΔRMSE (MOS−{ML1_NAME}) — positive = {ML1_NAME} better",
                            f"ΔRMSE (MOS − {ML1_NAME}) [K]",
                            f"hist_delta_rmse_mos_minus_{ML1_NAME}.png", OUT_DIR)
                    plot_hist(station_single["delta_skill_ml1_minus_mos_pct"],
                            f"ΔSkill ({ML1_NAME}−MOS) — positive = {ML1_NAME} better",
                            f"ΔSkill ({ML1_NAME} − MOS) [%]",
                            f"hist_delta_skill_{ML1_NAME}_minus_mos.png", OUT_DIR)

                    # Absolute RMSE maps (use shared vmin/vmax)
                    plot_abs_rmse_map(gdf_single, "rmse_raw", f"RAW RMSE ({season})", "RMSE (RAW) [K]", "map_rmse_raw.png", OUT_DIR,
                                    vmin=rmse_vmin, vmax=rmse_vmax)
                    plot_abs_rmse_map(gdf_single, "rmse_mos", f"MOS RMSE ({season})", "RMSE (MOS) [K]", "map_rmse_mos.png", OUT_DIR,
                                    vmin=rmse_vmin, vmax=rmse_vmax)
                    plot_abs_rmse_map(gdf_single, f"rmse_{ML1_TAG}", f"{ML1_NAME} RMSE ({season})",
                                    f"RMSE ({ML1_NAME}) [K]", f"map_rmse_{ML1_NAME}.png", OUT_DIR,
                                    vmin=rmse_vmin, vmax=rmse_vmax)

                    # Per-plot autoscale for each delta map
                    plot_delta_map(
                        gdf_single, "delta_rmse_mos_minus_ml1",
                        f"MOS vs {ML1_NAME} — ΔRMSE ({season})\n(positive = {ML1_NAME} better)",
                        f"ΔRMSE (MOS − {ML1_NAME}) [K]",
                        f"map_delta_rmse_mos_vs_{ML1_NAME}.png", OUT_DIR,
                        # <<< change is here:
                        norm=auto_diverging_norm(gdf_single["delta_rmse_mos_minus_ml1"], q=0.95, min_span=0.05)
                    )

                    plot_delta_map(
                        gdf_single, "delta_rmse_ml1_minus_raw",
                        f"{ML1_NAME} vs RAW — ΔRMSE ({season})\n(negative = {ML1_NAME} better)",
                        f"ΔRMSE ({ML1_NAME} − RAW) [K]",
                        f"map_delta_rmse_{ML1_NAME}_vs_raw.png", OUT_DIR,
                        # <<< and here:
                        norm=auto_diverging_norm(gdf_single["delta_rmse_ml1_minus_raw"], q=0.95, min_span=0.05)
                    )


                    # Bias maps (use shared diverging norm; set a diverging cmap if you want, e.g. "RdBu_r")
                    plot_bias_map(gdf_single, f"bias_{ML1_TAG}",
                                f"{ML1_NAME} mean bias ({season}) — positive = overestimate",
                                f"map_bias_{ML1_NAME}.png", OUT_DIR, cmap="RdBu_r", norm=bias_norm)
                    plot_bias_map(gdf_single, "bias_mos",
                                f"MOS mean bias ({season}) — positive = overestimate",
                                "map_bias_mos.png", OUT_DIR, cmap="RdBu_r", norm=bias_norm)


                    # Skill% map
                    norm = robust_symmetric_norm(gdf_single["delta_skill_ml1_minus_mos_pct"])
                    world = gpd.read_file(WORLD_SHP)
                    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
                    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
                    gdf_single.plot(
                        ax=ax, column="delta_skill_ml1_minus_mos_pct", cmap=SKILL_CMAP, norm=norm,
                        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
                        legend=True, legend_kwds={"label": f"ΔSkill ({ML1_NAME} − MOS) [%]", "orientation": "horizontal", "shrink": 0.7},
                    )
                    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
                    ax.set_title(f"{ML1_NAME} vs MOS — Skill Difference ({season})\n(positive = {ML1_NAME} better)")
                    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    plt.tight_layout(); plt.savefig(OUT_DIR / f"map_delta_skill_{ML1_NAME}_vs_mos.png", bbox_inches="tight"); plt.close()

                    plot_hist(station_single[f"bias_{ML1_TAG}"],
                            f"{ML1_NAME} mean bias",
                            "Mean bias (Model − Obs) [K]",
                            f"hist_bias_{ML1_NAME}.png", OUT_DIR)

                    plot_hist(station_single["bias_mos"],
                            "MOS mean bias",
                            "Mean bias (Model − Obs) [K]",
                            "hist_bias_mos.png", OUT_DIR)

                else:
                    # ======= Your existing “pair” pipeline (ML1 vs ML2, Best-ML vs MOS) =======

                    # Align everyone to MOS rows in that window
                    joined = align_mos_window_join(mos_eval, ml1_eval, ml2_eval, t0, t1)
                    print("Rows after alignment:", len(joined))
                    print("Stations after alignment:", joined["SID"].nunique())

                    # 3) Compare ML1 vs ML2 first (per-station), pick the best ML  [MOS-driven: from `joined`]
                    station_two_ml = compute_station_metrics_two_ml(joined)

                    window = f"{t0:%b %Y} – {t1:%b %Y}"
                    print(f"\n=== Station-level: ML1 vs ML2 (MOS-driven window {season}) ===")
                    if len(station_two_ml) == 0:
                        print("No stations after MOS-driven alignment.")
                    else:
                        ml2_better = (station_two_ml["delta_rmse_ML1_minus_ML2"] > 0).sum()
                        ml1_better = (station_two_ml["delta_rmse_ML1_minus_ML2"] < 0).sum()
                        print(f"Stations total: {len(station_two_ml)}")
                        print(f"{ML2_NAME} better (RMSE): {ml2_better} ({ml2_better/len(station_two_ml)*100:.1f}%)")
                        print(f"{ML1_NAME} better (RMSE): {ml1_better} ({ml1_better/len(station_two_ml)*100:.1f}%)")
                        print(f"Mean ΔRMSE ({ML1_NAME}−{ML2_TAG}) [K]: {station_two_ml['delta_rmse_ML1_minus_ML2'].mean():+.3f}")
                        print(f"Median ΔRMSE [K]: {station_two_ml['delta_rmse_ML1_minus_ML2'].median():+.3f}")
                        print(f"Mean ΔSkill ({ML2_NAME}−{ML1_NAME}) [%]: {station_two_ml['delta_skill_ML2_minus_ML1_pct'].mean():+.2f}")
                        print(f"Median ΔSkill [%]: {station_two_ml['delta_skill_ML2_minus_ML1_pct'].median():+.2f}")

                    # 4) Now compare the per-station BEST-ML vs MOS, still on the same exact rows
                    station = compute_station_metrics_bestml_vs_mos(joined, station_two_ml)

                    # --- Ensure keys line up for the merge
                    station["SID"] = station["SID"].astype(str)
                    station_two_ml["SID"] = station_two_ml["SID"].astype(str)

                    # --- Sanity: make sure ML↔ML columns exist in station_two_ml
                    required_cols = ["SID", "delta_rmse_ML1_minus_ML2", "delta_skill_ML2_minus_ML1_pct"]
                    missing = [c for c in required_cols if c not in station_two_ml.columns]
                    if missing:
                        raise KeyError(f"station_two_ml is missing columns: {missing}. "
                                    f"Available: {list(station_two_ml.columns)}")

                    # --- Merge ML↔ML deltas into `station`
                    station = station.merge(
                        station_two_ml[required_cols + [f"rmse_{ML1_NAME}", f"rmse_{ML2_NAME}"]],
                        on="SID",
                        how="left",
                        validate="one_to_one",
                    )

                    # --- Final guard before plotting
                    for need in ["delta_rmse_ML1_minus_ML2", "delta_skill_ML2_minus_ML1_pct"]:
                        if need not in station.columns:
                            # Fallback: use station_two_ml directly for ML↔ML plots/hists
                            print(f"Warning: '{need}' not found in 'station' after merge; "
                                f"will plot ML↔ML figures from 'station_two_ml'.")


                    # 5) Optional: normalized RMSE via obs std on the same aligned set
                    if DO_NRMSE:
                        obs_std = compute_obs_std_from_joined(joined)  # SID, obs_std
                        station = station.merge(obs_std, on="SID", how="left")
                        station["nrmse_raw"] = station["rmse_raw"] / station["obs_std"]
                        station["nrmse_mos"] = station["rmse_mos"] / station["obs_std"]
                        station["nrmse_best_ml"] = station["rmse_best_ml"] / station["obs_std"]
                        station["delta_nrmse_mos_minus_bestml"] = station["nrmse_mos"] - station["nrmse_best_ml"]

                    # 6) Save numeric table (optional)
                    # station.to_csv(OUT_DIR / f"station_metrics_bestml_vs_mos_{t0:%Y%m%d}_{t1:%Y%m%d}.csv", index=False)

                    # 7) Summaries for BEST-ML vs MOS
                    print(f"\n=== Station-level: Best-ML vs MOS (MOS-driven window {season}) ===")
                    if len(station) == 0:
                        print("No stations after MOS-driven alignment.")
                    else:
                        ml_better = (station["delta_rmse_mos_minus_bestml"] > 0).sum()
                        mos_better = (station["delta_rmse_mos_minus_bestml"] < 0).sum()
                        print(f"Stations total: {len(station)}")
                        print(f"Best-ML better (RMSE): {ml_better} ({ml_better/len(station)*100:.1f}%)")
                        print(f"MOS better (RMSE):    {mos_better} ({mos_better/len(station)*100:.1f}%)")
                        print(f"Mean ΔRMSE (MOS−BestML) [K]: {station['delta_rmse_mos_minus_bestml'].mean():+.3f}")
                        print(f"Median ΔRMSE [K]:           {station['delta_rmse_mos_minus_bestml'].median():+.3f}")
                        print(f"Mean ΔSkill (BestML−MOS) [%]: {station['delta_skill_bestml_minus_mos_pct'].mean():+.2f}")
                        print(f"Median ΔSkill [%]:            {station['delta_skill_bestml_minus_mos_pct'].median():+.2f}")

                        # --- Build two GeoDataFrames: one for ML↔ML, one for Best-ML↔MOS ---
                    gdf_ml   = attach_stations_gdf(station_two_ml)  # has rmse_{ML1_NAME}, rmse_{ML2_NAME}, deltas
                    gdf_best = attach_stations_gdf(station)         # has rmse_raw, rmse_mos, rmse_best_ml, deltas vs MOS


                    # --- Shared scales for comparability (PAIRED mode) ---
                    # 1) Absolute RMSE across RAW, MOS, Best-ML, and the individual MLs
                    rmse_vmin, rmse_vmax = shared_sequential_limits([
                        gdf_best["rmse_raw"], gdf_best["rmse_mos"], gdf_best["rmse_best_ml"],
                        gdf_ml[f"rmse_{ML1_TAG}"], gdf_ml[f"rmse_{ML2_TAG}"],
                    ])

                    # 2) Bias across Best-ML, MOS, RAW, and the two MLs
                    bias_norm = shared_diverging_norm([
                        gdf_best.get("bias_best_ml", pd.Series(dtype=float)),
                        gdf_best.get("bias_mos", pd.Series(dtype=float)),
                        gdf_best.get("bias_raw", pd.Series(dtype=float)),
                        gdf_ml.get(f"bias_{ML1_TAG}", pd.Series(dtype=float)),
                        gdf_ml.get(f"bias_{ML2_TAG}", pd.Series(dtype=float)),
                    ])


                    # --- Histograms ---
                    # ML1 vs ML2 (use station_two_ml)
                    plot_hist(
                        station_two_ml["delta_rmse_ML1_minus_ML2"],
                        f"ΔRMSE ({ML1_NAME}−{ML2_NAME}) — positive = {ML2_NAME} better",
                        f"ΔRMSE ({ML1_NAME}−{ML2_NAME}) [K]",
                        f"hist_delta_rmse_{ML1_NAME}_minus_{ML2_NAME}.png", OUT_DIR
                    )
                    plot_hist(
                        station_two_ml["delta_skill_ML2_minus_ML1_pct"],
                        f"ΔSkill ({ML2_NAME}−{ML1_NAME}) — positive = {ML2_NAME} better",
                        f"ΔSkill ({ML2_NAME}−{ML1_NAME}) [%]",
                        f"hist_delta_skill_{ML2_NAME}_minus_{ML1_NAME}.png", OUT_DIR
                    )

                    # Best-ML vs MOS (use station)
                    plot_hist(
                        station["delta_rmse_mos_minus_bestml"],
                        "ΔRMSE (MOS−Best-ML) — positive = Best-ML better",
                        "ΔRMSE (MOS − Best-ML) [K]",
                        "hist_delta_rmse_mos_minus_bestml.png", OUT_DIR
                    )
                    plot_hist(
                        station["delta_skill_bestml_minus_mos_pct"],
                        "ΔSkill (Best-ML−MOS) — positive = Best-ML better",
                        "ΔSkill (Best-ML − MOS) [%]",
                        "hist_delta_skill_bestml_minus_mos_pct.png", OUT_DIR
                    )

                    # Absolute RMSE (shared vmin/vmax)
                    plot_abs_rmse_map(gdf_best, "rmse_raw",     f"RAW RMSE ({season})",      "RMSE (RAW) [K]",      "map_rmse_raw.png",     OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)
                    plot_abs_rmse_map(gdf_best, "rmse_mos",     f"MOS RMSE ({season})",      "RMSE (MOS) [K]",      "map_rmse_mos.png",     OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)
                    plot_abs_rmse_map(gdf_best, "rmse_best_ml", f"Best-ML RMSE ({season})",  "RMSE (Best-ML) [K]",  "map_rmse_best_ml.png", OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)
                    plot_abs_rmse_map(gdf_ml,   f"rmse_{ML1_TAG}", f"{ML1_NAME} RMSE ({season})", "RMSE ({ML1_NAME}) [K]", f"map_rmse_{ML1_NAME}.png", OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)
                    plot_abs_rmse_map(gdf_ml,   f"rmse_{ML2_TAG}", f"{ML2_NAME} RMSE ({season})", "RMSE ({ML2_NAME}) [K]", f"map_rmse_{ML2_NAME}.png", OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)

                    # ML1 vs ML2 delta: autoscale to THIS map
                    plot_delta_map(
                        gdf_ml, "delta_rmse_ML1_minus_ML2",
                        title=f"{ML1_NAME} vs {ML2_NAME} — ΔRMSE ({season})\n(positive = {ML2_NAME} better)",
                        legend_label=f"ΔRMSE ({ML1_NAME} − {ML2_NAME}) [K]",
                        out_name=f"map_delta_rmse_{ML1_NAME}_vs_{ML2_NAME}.png",
                        OUT_DIR=OUT_DIR,
                        # <<< per-plot norm
                        norm=auto_diverging_norm(gdf_ml["delta_rmse_ML1_minus_ML2"], q=0.95, min_span=0.05)
                    )

                    # MOS vs Best-ML delta: autoscale
                    plot_delta_map(
                        gdf_best, "delta_rmse_mos_minus_bestml",
                        title=f"MOS vs Best-ML — ΔRMSE ({season})\n(positive = Best-ML better)",
                        legend_label="ΔRMSE (MOS − Best-ML) [K]",
                        out_name="map_delta_rmse_mos_vs_bestml.png",
                        OUT_DIR=OUT_DIR,
                        # <<< per-plot norm
                        norm=auto_diverging_norm(gdf_best["delta_rmse_mos_minus_bestml"], q=0.95, min_span=0.05)
                    )

                    # Best-ML vs RAW delta: autoscale
                    plot_delta_map(
                        gdf_best, "delta_rmse_bestml_minus_raw",
                        title=f"Best-ML vs RAW — ΔRMSE ({season})\n(negative = Best-ML better)",
                        legend_label="ΔRMSE (Best-ML − RAW) [K]",
                        out_name="map_delta_rmse_bestml_vs_raw.png",
                        OUT_DIR=OUT_DIR,
                        # <<< per-plot norm
                        norm=auto_diverging_norm(gdf_best["delta_rmse_bestml_minus_raw"], q=0.95, min_span=0.05)
                    )


                    # Bias maps (shared bias_norm; pick diverging cmap if desired)
                    plot_bias_map(gdf_best, "bias_best_ml",
                                f"Best-ML mean bias ({season}) — positive = overestimate",
                                "map_bias_best_ml.png", OUT_DIR, cmap="RdBu_r", norm=bias_norm)
                    plot_bias_map(gdf_best, "bias_mos",
                                f"MOS mean bias ({season}) — positive = overestimate",
                                "map_bias_mos.png", OUT_DIR, cmap="RdBu_r", norm=bias_norm)


                    # Skill% map (Best-ML − MOS)
                    norm = robust_symmetric_norm(gdf_best["delta_skill_bestml_minus_mos_pct"])
                    world = gpd.read_file(WORLD_SHP)
                    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
                    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
                    gdf_best.plot(
                        ax=ax, column="delta_skill_bestml_minus_mos_pct",
                        cmap=SKILL_CMAP, norm=norm,
                        markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
                        legend=True, legend_kwds={"label": "ΔSkill (Best-ML − MOS) [%]", "orientation": "horizontal", "shrink": 0.7},
                    )
                    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
                    ax.set_title(f"Best-ML vs MOS — Skill Difference ({season})\n(positive = Best-ML better)")
                    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    plt.tight_layout(); plt.savefig(OUT_DIR / "map_delta_skill_bestml_vs_mos.png", bbox_inches="tight"); plt.close()

                    plot_hist(station["bias_best_ml"],
                            "Best-ML mean bias",
                            "Mean bias (Model − Obs) [K]",
                            "hist_bias_best_ml.png", OUT_DIR)

                    plot_hist(station["bias_mos"],
                            "MOS mean bias",
                            "Mean bias (Model − Obs) [K]",
                            "hist_bias_mos.png", OUT_DIR)

                # Optional: nRMSE delta map (still from gdf_best)
                """if DO_NRMSE and "delta_nrmse_mos_minus_bestml" in gdf_best.columns:
                    plot_delta_map(
                        gdf_best,
                        column="delta_nrmse_mos_minus_bestml",
                        title=f"MOS vs Best-ML — ΔnRMSE ({season})\n(positive = Best-ML better)",
                        legend_label="ΔnRMSE (MOS − Best-ML) [σ units]",
                        out_name="map_delta_nrmse_mos_vs_bestml.png",
                    )"""

                print("Saved figures to:", OUT_DIR)

    else:
        # Load
        mos_eval = load_eval_rows_evaldir(
            MOS_DIR,
            pattern=f"eval_rows_{SPLIT}_MOS_20*.parquet",
            cols=["SID", SPLIT, "validtime", "leadtime", RAW, OBS, MOS_CORR],
        )
        ml1_eval = load_eval_rows_evaldir(
            ML1_DIR,
            pattern=f"eval_rows_{SPLIT}_{ML1_TAG}_20*.parquet",
            cols=["SID", SPLIT, "validtime", "leadtime", RAW, OBS, ML1_CORR],
        )
        ml2_eval = load_eval_rows_evaldir(
            ML_2_DIR,
            pattern=f"eval_rows_{SPLIT}_{ML2_TAG}_20*.parquet",
            cols=["SID", SPLIT, "validtime", "leadtime", RAW, OBS, ML2_CORR],
        )

        if LDT_FILTER:
            # --- Filter to leadtime == 24 only ---
            mos_eval  = mos_eval[mos_eval["leadtime"] <= 24].copy()
            ml1_eval  = ml1_eval[ml1_eval["leadtime"] <= 24].copy()
            ml2_eval  = ml2_eval[ml2_eval["leadtime"] <= 24].copy()
            print(f"Filtered to leadtime <= 24: MOS={len(mos_eval)}, ML1={len(ml1_eval)}, ML2={len(ml2_eval)}")
            OUT_DIR = HOME / "thesis_project" / "figures" / "station_plots" / "evaluation" / "2019" / "ldt24"
            OUT_DIR.mkdir(parents=True, exist_ok=True)
        else:
            OUT_DIR = HOME / "thesis_project" / "figures" / "station_plots" / "evaluation" / "2019" 
            OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Derive window from MOS (non-null MOS/RAW/OBS)
        t0, t1 = mos_coverage_window(mos_eval)
        print(f"MOS coverage window: {t0} → {t1}")
        season = f"{t0:%b %Y} – {t1:%b %Y}"

        if COMPARE_MODE == "single":
            # ======= MOS vs ML1 only =======
            joined = align_mos_window_join_single(mos_eval, ml1_eval, t0, t1)
            print("Rows after alignment (single):", len(joined))
            print("Stations after alignment (single):", joined["SID"].nunique())

            station_single = compute_station_metrics_single_ml_vs_mos(joined)

            """# Optional nRMSE
            if DO_NRMSE:
                obs_std = compute_obs_std_from_joined(joined)
                station_single = station_single.merge(obs_std, on="SID", how="left")
                station_single["nrmse_raw"] = station_single["rmse_raw"] / station_single["obs_std"]
                station_single["nrmse_mos"] = station_single["rmse_mos"] / station_single["obs_std"]
                station_single[f"nrmse_{ML1_NAME}"] = station_single[f"rmse_{ML1_NAME}"] / station_single["obs_std"]
                station_single["delta_nrmse_mos_minus_ml1"] = station_single["nrmse_mos"] - station_single[f"nrmse_{ML1_NAME}"]"""

            # Summaries
            print(f"\n=== Station-level: {ML1_NAME} vs MOS (MOS-driven {season}) ===")
            print(f"Stations total: {len(station_single)}")
            ml1_better = (station_single["delta_rmse_mos_minus_ml1"] > 0).sum()
            mos_better = (station_single["delta_rmse_mos_minus_ml1"] < 0).sum()
            print(f"{ML1_NAME} better (RMSE): {ml1_better} ({ml1_better/len(station_single)*100:.1f}%)")
            print(f"MOS better (RMSE):   {mos_better} ({mos_better/len(station_single)*100:.1f}%)")
            print(f"Mean ΔRMSE (MOS−{ML1_NAME}) [K]: {station_single['delta_rmse_mos_minus_ml1'].mean():+.3f}")
            print(f"Median ΔRMSE [K]:             {station_single['delta_rmse_mos_minus_ml1'].median():+.3f}")

            # Plots: use only ML1
            gdf_single = attach_stations_gdf(station_single)

            # --- Shared scales for comparability (SINGLE mode) ---
            # 1) Absolute RMSE: share [vmin, vmax] across RAW / MOS / ML1
            rmse_vmin, rmse_vmax = shared_sequential_limits([
                gdf_single["rmse_raw"], gdf_single["rmse_mos"], gdf_single[f"rmse_{ML1_TAG}"]
            ])

            # 2) Bias: share across RAW / MOS / ML1
            bias_norm = shared_diverging_norm([
                gdf_single.get("bias_raw", pd.Series(dtype=float)),
                gdf_single.get("bias_mos", pd.Series(dtype=float)),
                gdf_single.get(f"bias_{ML1_TAG}", pd.Series(dtype=float)),
            ])

            # Histograms
            plot_hist(station_single["delta_rmse_mos_minus_ml1"],
                    f"ΔRMSE (MOS−{ML1_NAME}) — positive = {ML1_NAME} better",
                    f"ΔRMSE (MOS − {ML1_NAME}) [K]",
                    f"hist_delta_rmse_mos_minus_{ML1_NAME}.png", OUT_DIR)
            plot_hist(station_single["delta_skill_ml1_minus_mos_pct"],
                    f"ΔSkill ({ML1_NAME}−MOS) — positive = {ML1_NAME} better",
                    f"ΔSkill ({ML1_NAME} − MOS) [%]",
                    f"hist_delta_skill_{ML1_NAME}_minus_mos.png", OUT_DIR)

            # Absolute RMSE maps (use shared vmin/vmax)
            plot_abs_rmse_map(gdf_single, "rmse_raw", f"RAW RMSE ({season})", "RMSE (RAW) [K]", "map_rmse_raw.png", OUT_DIR,
                            vmin=rmse_vmin, vmax=rmse_vmax)
            plot_abs_rmse_map(gdf_single, "rmse_mos", f"MOS RMSE ({season})", "RMSE (MOS) [K]", "map_rmse_mos.png", OUT_DIR,
                            vmin=rmse_vmin, vmax=rmse_vmax)
            plot_abs_rmse_map(gdf_single, f"rmse_{ML1_TAG}", f"{ML1_NAME} RMSE ({season})",
                            f"RMSE ({ML1_NAME}) [K]", f"map_rmse_{ML1_NAME}.png", OUT_DIR,
                            vmin=rmse_vmin, vmax=rmse_vmax)

            # MOS vs Best-ML delta: autoscale
            plot_delta_map(
                gdf_best, "delta_rmse_mos_minus_bestml",
                title=f"MOS vs Best-ML — ΔRMSE ({season})\n(positive = Best-ML better)",
                legend_label="ΔRMSE (MOS − Best-ML) [K]",
                out_name="map_delta_rmse_mos_vs_bestml.png",
                OUT_DIR=OUT_DIR,
                # <<< per-plot norm
                norm=auto_diverging_norm(gdf_best["delta_rmse_mos_minus_bestml"], q=0.95, min_span=0.05)
            )

            # Best-ML vs RAW delta: autoscale
            plot_delta_map(
                gdf_best, "delta_rmse_bestml_minus_raw",
                title=f"Best-ML vs RAW — ΔRMSE ({season})\n(negative = Best-ML better)",
                legend_label="ΔRMSE (Best-ML − RAW) [K]",
                out_name="map_delta_rmse_bestml_vs_raw.png",
                OUT_DIR=OUT_DIR,
                # <<< per-plot norm
                norm=auto_diverging_norm(gdf_best["delta_rmse_bestml_minus_raw"], q=0.95, min_span=0.05)
            )

            # Bias maps (use shared diverging norm; set a diverging cmap if you want, e.g. "RdBu_r")
            plot_bias_map(gdf_single, f"bias_{ML1_TAG}",
                        f"{ML1_NAME} mean bias ({season}) — positive = overestimate",
                        f"map_bias_{ML1_NAME}.png", OUT_DIR, cmap="RdBu_r", norm=bias_norm)
            plot_bias_map(gdf_single, "bias_mos",
                        f"MOS mean bias ({season}) — positive = overestimate",
                        "map_bias_mos.png", OUT_DIR, cmap="RdBu_r", norm=bias_norm)

            # Skill% map
            norm = robust_symmetric_norm(gdf_single["delta_skill_ml1_minus_mos_pct"])
            world = gpd.read_file(WORLD_SHP)
            fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
            world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
            gdf_single.plot(
                ax=ax, column="delta_skill_ml1_minus_mos_pct", cmap=SKILL_CMAP, norm=norm,
                markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
                legend=True, legend_kwds={"label": f"ΔSkill ({ML1_NAME} − MOS) [%]", "orientation": "horizontal", "shrink": 0.7},
            )
            ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
            ax.set_title(f"{ML1_NAME} vs MOS — Skill Difference ({season})\n(positive = {ML1_NAME} better)")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout(); plt.savefig(OUT_DIR / f"map_delta_skill_{ML1_NAME}_vs_mos.png", bbox_inches="tight"); plt.close()


            plot_hist(station_single[f"bias_{ML1_TAG}"],
                    f"{ML1_NAME} mean bias",
                    "Mean bias (Model − Obs) [K]",
                    f"hist_bias_{ML1_NAME}.png", OUT_DIR)

            plot_hist(station_single["bias_mos"],
                    "MOS mean bias",
                    "Mean bias (Model − Obs) [K]",
                    "hist_bias_mos.png", OUT_DIR)

        else:
            # ======= Your existing “pair” pipeline (ML1 vs ML2, Best-ML vs MOS) =======

            # Align everyone to MOS rows in that window
            joined = align_mos_window_join(mos_eval, ml1_eval, ml2_eval, t0, t1)
            print("Rows after alignment:", len(joined))
            print("Stations after alignment:", joined["SID"].nunique())

            # 3) Compare ML1 vs ML2 first (per-station), pick the best ML  [MOS-driven: from `joined`]
            station_two_ml = compute_station_metrics_two_ml(joined)

            season = f"{t0:%b %Y} – {t1:%b %Y}"
            print(f"\n=== Station-level: ML1 vs ML2 (MOS-driven window {season}) ===")
            if len(station_two_ml) == 0:
                print("No stations after MOS-driven alignment.")
            else:
                ml2_better = (station_two_ml["delta_rmse_ML1_minus_ML2"] > 0).sum()
                ml1_better = (station_two_ml["delta_rmse_ML1_minus_ML2"] < 0).sum()
                print(f"Stations total: {len(station_two_ml)}")
                print(f"{ML2_NAME} better (RMSE): {ml2_better} ({ml2_better/len(station_two_ml)*100:.1f}%)")
                print(f"{ML1_NAME} better (RMSE): {ml1_better} ({ml1_better/len(station_two_ml)*100:.1f}%)")
                print(f"Mean ΔRMSE ({ML1_NAME}−{ML2_NAME}) [K]: {station_two_ml['delta_rmse_ML1_minus_ML2'].mean():+.3f}")
                print(f"Median ΔRMSE [K]: {station_two_ml['delta_rmse_ML1_minus_ML2'].median():+.3f}")
                print(f"Mean ΔSkill ({ML2_NAME}−{ML1_NAME}) [%]: {station_two_ml['delta_skill_ML2_minus_ML1_pct'].mean():+.2f}")
                print(f"Median ΔSkill [%]: {station_two_ml['delta_skill_ML2_minus_ML1_pct'].median():+.2f}")

            # 4) Now compare the per-station BEST-ML vs MOS, still on the same exact rows
            station = compute_station_metrics_bestml_vs_mos(joined, station_two_ml)

            # --- Ensure keys line up for the merge
            station["SID"] = station["SID"].astype(str)
            station_two_ml["SID"] = station_two_ml["SID"].astype(str)

            # --- Sanity: make sure ML↔ML columns exist in station_two_ml
            required_cols = ["SID", "delta_rmse_ML1_minus_ML2", "delta_skill_ML2_minus_ML1_pct"]
            missing = [c for c in required_cols if c not in station_two_ml.columns]
            if missing:
                raise KeyError(f"station_two_ml is missing columns: {missing}. "
                            f"Available: {list(station_two_ml.columns)}")

            # --- Merge ML↔ML deltas into `station`
            station = station.merge(
                station_two_ml[required_cols + [f"rmse_{ML1_NAME}", f"rmse_{ML2_NAME}"]],
                on="SID",
                how="left",
                validate="one_to_one",
            )

            # --- Final guard before plotting
            for need in ["delta_rmse_ML1_minus_ML2", "delta_skill_ML2_minus_ML1_pct"]:
                if need not in station.columns:
                    # Fallback: use station_two_ml directly for ML↔ML plots/hists
                    print(f"Warning: '{need}' not found in 'station' after merge; "
                        f"will plot ML↔ML figures from 'station_two_ml'.")


            # 5) Optional: normalized RMSE via obs std on the same aligned set
            if DO_NRMSE:
                obs_std = compute_obs_std_from_joined(joined)  # SID, obs_std
                station = station.merge(obs_std, on="SID", how="left")
                station["nrmse_raw"] = station["rmse_raw"] / station["obs_std"]
                station["nrmse_mos"] = station["rmse_mos"] / station["obs_std"]
                station["nrmse_best_ml"] = station["rmse_best_ml"] / station["obs_std"]
                station["delta_nrmse_mos_minus_bestml"] = station["nrmse_mos"] - station["nrmse_best_ml"]

            # 6) Save numeric table (optional)
            # station.to_csv(OUT_DIR / f"station_metrics_bestml_vs_mos_{t0:%Y%m%d}_{t1:%Y%m%d}.csv", index=False)

            # 7) Summaries for BEST-ML vs MOS
            print(f"\n=== Station-level: Best-ML vs MOS (MOS-driven window {season}) ===")
            if len(station) == 0:
                print("No stations after MOS-driven alignment.")
            else:
                ml_better = (station["delta_rmse_mos_minus_bestml"] > 0).sum()
                mos_better = (station["delta_rmse_mos_minus_bestml"] < 0).sum()
                print(f"Stations total: {len(station)}")
                print(f"Best-ML better (RMSE): {ml_better} ({ml_better/len(station)*100:.1f}%)")
                print(f"MOS better (RMSE):    {mos_better} ({mos_better/len(station)*100:.1f}%)")
                print(f"Mean ΔRMSE (MOS−BestML) [K]: {station['delta_rmse_mos_minus_bestml'].mean():+.3f}")
                print(f"Median ΔRMSE [K]:           {station['delta_rmse_mos_minus_bestml'].median():+.3f}")
                print(f"Mean ΔSkill (BestML−MOS) [%]: {station['delta_skill_bestml_minus_mos_pct'].mean():+.2f}")
                print(f"Median ΔSkill [%]:            {station['delta_skill_bestml_minus_mos_pct'].median():+.2f}")

                # --- Build two GeoDataFrames: one for ML↔ML, one for Best-ML↔MOS ---
            gdf_ml   = attach_stations_gdf(station_two_ml)  # has rmse_{ML1_NAME}, rmse_{ML2_NAME}, deltas
            gdf_best = attach_stations_gdf(station)         # has rmse_raw, rmse_mos, rmse_best_ml, deltas vs MOS


            # --- Shared scales for comparability (PAIRED mode) ---
            # 1) Absolute RMSE across RAW, MOS, Best-ML, and the individual MLs
            rmse_vmin, rmse_vmax = shared_sequential_limits([
                gdf_best["rmse_raw"], gdf_best["rmse_mos"], gdf_best["rmse_best_ml"],
                gdf_ml[f"rmse_{ML1_TAG}"], gdf_ml[f"rmse_{ML2_TAG}"],
            ])

            # 2) Bias across Best-ML, MOS, RAW, and the two MLs
            bias_norm = shared_diverging_norm([
                gdf_best.get("bias_best_ml", pd.Series(dtype=float)),
                gdf_best.get("bias_mos", pd.Series(dtype=float)),
                gdf_best.get("bias_raw", pd.Series(dtype=float)),
                gdf_ml.get(f"bias_{ML1_TAG}", pd.Series(dtype=float)),
                gdf_ml.get(f"bias_{ML2_TAG}", pd.Series(dtype=float)),
            ])


            # --- Histograms ---
            # ML1 vs ML2 (use station_two_ml)
            plot_hist(
                station_two_ml["delta_rmse_ML1_minus_ML2"],
                f"ΔRMSE ({ML1_NAME}−{ML2_NAME}) — positive = {ML2_NAME} better",
                f"ΔRMSE ({ML1_NAME}−{ML2_NAME}) [K]",
                f"hist_delta_rmse_{ML1_NAME}_minus_{ML2_NAME}.png", OUT_DIR
            )
            plot_hist(
                station_two_ml["delta_skill_ML2_minus_ML1_pct"],
                f"ΔSkill ({ML2_NAME}−{ML1_NAME}) — positive = {ML2_NAME} better",
                f"ΔSkill ({ML2_NAME}−{ML1_NAME}) [%]",
                f"hist_delta_skill_{ML2_NAME}_minus_{ML1_NAME}.png", OUT_DIR
            )

            # Best-ML vs MOS (use station)
            plot_hist(
                station["delta_rmse_mos_minus_bestml"],
                "ΔRMSE (MOS−Best-ML) — positive = Best-ML better",
                "ΔRMSE (MOS − Best-ML) [K]",
                "hist_delta_rmse_mos_minus_bestml.png", OUT_DIR
            )
            plot_hist(
                station["delta_skill_bestml_minus_mos_pct"],
                "ΔSkill (Best-ML−MOS) — positive = Best-ML better",
                "ΔSkill (Best-ML − MOS) [%]",
                "hist_delta_skill_bestml_minus_mos_pct.png", OUT_DIR
            )

            # Absolute RMSE (shared vmin/vmax)
            plot_abs_rmse_map(gdf_best, "rmse_raw",     f"RAW RMSE ({season})",      "RMSE (RAW) [K]",      "map_rmse_raw.png",     OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)
            plot_abs_rmse_map(gdf_best, "rmse_mos",     f"MOS RMSE ({season})",      "RMSE (MOS) [K]",      "map_rmse_mos.png",     OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)
            plot_abs_rmse_map(gdf_best, "rmse_best_ml", f"Best-ML RMSE ({season})",  "RMSE (Best-ML) [K]",  "map_rmse_best_ml.png", OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)
            plot_abs_rmse_map(gdf_ml,   f"rmse_{ML1_TAG}", f"{ML1_NAME} RMSE ({season})", "RMSE ({ML1_NAME}) [K]", f"map_rmse_{ML1_NAME}.png", OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)
            plot_abs_rmse_map(gdf_ml,   f"rmse_{ML2_TAG}", f"{ML2_NAME} RMSE ({season})", "RMSE ({ML2_NAME}) [K]", f"map_rmse_{ML2_NAME}.png", OUT_DIR, vmin=rmse_vmin, vmax=rmse_vmax)

            # ML1 vs ML2 delta: autoscale to THIS map
            plot_delta_map(
                gdf_ml, "delta_rmse_ML1_minus_ML2",
                title=f"{ML1_NAME} vs {ML2_NAME} — ΔRMSE ({season})\n(positive = {ML2_NAME} better)",
                legend_label=f"ΔRMSE ({ML1_NAME} − {ML2_NAME}) [K]",
                out_name=f"map_delta_rmse_{ML1_NAME}_vs_{ML2_NAME}.png",
                OUT_DIR=OUT_DIR,
                # <<< per-plot norm
                norm=auto_diverging_norm(gdf_ml["delta_rmse_ML1_minus_ML2"], q=0.95, min_span=0.05)
            )

            # MOS vs Best-ML delta: autoscale
            plot_delta_map(
                gdf_best, "delta_rmse_mos_minus_bestml",
                title=f"MOS vs Best-ML — ΔRMSE ({season})\n(positive = Best-ML better)",
                legend_label="ΔRMSE (MOS − Best-ML) [K]",
                out_name="map_delta_rmse_mos_vs_bestml.png",
                OUT_DIR=OUT_DIR,
                # <<< per-plot norm
                norm=auto_diverging_norm(gdf_best["delta_rmse_mos_minus_bestml"], q=0.95, min_span=0.05)
            )

            # Best-ML vs RAW delta: autoscale
            plot_delta_map(
                gdf_best, "delta_rmse_bestml_minus_raw",
                title=f"Best-ML vs RAW — ΔRMSE ({season})\n(negative = Best-ML better)",
                legend_label="ΔRMSE (Best-ML − RAW) [K]",
                out_name="map_delta_rmse_bestml_vs_raw.png",
                OUT_DIR=OUT_DIR,
                # <<< per-plot norm
                norm=auto_diverging_norm(gdf_best["delta_rmse_bestml_minus_raw"], q=0.95, min_span=0.05)
            )

            # Bias maps (shared bias_norm; pick diverging cmap if desired)
            plot_bias_map(gdf_best, "bias_best_ml",
                        f"Best-ML mean bias ({season}) — positive = overestimate",
                        "map_bias_best_ml.png", OUT_DIR, cmap="RdBu_r", norm=bias_norm)
            plot_bias_map(gdf_best, "bias_mos",
                        f"MOS mean bias ({season}) — positive = overestimate",
                        "map_bias_mos.png", OUT_DIR, cmap="RdBu_r", norm=bias_norm)


            # Skill% map (Best-ML − MOS)
            norm = robust_symmetric_norm(gdf_best["delta_skill_bestml_minus_mos_pct"])
            world = gpd.read_file(WORLD_SHP)
            fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
            world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
            gdf_best.plot(
                ax=ax, column="delta_skill_bestml_minus_mos_pct",
                cmap=SKILL_CMAP, norm=norm,
                markersize=20, edgecolor="black", linewidth=0.3, alpha=0.9,
                legend=True, legend_kwds={"label": "ΔSkill (Best-ML − MOS) [%]", "orientation": "horizontal", "shrink": 0.7},
            )
            ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
            ax.set_title(f"Best-ML vs MOS — Skill Difference ({season})\n(positive = Best-ML better)")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout(); plt.savefig(OUT_DIR / "map_delta_skill_bestml_vs_mos.png", bbox_inches="tight"); plt.close()


            plot_hist(station["bias_best_ml"],
                    "Best-ML mean bias",
                    "Mean bias (Model − Obs) [K]",
                    "hist_bias_best_ml.png", OUT_DIR)

            plot_hist(station["bias_mos"],
                    "MOS mean bias",
                    "Mean bias (Model − Obs) [K]",
                    "hist_bias_mos.png", OUT_DIR)

        # Optional: nRMSE delta map (still from gdf_best)
        """if DO_NRMSE and "delta_nrmse_mos_minus_bestml" in gdf_best.columns:
            plot_delta_map(
                gdf_best,
                column="delta_nrmse_mos_minus_bestml",
                title=f"MOS vs Best-ML — ΔnRMSE ({season})\n(positive = Best-ML better)",
                legend_label="ΔnRMSE (MOS − Best-ML) [σ units]",
                out_name="map_delta_nrmse_mos_vs_bestml.png",
            )"""

        print("Saved figures to:", OUT_DIR)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
