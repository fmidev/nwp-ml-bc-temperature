import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---- Paths
MY_DATA_DIR = Path.home() / "thesis_project" / "data"
OUT_DIR = Path.home() / "thesis_project" / "figures" / "station_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAPS_DIR = MY_DATA_DIR / "maps"
world = gpd.read_file(MAPS_DIR / "ne_110m_admin_0_countries.shp")

# ---- Map bounds (from your MOS wiki box)
LON_MIN, LAT_MIN = -25.0, 25.5
LON_MAX, LAT_MAX =  42.0, 72.0

stations_csv = MY_DATA_DIR / "stations.csv"
# pick the metrics file you saved earlier (by station)
# ---- Load stations
st = pd.read_csv(stations_csv)
st["SID"] = st["SID"].astype(str)

# ---- Load & average metrics across all years
metrics_dir = Path.home() / "thesis_project" / "metrics"
files = sorted(metrics_dir.glob("metrics_by_station_analysistime_*.csv"))
if not files:
    raise FileNotFoundError("No per-station metrics files found for any year.")

all_years = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
all_years["SID"] = all_years["SID"].astype(str)

# Identify columns
rmse_cols = [c for c in all_years.columns if c.startswith("rmse_")]
mae_cols  = [c for c in all_years.columns if c.startswith("mae_")]
mbe_cols  = [c for c in all_years.columns if c.startswith("mbe_")]
# keep a copy of per-row counts if present
has_n = "n" in all_years.columns
if not has_n:
    all_years["n"] = 1  # fallback weights

def combine_rmse(group, col):
    w = group["n"].to_numpy()
    r = group[col].to_numpy()
    # ignore NaNs in a weighted way
    m = np.isfinite(r) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.sum(w[m] * r[m]**2) / np.sum(w[m])))

def combine_weighted_mean(group, col):
    w = group["n"].to_numpy()
    x = group[col].to_numpy()
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

agg_rows = []
for sid, g in all_years.groupby("SID", as_index=False):
    row = {"SID": sid}
    # RMSE columns (proper combination)
    for c in rmse_cols:
        row[c] = combine_rmse(g, c)
    # MAE / MBE (weighted means)
    for c in mae_cols + mbe_cols:
        row[c] = combine_weighted_mean(g, c)
    # total sample count (optional)
    row["n_total"] = int(g["n"].fillna(0).sum())
    agg_rows.append(row)

avg_metrics = pd.DataFrame(agg_rows)

# Recompute skill from combined RMSEs if present
if {"rmse_raw", "rmse_corrected_base"}.issubset(avg_metrics.columns):
    avg_metrics["skill_vs_raw_corrected_base"] = 1.0 - (
        avg_metrics["rmse_corrected_base"] / avg_metrics["rmse_raw"]
    )
if {"rmse_raw", "rmse_corrected_tuned"}.issubset(avg_metrics.columns):
    avg_metrics["skill_vs_raw_corrected_tuned"] = 1.0 - (
        avg_metrics["rmse_corrected_tuned"] / avg_metrics["rmse_raw"]
    )

# (Optional) drop correlation columns if they exist; averaging r/R² is non-trivial
avg_metrics = avg_metrics.drop(
    columns=[c for c in avg_metrics.columns if c.startswith(("r_", "r2_"))],
    errors="ignore"
)

# ---- Merge averaged metrics onto stations
df = st.merge(avg_metrics, on="SID", how="left")

# Make GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326",
)


# ---- Compute difference (negative = improvement, positive = worse)
gdf["delta_rmse_tuned"] =  gdf["rmse_raw"] - gdf["rmse_corrected_tuned"]

# ---- Robust symmetric scaling around 0
vals = pd.to_numeric(gdf["delta_rmse_tuned"], errors="coerce").abs()
vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
span = float(np.nanquantile(vals, 0.95)) if len(vals) else 1.0
norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

# ---- Plot map
fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

gdf.plot(
    ax=ax,
    column="delta_rmse_tuned",
    cmap="PuOr",  # diverging colormap, safe for colorblind users
    norm=norm,
    markersize=18,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.9,
    legend=True,
    legend_kwds={
        "label": "ΔRMSE (raw - tuned + weighted) [°C]",
        "orientation": "horizontal",
        "shrink": 0.7
    },
)

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title("Station ΔRMSE (tuned + weighted vs raw)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "station_delta_rmse_tuned.png", bbox_inches="tight")
plt.close()


# ========== Variant A: RMSE map (e.g., corrected_tuned) ==========
metric_col = "rmse_corrected_tuned"     # or "rmse_corrected_base" / "rmse_raw"
label = "RMSE (°C)"

# Robust vmin/vmax to avoid outlier stretch
vals = gdf[metric_col].replace([np.inf, -np.inf], np.nan).dropna()
vmin, vmax = np.quantile(vals, [0.05, 0.95]) if len(vals) else (None, None)

fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

gdf.plot(
    ax=ax,
    column=metric_col,
    cmap="hot_r",
    markersize=18,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.9,
    legend=True,
    vmin=vmin, vmax=vmax,
    legend_kwds={"label": label, "orientation": "horizontal", "shrink": 0.7},
)

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title("Station RMSE — corrected (tuned + weighted)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "station_rmse_corrected_tuned.png", bbox_inches="tight")
plt.close()

# ========== Variant B: Skill vs raw (1 - RMSE(model)/RMSE(raw)) in % ==========
metric_col = "skill_vs_raw_corrected_tuned"  # or ..._corrected_base
label = "Skill vs raw (%)"

# convert to percent for colorbar (optional)
gdf["_skill_pct"] = gdf[metric_col] * 100.0

vals = gdf["_skill_pct"].replace([np.inf, -np.inf], np.nan).dropna()
# center color at 0%: improvements >0, degradations <0
span = np.nanmax(np.abs(vals)) if len(vals) else 1.0
norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

colors = ["#570040", "white", "#005717"]
cmap = LinearSegmentedColormap.from_list("my_diverging", colors)

fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

gdf.plot(
    ax=ax,
    column="_skill_pct",
    cmap=cmap,
    norm=norm,
    markersize=18,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.9,
    legend=True,
    legend_kwds={"label": label, "orientation": "horizontal", "shrink": 0.7},
)

ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title("Station skill vs raw — corrected (tuned + weighted)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "station_skill_corrected_tuned.png", bbox_inches="tight")
plt.close()

# ========== Variant C: Mean Bias Error (MBE) map ==========
metric_col = "mbe_corrected_tuned"  # prediction - observation
label = "Mean Bias Error (°C)"

vals = gdf[metric_col].replace([np.inf, -np.inf], np.nan).dropna()
span = np.nanmax(np.abs(vals)) if len(vals) else 1.0
norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

colors = ["#25E6D6", "white", "#E62534"]
cmap = LinearSegmentedColormap.from_list("my_diverging", colors)

fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

gdf.plot(
    ax=ax,
    column=metric_col,
    cmap=cmap,
    norm=norm,
    markersize=18,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.9,
    legend=True,
    legend_kwds={"label": label, "orientation": "horizontal", "shrink": 0.7},
)

ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title("Station mean bias — corrected (tuned + weighted)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "station_mbe_corrected_tuned.png", bbox_inches="tight")
plt.close()
