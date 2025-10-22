
import os
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# User settings (edit these)
# =====================================================
TARGET_STATION = "115797"                    # station ID (string)
TARGET_INIT    = "2025-01-01 00:00:00"       # analysistime (string as in your data)

# Your ML model tag used when saving eval rows (=> column "corrected_<ML_TAG>")
ML_TAG = "tuned_ah_full"                        # e.g. "tuned_full" -> column "corrected_tuned_full"

SHOW_PLOT = True
SAVE_PLOT = True
FIG_DPI   = 150

# Directories
HOME     = Path.home()
METRICS  = HOME / "thesis_project" / "metrics"
MOS_DIR  = METRICS / "mos"
ML_DIR   = METRICS / "full_tuned_ah"                            # ML eval_rows live directly in metrics/
OUT_DIR  = HOME / "thesis_project" / "figures" / "MOSvsML_timeseries" / f"{TARGET_STATION}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns
SPLIT = "analysistime"
KEYS  = ["SID", SPLIT, "validtime", "leadtime"]
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS   = "corrected_mos"
MLCOL = f"corrected_{ML_TAG}"

# =====================================================
# Helpers
# =====================================================
def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d*d)))

def add_fh(df):
    return df.with_columns(
        ( (pl.col("validtime").cast(pl.Datetime) - pl.col("analysistime").cast(pl.Datetime))
          .dt.hours() ).cast(pl.Int32).alias("fh")
    )

# =====================================================
# Load MOS eval rows
# =====================================================
mos_files = sorted(MOS_DIR.glob("eval_rows_analysistime_MOS_*.parquet"))
if not mos_files:
    raise FileNotFoundError(f"No MOS eval_rows files in {MOS_DIR}")

mos_frames = []
for f in mos_files:
    df = pl.read_parquet(f, columns=KEYS + [RAW, OBS, MOS])
    mos_frames.append(df)

mos_all = pl.concat(mos_frames, how="vertical_relaxed").with_columns([
    pl.col("SID").cast(pl.Utf8),
    pl.col(SPLIT).cast(pl.Utf8),
])
print(f"[INFO] MOS rows loaded: {mos_all.height:,}")


# Filter MOS for station/init
mos_plot = mos_all.filter(
    (pl.col("SID") == TARGET_STATION) & (pl.col(SPLIT) == TARGET_INIT)
)

if mos_plot.height == 0:
    raise ValueError(f"No MOS rows for SID={TARGET_STATION}, init={TARGET_INIT}")

print(f"Predicted temperatures MOS= {mos_plot.select(pl.col(MOS).n_unique()).item()}")
# =====================================================
# Load ML eval rows
# =====================================================
ml_files = sorted(ML_DIR.glob(f"eval_rows_{SPLIT}_{ML_TAG}_*.parquet"))
if not ml_files:
    raise FileNotFoundError(f"No ML eval_rows files matching eval_rows_{SPLIT}_{ML_TAG}_*.parquet in {ML_DIR}")

ml_frames = []
for f in ml_files:
    cols = KEYS + [RAW, OBS, MLCOL]
    # tolerate older files lacking RAW (we’ll drop if missing)
    existing = [c for c in cols if c in pl.scan_parquet(str(f)).collect_schema().keys()]
    df = pl.read_parquet(f, columns=existing).with_columns([
        pl.col("SID").cast(pl.Utf8),
        pl.col(SPLIT).cast(pl.Utf8),
    ])
    ml_frames.append(df)

ml_all = pl.concat(ml_frames, how="vertical_relaxed")
print(f"[INFO] ML rows loaded: {ml_all.height:,}")

ml_plot = ml_all.filter(
    (pl.col("SID") == TARGET_STATION) & (pl.col(SPLIT) == TARGET_INIT)
)


def leadtime_coverage(df, name):
    print(f"[COV] {name}: rows={df.height}, unique leadtimes={df.select(pl.col('leadtime').n_unique()).item()}")

leadtime_coverage(mos_plot, "MOS")
leadtime_coverage(ml_plot,  "ML")


if ml_plot.height == 0:
    print(f"[WARN] No ML rows for SID={TARGET_STATION}, init={TARGET_INIT}. Plotting without ML.")
    # use MOS-only
    joined = mos_plot
else:
    # inner-join to align samples (same validtime/leadtime)
    # use ML (raw forecast) as the base
    joined = (
        ml_plot.join(
            mos_plot.select(KEYS + [MOS]),  # only keep relevant cols from MOS
            on=KEYS,
            how="left",                     # keep all ML rows even if MOS missing
            suffix="_mos"
        )
    )

    # if both sides had RAW/OBS, de-duplicate to single columns
    # prefer the MOS side’s names for RAW/OBS
    for col in [RAW, OBS]:
        col_ml = f"{col}_ml"
        if col_ml in joined.columns and col in joined.columns:
            joined = joined.drop(col_ml)

# =====================================================
# Prepare pandas for plotting & metric prints
# =====================================================
plot_pd = joined.to_pandas()
plot_pd = plot_pd.sort_values("validtime")
plot_pd["validtime"] = pd.to_datetime(plot_pd["validtime"])

# Sanity prints
print(f"[INFO] Aligned rows for plot: {len(plot_pd)}")
print(f"[INFO] Columns present: {sorted(plot_pd.columns)}")

# Per-series RMSE vs obs (for this station/init only)
if OBS in plot_pd.columns:
    if RAW in plot_pd.columns:
        print(f"RMSE RAW vs OBS: {rmse(plot_pd[RAW], plot_pd[OBS]):.3f}")
    if MOS in plot_pd.columns:
        print(f"RMSE MOS vs OBS: {rmse(plot_pd[MOS], plot_pd[OBS]):.3f}")
    if MLCOL in plot_pd.columns:
        print(f"RMSE ML  vs OBS: {rmse(plot_pd[MLCOL], plot_pd[OBS]):.3f}")

# =====================================================
# Plot
# =====================================================
plt.figure(figsize=(12, 5))

if RAW in plot_pd.columns:
    plt.plot(plot_pd["validtime"], plot_pd[RAW], label="Raw forecast (T2)", linewidth=1.5)
if MOS in plot_pd.columns:
    plt.plot(plot_pd["validtime"], plot_pd[MOS], label="MOS corrected", linewidth=2, color="tab:orange")
if MLCOL in plot_pd.columns:
    plt.plot(plot_pd["validtime"], plot_pd[MLCOL], label=f"ML corrected ({ML_TAG})", linewidth=2, color="tab:green")

if OBS in plot_pd.columns and plot_pd[OBS].notna().any():
    mask = plot_pd[OBS].notna()
    plt.scatter(plot_pd.loc[mask, "validtime"], plot_pd.loc[mask, OBS],
                s=25, marker="o", color="black", label="Observation")

plt.title(f"Station {TARGET_STATION} — Init {TARGET_INIT}")
plt.xlabel("Valid time")
plt.ylabel("Temperature (K)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save/show
safe_init = TARGET_INIT.replace(":", "").replace(" ", "_")
out_path = OUT_DIR / f"timeseries_MOS_ML_{TARGET_STATION}_{safe_init}.png"
if SAVE_PLOT:
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[OK] Saved plot → {out_path}")
    plt.close()



"""
# =====================================================
# Extra plot: ML/OBS (in °C) vs external file values (by leadtime)
# =====================================================

# ---- User setting for the reference CSV ----
REF_CSV = HOME / "thesis_project" / "verific_102035_09_01.csv"  # <-- set to your file path
REF_LABEL = "External value"  # legend label for the file's series

# ---- Load & prepare reference data ----
# Expecting columns:
# producer_id,analysis_time,target_id,parameter_id,forecaster_id,leadtime,value
# ---- Load & prepare reference data (robust tz parsing) ----
ref_raw = pl.read_csv(REF_CSV)

ref = (
    ref_raw
    .with_columns([
        pl.col("target_id").cast(pl.Utf8).alias("SID"),
        # Normalize "+00" → "+00:00", then parse as UTC
        pl.col("analysis_time")
          .cast(pl.Utf8)
          .str.replace(r"\+00$", "+00:00")  # "+00" -> "+00:00"
          .str.strptime(
              pl.Datetime(time_zone="UTC"),
              format="%Y-%m-%d %H:%M:%S%z"     # <- use 'format' not 'fmt'
          )
          .alias("analysis_dt_utc"),
        pl.col("leadtime").cast(pl.Int32),
        pl.col("value").cast(pl.Float64).alias("ref_C"),
    ])
    .with_columns([
        pl.col("analysis_dt_utc").dt.replace_time_zone(None).alias("analysis_dt_naive")
    ])
    .filter(
        (pl.col("SID") == TARGET_STATION) &
        (pl.col("analysis_dt_naive").dt.strftime("%Y-%m-%d %H:%M:%S") == TARGET_INIT)
    )
    .with_columns([
        (pl.col("analysis_dt_utc") + pl.duration(hours=pl.col("leadtime"))).alias("validtime_utc")
    ])
    .with_columns([
        pl.col("validtime_utc").dt.replace_time_zone(None).alias("validtime")
    ])
    .select(["leadtime", "validtime", "ref_C"])
)



if ref.height == 0:
    raise ValueError(f"No rows in {REF_CSV} for SID={TARGET_STATION}, init={TARGET_INIT}")

print(f"[INFO] External file rows (after filter): {ref.height}")

# ---- Align with ML rows on leadtime ----
# Use the ML side you already loaded/filtered (ml_plot). If none, we still allow plotting OBS/RAW if present.
ml_for_join = ml_plot.select([pl.col("leadtime"), pl.col(RAW).alias(RAW), pl.col(OBS).alias(OBS), pl.col(MLCOL).alias(MLCOL)]) if ml_plot.height > 0 else None

if ml_for_join is None or ml_for_join.height == 0:
    # Fallback: try aligning with MOS frame just to get RAW/OBS if available
    base_for_join = mos_plot.select([pl.col("leadtime"), pl.col(RAW).alias(RAW), pl.col(OBS).alias(OBS)]) if mos_plot.height > 0 else None
else:
    base_for_join = ml_for_join

if base_for_join is None or base_for_join.height == 0:
    raise ValueError("No ML or MOS rows available to align with the reference file.")

joined_ref = base_for_join.join(ref, on="leadtime", how="inner")
print(f"[INFO] Aligned rows with external reference: {joined_ref.height}")

# ---- Prepare pandas & convert Kelvin → Celsius for ML/OBS/RAW ----
plot2 = joined_ref.to_pandas().sort_values("validtime")
plot2["validtime"] = pd.to_datetime(plot2["validtime"])

def K2C(series):
    return series.astype(float) - 273.15

if OBS in plot2.columns:
    if plot2[OBS].notna().any():
        plot2["OBS_C"] = K2C(plot2[OBS])
if RAW in plot2.columns:
    if plot2[RAW].notna().any():
        plot2["RAW_C"] = K2C(plot2[RAW])
if MLCOL in plot2.columns:
    if plot2[MLCOL].notna().any():
        plot2["ML_C"] = K2C(plot2[MLCOL])

# ---- Optional sanity prints: RMSE vs external file (°C) ----
def rmse_pd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d*d)))

if "RAW_C" in plot2 and "ref_C" in plot2:
    print(f"RMSE RAW (°C) vs REF: {rmse_pd(plot2['RAW_C'], plot2['ref_C']):.3f}")
if "ML_C" in plot2 and "ref_C" in plot2:
    print(f"RMSE ML  (°C) vs REF: {rmse_pd(plot2['ML_C'], plot2['ref_C']):.3f}")
if "OBS_C" in plot2 and "ref_C" in plot2:
    print(f"RMSE OBS (°C) vs REF: {rmse_pd(plot2['OBS_C'], plot2['ref_C']):.3f}")

# ---- Plot (like the MOS figure, but with the external series) ----
plt.figure(figsize=(12, 5))

if "RAW_C" in plot2:
    plt.plot(plot2["validtime"], plot2["RAW_C"], label="Raw forecast (T2, °C)", linewidth=1.5)

if "ML_C" in plot2:
    plt.plot(plot2["validtime"], plot2["ML_C"], label=f"ML corrected ({ML_TAG}, °C)", linewidth=2)

# External file series
plt.plot(plot2["validtime"], plot2["ref_C"], label=f"{REF_LABEL} (°C)", linewidth=2)

# Observations as points (°C)
if "OBS_C" in plot2 and np.isfinite(plot2["OBS_C"]).any():
    mask = np.isfinite(plot2["OBS_C"])
    plt.scatter(plot2.loc[mask, "validtime"], plot2.loc[mask, "OBS_C"],
                s=25, marker="o", label="Observation (°C)")

plt.title(f"Station {TARGET_STATION} — Init {TARGET_INIT}  (ML/OBS in °C vs external)")
plt.xlabel("Valid time")
plt.ylabel("Temperature (°C)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save/show
out_path2 = OUT_DIR / f"timeseries_ML_vs_external_{TARGET_STATION}_{safe_init}.png"
if SAVE_PLOT:
    plt.savefig(out_path2, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[OK] Saved plot → {out_path2}")
    plt.close()

"""