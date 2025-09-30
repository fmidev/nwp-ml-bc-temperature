import polars as pl
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Paths 
PATH = Path.home() / "thesis_project" / "data" / "ml_data" / "ml_data_*.parquet"
MODEL_PATH_BASE  = "bias_model_tuned_best.json"         
MODEL_PATH_TUNED = "bias_model_tuned_weighted_best.json"      
OUT = Path.home() / "thesis_project" / "figures" / "corrected_xgboost"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_INIT    = "2023-07-05 12:00:00"     # The single analysistime wanted
TARGET_STATION = "114576"                   # Set the SID

# Expected range of the output
EXPECTED = list(range(125))

# Parameters
TEMP_FC = "T2"  
LABEL_OBS = "obs_TA"
weather = [
    "MSL", TEMP_FC, "D2", "U10", "V10", "LCC", "MCC", "SKT",
    "MX2T", "MN2T", "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1"]
meta = ["leadtime", "lon", "lat", "elev", "sin_hod", "cos_hod", "sin_doy", "cos_doy"]

# Features
FEATS = weather + meta

# Input information for the correction
ID    = ["SID", "analysistime", "validtime", "leadtime"]


# Load the trained models
base_model = XGBRegressor()
base_model.load_model(MODEL_PATH_BASE)

tuned_model = XGBRegressor()
tuned_model.load_model(MODEL_PATH_TUNED)

# Check for duplicate features
FEATS_NODUP = [c for c in FEATS if c not in ID]

# Slices exactly one initialization and station
scan = pl.scan_parquet(str(PATH)).select(ID + FEATS_NODUP)
pred_slice = (scan.filter((pl.col("analysistime") == TARGET_INIT) & (pl.col("SID") == TARGET_STATION)))

# Assert single init present in slice
ats = pred_slice.select(pl.col("analysistime").unique()).collect()
assert ats.height == 1, f"Expected 1 analysistime, saw {ats.height}: {ats}"

# Assert only one station present
sids = pred_slice.select(pl.col("SID").unique()).collect()
assert sids.height == 1, f"Expected 1 SID after filter, saw {sids.height}: {sids}"

# Deduplicate to one row per (SID, leadtime), keep latest validtime if duplicates
pred_slice = (pred_slice.sort(["SID", "leadtime", "validtime"]).unique(subset=["SID", "leadtime"], keep="last"))

# ---- PREDICT BIAS (both models) ----

# Select once (leadtime is already included via ID)
pred_df = pred_slice.select(ID + FEATS_NODUP).collect(engine="streaming")

# Now build X_pred using the ORIGINAL training FEATS (which *includes* leadtime)
missing = [c for c in FEATS if c not in pred_df.columns]
extra   = [c for c in pred_df.columns if c in FEATS and pred_df[c].dtype is None]
assert not missing, f"Missing feature columns at inference: {missing}"

# Build EXPECTED from actual order present (keeps first occurrence of each leadtime)
ordered = pred_df.sort("validtime")["leadtime"].to_list()
EXPECTED = []
seen = set()
for lt in ordered:
    if lt not in seen:
        EXPECTED.append(lt)
        seen.add(lt)

# Need base forecast column to add bias back
if pred_df[TEMP_FC].null_count() > 0:
    missing = int(pred_df[TEMP_FC].null_count())
    raise ValueError(f"{TEMP_FC} has {missing} NaNs in inference slice; cannot add bias back.")

X_pred = pred_df.select(FEATS).to_numpy().astype(np.float32, copy=False)

# Predict bias with both models
bias_hat_base  = base_model.predict(X_pred)
bias_hat_tuned = tuned_model.predict(X_pred)

# From bias correct the prediction
raw_fc            = pred_df[TEMP_FC].to_numpy()
corrected_base    = raw_fc + bias_hat_base
corrected_tuned   = raw_fc + bias_hat_tuned

# Long outputs, consistent column names
out_long_base = (
    pred_df
    .with_columns(pl.Series("corrected_base", corrected_base))
    .select(["SID", "analysistime", "validtime", "leadtime", "corrected_base"])
    .sort(["SID", "leadtime"])
)

out_long_tuned = (
    pred_df
    .with_columns(pl.Series("corrected_tuned", corrected_tuned))
    .select(["SID", "analysistime", "validtime", "leadtime", "corrected_tuned"])
    .sort(["SID", "leadtime"])
)

# Guarantee horizons (fill missing with NaN)
sid_value = pred_df["SID"].unique().to_list()[0]
grid = pl.DataFrame({
    "SID": [sid_value] * len(EXPECTED),
    "leadtime": EXPECTED,
})


# Frame with raw forecast + both corrected temps
plot_df = (
    pred_df
    .select(["SID", "analysistime", "validtime", "leadtime", TEMP_FC])
    .rename({TEMP_FC: "raw_fc"})
    .join(out_long_base,  on=["SID","analysistime","validtime","leadtime"], how="left")
    .join(out_long_tuned, on=["SID","analysistime","validtime","leadtime"], how="left")
)

# Bring in observations for the same station + init, dedup like predictions
obs_scan = (
    pl.scan_parquet(str(PATH))
      .select(["SID", "analysistime", "validtime", "leadtime", LABEL_OBS])
      .filter((pl.col("analysistime") == TARGET_INIT) & (pl.col("SID") == TARGET_STATION))
      .sort(["SID", "leadtime", "validtime"])
      .unique(subset=["SID", "leadtime"], keep="last")
)
obs_df = obs_scan.collect(engine="streaming")

plot_df = plot_df.join(
    obs_df,
    on=["SID", "analysistime", "validtime", "leadtime"],
    how="left"
)

# Order and convert to pandas for matplotlib
plot_pd = plot_df.sort("validtime").to_pandas()
plot_pd["validtime"] = pd.to_datetime(plot_pd["validtime"], errors="coerce")

# Plot: raw + baseline + tuned + (scatter obs)
plt.figure(figsize=(12, 5))
plt.plot(plot_pd["validtime"], plot_pd["raw_fc"], label="Raw forecast (T2)", linewidth=1.5)
plt.plot(plot_pd["validtime"], plot_pd["corrected_base"],  label="Corrected (tuned)", linewidth=2)
plt.plot(plot_pd["validtime"], plot_pd["corrected_tuned"], label="Corrected (tuned + weighted)", linewidth=2)

if LABEL_OBS in plot_pd.columns and plot_pd[LABEL_OBS].notna().any():
    mask = plot_pd[LABEL_OBS].notna()
    plt.scatter(plot_pd.loc[mask, "validtime"], plot_pd.loc[mask, LABEL_OBS],
                s=18, marker="o", label="Observation")

plt.title(f"Station {TARGET_STATION} — init {TARGET_INIT}")
plt.xlabel("Valid time")
plt.ylabel("Temperature (°C)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / f"corrected_temp_compare_{TARGET_STATION}_{TARGET_INIT}.png")

# Print quick metrics if obs available
if LABEL_OBS in plot_pd.columns and plot_pd[LABEL_OBS].notna().any():
    m = plot_pd[LABEL_OBS].notna()
    def rmse(y, x): return float(np.sqrt(np.mean((y - x)**2)))
    raw_rmse   = rmse(plot_pd.loc[m, LABEL_OBS], plot_pd.loc[m, "raw_fc"])
    base_rmse  = rmse(plot_pd.loc[m, LABEL_OBS], plot_pd.loc[m, "corrected_base"])
    tuned_rmse = rmse(plot_pd.loc[m, LABEL_OBS], plot_pd.loc[m, "corrected_tuned"])
    print(f"RMSE raw: {raw_rmse:.3f} | tuned: {base_rmse:.3f} | tuned+weighted: {tuned_rmse:.3f}")

