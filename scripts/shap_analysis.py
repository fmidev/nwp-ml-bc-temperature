# SHAP analysis for the XGBoost model
import json
import numpy as np
import polars as pl
import shap
from xgboost import XGBRegressor
from pathlib import Path
import matplotlib.pyplot as plt

# Paths 
DATA_PATH   = Path.home() / "thesis_project" / "data" / "ml_data" / "ml_data_*.parquet"
MODEL_PATH  = "bias_model_tuned_weighted_best.json"
#FEATS_PATH  = "bias_model_features.json"   # <- save your FEATS at train time (see note below)
OUT = Path.home() / "thesis_project" / "figures" / "SHAP"
OUT.mkdir(parents=True, exist_ok=True)

# If you didn’t save features to json, define them here (order must match training):
weather = ["MSL","T2","D2","U10","V10","LCC","MCC","SKT","MX2T","MN2T","T_925","T2_ENSMEAN_MA1","T2_M1","T_925_M1"]
meta    = ["leadtime","lon","lat","elev","sin_hod","cos_hod","sin_doy","cos_doy"]
FEATS   = weather + meta

# How many samples to explain (SHAP is O(n_features * n_trees))
N_EXPLAIN = 30000

# Optional: restrict to a recent time window for relevance
HOLDOUT_DAYS = 365   # last ~year
# ----------------------------

# Load the model
model = XGBRegressor()
model.load_model(MODEL_PATH)

# Load features used in training
"""try:
    with open(FEATS_PATH) as f:
        FEATS = json.load(f)
except FileNotFoundError:
    raise RuntimeError(
        "bias_model_features.json not found. "
        "Save the exact feature list at training time, e.g.:\n"
        '  json.dump(FEATS, open("bias_model_features.json","w"))'
    )"""

# Build a validation/holdout slice to explain
# We only need features (not labels) for SHAP.
lf = pl.scan_parquet(str(DATA_PATH)).select(FEATS + ["analysistime"])

# If analysistime is a string it can be limited to the last year as follows:
lf = lf.with_columns(
    pl.col("analysistime").str.strptime(pl.Datetime, strict=False).alias("analysistime_dt")
)

df = lf.collect(engine="streaming")
if HOLDOUT_DAYS:
    import datetime as dt
    cutoff = df["analysistime_dt"].max() - dt.timedelta(days=HOLDOUT_DAYS)
    df = df.filter(pl.col("analysistime_dt") >= cutoff)

# Drop helper column and keep only features
dfX = df.select(FEATS)

# Convert to numpy (float32)
X_all = dfX.to_numpy().astype(np.float32, copy=False)

# SAMPLE to keep SHAP fast and memory-safe
n = X_all.shape[0]
if n == 0:
    raise ValueError("No rows found for SHAP slice. Check your filters / data path.")
idx = np.random.default_rng(0).choice(n, size=min(n, N_EXPLAIN), replace=False)
X_sample = X_all[idx]

# Build SHAP explainer 
# TreeExplainer works directly on XGBoost models
# Use "approximate" for speed on very large models if needed.
explainer = shap.TreeExplainer(model) 

# Compute SHAP values
# Newer SHAP:
try:
    sv = explainer(X_sample)  # sv.values, sv.base_values
    shap_values = sv.values
    base_value  = np.mean(sv.base_values)
# Older SHAP fallback
except Exception:
    shap_values = explainer.shap_values(X_sample)
    base_value  = explainer.expected_value if np.isscalar(explainer.expected_value) else np.mean(explainer.expected_value)

# (n_samples, n_features)
print("SHAP matrix shape:", shap_values.shape)  

# Plots
# 1) Beeswarm summary (global importance + direction)
plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=FEATS, show=False)
plt.tight_layout()
plt.savefig(OUT / "beeswarm.png")
plt.close()

# 2) Bar plot (mean(|SHAP|) ranking)
plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=FEATS, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(OUT / "shap_bar.png")
plt.close()

# 3) Dependence plots (pick a couple of key features)
#    Example: how bias correction varies with leadtime and with T2
for feat in ["leadtime", "T2"]:
    if feat in FEATS:
        shap.dependence_plot(
            feat, shap_values, X_sample, feature_names=FEATS, interaction_index=None, show=False
        )
        plt.tight_layout()
        plt.savefig(OUT / f"dependece_{feat}.png")
