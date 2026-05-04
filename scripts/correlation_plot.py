# CORRELATION HEATMAP — keep lon/lat/elev reliably
import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PATH = Path.home() / "thesis_project" / "data" / "ml_data_full" / "ml_data_full*.parquet"
OUT  = Path.home() / "thesis_project" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Features/label
TEMP_FC   = "T2"
LABEL_OBS = "obs_TA"
weather   = ['MSL','T2','D2','U10','V10','LCC','MCC','SKT','MX2T','MN2T','T_925','T2_ENSMEAN_MA1','T2_M1','T_925_M1']
meta      = ["leadtime","lon","lat","elev",'sin_hod','cos_hod','sin_doy','cos_doy',"analysishour"]
cols      = weather + meta + [LABEL_OBS]

N_SAMPLE = 200_000  # adjust for RAM

# --- Build lazy frame; KEEP columns explicitly, and cast geo/meta numeric
lf = (
    pl.scan_parquet(str(PATH))
    .select(cols)
    .with_columns(
        pl.col("analysishour").cast(pl.Int16,   strict=False),
        pl.col(["lon","lat","elev"]).cast(pl.Float32, strict=False),
        (pl.col(LABEL_OBS) - pl.col(TEMP_FC)).alias("bias"),
    )
    # Optional: shrink remaining float columns to Float32 to save RAM
    .with_columns(cs.float().cast(pl.Float32, strict=False))
)

# --- Deterministic unbiased “shuffle” using a row index (works on old Polars)
lf_sampled = (
    lf.with_row_index("_i")
      .with_columns(_k=((pl.col("_i") * 1103515245 + 12345) % 2_147_483_647).cast(pl.UInt32))
      .sort("_k")
      .head(N_SAMPLE)
      .drop(["_i","_k"])
)

# --- Collect bounded table then finish in pandas
df = lf_sampled.collect(engine="streaming").to_pandas()

# Make 100% sure lon/lat/elev are numeric in pandas too
for c in ["lon", "lat", "elev", "analysishour"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# If a column is entirely NaN after coercion (e.g., missing in some parquet parts),
# you’ll still keep it—Spearman will just yield NaNs for that column.
# Compute correlation on numeric columns:
corr = df.corr(method="spearman", numeric_only=True)

# If you want to guarantee these columns appear even if NaN/constant,
# reindex the corr matrix to include them:
must_have = [c for c in ["lon","lat","elev"] if c in df.columns]
corr = corr.reindex(index=corr.index.union(must_have), columns=corr.columns.union(must_have))

# Plot
plt.figure(figsize=(10, 8))
im = plt.imshow(corr, aspect='auto', cmap="coolwarm")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Feature/Target Correlation (Spearman)", fontsize=20)
plt.tight_layout()
plt.savefig(OUT / "correlation_map.svg", dpi=200)

# Quick sanity check printouts (optional)
print("Columns present:", sorted(df.columns.tolist()))
print("dtypes (subset):", df[["lon","lat","elev"]].dtypes if set(["lon","lat","elev"]).issubset(df.columns) else "geo columns missing")
print(corr.loc[must_have, ["bias"]].sort_values(by="bias", ascending=False) if must_have else "No geo in corr")



