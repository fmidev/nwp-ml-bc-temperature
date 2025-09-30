# CORRELATION HEATMAP
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 

# Paths 
PATH = Path.home() / "thesis_project" / "data" / "ml_data" / "ml_data_*.parquet"
OUT = Path.home() / "thesis_project" / "figures" 


# Features and the label
TEMP_FC   = "T2"
LABEL_OBS = "obs_TA"
weather = ['MSL', 'T2', 'D2', 'U10', 'V10', 'LCC', 'MCC', 'SKT', 'MX2T', 'MN2T', 'T_925', 'T2_ENSMEAN_MA1', 'T2_M1', 'T_925_M1']    
meta    = ["leadtime","lon","lat","elev",'sin_hod', 'cos_hod', 'sin_doy', 'cos_doy']

# Columns 
cols = weather + meta + [LABEL_OBS]

# Load a sample to keep plots readable
scan = pl.scan_parquet(PATH).select(cols)
df_pl = scan.collect(streaming=True)
df = df_pl.to_pandas()

# Build bias target
df["bias"] = df[LABEL_OBS] - df[TEMP_FC]

# Keep only numeric columns
num = df.select_dtypes(include=[np.number]).copy()

# Downsample rows for speed
num = num.sample(n=min(len(num), 200_000), random_state=0)

# Correlation (method can be chosen between Spearman and Pearson)
corr = num.corr(method="spearman")

# Plot heatmap with matplotlib
plt.figure(figsize=(10, 8))
im = plt.imshow(corr, aspect='auto')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Feature/Target Correlation (Spearman)")
plt.tight_layout()
plt.savefig(OUT / "correlation_map.png")

# Print top correlations with the bias target
print(
    corr["bias"].sort_values(ascending=False).head(10),
    "\n\n",
    corr["bias"].sort_values(ascending=True).head(10)
)
