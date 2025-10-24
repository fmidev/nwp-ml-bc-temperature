import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path.home() / "thesis_project" / "metrics"
OUT_DIR = Path.home() / "thesis_project" / "figures" / "hit_rate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = DATA_DIR / "hit_rate_from2019_summer.csv"   

# Read the combined results
df = pd.read_csv(csv_path)

# Filter only the by-leadtime rows
df_by = df[df["level"] == "by_leadtime"].copy()

# Ensure leadtime is numeric and sorted
df_by["leadtime"] = pd.to_numeric(df_by["leadtime"], errors="coerce")
df_by = df_by.dropna(subset=["leadtime"]).sort_values("leadtime")

# Pivot to wide format: one column per model
wide = df_by.pivot(index="leadtime", columns="model", values="hit_rate")

#  Plot colors
colors = {
    "DIFF": "#999999",  
    "MOS": "#637AB9",  
    "ML":  "#B95E82",   
}

# Assuming 'wide' DataFrame with columns 'MOS' and 'EC_ML'
plt.figure(figsize=(8, 5))

# Left y-axis: hit rates
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(wide.index, wide["MOS"], "o-", label="MOS", color=colors["MOS"])
ax1.plot(wide.index, wide["EC_ML"], "o-", label="EC_ML", color=colors["ML"])
ax1.set_xlabel("Leadtime (h)")
ax1.set_ylabel("Hit Rate")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

# Right y-axis: difference
ax2 = ax1.twinx()
diff = wide["EC_ML"] - wide["MOS"]
ax2.plot(wide.index, diff, "-", color=colors["DIFF"], label="Δ (EC_ML - MOS)")
ax2.set_ylabel("Hit Rate Difference")
ax2.axhline(0, color="gray", linewidth=1, linestyle="--")

ax2.legend(loc="center right")
plt.title("Hit Rate Comparison by Leadtime Autumn")
plt.tight_layout()
plt.savefig(OUT_DIR / "hit_rate_by_leadtime_plot_from2019_summer.png")
