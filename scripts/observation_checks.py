import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np

# Paths
MY_DATA_DIR = Path.home() / "thesis_project" / "data"
OUT = Path.home() / "thesis_project" / "figures" 

csv_file = MY_DATA_DIR / "observations.csv"


# Load data
df = pd.read_csv(csv_file)

# Change temperature from Kelvin to Celcius for easier checking of values
temps = df["obs_TA"]-273.15

# Check summary of observed temperatures
print(temps.describe())

# Print the proportion of NAN values
print(df["obs_TA"].isna().sum()/len(temps))


# Plot a histogram of the temperatures
plt.figure(figsize=(8, 6), dpi=120)
plt.hist(temps, bins=30, color="lightblue", edgecolor="black", alpha=0.8)

plt.title("Distribution of Observed Temperatures")
plt.xlabel("Temperature (celsius)")
plt.ylabel("Number of Observations")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig(OUT / "obs_T_hist.png")
plt.close()

# Threshold for NAN values
# Stations with 3h time resolution should have ~2/3 NAN values
# We don't want to flag these
THRESH = 0.75

# Calculate fraction of NAN values per station
nan_frac = (
    df.groupby("SID")["obs_TA"]
      .apply(lambda s: s.isna().mean())
      .rename("nan_fraction")
)

# Flag stations that have over 75% NAN values 
flagged = nan_frac[nan_frac > THRESH].sort_values(ascending=False)
print(f"{len(flagged)} stations flagged (> {THRESH:.0%} NaNs)")
print(flagged)

flagged.to_csv(MY_DATA_DIR / "stations_with_75%_NAN.csv")
