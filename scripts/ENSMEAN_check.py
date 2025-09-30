import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker


MY_DATA_DIR = Path.home() / "thesis_project" / "data" / "ml_data"
OUT = Path.home() / "thesis_project" / "figures" / "ENSMEAN"

file = MY_DATA_DIR / "ml_data_2022-11.parquet"

# Load data
df = pd.read_parquet(file)

# Get the ensemble mean predictions
temps = df["T2_ENSMEAN_MA1"]
temps = temps - 273.15

# Calculate the average ensemble mean prediction across stations for each timepoint
temps_avg = df.groupby(["validtime"])["T2_ENSMEAN_MA1"].mean()
temps_avg = temps_avg.dropna()
temps_avg = temps_avg - 273.15

# Check summary of ensemble mean predictions
print(temps.describe())



# Plot a timeseries of the ensemble mean predictions
plt.figure(figsize=(8, 6), dpi=120)
plt.plot(temps_avg.index, temps_avg)

plt.title("Ensemble Mean Prediction 2022-11")
plt.xlabel("Date")
plt.ylabel("Avg Ensemble Mean (Celcius)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))

plt.xticks(rotation=90)

# Save the figure
plt.tight_layout()
plt.savefig(OUT / "ENSMEAN_T_2022_11_ts_ml.png")
plt.close()


# Plot the histogram of the ensemble mean temperature predictions 
plt.figure(figsize=(8, 6), dpi=120)
plt.hist(temps, bins=30, color="lightblue", edgecolor="black", alpha=0.8)

plt.title("Distribution of Ensemble Mean Predictions 2022-11")
plt.xlabel("Temperature (Celcius)")
plt.ylabel("Number of Predicted Values")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig(OUT / "ENSMEAN_T_2022_11_hist_ml.png")
plt.close()