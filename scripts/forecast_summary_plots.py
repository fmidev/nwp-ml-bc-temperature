import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Path to the data
MY_DATA_DIR = Path.home() / "thesis_project" / "data"

# Output directory 
OUT = Path.home() / "thesis_project" / "figures" / "forecast_summary"
OUT.mkdir(exist_ok=True)

# Set the colormap for the plot
cmap = plt.get_cmap("tab20")

# Read the data into a dataframe 
df = pd.read_csv(MY_DATA_DIR / "summary_all_forecasts.csv")

# Get the year and month from the file and concatenate them for the plotting
df["ym_dt"] = pd.to_datetime(
    {"year": df["year"], "month": df["month"], "day": 1}, errors="coerce"
)
df["year_month"] = (
   df["year"].astype("Int64").astype(str) + "-" + df["month"].astype("Int64").astype(str).str.zfill(2)
)

# List the time variables and weather variables to split the information in to two different figures 
time_vars = ["analysishour", "validtime", "leadtime", "analysishour"]
weather_vars = ["MSL","T2","D2","U10","V10","LCC","MCC","SKT","MX2T","MN2T","T_925","T2_ENSMEAN_MA1","T2_M1","T_925_M1","SSR_Acc","STR_Acc"] 


# Plot the fraction of Nan values for the time variables 
for col in df["column"].unique():
    subset = df[df["column"] == col]
    if col in time_vars:
        plt.plot(subset["ym_dt"], subset["nan_frac"].astype(float), marker="o", label=col)


step = 3
tick_idx = np.arange(0, len(subset), step)
plt.xticks(subset["ym_dt"].iloc[tick_idx], subset["year_month"].iloc[tick_idx], rotation=90)
plt.ylabel("NaN fraction")
plt.title("NaN fraction by month")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "nan_frac_forecasts_time.png")
plt.close()


# Plot the fraction of Nan values for the weather variables 
for i, col in enumerate(weather_vars):
    subset = df[df["column"] == col]
    plt.plot(subset["ym_dt"], subset["nan_frac"].astype(float), marker="o", label=col, color = cmap(i))


step = 4
tick_idx = np.arange(0, len(subset), step)
plt.xticks(subset["ym_dt"].iloc[tick_idx], subset["year_month"].iloc[tick_idx], rotation=90)
plt.ylabel("NaN fraction")
plt.title("NaN fraction by month")
plt.legend(
    bbox_to_anchor=(1.02, 1),  
    loc="upper left",          
    borderaxespad=0,
    ncol=1                     
)
plt.tight_layout()
plt.savefig(OUT / "nan_frac_forecasts_params.png")
plt.close()

# For the weather variables plot the min, max and med value across all stations (not separeted by station, shows min out of all stations ...) 
for col in weather_vars:
    subset = df[df["column"] == col].sort_values("ym_dt")
    
    plt.plot(subset["ym_dt"], subset["min"], marker="o", linestyle="-", label="min")
    plt.plot(subset["ym_dt"], subset["median"], marker="o", linestyle="-", label="median")
    plt.plot(subset["ym_dt"], subset["max"], marker="o", linestyle="-", label="max")
    
    step = 3
    tick_idx = np.arange(0, len(subset), step)
    plt.xticks(subset["ym_dt"].iloc[tick_idx], subset["year_month"].iloc[tick_idx], rotation=90)
    plt.title(f"{col} summary over months")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    outfile = OUT / f"{col}_summary.png"
    plt.savefig(outfile)
    plt.close()

