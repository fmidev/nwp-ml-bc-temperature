import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
MY_DATA_DIR = Path.home() / "thesis_project" / "data"
OUT_DIR = Path.home() / "thesis_project" / "figures"

MAPS_DIR = Path.home() / "thesis_project" / "data" / "maps"
world = gpd.read_file(MAPS_DIR / "ne_110m_admin_0_countries.shp")

# Load the data
csv_file = MY_DATA_DIR / "stations.csv"


# From MOS wiki the approximate area of the stations 
LON_MIN, LAT_MIN = -25.0, 25.5   
LON_MAX, LAT_MAX =  42.0, 72.0 

df = pd.read_csv(csv_file)

# Make geodataframe 
gdf = gpd.GeoDataFrame(df,
                       geometry = gpd.points_from_xy(df["lon"], df["lat"]),
                       crs = "EPSG:4326")


fig, ax = plt.subplots(figsize = (8,10), dpi = 150)

# Plot the base map 
world.plot(ax = ax, color ="#f2f2f2", edgecolor="#999999", linewidth=0.5)

# Plot the station points with color based on elevation
gdf.plot(ax=ax,
         column="elev" if "elev" in gdf.columns else None,
         cmap="viridis",
         markersize=18,
         edgecolor="black",
         linewidth=0.3,
         alpha=0.9,
         legend=True,
         legend_kwds={"label": "Elevation (m)",
            "orientation": "horizontal",
            "shrink": 0.6})

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)

ax.set_title("Map of the stations")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.3)

# Save figure
plt.tight_layout()
plt.savefig(OUT_DIR / "station_map.png", bbox_inches="tight")
plt.close()

elev = df["elev"]


# Plot histogram
plt.figure(figsize=(8, 6), dpi=120)
plt.hist(elev, bins=30, color="lightblue", edgecolor="black", alpha=0.8)

plt.title("Distribution of Station Elevations")
plt.xlabel("Elevation (m)")
plt.ylabel("Number of Stations")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig(OUT_DIR / "elevation_hist.png")
plt.close()