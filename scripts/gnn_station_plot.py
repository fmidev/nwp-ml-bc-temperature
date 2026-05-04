import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString
from pathlib import Path
import joblib
import torch  # only needed if edge_index is a tensor
import json

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path.home() / "thesis_project"
MY_DATA_DIR = BASE_DIR / "data"

# Use the same run directory as training
GRAPH_DIR = BASE_DIR / "models" / "gnn_bias_correction" / "full_gnn_gat_lsm"

OUT_DIR = BASE_DIR / "figures" / "station_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAPS_DIR = BASE_DIR / "data" / "maps"
world = gpd.read_file(MAPS_DIR / "ne_110m_admin_0_countries.shp")

LON_MIN, LAT_MIN = -25.0, 25.5
LON_MAX, LAT_MAX = 42.0, 72.0

# -------------------------------------------------------------------
# 1. Load stations in the SAME order as build_graph used
# -------------------------------------------------------------------
# In training, build_graph(stations) internally did:
#   stn = stations.sort_values("SID").reset_index(drop=True)
# and then used stn's row order as node indices.
# stations.parquet was saved BEFORE that sort, so we must sort here.
stations_path = GRAPH_DIR / "stations.parquet"
df = pd.read_parquet(stations_path)

# IMPORTANT: sort by SID to reproduce the stn used inside build_graph
df = df.sort_values("SID").reset_index(drop=True)
df["node_idx"] = df.index

gdf_stations = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326"
)

lon_col = "lon"
lat_col = "lat"

coast_threshold_km = 20.0  # MUST match training build_graph default/setting
df["is_coast"] = df["dist_sea"] <= coast_threshold_km
print(df["is_coast"].value_counts())
# -------------------------------------------------------------------
# 2. Load edge_index from joblib (from the same run dir)
# -------------------------------------------------------------------
edge_index_path = GRAPH_DIR / "edge_index.pkl"
edge_index = joblib.load(edge_index_path)

# Ensure numpy array
if isinstance(edge_index, torch.Tensor):
    edge_index = edge_index.cpu().numpy()
elif isinstance(edge_index, list):
    edge_index = np.array(edge_index)

src = edge_index[0]
dst = edge_index[1]

# Avoid double-drawing undirected edges
mask = src < dst
src = src[mask]
dst = dst[mask]

# -------------------------------------------------------------------
# 3. Attach coordinates correctly using node_idx
# -------------------------------------------------------------------
edges_df = pd.DataFrame({"src": src, "dst": dst})

edges_df = edges_df.merge(
    df[["node_idx", "is_coast"]],
    left_on="src",
    right_on="node_idx",
    how="left"
).rename(columns={"is_coast": "is_coast_src"}).drop(columns=["node_idx"])

edges_df = edges_df.merge(
    df[["node_idx", "is_coast"]],
    left_on="dst",
    right_on="node_idx",
    how="left"
).rename(columns={"is_coast": "is_coast_dst"}).drop(columns=["node_idx"])

edges_df["cross_coast"] = edges_df["is_coast_src"] != edges_df["is_coast_dst"]
print("Cross-coast edges (should be 0):", edges_df["cross_coast"].sum())


# Merge src coords
edges_df = edges_df.merge(
    gdf_stations[["node_idx", lon_col, lat_col]],
    left_on="src",
    right_on="node_idx",
    how="left"
).rename(columns={lon_col: "lon_src", lat_col: "lat_src"}).drop(columns=["node_idx"])

# Merge dst coords
edges_df = edges_df.merge(
    gdf_stations[["node_idx", lon_col, lat_col]],
    left_on="dst",
    right_on="node_idx",
    how="left"
).rename(columns={lon_col: "lon_dst", lat_col: "lat_dst"}).drop(columns=["node_idx"])

# -------------------------------------------------------------------
# 4. Line geometry + great-circle distance
# -------------------------------------------------------------------
edges_df["geometry"] = edges_df.apply(
    lambda row: LineString([(row["lon_src"], row["lat_src"]),
                            (row["lon_dst"], row["lat_dst"])]),
    axis=1
)

edges_gdf = gpd.GeoDataFrame(edges_df, geometry="geometry", crs="EPSG:4326")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

edges_gdf["dist_km"] = haversine_km(
    edges_gdf["lat_src"].values,
    edges_gdf["lon_src"].values,
    edges_gdf["lat_dst"].values,
    edges_gdf["lon_dst"].values,
)

print(edges_gdf["dist_km"].describe())

# Optional: read graph_params.json to see radius used
graph_params_path = GRAPH_DIR / "graph_params.json"
if graph_params_path.exists():
    with open(graph_params_path, "r") as f:
        graph_params = json.load(f)
    radius_km_used = graph_params.get("radius_km", None)
else:
    graph_params = {}
    radius_km_used = None

# -------------------------------------------------------------------
# 5. Optional subsampling for plotting
# -------------------------------------------------------------------
max_edges_to_plot = 2000
if len(edges_gdf) > max_edges_to_plot:
    edges_gdf_plot = edges_gdf.sample(max_edges_to_plot, random_state=42)
else:
    edges_gdf_plot = edges_gdf

# -------------------------------------------------------------------
# 6. Plot
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)

edges_gdf_plot.plot(
    ax=ax,
    column="dist_km",
    cmap="viridis",
    linewidth=0.5,
    alpha=0.7,
    legend=True,
    legend_kwds={"shrink": 0.7}
)

gdf_stations.plot(
    ax=ax,
    markersize=18,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.9,
    color="black"
)

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)

title = "GNN Station Graph Connections (LSM)"
if radius_km_used is not None:
    title += f" (radius_km ≈ {radius_km_used})"
ax.set_title(title, fontsize=16)

ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "station_graph_edges_lsm.svg", bbox_inches="tight")
plt.close()

