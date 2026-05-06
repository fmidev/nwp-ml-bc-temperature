import argparse
from pathlib import Path
import json

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString
import joblib
import torch  # only needed if edge_index is a tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GNN station graph edges from a saved training graph directory."
    )

    parser.add_argument(
        "--graph-dir",
        required=True,
        type=str,
        help=(
            "Directory containing stations.parquet, edge_index.pkl, "
            "and optionally graph_params.json."
        ),
    )

    parser.add_argument(
        "--world-shp",
        required=True,
        type=str,
        help="Path to Natural Earth world shapefile, e.g. ne_110m_admin_0_countries.shp.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where the station graph plot will be saved.",
    )

    parser.add_argument(
        "--output-name",
        default="station_graph_edges_lsm.svg",
        type=str,
        help="Output figure filename. Default: station_graph_edges_lsm.svg.",
    )

    parser.add_argument(
        "--coast-threshold-km",
        default=20.0,
        type=float,
        help=(
            "Distance-to-sea threshold used to classify coastal stations. "
            "Should match the training graph setting. Default: 20.0."
        ),
    )

    parser.add_argument(
        "--max-edges-to-plot",
        default=2000,
        type=int,
        help="Maximum number of edges to plot. Default: 2000.",
    )

    parser.add_argument(
        "--random-seed",
        default=42,
        type=int,
        help="Random seed used when subsampling edges. Default: 42.",
    )

    parser.add_argument(
        "--lon-min",
        default=-25.0,
        type=float,
        help="Minimum longitude for map extent.",
    )

    parser.add_argument(
        "--lon-max",
        default=42.0,
        type=float,
        help="Maximum longitude for map extent.",
    )

    parser.add_argument(
        "--lat-min",
        default=25.5,
        type=float,
        help="Minimum latitude for map extent.",
    )

    parser.add_argument(
        "--lat-max",
        default=72.0,
        type=float,
        help="Maximum latitude for map extent.",
    )

    parser.add_argument(
        "--title",
        default="GNN Station Graph Connections",
        type=str,
        help="Base plot title.",
    )

    return parser.parse_args()


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance in kilometers.
    """
    radius_earth_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return radius_earth_km * c


def load_edge_index(edge_index_path: Path) -> np.ndarray:
    """
    Load edge_index from joblib and return it as a NumPy array.
    """
    edge_index = joblib.load(edge_index_path)

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    elif isinstance(edge_index, list):
        edge_index = np.array(edge_index)

    edge_index = np.asarray(edge_index)

    if edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index with shape (2, n_edges), got {edge_index.shape}"
        )

    return edge_index


def main():
    args = parse_args()

    graph_dir = Path(args.graph_dir)
    world_shp = Path(args.world_shp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations_path = graph_dir / "stations.parquet"
    edge_index_path = graph_dir / "edge_index.pkl"
    graph_params_path = graph_dir / "graph_params.json"

    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")

    if not stations_path.exists():
        raise FileNotFoundError(f"Stations parquet not found: {stations_path}")

    if not edge_index_path.exists():
        raise FileNotFoundError(f"Edge index file not found: {edge_index_path}")

    if not world_shp.exists():
        raise FileNotFoundError(f"World shapefile not found: {world_shp}")

    # -------------------------------------------------------------------
    # 1. Load stations in the SAME order as build_graph used
    # -------------------------------------------------------------------
    # In training, build_graph(stations) internally did:
    #   stn = stations.sort_values("SID").reset_index(drop=True)
    # and then used stn's row order as node indices.
    # stations.parquet was saved BEFORE that sort, so we must sort here.
    df = pd.read_parquet(stations_path)

    required_station_cols = {"SID", "lon", "lat"}
    missing = required_station_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"stations.parquet is missing required columns: {sorted(missing)}"
        )

    df["SID"] = df["SID"].astype(str)

    df = df.sort_values("SID").reset_index(drop=True)
    df["node_idx"] = df.index

    gdf_stations = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    if "dist_sea" in df.columns:
        df["is_coast"] = df["dist_sea"] <= args.coast_threshold_km
        print("Coast classification counts:")
        print(df["is_coast"].value_counts())
    else:
        df["is_coast"] = False
        print("[WARN] dist_sea column not found; treating all stations as non-coastal.")

    # -------------------------------------------------------------------
    # 2. Load edge_index from joblib
    # -------------------------------------------------------------------
    edge_index = load_edge_index(edge_index_path)

    src = edge_index[0]
    dst = edge_index[1]

    # Avoid double-drawing undirected edges
    mask = src < dst
    src = src[mask]
    dst = dst[mask]

    # -------------------------------------------------------------------
    # 3. Attach node/coast metadata and coordinates
    # -------------------------------------------------------------------
    edges_df = pd.DataFrame(
        {
            "src": src,
            "dst": dst,
        }
    )

    edges_df = (
        edges_df.merge(
            df[["node_idx", "is_coast"]],
            left_on="src",
            right_on="node_idx",
            how="left",
        )
        .rename(columns={"is_coast": "is_coast_src"})
        .drop(columns=["node_idx"])
    )

    edges_df = (
        edges_df.merge(
            df[["node_idx", "is_coast"]],
            left_on="dst",
            right_on="node_idx",
            how="left",
        )
        .rename(columns={"is_coast": "is_coast_dst"})
        .drop(columns=["node_idx"])
    )

    edges_df["cross_coast"] = edges_df["is_coast_src"] != edges_df["is_coast_dst"]

    print("Cross-coast edges:", int(edges_df["cross_coast"].sum()))

    edges_df = (
        edges_df.merge(
            gdf_stations[["node_idx", "lon", "lat"]],
            left_on="src",
            right_on="node_idx",
            how="left",
        )
        .rename(columns={"lon": "lon_src", "lat": "lat_src"})
        .drop(columns=["node_idx"])
    )

    edges_df = (
        edges_df.merge(
            gdf_stations[["node_idx", "lon", "lat"]],
            left_on="dst",
            right_on="node_idx",
            how="left",
        )
        .rename(columns={"lon": "lon_dst", "lat": "lat_dst"})
        .drop(columns=["node_idx"])
    )

    coord_cols = ["lon_src", "lat_src", "lon_dst", "lat_dst"]
    if edges_df[coord_cols].isna().any().any():
        n_bad = edges_df[edges_df[coord_cols].isna().any(axis=1)].shape[0]
        raise ValueError(
            f"{n_bad} edges have missing station coordinates. "
            "Check edge_index and station ordering."
        )

    # -------------------------------------------------------------------
    # 4. Line geometry + great-circle distance
    # -------------------------------------------------------------------
    edges_df["geometry"] = edges_df.apply(
        lambda row: LineString(
            [
                (row["lon_src"], row["lat_src"]),
                (row["lon_dst"], row["lat_dst"]),
            ]
        ),
        axis=1,
    )

    edges_gdf = gpd.GeoDataFrame(
        edges_df,
        geometry="geometry",
        crs="EPSG:4326",
    )

    edges_gdf["dist_km"] = haversine_km(
        edges_gdf["lat_src"].values,
        edges_gdf["lon_src"].values,
        edges_gdf["lat_dst"].values,
        edges_gdf["lon_dst"].values,
    )

    print("Edge distance summary:")
    print(edges_gdf["dist_km"].describe())

    # -------------------------------------------------------------------
    # 5. Optional graph params
    # -------------------------------------------------------------------
    if graph_params_path.exists():
        with open(graph_params_path, "r") as f:
            graph_params = json.load(f)

        radius_km_used = graph_params.get("radius_km", None)
    else:
        graph_params = {}
        radius_km_used = None

    # -------------------------------------------------------------------
    # 6. Optional subsampling for plotting
    # -------------------------------------------------------------------
    if args.max_edges_to_plot <= 0:
        edges_gdf_plot = edges_gdf
    elif len(edges_gdf) > args.max_edges_to_plot:
        edges_gdf_plot = edges_gdf.sample(
            args.max_edges_to_plot,
            random_state=args.random_seed,
        )
    else:
        edges_gdf_plot = edges_gdf

    print(f"Total undirected edges: {len(edges_gdf):,}")
    print(f"Edges plotted: {len(edges_gdf_plot):,}")

    # -------------------------------------------------------------------
    # 7. Plot
    # -------------------------------------------------------------------
    world = gpd.read_file(world_shp)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

    world.plot(
        ax=ax,
        color="#f2f2f2",
        edgecolor="#999999",
        linewidth=0.5,
    )

    edges_gdf_plot.plot(
        ax=ax,
        column="dist_km",
        cmap="viridis",
        linewidth=0.5,
        alpha=0.7,
        legend=True,
        legend_kwds={"shrink": 0.7},
    )

    gdf_stations.plot(
        ax=ax,
        markersize=18,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        color="black",
    )

    ax.set_xlim(args.lon_min, args.lon_max)
    ax.set_ylim(args.lat_min, args.lat_max)

    title = args.title

    if radius_km_used is not None:
        title += f" radius_km ≈ {radius_km_used}"

    ax.set_title(title, fontsize=16)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path = out_dir / args.output_name

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved graph plot to: {output_path}")


if __name__ == "__main__":
    main()

