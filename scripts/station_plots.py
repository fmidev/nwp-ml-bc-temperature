import argparse
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot station locations on a map and station elevation histogram."
    )

    parser.add_argument(
        "--stations-csv",
        required=True,
        type=str,
        help="Path to stations.csv. Must contain at least lon and lat columns.",
    )

    parser.add_argument(
        "--maps-dir",
        required=True,
        type=str,
        help=(
            "Directory containing the Natural Earth shapefile. "
            "Expected file: ne_110m_admin_0_countries.shp"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where station_map.png and elevation_hist.png will be saved.",
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

    return parser.parse_args()


def main():
    args = parse_args()

    stations_csv = Path(args.stations_csv)
    maps_dir = Path(args.maps_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_path = maps_dir / "ne_110m_admin_0_countries.shp"

    if not stations_csv.exists():
        raise FileNotFoundError(f"Stations CSV not found: {stations_csv}")

    if not world_path.exists():
        raise FileNotFoundError(f"Map shapefile not found: {world_path}")

    df = pd.read_csv(stations_csv)

    required_cols = {"lon", "lat"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Stations CSV is missing required columns: {sorted(missing)}")

    world = gpd.read_file(world_path)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)

    world.plot(
        ax=ax,
        color="#f2f2f2",
        edgecolor="#999999",
        linewidth=0.5,
    )

    plot_kwargs = {
        "ax": ax,
        "markersize": 18,
        "edgecolor": "black",
        "linewidth": 0.3,
        "alpha": 0.9,
    }

    if "elev" in gdf.columns:
        plot_kwargs.update(
            {
                "column": "elev",
                "cmap": "viridis_r",
                "legend": True,
                "legend_kwds": {
                    "label": "Elevation (m)",
                    "orientation": "horizontal",
                    "shrink": 0.6,
                },
            }
        )

    gdf.plot(**plot_kwargs)

    ax.set_xlim(args.lon_min, args.lon_max)
    ax.set_ylim(args.lat_min, args.lat_max)

    ax.set_title("Map of the stations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)

    station_map_path = out_dir / "station_map.png"
    plt.tight_layout()
    plt.savefig(station_map_path, bbox_inches="tight")
    plt.close()

    print(f"Saved station map to: {station_map_path}")

    if "elev" in df.columns:
        elev = pd.to_numeric(df["elev"], errors="coerce").dropna()

        plt.figure(figsize=(8, 6), dpi=120)
        plt.hist(
            elev,
            bins=30,
            color="lightblue",
            edgecolor="black",
            alpha=0.8,
        )

        plt.title("Distribution of Station Elevations")
        plt.xlabel("Elevation (m)")
        plt.ylabel("Number of Stations")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        elevation_hist_path = out_dir / "elevation_hist.png"
        plt.tight_layout()
        plt.savefig(elevation_hist_path)
        plt.close()

        print(f"Saved elevation histogram to: {elevation_hist_path}")
    else:
        print("No 'elev' column found; skipped elevation histogram.")

if __name__ == "__main__":
    main()