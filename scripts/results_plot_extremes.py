"""
Evaluate TMAX/TMIN inference results at event level instead of hourly row level.

This script reads one inference parquet file produced by xgboost_inference_clean.py,
collapses copied hourly rows into one event row per station + analysistime + event,
computes station-level metrics, saves event/station outputs, and generates Europe
map plots similar to results_plot.py.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


RAW_COL = "raw_fc"

LON_MIN, LAT_MIN = -25.0, 25.5
LON_MAX, LAT_MAX = 42.0, 72.0

DIV_CMAP = "PuOr"
SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill_cmp",
    ["#570040", "white", "#005717"],
)
BIAS_CMAP = "coolwarm"
RMSE_CMAP = "hot_r"

WORLD_GDF_CACHE = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate TMAX/TMIN inference rows to event level, compute station "
            "metrics, and generate maps."
        )
    )

    parser.add_argument("--input-file", required=True, type=str)
    parser.add_argument("--variable", required=True, choices=["tmax", "tmin"])
    parser.add_argument("--model-tag", required=True, type=str)
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--obs-col", required=True, type=str)
    parser.add_argument("--stations-csv", required=True, type=str)
    parser.add_argument("--world-shp", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--raw-name", default="ECMWF", type=str)
    parser.add_argument("--value-unit", default="K", type=str)
    parser.add_argument("--threads", default="16", type=str)
    parser.add_argument(
        "--aggregation",
        default="mean",
        choices=["mean", "anchor"],
        help="mean = average all rows in the 12-hour event window. anchor = use only the 18/06 UTC row.",
    )

    return parser.parse_args()


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def rmse(arr) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def mae(arr) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(np.abs(a))) if a.size else float("nan")


def mean_bias(arr) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def skill_pct(mae_raw_value: float, mae_ml_value: float) -> float:
    if not np.isfinite(mae_raw_value) or mae_raw_value == 0:
        return float("nan")
    return float(100.0 * (1.0 - mae_ml_value / mae_raw_value))


def shared_diverging_norm(series_list, q=0.95, vcenter=0.0):
    vals = pd.concat(
        [pd.to_numeric(s, errors="coerce") for s in series_list],
        ignore_index=True,
    )
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

    if vals.empty:
        span = 1.0
    else:
        span = float(np.nanquantile(np.abs(vals), q))
        if not np.isfinite(span) or span == 0:
            span = float(np.nanmax(np.abs(vals))) or 1.0

    return TwoSlopeNorm(vmin=-span, vcenter=vcenter, vmax=span)


def shared_sequential_limits(series_list, lo=5, hi=95):
    vals = pd.concat(
        [pd.to_numeric(s, errors="coerce") for s in series_list],
        ignore_index=True,
    )
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

    if vals.empty:
        return None, None

    return float(np.nanpercentile(vals, lo)), float(np.nanpercentile(vals, hi))


def auto_diverging_norm(series: pd.Series, q=0.95, vcenter=0.0, min_span=None):
    vals = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if vals.empty:
        span = 1.0
    else:
        span = float(np.nanquantile(np.abs(vals), q))
        if not np.isfinite(span) or span == 0:
            span = float(np.nanmax(np.abs(vals))) or 1.0

    if min_span is not None:
        span = max(span, float(min_span))

    return TwoSlopeNorm(vmin=-span, vcenter=vcenter, vmax=span)


def parquet_columns(file_path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(file_path).schema_arrow.names


def load_input_rows(input_file: Path, obs_col: str, corr_col: str) -> pd.DataFrame:
    needed_cols = ["SID", "analysistime", "validtime", "leadtime", RAW_COL, obs_col, corr_col]
    existing_cols = parquet_columns(input_file)
    missing = [col for col in needed_cols if col not in existing_cols]

    if missing:
        raise ValueError(
            f"Input parquet is missing required columns: {missing}. "
            f"Available columns: {existing_cols}"
        )

    df = pd.read_parquet(input_file, columns=needed_cols)
    df["SID"] = df["SID"].astype(str)
    df["analysistime"] = pd.to_datetime(df["analysistime"], errors="coerce").dt.tz_localize(None)
    df["validtime"] = pd.to_datetime(df["validtime"], errors="coerce").dt.tz_localize(None)
    df["leadtime"] = pd.to_numeric(df["leadtime"], errors="coerce")
    df[RAW_COL] = pd.to_numeric(df[RAW_COL], errors="coerce")
    df[obs_col] = pd.to_numeric(df[obs_col], errors="coerce")
    df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")

    df = df.dropna(subset=["SID", "analysistime", "validtime"]).copy()
    return df


def assign_event_time(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    out = df.copy()
    hour = out["validtime"].dt.hour
    day = out["validtime"].dt.normalize()

    if variable == "tmax":
        out = out[hour.between(7, 18)].copy()
        out["event_time"] = day.loc[out.index] + pd.Timedelta(hours=18)
        out["event_anchor_hour"] = 18
    else:
        mask = hour.isin([19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        out = out[mask].copy()
        hour = out["validtime"].dt.hour
        same_day_06 = out["validtime"].dt.normalize() + pd.Timedelta(hours=6)
        next_day_06 = same_day_06 + pd.Timedelta(days=1)
        out["event_time"] = same_day_06.where(hour <= 6, next_day_06)
        out["event_anchor_hour"] = 6

    if out.empty:
        raise ValueError(f"No rows remained after {variable.upper()} event-hour filtering.")

    out["valid_hour"] = out["validtime"].dt.hour
    out = out.sort_values(["SID", "analysistime", "event_time", "validtime", "leadtime"]).reset_index(drop=True)
    return out


def aggregate_events(df: pd.DataFrame, variable: str, aggregation: str, obs_col: str, corr_col: str) -> pd.DataFrame:
    work = assign_event_time(df, variable)
    keys = ["SID", "analysistime", "event_time"]

    base = (
        work.groupby(keys, as_index=False)
        .agg(
            obs_event=(obs_col, "first"),
            n_rows=("validtime", "size"),
            validtime_start=("validtime", "min"),
            validtime_end=("validtime", "max"),
            leadtime_min=("leadtime", "min"),
            leadtime_max=("leadtime", "max"),
        )
    )

    anchor_rows = work[work["validtime"] == work["event_time"]].copy()
    anchor = (
        anchor_rows.groupby(keys, as_index=False)
        .agg(
            anchor_validtime=("validtime", "first"),
            anchor_leadtime=("leadtime", "first"),
            raw_anchor=(RAW_COL, "first"),
            ml_anchor=(corr_col, "first"),
        )
    )

    if aggregation == "mean":
        values = (
            work.groupby(keys, as_index=False)
            .agg(
                raw_event=(RAW_COL, "mean"),
                ml_event=(corr_col, "mean"),
            )
        )
    else:
        values = anchor.rename(columns={"raw_anchor": "raw_event", "ml_anchor": "ml_event"})

    event_rows = (
        base.merge(anchor[["SID", "analysistime", "event_time", "anchor_validtime", "anchor_leadtime"]], on=keys, how="left")
        .merge(values[["SID", "analysistime", "event_time", "raw_event", "ml_event"]], on=keys, how="left")
    )

    event_rows["variable"] = variable.upper()
    event_rows["aggregation"] = aggregation
    event_rows["anchor_available"] = event_rows["anchor_validtime"].notna()
    event_rows["raw_error"] = event_rows["raw_event"] - event_rows["obs_event"]
    event_rows["ml_error"] = event_rows["ml_event"] - event_rows["obs_event"]

    return event_rows.sort_values(keys).reset_index(drop=True)


def compute_station_metrics(event_rows: pd.DataFrame) -> pd.DataFrame:
    usable = event_rows.dropna(subset=["obs_event", "raw_event", "ml_event"]).copy()

    records = []
    for sid, group in usable.groupby("SID", sort=True):
        raw_error = group["raw_error"].to_numpy(dtype=float)
        ml_error = group["ml_error"].to_numpy(dtype=float)

        rmse_raw_value = rmse(raw_error)
        rmse_ml_value = rmse(ml_error)
        mae_raw_value = mae(raw_error)
        mae_ml_value = mae(ml_error)

        records.append(
            {
                "SID": str(sid),
                "rmse_raw": rmse_raw_value,
                "rmse_ml": rmse_ml_value,
                "mae_raw": mae_raw_value,
                "mae_ml": mae_ml_value,
                "bias_raw": mean_bias(raw_error),
                "bias_ml": mean_bias(ml_error),
                "delta_rmse_raw_minus_ml": rmse_raw_value - rmse_ml_value,
                "delta_mae_raw_minus_ml": mae_raw_value - mae_ml_value,
                "skill_ml_vs_raw_pct": skill_pct(mae_raw_value, mae_ml_value),
                "n_events": int(len(group)),
                "mean_rows_per_event": float(group["n_rows"].mean()),
            }
        )

    return pd.DataFrame(records).sort_values("SID").reset_index(drop=True)


def summary_metrics(event_rows: pd.DataFrame) -> dict:
    usable = event_rows.dropna(subset=["obs_event", "raw_event", "ml_event"]).copy()

    raw_rmse_value = rmse(usable["raw_error"])
    ml_rmse_value = rmse(usable["ml_error"])
    raw_mae_value = mae(usable["raw_error"])
    ml_mae_value = mae(usable["ml_error"])

    return {
        "n_events_total": int(len(event_rows)),
        "n_events_usable": int(len(usable)),
        "n_stations": int(usable["SID"].nunique()),
        "mean_rows_per_event": float(usable["n_rows"].mean()) if not usable.empty else float("nan"),
        "row_count_distribution": usable["n_rows"].value_counts().sort_index(),
        "rmse_raw": raw_rmse_value,
        "rmse_ml": ml_rmse_value,
        "delta_rmse_raw_minus_ml": raw_rmse_value - ml_rmse_value,
        "mae_raw": raw_mae_value,
        "mae_ml": ml_mae_value,
        "delta_mae_raw_minus_ml": raw_mae_value - ml_mae_value,
        "bias_raw": mean_bias(usable["raw_error"]),
        "bias_ml": mean_bias(usable["ml_error"]),
    }


def print_summary(summary: dict):
    print("[SUMMARY] Aggregated events total:", summary["n_events_total"])
    print("[SUMMARY] Number of events:", summary["n_events_usable"])
    print("[SUMMARY] Number of stations:", summary["n_stations"])
    print(f"[SUMMARY] Mean rows per event: {summary['mean_rows_per_event']:.3f}")
    print("[SUMMARY] Row-count distribution by n_rows:")

    distribution = summary["row_count_distribution"]
    if distribution.empty:
        print("  <empty>")
    else:
        for n_rows, count in distribution.items():
            print(f"  n_rows={int(n_rows)}: {int(count)}")

    print(f"[SUMMARY] Overall raw RMSE: {summary['rmse_raw']:.4f}")
    print(f"[SUMMARY] Overall ML RMSE: {summary['rmse_ml']:.4f}")
    print(f"[SUMMARY] Delta RMSE raw minus ML: {summary['delta_rmse_raw_minus_ml']:.4f}")
    print(f"[SUMMARY] Overall raw MAE: {summary['mae_raw']:.4f}")
    print(f"[SUMMARY] Overall ML MAE: {summary['mae_ml']:.4f}")
    print(f"[SUMMARY] Delta MAE raw minus ML: {summary['delta_mae_raw_minus_ml']:.4f}")
    print(f"[SUMMARY] Raw bias: {summary['bias_raw']:.4f}")
    print(f"[SUMMARY] ML bias: {summary['bias_ml']:.4f}")


def attach_stations_gdf(station_metrics: pd.DataFrame, stations_csv: Path) -> gpd.GeoDataFrame:
    stations = pd.read_csv(stations_csv)
    stations["SID"] = stations["SID"].astype(str)

    required_cols = {"SID", "lon", "lat"}
    missing = required_cols - set(stations.columns)
    if missing:
        raise ValueError(f"Stations CSV is missing required columns: {sorted(missing)}")

    merged = stations.merge(station_metrics, on="SID", how="left")
    return gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["lon"], merged["lat"]),
        crs="EPSG:4326",
    )


def world_gdf(world_shp: Path):
    global WORLD_GDF_CACHE

    if WORLD_GDF_CACHE is None:
        WORLD_GDF_CACHE = gpd.read_file(world_shp)

    return WORLD_GDF_CACHE


def save_svg_pdf(out_dir: Path, stem: str):
    svg = out_dir / f"{stem}.svg"
    pdf = out_dir / f"{stem}.pdf"
    plt.savefig(svg, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"[OK] Saved {svg}")
    print(f"[OK] Saved {pdf}")


def plot_abs_rmse_map(gdf, world, column, title, legend_label, stem, out_dir, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax,
        column=column,
        cmap=RMSE_CMAP,
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={"label": legend_label, "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def plot_delta_map(gdf, world, column, title, legend_label, stem, out_dir, norm=None):
    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=0.05)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax,
        column=column,
        cmap=DIV_CMAP,
        norm=norm,
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        legend_kwds={"label": legend_label, "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def plot_bias_map(gdf, world, column, title, stem, out_dir, value_unit, norm=None):
    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=0.05)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax,
        column=column,
        cmap=BIAS_CMAP,
        norm=norm,
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        legend_kwds={"label": f"Mean bias (Model - Obs) [{value_unit}]", "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def plot_skill_map(gdf, world, column, title, stem, out_dir, raw_name, norm=None):
    if norm is None:
        norm = auto_diverging_norm(gdf[column], q=0.95, min_span=1.0)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.5)
    gdf.plot(
        ax=ax,
        column=column,
        cmap=SKILL_CMAP,
        norm=norm,
        markersize=20,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        legend=True,
        legend_kwds={"label": f"Skill vs {raw_name} (MAE) [%]", "orientation": "horizontal", "shrink": 0.7},
    )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_svg_pdf(out_dir, stem)
    plt.close(fig)


def make_plots(
    station_metrics: pd.DataFrame,
    summary: dict,
    stations_csv: Path,
    world_shp: Path,
    out_dir: Path,
    variable: str,
    aggregation: str,
    model_tag: str,
    model_name: str,
    raw_name: str,
    value_unit: str,
):
    gdf = attach_stations_gdf(station_metrics, stations_csv)
    world = world_gdf(world_shp)

    rmse_vmin, rmse_vmax = shared_sequential_limits([gdf["rmse_raw"], gdf["rmse_ml"]])
    bias_norm = shared_diverging_norm([gdf["bias_raw"], gdf["bias_ml"]])
    safe_tag = safe_name(model_tag)
    prefix = f"{variable.upper()}_{safe_tag}_{aggregation}"

    plot_abs_rmse_map(
        gdf,
        world,
        "rmse_raw",
        f"{raw_name} {variable.upper()} event RMSE ({aggregation}) - overall {summary['rmse_raw']:.2f} {value_unit}",
        f"RMSE ({raw_name}) [{value_unit}]",
        f"map_rmse_raw_{prefix}",
        out_dir,
        vmin=rmse_vmin,
        vmax=rmse_vmax,
    )
    plot_abs_rmse_map(
        gdf,
        world,
        "rmse_ml",
        f"{model_name} {variable.upper()} event RMSE ({aggregation}) - overall {summary['rmse_ml']:.2f} {value_unit}",
        f"RMSE ({model_name}) [{value_unit}]",
        f"map_rmse_ml_{prefix}",
        out_dir,
        vmin=rmse_vmin,
        vmax=rmse_vmax,
    )
    plot_delta_map(
        gdf,
        world,
        "delta_rmse_raw_minus_ml",
        f"{model_name} improvement over {raw_name}\n{variable.upper()} event delta RMSE ({aggregation}) - overall {summary['delta_rmse_raw_minus_ml']:+.2f} {value_unit}",
        f"Delta RMSE {raw_name} - {model_name} [{value_unit}]",
        f"map_delta_rmse_raw_minus_ml_{prefix}",
        out_dir,
    )
    plot_bias_map(
        gdf,
        world,
        "bias_raw",
        f"{raw_name} {variable.upper()} event bias ({aggregation}) - overall {summary['bias_raw']:+.2f} {value_unit}",
        f"map_bias_raw_{prefix}",
        out_dir,
        value_unit,
        norm=bias_norm,
    )
    plot_bias_map(
        gdf,
        world,
        "bias_ml",
        f"{model_name} {variable.upper()} event bias ({aggregation}) - overall {summary['bias_ml']:+.2f} {value_unit}",
        f"map_bias_ml_{prefix}",
        out_dir,
        value_unit,
        norm=bias_norm,
    )
    plot_skill_map(
        gdf,
        world,
        "skill_ml_vs_raw_pct",
        f"{model_name} skill vs {raw_name} ({aggregation}) - overall {skill_pct(summary['mae_raw'], summary['mae_ml']):+.2f}%",
        f"map_skill_ml_vs_raw_{prefix}",
        out_dir,
        raw_name,
    )


def ensure_paths(args):
    input_file = Path(args.input_file)
    stations_csv = Path(args.stations_csv)
    world_shp = Path(args.world_shp)
    output_dir = Path(args.output_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_file}")
    if not stations_csv.exists():
        raise FileNotFoundError(f"Stations CSV not found: {stations_csv}")
    if not world_shp.exists():
        raise FileNotFoundError(f"World shapefile not found: {world_shp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_file, stations_csv, world_shp, output_dir


def main():
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = args.threads
    os.environ["MKL_NUM_THREADS"] = args.threads

    input_file, stations_csv, world_shp, output_dir = ensure_paths(args)
    corr_col = f"corrected_{args.model_tag}"

    print(f"[INFO] Input parquet: {input_file}")
    print(f"[INFO] Variable: {args.variable.upper()}")
    print(f"[INFO] Aggregation: {args.aggregation}")
    print(f"[INFO] Corrected column: {corr_col}")

    rows = load_input_rows(input_file, args.obs_col, corr_col)
    print(f"[INFO] Rows loaded: {len(rows):,}")

    event_rows = aggregate_events(rows, args.variable, args.aggregation, args.obs_col, corr_col)
    print(f"[INFO] Event rows aggregated: {len(event_rows):,}")

    variable_name = args.variable.upper()
    event_out = output_dir / f"event_rows_{variable_name}_{args.model_tag}_{args.aggregation}.parquet"
    station_out = output_dir / f"station_metrics_{variable_name}_{args.model_tag}_{args.aggregation}.csv"

    event_rows.to_parquet(event_out, index=False)
    print(f"[OK] Saved {event_out}")

    usable_events = event_rows.dropna(subset=["obs_event", "raw_event", "ml_event"]).copy()
    if usable_events.empty:
        raise ValueError(
            "No usable event rows remained after aggregation. "
            "Check event-hour coverage and whether the requested observation/corrected columns contain values."
        )

    station_metrics = compute_station_metrics(event_rows)
    station_metrics.to_csv(station_out, index=False)
    print(f"[OK] Saved {station_out}")

    summary = summary_metrics(event_rows)
    print_summary(summary)

    make_plots(
        station_metrics=station_metrics,
        summary=summary,
        stations_csv=stations_csv,
        world_shp=world_shp,
        out_dir=output_dir,
        variable=args.variable,
        aggregation=args.aggregation,
        model_tag=args.model_tag,
        model_name=args.model_name,
        raw_name=args.raw_name,
        value_unit=args.value_unit,
    )

    print(f"[OK] All outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
