# Plot of the observations and averaged predictions across all analysis times
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
MY_DATA_DIR = Path.home() / "thesis_project" / "data" / "ml_data"
STATION_FILE = Path.home() / "thesis_project" / "data" / "stations.csv"
OUT = Path.home() / "thesis_project" / "figures" / "pred_obs"
OUT.mkdir(parents=True, exist_ok=True)


def load_station_names(station_file=STATION_FILE, sid_col="SID", name_col="name"):
    """ Helper function to load stations names from csv file
        Params:
            station_file = Path and file name of the csv file containing the station information
            sid_col = Name of the station ID column
            name_col = Name of the station name column
        Returns: Dictionary of the station IDs and names"""
    stations = pd.read_csv(station_file, dtype={sid_col: str})
    return dict(zip(stations[sid_col].astype(str), stations[name_col]))

def plot_month(filename, stationid, *, q_low=0.05, q_high=0.95, ylabel="Temperature (°C)", obs_mode="scatter", nearest_tolerance="90min", pred_resample=None):
    """
    Plot one month of observation vs predictions for a single station.
        Params: 
            filename = Name of the file 
            stationid = Station ID
            q_low = Lower quantile
            q_high = Upper quantile
            ylabel = Title for the y-label
            obs_mode = Mode for the observations 'scatter' | 'nearest_to_pred' | 'interp_to_pred'
            nearest_tolerance = Tolerance for aggregation; only used when obs_mode='nearest_to_pred'
            pred_resample = Prediction resample e.g. '3H' to aggregate predictions to 3-hour cadence
    """
    # Read the file into a data frame
    path = MY_DATA_DIR / filename
    df = pd.read_parquet(path)

    # Filter station based on station ID
    df = df[df["SID"].astype(str) == str(stationid)].copy()
    if df.empty:
        raise ValueError(f"No rows for SID={stationid} in {path.name}")

    # Check for required columns
    for col in ("validtime", "obs_TA", "T2"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}'.")

    # Change valid time to pandas datetime 
    df["validtime"] = pd.to_datetime(df["validtime"])

    # Change temperature from kelvin to celsius
    df["obs_C"] = df["obs_TA"] - 273.15
    df["T2_C"] = df["T2"] - 273.15

    # Observation (one per time — drop duplicates if repeated)
    obs = (
        df[["validtime", "obs_C"]]
        .drop_duplicates(subset=["validtime"])
        .set_index("validtime")
        .sort_index()
    )

    # Aggregate predictions at each timestamp to median + ribbon
    preds = df.set_index("validtime").sort_index()

    # Different aggregation method if pred_resample given
    #  Aggregate predictions to match a coarser cadence (e.g., '3H')
    if pred_resample: 
        agg = (
            preds["T2_C"]
            .resample(pred_resample)
            .agg(
                median="median",
                lo=lambda x: x.quantile(q_low) if len(x) else float("nan"),
                hi=lambda x: x.quantile(q_high) if len(x) else float("nan"),
            )
            .dropna(how="all")
        )
    # No resampling 
    else:
        agg = (
            preds.groupby(level=0)["T2_C"]
            .agg(
                median="median",
                lo=lambda x: x.quantile(q_low),
                hi=lambda x: x.quantile(q_high),
            )
            .sort_index()
        )

    # Title bits
    try:
        month_label = pd.to_datetime(filename.split("_")[-1].split(".")[0]).strftime("%B %Y")
    except Exception:
        month_label = "Selected month"
    station_names = load_station_names()
    station_name = station_names.get(str(stationid), "Unknown station")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(agg.index, agg["lo"], agg["hi"], alpha=0.2,
                    label=f"T2 {int(q_low*100)}–{int(q_high*100)}%")
    ax.plot(agg.index, agg["median"], linewidth=2, label="T2 median")

    # Observations: choose display mode
    # Scatter the observations 
    if obs_mode == "scatter":
        ax.scatter(obs.index, obs["obs_C"], s=18, label="Observation", zorder=5)

    # Snap obs to the prediction time grid using nearest within tolerance
    elif obs_mode == "nearest_to_pred":
        grid = agg.index
        tol = pd.Timedelta(nearest_tolerance)
        obs_on_grid = obs.reindex(grid, method="nearest", tolerance=tol)
        ax.plot(obs_on_grid.index, obs_on_grid["obs_C"], linewidth=2.5,
                label=f"Observation (nearest ≤ {nearest_tolerance})", zorder=5)
        # also show the raw sparse obs as faint points
        ax.scatter(obs.index, obs["obs_C"], s=14, alpha=0.6, label="Obs (raw)", zorder=6)

    # Interpolate obs across time onto the prediction grid
    elif obs_mode == "interp_to_pred":
        grid = agg.index
        obs_interp = obs.reindex(grid)
        obs_interp["obs_C"] = obs_interp["obs_C"].interpolate(
            method="time", limit_direction="both"
        )
        ax.plot(obs_interp.index, obs_interp["obs_C"], linewidth=2.5,
                label="Observation (interpolated)", zorder=5)
        ax.scatter(obs.index, obs["obs_C"], s=14, alpha=0.6, label="Obs (raw)", zorder=6)
    
    # Raise error if not correct obs_mode
    else:
        raise ValueError("obs_mode must be 'scatter', 'nearest_to_pred', or 'interp_to_pred'.")

    # Plot parameters
    ax.set_title(f"{station_name} (SID {stationid}) — {month_label}")
    ax.set_xlabel("Valid time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    # Save the plot
    out_path = OUT / f"{Path(filename).stem}_{stationid}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved {out_path}")

def main():

    # Filename 
    filename = "ml_data_2023-07.parquet"

    # Station ID
    stationid = "116245"

    # Different plotting modes 
    plot_month(filename, stationid, obs_mode="scatter")                 # points at 3-hour times
    # plot_month(filename, stationid, obs_mode="nearest_to_pred")       # snap obs to pred grid
    # plot_month(filename, stationid, obs_mode="interp_to_pred")        # continuous obs line
    # plot_month(filename, stationid, obs_mode="scatter", pred_resample="3H")  # also downsample preds

if __name__ == "__main__":
    main()


 