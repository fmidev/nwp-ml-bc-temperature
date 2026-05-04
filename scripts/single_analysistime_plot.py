
import os
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your ML model tag used when saving eval rows (=> column "corrected_<ML_TAG>")
ML_TAG = "tuned_full" 
# Name of the model for the plot  
ML_NAME = "EC_ML_XGBoost"     

GNN_TAG = "full_gnn_gat"
GNN_NAME = "EC_ML_GNN"

LSTM_TAG = "bias_lstm_stream"
LSTM_NAME = "EC_ML_LSTM"

# Plot settings
SHOW_PLOT = False
SAVE_PLOT = True
FIG_DPI   = 150

# Paths
HOME     = Path.home()
STATION_FILE = HOME / "thesis_project" / "data" / "stations.csv"
METRICS  = HOME / "thesis_project" / "metrics"
MOS_DIR  = METRICS / "mos"
ML_DIR   = METRICS / "tuned_full_new"
GNN_DIR = METRICS / "full_gnn_gat"     
LSTM_DIR = METRICS / "bias_lstm_stream"                       
OUT_DIR  = HOME / "thesis_project" / "figures" / "MOSvsML_timeseries" / "single_analysistime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns
SPLIT = "validtime"
KEYS  = ["SID", SPLIT, "analysistime", "leadtime"]
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS   = "corrected_mos"
MLCOL = f"corrected_{ML_TAG}"
GNNCOL = f"corrected_{GNN_TAG}"
LSTMCOL = f"corrected_{LSTM_TAG}"

ANOM = False

# -------------------------------
# Helpers
# -------------------------------

def load_station_names(station_file=STATION_FILE, sid_col="SID", name_col="name"):
    """ Helper function to load stations names from csv file
        Params:
            station_file = Path and file name of the csv file containing the station information
            sid_col = Name of the station ID column
            name_col = Name of the station name column
        Returns: Dictionary of the station IDs and names"""
    stations = pd.read_csv(station_file, dtype={sid_col: str})
    return dict(zip(stations[sid_col].astype(str), stations[name_col]))


def rmse(a, b):
    """Calculate the rmse for two sets of temperatures"""
    a = np.asarray(a, float) 
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d*d)))


# ------------------------------------------
# Data loading and preparation functions
# ------------------------------------------

def load_eval_rows_evaldir(eval_dir: Path, pattern: str, tag: str):
    """Load the data
        Params: 
            eval_dir = Directory of the files
            pattern = Filename pattern
            tag = Name of the column/model data is loaded for
        Return: Loaded data
    """
    # Gather the files
    files = sorted(eval_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")
  
    # Loop through files and collect needed data
    dfs = []
    for f in files:
        cols = KEYS + [RAW, OBS, MLCOL, MOS, GNNCOL, LSTMCOL]
        # tolerate older files lacking RAW (we’ll drop if missing)
        existing = [c for c in cols if c in pl.scan_parquet(str(f)).collect_schema().keys()]
        df = pl.read_parquet(f, columns=existing).with_columns([
            pl.col("SID").cast(pl.Utf8),
            pl.col(SPLIT).cast(pl.Utf8),
            pl.col("analysistime").cast(pl.Utf8),
        ])
        dfs.append(df)

    # Concatenate the into one dataframe
    out = pl.concat(dfs, how="vertical_relaxed")

    print(f"[INFO] {tag} rows loaded: {out.height:,}")
    return out

def filter_for_station_init(df: pl.DataFrame, target_station: str, target_init: str, tag: str):
    """Filter the dataframe to the wanted station and analysistime
        Params: 
            df = Dataframe
            target_station = SID of the wanted target station
            target_init = Date and time of the wanted analysistime
            tag = Name of the column/model data is loaded for
        Returns: Filtered dataframe"""

    # Filter data to the target station and analysistime
    plot_df = df.filter(
        (pl.col("SID") == pl.lit(str(target_station)).cast(pl.Utf8)) &
        (pl.col("analysistime") == pl.lit(str(target_init)).cast(pl.Utf8))
    )
    
    #if plot_df.height == 0:
    #    raise ValueError(f"No {tag} rows for SID={target_station}, init={target_init}")
    return plot_df

def leadtime_coverage(df, name):
    """Print the leadtime coverage for the model"""
    print(f"[COV] {name}: rows={df.height}, unique leadtimes={df.select(pl.col('leadtime').n_unique()).item()}")


# ----------------------
# Plot function
#-----------------------
def plot(df: pd.DataFrame, target_station: str, target_init: str, station_name: str):
    """Plot the temperature values for the models and observations for the chosen station and analysistime
        Params:
            df = Dataframe with the temperature values
            target_station = SID of chosen station
            target_init = Date and time of chosen analysistime
    """

    # Per-series RMSE vs obs (for this station/init only)
    if OBS in df.columns:
        text_lines = []
        if RAW in df.columns:
            text_lines.append(f"RMSE ECMWF vs OBS: {rmse(df[RAW], df[OBS]):.3f}")
        if MOS in df.columns:
            text_lines.append(f"RMSE MOS vs OBS: {rmse(df[MOS], df[OBS]):.3f}")
        if MLCOL in df.columns:
            text_lines.append(f"RMSE {ML_NAME}  vs OBS: {rmse(df[MLCOL], df[OBS]):.3f}")
        if GNNCOL in df.columns:
            text_lines.append(f"RMSE {GNN_NAME} vs OBS: {rmse(df[GNNCOL], df[OBS]):.3f}")
        if LSTMCOL in df.columns:
            text_lines.append(f"RMSE {LSTM_NAME} vs OBS: {rmse(df[LSTMCOL], df[OBS]):.3f}")

    plt.figure(figsize=(12, 5))

    # Plot the temperatures (models as timeseries and observations as points)
    if RAW in df.columns:
        plt.plot(df["validtime"], df[RAW], label="ECMWF", linewidth=1.5, color="#999999")
    if MOS in df.columns:
        plt.plot(df["validtime"], df[MOS], label="MOS", linewidth=2, color="#637AB9")
    if MLCOL in df.columns:
        plt.plot(df["validtime"], df[MLCOL], label=f"{ML_NAME}", linewidth=2, color="#B95E82")
    if GNNCOL in df.columns:
        plt.plot(df["validtime"], df[GNNCOL], label=f"{GNN_NAME}", linewidth=2, color="#7DB288")
    if LSTMCOL in df.columns:
        plt.plot(df["validtime"], df[LSTMCOL], label=f"{LSTM_NAME}", linewidth=2, color="#D18F49")

    if OBS in df.columns and df[OBS].notna().any():
        mask = df[OBS].notna()
        plt.scatter(df.loc[mask, "validtime"], df.loc[mask, OBS],
                    s=25, marker="o", color="black", label="Observation")

    # Plot parameters
    plt.title(f"{station_name} {target_station} — Init {target_init}")
    plt.xlabel("Valid time")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if text_lines:
        textstr = "\n".join(text_lines)
        ax = plt.gca()
        ax.text(
            0.98, 0.02, textstr,
            transform=ax.transAxes, va="bottom", ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="0.5")
        )
    plt.tight_layout()

    # Save
    safe_init = target_init.replace(":", "").replace(" ", "_")
    out_path = OUT_DIR / f"{target_station}"
    out_path.mkdir(parents=True, exist_ok=True)
    if SAVE_PLOT:
        plt.savefig(out_path / f"timeseries_MOS_ML_{target_station}_{safe_init}.svg", dpi=FIG_DPI, bbox_inches="tight")
        plt.savefig(out_path / f"timeseries_MOS_ML_{target_station}_{safe_init}.pdf", dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved plot → {out_path}")
        plt.close()


def main():

    
    station_names = load_station_names()
    # Load data from MOS and ML model
    mos_all = load_eval_rows_evaldir(MOS_DIR, f"eval_rows_{SPLIT}_MOS_*.parquet", MOS)
    ml_all = load_eval_rows_evaldir(ML_DIR, f"eval_rows_{SPLIT}_{ML_TAG}_20*.parquet", MLCOL)
    gnn_all = load_eval_rows_evaldir(GNN_DIR, f"eval_rows_{SPLIT}_{GNN_TAG}_20*.parquet", GNNCOL)
    lstm_all = load_eval_rows_evaldir(LSTM_DIR, f"eval_rows_analysistime_{LSTM_TAG}_20??_fin.parquet", LSTMCOL)

    if ANOM:
        df = pl.read_csv(METRICS / "all_anomalous_analysistimes_by_station.csv")


        for row in df.iter_rows(named=True):

            # List of wanted stations (string)
            target_station = row["SID"] 

            # List of wanted analysistimes (string)
            inits = row["anomalous_times_str"].split(", ")



            for target_init in inits:

                station_name = station_names.get(str(target_station))
                # Filter for the certain station and analysistime
                mos_plot = filter_for_station_init(mos_all, target_station, target_init, MOS)
                ml_plot = filter_for_station_init(ml_all, target_station, target_init, MLCOL)
                gnn_plot = filter_for_station_init(gnn_all, target_station, target_init, GNNCOL)
                lstm_plot = filter_for_station_init(lstm_all, target_station, target_init, LSTMCOL)

                # Print the leadtime coverage to check
                leadtime_coverage(mos_plot, "MOS")
                leadtime_coverage(ml_plot,  "ML")
                leadtime_coverage(gnn_plot, "GNN")
                leadtime_coverage(lstm_plot)

                # Pick a base that exists (prefer ML, then GNN, then MOS)
                if ml_plot.height > 0:
                    joined = ml_plot
                    base_name = "ML"
                elif gnn_plot.height > 0:
                    joined = gnn_plot
                    base_name = "GNN"
                elif lstm_plot.height > 0:
                    joined = lstm_plot
                    base_name = "LSTM"
                else:
                    joined = mos_plot
                    base_name = "MOS"

                print(f"[INFO] Using {base_name} as join base")

                # Join MOS corrected (if present and not already base)
                if mos_plot.height > 0 and base_name != "MOS":
                    joined = joined.join(
                        mos_plot.select(KEYS + [MOS]).unique(subset=KEYS),
                        on=KEYS,
                        how="left"
                    )

                # Join ML corrected (if present and not already base)
                if ml_plot.height > 0 and base_name != "ML":
                    joined = joined.join(
                        ml_plot.select(KEYS + [MLCOL, RAW, OBS]).unique(subset=KEYS),
                        on=KEYS,
                        how="left",
                        suffix="_ml"
                    )
                    # If duplicates were created, drop suffixed RAW/OBS
                    for col in [RAW, OBS]:
                        dup = f"{col}_ml"
                        if dup in joined.columns and col in joined.columns:
                            joined = joined.drop(dup)

                # Join GNN corrected (if present and not already base)
                if gnn_plot.height > 0 and base_name != "GNN":
                    joined = joined.join(
                        gnn_plot.select(KEYS + [GNNCOL, RAW, OBS]).unique(subset=KEYS),
                        on=KEYS,
                        how="left",
                        suffix="_gnn"
                    )
                    for col in [RAW, OBS]:
                        dup = f"{col}_gnn"
                        if dup in joined.columns and col in joined.columns:
                            joined = joined.drop(dup)
                
                # Join LSTM corrected (if present and not already base)
                if lstm_plot.height > 0 and base_name != "LSTM":
                    joined = joined.join(
                        lstm_plot.select(KEYS + [LSTMCOL, RAW, OBS]).unique(subset=KEYS),
                        on=KEYS,
                        how="left",
                        suffix="_lstm"
                    )
                    for col in [RAW, OBS]:
                        dup = f"{col}_lstm"
                        if dup in joined.columns and col in joined.columns:
                            joined = joined.drop(dup)


                # Transfer the polars into pandas and prepare the data for plotting 
                plot_pd = joined.to_pandas()
                plot_pd = plot_pd.sort_values("validtime")
                plot_pd["validtime"] = pd.to_datetime(plot_pd["validtime"])

                # Sanity prints
                print(f"[INFO] Aligned rows for plot: {len(plot_pd)}")
                print(f"[INFO] Columns present: {sorted(plot_pd.columns)}")

                # Make the figure
                plot(plot_pd, target_station, target_init, station_name)
    else:

        inits = ["2024-12-13 12:00:00", "2024-12-13 00:00:00", "2025-02-01 12:00:00", "2025-02-01 00:00:00", "2025-07-16 12:00:00", "2025-07-16 00:00:00"]
        target_station = "101932"

        for target_init in inits:

            station_name = station_names.get(str(target_station))
            # Filter for the certain station and analysistime
            mos_plot = filter_for_station_init(mos_all, target_station, target_init, MOS)
            ml_plot = filter_for_station_init(ml_all, target_station, target_init, MLCOL)
            gnn_plot = filter_for_station_init(gnn_all, target_station, target_init, GNNCOL)
            lstm_plot = filter_for_station_init(lstm_all, target_station, target_init, LSTMCOL)

            # Print the leadtime coverage to check
            leadtime_coverage(mos_plot, "MOS")
            leadtime_coverage(ml_plot,  "ML")
            leadtime_coverage(gnn_plot, "GNN")
            leadtime_coverage(lstm_plot, "LSTM")

            # Pick a base that exists (prefer ML, then GNN, then MOS)
            if ml_plot.height > 0:
                joined = ml_plot
                base_name = "ML"
            elif gnn_plot.height > 0:
                joined = gnn_plot
                base_name = "GNN"
            elif lstm_plot.height > 0:
                joined = lstm_plot
                base_name = "LSTM"
            else:
                joined = mos_plot
                base_name = "MOS"

            print(f"[INFO] Using {base_name} as join base")

            # Join MOS corrected (if present and not already base)
            if mos_plot.height > 0 and base_name != "MOS":
                joined = joined.join(
                    mos_plot.select(KEYS + [MOS]).unique(subset=KEYS),
                    on=KEYS,
                    how="left"
                )

            # Join ML corrected (if present and not already base)
            if ml_plot.height > 0 and base_name != "ML":
                joined = joined.join(
                    ml_plot.select(KEYS + [MLCOL, RAW, OBS]).unique(subset=KEYS),
                    on=KEYS,
                    how="left",
                    suffix="_ml"
                )
                # If duplicates were created, drop suffixed RAW/OBS
                for col in [RAW, OBS]:
                    dup = f"{col}_ml"
                    if dup in joined.columns and col in joined.columns:
                        joined = joined.drop(dup)

            # Join GNN corrected (if present and not already base)
            if gnn_plot.height > 0 and base_name != "GNN":
                joined = joined.join(
                    gnn_plot.select(KEYS + [GNNCOL, RAW, OBS]).unique(subset=KEYS),
                    on=KEYS,
                    how="left",
                    suffix="_gnn"
                )
                for col in [RAW, OBS]:
                    dup = f"{col}_gnn"
                    if dup in joined.columns and col in joined.columns:
                        joined = joined.drop(dup)

            # Join LSTM corrected (if present and not already base)
            if lstm_plot.height > 0 and base_name != "LSTM":
                joined = joined.join(
                    lstm_plot.select(KEYS + [LSTMCOL, RAW, OBS]).unique(subset=KEYS),
                    on=KEYS,
                    how="left",
                    suffix="_lstm"
                )
                for col in [RAW, OBS]:
                    dup = f"{col}_lstm"
                    if dup in joined.columns and col in joined.columns:
                        joined = joined.drop(dup)

            # Transfer the polars into pandas and prepare the data for plotting 
            plot_pd = joined.to_pandas()
            plot_pd = plot_pd.sort_values("validtime")
            plot_pd["validtime"] = pd.to_datetime(plot_pd["validtime"])

            # Sanity prints
            print(f"[INFO] Aligned rows for plot: {len(plot_pd)}")
            print(f"[INFO] Columns present: {sorted(plot_pd.columns)}")

            # Make the figure
            plot(plot_pd, target_station, target_init, station_name)

if __name__ == "__main__":
    main()


