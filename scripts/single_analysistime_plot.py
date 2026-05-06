import argparse
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# Columns
# ===============================

SPLIT = "validtime"
KEYS = ["SID", SPLIT, "analysistime", "leadtime"]

OBS = "obs_TA"
RAW = "raw_fc"
MOS = "corrected_mos"


# These are set from command-line arguments in main()
ML_TAG = None
ML_NAME = None
GNN_TAG = None
GNN_NAME = None
LSTM_TAG = None
LSTM_NAME = None

MLCOL = None
GNNCOL = None
LSTMCOL = None


# ===============================
# Argument parsing
# ===============================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot time series for one station/init or anomalous station/init cases, "
            "comparing raw forecast, MOS, XGBoost/ML, GNN, LSTM, and observations."
        )
    )

    parser.add_argument(
        "--station-file",
        required=True,
        type=str,
        help="Path to stations.csv. Used to map SID to station name.",
    )

    parser.add_argument(
        "--mos-dir",
        required=True,
        type=str,
        help="Directory containing MOS evaluation parquet files.",
    )

    parser.add_argument(
        "--ml-dir",
        required=True,
        type=str,
        help="Directory containing ML/XGBoost evaluation parquet files.",
    )

    parser.add_argument(
        "--gnn-dir",
        required=True,
        type=str,
        help="Directory containing GNN evaluation parquet files.",
    )

    parser.add_argument(
        "--lstm-dir",
        required=True,
        type=str,
        help="Directory containing LSTM evaluation parquet files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where plots will be saved.",
    )

    parser.add_argument(
        "--ml-tag",
        default="tuned_full",
        type=str,
        help="ML/XGBoost tag used in corrected_<tag> column and filenames.",
    )

    parser.add_argument(
        "--ml-name",
        default="EC_ML_XGBoost",
        type=str,
        help="Readable ML/XGBoost model name used in plot labels.",
    )

    parser.add_argument(
        "--gnn-tag",
        default="full_gnn_gat",
        type=str,
        help="GNN tag used in corrected_<tag> column and filenames.",
    )

    parser.add_argument(
        "--gnn-name",
        default="EC_ML_GNN",
        type=str,
        help="Readable GNN model name used in plot labels.",
    )

    parser.add_argument(
        "--lstm-tag",
        default="bias_lstm_stream",
        type=str,
        help="LSTM tag used in corrected_<tag> column and filenames.",
    )

    parser.add_argument(
        "--lstm-name",
        default="EC_ML_LSTM",
        type=str,
        help="Readable LSTM model name used in plot labels.",
    )

    parser.add_argument(
        "--target-station",
        default="101932",
        type=str,
        help="Station ID to plot in non-anomaly mode. Default: 101932.",
    )

    parser.add_argument(
        "--inits",
        nargs="+",
        default=[
            "2024-12-13 12:00:00",
            "2024-12-13 00:00:00",
            "2025-02-01 12:00:00",
            "2025-02-01 00:00:00",
            "2025-07-16 12:00:00",
            "2025-07-16 00:00:00",
        ],
        help=(
            "One or more analysistime values to plot in non-anomaly mode. "
            "Example: --inits '2024-12-13 12:00:00' '2025-02-01 00:00:00'"
        ),
    )

    parser.add_argument(
        "--anom",
        action="store_true",
        help="Use anomalous analysistimes from --anom-csv instead of --target-station/--inits.",
    )

    parser.add_argument(
        "--anom-csv",
        default=None,
        type=str,
        help=(
            "CSV file for anomaly mode. Expected columns: SID, anomalous_times_str. "
            "Required if --anom is used."
        ),
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show plots interactively.",
    )

    parser.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Do not save plots to disk.",
    )

    parser.add_argument(
        "--fig-dpi",
        default=150,
        type=int,
        help="Figure DPI. Default: 150.",
    )

    return parser.parse_args()


# ===============================
# Helpers
# ===============================

def load_station_names(station_file: Path, sid_col="SID", name_col="name"):
    """
    Load station names from CSV.

    Returns:
        Dictionary mapping station IDs to names.
    """
    stations = pd.read_csv(station_file, dtype={sid_col: str})

    if sid_col not in stations.columns:
        raise ValueError(f"Station file is missing SID column: {sid_col}")

    if name_col not in stations.columns:
        print(f"[WARN] Station file has no '{name_col}' column. Using SID as station name.")
        return dict(zip(stations[sid_col].astype(str), stations[sid_col].astype(str)))

    return dict(zip(stations[sid_col].astype(str), stations[name_col]))


def rmse(a, b):
    """Calculate RMSE for two temperature series."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    m = np.isfinite(a) & np.isfinite(b)

    if not m.any():
        return np.nan

    d = a[m] - b[m]

    return float(np.sqrt(np.mean(d * d)))


def safe_init_name(target_init: str) -> str:
    return (
        str(target_init)
        .replace(":", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


# ===============================
# Data loading and preparation
# ===============================

def load_eval_rows_evaldir(eval_dir: Path, pattern: str, tag: str):
    """
    Load evaluation rows from parquet files.
    """
    files = sorted(eval_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")

    dfs = []

    for f in files:
        cols = KEYS + [RAW, OBS, MLCOL, MOS, GNNCOL, LSTMCOL]

        schema = pl.scan_parquet(str(f)).collect_schema().names()
        existing = [c for c in cols if c in schema]

        if not existing:
            continue

        df = pl.read_parquet(f, columns=existing)

        exprs = []

        if "SID" in df.columns:
            exprs.append(pl.col("SID").cast(pl.Utf8))

        if SPLIT in df.columns:
            exprs.append(pl.col(SPLIT).cast(pl.Utf8))

        if "analysistime" in df.columns:
            exprs.append(pl.col("analysistime").cast(pl.Utf8))

        if "leadtime" in df.columns:
            exprs.append(pl.col("leadtime").cast(pl.Int64, strict=False))

        if exprs:
            df = df.with_columns(exprs)

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No usable columns found in files matching: {eval_dir}/{pattern}")

    out = pl.concat(dfs, how="vertical_relaxed")

    print(f"[INFO] {tag} rows loaded: {out.height:,}")

    return out


def filter_for_station_init(
    df: pl.DataFrame,
    target_station: str,
    target_init: str,
    tag: str,
):
    """
    Filter dataframe to the wanted station and analysistime.
    """
    plot_df = df.filter(
        (pl.col("SID") == pl.lit(str(target_station)).cast(pl.Utf8))
        & (pl.col("analysistime") == pl.lit(str(target_init)).cast(pl.Utf8))
    )

    return plot_df


def leadtime_coverage(df: pl.DataFrame, name: str):
    """Print leadtime coverage for a model."""
    if df.height == 0:
        print(f"[COV] {name}: rows=0, unique leadtimes=0")
        return

    print(
        f"[COV] {name}: "
        f"rows={df.height}, "
        f"unique leadtimes={df.select(pl.col('leadtime').n_unique()).item()}"
    )


def join_model_outputs(
    mos_plot: pl.DataFrame,
    ml_plot: pl.DataFrame,
    gnn_plot: pl.DataFrame,
    lstm_plot: pl.DataFrame,
):
    """
    Join available model outputs for one station/init.

    Chooses a base dataframe that exists, then left-joins the others.
    """
    if ml_plot.height > 0:
        joined = ml_plot
        base_name = "ML"
    elif gnn_plot.height > 0:
        joined = gnn_plot
        base_name = "GNN"
    elif lstm_plot.height > 0:
        joined = lstm_plot
        base_name = "LSTM"
    elif mos_plot.height > 0:
        joined = mos_plot
        base_name = "MOS"
    else:
        return pl.DataFrame(), "none"

    print(f"[INFO] Using {base_name} as join base")

    if mos_plot.height > 0 and base_name != "MOS":
        joined = joined.join(
            mos_plot.select(KEYS + [MOS]).unique(subset=KEYS),
            on=KEYS,
            how="left",
        )

    if ml_plot.height > 0 and base_name != "ML":
        joined = joined.join(
            ml_plot.select(KEYS + [MLCOL, RAW, OBS]).unique(subset=KEYS),
            on=KEYS,
            how="left",
            suffix="_ml",
        )

        for col in [RAW, OBS]:
            dup = f"{col}_ml"
            if dup in joined.columns and col in joined.columns:
                joined = joined.drop(dup)

    if gnn_plot.height > 0 and base_name != "GNN":
        joined = joined.join(
            gnn_plot.select(KEYS + [GNNCOL, RAW, OBS]).unique(subset=KEYS),
            on=KEYS,
            how="left",
            suffix="_gnn",
        )

        for col in [RAW, OBS]:
            dup = f"{col}_gnn"
            if dup in joined.columns and col in joined.columns:
                joined = joined.drop(dup)

    if lstm_plot.height > 0 and base_name != "LSTM":
        joined = joined.join(
            lstm_plot.select(KEYS + [LSTMCOL, RAW, OBS]).unique(subset=KEYS),
            on=KEYS,
            how="left",
            suffix="_lstm",
        )

        for col in [RAW, OBS]:
            dup = f"{col}_lstm"
            if dup in joined.columns and col in joined.columns:
                joined = joined.drop(dup)

    return joined, base_name


def prepare_plot_dataframe(joined: pl.DataFrame) -> pd.DataFrame:
    """
    Convert joined Polars dataframe to Pandas and sort by validtime.
    """
    plot_pd = joined.to_pandas()

    plot_pd = plot_pd.sort_values("validtime")
    plot_pd["validtime"] = pd.to_datetime(plot_pd["validtime"], errors="coerce")

    return plot_pd


# ===============================
# Plot function
# ===============================

def plot_timeseries(
    df: pd.DataFrame,
    target_station: str,
    target_init: str,
    station_name: str,
    out_dir: Path,
    save_plot: bool,
    show_plot: bool,
    fig_dpi: int,
):
    """
    Plot temperature values for models and observations for one station/init.
    """
    text_lines = []

    if OBS in df.columns:
        if RAW in df.columns:
            text_lines.append(f"RMSE ECMWF vs OBS: {rmse(df[RAW], df[OBS]):.3f}")

        if MOS in df.columns:
            text_lines.append(f"RMSE MOS vs OBS: {rmse(df[MOS], df[OBS]):.3f}")

        if MLCOL in df.columns:
            text_lines.append(f"RMSE {ML_NAME} vs OBS: {rmse(df[MLCOL], df[OBS]):.3f}")

        if GNNCOL in df.columns:
            text_lines.append(f"RMSE {GNN_NAME} vs OBS: {rmse(df[GNNCOL], df[OBS]):.3f}")

        if LSTMCOL in df.columns:
            text_lines.append(f"RMSE {LSTM_NAME} vs OBS: {rmse(df[LSTMCOL], df[OBS]):.3f}")

    plt.figure(figsize=(12, 5))

    if RAW in df.columns:
        plt.plot(
            df["validtime"],
            df[RAW],
            label="ECMWF",
            linewidth=1.5,
            color="#999999",
        )

    if MOS in df.columns:
        plt.plot(
            df["validtime"],
            df[MOS],
            label="MOS",
            linewidth=2,
            color="#637AB9",
        )

    if MLCOL in df.columns:
        plt.plot(
            df["validtime"],
            df[MLCOL],
            label=ML_NAME,
            linewidth=2,
            color="#B95E82",
        )

    if GNNCOL in df.columns:
        plt.plot(
            df["validtime"],
            df[GNNCOL],
            label=GNN_NAME,
            linewidth=2,
            color="#7DB288",
        )

    if LSTMCOL in df.columns:
        plt.plot(
            df["validtime"],
            df[LSTMCOL],
            label=LSTM_NAME,
            linewidth=2,
            color="#D18F49",
        )

    if OBS in df.columns and df[OBS].notna().any():
        mask = df[OBS].notna()

        plt.scatter(
            df.loc[mask, "validtime"],
            df.loc[mask, OBS],
            s=25,
            marker="o",
            color="black",
            label="Observation",
        )

    plt.title(f"{station_name} {target_station} — Init {target_init}")
    plt.xlabel("Valid time")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if text_lines:
        textstr = "\n".join(text_lines)
        ax = plt.gca()

        ax.text(
            0.98,
            0.02,
            textstr,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.8,
                edgecolor="0.5",
            ),
        )

    plt.tight_layout()

    safe_init = safe_init_name(target_init)
    station_out_dir = out_dir / f"{target_station}"
    station_out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = station_out_dir / f"timeseries_MOS_ML_{target_station}_{safe_init}.svg"
    pdf_path = station_out_dir / f"timeseries_MOS_ML_{target_station}_{safe_init}.pdf"

    if save_plot:
        plt.savefig(svg_path, dpi=fig_dpi, bbox_inches="tight")
        plt.savefig(pdf_path, dpi=fig_dpi, bbox_inches="tight")
        print(f"[OK] Saved plot to: {station_out_dir}")

    if show_plot:
        plt.show()

    plt.close()


# ===============================
# Main workflow helpers
# ===============================

def process_one_case(
    target_station: str,
    target_init: str,
    station_names: dict[str, str],
    mos_all: pl.DataFrame,
    ml_all: pl.DataFrame,
    gnn_all: pl.DataFrame,
    lstm_all: pl.DataFrame,
    out_dir: Path,
    save_plot: bool,
    show_plot: bool,
    fig_dpi: int,
):
    station_name = station_names.get(str(target_station), str(target_station))

    mos_plot = filter_for_station_init(mos_all, target_station, target_init, MOS)
    ml_plot = filter_for_station_init(ml_all, target_station, target_init, MLCOL)
    gnn_plot = filter_for_station_init(gnn_all, target_station, target_init, GNNCOL)
    lstm_plot = filter_for_station_init(lstm_all, target_station, target_init, LSTMCOL)

    leadtime_coverage(mos_plot, "MOS")
    leadtime_coverage(ml_plot, "ML")
    leadtime_coverage(gnn_plot, "GNN")
    leadtime_coverage(lstm_plot, "LSTM")

    joined, base_name = join_model_outputs(
        mos_plot=mos_plot,
        ml_plot=ml_plot,
        gnn_plot=gnn_plot,
        lstm_plot=lstm_plot,
    )

    if joined.height == 0:
        print(
            f"[WARN] No rows for station={target_station}, "
            f"analysistime={target_init}; skipping."
        )
        return

    plot_pd = prepare_plot_dataframe(joined)

    print(f"[INFO] Aligned rows for plot: {len(plot_pd)}")
    print(f"[INFO] Columns present: {sorted(plot_pd.columns)}")

    plot_timeseries(
        plot_pd,
        target_station=target_station,
        target_init=target_init,
        station_name=station_name,
        out_dir=out_dir,
        save_plot=save_plot,
        show_plot=show_plot,
        fig_dpi=fig_dpi,
    )


# ===============================
# Main
# ===============================

def main():
    global ML_TAG, ML_NAME, GNN_TAG, GNN_NAME, LSTM_TAG, LSTM_NAME
    global MLCOL, GNNCOL, LSTMCOL

    args = parse_args()

    station_file = Path(args.station_file)
    mos_dir = Path(args.mos_dir)
    ml_dir = Path(args.ml_dir)
    gnn_dir = Path(args.gnn_dir)
    lstm_dir = Path(args.lstm_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not station_file.exists():
        raise FileNotFoundError(f"Station file not found: {station_file}")

    ML_TAG = args.ml_tag
    ML_NAME = args.ml_name

    GNN_TAG = args.gnn_tag
    GNN_NAME = args.gnn_name

    LSTM_TAG = args.lstm_tag
    LSTM_NAME = args.lstm_name

    MLCOL = f"corrected_{ML_TAG}"
    GNNCOL = f"corrected_{GNN_TAG}"
    LSTMCOL = f"corrected_{LSTM_TAG}"

    save_plot = not args.no_save_plot
    show_plot = args.show_plot
    fig_dpi = args.fig_dpi

    if args.anom and args.anom_csv is None:
        raise ValueError("--anom-csv is required when using --anom.")

    print(f"[INFO] MOS directory: {mos_dir}")
    print(f"[INFO] ML directory: {ml_dir}")
    print(f"[INFO] GNN directory: {gnn_dir}")
    print(f"[INFO] LSTM directory: {lstm_dir}")
    print(f"[INFO] Output directory: {out_dir}")

    print(f"[INFO] ML column: {MLCOL}")
    print(f"[INFO] GNN column: {GNNCOL}")
    print(f"[INFO] LSTM column: {LSTMCOL}")

    station_names = load_station_names(station_file)

    # Load data from MOS and all ML models
    mos_all = load_eval_rows_evaldir(
        mos_dir,
        f"eval_rows_{SPLIT}_MOS_*.parquet",
        MOS,
    )

    ml_all = load_eval_rows_evaldir(
        ml_dir,
        f"eval_rows_{SPLIT}_{ML_TAG}_20*.parquet",
        MLCOL,
    )

    gnn_all = load_eval_rows_evaldir(
        gnn_dir,
        f"eval_rows_{SPLIT}_{GNN_TAG}_20*.parquet",
        GNNCOL,
    )

    lstm_all = load_eval_rows_evaldir(
        lstm_dir,
        f"eval_rows_analysistime_{LSTM_TAG}_20??_fin.parquet",
        LSTMCOL,
    )

    if args.anom:
        anom_csv = Path(args.anom_csv)

        if not anom_csv.exists():
            raise FileNotFoundError(f"Anomaly CSV not found: {anom_csv}")

        df = pl.read_csv(anom_csv)

        required_cols = {"SID", "anomalous_times_str"}
        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"Anomaly CSV is missing required columns: {sorted(missing)}")

        for row in df.iter_rows(named=True):
            target_station = str(row["SID"])
            inits = str(row["anomalous_times_str"]).split(", ")

            for target_init in inits:
                process_one_case(
                    target_station=target_station,
                    target_init=target_init,
                    station_names=station_names,
                    mos_all=mos_all,
                    ml_all=ml_all,
                    gnn_all=gnn_all,
                    lstm_all=lstm_all,
                    out_dir=out_dir,
                    save_plot=save_plot,
                    show_plot=show_plot,
                    fig_dpi=fig_dpi,
                )

    else:
        target_station = str(args.target_station)

        for target_init in args.inits:
            process_one_case(
                target_station=target_station,
                target_init=target_init,
                station_names=station_names,
                mos_all=mos_all,
                ml_all=ml_all,
                gnn_all=gnn_all,
                lstm_all=lstm_all,
                out_dir=out_dir,
                save_plot=save_plot,
                show_plot=show_plot,
                fig_dpi=fig_dpi,
            )


if __name__ == "__main__":
    main()


