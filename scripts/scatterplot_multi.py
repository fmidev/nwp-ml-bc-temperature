import os
from pathlib import Path
import calendar
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator


# User settings
ML_TAG = "tuned_ah_2019"
ML_NAME = "EC_ML_XGBoost_2019"

SHOW_PLOT = False
SAVE_PLOT = True
FIG_DPI   = 150


# Directories
HOME     = Path.home()
STATION_FILE = HOME / "thesis_project" / "data" / "stations.csv"
METRICS  = HOME / "thesis_project" / "metrics" 
MOS_DIR  = METRICS / "mos"
ML_DIR   = METRICS / "2019_tuned_ah"
OUT_DIR  = HOME / "thesis_project" / "figures" / "MOSvsML_timeseries" / "north-Finland" / "ldt48"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns
SPLIT = "analysistime"
KEYS  = [SPLIT, "validtime", "leadtime"]
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS   = "corrected_mos"
MLCOL = f"corrected_{ML_TAG}"

SEASONAL = False


LEADTIME = 48

#---------------
# Helpers
#---------------

def rmse(a, b):
    """Calculate rmse between two sets of values"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def month_windows(start_str, end_str):
    """
    Function to window data into montly windows
    Yield (start_dt, end_dt_inclusive) for each month overlapping [start,end].
    """
    start = pd.Timestamp(start_str).to_pydatetime()
    end   = pd.Timestamp(end_str).to_pydatetime()
    y, m = start.year, start.month
    cur = datetime(y, m, 1)
    while cur <= end.replace(day=1):
        nd = calendar.monthrange(cur.year, cur.month)[1]
        m_start = max(cur, start)
        m_end   = min(datetime(cur.year, cur.month, nd, 23, 59, 59), end)
        yield m_start, m_end
        # next month
        ny, nm = (cur.year + (cur.month == 12), 1 if cur.month == 12 else cur.month + 1)
        cur = datetime(ny, nm, 1)

def _pair_stats(y_true, y_pred):
    """Calculates statistics for observations vs predictions
        Params:
            y_true = Observations
            y_pred = Predictions
        Returns: 
            n = number of points
            bias = Bias of the predictions (over/under estimation)
            rmse = RMSE of the predictions
            r2 = R squared value of the predictions
    """
    a = np.asarray(y_true, float)
    b = np.asarray(y_pred, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    n = a.size
    if n == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "r2": np.nan}
    bias = float(np.mean(b - a))
    rmse_v = float(np.sqrt(np.mean((b - a) ** 2)))
    if np.std(a) == 0 or np.std(b) == 0:
        r2 = np.nan
    else:
        r = float(np.corrcoef(a, b)[0, 1])
        r2 = r * r
    return {"n": int(n), "bias": bias, "rmse": rmse_v, "r2": r2}

def mos_coverage_window(mos_eval: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return mos_eval[SPLIT].min(), mos_eval[SPLIT].max()
#-----------------------------------------
# Data loading and preparation functions
#-----------------------------------------

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
        cols = KEYS + [RAW, OBS, MLCOL, MOS, "SID"]
        # tolerate older files lacking RAW (we’ll drop if missing)
        existing = [c for c in cols if c in pl.scan_parquet(str(f)).collect_schema().keys()]
        df = pl.read_parquet(f, columns=existing).with_columns([
            pl.col("SID").cast(pl.Utf8),
            pl.col(SPLIT).cast(pl.Utf8),
        ])
        dfs.append(df)

    # Concatenate the into one dataframe
    out = pl.concat(dfs, how="vertical_relaxed")

    print(f"[INFO] {tag} rows loaded: {out.height:,}")
    return out

def filter_for_station_init(df: pl.DataFrame, target_station: str, start_init: str, end_init:str, tag: str):
    """Filter the dataframe to the wanted station and analysistime
        Params: 
            df = Dataframe
            target_station = SID of the wanted target station
            target_init = Date and time of the wanted analysistime
            tag = Name of the column/model data is loaded for
        Returns: Filtered dataframe"""

    # Filter data to the target station and analysistime
    plot_df = df.filter(
        (pl.col("SID") == target_station) & (pl.col(SPLIT) >= start_init)
        & (pl.col(SPLIT) <= end_init)
    )   

    if plot_df.height == 0:
        raise ValueError(f"No {tag} rows for SID={target_station}, init={start_init}_{end_init}")
    
    return plot_df

def data_prep_for_plot(ml_plot, mos_plot, start_init, end_init):
    """Function to join the prediction data and prepare the dataset for plotting
        Params: 
            ml_plot = Dataframe with ML model predictions (and observations)
            mos_plot = Dataframe with MOS model predictions (and observations)
            target_station = Unique station ID of the chosen station
            start_init = Start date and time of the time window
            end_init = End date and time of the time window
        Returns: Joined and prepared dataframe
    """

    # If no ML predictions present use only MOS
    if ml_plot.height == 0:
        print(f"[WARN] No ML rows for init={start_init}_{end_init}. Plotting without ML.")
        # Use MOS-only
        joined = mos_plot
    # If no MOS predictions present use only ML
    elif mos_plot.height == 0:
        print(f"[WARN] No MOS rows for init={start_init}_{end_init}. Plotting without MOS.")
        # Use ML-only
        joined = ml_plot
    # Otherwise use both
    else:
        # Inner-join to align samples (same validtime/leadtime)
        # Use ML (raw forecast) as the base
        joined = ml_plot.join(
            mos_plot.select(KEYS + [MOS]),
            on=KEYS,
            how="left",
            suffix="_mos"
        )
        for col in [RAW, OBS]:
            col_ml = f"{col}_ml"
            if col_ml in joined.columns and col in joined.columns:
                joined = joined.drop(col_ml)

    # Convert to pandas and prepare for plotting
    plot_pd = joined.to_pandas()
    plot_pd["validtime"] = pd.to_datetime(plot_pd["validtime"])
    plot_pd["analysistime"] = pd.to_datetime(plot_pd["analysistime"])
    plot_pd = plot_pd.sort_values(["analysistime", "validtime"])

    return plot_pd


#------------------------------
# Plot functions
#------------------------------

def scatter_density_one(
    df: pd.DataFrame, obs_col: str, pred_col: str, station_id: str, station_name: str, start_init: str, end_init: str,
    tag: str, cmap: str, gridsize: int = 60, mincnt: int = 1,
    use_log_counts: bool = True,    # Enables log color + ticks
    err_thresh: float | None = None,     # symmetric |pred-obs| <= T
    err_thresh_pos: float | None = None, # asymmetric upper:  pred-obs >= +Tpos
    err_thresh_neg: float | None = None, # asymmetric lower:  pred-obs <= -Tneg
    shade_alpha: float = 0.12, band_alpha: float = 0.06, show_band: bool = False,
    show_pct_outside: bool = True,  # Annotate percent outside the band
    units: str = "K",
    use_variable_threshold: bool = False,  # if True, overrides err_thresh/pos/neg
    cold_cut: float = 258.15,
    cool_cut: float = 268.15,
    cold_T: float = 5.0,    # K
    cool_T: float = 3.5,    # K
    warm_T: float = 2.5,    # K
    season = None
):
    
    """
    Creates a scatter plot of the predictions vs observations colored by density.
    Includes a band based on the hit rate and statistics to add more information. 
    The scatter points are hexbins where the points are logarithmically binned
    """
    # Extract finite pairs
    a = pd.to_numeric(df.get(obs_col, pd.Series(dtype=float)), errors="coerce")
    b = pd.to_numeric(df.get(pred_col, pd.Series(dtype=float)), errors="coerce")
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m].to_numpy(); b = b[m].to_numpy()
    if a.size == 0:
        print(f"[WARN] No finite {obs_col}/{pred_col} pairs for {station_id} ({tag})")
        return

    # Get the statistics
    st = _pair_stats(a, b)


    # Percentiles (0.5–99.5)
    x_lo, x_hi = np.nanpercentile(a, [0.5, 99.5])
    y_lo, y_hi = np.nanpercentile(b, [0.5, 99.5])

    # Calculate span and center
    sx, sy = (x_hi - x_lo), (y_hi - y_lo)
    if not np.isfinite(sx) or sx <= 0: sx = 1.0
    if not np.isfinite(sy) or sy <= 0: sy = 1.0

    # Extra padding (≈ 20% + one hex width)
    pad_x = 0.2 * sx + sx / gridsize
    pad_y = 0.2 * sy + sy / gridsize

    # Keep equal axis ranges (square aspect)
    x_center = (x_hi + x_lo) / 2
    y_center = (y_hi + y_lo) / 2
    half_span = 0.5 * max(sx + 2 * pad_x, sy + 2 * pad_y)

    x0, x1 = x_center - half_span, x_center + half_span
    y0, y1 = y_center - half_span, y_center + half_span

    # Add a bit of extra breathing room if needed
    margin = 0.01 * half_span
    x0 -= margin; x1 += margin
    y0 -= margin; y1 += margin

    # Plot
    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    # Logarithmic counts to hexbins + plot the hexbins
    norm = LogNorm() if use_log_counts else None
    hb = ax.hexbin(
        a, b,
        gridsize=gridsize,
        mincnt=mincnt,
        norm=norm,
        cmap=cmap,
        extent=(x0, x1, y0, y1),  # lattice matches limits → no cropped hexes
    )

    # Colorbar configuration
    cb = fig.colorbar(hb, ax=ax, shrink=0.9)
    cb.set_label("Count per hexbin")
    if use_log_counts:
        cb.locator = LogLocator(base=10, subs="all")
        cb.update_ticks()

    # Background shading for thresholds, if requested
    want_fixed = (err_thresh is not None) or (err_thresh_pos is not None) or (err_thresh_neg is not None)

    if use_variable_threshold or want_fixed:
        xs = np.linspace(x0, x1, 512)

        # Piecewise thershold boundary by observation (x)
        # Based on the hit rate boundaries 
        # Cold <=258.15K
        # 258.15K < Cool <= 268.15K
        # Warm > 268.15K
        if use_variable_threshold:
            Tx = np.where(xs <= cold_cut, cold_T,
                 np.where(xs <= cool_cut, cool_T, warm_T))
            y_plus  = xs + Tx
            y_minus = xs - Tx

            label_band_main = f"|pred−obs| ≤ T(obs);" 
            upper_lbl = "pred−obs ≥ +T(obs)"
            lower_lbl = "pred−obs ≤ −T(obs)"

        # Fixed thersholds
        else:
            Tpos = err_thresh_pos if err_thresh_pos is not None else (err_thresh or 0.0)
            Tneg = err_thresh_neg if err_thresh_neg is not None else (err_thresh or 0.0)
            y_plus  = xs + Tpos
            y_minus = xs - Tneg

            label_band_main = f"|pred−obs| ≤ {max(Tpos, Tneg):g} {units}" if show_band else None
            upper_lbl = f"pred−obs ≥ +{Tpos:g} {units}"
            lower_lbl = f"pred−obs ≤ −{Tneg:g} {units}"

        # Shade the out-of-band regions
        ax.fill_between(xs, y_plus, y1, alpha=shade_alpha, label=upper_lbl, color= "#E64A19")
        ax.fill_between(xs, y0, y_minus, alpha=shade_alpha, label=lower_lbl, color= "#004E9B")

        # Shade band interior
        if show_band:
            ax.fill_between(xs, y_minus, y_plus, alpha=band_alpha, color = "#66E247",
                            label=label_band_main if label_band_main else None)

        # Boundary lines
        ax.plot(xs, y_plus,  linestyle="-", linewidth=1.0, color= "#E64A19")
        ax.plot(xs, y_minus, linestyle="-", linewidth=1.0, color= "#004E9B")

        # % outside text and calculation
        if show_pct_outside:
            err = b - a
            if use_variable_threshold:
                # per-sample T depends on obs 'a'
                Ta = np.where(a <= cold_cut, cold_T,
                     np.where(a <= cool_cut, cool_T, warm_T))
                out_frac = np.mean(np.abs(err) >= Ta) * 100.0
            else:
                Tpos = err_thresh_pos if err_thresh_pos is not None else (err_thresh or 0.0)
                Tneg = err_thresh_neg if err_thresh_neg is not None else (err_thresh or 0.0)
                out_frac = np.mean((err >= Tpos) | (err <= -Tneg)) * 100.0

            # % outside text box configuration
            ax.text(
                0.02, 0.8,
                f"{out_frac:.1f}% outside band",
                transform=ax.transAxes,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round", alpha=0.20)
            )

    # Plot parameters
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.plot([x0, x1], [x0, x1], linewidth=1.2, color = "#444444")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observation [{units}]", fontsize = 12)
    ax.set_ylabel(f"{tag} corrected prediction [{units}]" if tag in ("MOS",ML_NAME) else f"Prediction [{units}]", fontsize = 12)
    title_range = f"{pd.to_datetime(start_init):%Y-%m-%d} .. {pd.to_datetime(end_init):%Y-%m-%d}"
    ax.set_title(f"{station_name} {station_id} \n Inits {title_range}\nObs vs {tag} (density)", fontsize = 20)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()

    # Statistics text box text
    s_txt = (
        f"N = {st['n']}\n"
        f"Bias = {st['bias']:.3g}\n"
        f"RMSE = {st['rmse']:.3g}\n"
        f"R\u00b2 = {st['r2']:.3g}" if np.isfinite(st["r2"]) else
        f"N = {st['n']}\nBias = {st['bias']:.3g}\nRMSE = {st['rmse']:.3g}\nR\u00b2 = NA"
    )

    # Statistics text box configuration
    ax.text(
        0.02, 0.98, s_txt,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    # Save/show plot
    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_path = OUT_DIR / f"{station_id}" / month_tag
    out_path.mkdir(parents=True, exist_ok=True)
    out_svg = out_path / f"scatter_density_{tag}_{station_id}_{month_tag}_{season}.svg"
    if SAVE_PLOT:
        fig.savefig(out_svg, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {out_svg}")
        plt.close(fig)
    if SHOW_PLOT: plt.show()
    else: plt.close(fig)



def main():
    # Set wanted stations 
    stations = ["101886", "101928", "101932", "102016", "102033", "102035"]

    # Set wanted start and end dates (now the whole test period)
    month_start = "2024-09-01 00:00:00"
    month_end   = "2025-08-31 12:00:00"  # inclusive end of the whole loop range

    if SEASONAL:
        seasons = ["autumn", "winter", "spring", "summer"]

        # Load ML predictions once (covers all seasons if your files do)
        ml_data_full = load_eval_rows_evaldir(ML_DIR, f"eval_rows_{SPLIT}_{ML_TAG}_*.parquet", MLCOL)

        for season in seasons:
            # Load MOS once per season
            mos_data_season = load_eval_rows_evaldir(MOS_DIR, f"eval_rows_{SPLIT}_MOS_*_{season}.parquet", MOS)

            # Determine coverage window for the season (shared across stations)
            start_init, end_init = mos_coverage_window(mos_data_season)

            leadtime = LEADTIME
            mos_ltd = mos_data_season.filter(pl.col("leadtime") <= leadtime)
            ml_ltd  = ml_data_full.filter(pl.col("leadtime") <= leadtime)

            all_stations_mos_ltd = []
            all_stations_ml_ltd  = []
            for sid in stations:
                station_id = str(sid)
                mos_filter = filter_for_station_init(mos_ltd, station_id, start_init, end_init, MOS)
                ml_filter  = filter_for_station_init(ml_ltd,  station_id, start_init, end_init, ML_TAG)
                if mos_filter is not None and len(mos_filter) > 0:
                    all_stations_mos_ltd.append(mos_filter)
                if ml_filter is not None and len(ml_filter) > 0:
                    all_stations_ml_ltd.append(ml_filter)

            all_mos_ldt = pl.concat(all_stations_mos_ltd)
            all_ml_ldt  = pl.concat(all_stations_ml_ltd)

            if all_mos_ldt.height == 0 and all_ml_ldt.height == 0:
                print(f"[{season} leadtime≤{leadtime}] No data after station filtering — skipping.")
            else:
                plot_pd = data_prep_for_plot(all_ml_ldt, all_mos_ldt, start_init=start_init, end_init=end_init)
                station_id   = "North Finland"
                station_name = "All stations"

                if MOS in plot_pd.columns:
                    scatter_density_one(
                        df=plot_pd, obs_col=OBS, pred_col=MOS,
                        station_id=station_id, station_name=station_name,
                        start_init=start_init, end_init=end_init,
                        tag=f"MOS (≤{leadtime}h)", use_log_counts=True, cmap="turbo",
                        use_variable_threshold=True, cold_cut=258.15, cold_T=5.0,
                        cool_cut=268.15,  cool_T=3.5, warm_T=2.5,
                        show_band=True, units="K", season=season
                    )
                if MLCOL in plot_pd.columns:
                    scatter_density_one(
                        df=plot_pd, obs_col=OBS, pred_col=MLCOL,
                        station_id=station_id, station_name=station_name,
                        start_init=start_init, end_init=end_init,
                        tag=f"{ML_NAME} (≤{leadtime}h)", use_log_counts=True, cmap="turbo",
                        use_variable_threshold=True, cold_cut=258.15, cold_T=5.0,
                        cool_cut=268.15,  cool_T=3.5, warm_T=2.5,
                        show_band=True, units="K", season=season
                    )
                if RAW in plot_pd.columns:
                    scatter_density_one(
                        df=plot_pd, obs_col=OBS, pred_col=RAW,
                        station_id=station_id, station_name=station_name,
                        start_init=start_init, end_init=end_init,
                        tag=f"ECMWF (≤{leadtime}h)", use_log_counts=True, cmap="turbo",
                        use_variable_threshold=True, cold_cut=258.15, cold_T=5.0,
                        cool_cut=268.15,  cool_T=3.5, warm_T=2.5,
                        show_band=True, units="K", season=season
                    )

    else:
        # Non-seasonal: plot per month window, concatenated across all stations
        mos_data_full = load_eval_rows_evaldir(MOS_DIR, f"eval_rows_{SPLIT}_MOS_*.parquet", MOS)
        ml_data_full  = load_eval_rows_evaldir(ML_DIR,  f"eval_rows_{SPLIT}_{ML_TAG}_*.parquet", MLCOL)

        for m_start, m_end in month_windows(month_start, month_end):
            start_init = m_start.strftime("%Y-%m-%d %H:%M:%S")
            end_init   = m_end.strftime("%Y-%m-%d %H:%M:%S")

            leadtime = LEADTIME
            mos_ltd = mos_data_full.filter(pl.col("leadtime") <= leadtime)
            ml_ltd  = ml_data_full.filter(pl.col("leadtime") <= leadtime)

            all_stations_mos_ltd = []
            all_stations_ml_ltd  = []
            for sid in stations:
                station_id = str(sid)
                mos_filter = filter_for_station_init(mos_ltd, station_id, start_init, end_init, MOS)
                ml_filter  = filter_for_station_init(ml_ltd,  station_id, start_init, end_init, ML_TAG)
                if mos_filter is not None and len(mos_filter) > 0:
                    all_stations_mos_ltd.append(mos_filter)
                if ml_filter is not None and len(ml_filter) > 0:
                    all_stations_ml_ltd.append(ml_filter)

            all_mos_ltd = pl.concat(all_stations_mos_ltd)
            all_ml_ltd  = pl.concat(all_stations_ml_ltd)

            if all_mos_ltd.height == 0 and all_ml_ldt.height == 0:
                print(f"[{start_init} – {end_init}, leadtime≤{leadtime}] No data after station filtering — skipping.")
            else:
                plot_pd = data_prep_for_plot(all_ml_ltd, all_mos_ltd, start_init=start_init, end_init=end_init)
                station_id   = "North Finland"
                station_name = "All stations"

                if MOS in plot_pd.columns:
                    scatter_density_one(
                        df=plot_pd, obs_col=OBS, pred_col=MOS,
                        station_id=station_id, station_name=station_name,
                        start_init=start_init, end_init=end_init,
                        tag=f"MOS (≤{leadtime}h)", use_log_counts=True, cmap="turbo",
                        use_variable_threshold=True, cold_cut=258.15, cold_T=5.0,
                        cool_cut=268.15,  cool_T=3.5, warm_T=2.5,
                        show_band=True, units="K",
                    )
                if MLCOL in plot_pd.columns:
                    scatter_density_one(
                        df=plot_pd, obs_col=OBS, pred_col=MLCOL,
                        station_id=station_id, station_name=station_name,
                        start_init=start_init, end_init=end_init,
                        tag=f"{ML_NAME} (≤{leadtime}h)", use_log_counts=True, cmap="turbo",
                        use_variable_threshold=True, cold_cut=258.15, cold_T=5.0,
                        cool_cut=268.15,  cool_T=3.5, warm_T=2.5,
                        show_band=True, units="K",
                    )
                if RAW in plot_pd.columns:
                    scatter_density_one(
                        df=plot_pd, obs_col=OBS, pred_col=RAW,
                        station_id=station_id, station_name=station_name,
                        start_init=start_init, end_init=end_init,
                        tag=f"ECMWF (≤{leadtime}h)", use_log_counts=True, cmap="turbo",
                        use_variable_threshold=True, cold_cut=258.15, cold_T=5.0,
                        cool_cut=268.15,  cool_T=3.5, warm_T=2.5,
                        show_band=True, units="K",
                    )




if __name__ == "__main__":
    main()