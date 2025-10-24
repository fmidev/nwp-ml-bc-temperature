import os
from pathlib import Path
import calendar
from datetime import datetime, timedelta
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# User settings
ML_TAG = "tuned_ah_2019"

SHOW_PLOT = False
SAVE_PLOT = True
FIG_DPI   = 150


# Directories
HOME     = Path.home()
STATION_FILE = HOME / "thesis_project" / "data" / "stations.csv"
METRICS  = HOME / "thesis_project" / "metrics" 
MOS_DIR  = METRICS / "mos"
ML_DIR   = METRICS / "2019_tuned_ah"
OUT_DIR  = HOME / "thesis_project" / "figures" / "MOSvsML_timeseries" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns
SPLIT = "analysistime"
KEYS  = ["SID", SPLIT, "validtime", "leadtime"]
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS   = "corrected_mos"
MLCOL = f"corrected_{ML_TAG}"

#---------------
# Helpers
#---------------

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
        cols = KEYS + [RAW, OBS, MLCOL, MOS]
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

def data_prep_for_plot(ml_plot, mos_plot, target_station, start_init, end_init):
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
        print(f"[WARN] No ML rows for SID={target_station}, init={start_init}_{end_init}. Plotting without ML.")
        # Use MOS-only
        joined = mos_plot
    # If no MOS predictions present use only ML
    elif mos_plot.height == 0:
        print(f"[WARN] No MOS rows for SID={target_station}, init={start_init}_{end_init}. Plotting without MOS.")
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

def plot_month(df, station_id, station_name, *, q_low=0.05, q_high=0.95, pred_resample=None,
                start_init, end_init):
    """
    Plot one month of observation vs predictions for a single station.
        Params: 
            df = Dataframe with predictions and observations 
            station_id = Unique station ID of the chosen station
            station_name = Name of the chosen station
            q_low = Lower quantile
            q_high = Upper quantile
            pred_resample = Prediction resample e.g. '3H' to aggregate predictions to 3-hour cadence
            start_init = Start date and time of the time window
            end_init = End date and time of the time window
    """

    # Plot colors and levels
    colors = {
        "RAW": "#999999",   # muted grey
        "MOS": "#637AB9",   # blue-toned (MOS)
        "ML":  "#B95E82",   # magenta-toned (ML)
    }
    alphas = {
        "RAW": 0.7,
        "MOS": 0.95,
        "ML":  0.95,
    }

    # Defensive copy
    df = df.copy()

    # Observation series (deduplicate by time)
    obs = (
        df[["validtime", OBS]]
        .drop_duplicates(subset=["validtime"])
        .set_index("validtime")
        .sort_index()
    )

    # Title bits
    try:
        month_label = f"{pd.to_datetime(start_init):%Y%m}_{pd.to_datetime(end_init):%Y%m}"
    except Exception:
        month_label = "Selected month"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Names for the models 
    tag_to_col = {"RAW": RAW, "MOS": MOS, "ML": MLCOL}

    for col_type, col in tag_to_col.items():
        series = (
            df[["validtime", f"{col}"]]
            .dropna()
            .set_index("validtime")
            .sort_index()
            .rename(columns={f"{col}": "T2"})
        )

        # Aggregate to median + quantile band
        if pred_resample:
            agg = (
                series["T2"]
                .resample(pred_resample)
                .agg(
                    median="median",
                    lo=lambda x: x.quantile(q_low) if len(x) else np.nan,
                    hi=lambda x: x.quantile(q_high) if len(x) else np.nan,
                )
                .dropna(how="all")
            )
        else:
            agg = (
                series.groupby(level=0)["T2"]
                .agg(
                    median="median",
                    lo=lambda x: x.quantile(q_low),
                    hi=lambda x: x.quantile(q_high),
                )
                .sort_index()
            )

        # Ribbon
        ax.fill_between(
            agg.index, agg["lo"], agg["hi"],
            color=colors[col_type],
            alpha=min(0.35, alphas[col_type] * 0.35),
            linewidth=0,
            label=f"{col_type} {int(q_low*100)}–{int(q_high*100)}%"
        )

        # Median line
        ax.plot(
            agg.index, agg["median"],
            color=colors[col_type],
            alpha=alphas[col_type],
            linewidth=2,
            label=f"{col_type} median"
        )

    # Scatter observations on top
    ax.scatter(
        obs.index, obs[OBS],
        s=18, label="Observation", zorder=5, color= "#444444"
    )

    # Plot parameters
    ax.set_title(f"{station_name} (SID {station_id}) — {month_label}")
    ax.set_xlabel("Valid time")
    ax.set_ylabel("Temperature (K)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()

    # Save
    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_path = OUT_DIR / f"{station_id}" / month_tag
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"timeseries_mean_{station_id}_{start_init}_{end_init}.png", dpi=160)
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_for_leadtime(df, station_id, station_name, start_init, end_init, leadtime):
    """Plots a timeseries for each model of the predictions with chosen leadtime
        Params:
            df = Dataframe with predictions and observations
            station_id = Unique ID of the chosen station
            station_name = Name of the chosen station
            start_init = Start date and time of the time window
            end_init = End date and time of the time window
            leadtime = Chosen leadtime
    """

    # Filter based on chosen leadtime
    try:
        subset = df.query(f"leadtime == {leadtime}")
    except Exception:
        print(f"[WARN] No leadtime column or invalid values for {station_id}")
        return

    if subset.empty:
        print(f"[WARN] No 24h leadtime forecasts for {station_id}")
        return

    plt.figure(figsize=(12, 5))

    # Plot each model if present
    if RAW in subset.columns:
        plt.plot(subset["validtime"], subset[RAW],
                linestyle="-", lw=1.0, color="#999999", alpha=0.7, label=f"Raw forecast ({leadtime}h lead)")
    if MOS in subset.columns:
        plt.plot(subset["validtime"], subset[MOS],
                lw=2.3, color="#637AB9", alpha=0.95, label=f"MOS corrected ({leadtime}h lead)")
    if MLCOL in subset.columns:
        plt.plot(subset["validtime"], subset[MLCOL],
                lw=2.3, color="#B95E82", alpha=0.95, label=f"ML corrected ({ML_TAG}) ({leadtime}h lead)")

    # Observations at those same validtimes
    if OBS in subset.columns:
        plt.scatter(subset["validtime"], subset[OBS],
                    s=12, color="black", label="Observation", zorder=3)

    # Plot parameters
    plt.title(f"{station_name} {station_id} — Inits {pd.to_datetime(start_init):%Y-%m-%d}..{pd.to_datetime(end_init):%Y-%m-%d}\n{leadtime}h lead forecasts")
    plt.xlabel("Valid time")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.legend(framealpha=0.85)
    plt.tight_layout()

    # Save/show plot
    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_dir = OUT_DIR / station_id / month_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fig2 = out_dir / f"timeseries_ldt{leadtime}_{station_id}_{start_init}_{end_init}.png"

    if SAVE_PLOT:
        plt.savefig(fig2, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {fig2}")
        plt.close()
    if SHOW_PLOT:
        plt.show()
        plt.close()
    else:
        plt.close()

def rmse_plot(df, station_id, station_name, start_init, end_init):
    """Plot the rmse as a timeseries for the different models
        Params: 
            df = Dataframe with the predictions and observations
            station_id = Unique ID of the chosen station
            station_name = Name of the chosen station
            start_init = Start date and time of the time window
            end_init = End date and time of the time window
    """
    # Colors for the plot
    colors = {
        "RAW": "#999999",   
        "MOS": "#637AB9",   
        "ML":  "#B95E82",   
    }

    # Month tag for output directory
    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
     
    # Calculate the rmse between predictions and observations for all of the models
    metrics = (
        df.groupby("analysistime", group_keys=False)
        .apply(
            lambda d: pd.Series({
                "RMSE_RAW": rmse(d["raw_fc"], d["obs_TA"]) if "raw_fc" in d and "obs_TA" in d else np.nan,
                "RMSE_MOS": rmse(d["corrected_mos"], d["obs_TA"]) if "corrected_mos" in d and "obs_TA" in d else np.nan,
                "RMSE_ML":  rmse(d.get(f"corrected_{ML_TAG}"), d["obs_TA"]) if f"corrected_{ML_TAG}" in d and "obs_TA" in d else np.nan,
            }),
            include_groups=False,
        )
        .reset_index()
    )
     

    plt.figure(figsize=(10, 4))

    # If the rmse for certain model present plot it
    if "RMSE_RAW" in metrics:
        plt.plot(metrics["analysistime"], metrics["RMSE_RAW"], label="RAW", lw=1.2, color= colors["RAW"])
    if "RMSE_MOS" in metrics:
        plt.plot(metrics["analysistime"], metrics["RMSE_MOS"], label="MOS", lw=2, color= colors["MOS"])
    if "RMSE_ML" in metrics:
        plt.plot(metrics["analysistime"], metrics["RMSE_ML"],  label=f"ML ({ML_TAG})", lw=2, color= colors["ML"])

    # Plot parameters
    plt.title(f"RMSE vs analysistime — Station {station_name} {station_id} Inits:{pd.to_datetime(start_init):%Y-%m-%d}..{pd.to_datetime(end_init):%Y-%m-%d}")
    plt.xlabel("Analysistime"); plt.ylabel("RMSE (K)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    # Save/show plot
    out_dir = OUT_DIR / station_id / month_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fig2 = out_dir / f"RMSE_timeseries_{station_id}_{month_tag}.png"
    if SAVE_PLOT:
        plt.savefig(fig2, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {fig2}")
        plt.close()
    if SHOW_PLOT:
        plt.show()
        plt.close
    else: plt.close()

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
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count per hexbin")
    if use_log_counts:
        cb.locator = LogLocator(base=10)
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
        ax.fill_between(xs, y_plus, y1, alpha=shade_alpha, label=upper_lbl, color= "#004E9B")
        ax.fill_between(xs, y0, y_minus, alpha=shade_alpha, label=lower_lbl, color= "#E64A19")

        # Shade band interior
        if show_band:
            ax.fill_between(xs, y_minus, y_plus, alpha=band_alpha, color = "#C7FFB9",
                            label=label_band_main if label_band_main else None)

        # Boundary lines
        ax.plot(xs, y_plus,  linestyle="-", linewidth=1.0, color= "#004E9B")
        ax.plot(xs, y_minus, linestyle="-", linewidth=1.0, color= "#E64A19")

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
    ax.set_xlabel(f"Observation [{units}]")
    ax.set_ylabel(f"{tag} corrected prediction [{units}]" if tag in ("MOS","ML") else f"Prediction [{units}]")
    title_range = f"{pd.to_datetime(start_init):%Y-%m-%d} .. {pd.to_datetime(end_init):%Y-%m-%d}"
    ax.set_title(f"{station_name} {station_id} — Inits {title_range}\nObs vs {tag} (density)")
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
    out_png = out_path / f"scatter_density_{tag}_{station_id}_{month_tag}.png"
    if SAVE_PLOT:
        fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {out_png}")
        plt.close(fig)
    if SHOW_PLOT: plt.show()
    else: plt.close(fig)

# facet_by_lead needs to be fixed doesn't look like what I want it to look like yet
def facet_by_lead(
    plot_pd: pd.DataFrame,
    obs_col: str,
    pred_col: str,
    station_id: str,
    start_init: str,
    end_init: str,
    leads: tuple[int, ...] = (12, 24, 48),
    cmap: str = "turbo",
    units: str = "K",
    tag: str = "ML",
    out_dir: Path | None = None,

    # --- NEW: adaptive binning controls ---
    adaptive_binning: bool = True,
    scatter_threshold: int = 120,      # if N < this → use scatter instead of hexbin
    target_pts_per_hex: float = 8.0,   # aim for ~this many points per hex
    gridsize_min: int = 12,
    gridsize_max: int = 50,
    use_log_counts_when_dense: bool = True,
):
    # Prepare per-lead data
    prepared = {}
    for L in leads:
        sub = plot_pd.query("leadtime == @L")
        a = pd.to_numeric(sub.get(obs_col, pd.Series(dtype=float)), errors="coerce").to_numpy()
        b = pd.to_numeric(sub.get(pred_col, pd.Series(dtype=float)), errors="coerce").to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        prepared[L] = (a[m], b[m])


    arr_x, arr_y = [], []
    for L in leads:
        aL, bL = prepared[L]
        if aL.size and bL.size:
            arr_x.append(aL)
            arr_y.append(bL)

    if not arr_x:  # <- nothing to concatenate
        print(f"[WARN] No data for station {station_id} for leads {leads} "
            f"in [{start_init} .. {end_init}]")
        return

    all_a = np.concatenate(arr_x)
    all_b = np.concatenate(arr_y)

    # Stable framing (0.5–99.5 pct + fixed padding + square box)
    x_lo, x_hi = np.nanpercentile(all_a, [0.5, 99.5])
    y_lo, y_hi = np.nanpercentile(all_b, [0.5, 99.5])
    sx, sy = (x_hi - x_lo), (y_hi - y_lo)
    sx = 1.0 if (not np.isfinite(sx) or sx <= 0) else sx
    sy = 1.0 if (not np.isfinite(sy) or sy <= 0) else sy
    pad_x = 0.05 * sx + sx / 40   # ≈5% + small margin
    pad_y = 0.05 * sy + sy / 40
    x0, x1 = x_lo - pad_x, x_hi + pad_x
    y0, y1 = y_lo - pad_y, y_hi + pad_y
    cx, cy = (x0 + x1)/2, (y0 + y1)/2
    half = 0.5 * max(x1 - x0, y1 - y0)
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half

    n = len(leads)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharex=True, sharey=True, constrained_layout=True)
    if n == 1: axes = [axes]
    last_hb = None

    for ax, L in zip(axes, leads):
        a, b = prepared[L]
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.plot([x0, x1], [x0, x1], linewidth=1.0, color="#4f6fad")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_title(f"Lead {L}h")

        if a.size == 0:
            continue

        # --- ADAPTIVE CHOICE: scatter vs hexbin ---
        if adaptive_binning and a.size < scatter_threshold:
            # Sparse → scatter with good visibility
            ax.scatter(a, b, s=18, alpha=0.85, linewidths=0,
                       c="#2b2b2b", zorder=3)
            # No colorbar update from scatter
        else:
            # Dense enough → hexbin; choose gridsize from N
            if adaptive_binning:
                # approximate gridsize so bins ≈ N / target_pts_per_hex
                gs = int(np.sqrt(max(a.size / max(target_pts_per_hex, 1e-9), 1.0)))
                gridsize = int(np.clip(gs, gridsize_min, gridsize_max))
            # Norm: log when dense, linear when moderately sparse
            use_log = use_log_counts_when_dense and (a.size >= 5*target_pts_per_hex)
            norm = LogNorm() if use_log else None

            last_hb = ax.hexbin(
                a, b,
                gridsize=gridsize,
                mincnt=1,
                norm=norm,
                cmap=cmap,
                extent=(x0, x1, y0, y1),
            )

        # Optional: per-panel stats box (compact)
        # st = _pair_stats(a, b)
        # ax.text(0.03, 0.97,
        #         f"N={st['n']}\nBias={st['bias']:.2g}\nRMSE={st['rmse']:.2g}\nR²={st['r2']:.2g}",
        #         transform=ax.transAxes, ha="left", va="top",
        #         bbox=dict(boxstyle="round", alpha=0.2), fontsize=10)

        ax.set_xlabel(f"Observation [{units}]")
    axes[0].set_ylabel(f"{tag} corrected prediction [{units}]")

    # Shared colorbar only if we drew any hexbins
    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes, label="Count per hexbin")
        if isinstance(last_hb.get_array(), np.ndarray) and last_hb.get_array().size:
            # nice ticks for log; integer ticks for linear
            if isinstance(last_hb.norm, LogNorm):
                cbar.locator = LogLocator(base=10)
                cbar.update_ticks()
            else:
                cbar.locator = MaxNLocator(integer=True, nbins=5)
                cbar.update_ticks()

    title_range = f"{pd.to_datetime(start_init):%Y-%m-%d} .. {pd.to_datetime(end_init):%Y-%m-%d}"
    fig.suptitle(f"{station_id} — Inits {title_range}\nObs vs {tag} by lead time", y=1.02)


    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_path = OUT_DIR / f"{station_id}" / month_tag
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"scatter_density_facets_{tag}_{station_id}_{month_tag}.png"
    if SAVE_PLOT:
        fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {fname}")
        plt.close(fig)
    if SHOW_PLOT: 
        plt.show()
        plt.close(fig)




def main():

    """stations = ["100917", "100932", "100896", "101044", "101065", "101118", "101237", "101268", "101339",
                "101398", "101430", "101537", "101570", "101608", "101725", "101794", "101886", "101928",
                "101932", "10201", "102033", "102035"]  # ← add as many as you need"""

    # Set wanted stations 
    stations = ["101932", "101118", "100932"]

    # Set wanted start and end dates (now the whole test period)
    month_start = "2024-09-01 00:00:00"
    month_end = "2025-08-31 12:00:00"  # inclusive end of the whole loop range

    # Load station names
    station_names = load_station_names()

    # Load the prediction data
    mos_data = load_eval_rows_evaldir(MOS_DIR, f"eval_rows_{SPLIT}_MOS_*.parquet", MOS)
    ml_data = load_eval_rows_evaldir(ML_DIR, f"eval_rows_{SPLIT}_{ML_TAG}_*.parquet", MLCOL)

    # Loop over stations and months
    for sid in stations:
        # Window the data to individual months
        for m_start, m_end in month_windows(month_start, month_end):
            
            # Station ID and name
            station_id=str(sid)
            station_name = station_names.get(str(station_id))

            # Analysistime window should cover the whole month
            start_init=m_start.strftime("%Y-%m-%d %H:%M:%S")
            end_init=m_end.strftime("%Y-%m-%d %H:%M:%S")

            # Filter data to needed station and time period 
            mos_filter = filter_for_station_init(mos_data, station_id, start_init, end_init, MOS)
            ml_filter = filter_for_station_init(ml_data, station_id, start_init, end_init, ML_TAG)

            # Join and prepare the data for plotting 
            plot_pd = data_prep_for_plot(ml_filter, mos_filter, station_id, start_init, end_init)

            # Plot monthly mean predictions and the observations 
            plot_month(df= plot_pd, 
                       station_id=station_id, station_name=station_name,
                        start_init= start_init, end_init=end_init)

            # Plot only individual leadtimes (now chosen 24h and 48h)
            plot_for_leadtime(plot_pd, station_id, station_name, start_init, end_init, 24)
            plot_for_leadtime(plot_pd, station_id, station_name, start_init, end_init, 48)

            # Plot the rmse changes for the models
            rmse_plot(plot_pd, station_id, station_name, start_init, end_init)

            # Density scatter plot of predictions vs observations
            if MOS in plot_pd.columns:
                scatter_density_one(
                    df=plot_pd,
                    obs_col=OBS,
                    pred_col=MOS,
                    station_id=station_id,
                    station_name=station_name,
                    start_init=start_init,
                    end_init=end_init,
                    tag="MOS",
                    use_log_counts=True,
                    cmap="turbo",
                    use_variable_threshold=True,
                    cold_cut=258.15, cold_T=5.0,
                    cool_cut=268.15, cool_T=3.5,
                    warm_T=2.5,
                    show_band=True,          
                    units="K",
                )

            if MLCOL in plot_pd.columns:
                scatter_density_one(
                    df=plot_pd,
                    obs_col=OBS,
                    pred_col=MLCOL,
                    station_id=station_id,
                    station_name=station_name,
                    start_init=start_init,
                    end_init=end_init,
                    tag="ML",
                    use_log_counts=True,
                    cmap="turbo",
                    use_variable_threshold=True,
                    cold_cut=258.15, cold_T=5.0,
                    cool_cut=268.15, cool_T=3.5,
                    warm_T=2.5,
                    show_band=True,          
                    units="K",
                )

           
            # Facets by lead time for MOS/ML
            if MLCOL in plot_pd.columns:
                facet_by_lead(
                    plot_pd=plot_pd, obs_col=OBS, pred_col=MLCOL,
                    station_id=station_id, start_init=start_init, end_init=end_init,
                    leads=(12, 24, 48),
                    cmap="turbo", units="K", tag="ML",
                    # adaptive defaults:
                    adaptive_binning=True,
                    scatter_threshold=12,
                    target_pts_per_hex=8.0,
                    gridsize_min=12, gridsize_max=50,
                    use_log_counts_when_dense=True,
                )

            if MOS in plot_pd.columns:
                facet_by_lead(
                    plot_pd=plot_pd, obs_col=OBS, pred_col=MOS,
                    station_id=station_id, start_init=start_init, end_init=end_init,
                    leads=(12, 24, 48),
                    cmap="turbo", units="K", tag="MOS",
                    adaptive_binning=True,
                    scatter_threshold=12,
                    target_pts_per_hex=8.0,
                    gridsize_min=12, gridsize_max=50,
                    use_log_counts_when_dense=True,
                )

if __name__ == "__main__":
    main()