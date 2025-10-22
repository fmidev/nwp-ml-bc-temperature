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

# =====================================================
# User settings
# =====================================================
"""STATIONS = ["100917", "100932", "100896", "101044", "101065", "101118", "101237", "101268", "101339",
            "101398", "101430", "101537", "101570", "101608", "101725", "101794", "101886", "101928",
            "101932", "10201", "102033", "102035"]  # ← add as many as you need"""

STATIONS = ["101932", "101118", "100932"]
MONTH_START = "2024-09-01 00:00:00"
MONTH_END   = "2025-08-31 12:00:00"  # inclusive end of the whole loop range

ML_TAG = "tuned_ah_2019"

SHOW_PLOT = False
SAVE_PLOT = True
FIG_DPI   = 150

# Ribbons/aggregation
BIN = "6h"           # snap valid times to bins (e.g. '3h', '6h', '1D')
ROLL = 1             # rolling mean over bins (integer window size)
QLOW, QHIGH = 0.25, 0.75   # ribbon quantiles (IQR by default)

# Directories
HOME     = Path.home()
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

# =====================================================
# Helpers
# =====================================================
def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def month_windows(start_str, end_str):
    """Yield (start_dt, end_dt_inclusive) for each month overlapping [start,end]."""
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



# =======================
# Scatter-density helpers
# =======================
import matplotlib.colors as mcolors

def _pair_stats(y_true, y_pred):
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

def scatter_density_one(
    df: pd.DataFrame,
    obs_col: str,
    pred_col: str,
    station_id: str,
    start_init: str,
    end_init: str,
    out_dir: Path,
    tag: str,
    gridsize: int = 60,
    mincnt: int = 1,
    use_log_counts: bool = True,    # (2) enables log color + ticks
    cmap: str = "viridis",
    # ---- (3) threshold band options ----
    err_thresh: float | None = None,     # symmetric |pred-obs| <= T
    err_thresh_pos: float | None = None, # asymmetric upper:  pred-obs >= +Tpos
    err_thresh_neg: float | None = None, # asymmetric lower:  pred-obs <= -Tneg
    shade_alpha: float = 0.12,
    band_alpha: float = 0.06,
    show_band: bool = False,
    show_pct_outside: bool = True,  # (3) annotate percent outside the band
    units: str = "K",
    use_variable_threshold: bool = False,  # if True, overrides err_thresh/pos/neg
    cold_cut: float = 258.15,
    cool_cut: float = 268.15,
    cold_T: float = 5.0,    # K
    cool_T: float = 3.5,    # K
    warm_T: float = 2.5,    # K

    # --- new: control robust limits ---
    limit_percentiles: tuple[float, float] = (1, 99),
    pad_fraction: float = 0.12,
):
    # Extract finite pairs
    a = pd.to_numeric(df.get(obs_col, pd.Series(dtype=float)), errors="coerce")
    b = pd.to_numeric(df.get(pred_col, pd.Series(dtype=float)), errors="coerce")
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m].to_numpy(); b = b[m].to_numpy()
    if a.size == 0:
        print(f"[WARN] No finite {obs_col}/{pred_col} pairs for {station_id} ({tag})")
        return

    # Stats (kept exactly as you had)
    st = _pair_stats(a, b)

    # --- Improved robust axis limits ---
    # Use wider percentiles (0.5–99.5) and a fixed extra margin
    x_lo, x_hi = np.nanpercentile(a, [0.5, 99.5])
    y_lo, y_hi = np.nanpercentile(b, [0.5, 99.5])

    # Calculate span and center
    sx, sy = (x_hi - x_lo), (y_hi - y_lo)
    if not np.isfinite(sx) or sx <= 0: sx = 1.0
    if not np.isfinite(sy) or sy <= 0: sy = 1.0

    # Extra padding (≈ 5% + one hex width)
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

    # -------------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    # (2) Log colorbar + nice ticks
    norm = LogNorm() if use_log_counts else None
    hb = ax.hexbin(
        a, b,
        gridsize=gridsize,
        mincnt=mincnt,
        norm=norm,
        cmap=cmap,
        extent=(x0, x1, y0, y1),  # lattice matches limits → no cropped hexes
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count per hexbin")
    if use_log_counts:
        cb.locator = LogLocator(base=10)
        cb.update_ticks()

    # (3) Background shading for thresholds, if requested
    want_fixed = (err_thresh is not None) or (err_thresh_pos is not None) or (err_thresh_neg is not None)

    if use_variable_threshold or want_fixed:
        xs = np.linspace(x0, x1, 512)

        if use_variable_threshold:
            # piecewise T(x) by observation (x)
            # cold: x <= cold_cut  -> cold_T
            # cool: (cold_cut, cool_cut] -> cool_T
            # warm: x > cool_cut -> warm_T
            Tx = np.where(xs <= cold_cut, cold_T,
                 np.where(xs <= cool_cut, cool_T, warm_T))
            y_plus  = xs + Tx
            y_minus = xs - Tx

            label_band_main = f"|pred−obs| ≤ T(obs);" #\n 258.15K≤{cold_T:g}K, 268.15K≤{cool_T:g}K, ≤{warm_T:g}K"
            upper_lbl = "pred−obs ≥ +T(obs)"
            lower_lbl = "pred−obs ≤ −T(obs)"

        else:
            # original fixed thresholds (symmetric or asymmetric)
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

        # Optional: shade band interior
        if show_band:
            ax.fill_between(xs, y_minus, y_plus, alpha=band_alpha, color = "#C7FFB9",
                            label=label_band_main if label_band_main else None)

        # Boundary lines
        ax.plot(xs, y_plus,  linestyle="-", linewidth=1.0, color= "#004E9B")
        ax.plot(xs, y_minus, linestyle="-", linewidth=1.0, color= "#E64A19")

        # (3) % outside text badge
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

            ax.text(
                0.02, 0.8,
                f"{out_frac:.1f}% outside band",
                transform=ax.transAxes,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round", alpha=0.20)
            )

    # 1:1 ref + cosmetics
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.plot([x0, x1], [x0, x1], linewidth=1.2, color = "#444444")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observation [{units}]")
    ax.set_ylabel(f"{tag} corrected prediction [{units}]" if tag in ("MOS","ML") else f"Prediction [{units}]")
    title_range = f"{pd.to_datetime(start_init):%Y-%m-%d} .. {pd.to_datetime(end_init):%Y-%m-%d}"
    ax.set_title(f"Station {station_id} — Inits {title_range}\nObs vs {tag} (density)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()

    # --- original stats box (unchanged text/format/position) ---
    s_txt = (
        f"N = {st['n']}\n"
        f"Bias = {st['bias']:.3g}\n"
        f"RMSE = {st['rmse']:.3g}\n"
        f"R\u00b2 = {st['r2']:.3g}" if np.isfinite(st["r2"]) else
        f"N = {st['n']}\nBias = {st['bias']:.3g}\nRMSE = {st['rmse']:.3g}\nR\u00b2 = NA"
    )
    ax.text(  # use ax.text (not plt.text) and same location/style as before
        0.02, 0.98, s_txt,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    # Save
    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"scatter_density_{tag}_{station_id}_{month_tag}.png"
    if SAVE_PLOT:
        fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {out_png}")
    if SHOW_PLOT: plt.show()
    else: plt.close(fig)



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
    fig.suptitle(f"Station {station_id} — Inits {title_range}\nObs vs {tag} by lead time", y=1.02)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"scatter_density_facets_{tag}_{station_id}_{pd.to_datetime(start_init):%Y%m}.png"
        if SAVE_PLOT:
            fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
            print(f"[OK] Saved {fname}")
    if SHOW_PLOT: plt.show()
    else: plt.close(fig)




# =====================================================
# Load once
# =====================================================
mos_files = sorted(MOS_DIR.glob("eval_rows_analysistime_MOS_*.parquet"))
if not mos_files:
    raise FileNotFoundError(f"No MOS eval_rows files in {MOS_DIR}")

mos_frames = [pl.read_parquet(f, columns=KEYS + [RAW, OBS, MOS]) for f in mos_files]
mos_all = pl.concat(mos_frames, how="vertical_relaxed").with_columns([
    pl.col("SID").cast(pl.Utf8),
    pl.col(SPLIT).cast(pl.Utf8),
])

ml_files = sorted(ML_DIR.glob(f"eval_rows_{SPLIT}_{ML_TAG}_*.parquet"))
if not ml_files:
    raise FileNotFoundError(f"No ML eval_rows files matching eval_rows_{SPLIT}_{ML_TAG}_*.parquet in {ML_DIR}")

ml_frames = []
for f in ml_files:
    cols = KEYS + [RAW, OBS, MLCOL]
    existing = [c for c in cols if c in pl.scan_parquet(str(f)).collect_schema().keys()]
    df = pl.read_parquet(f, columns=existing).with_columns([
        pl.col("SID").cast(pl.Utf8),
        pl.col(SPLIT).cast(pl.Utf8),
    ])
    ml_frames.append(df)
ml_all = pl.concat(ml_frames, how="vertical_relaxed")

print(f"[INFO] MOS rows loaded: {mos_all.height:,}")
print(f"[INFO] ML  rows loaded: {ml_all.height:,}")

# =====================================================
# Core runner for one (station, month)
# =====================================================
def run_one(station_id: str, start_init: str, end_init: str):
    # Filter by station & analysistime window
    mos_plot = mos_all.filter(
        (pl.col("SID") == station_id)
        & (pl.col(SPLIT) >= start_init)
        & (pl.col(SPLIT) <= end_init)
    )
    ml_plot = ml_all.filter(
        (pl.col("SID") == station_id)
        & (pl.col(SPLIT) >= start_init)
        & (pl.col(SPLIT) <= end_init)
    )

    if mos_plot.height == 0 and ml_plot.height == 0:
        print(f"[WARN] No data for SID={station_id} in [{start_init} .. {end_init}]")
        return

    if ml_plot.height == 0:
        joined = mos_plot
    else:
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

    # → pandas
    plot_pd = joined.to_pandas()
    plot_pd["validtime"] = pd.to_datetime(plot_pd["validtime"])
    plot_pd["analysistime"] = pd.to_datetime(plot_pd["analysistime"])
    plot_pd = plot_pd.sort_values(["analysistime", "validtime"])

    # =======================
    # Mean-per-validtime + ribbons
    # =======================

    colors = {
        "RAW": "#999999",   
        "MOS": "#637AB9",  
        "ML":  "#B95E82",   
    }

    alphas = {
        "RAW": 0.7,
        "MOS": 0.95,
        "ML":  0.95,
    }


    plot_pd["valid_bin"] = plot_pd["validtime"].dt.floor(BIN)

    has_raw = RAW in plot_pd.columns
    has_mos = MOS in plot_pd.columns
    has_ml  = MLCOL in plot_pd.columns

    def qlow(x):  return np.nanpercentile(x, QLOW*100)
    def qhigh(x): return np.nanpercentile(x, QHIGH*100)

    agg_spec = {}
    if has_raw: agg_spec[RAW] = ["mean", qlow, qhigh]
    if has_mos: agg_spec[MOS] = ["mean", qlow, qhigh]
    if has_ml:  agg_spec[MLCOL] = ["mean", qlow, qhigh]
    agg_spec[OBS] = ["mean"]

    agg = (plot_pd.groupby("valid_bin").agg(agg_spec)).sort_index()
    agg.columns = ["_".join(filter(None, c)) for c in agg.columns]

    if ROLL > 1:
        agg = agg.rolling(ROLL, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(12, 5))

    def plot_mean_with_ribbon(base, label, color, alpha_line=1.0, alpha_fill=0.18, lw=2.0):
        m  = agg.get(f"{base}_mean")
        lo = agg.get(f"{base}_qlow")
        hi = agg.get(f"{base}_qhigh")
        if m is None:
            return
        plt.plot(agg.index, m, label=f"{label} (mean)", color=color,
                linewidth=lw, alpha=alpha_line)
        if lo is not None and hi is not None:
            plt.fill_between(agg.index, lo, hi, color=color,
                            alpha=alpha_fill, linewidth=0)

    if has_raw:
        plot_mean_with_ribbon(RAW, "Raw forecast", colors["RAW"],
                            alpha_line=alphas["RAW"], alpha_fill=0.1, lw=1.0)
    if has_mos:
        plot_mean_with_ribbon(MOS, "MOS corrected", colors["MOS"],
                            alpha_line=alphas["MOS"], alpha_fill=0.25, lw=2.3)
    if has_ml:
        plot_mean_with_ribbon(MLCOL, f"ML corrected ({ML_TAG})", colors["ML"],
                            alpha_line=alphas["ML"], alpha_fill=0.25, lw=2.3)

    if f"{OBS}_mean" in agg.columns:
        obs_mean = agg[f"{OBS}_mean"]
        ok = np.isfinite(obs_mean)
        plt.scatter(agg.index[ok], obs_mean[ok], s=12, color="black", label="Observation", zorder=3)

    title_range = f"{pd.to_datetime(start_init):%Y-%m-%d} .. {pd.to_datetime(end_init):%Y-%m-%d}"
    plt.title(f"Station {station_id} — Inits {title_range}\nMean forecast per valid time (ribbons = {int(QLOW*100)}–{int(QHIGH*100)} pct range)")
    plt.xlabel("Valid time"); plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3); plt.legend(framealpha=0.85); plt.tight_layout()

    # Paths
    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_dir = OUT_DIR / station_id / month_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1 = out_dir / f"timeseries_mean_with_ribbon_{station_id}_{month_tag}.png"
    if SAVE_PLOT:
        plt.savefig(fig1, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {fig1}")
    if SHOW_PLOT: plt.show()
    else: plt.close()



    # =======================
    # Mean forecasts at observation times (no binning)
    # =======================

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

      # IQR ribbon; change to (0.10, 0.90) for wider bands
    MIN_COUNT_FOR_RIBBON = 5

    # Determine which validtimes have observations
    obs_times = pd.to_datetime(plot_pd.loc[plot_pd["obs_TA"].notna(), "validtime"].unique())
    obs_times = [t for t in obs_times if t.hour in (0, 6, 12, 18)]

    obs_times = np.sort(obs_times)

    tol = pd.Timedelta("30min")  # tolerance for matching forecasts near obs times

    records = []
    for vt in obs_times:
        subset = plot_pd[(plot_pd["validtime"] >= vt - tol) &
                        (plot_pd["validtime"] <= vt + tol)]
        if subset.empty:
            continue

        row = {"validtime": vt}

        # Helper to attach mean/lo/hi/count for a forecast column if present
        def add_stats(df, col, prefix):
            if col in df:
                vals = df[col].to_numpy(dtype=float)
                m = np.isfinite(vals)
                if m.any():
                    row[f"{prefix}_mean"]  = float(np.nanmean(vals[m]))
                    row[f"{prefix}_lo"]    = float(np.nanpercentile(vals[m], QLOW*100))
                    row[f"{prefix}_hi"]    = float(np.nanpercentile(vals[m], QHIGH*100))
                    row[f"{prefix}_count"] = int(np.sum(m))

        add_stats(subset, RAW,   "RAW")
        add_stats(subset, MOS,   "MOS")
        add_stats(subset, MLCOL, "ML")

        # Observations: average (usually one value)
        if "obs_TA" in subset:
            obs_vals = subset["obs_TA"].to_numpy(dtype=float)
            m = np.isfinite(obs_vals)
            row["OBS_mean"] = float(np.nanmean(obs_vals[m])) if m.any() else np.nan

        records.append(row)

    summary = pd.DataFrame(records).sort_values("validtime")
    if summary.empty:
        print(f"[WARN] No valid obs/forecast matches for {station_id}")
        return

    # =======================
    # Plot (same linestyles + ribbons)
    # =======================
    plt.figure(figsize=(12, 5))

    def plot_line_with_ribbon(prefix, label, color, line_alpha, lw, linestyle="-", fill_alpha=0.22):
        mean_col, lo_col, hi_col, n_col = f"{prefix}_mean", f"{prefix}_lo", f"{prefix}_hi", f"{prefix}_count"
        if mean_col not in summary:
            return
        # Line
        plt.plot(summary["validtime"], summary[mean_col],
                linestyle=linestyle, color=color, alpha=line_alpha, lw=lw, label=label)
        # Ribbon (only if quantiles exist and sample size is sufficient)
        if lo_col in summary and hi_col in summary and n_col in summary:
            mask = summary[n_col].fillna(0).astype(int) >= MIN_COUNT_FOR_RIBBON
            if mask.any():
                vt = summary.loc[mask, "validtime"]
                lo = summary.loc[mask, lo_col]
                hi = summary.loc[mask, hi_col]
                plt.fill_between(vt, lo, hi, color=color, alpha=fill_alpha, linewidth=0)

    # RAW (de-emphasized, dashed, lighter ribbon)
    plot_line_with_ribbon("RAW", "Raw forecast (mean)", colors["RAW"],
                        line_alpha=alphas["RAW"], lw=1.0, linestyle="-", fill_alpha=0.10)

    # MOS (solid, prominent)
    plot_line_with_ribbon("MOS", "MOS corrected (mean)", colors["MOS"],
                        line_alpha=alphas["MOS"], lw=2.3, linestyle="-", fill_alpha=0.25)

    # ML (solid, prominent)
    plot_line_with_ribbon("ML",  f"ML corrected ({ML_TAG}) (mean)", colors["ML"],
                        line_alpha=alphas["ML"], lw=2.3, linestyle="-", fill_alpha=0.25)

    # Observations
    if "OBS_mean" in summary:
        plt.scatter(summary["validtime"], summary["OBS_mean"],
                    s=12, color="black", label="Observation", zorder=3)

    title_range = f"{pd.to_datetime(start_init):%Y-%m-%d} .. {pd.to_datetime(end_init):%Y-%m-%d}"
    plt.title(f"Station {station_id} — Inits {title_range}\nForecast means at observation times (ribbons = {int(QLOW*100)}–{int(QHIGH*100)} pct range)")
    plt.xlabel("Valid time"); plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3); plt.legend(framealpha=0.85); plt.tight_layout()

    # Save/show (same as your current code)
    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_dir = OUT_DIR / station_id / month_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1 = out_dir / f"timeseries_mean_at_obs_with_ribbon_{station_id}_{month_tag}.png"
    if SAVE_PLOT:
        plt.savefig(fig1, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {fig1}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()



    # =======================
    # Plot for leadtime = 24h
    # =======================
    try:
        subset_24 = plot_pd.query("leadtime == 24")
    except Exception:
        print(f"[WARN] No leadtime column or invalid values for {station_id}")
        return

    if subset_24.empty:
        print(f"[WARN] No 24h leadtime forecasts for {station_id}")
        return

    # Select observation times that actually have observations
    obs_times = (
        pd.to_datetime(subset_24.loc[subset_24["obs_TA"].notna(), "validtime"]
                    .drop_duplicates()
                    .sort_values())
    )

    # Pick every 6th observation to reduce density
    #obs_times_sampled = obs_times[::6]

    #subset_24 = subset_24[subset_24["validtime"].isin(obs_times_sampled)]

    if subset_24.empty:
        print(f"[WARN] No matching observation/forecast pairs after sampling for {station_id}")
        return

    plt.figure(figsize=(12, 5))

    # Plot each forecast system
    if RAW in subset_24.columns:
        plt.plot(subset_24["validtime"], subset_24[RAW],
                linestyle="-", lw=1.0, color="#999999", alpha=0.7, label="Raw forecast (24h lead)")

    if MOS in subset_24.columns:
        plt.plot(subset_24["validtime"], subset_24[MOS],
                lw=2.3, color="#637AB9", alpha=0.95, label="MOS corrected (24h lead)")

    if MLCOL in subset_24.columns:
        plt.plot(subset_24["validtime"], subset_24[MLCOL],
                lw=2.3, color="#B95E82", alpha=0.95, label=f"ML corrected ({ML_TAG}) (24h lead)")

    # Observations at those same validtimes
    if OBS in subset_24.columns:
        plt.scatter(subset_24["validtime"], subset_24[OBS],
                    s=12, color="black", label="Observation", zorder=3)

    plt.title(f"Station {station_id} — Inits {pd.to_datetime(start_init):%Y-%m-%d}..{pd.to_datetime(end_init):%Y-%m-%d}\n24h lead forecasts")
    plt.xlabel("Valid time")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.legend(framealpha=0.85)
    plt.tight_layout()

    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_dir = OUT_DIR / station_id / month_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fig2 = out_dir / f"timeseries_ldt24_{station_id}_{month_tag}.png"

    if SAVE_PLOT:
        plt.savefig(fig2, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {fig2}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


    # =======================
    # Plot for leadtime = 48h
    # =======================
    try:
        subset_24 = plot_pd.query("leadtime == 48")
    except Exception:
        print(f"[WARN] No leadtime column or invalid values for {station_id}")
        return

    if subset_24.empty:
        print(f"[WARN] No 24h leadtime forecasts for {station_id}")
        return

    # Select observation times that actually have observations
    obs_times = (
        pd.to_datetime(subset_24.loc[subset_24["obs_TA"].notna(), "validtime"]
                    .drop_duplicates()
                    .sort_values())
    )

    # Pick every 6th observation to reduce density
    #obs_times_sampled = obs_times[::6]

    #subset_24 = subset_24[subset_24["validtime"].isin(obs_times_sampled)]

    if subset_24.empty:
        print(f"[WARN] No matching observation/forecast pairs after sampling for {station_id}")
        return

    plt.figure(figsize=(12, 5))

    # Plot each forecast system
    if RAW in subset_24.columns:
        plt.plot(subset_24["validtime"], subset_24[RAW],
                linestyle="-", lw=1.0, color="#999999", alpha=0.7, label="Raw forecast (48h lead)")

    if MOS in subset_24.columns:
        plt.plot(subset_24["validtime"], subset_24[MOS],
                lw=2.3, color="#637AB9", alpha=0.95, label="MOS corrected (48h lead)")

    if MLCOL in subset_24.columns:
        plt.plot(subset_24["validtime"], subset_24[MLCOL],
                lw=2.3, color="#B95E82", alpha=0.95, label=f"ML corrected ({ML_TAG}) (48h lead)")

    # Observations at those same validtimes
    if OBS in subset_24.columns:
        plt.scatter(subset_24["validtime"], subset_24[OBS],
                    s=12, color="black", label="Observation", zorder=3)

    plt.title(f"Station {station_id} — Inits {pd.to_datetime(start_init):%Y-%m-%d}..{pd.to_datetime(end_init):%Y-%m-%d}\n48h lead forecasts")
    plt.xlabel("Valid time")
    plt.ylabel("Temperature (K)")
    plt.grid(True, alpha=0.3)
    plt.legend(framealpha=0.85)
    plt.tight_layout()

    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_dir = OUT_DIR / station_id / month_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fig2 = out_dir / f"timeseries_ldt48_{station_id}_{month_tag}.png"

    if SAVE_PLOT:
        plt.savefig(fig2, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {fig2}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


    # =======================
    # NEW: Density scatter with (2) log ticks & (3) % outside
    # =======================
    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    out_dir = OUT_DIR / station_id / month_tag

    if MOS in plot_pd.columns:
        scatter_density_one(
            df=plot_pd,
            obs_col=OBS,
            pred_col=MOS,
            station_id=station_id,
            start_init=start_init,
            end_init=end_init,
            out_dir=out_dir,
            tag="MOS",
            use_log_counts=True,
            cmap="turbo",
            # NEW:
            use_variable_threshold=True,
            cold_cut=258.15, cold_T=5.0,
            cool_cut=268.15, cool_T=3.5,
            warm_T=2.5,
            show_band=True,          # if you want the interior band softly shaded
            units="K",
        )

    if MLCOL in plot_pd.columns:
        scatter_density_one(
            df=plot_pd,
            obs_col=OBS,
            pred_col=MLCOL,
            station_id=station_id,
            start_init=start_init,
            end_init=end_init,
            out_dir=out_dir,
            tag="ML",
            use_log_counts=True,
            cmap="turbo",
            # NEW:
            use_variable_threshold=True,
            cold_cut=258.15, cold_T=5.0,
            cool_cut=268.15, cool_T=3.5,
            warm_T=2.5,
            show_band=True,          # if you want the interior band softly shaded
            units="K",
        )

    # =======================
    # NEW: (6) Facets by lead time for MOS/ML
    # =======================
    if MLCOL in plot_pd.columns:
        facet_by_lead(
            plot_pd=plot_pd, obs_col=OBS, pred_col=MLCOL,
            station_id=station_id, start_init=start_init, end_init=end_init,
            leads=(12, 24, 48),
            cmap="turbo", units="K", tag="ML",
            out_dir=out_dir,
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
            out_dir=out_dir,
            adaptive_binning=True,
            scatter_threshold=12,
            target_pts_per_hex=8.0,
            gridsize_min=12, gridsize_max=50,
            use_log_counts_when_dense=True,
        )


    # (Optional) RAW
    # if RAW in plot_pd.columns:
    #     scatter_density_one(
    #         df=plot_pd,
    #         obs_col=OBS,
    #         pred_col=RAW,
    #         station_id=station_id,
    #         start_init=start_init,
    #         end_init=end_init,
    #         out_dir=out_dir,
    #         tag="RAW",
    #         gridsize=60,
    #         mincnt=1,
    #         use_log_counts=True,
    #     )


    # =======================
    # RMSE vs analysistime (aggregated over all leadtimes)
    # =======================
    def safe_rmse_col(d, col): 
        return rmse(d[col], d[OBS]) if col in d and OBS in d else np.nan

    metrics = (
        plot_pd.groupby("analysistime", group_keys=False)
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
    if "RMSE_RAW" in metrics: plt.plot(metrics["analysistime"], metrics["RMSE_RAW"], label="RAW", lw=1.2, color= colors["RAW"])
    if "RMSE_MOS" in metrics: plt.plot(metrics["analysistime"], metrics["RMSE_MOS"], label="MOS", lw=2, color= colors["MOS"])
    if "RMSE_ML" in metrics:  plt.plot(metrics["analysistime"], metrics["RMSE_ML"],  label=f"ML ({ML_TAG})", lw=2, color= colors["ML"])
    plt.title(f"RMSE vs analysistime — Station {station_id} ({month_tag})")
    plt.xlabel("Analysistime"); plt.ylabel("RMSE (K)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    fig2 = out_dir / f"RMSE_timeseries_{station_id}_{month_tag}.png"
    if SAVE_PLOT:
        plt.savefig(fig2, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved {fig2}")
    if SHOW_PLOT: plt.show()
    else: plt.close()

# =====================================================
# Loop over stations × months
# =====================================================
for sid in STATIONS:
    for m_start, m_end in month_windows(MONTH_START, MONTH_END):
        # analysistime window should cover the whole month
        run_one(station_id=str(sid),
                start_init=m_start.strftime("%Y-%m-%d %H:%M:%S"),
                end_init=m_end.strftime("%Y-%m-%d %H:%M:%S"))

