import argparse
import calendar
from datetime import datetime
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator


# =========================
# Columns
# =========================

SPLIT = "analysistime"
KEYS = [SPLIT, "validtime", "leadtime"]

OBS = "obs_TA"
RAW = "raw_fc"
MOS = "corrected_mos"


# =========================
# Argument parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create MOS vs ML scatter-density plots for temperature predictions, "
            "optionally filtered by station subset, month windows, seasons, and leadtime."
        )
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
        help="Directory containing ML evaluation parquet files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where figures will be saved.",
    )

    parser.add_argument(
        "--ml-tag",
        default="tuned_ah_2019",
        type=str,
        help=(
            "ML model tag used in filenames and corrected_<ml-tag> column. "
            "Default: tuned_ah_2019."
        ),
    )

    parser.add_argument(
        "--ml-name",
        default="EC_ML_XGBoost_2019",
        type=str,
        help="Readable ML model name used in plot labels. Default: EC_ML_XGBoost_2019.",
    )

    parser.add_argument(
        "--leadtime",
        default=48,
        type=int,
        help="Maximum leadtime to include. Default: 48.",
    )

    parser.add_argument(
        "--seasonal",
        action="store_true",
        help="Run seasonal plots using seasonal MOS files.",
    )

    parser.add_argument(
        "--non-seasonal",
        action="store_true",
        help="Run non-seasonal monthly plots.",
    )

    parser.add_argument(
        "--month-start",
        default="2024-09-01 00:00:00",
        type=str,
        help=(
            "Start datetime for non-seasonal monthly plots. "
            "Default: 2024-09-01 00:00:00."
        ),
    )

    parser.add_argument(
        "--month-end",
        default="2025-08-31 12:00:00",
        type=str,
        help=(
            "End datetime for non-seasonal monthly plots. "
            "Default: 2025-08-31 12:00:00."
        ),
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["autumn", "winter", "spring", "summer"],
        help=(
            "Seasons to process in seasonal mode. "
            "Default: autumn winter spring summer."
        ),
    )

    parser.add_argument(
        "--stations-file",
        default=None,
        type=str,
        help=(
            "Optional text file containing station IDs to include, one per line. "
            "If omitted, all stations in the data are used."
        ),
    )

    parser.add_argument(
        "--station-group-id",
        default="All stations",
        type=str,
        help="Station/group ID used in output path and filenames. Default: All stations.",
    )

    parser.add_argument(
        "--station-group-name",
        default="All stations",
        type=str,
        help="Station/group name used in plot titles. Default: All stations.",
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show plots interactively.",
    )

    parser.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Do not save plots.",
    )

    parser.add_argument(
        "--fig-dpi",
        default=150,
        type=int,
        help="Figure DPI. Default: 150.",
    )

    return parser.parse_args()


def load_station_subset(stations_file: str | None) -> list[str] | None:
    """
    Load optional station subset.

    Returns:
        None if no station file is given, meaning use all stations.
        Otherwise returns a list of station IDs as strings.
    """
    if stations_file is None:
        return None

    path = Path(stations_file)

    if not path.exists():
        raise FileNotFoundError(f"Stations file not found: {path}")

    with open(path, "r") as f:
        stations = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not stations:
        raise ValueError(f"No station IDs found in station file: {path}")

    return stations


def safe_name(text: str) -> str:
    """Make a string safe for filenames and paths."""
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("≤", "le")
        .replace("(", "")
        .replace(")", "")
    )


# =========================
# Helpers
# =========================

def rmse(a, b):
    """Calculate RMSE between two sets of values."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    m = np.isfinite(a) & np.isfinite(b)

    if not m.any():
        return np.nan

    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def month_windows(start_str, end_str):
    """
    Yield monthly windows as (start_dt, end_dt_inclusive)
    for each month overlapping [start, end].
    """
    start = pd.Timestamp(start_str).to_pydatetime()
    end = pd.Timestamp(end_str).to_pydatetime()

    y, m = start.year, start.month
    cur = datetime(y, m, 1)

    while cur <= end.replace(day=1):
        nd = calendar.monthrange(cur.year, cur.month)[1]

        m_start = max(cur, start)
        m_end = min(datetime(cur.year, cur.month, nd, 23, 59, 59), end)

        yield m_start, m_end

        ny, nm = (
            cur.year + (cur.month == 12),
            1 if cur.month == 12 else cur.month + 1,
        )
        cur = datetime(ny, nm, 1)


def _pair_stats(y_true, y_pred):
    """
    Calculate statistics for observations vs predictions.

    Returns:
        n, bias, RMSE, and R².
    """
    a = np.asarray(y_true, float)
    b = np.asarray(y_pred, float)

    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]

    n = a.size

    if n == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }

    bias = float(np.mean(b - a))
    rmse_v = float(np.sqrt(np.mean((b - a) ** 2)))

    if np.std(a) == 0 or np.std(b) == 0:
        r2 = np.nan
    else:
        r = float(np.corrcoef(a, b)[0, 1])
        r2 = r * r

    return {
        "n": int(n),
        "bias": bias,
        "rmse": rmse_v,
        "r2": r2,
    }


def mos_coverage_window(mos_eval: pl.DataFrame) -> tuple[datetime, datetime]:
    usable = mos_eval.filter(
        pl.col(MOS).is_not_null()
        & pl.col(RAW).is_not_null()
        & pl.col(OBS).is_not_null()
        & pl.col(SPLIT).is_not_null()
    )

    if usable.height == 0:
        raise ValueError("MOS eval rows have no usable values.")

    return usable.select(
        pl.col(SPLIT).min().alias("t0"),
        pl.col(SPLIT).max().alias("t1"),
    ).row(0)


# =========================
# Data loading and preparation
# =========================

def load_eval_rows_evaldir(
    eval_dir: Path,
    pattern: str,
    tag: str,
    ml_col: str,
):
    """
    Load evaluation rows from parquet files.
    """
    files = sorted(eval_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")

    dfs = []

    for f in files:
        cols = KEYS + [RAW, OBS, ml_col, MOS, "SID"]

        schema = pl.scan_parquet(str(f)).collect_schema().names()
        existing = [c for c in cols if c in schema]

        if not existing:
            continue

        df = pl.read_parquet(f, columns=existing)

        exprs = []

        if "SID" in df.columns:
            exprs.append(pl.col("SID").cast(pl.Utf8))

        if SPLIT in df.columns:
            exprs.append(
                pl.col(SPLIT)
                .cast(pl.Utf8)
                .str.to_datetime(strict=False)
                .alias(SPLIT)
            )

        if "validtime" in df.columns:
            exprs.append(
                pl.col("validtime")
                .cast(pl.Utf8)
                .str.to_datetime(strict=False)
                .alias("validtime")
            )

        if "leadtime" in df.columns:
            exprs.append(pl.col("leadtime").cast(pl.Int64, strict=False))

        if exprs:
            df = df.with_columns(exprs)

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No usable parquet data found in: {eval_dir}/{pattern}")

    out = pl.concat(dfs, how="vertical_relaxed")

    print(f"[INFO] {tag} rows loaded: {out.height:,}")

    return out


def filter_for_station_init(
    df: pl.DataFrame,
    station_subset: list[str] | None,
    start_init,
    end_init,
    tag: str,
):
    """
    Filter dataframe to optional station subset and analysistime window.

    If station_subset is None, all stations are used.
    """
    filter_expr = (
        (pl.col(SPLIT) >= start_init)
        & (pl.col(SPLIT) <= end_init)
    )

    if station_subset is not None:
        filter_expr = filter_expr & pl.col("SID").is_in([str(s) for s in station_subset])

    plot_df = df.filter(filter_expr)

    if plot_df.height == 0:
        station_msg = (
            "all stations"
            if station_subset is None
            else f"{len(station_subset)} selected stations"
        )

        raise ValueError(
            f"No {tag} rows for {station_msg}, "
            f"window={start_init} -> {end_init}"
        )

    return plot_df


def data_prep_for_plot(ml_plot, mos_plot, start_init, end_init):
    """
    Join ML and MOS prediction data and prepare a pandas dataframe for plotting.
    """
    if ml_plot.height == 0:
        print(f"[WARN] No ML rows for init={start_init}_{end_init}. Plotting without ML.")
        joined = mos_plot

    elif mos_plot.height == 0:
        print(f"[WARN] No MOS rows for init={start_init}_{end_init}. Plotting without MOS.")
        joined = ml_plot

    else:
        joined = ml_plot.join(
            mos_plot.select(KEYS + [MOS]),
            on=KEYS,
            how="left",
            suffix="_mos",
        )

        for col in [RAW, OBS]:
            col_ml = f"{col}_ml"

            if col_ml in joined.columns and col in joined.columns:
                joined = joined.drop(col_ml)

    plot_pd = joined.to_pandas()

    plot_pd["validtime"] = pd.to_datetime(plot_pd["validtime"], errors="coerce")
    plot_pd["analysistime"] = pd.to_datetime(plot_pd["analysistime"], errors="coerce")

    plot_pd = plot_pd.sort_values(["analysistime", "validtime"])

    return plot_pd


# =========================
# Plot functions
# =========================

def scatter_density_one(
    df: pd.DataFrame,
    obs_col: str,
    pred_col: str,
    station_id: str,
    station_name: str,
    start_init,
    end_init,
    tag: str,
    cmap: str,
    out_dir: Path,
    fig_dpi: int,
    save_plot: bool,
    show_plot: bool,
    ml_name: str,
    gridsize: int = 60,
    mincnt: int = 1,
    use_log_counts: bool = True,
    err_thresh: float | None = None,
    err_thresh_pos: float | None = None,
    err_thresh_neg: float | None = None,
    shade_alpha: float = 0.12,
    band_alpha: float = 0.06,
    show_band: bool = False,
    show_pct_outside: bool = True,
    units: str = "K",
    use_variable_threshold: bool = False,
    cold_cut: float = 258.15,
    cool_cut: float = 268.15,
    cold_T: float = 5.0,
    cool_T: float = 3.5,
    warm_T: float = 2.5,
    season=None,
):
    """
    Create a scatter-density plot of predictions vs observations.
    """
    a = pd.to_numeric(df.get(obs_col, pd.Series(dtype=float)), errors="coerce")
    b = pd.to_numeric(df.get(pred_col, pd.Series(dtype=float)), errors="coerce")

    m = np.isfinite(a) & np.isfinite(b)

    a = a[m].to_numpy()
    b = b[m].to_numpy()

    if a.size == 0:
        print(f"[WARN] No finite {obs_col}/{pred_col} pairs for {station_id} ({tag})")
        return

    st = _pair_stats(a, b)

    x_lo, x_hi = np.nanpercentile(a, [0.5, 99.5])
    y_lo, y_hi = np.nanpercentile(b, [0.5, 99.5])

    sx, sy = (x_hi - x_lo), (y_hi - y_lo)

    if not np.isfinite(sx) or sx <= 0:
        sx = 1.0

    if not np.isfinite(sy) or sy <= 0:
        sy = 1.0

    pad_x = 0.2 * sx + sx / gridsize
    pad_y = 0.2 * sy + sy / gridsize

    x_center = (x_hi + x_lo) / 2
    y_center = (y_hi + y_lo) / 2

    half_span = 0.5 * max(sx + 2 * pad_x, sy + 2 * pad_y)

    x0, x1 = x_center - half_span, x_center + half_span
    y0, y1 = y_center - half_span, y_center + half_span

    margin = 0.01 * half_span
    x0 -= margin
    x1 += margin
    y0 -= margin
    y1 += margin

    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    norm = LogNorm() if use_log_counts else None

    hb = ax.hexbin(
        a,
        b,
        gridsize=gridsize,
        mincnt=mincnt,
        norm=norm,
        cmap=cmap,
        extent=(x0, x1, y0, y1),
    )

    cb = fig.colorbar(hb, ax=ax, shrink=0.9)
    cb.set_label("Count per hexbin")

    if use_log_counts:
        cb.locator = LogLocator(base=10, subs="all")
        cb.update_ticks()

    want_fixed = (
        (err_thresh is not None)
        or (err_thresh_pos is not None)
        or (err_thresh_neg is not None)
    )

    if use_variable_threshold or want_fixed:
        xs = np.linspace(x0, x1, 512)

        if use_variable_threshold:
            tx = np.where(
                xs <= cold_cut,
                cold_T,
                np.where(xs <= cool_cut, cool_T, warm_T),
            )

            y_plus = xs + tx
            y_minus = xs - tx

            label_band_main = "|pred−obs| ≤ T(obs)"
            upper_lbl = "pred−obs ≥ +T(obs)"
            lower_lbl = "pred−obs ≤ −T(obs)"

        else:
            t_pos = err_thresh_pos if err_thresh_pos is not None else (err_thresh or 0.0)
            t_neg = err_thresh_neg if err_thresh_neg is not None else (err_thresh or 0.0)

            y_plus = xs + t_pos
            y_minus = xs - t_neg

            label_band_main = f"|pred−obs| ≤ {max(t_pos, t_neg):g} {units}" if show_band else None
            upper_lbl = f"pred−obs ≥ +{t_pos:g} {units}"
            lower_lbl = f"pred−obs ≤ −{t_neg:g} {units}"

        ax.fill_between(xs, y_plus, y1, alpha=shade_alpha, label=upper_lbl, color="#E64A19")
        ax.fill_between(xs, y0, y_minus, alpha=shade_alpha, label=lower_lbl, color="#004E9B")

        if show_band:
            ax.fill_between(
                xs,
                y_minus,
                y_plus,
                alpha=band_alpha,
                color="#66E247",
                label=label_band_main if label_band_main else None,
            )

        ax.plot(xs, y_plus, linestyle="-", linewidth=1.0, color="#E64A19")
        ax.plot(xs, y_minus, linestyle="-", linewidth=1.0, color="#004E9B")

        if show_pct_outside:
            err = b - a

            if use_variable_threshold:
                ta = np.where(
                    a <= cold_cut,
                    cold_T,
                    np.where(a <= cool_cut, cool_T, warm_T),
                )
                out_frac = np.mean(np.abs(err) >= ta) * 100.0

            else:
                t_pos = err_thresh_pos if err_thresh_pos is not None else (err_thresh or 0.0)
                t_neg = err_thresh_neg if err_thresh_neg is not None else (err_thresh or 0.0)
                out_frac = np.mean((err >= t_pos) | (err <= -t_neg)) * 100.0

            ax.text(
                0.02,
                0.8,
                f"{out_frac:.1f}% outside band",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round", alpha=0.20),
            )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.plot([x0, x1], [x0, x1], linewidth=1.2, color="#444444")
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"Observation [{units}]", fontsize=12)

    if tag in ("MOS", ml_name):
        ylabel = f"{tag} corrected prediction [{units}]"
    else:
        ylabel = f"Prediction [{units}]"

    ax.set_ylabel(ylabel, fontsize=12)

    title_range = f"{pd.to_datetime(start_init):%Y-%m-%d} .. {pd.to_datetime(end_init):%Y-%m-%d}"

    ax.set_title(
        f"{station_name} {station_id}\n"
        f"Inits {title_range}\n"
        f"Obs vs {tag} (density)",
        fontsize=20,
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.85)

    s_txt = (
        f"N = {st['n']}\n"
        f"Bias = {st['bias']:.3g}\n"
        f"RMSE = {st['rmse']:.3g}\n"
        f"R² = {st['r2']:.3g}"
        if np.isfinite(st["r2"])
        else
        f"N = {st['n']}\n"
        f"Bias = {st['bias']:.3g}\n"
        f"RMSE = {st['rmse']:.3g}\n"
        f"R² = NA"
    )

    ax.text(
        0.02,
        0.98,
        s_txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.2),
    )

    fig.tight_layout()

    month_tag = f"{pd.to_datetime(start_init):%Y%m}"
    season_tag = safe_name(season if season is not None else "full")
    station_tag = safe_name(station_id)
    model_tag = safe_name(tag)

    out_path = out_dir / station_tag / month_tag
    out_path.mkdir(parents=True, exist_ok=True)

    out_svg = out_path / f"scatter_density_{model_tag}_{station_tag}_{month_tag}_{season_tag}.svg"

    if save_plot:
        fig.savefig(out_svg, dpi=fig_dpi, bbox_inches="tight")
        print(f"[OK] Saved {out_svg}")

    if show_plot:
        plt.show()

    plt.close(fig)


def plot_three_models(
    plot_pd: pd.DataFrame,
    station_id: str,
    station_name: str,
    start_init,
    end_init,
    leadtime: int,
    ml_col: str,
    ml_name: str,
    out_dir: Path,
    fig_dpi: int,
    save_plot: bool,
    show_plot: bool,
    season=None,
):
    """Create MOS, ML, and ECMWF plots when the needed columns exist."""
    common_kwargs = dict(
        df=plot_pd,
        station_id=station_id,
        station_name=station_name,
        start_init=start_init,
        end_init=end_init,
        use_log_counts=True,
        cmap="turbo",
        use_variable_threshold=True,
        cold_cut=258.15,
        cold_T=5.0,
        cool_cut=268.15,
        cool_T=3.5,
        warm_T=2.5,
        show_band=True,
        units="K",
        out_dir=out_dir,
        fig_dpi=fig_dpi,
        save_plot=save_plot,
        show_plot=show_plot,
        ml_name=ml_name,
        season=season,
    )

    if MOS in plot_pd.columns:
        scatter_density_one(
            obs_col=OBS,
            pred_col=MOS,
            tag=f"MOS_le{leadtime}h",
            **common_kwargs,
        )

    if ml_col in plot_pd.columns:
        scatter_density_one(
            obs_col=OBS,
            pred_col=ml_col,
            tag=f"{ml_name}_le{leadtime}h",
            **common_kwargs,
        )

    if RAW in plot_pd.columns:
        scatter_density_one(
            obs_col=OBS,
            pred_col=RAW,
            tag=f"ECMWF_le{leadtime}h",
            **common_kwargs,
        )


# =========================
# Main
# =========================

def main():
    args = parse_args()

    if args.seasonal and args.non_seasonal:
        raise ValueError("Use either --seasonal or --non-seasonal, not both.")

    seasonal = True if args.seasonal else False

    mos_dir = Path(args.mos_dir)
    ml_dir = Path(args.ml_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ml_tag = args.ml_tag
    ml_name = args.ml_name
    ml_col = f"corrected_{ml_tag}"

    leadtime = args.leadtime
    save_plot = not args.no_save_plot
    show_plot = args.show_plot
    fig_dpi = args.fig_dpi

    station_subset = load_station_subset(args.stations_file)

    if station_subset is None:
        print("[INFO] Using all stations in the data.")
    else:
        print(f"[INFO] Using station subset from {args.stations_file}: {len(station_subset)} stations")

    print(f"[INFO] MOS directory: {mos_dir}")
    print(f"[INFO] ML directory: {ml_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] ML tag: {ml_tag}")
    print(f"[INFO] ML column: {ml_col}")
    print(f"[INFO] ML name: {ml_name}")
    print(f"[INFO] Leadtime <= {leadtime}")
    print(f"[INFO] Seasonal mode: {seasonal}")

    station_id = args.station_group_id
    station_name = args.station_group_name

    if seasonal:
        ml_data_full = load_eval_rows_evaldir(
            ml_dir,
            f"eval_rows_{SPLIT}_{ml_tag}_*.parquet",
            ml_col,
            ml_col=ml_col,
        )

        for season in args.seasons:
            print(f"\n[INFO] Processing season: {season}")

            mos_data_season = load_eval_rows_evaldir(
                mos_dir,
                f"eval_rows_{SPLIT}_MOS_*_{season}.parquet",
                MOS,
                ml_col=ml_col,
            )

            start_init, end_init = mos_coverage_window(mos_data_season)

            mos_ltd = mos_data_season.filter(pl.col("leadtime") <= leadtime)
            ml_ltd = ml_data_full.filter(pl.col("leadtime") <= leadtime)

            try:
                all_mos_ltd = filter_for_station_init(
                    mos_ltd,
                    station_subset,
                    start_init,
                    end_init,
                    MOS,
                )
            except ValueError as e:
                print(f"[WARN] {e}")
                all_mos_ltd = pl.DataFrame()

            try:
                all_ml_ltd = filter_for_station_init(
                    ml_ltd,
                    station_subset,
                    start_init,
                    end_init,
                    ml_tag,
                )
            except ValueError as e:
                print(f"[WARN] {e}")
                all_ml_ltd = pl.DataFrame()

            if all_mos_ltd.height == 0 and all_ml_ltd.height == 0:
                print(f"[{season} leadtime <= {leadtime}] No data after filtering; skipping.")
                continue

            plot_pd = data_prep_for_plot(
                all_ml_ltd,
                all_mos_ltd,
                start_init=start_init,
                end_init=end_init,
            )

            plot_three_models(
                plot_pd=plot_pd,
                station_id=station_id,
                station_name=station_name,
                start_init=start_init,
                end_init=end_init,
                leadtime=leadtime,
                ml_col=ml_col,
                ml_name=ml_name,
                out_dir=out_dir,
                fig_dpi=fig_dpi,
                save_plot=save_plot,
                show_plot=show_plot,
                season=season,
            )

    else:
        print("\n[INFO] Processing non-seasonal monthly plots")

        mos_data_full = load_eval_rows_evaldir(
            mos_dir,
            f"eval_rows_{SPLIT}_MOS_*.parquet",
            MOS,
            ml_col=ml_col,
        )

        ml_data_full = load_eval_rows_evaldir(
            ml_dir,
            f"eval_rows_{SPLIT}_{ml_tag}_*.parquet",
            ml_col,
            ml_col=ml_col,
        )

        for m_start, m_end in month_windows(args.month_start, args.month_end):
            start_init = m_start
            end_init = m_end

            print(f"\n[INFO] Processing month window: {start_init} -> {end_init}")

            mos_ltd = mos_data_full.filter(pl.col("leadtime") <= leadtime)
            ml_ltd = ml_data_full.filter(pl.col("leadtime") <= leadtime)

            try:
                all_mos_ltd = filter_for_station_init(
                    mos_ltd,
                    station_subset,
                    start_init,
                    end_init,
                    MOS,
                )
            except ValueError as e:
                print(f"[WARN] {e}")
                all_mos_ltd = pl.DataFrame()

            try:
                all_ml_ltd = filter_for_station_init(
                    ml_ltd,
                    station_subset,
                    start_init,
                    end_init,
                    ml_tag,
                )
            except ValueError as e:
                print(f"[WARN] {e}")
                all_ml_ltd = pl.DataFrame()

            if all_mos_ltd.height == 0 and all_ml_ltd.height == 0:
                print(
                    f"[{start_init} -> {end_init}, leadtime <= {leadtime}] "
                    f"No data after filtering; skipping."
                )
                continue

            plot_pd = data_prep_for_plot(
                all_ml_ltd,
                all_mos_ltd,
                start_init=start_init,
                end_init=end_init,
            )

            plot_three_models(
                plot_pd=plot_pd,
                station_id=station_id,
                station_name=station_name,
                start_init=start_init,
                end_init=end_init,
                leadtime=leadtime,
                ml_col=ml_col,
                ml_name=ml_name,
                out_dir=out_dir,
                fig_dpi=fig_dpi,
                save_plot=save_plot,
                show_plot=show_plot,
                season="full",
            )


if __name__ == "__main__":
    main()