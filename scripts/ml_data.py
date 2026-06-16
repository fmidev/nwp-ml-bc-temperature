"""
Create monthly XGBoost-ready MOS training data with multiple targets.

This script reads monthly forecast files, yearly observation CSV files, and station
metadata. It writes one clean parquet file per forecast month.

The clean output contains shared columns only once:
    ID/time columns:
        SID, analysistime, validtime

    XGBoost features:
        FEATS

    target columns:
        target_bias_TA    = obs_TA - T2
        target_bias_TD    = obs_TD - D2
        target_bias_TMAX  = obs_TMAX_aligned - MX2T
        target_bias_TMIN  = obs_TMIN_aligned - MN2T

Daily max/min alignment follows the logic:
    - observed max at 18 UTC is copied to valid hours 07..18
    - observed min at 06 UTC is copied to valid hours 19..23 and 00..06
    - only forecast rows with leadtime >= --extreme-min-leadtime receive TMAX/TMIN targets

For TMIN:
    - valid hours 00..06 use same-day 06 UTC observation
    - valid hours 19..23 use next-day 06 UTC observation

Training scripts can then select a target with, for example:
    --target-col target_bias_TD
    --target-col target_bias_TMAX

Example:
    python3.11 ml_data_multi_target.py \
      --forecast-input "/training-data/MOS_data_for_all_stations/forecasts" \
      --observations "/training-data/MOS_data_for_all_stations/observations2" \
      --stations "/training-data/MOS_data_for_all_stations/mos_stations_fmisid_Europe.csv" \
      --output-dir "/training-data/xgb_MOS_tmp" \
      --clean-xgb-output-dir "/training-data/xgb_MOS_clean_multi_rg100k_with_id" \
      --clean-output-prefix "ml_data_clean" \
      --clean-row-group-size 100000 \
      --clean-compression "zstd" \
      --clean-only \
      --threads 32
"""

import argparse
import gc
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Model columns
# -----------------------------------------------------------------------------

SID_COL_DEFAULT = "SID"
ANALYSIS_TIME_COL_DEFAULT = "analysistime"
VALID_TIME_COL_DEFAULT = "validtime"
LEADTIME_COL_DEFAULT = "leadtime"

TEMP_FC = "T2"
DEW_FC = "D2"
TMAX_FC = "MX2T"
TMIN_FC = "MN2T"

TARGET_TA = "target_bias_TA"
TARGET_TD = "target_bias_TD"
TARGET_TMAX = "target_bias_TMAX"
TARGET_TMIN = "target_bias_TMIN"
TARGET_COLS = [TARGET_TA, TARGET_TD, TARGET_TMAX, TARGET_TMIN]

# Backward-compatible single target name, optional.
TARGET_BIAS_LEGACY = "target_bias"

weather = [
    "MSL", TEMP_FC, DEW_FC, "U10", "V10", "LCC", "MCC", "SKT",
    TMAX_FC, TMIN_FC, "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1",
]

meta = [
    "leadtime", "lon", "lat", "elev",
    "sin_hod", "cos_hod", "sin_doy", "cos_doy",
    "analysishour",
]

FEATS = weather + meta


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create monthly clean XGBoost parquet files with shared features and "
            "multiple target columns: TA, TD, TMAX, TMIN."
        )
    )

    parser.add_argument(
        "--forecast-input",
        required=True,
        type=str,
        help=(
            "Forecast input directory, single file, or glob. Supports CSV and parquet. "
            "For your data this is usually /training-data/MOS_data_for_all_stations/forecasts."
        ),
    )

    parser.add_argument(
        "--forecast-pattern",
        default=None,
        type=str,
        help=(
            "Pattern used if --forecast-input is a directory. If omitted, tries "
            "*.csv first and then *.parquet."
        ),
    )

    parser.add_argument(
        "--observations",
        required=True,
        type=str,
        help=(
            "Observation CSV file or directory with yearly observation CSVs, e.g. "
            "/training-data/MOS_data_for_all_stations/observations2."
        ),
    )

    parser.add_argument(
        "--obs-pattern",
        default="observations_*.csv",
        type=str,
        help="Pattern used when --observations is a directory. Default: observations_*.csv.",
    )

    parser.add_argument(
        "--stations",
        required=True,
        type=str,
        help="Station metadata CSV. Must contain SID and usually lon/lat/elev.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory for optional full combined parquet output.",
    )

    parser.add_argument(
        "--clean-xgb-output-dir",
        required=True,
        type=str,
        help="Directory for clean multi-target XGBoost parquet files.",
    )

    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only write clean parquet files, not full combined parquet files.",
    )

    parser.add_argument(
        "--output-prefix",
        default="combined",
        type=str,
        help="Prefix for full combined parquet files. Default: combined.",
    )

    parser.add_argument(
        "--clean-output-prefix",
        default="ml_data_clean",
        type=str,
        help="Prefix for clean parquet files. Default: ml_data_clean.",
    )

    parser.add_argument(
        "--clean-row-group-size",
        default=100_000,
        type=int,
        help="Parquet row group size for clean files. Default: 100000.",
    )

    parser.add_argument(
        "--clean-compression",
        default="zstd",
        type=str,
        help="Parquet compression for clean files. Default: zstd.",
    )

    parser.add_argument(
        "--threads",
        default="16",
        type=str,
        help="OMP/MKL thread count. Default: 16.",
    )

    parser.add_argument("--station-id-col", default=SID_COL_DEFAULT, type=str)
    parser.add_argument("--analysis-time-col", default=ANALYSIS_TIME_COL_DEFAULT, type=str)
    parser.add_argument("--valid-time-col", default=VALID_TIME_COL_DEFAULT, type=str)
    parser.add_argument("--leadtime-col", default=LEADTIME_COL_DEFAULT, type=str)
    parser.add_argument("--obs-time-col", default="obstime", type=str)

    parser.add_argument(
        "--obs-ta-col",
        default="obs_TA",
        type=str,
        help="Hourly observed air temperature column. Default: obs_TA.",
    )

    parser.add_argument(
        "--obs-td-col",
        default="obs_TD",
        type=str,
        help="Hourly observed dewpoint column. Default: obs_TD.",
    )

    parser.add_argument(
        "--obs-tmax-col",
        default="obs_TAMAX12H",
        type=str,
        help="Observed daily maximum temperature column. Default: obs_TAMAX12H.",
    )

    parser.add_argument(
        "--obs-tmin-col",
        default="obs_TAMIN12H",
        type=str,
        help="Observed daily minimum temperature column. Default: obs_TAMIN12H.",
    )

    parser.add_argument(
        "--extreme-min-leadtime",
        default=12,
        type=int,
        help=(
            "Minimum leadtime for TMAX/TMIN targets, matching old logic leadtime >= 12. "
            "Default: 12."
        ),
    )

    parser.add_argument(
        "--strict-clean-features",
        action="store_true",
        help="Fail if a feature column is missing. Otherwise missing features are NaN.",
    )

    parser.add_argument(
        "--write-legacy-target-bias",
        action="store_true",
        help=(
            "Also write target_bias as an alias of target_bias_TA for backwards compatibility."
        ),
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------

def resolve_files(input_path: str, pattern: str | None = None) -> list[Path]:
    p = Path(input_path)

    if p.is_dir():
        if pattern is not None:
            files = sorted(p.glob(pattern))
        else:
            files = sorted(p.glob("*.csv"))
            if not files:
                files = sorted(p.glob("*.parquet"))
    elif p.is_file():
        files = [p]
    else:
        from glob import glob
        files = [Path(x) for x in sorted(glob(input_path))]

    if not files:
        raise FileNotFoundError(f"No files found from: {input_path}")

    return files


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def ym_from_filename(path: Path) -> str:
    """
    Supports:
        mos_archive_data_202112.csv
        ml_data_clean_2021-12.parquet
        part-2021-12-...
        any filename containing YYYY-MM or YYYYMM
    """
    name = path.name

    m = re.search(r"(\d{4})-(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.search(r"(\d{4})(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.match(r"part-(\d{4})-(\d{1,2})-", name)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"

    raise ValueError(f"Could not parse YYYY-MM from filename: {path}")


def years_needed_for_month(month: str) -> list[str]:
    """
    Forecast month may need observations from same month/year and next year for
    leadtimes/validtimes crossing year boundary, especially December and TMIN next-day 06.
    """
    year = int(month[:4])
    mm = int(month[5:7])
    years = {str(year)}
    if mm == 12:
        years.add(str(year + 1))
    return sorted(years)


def observation_files_for_years(obs_path: str, years: list[str], obs_pattern: str) -> list[Path]:
    p = Path(obs_path)

    if p.is_file():
        return [p]

    if not p.is_dir():
        raise FileNotFoundError(f"Observation path does not exist: {obs_path}")

    files = []
    for y in years:
        exact = p / f"observations_{y}.csv"
        if exact.exists():
            files.append(exact)
            continue

        matches = sorted(p.glob(obs_pattern.replace("*", f"*{y}*")))
        if matches:
            files.extend(matches)

    if not files:
        raise FileNotFoundError(f"No observation files found for years={years} in {p}")

    # Remove duplicates while preserving order.
    out = []
    seen = set()
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


# -----------------------------------------------------------------------------
# Data cleaning helpers
# -----------------------------------------------------------------------------

def key_formatting(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = [c for c in keys if c in df.columns]
    if not cols:
        return df

    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("string").str.strip()
    return df


def parse_datetime_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def ensure_analysishour(df: pd.DataFrame, analysis_time_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[analysis_time_col], errors="coerce")
    out["analysishour"] = dt.dt.hour.astype("float32")
    return out


def ensure_station_metadata(df: pd.DataFrame, stations: pd.DataFrame, sid_col: str) -> pd.DataFrame:
    out = df.merge(stations, on=sid_col, how="left", validate="m:1")
    return out


def deduplicate_observation_column(
    obs: pd.DataFrame,
    sid_col: str,
    time_col: str,
    value_col: str,
    label: str,
) -> pd.DataFrame:
    if value_col not in obs.columns:
        return pd.DataFrame(columns=[sid_col, time_col, value_col])

    small = obs[[sid_col, time_col, value_col]].copy()
    small[value_col] = pd.to_numeric(small[value_col], errors="coerce")
    small = small.dropna(subset=[sid_col, time_col])

    before = len(small)
    dup_count = int(small.duplicated(subset=[sid_col, time_col], keep=False).sum())

    if dup_count:
        print(
            f"[WARN] Duplicate observation rows found for {label} by {sid_col}+{time_col}: "
            f"{dup_count:,}. Collapsing duplicates by mean.",
            flush=True,
        )

    small = (
        small.groupby([sid_col, time_col], as_index=False)[value_col]
        .mean()
    )

    after = len(small)
    if after != before:
        print(
            f"[INFO] {label} observation rows after deduplication: {after:,} / {before:,}",
            flush=True,
        )

    return small


def add_tmax_tmin_lookup_times(
    df: pd.DataFrame,
    valid_time_col: str,
    leadtime_col: str,
    extreme_min_leadtime: int,
) -> pd.DataFrame:
    """
    Implements the old array logic in dataframe form.

    Old logic:
        ilocs_tmax = where(hour == 18 and leadtime >= 12)
        ilocs_tmin = where(hour == 6 and leadtime >= 12)
        for i in range(0, 12): copy value to ilocs - i

    Equivalent validtime mapping:
        TMAX rows: valid hour 07..18 use same-day 18 UTC observation.
        TMIN rows: valid hour 19..23 use next-day 06 UTC observation.
                   valid hour 00..06 use same-day 06 UTC observation.

    Additionally, only rows with leadtime >= extreme_min_leadtime get an extrema target.
    """
    out = df.copy()

    valid_dt = pd.to_datetime(out[valid_time_col], errors="coerce")
    valid_date = valid_dt.dt.floor("D")
    hour = valid_dt.dt.hour
    leadtime = pd.to_numeric(out[leadtime_col], errors="coerce")

    out["tmax_obs_time"] = pd.NaT
    out["tmin_obs_time"] = pd.NaT

    lead_ok = leadtime >= int(extreme_min_leadtime)

    mask_tmax = lead_ok & hour.between(7, 18)
    out.loc[mask_tmax, "tmax_obs_time"] = valid_date[mask_tmax] + pd.Timedelta(hours=18)

    mask_tmin_same_day = lead_ok & hour.between(0, 6)
    out.loc[mask_tmin_same_day, "tmin_obs_time"] = (
        valid_date[mask_tmin_same_day] + pd.Timedelta(hours=6)
    )

    mask_tmin_next_day = lead_ok & hour.between(19, 23)
    out.loc[mask_tmin_next_day, "tmin_obs_time"] = (
        valid_date[mask_tmin_next_day] + pd.Timedelta(days=1, hours=6)
    )

    return out


def merge_hourly_observations(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    sid_col: str,
    obs_time_col: str,
    valid_time_col: str,
    obs_ta_col: str,
    obs_td_col: str,
) -> pd.DataFrame:
    out = df.copy()

    hourly_cols = [c for c in [obs_ta_col, obs_td_col] if c in obs.columns]
    if not hourly_cols:
        print("[WARN] No hourly TA/TD observation columns found.", flush=True)
        return out

    parts = []
    for c in hourly_cols:
        part = deduplicate_observation_column(
            obs,
            sid_col=sid_col,
            time_col=obs_time_col,
            value_col=c,
            label=c,
        )
        parts.append(part)

    # Merge separate deduplicated value columns into one lookup table.
    hourly = None
    for part in parts:
        if hourly is None:
            hourly = part
        else:
            hourly = hourly.merge(part, on=[sid_col, obs_time_col], how="outer", validate="1:1")

    hourly = hourly.rename(columns={obs_time_col: valid_time_col})

    out = out.merge(
        hourly,
        on=[sid_col, valid_time_col],
        how="left",
        validate="m:1",
    )

    return out


def merge_extreme_observations(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    sid_col: str,
    obs_time_col: str,
    obs_tmax_col: str,
    obs_tmin_col: str,
) -> pd.DataFrame:
    out = df.copy()

    if obs_tmax_col in obs.columns:
        tmax_lookup = deduplicate_observation_column(
            obs,
            sid_col=sid_col,
            time_col=obs_time_col,
            value_col=obs_tmax_col,
            label=obs_tmax_col,
        ).rename(
            columns={
                obs_time_col: "tmax_obs_time",
                obs_tmax_col: "obs_TMAX_aligned",
            }
        )

        out = out.merge(
            tmax_lookup,
            on=[sid_col, "tmax_obs_time"],
            how="left",
            validate="m:1",
        )
    else:
        print(f"[WARN] Observation column {obs_tmax_col} not found; TMAX target will be NaN.", flush=True)
        out["obs_TMAX_aligned"] = np.nan

    if obs_tmin_col in obs.columns:
        tmin_lookup = deduplicate_observation_column(
            obs,
            sid_col=sid_col,
            time_col=obs_time_col,
            value_col=obs_tmin_col,
            label=obs_tmin_col,
        ).rename(
            columns={
                obs_time_col: "tmin_obs_time",
                obs_tmin_col: "obs_TMIN_aligned",
            }
        )

        out = out.merge(
            tmin_lookup,
            on=[sid_col, "tmin_obs_time"],
            how="left",
            validate="m:1",
        )
    else:
        print(f"[WARN] Observation column {obs_tmin_col} not found; TMIN target will be NaN.", flush=True)
        out["obs_TMIN_aligned"] = np.nan

    return out


def make_clean_multi_target_frame(
    final_df: pd.DataFrame,
    sid_col: str,
    analysis_time_col: str,
    valid_time_col: str,
    obs_ta_col: str,
    obs_td_col: str,
    strict_clean_features: bool,
    write_legacy_target_bias: bool,
) -> pd.DataFrame:
    clean = final_df.copy()

    missing_features = [c for c in FEATS if c not in clean.columns]
    if missing_features and strict_clean_features:
        raise ValueError(f"Missing XGBoost feature columns: {missing_features}")

    for c in missing_features:
        clean[c] = np.nan

    # Cast features to float32. Missing feature values remain NaN.
    for c in FEATS:
        clean[c] = pd.to_numeric(clean[c], errors="coerce").astype("float32")

    # Build targets independently. Do not drop rows globally just because one target is missing.
    if obs_ta_col in clean.columns and TEMP_FC in clean.columns:
        obs_ta = pd.to_numeric(clean[obs_ta_col], errors="coerce").astype("float32")
        fc_ta = pd.to_numeric(clean[TEMP_FC], errors="coerce").astype("float32")
        clean[TARGET_TA] = (obs_ta - fc_ta).astype("float32")
    else:
        clean[TARGET_TA] = np.float32(np.nan)

    if obs_td_col in clean.columns and DEW_FC in clean.columns:
        obs_td = pd.to_numeric(clean[obs_td_col], errors="coerce").astype("float32")
        fc_td = pd.to_numeric(clean[DEW_FC], errors="coerce").astype("float32")
        clean[TARGET_TD] = (obs_td - fc_td).astype("float32")
    else:
        clean[TARGET_TD] = np.float32(np.nan)

    if "obs_TMAX_aligned" in clean.columns and TMAX_FC in clean.columns:
        obs_tmax = pd.to_numeric(clean["obs_TMAX_aligned"], errors="coerce").astype("float32")
        fc_tmax = pd.to_numeric(clean[TMAX_FC], errors="coerce").astype("float32")
        clean[TARGET_TMAX] = (obs_tmax - fc_tmax).astype("float32")
    else:
        clean[TARGET_TMAX] = np.float32(np.nan)

    if "obs_TMIN_aligned" in clean.columns and TMIN_FC in clean.columns:
        obs_tmin = pd.to_numeric(clean["obs_TMIN_aligned"], errors="coerce").astype("float32")
        fc_tmin = pd.to_numeric(clean[TMIN_FC], errors="coerce").astype("float32")
        clean[TARGET_TMIN] = (obs_tmin - fc_tmin).astype("float32")
    else:
        clean[TARGET_TMIN] = np.float32(np.nan)

    for target in TARGET_COLS:
        clean[target] = pd.to_numeric(clean[target], errors="coerce").astype("float32")
        n_valid = int(clean[target].notna().sum())
        print(f"[INFO] {target}: valid rows={n_valid:,} / {len(clean):,}", flush=True)

    if write_legacy_target_bias:
        clean[TARGET_BIAS_LEGACY] = clean[TARGET_TA]

    # Preserve useful ID/time columns for inference and plotting.
    id_cols = [sid_col, analysis_time_col, valid_time_col]
    id_cols = [c for c in id_cols if c in clean.columns]

    target_cols = TARGET_COLS.copy()
    if write_legacy_target_bias:
        target_cols.append(TARGET_BIAS_LEGACY)

    # Remove helper lookup timestamps from clean output. They are only intermediate.
    out_cols = id_cols + FEATS + target_cols
    out_cols = list(dict.fromkeys(out_cols))

    return clean[out_cols]


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = args.threads
    os.environ["MKL_NUM_THREADS"] = args.threads

    sid_col = args.station_id_col
    analysis_time_col = args.analysis_time_col
    valid_time_col = args.valid_time_col
    leadtime_col = args.leadtime_col
    obs_time_col = args.obs_time_col

    forecast_files = resolve_files(args.forecast_input, args.forecast_pattern)
    output_dir = Path(args.output_dir)
    clean_output_dir = Path(args.clean_xgb_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    stations_path = Path(args.stations)
    if not stations_path.exists():
        raise FileNotFoundError(f"Stations file not found: {stations_path}")

    print(f"[INFO] Forecast files: {len(forecast_files)}", flush=True)
    print(f"[INFO] Observations: {args.observations}", flush=True)
    print(f"[INFO] Stations: {stations_path}", flush=True)
    print(f"[INFO] Output directory: {output_dir}", flush=True)
    print(f"[INFO] Clean output directory: {clean_output_dir}", flush=True)
    print(f"[INFO] Row group size: {args.clean_row_group_size}", flush=True)

    stations = pd.read_csv(stations_path)
    stations = key_formatting(stations, [sid_col])
    if sid_col not in stations.columns:
        raise ValueError(f"Stations file is missing station ID column: {sid_col}")

    for file_idx, forecast_file in enumerate(forecast_files, start=1):
        month = ym_from_filename(forecast_file)
        years = years_needed_for_month(month)

        print("=" * 80, flush=True)
        print(f"[INFO] Processing forecast month {month} ({file_idx}/{len(forecast_files)})", flush=True)
        print(f"[INFO] Reading forecast: {forecast_file.name}", flush=True)

        df = read_table(forecast_file)
        print(f"[INFO] Read forecast file={forecast_file.name}, rows={len(df):,}", flush=True)

        required_forecast_cols = {sid_col, analysis_time_col, valid_time_col, leadtime_col}
        missing_forecast = required_forecast_cols - set(df.columns)
        if missing_forecast:
            raise ValueError(f"Forecast file {forecast_file} missing columns: {sorted(missing_forecast)}")

        df = key_formatting(df, [sid_col])
        df[analysis_time_col] = pd.to_datetime(df[analysis_time_col], errors="coerce")
        df[valid_time_col] = pd.to_datetime(df[valid_time_col], errors="coerce")
        df[leadtime_col] = pd.to_numeric(df[leadtime_col], errors="coerce")

        # Ensure stations join works with string SID.
        df = ensure_station_metadata(df, stations, sid_col)
        df = ensure_analysishour(df, analysis_time_col)

        print(
            f"[INFO] Month {month}: forecast rows={len(df):,}, needed observation years={years}",
            flush=True,
        )

        obs_files = observation_files_for_years(args.observations, years, args.obs_pattern)
        obs_parts = []
        for obs_file in obs_files:
            print(f"[INFO] Reading observations: {obs_file.name}", flush=True)
            obs_parts.append(pd.read_csv(obs_file))

        obs = pd.concat(obs_parts, ignore_index=True) if obs_parts else pd.DataFrame()
        print(f"[INFO] Loaded observation rows={len(obs):,}", flush=True)

        required_obs_base = {sid_col, obs_time_col}
        missing_obs_base = required_obs_base - set(obs.columns)
        if missing_obs_base:
            raise ValueError(f"Observations missing required columns: {sorted(missing_obs_base)}")

        obs = key_formatting(obs, [sid_col])
        obs[obs_time_col] = pd.to_datetime(obs[obs_time_col], errors="coerce")

        # Merge hourly TA/TD observations by SID + validtime.
        df = merge_hourly_observations(
            df=df,
            obs=obs,
            sid_col=sid_col,
            obs_time_col=obs_time_col,
            valid_time_col=valid_time_col,
            obs_ta_col=args.obs_ta_col,
            obs_td_col=args.obs_td_col,
        )

        # Add lookup times and merge daily extrema observations.
        df = add_tmax_tmin_lookup_times(
            df=df,
            valid_time_col=valid_time_col,
            leadtime_col=leadtime_col,
            extreme_min_leadtime=args.extreme_min_leadtime,
        )

        df = merge_extreme_observations(
            df=df,
            obs=obs,
            sid_col=sid_col,
            obs_time_col=obs_time_col,
            obs_tmax_col=args.obs_tmax_col,
            obs_tmin_col=args.obs_tmin_col,
        )

        # Sort by SID, analysistime, leadtime, validtime.
        df = (
            df.sort_values(
                [sid_col, analysis_time_col, leadtime_col, valid_time_col],
                ascending=[True, True, True, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        # Drop exact duplicate full rows only. This does NOT remove distinct leadtimes.
        before_dups = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        after_dups = len(df)
        print(f"[INFO] Exact duplicate rows removed: {before_dups - after_dups:,}", flush=True)

        if not args.clean_only:
            full_path = output_dir / f"{args.output_prefix}_{month}.parquet"
            df.to_parquet(full_path, index=False)
            print(f"[INFO] Wrote combined file {full_path} | rows={len(df):,}", flush=True)

        clean_df = make_clean_multi_target_frame(
            final_df=df,
            sid_col=sid_col,
            analysis_time_col=analysis_time_col,
            valid_time_col=valid_time_col,
            obs_ta_col=args.obs_ta_col,
            obs_td_col=args.obs_td_col,
            strict_clean_features=args.strict_clean_features,
            write_legacy_target_bias=args.write_legacy_target_bias,
        )

        clean_path = clean_output_dir / f"{args.clean_output_prefix}_{month}.parquet"
        clean_df.to_parquet(
            clean_path,
            index=False,
            compression=args.clean_compression,
            row_group_size=args.clean_row_group_size,
        )

        print(
            f"[INFO] Wrote clean multi-target file {clean_path} | "
            f"rows={len(clean_df):,} | cols={len(clean_df.columns)} | "
            f"row_group_size={args.clean_row_group_size}",
            flush=True,
        )

        if sid_col in clean_df.columns:
            print(f"[INFO] Unique stations: {clean_df[sid_col].nunique():,}", flush=True)

        del df, clean_df, obs, obs_parts
        gc.collect()


if __name__ == "__main__":
    main()
