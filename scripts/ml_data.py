"""
Combine forecast CSV files, yearly observation CSV files, and station metadata,
then apply ML preprocessing and write one parquet file per forecast month.

Forecast input:
    CSV files named like:
        forecast_202001.csv
        forecast_202002.csv

Observation input:
    A directory, single file, or glob of yearly CSV files.
    Filenames should contain a year, for example:
        observations_2020.csv
        observations_2021.csv

Station input:
    One CSV file.

Output:
    One parquet file per forecast month, for example:
        ml_data_2020-01.parquet

Example:
    python make_ml_data_from_csv.py \
      --forecast-input "$HOME/thesis_project/data/forecast_csv" \
      --observations "$HOME/thesis_project/data/observations_yearly" \
      --stations "$HOME/thesis_project/data/stations_with_tags.csv" \
      --output-dir "$HOME/thesis_project/data/ml_data"
"""

import argparse
import gc
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine monthly forecast CSV files with yearly observation CSV files "
            "and station metadata, apply ML preprocessing, and output one parquet "
            "file per forecast month."
        )
    )

    parser.add_argument(
        "--forecast-input",
        required=True,
        type=str,
        help=(
            "Forecast CSV input directory, single CSV file, or glob pattern. "
            "Examples: /path/to/forecast_csv or /path/to/forecast_csv/*.csv"
        ),
    )

    parser.add_argument(
        "--observations",
        required=True,
        type=str,
        help=(
            "Observation CSV input directory, single CSV file, or glob pattern. "
            "Expected yearly files with years in filenames, for example observations_2020.csv."
        ),
    )

    parser.add_argument(
        "--stations",
        required=True,
        type=str,
        help="Path to station metadata CSV file, for example stations_with_tags.csv.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where monthly ML parquet files will be saved.",
    )

    parser.add_argument(
        "--forecast-pattern",
        default="*.csv",
        type=str,
        help="Filename pattern used when --forecast-input is a directory. Default: *.csv",
    )

    parser.add_argument(
        "--observation-pattern",
        default="*.csv",
        type=str,
        help="Filename pattern used when --observations is a directory. Default: *.csv",
    )

    parser.add_argument(
        "--threads",
        default="16",
        type=str,
        help="Thread count for OMP_NUM_THREADS and MKL_NUM_THREADS. Default: 16.",
    )

    parser.add_argument(
        "--obs-time-col",
        default="obstime",
        type=str,
        help="Observation timestamp column in observations CSV. Default: obstime.",
    )

    parser.add_argument(
        "--obs-value-col",
        default="obs_TA",
        type=str,
        help="Observation value column in observations CSV. Default: obs_TA.",
    )

    parser.add_argument(
        "--station-id-col",
        default="SID",
        type=str,
        help="Station ID column name. Default: SID.",
    )

    parser.add_argument(
        "--analysis-time-col",
        default="analysistime",
        type=str,
        help="Forecast analysis time column. Default: analysistime.",
    )

    parser.add_argument(
        "--valid-time-col",
        default="validtime",
        type=str,
        help="Forecast valid time column. Default: validtime.",
    )

    parser.add_argument(
        "--leadtime-col",
        default="leadtime",
        type=str,
        help="Forecast leadtime column. Default: leadtime.",
    )

    parser.add_argument(
        "--output-prefix",
        default="ml_data",
        type=str,
        help="Prefix for output parquet files. Default: ml_data.",
    )

    parser.add_argument(
        "--ensmean-col",
        default="T2_ENSMEAN_MA1",
        type=str,
        help="Column where values <= 100 are replaced with NaN. Default: T2_ENSMEAN_MA1.",
    )

    return parser.parse_args()


def resolve_csv_files(input_path: str, pattern: str = "*.csv") -> list[Path]:
    """
    Resolve input into a sorted list of CSV files.

    The input can be:
      - a directory
      - one CSV file
      - a glob pattern
    """
    p = Path(input_path)

    if p.is_dir():
        files = sorted(p.glob(pattern))
    elif p.is_file():
        files = [p]
    else:
        files = [Path(x) for x in sorted(glob(input_path))]

    if not files:
        raise FileNotFoundError(f"No CSV files found from: {input_path}")

    return files


def ym_from_forecast_filename(pth: Path) -> str:
    """
    Parse year and month from forecast filename.

    Expected forecast filename convention:
        name_YYYYMM.csv

    Examples:
        forecast_202001.csv -> 2020-01
        forecast_202012.csv -> 2020-12
        my_forecast_202103.csv -> 2021-03

    Returns:
        YYYY-MM
    """
    m = re.search(r"_(\d{4})(\d{2})\.csv$", pth.name)

    if not m:
        raise ValueError(
            f"Cannot parse month from forecast file: {pth.name}. "
            "Expected filename like forecast_202001.csv or name_YYYYMM.csv"
        )

    year = m.group(1)
    month = m.group(2)

    month_int = int(month)

    if month_int < 1 or month_int > 12:
        raise ValueError(f"Invalid month in forecast filename {pth.name}: {month}")

    return f"{year}-{month}"


def year_from_observation_filename(pth: Path) -> str:
    """
    Parse year from observation filename.

    Accepts any filename containing a 4-digit year.

    Examples:
        observations_2020.csv -> 2020
        obs_2021.csv -> 2021
        2022_observations.csv -> 2022
    """
    m = re.search(r"(19\d{2}|20\d{2})", pth.name)

    if not m:
        raise ValueError(
            f"Cannot parse year from observation file: {pth.name}. "
            "Expected filename to contain a year, for example observations_2020.csv"
        )

    return m.group(1)


def key_formatting(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Format join key columns as stripped string columns.
    """
    cols = [c for c in keys if c in df.columns]

    if not cols:
        return df

    df = df.copy()

    for c in cols:
        df[c] = df[c].astype("string").str.strip()

    return df


def load_observations_for_years(
    obs_files_by_year: dict[str, Path],
    years: set[str],
    sid_col: str,
    obs_time_col: str,
    valid_time_col: str,
    obs_value_col: str,
) -> pd.DataFrame:
    """
    Load only observation files for the needed years.
    """
    parts = []

    for year in sorted(years):
        obs_file = obs_files_by_year.get(year)

        if obs_file is None:
            print(f"[WARN] No observation file found for year {year}")
            continue

        print(f"[INFO] Reading observations for {year}: {obs_file.name}")

        df_obs = pd.read_csv(
            obs_file,
            dtype={
                sid_col: "string",
                obs_time_col: "string",
            },
            usecols=lambda c: c in {sid_col, obs_time_col, obs_value_col},
        )

        required_cols = {sid_col, obs_time_col, obs_value_col}
        missing_cols = required_cols - set(df_obs.columns)

        if missing_cols:
            raise ValueError(
                f"Observation file {obs_file} is missing columns: "
                f"{sorted(missing_cols)}"
            )

        df_obs = key_formatting(df_obs, [sid_col, obs_time_col])

        df_obs = df_obs.rename(columns={obs_time_col: valid_time_col})
        df_obs = df_obs[[sid_col, valid_time_col, obs_value_col]]

        parts.append(df_obs)

    if not parts:
        return pd.DataFrame(columns=[sid_col, valid_time_col, obs_value_col])

    return pd.concat(parts, ignore_index=True)


def add_time_features(
    df: pd.DataFrame,
    valid_time_col: str,
) -> pd.DataFrame:
    """
    Add cyclic hour-of-day and day-of-year features.

    Creates:
        sin_hod
        cos_hod
        sin_doy
        cos_doy
    """
    df = df.copy()

    df["ts"] = pd.to_datetime(df[valid_time_col], errors="coerce")

    # Hour-of-day cyclic features
    h = (
        df["ts"].dt.hour
        + df["ts"].dt.minute / 60
        + df["ts"].dt.second / 3600
    )

    theta_h = 2 * np.pi * (h / 24.0)

    df["sin_hod"] = np.sin(theta_h)
    df["cos_hod"] = np.cos(theta_h)

    # Day-of-year cyclic features, leap-year aware
    doy = df["ts"].dt.dayofyear.astype("Float64")

    sec = (
        df["ts"].dt.hour * 3600
        + df["ts"].dt.minute * 60
        + df["ts"].dt.second
    ).astype("Float64")

    frac_doy = (doy - 1) + (sec / 86400.0)

    year_len = (
        df["ts"]
        .dt.is_leap_year
        .map({True: 366.0, False: 365.0})
        .astype("Float64")
    )

    theta_y = 2 * np.pi * (frac_doy / year_len)

    df["sin_doy"] = np.sin(theta_y)
    df["cos_doy"] = np.cos(theta_y)

    # Remove helper timestamp column
    df = df.drop(columns=["ts"])

    return df


def apply_ml_preprocessing(
    df: pd.DataFrame,
    valid_time_col: str,
    ensmean_col: str,
) -> pd.DataFrame:
    """
    Apply ML preprocessing:
      - drop radiation columns if present
      - set ENSMEAN values <= 100 to NaN
      - add cyclic time features
      - remove closed and late-opened stations
      - keep only hourly stations tag == '1h'
      - remove duplicate rows
    """
    df = df.copy()

    # Drop radiation parameters if present
    drop_cols = [c for c in ["SSR_Acc", "STR_Acc"] if c in df.columns]

    if drop_cols:
        df = df.drop(columns=drop_cols)

    # ENSMEAN to NaN when <= 100
    if ensmean_col in df.columns:
        df[ensmean_col] = pd.to_numeric(df[ensmean_col], errors="coerce")
        df[ensmean_col] = df[ensmean_col].where(df[ensmean_col] > 100)
    else:
        print(f"[WARN] ENSMEAN column not found: {ensmean_col}")

    # Add time features
    df = add_time_features(df, valid_time_col=valid_time_col)

    """
    # Remove closed stations
    if "closed" in df.columns:
       df = df[~df["closed"].fillna(False)]
    else:
        print("[WARN] Column not found: closed")

    # Remove stations that opened late
    if "opened_late" in df.columns:
        df = df[~df["opened_late"].fillna(False)]
    else:
        print("[WARN] Column not found: opened_late")

    # Keep only stations that send observations hourly
    if "tag" in df.columns:
        df = df[df["tag"].astype("string") == "1h"]
    else:
        print("[WARN] Column not found: tag")"""

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    print(f"[INFO] Duplicate rows removed: {removed:,}")

    return df


def main():
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = args.threads
    os.environ["MKL_NUM_THREADS"] = args.threads

    sid_col = args.station_id_col
    obs_time_col = args.obs_time_col
    obs_value_col = args.obs_value_col
    analysis_time_col = args.analysis_time_col
    valid_time_col = args.valid_time_col
    leadtime_col = args.leadtime_col

    forecast_files = resolve_csv_files(
        args.forecast_input,
        args.forecast_pattern,
    )

    observation_files = resolve_csv_files(
        args.observations,
        args.observation_pattern,
    )

    stations_path = Path(args.stations)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not stations_path.exists():
        raise FileNotFoundError(f"Stations file not found: {stations_path}")

    print(f"[INFO] Forecast files: {len(forecast_files)}")
    print(f"[INFO] Observation files: {len(observation_files)}")
    print(f"[INFO] Stations: {stations_path}")
    print(f"[INFO] Output directory: {output_dir}")

    # Read station metadata once
    df_stations = pd.read_csv(
        stations_path,
        dtype={
            sid_col: "string",
        },
    )

    if sid_col not in df_stations.columns:
        raise ValueError(f"Stations file is missing station ID column: {sid_col}")
    
    # Drop unneeded station metadata columns before merging
    station_columns_to_drop = [
        "name",
        "producer",
    ]

    drop_station_cols = [
        c for c in station_columns_to_drop
        if c in df_stations.columns and c != sid_col
    ]

    if drop_station_cols:
        df_stations = df_stations.drop(columns=drop_station_cols)
        print(f"[INFO] Dropped station columns: {drop_station_cols}")

    df_stations = key_formatting(df_stations, [sid_col])

    # Map observation year -> observation CSV file
    obs_files_by_year = {}

    for obs_file in observation_files:
        year = year_from_observation_filename(obs_file)

        if year in obs_files_by_year:
            raise ValueError(
                f"Multiple observation files found for year {year}: "
                f"{obs_files_by_year[year]} and {obs_file}"
            )

        obs_files_by_year[year] = obs_file

    print(f"[INFO] Observation years available: {sorted(obs_files_by_year)}")

    # Group forecast CSV files by forecast month
    forecast_files_by_month = defaultdict(list)

    for forecast_file in forecast_files:
        month_key = ym_from_forecast_filename(forecast_file)
        forecast_files_by_month[month_key].append(forecast_file)

    print(f"[INFO] Forecast months found: {len(forecast_files_by_month)}")

    # Process one forecast month at a time
    for month_key, month_files in sorted(forecast_files_by_month.items()):
        print("=" * 80)
        print(
            f"[INFO] Processing forecast month {month_key} "
            f"with {len(month_files)} forecast file(s)"
        )

        forecast_parts = []
        needed_obs_years = set()

        for forecast_file in month_files:
            print(f"[INFO] Reading forecast: {forecast_file.name}")

            df_forecast = pd.read_csv(
                forecast_file,
                dtype={
                    sid_col: "string",
                    analysis_time_col: "string",
                    valid_time_col: "string",
                },
            )

            required_forecast_cols = {
                sid_col,
                analysis_time_col,
                valid_time_col,
                leadtime_col,
            }

            missing_forecast_cols = required_forecast_cols - set(df_forecast.columns)

            if missing_forecast_cols:
                raise ValueError(
                    f"Forecast file {forecast_file} is missing columns: "
                    f"{sorted(missing_forecast_cols)}"
                )

            df_forecast = key_formatting(
                df_forecast,
                [sid_col, analysis_time_col, valid_time_col],
            )

            # Track observation years needed from forecast validtime.
            # This matters if a forecast month crosses into the next year.
            years = (
                df_forecast[valid_time_col]
                .str.slice(0, 4)
                .dropna()
                .unique()
                .tolist()
            )

            needed_obs_years.update(years)

            forecast_parts.append(df_forecast)

            print(
                f"[INFO] Read forecast file={forecast_file.name}, "
                f"rows={len(df_forecast):,}"
            )

        # Combine forecasts for this month only
        df_month = pd.concat(forecast_parts, ignore_index=True)

        del forecast_parts
        gc.collect()

        print(
            f"[INFO] Month {month_key}: forecast rows={len(df_month):,}, "
            f"needed observation years={sorted(needed_obs_years)}"
        )

        # Merge station metadata
        df_month = df_month.merge(
            df_stations,
            on=sid_col,
            how="left",
            validate="m:1",
        )

        # Load only needed yearly observation files
        df_obs = load_observations_for_years(
            obs_files_by_year=obs_files_by_year,
            years=needed_obs_years,
            sid_col=sid_col,
            obs_time_col=obs_time_col,
            valid_time_col=valid_time_col,
            obs_value_col=obs_value_col,
        )

        print(f"[INFO] Loaded observation rows={len(df_obs):,}")

        # Merge observations onto forecasts
        final_df = df_month.merge(
            df_obs,
            on=[sid_col, valid_time_col],
            how="left",
            validate="m:1",
        )

        
        del df_month
        del df_obs
        gc.collect()

        # Apply ML preprocessing
        final_df = apply_ml_preprocessing(
            final_df,
            valid_time_col=valid_time_col,
            ensmean_col=args.ensmean_col,
        )

        # Sort this forecast month
        final_df["_lt"] = pd.to_numeric(final_df[leadtime_col], errors="coerce")

        final_df = (
            final_df.sort_values(
                [sid_col, analysis_time_col, "_lt", valid_time_col],
                ascending=[True, True, True, True],
                kind="stable",
            )
            .drop(columns="_lt")
            .reset_index(drop=True)
        )

        # Derive analysishour from analysistime if the column already exists
        if "analysishour" in final_df.columns:
            final_df["analysishour"] = final_df[analysis_time_col].str[11:13]

        out_path = output_dir / f"{args.output_prefix}_{month_key}.parquet"

        final_df.to_parquet(out_path, index=False)

        print(f"[INFO] Unique stations: {final_df[sid_col].nunique():,}")
        print(f"[INFO] DataFrame memory usage, bytes: {final_df.memory_usage(deep=True).sum():,}")
        print(f"[INFO] Wrote {out_path} | rows={len(final_df):,}")

        del final_df
        gc.collect()

    print("=" * 80)
    print("[INFO] Done")


if __name__ == "__main__":
    main()