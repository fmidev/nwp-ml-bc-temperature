import os
from pathlib import Path
import numpy as np
import pandas as pd
import argparse



# Columns
SPLIT = "validtime"
OBS   = "obs_TA"
RAW   = "raw_fc"
MOS_CORR = "corrected_mos"

ML1_TAG = None
ML1_CORR = None
MLNAME = None

SEASONAL = True


# ------------------------------------------------
# Data loading and preparation functions
# ------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute hit rates by aligning MOS and ML evaluation rows."
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
        help="Directory where hit-rate CSV files will be saved.",
    )

    parser.add_argument(
        "--ml-tag",
        default="tuned_full",
        type=str,
        help="ML model tag used in corrected_<tag> column and input filenames.",
    )

    parser.add_argument(
        "--ml-name",
        default="XGBoost",
        type=str,
        help="Readable model name used in output filenames and labels.",
    )

    parser.add_argument(
        "--seasonal",
        action="store_true",
        help="Compute separate seasonal hit-rate files.",
    )

    parser.add_argument(
        "--non-seasonal",
        action="store_true",
        help="Compute one combined hit-rate file instead of seasonal files.",
    )

    parser.add_argument(
        "--stations-file",
        default=None,
        type=str,
        help=(
            "Optional text file containing station IDs to include, one per line. "
            "If omitted, all stations in the aligned data are used."
        ),
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

    with open(path, "r") as f:
        stations = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not stations:
        raise ValueError(f"No station IDs found in station file: {path}")

    return stations

def _read_parquet_subset(file_path: Path, cols: list[str]) -> pd.DataFrame:
    """Read only columns that exist in the parquet file."""
    preview = pd.read_parquet(file_path)
    existing = [c for c in cols if c in preview.columns]
    if not existing:
        return pd.DataFrame()
    return pd.read_parquet(file_path, columns=existing)


def _normalize_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common dtypes across files."""
    out = df.copy()

    if "SID" in out.columns:
        out["SID"] = out["SID"].astype(str)

    if SPLIT in out.columns:
        out[SPLIT] = pd.to_datetime(out[SPLIT], errors="coerce").dt.tz_localize(None)

    if "analysistime" in out.columns:
        out["analysistime"] = pd.to_datetime(out["analysistime"], errors="coerce").dt.tz_localize(None)

    if "leadtime" in out.columns:
        out["leadtime"] = pd.to_numeric(out["leadtime"], errors="coerce")

    return out


def load_eval_rows_evaldir(eval_dir: Path, pattern: str, cols: list[str]) -> pd.DataFrame:
    """
    Generic loader for MOS / regular validtime-split model files.
    """
    files = sorted(eval_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {eval_dir}/{pattern}")

    dfs = []
    for f in files:
        df = _read_parquet_subset(f, cols)
        df = _normalize_eval_df(df)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    if "SID" in out.columns:
        out["SID"] = out["SID"].astype(str)
    if SPLIT in out.columns:
        out[SPLIT] = pd.to_datetime(out[SPLIT], errors="coerce").dt.tz_localize(None)

    return out


def load_model_eval_rows(model_dir: Path, ml_tag: str) -> pd.DataFrame:
    """
    Load ML eval rows for either:
      1) LSTM-style analysistime-split parquet files with split_set column
      2) regular validtime-split parquet files

    Auto-detects layout by filename pattern.
    """
    model_col = f"corrected_{ml_tag}"

    lstm_pattern = f"eval_rows_analysistime_{ml_tag}_20??_fin.parquet"
    validtime_pattern = f"eval_rows_{SPLIT}_{ml_tag}_20*.parquet"

    lstm_files = sorted(model_dir.glob(lstm_pattern))
    validtime_files = sorted(model_dir.glob(validtime_pattern))

    if lstm_files:
        files = lstm_files
        is_lstm_layout = True
        print(f"[INFO] Detected LSTM analysistime-split layout for {ml_tag}")
    elif validtime_files:
        files = validtime_files
        is_lstm_layout = False
        print(f"[INFO] Detected validtime-split layout for {ml_tag}")
    else:
        raise FileNotFoundError(
            f"No files matched either:\n"
            f"  {model_dir}/{lstm_pattern}\n"
            f"  {model_dir}/{validtime_pattern}"
        )

    dfs = []
    cols = ["SID", "validtime", "analysistime", "leadtime", RAW, OBS, model_col, "split_set"]

    for f in files:
        df = _read_parquet_subset(f, cols)
        df = _normalize_eval_df(df)

        if is_lstm_layout and "split_set" in df.columns:
            df = df[df["split_set"] == "test"].copy()
            df = df.drop(columns=["split_set"], errors="ignore")

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    if "SID" in out.columns:
        out["SID"] = out["SID"].astype(str)
    if SPLIT in out.columns:
        out[SPLIT] = pd.to_datetime(out[SPLIT], errors="coerce").dt.tz_localize(None)

    print(f"[INFO] corrected_{ml_tag} rows loaded: {len(out):,}")
    return out


def mos_coverage_window(mos_eval: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get the time window based on MOS."""
    df = mos_eval.dropna(subset=[MOS_CORR, RAW, OBS]).copy()
    if df.empty:
        raise ValueError("MOS eval rows have no usable values.")
    return df[SPLIT].min(), df[SPLIT].max()


def align_mos_window(mos_eval: pd.DataFrame, ml_eval: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """
    Use MOS to define the window+samples, inner-join ML.
    """
    mos = mos_eval[(mos_eval[SPLIT] >= t0) & (mos_eval[SPLIT] <= t1)].copy()
    keys = ["SID", SPLIT, "analysistime", "leadtime"]
    keep_cols = keys + [RAW, OBS, MOS_CORR, ML1_CORR]

    joined = (
        mos.merge(ml_eval[keys + [ML1_CORR]], on=keys, how="inner")
           .dropna(subset=[OBS, RAW, MOS_CORR, ML1_CORR])[keep_cols]
           .copy()
    )

    if not joined.empty:
        print(f"MOS-driven window (single-ML): {joined[SPLIT].min()} → {joined[SPLIT].max()}")
    else:
        print("Warning: No aligned rows after MOS-driven single-ML join.")
    return joined


def hit_rate(
    df: pd.DataFrame,
    pred_cols: list[str] | tuple[str, ...] = (MOS_CORR, ML1_CORR),
    obs_col: str = OBS,
    by: str | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute hit rates for one or more prediction columns in a single call.

    Returns a dict with:
      - 'overall': rows per model with hits, n, hit_rate
      - 'by' (optional): grouped breakdown with your 'by' columns + model + hits, n, hit_rate
    """
    if df.empty:
        return {"overall": pd.DataFrame(columns=["model", "hits", "n", "hit_rate"])}

    if isinstance(pred_cols, str):
        pred_cols = [pred_cols]

    obs = df[obs_col]
    cold = obs <= 258.15
    cool = (obs > 258.15) & (obs <= 268.15)
    warm = obs > 268.15

    overall_rows = []
    by_rows = []

    if by is not None and not isinstance(by, list):
        by = [by]

    for col in pred_cols:
        if col not in df.columns:
            overall_rows.append({"model": col, "hits": 0, "n": 0, "hit_rate": float("nan")})
            continue

        diff = (obs - df[col]).abs()
        mask = (cold & (diff < 5.0)) | (cool & (diff < 3.5)) | (warm & (diff < 2.5))

        n = int(mask.size)
        hits = int(mask.sum())
        overall_rows.append({
            "model": col,
            "hits": hits,
            "n": n,
            "hit_rate": hits / n if n else float("nan")
        })

        if by:
            g = df.assign(__hit__=mask).groupby(by, observed=True)["__hit__"]
            grp = g.agg(hits=lambda s: int(s.sum()), n=lambda s: int(s.size)).reset_index()
            grp["hit_rate"] = grp["hits"] / grp["n"]
            grp["model"] = col
            by_rows.append(grp)

    overall = (
        pd.DataFrame(overall_rows)
        .sort_values("hit_rate", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    out = {"overall": overall}

    if by_rows:
        by_df = pd.concat(by_rows, ignore_index=True)
        cols = ["model"] + [c for c in by_df.columns if c != "model"]
        out["by"] = by_df[cols]

    return out


def main():
    global ML1_TAG, ML1_CORR, MLNAME

    args = parse_args()

    mos_dir = Path(args.mos_dir)
    ml1_dir = Path(args.ml_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ML1_TAG = args.ml_tag
    ML1_CORR = f"corrected_{ML1_TAG}"
    MLNAME = args.ml_name

    if args.seasonal and args.non_seasonal:
        raise ValueError("Use either --seasonal or --non-seasonal, not both.")

    if args.non_seasonal:
        seasonal = False
    else:
        # Default to seasonal if neither option is given,
        # matching your original SEASONAL = True behavior.
        seasonal = True

    station_subset = load_station_subset(args.stations_file)

    ml1_eval = load_model_eval_rows(ml1_dir, ML1_TAG)

    rename_map = {
        RAW: "Raw forecast",
        MOS_CORR: "MOS",
        ML1_CORR: f"EC_ML_{MLNAME}",
    }

    if seasonal:
        AVAILABLE = {
            "2024": ["autumn"],
            "2025": ["winter", "spring", "summer"],
        }

        for year, seasons in AVAILABLE.items():
            for season in seasons:
                mos_eval = load_eval_rows_evaldir(
                    mos_dir,
                    pattern=f"eval_rows_{SPLIT}_MOS_{year}_{season}.parquet",
                    cols=[
                        "SID",
                        SPLIT,
                        "analysistime",
                        "leadtime",
                        RAW,
                        OBS,
                        MOS_CORR,
                    ],
                )

                t0, t1 = mos_coverage_window(mos_eval)
                print(f"MOS coverage window: {t0} → {t1}")

                joined = align_mos_window(mos_eval, ml1_eval, t0, t1)

                if station_subset is not None:
                    joined = joined[joined["SID"].isin(station_subset)].copy()
                    print(f"Applied station subset from: {args.stations_file}")
                else:
                    print("Using all stations in aligned data.")

                print("Rows after alignment:", len(joined))
                print("Stations after alignment:", joined["SID"].nunique())

                results = hit_rate(
                    joined,
                    pred_cols=[RAW, MOS_CORR, ML1_CORR],
                    by="leadtime",
                )

                results["overall"]["model"] = results["overall"]["model"].replace(rename_map)

                if "by" in results:
                    results["by"]["model"] = results["by"]["model"].replace(rename_map)

                print("Overall hit rates:")
                print(results["overall"].to_string(index=False))

                overall = results["overall"].copy()
                overall["level"] = "overall"

                if "by" in results:
                    by_df = results["by"].copy()
                    by_df["level"] = "by_leadtime"
                    combined = pd.concat(
                        [overall, by_df],
                        ignore_index=True,
                        sort=False,
                    )
                else:
                    combined = overall

                out_path = out_dir / f"hit_rate_{MLNAME}_{season}.csv"
                combined.to_csv(out_path, index=False)
                print(f"\nCombined results saved to: {out_path}")

    else:
        mos_eval = load_eval_rows_evaldir(
            mos_dir,
            pattern=f"eval_rows_{SPLIT}_MOS_202*.parquet",
            cols=[
                "SID",
                SPLIT,
                "analysistime",
                "leadtime",
                RAW,
                OBS,
                MOS_CORR,
            ],
        )

        t0, t1 = mos_coverage_window(mos_eval)
        print(f"MOS coverage window: {t0} → {t1}")

        joined = align_mos_window(mos_eval, ml1_eval, t0, t1)

        if station_subset is not None:
            joined = joined[joined["SID"].isin(station_subset)].copy()
            print(f"Applied station subset from: {args.stations_file}")
        else:
            print("Using all stations in aligned data.")

        print("Rows after alignment:", len(joined))
        print("Stations after alignment:", joined["SID"].nunique())

        results = hit_rate(
            joined,
            pred_cols=[RAW, MOS_CORR, ML1_CORR],
            by="leadtime",
        )

        results["overall"]["model"] = results["overall"]["model"].replace(rename_map)

        if "by" in results:
            results["by"]["model"] = results["by"]["model"].replace(rename_map)

        print("Overall hit rates:")
        print(results["overall"].to_string(index=False))

        overall = results["overall"].copy()
        overall["level"] = "overall"

        if "by" in results:
            by_df = results["by"].copy()
            by_df["level"] = "by_leadtime"
            combined = pd.concat(
                [overall, by_df],
                ignore_index=True,
                sort=False,
            )
        else:
            combined = overall

        out_path = out_dir / f"hit_rate_{MLNAME}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nCombined results saved to: {out_path}")

if __name__ == "__main__":
    main()