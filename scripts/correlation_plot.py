# CORRELATION HEATMAP — keep lon/lat/elev reliably
import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# Features/label
TEMP_FC   = "T2"
LABEL_OBS = "obs_TA"
weather   = ['MSL','T2','D2','U10','V10','LCC','MCC','SKT','MX2T','MN2T','T_925','T2_ENSMEAN_MA1','T2_M1','T_925_M1']
meta      = ["leadtime","lon","lat","elev",'sin_hod','cos_hod','sin_doy','cos_doy',"analysishour"]
cols      = weather + meta + [LABEL_OBS]

N_SAMPLE = 200_000  # adjust for RAM

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a Spearman correlation heatmap from parquet ML data."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help=(
            "Input parquet file, directory, or glob pattern. "
            "Examples: /path/to/ml_data_full/*.parquet or /path/to/ml_data_full"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where the correlation heatmap will be saved.",
    )

    parser.add_argument(
        "--output-name",
        default="correlation_map.svg",
        type=str,
        help="Name of the output figure file. Default: correlation_map.svg",
    )

    parser.add_argument(
        "--sample-size",
        default=200_000,
        type=int,
        help="Number of rows to sample before computing correlations. Default: 200000.",
    )

    return parser.parse_args()

def resolve_input_path(input_arg):
    input_path = Path(input_arg)

    if input_path.is_dir():
        return str(input_path / "*.parquet")

    return str(input_path)


def main():
    args = parse_args()

    parquet_path = resolve_input_path(args.input)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build lazy frame; KEEP columns explicitly, and cast geo/meta numeric
    lf = (
        pl.scan_parquet(parquet_path)
        .select(cols)
        .with_columns(
            pl.col("analysishour").cast(pl.Int16, strict=False),
            pl.col(["lon", "lat", "elev"]).cast(pl.Float32, strict=False),
            (pl.col(LABEL_OBS) - pl.col(TEMP_FC)).alias("bias"),
        )
        # Optional: shrink remaining float columns to Float32 to save RAM
        .with_columns(cs.float().cast(pl.Float32, strict=False))
    )

    # Deterministic unbiased “shuffle” using a row index
    lf_sampled = (
        lf.with_row_index("_i")
          .with_columns(
              _k=((pl.col("_i") * 1103515245 + 12345) % 2_147_483_647)
              .cast(pl.UInt32)
          )
          .sort("_k")
          .head(args.sample_size)
          .drop(["_i", "_k"])
    )

    # Collect bounded table then finish in pandas
    df = lf_sampled.collect(engine="streaming").to_pandas()

    # Make sure lon/lat/elev are numeric in pandas too
    for c in ["lon", "lat", "elev", "analysishour"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute correlation on numeric columns
    corr = df.corr(method="spearman", numeric_only=True)

    # Guarantee these columns appear even if NaN/constant
    must_have = [c for c in ["lon", "lat", "elev"] if c in df.columns]
    corr = corr.reindex(
        index=corr.index.union(must_have),
        columns=corr.columns.union(must_have),
    )

    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Feature/Target Correlation (Spearman)", fontsize=20)
    plt.tight_layout()

    output_path = out_dir / args.output_name
    plt.savefig(output_path, dpi=200)

    print("Saved to:", output_path)
    print("Columns present:", sorted(df.columns.tolist()))

    if set(["lon", "lat", "elev"]).issubset(df.columns):
        print("dtypes (subset):", df[["lon", "lat", "elev"]].dtypes)
    else:
        print("geo columns missing")

    if must_have:
        print(corr.loc[must_have, ["bias"]].sort_values(by="bias", ascending=False))
    else:
        print("No geo in corr")

if __name__ == "__main__":
    main()
