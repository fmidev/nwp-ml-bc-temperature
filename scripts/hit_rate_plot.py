import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot hit-rate comparison by leadtime from a combined hit-rate CSV."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        type=str,
        help="Path to the combined hit-rate CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where the plot will be saved.",
    )

    parser.add_argument(
        "--output-name",
        default="hit_rate_by_leadtime_plot.png",
        type=str,
        help="Output plot filename. Default: hit_rate_by_leadtime_plot.png",
    )

    parser.add_argument(
        "--mos-name",
        default="MOS",
        type=str,
        help="Model name used for MOS in the CSV. Default: MOS.",
    )

    parser.add_argument(
        "--ml-name",
        default="EC_ML",
        type=str,
        help="Model name used for ML in the CSV. Default: EC_ML.",
    )

    parser.add_argument(
        "--title",
        default="Hit Rate Comparison by Leadtime",
        type=str,
        help="Plot title.",
    )

    parser.add_argument(
        "--dpi",
        default=200,
        type=int,
        help="Figure DPI. Default: 200.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"level", "leadtime", "model", "hit_rate"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    # Filter only the by-leadtime rows
    df_by = df[df["level"] == "by_leadtime"].copy()

    if df_by.empty:
        raise ValueError("No rows found where level == 'by_leadtime'.")

    # Ensure leadtime is numeric and sorted
    df_by["leadtime"] = pd.to_numeric(df_by["leadtime"], errors="coerce")
    df_by = df_by.dropna(subset=["leadtime"]).sort_values("leadtime")

    if df_by.empty:
        raise ValueError("No valid numeric leadtime values found.")

    # Pivot to wide format: one column per model
    wide = df_by.pivot(index="leadtime", columns="model", values="hit_rate")

    missing_models = [
        model_name
        for model_name in [args.mos_name, args.ml_name]
        if model_name not in wide.columns
    ]

    if missing_models:
        raise ValueError(
            f"Requested model column(s) not found in CSV: {missing_models}. "
            f"Available models are: {list(wide.columns)}"
        )

    # Plot colors
    colors = {
        "DIFF": "#999999",
        "MOS": "#637AB9",
        "ML": "#B95E82",
    }

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left y-axis: hit rates
    ax1.plot(
        wide.index,
        wide[args.mos_name],
        "o-",
        label=args.mos_name,
        color=colors["MOS"],
    )

    ax1.plot(
        wide.index,
        wide[args.ml_name],
        "o-",
        label=args.ml_name,
        color=colors["ML"],
    )

    ax1.set_xlabel("Leadtime (h)")
    ax1.set_ylabel("Hit Rate")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Right y-axis: difference
    ax2 = ax1.twinx()
    diff = wide[args.ml_name] - wide[args.mos_name]

    ax2.plot(
        wide.index,
        diff,
        "-",
        color=colors["DIFF"],
        label=f"Δ ({args.ml_name} - {args.mos_name})",
    )

    ax2.set_ylabel("Hit Rate Difference")
    ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax2.legend(loc="center right")

    plt.title(args.title)
    plt.tight_layout()

    output_path = out_dir / args.output_name
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
