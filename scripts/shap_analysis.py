# SHAP analysis for the XGBoost model

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import shap
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


# ------------------------------------------------
# Default features
# ------------------------------------------------
# If you provide --features-path, this list is replaced by the JSON feature list.
# The feature order must match the order used during training.

weather = [
    "MSL", "T2", "D2", "U10", "V10", "LCC", "MCC", "SKT",
    "MX2T", "MN2T", "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1",
]

meta = [
    "leadtime", "lon", "lat", "elev",
    "sin_hod", "cos_hod", "sin_doy", "cos_doy", "analysishour",
]

FEATS = weather + meta


# ------------------------------------------------
# Argument parsing
# ------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis for a trained XGBoost model."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help=(
            "Input parquet file, directory, or glob pattern. "
            "Examples: /path/to/ml_data_full/ml_data_full_*.parquet "
            "or /path/to/ml_data_full"
        ),
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to the trained XGBoost model JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where SHAP figures will be saved.",
    )

    parser.add_argument(
        "--n-explain",
        default=30000,
        type=int,
        help="Number of samples to explain with SHAP. Default: 30000.",
    )

    parser.add_argument(
        "--holdout-days",
        default=365,
        type=int,
        help=(
            "Use only the last N days based on analysistime. "
            "Use 0 to disable holdout filtering. Default: 365."
        ),
    )

    parser.add_argument(
        "--features-path",
        default=None,
        type=str,
        help=(
            "Optional JSON file containing the exact feature list used in training. "
            "If omitted, the feature list defined in this script is used."
        ),
    )

    parser.add_argument(
        "--random-seed",
        default=0,
        type=int,
        help="Random seed for SHAP sampling. Default: 0.",
    )

    return parser.parse_args()


def resolve_input_path(input_arg: str) -> str:
    """
    Accept either:
      - a parquet file
      - a directory containing parquet files
      - a glob pattern
    """
    input_path = Path(input_arg)

    if input_path.is_dir():
        return str(input_path / "*.parquet")

    return str(input_path)


def load_features(features_path: str | None) -> list[str]:
    """
    Load feature names from a JSON file if provided.
    Otherwise use the default FEATS list.
    """
    if features_path is None:
        return FEATS

    features_path = Path(features_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    with open(features_path, "r") as f:
        feats = json.load(f)

    if not isinstance(feats, list) or not all(isinstance(c, str) for c in feats):
        raise ValueError(
            "Features JSON must contain a list of strings, for example: "
            '["MSL", "T2", "D2", ...]'
        )

    return feats


# ------------------------------------------------
# Main SHAP workflow
# ------------------------------------------------

def main():
    args = parse_args()

    data_path = resolve_input_path(args.input)
    model_path = Path(args.model_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = load_features(args.features_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.n_explain <= 0:
        raise ValueError("--n-explain must be greater than 0.")

    if args.holdout_days < 0:
        raise ValueError("--holdout-days must be 0 or greater.")

    print(f"Input data: {data_path}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {out_dir}")
    print(f"Number of features: {len(feats)}")
    print(f"N_EXPLAIN: {args.n_explain}")
    print(f"HOLDOUT_DAYS: {args.holdout_days}")

    # ----------------------------
    # Load model
    # ----------------------------
    model = XGBRegressor()
    model.load_model(model_path)

    # ----------------------------
    # Load data for SHAP
    # ----------------------------
    needed_cols = feats + ["analysistime"]

    lf = pl.scan_parquet(data_path).select(needed_cols)

    lf = lf.with_columns(
        pl.col("analysistime")
        .str.strptime(pl.Datetime, strict=False)
        .alias("analysistime_dt")
    )

    if "analysishour" in feats:
        lf = lf.with_columns(
            pl.col("analysishour").cast(pl.Int8, strict=False)
        )

    df = lf.collect(engine="streaming")

    if df.height == 0:
        raise ValueError("No rows found in input data.")

    # Optional: restrict to last N days based on analysistime
    if args.holdout_days > 0:
        import datetime as dt

        max_time = df["analysistime_dt"].max()

        if max_time is None:
            raise ValueError(
                "Could not determine max analysistime_dt. "
                "Check that analysistime can be parsed as datetime."
            )

        cutoff = max_time - dt.timedelta(days=args.holdout_days)
        df = df.filter(pl.col("analysistime_dt") >= cutoff)

        print(f"Using SHAP data from analysistime >= {cutoff}")

    if df.height == 0:
        raise ValueError(
            "No rows found for SHAP slice after holdout filtering. "
            "Check --holdout-days or the input data."
        )

    # Drop helper column and keep only model features
    dfX = df.select(feats)

    # Convert to numpy
    X_all = dfX.to_numpy().astype(np.float32, copy=False)

    n = X_all.shape[0]
    if n == 0:
        raise ValueError("No rows available for SHAP after feature selection.")

    # ----------------------------
    # Sample rows for SHAP
    # ----------------------------
    rng = np.random.default_rng(args.random_seed)
    sample_size = min(n, args.n_explain)
    idx = rng.choice(n, size=sample_size, replace=False)
    X_sample = X_all[idx]

    print(f"Rows available for SHAP: {n:,}")
    print(f"Rows sampled for SHAP: {sample_size:,}")

    # ----------------------------
    # Build SHAP explainer
    # ----------------------------
    explainer = shap.TreeExplainer(model)

    try:
        sv = explainer(X_sample)
        shap_values = sv.values
        base_value = np.mean(sv.base_values)
    except Exception:
        shap_values = explainer.shap_values(X_sample)
        base_value = (
            explainer.expected_value
            if np.isscalar(explainer.expected_value)
            else np.mean(explainer.expected_value)
        )

    print("SHAP matrix shape:", shap_values.shape)
    print("Base value:", base_value)

    max_display = len(feats)

    # ----------------------------
    # 1) Beeswarm summary plot
    # ----------------------------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feats,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()

    beeswarm_path = out_dir / "beeswarm.png"
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {beeswarm_path}")

    # ----------------------------
    # 2) Bar plot: mean absolute SHAP
    # ----------------------------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feats,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()

    bar_path = out_dir / "shap_bar.png"
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {bar_path}")

    # ----------------------------
    # 3) Dependence plots
    # ----------------------------
    for feat in ["leadtime", "T2"]:
        if feat in feats:
            shap.dependence_plot(
                feat,
                shap_values,
                X_sample,
                feature_names=feats,
                interaction_index=None,
                show=False,
            )
            plt.tight_layout()

            dep_path = out_dir / f"dependence_{feat}.png"
            plt.savefig(dep_path, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"Saved: {dep_path}")
        else:
            print(f"Skipped dependence plot for {feat}: feature not found.")


if __name__ == "__main__":
    main()
