import os

# Restrict CUDA visibility to a single GPU before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Limit CPU thread usage for numerical libraries to avoid oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse
import json
from pathlib import Path
from glob import glob
from datetime import datetime, timedelta
from collections import defaultdict, deque

import numpy as np
import polars as pl
import torch
from torch import nn

# -------------------------
# Fixed columns (must match training constants)
# -------------------------
FEATS = [
    "T2", "SKT", "MX2T", "MN2T", "D2", "T_925", "MSL", "U10", "V10", "T2_M1", "T_925_M1",
    "T2_ENSMEAN_MA1", "LCC", "MCC", "sin_hod", "cos_hod", "sin_doy", "cos_doy",
    "analysishour", "leadtime", "lon", "lat", "elev"
]
LABEL_OBS = "obs_TA"
TEMP_FC = "T2"
SID_COL = "SID"

# -------------------------
# Paths / config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOME = Path.home()
PREP_DIR = HOME / "thesis_project" / "data" / "ml_data_prepared"
PREP_GLOB = PREP_DIR / "ml_data_prep_*.parquet"

MODEL_DIR = HOME / "thesis_project" / "models"

MODEL_TAG = "bias_lstm_stream"
CORR_COL = f"corrected_{MODEL_TAG}"

OUTDIR = HOME / "thesis_project" / "metrics" / MODEL_TAG
OUTDIR.mkdir(parents=True, exist_ok=True)

SPLIT_COLUMN = "analysistime"

# Inference batch size used when sending padded sequences to the model.
PRED_BATCH = 16384

TEST_START = datetime(2024, 9, 10)
TEST_END_INCL = datetime(2025, 8, 31)
TEST_END_EXCL = TEST_END_INCL + timedelta(days=1)

# Optional lookback window for scanning earlier analysis runs.
# Set to 0 when using current-run-only sequence construction.
HISTORY_BUFFER_DAYS = 60


# -------------------------
# Helpers
# -------------------------
def list_files(path_glob: Path):
    """
    Return all files matching a glob pattern, sorted lexicographically.

    Args:
        path_glob: Glob pattern pointing to prepared parquet files.

    Returns:
        List of matching file paths as strings.

    Raises:
        FileNotFoundError: If no files match the pattern.
    """
    files = sorted(glob(str(path_glob)))
    if not files:
        raise FileNotFoundError(f"No files matched: {path_glob}")
    return files


def load_json(path: Path):
    """
    Load and return a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a Python object.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_stats(stats_path: Path):
    """
    Load feature normalization statistics from disk.

    The statistics file is expected to contain `mu` and `sd` mappings
    for each input feature.

    Args:
        stats_path: Path to the normalization statistics JSON file.

    Returns:
        Tuple of:
            - mu: dict mapping feature name to mean
            - sd: dict mapping feature name to standard deviation
    """
    d = load_json(stats_path)
    mu = {k: float(v) for k, v in d["mu"].items()}
    sd = {k: float(v) for k, v in d["sd"].items()}
    return mu, sd


def find_latest_run_dir(model_dir: Path) -> Path:
    """
    Find the most recently modified bias_lstm run directory.

    Args:
        model_dir: Parent directory containing run subdirectories.

    Returns:
        Path to the newest matching run directory.

    Raises:
        FileNotFoundError: If no matching run directories are found.
    """
    cands = [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("bias_lstm_")]
    if not cands:
        raise FileNotFoundError(f"No run dirs found in {model_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_run_artifacts(run_dir: Path):
    """
    Load model configuration, normalization statistics, and checkpoint path.

    This function assumes the run directory contains:
      - final_training_config.json
      - bias_lstm_stream_stats.json
      - bias_lstm_stream.pt

    Args:
        run_dir: Path to a single training run directory.

    Returns:
        Tuple containing:
            cfg, mu, sd, seq_len, hidden, num_layers, dropout, model_path

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "final_training_config.json"
    stats_path = run_dir / "bias_lstm_stream_stats.json"
    model_path = run_dir / "bias_lstm_stream.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing: {stats_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")

    cfg = load_json(cfg_path)
    mu, sd = load_stats(stats_path)

    # Rebuild the model using the same architecture parameters as training.
    seq_len = int(cfg["seq_len"])
    hidden = int(cfg["hidden"])
    num_layers = int(cfg["num_layers"])
    dropout = float(cfg["dropout"])

    return cfg, mu, sd, seq_len, hidden, num_layers, dropout, model_path


# -------------------------
# Model
# -------------------------
class BiasLSTM(nn.Module):
    """
    LSTM-based bias-correction model for station time series.

    The model encodes a padded sequence of past feature vectors and predicts
    a scalar forecast bias from the final valid time step in each sequence.
    """

    def __init__(self, in_dim, hidden=128, num_layers=2, dropout=0.2):
        """
        Initialize the recurrent encoder and regression head.

        Args:
            in_dim: Number of input features per time step.
            hidden: Hidden size of the LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability used between LSTM layers and in the head.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, pad_mask):
        """
        Run the model on a batch of padded sequences.

        Args:
            x: Tensor of shape [batch, seq_len, in_dim].
            pad_mask: Tensor of shape [batch, seq_len], where 1 indicates
                a real time step and 0 indicates padding.

        Returns:
            Tensor of shape [batch] containing predicted bias values.
        """
        out, _ = self.lstm(x)

        # Determine the index of the last non-padded element in each sequence.
        lengths = pad_mask.sum(dim=1).clamp(min=1).long()
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))

        # Gather the final valid hidden state and feed it to the regression head.
        last = out.gather(1, idx).squeeze(1)
        return self.head(last).squeeze(-1)


# -------------------------
# Transform
# -------------------------
def transform_block(df: pl.DataFrame, feats, mu, sd) -> np.ndarray:
    """
    Convert a Polars block into the model input matrix.

    Each original feature is expanded into:
      - a z-scored value
      - a binary missingness indicator

    The final column contains `dt_valid_hours`.

    Args:
        df: Input dataframe containing the required feature columns.
        feats: Ordered list of feature names.
        mu: Per-feature means from training.
        sd: Per-feature standard deviations from training.

    Returns:
        NumPy array of shape [n_rows, 2 * len(feats) + 1] with dtype float32.
    """
    D = 2 * len(feats) + 1
    X = np.empty((df.height, D), dtype=np.float32)
    k = 0

    for f in feats:
        col = df[f].to_numpy()
        miss = np.isnan(col)

        # Store normalized values in one column and missingness flags in the next.
        z = np.zeros_like(col, dtype=np.float32)
        if (~miss).any():
            z[~miss] = ((col[~miss] - mu[f]) / sd[f]).astype(np.float32, copy=False)

        X[:, k] = z
        X[:, k + 1] = miss.astype(np.float32)
        k += 2

    dt = df["dt_valid_hours"].to_numpy()
    dt = np.nan_to_num(dt, nan=0.0).astype(np.float32, copy=False)
    X[:, -1] = dt
    return X


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def predict_period(
    prep_files,
    model,
    mu,
    sd,
    seq_len,
    batch_size,
    plot_vt_start,
    plot_vt_end_excl,
    val_start,
    val_end,
    test_start,
    history_buffer_days=0,
):
    """
    Run streaming inference over prepared parquet files for a valid-time window.

    Sequence history is built separately for each `(SID, analysistime_dt)` pair,
    meaning each forecast run keeps its own time-series context. Predictions are
    only emitted for rows whose valid times fall inside the requested plotting
    window and whose split is not marked as embargo.

    Args:
        prep_files: Iterable of prepared parquet file paths.
        model: Loaded PyTorch model.
        mu: Feature means.
        sd: Feature standard deviations.
        seq_len: Maximum sequence length used by the model.
        batch_size: Number of sequences to score per inference batch.
        plot_vt_start: Inclusive valid-time start for emitted predictions.
        plot_vt_end_excl: Exclusive valid-time end for emitted predictions.
        val_start: Validation period start, based on analysistime.
        val_end: Validation period end, based on analysistime.
        test_start: Test period start, based on analysistime.
        history_buffer_days: Optional lookback in days for scanning earlier runs.

    Returns:
        Polars DataFrame containing raw forecast, observation, corrected forecast,
        and metadata for all emitted rows. Returns an empty DataFrame if no rows
        are produced.
    """
    model.eval()

    # Only scan analysis times that could contribute to the requested valid-time window.
    at_min = plot_vt_start - timedelta(days=history_buffer_days)
    at_max = plot_vt_end_excl

    # Keep a rolling sequence buffer per station and analysis run.
    buf = defaultdict(lambda: deque(maxlen=seq_len))

    out_chunks = []
    Xb, Mb, meta = [], [], []

    def flush():
        """
        Score the currently buffered batch and append results to `out_chunks`.

        Uses mixed precision on CUDA when available.
        """
        nonlocal Xb, Mb, meta, out_chunks
        if not Xb:
            return

        X = torch.from_numpy(np.stack(Xb, axis=0)).to(DEVICE, non_blocking=True)
        M = torch.from_numpy(np.stack(Mb, axis=0)).to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            bias_hat = model(X, M).float().cpu().numpy()

        sid, at, vt, lt, raw_fc, obs, split_set = zip(*meta)
        corrected = np.asarray(raw_fc, dtype=np.float32) + bias_hat.astype(np.float32)

        out_chunks.append(
            pl.DataFrame(
                {
                    "SID": sid,
                    "analysistime_dt": at,
                    "validtime_dt": vt,
                    "analysistime": at,  # Convenience duplicates for downstream plotting scripts.
                    "validtime": vt,
                    "leadtime": lt,
                    "raw_fc": np.asarray(raw_fc, dtype=np.float32),
                    LABEL_OBS: np.asarray(obs, dtype=np.float32),
                    CORR_COL: corrected,
                    "split_set": split_set,
                }
            )
        )

        Xb, Mb, meta = [], [], []

    needed = [SID_COL, "analysistime_dt", "validtime_dt", "dt_valid_hours", *FEATS, LABEL_OBS]

    for fp in prep_files:
        # Scan lazily and keep only rows relevant for the target analysis-time window.
        lf = (
            pl.scan_parquet(fp)
            .select(needed)
            .filter((pl.col("analysistime_dt") >= pl.lit(at_min)) & (pl.col("analysistime_dt") < pl.lit(at_max)))
            .filter(pl.col(TEMP_FC).is_not_null() & pl.col(LABEL_OBS).is_not_null())
        )

        if lf.head(1).collect(engine="streaming").height == 0:
            continue

        df = lf.collect(engine="streaming")
        if df.height == 0:
            continue

        X_block = transform_block(df, FEATS, mu, sd)

        sid = df[SID_COL].to_numpy()
        raw_fc = df[TEMP_FC].to_numpy().astype(np.float32)
        obs = df[LABEL_OBS].to_numpy().astype(np.float32)
        at = df["analysistime_dt"].to_list()
        vt = df["validtime_dt"].to_list()
        lt = df["leadtime"].to_numpy()

        for i in range(df.height):
            key = (int(sid[i]), at[i])
            hist = buf[key]

            # Build each sequence from earlier valid times within the same forecast run.
            if len(hist) >= 1:
                hist_arr = np.stack(hist, axis=0)

                # Left-pad short histories with zeros so all sequences have length `seq_len`.
                if hist_arr.shape[0] < seq_len:
                    pad = seq_len - hist_arr.shape[0]
                    Xseq = np.vstack([np.zeros((pad, X_block.shape[1]), np.float32), hist_arr])
                    Mseq = np.concatenate(
                        [np.zeros(pad, np.float32), np.ones(hist_arr.shape[0], np.float32)]
                    ).astype(np.float32)
                else:
                    Xseq = hist_arr[-seq_len:]
                    Mseq = np.ones(seq_len, np.float32)

                vti = vt[i]

                # Emit predictions only inside the requested valid-time window.
                if (vti >= plot_vt_start) and (vti < plot_vt_end_excl):
                    split_set = split_set_from_at(at[i], val_start, val_end, test_start)

                    # Skip embargo rows in reporting output.
                    if split_set != "embargo":
                        Xb.append(Xseq)
                        Mb.append(Mseq)
                        meta.append(
                            (
                                int(sid[i]),
                                at[i],
                                vti,
                                int(lt[i]),
                                float(raw_fc[i]),
                                float(obs[i]),
                                split_set,
                            )
                        )

                        if len(Xb) >= batch_size:
                            flush()

            # Add the current row after sequence construction so it is only
            # available to future time steps, not to itself.
            hist.append(X_block[i])

    flush()
    if not out_chunks:
        return pl.DataFrame()

    return pl.concat(out_chunks, how="vertical_relaxed")


def to_plot_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert inference output into the column layout expected by plotting scripts.

    Existing `analysistime` and `validtime` columns are dropped before renaming
    datetime source columns to avoid duplication. Output timestamps are formatted
    as strings.

    Args:
        df: Inference output DataFrame.

    Returns:
        DataFrame with plotting-friendly column names and timestamp strings.
    """
    # Drop pre-existing columns so the rename does not create duplicate names.
    drop_cols = [c for c in ["analysistime", "validtime"] if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    df = df.rename({"analysistime_dt": "analysistime", "validtime_dt": "validtime"})

    df = df.with_columns(
        pl.col("analysistime").dt.strftime("%Y-%m-%d %H:%M:%S"),
        pl.col("validtime").dt.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return df


def split_set_from_at(at: datetime, val_start: datetime, val_end: datetime, test_start: datetime):
    """
    Assign a dataset split label from analysis time.

    Args:
        at: Analysis time of the row.
        val_start: Validation period start.
        val_end: Validation period end.
        test_start: Test period start.

    Returns:
        One of: "train", "val", "embargo", or "test".
    """
    if at < val_start:
        return "train"
    if at < val_end:
        return "val"
    if at < test_start:
        return "embargo"
    return "test"


def main():
    """
    Load the latest trained BiasLSTM run, generate corrected forecasts for the
    configured test period, and write yearly parquet outputs for plotting.

    Command-line arguments:
        --run_dir: Optional path to a specific training run directory.
                   If omitted, the most recently modified run is used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to a bias_lstm_YYYYMMDD_HHMMSS run dir. If omitted, uses newest.",
    )
    args = parser.parse_args()

    prep_files = list_files(PREP_GLOB)
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(MODEL_DIR)
    print("Using run_dir:", run_dir)

    cfg, mu, sd, seq_len, hidden, num_layers, dropout, model_path = load_run_artifacts(run_dir)

    val_start_final = datetime.fromisoformat(cfg["val_start_final"])
    val_end_final = datetime.fromisoformat(cfg.get("val_end_final", cfg["test_start"]))
    test_start = datetime.fromisoformat(cfg["test_start"])

    in_dim = 2 * len(FEATS) + 1
    model = BiasLSTM(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    print("Loaded model config from final_training_config.json:")
    print(f"  seq_len={seq_len} hidden={hidden} layers={num_layers} dropout={dropout}")
    print(f"Inference window: {TEST_START} .. {TEST_END_INCL} (inclusive)")

    # Use a valid-time window for output generation so results match other models.
    plot_vt_start = test_start
    plot_vt_end_incl = datetime(2025, 8, 31)
    plot_vt_end_excl = plot_vt_end_incl + timedelta(days=1)

    all_df = predict_period(
        prep_files,
        model,
        mu,
        sd,
        seq_len=seq_len,
        batch_size=PRED_BATCH,
        plot_vt_start=plot_vt_start,
        plot_vt_end_excl=plot_vt_end_excl,
        val_start=val_start_final,
        val_end=val_end_final,
        test_start=test_start,
        history_buffer_days=0,  # Current-run-only inference does not need earlier analysis times.
    )

    if all_df.height == 0:
        print("No rows produced for the requested test period.")
        return

    # Save one parquet file per analysis year for downstream evaluation/plotting.
    years = sorted(all_df.select(pl.col("analysistime_dt").dt.year().unique()).to_series().to_list())
    for year in years:
        df_y = all_df.filter(pl.col("analysistime_dt").dt.year() == year)
        df_y = to_plot_schema(df_y)

        tag = f"{SPLIT_COLUMN}_{MODEL_TAG}_{year}"
        out_path = OUTDIR / f"eval_rows_{tag}_fin.parquet"
        df_y.write_parquet(out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
