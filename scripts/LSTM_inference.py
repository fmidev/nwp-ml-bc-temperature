import os
import argparse


def parse_pre_args():
    """
    Parse CUDA_VISIBLE_DEVICES and thread settings before importing torch.

    CUDA_VISIBLE_DEVICES should be set before torch is imported.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        type=str,
        help="CUDA_VISIBLE_DEVICES value, for example '0', '1', or empty string.",
    )

    parser.add_argument(
        "--threads",
        default="4",
        type=str,
        help="Thread count for OMP_NUM_THREADS, MKL_NUM_THREADS, and POLARS_MAX_THREADS.",
    )

    args, _ = parser.parse_known_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    os.environ.setdefault("OMP_NUM_THREADS", args.threads)
    os.environ.setdefault("MKL_NUM_THREADS", args.threads)
    os.environ.setdefault("POLARS_MAX_THREADS", args.threads)


parse_pre_args()

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
# Fixed columns
# Must match training constants
# -------------------------

FEATS = [
    "T2", "SKT", "MX2T", "MN2T", "D2", "T_925", "MSL", "U10", "V10",
    "T2_M1", "T_925_M1", "T2_ENSMEAN_MA1",
    "LCC", "MCC",
    "sin_hod", "cos_hod", "sin_doy", "cos_doy",
    "analysishour", "leadtime", "lon", "lat", "elev",
]

LABEL_OBS = "obs_TA"
TEMP_FC = "T2"
SID_COL = "SID"


# -------------------------
# Runtime globals
# Set from command-line args in main()
# -------------------------

DEVICE = None

MODEL_TAG = None
CORR_COL = None

SPLIT_COLUMN = "analysistime"


# -------------------------
# Arguments
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BiasLSTM inference and write yearly evaluation parquet files."
    )

    parser.add_argument(
        "--prep-input",
        required=True,
        type=str,
        help=(
            "Prepared parquet input path, directory, or glob pattern. "
            "Examples: /path/to/ml_data_prepared/ml_data_prep_*.parquet "
            "or /path/to/ml_data_prepared"
        ),
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help=(
            "Parent directory containing bias_lstm_* run directories. "
            "Used only when --run-dir is not provided."
        ),
    )

    parser.add_argument(
        "--run-dir",
        default=None,
        type=str,
        help=(
            "Specific BiasLSTM training run directory. "
            "If omitted, the newest bias_lstm_* directory inside --model-dir is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where evaluation parquet files will be saved.",
    )

    parser.add_argument(
        "--model-tag",
        default="bias_lstm_stream",
        type=str,
        help="Model tag used in corrected_<tag> column and output filenames.",
    )

    parser.add_argument(
        "--test-end",
        default="2025-08-31",
        type=str,
        help=(
            "Inclusive validtime end date for output generation. "
            "Default: 2025-08-31."
        ),
    )

    parser.add_argument(
        "--history-buffer-days",
        default=0,
        type=int,
        help=(
            "Optional lookback days for scanning earlier analysis runs. "
            "Default: 0 for current-run-only sequence construction."
        ),
    )

    parser.add_argument(
        "--pred-batch",
        default=16384,
        type=int,
        help="Inference batch size. Default: 16384.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. Default: auto.",
    )

    parser.add_argument(
        "--threads",
        default="4",
        type=str,
        help="Thread count. Parsed early before torch import.",
    )

    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        type=str,
        help="CUDA_VISIBLE_DEVICES value. Parsed early before torch import.",
    )

    return parser.parse_args()


def resolve_prep_glob(prep_input: str) -> str:
    """
    Accept a prepared parquet file, directory, or glob pattern.
    """
    p = Path(prep_input)

    if p.is_dir():
        return str(p / "ml_data_prep_*.parquet")

    return str(p)


# -------------------------
# Helpers
# -------------------------

def list_files(path_glob: str | Path):
    """
    Return all files matching a glob pattern, sorted lexicographically.
    """
    files = sorted(glob(str(path_glob)))

    if not files:
        raise FileNotFoundError(f"No files matched: {path_glob}")

    return files


def load_json(path: Path):
    """
    Load and return a JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_stats(stats_path: Path):
    """
    Load feature normalization statistics from disk.
    """
    d = load_json(stats_path)

    mu = {k: float(v) for k, v in d["mu"].items()}
    sd = {k: float(v) for k, v in d["sd"].items()}

    return mu, sd


def find_latest_run_dir(model_dir: Path) -> Path:
    """
    Find the most recently modified bias_lstm run directory.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    cands = [
        p
        for p in model_dir.iterdir()
        if p.is_dir() and p.name.startswith("bias_lstm_")
    ]

    if not cands:
        raise FileNotFoundError(f"No run dirs found in {model_dir}")

    return max(cands, key=lambda p: p.stat().st_mtime)


def load_run_artifacts(run_dir: Path, model_tag: str):
    """
    Load model configuration, normalization statistics, and checkpoint path.

    Expects:
      - final_training_config.json
      - <model_tag>_stats.json
      - <model_tag>.pt
    """
    run_dir = Path(run_dir)

    cfg_path = run_dir / "final_training_config.json"
    stats_path = run_dir / f"{model_tag}_stats.json"
    model_path = run_dir / f"{model_tag}.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing: {stats_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")

    cfg = load_json(cfg_path)
    mu, sd = load_stats(stats_path)

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
    """

    def __init__(self, in_dim, hidden=128, num_layers=2, dropout=0.2):
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
        out, _ = self.lstm(x)

        lengths = pad_mask.sum(dim=1).clamp(min=1).long()
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))

        last = out.gather(1, idx).squeeze(1)

        return self.head(last).squeeze(-1)


# -------------------------
# Transform
# -------------------------

def transform_block(df: pl.DataFrame, feats, mu, sd) -> np.ndarray:
    """
    Convert a Polars block into the model input matrix.

    Each feature becomes:
      - z-scored value
      - missingness indicator

    Final column is dt_valid_hours.
    """
    d = 2 * len(feats) + 1
    x = np.empty((df.height, d), dtype=np.float32)

    k = 0

    for f in feats:
        col = df[f].to_numpy()
        miss = np.isnan(col)

        z = np.zeros_like(col, dtype=np.float32)

        if (~miss).any():
            z[~miss] = ((col[~miss] - mu[f]) / sd[f]).astype(np.float32, copy=False)

        x[:, k] = z
        x[:, k + 1] = miss.astype(np.float32)

        k += 2

    dt = df["dt_valid_hours"].to_numpy()
    dt = np.nan_to_num(dt, nan=0.0).astype(np.float32, copy=False)

    x[:, -1] = dt

    return x


# -------------------------
# Split helpers
# -------------------------

def split_set_from_at(
    at: datetime,
    val_start: datetime,
    val_end: datetime,
    test_start: datetime,
):
    """
    Assign a dataset split label from analysis time.
    """
    if at < val_start:
        return "train"

    if at < val_end:
        return "val"

    if at < test_start:
        return "embargo"

    return "test"


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
    Run streaming inference over prepared parquet files for a validtime window.
    """
    model.eval()

    at_min = plot_vt_start - timedelta(days=history_buffer_days)
    at_max = plot_vt_end_excl

    buf = defaultdict(lambda: deque(maxlen=seq_len))

    out_chunks = []
    xb, mb, meta = [], [], []

    def flush():
        nonlocal xb, mb, meta, out_chunks

        if not xb:
            return

        x = torch.from_numpy(np.stack(xb, axis=0)).to(DEVICE, non_blocking=True)
        m = torch.from_numpy(np.stack(mb, axis=0)).to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            bias_hat = model(x, m).float().cpu().numpy()

        sid, at, vt, lt, raw_fc, obs, split_set = zip(*meta)

        corrected = np.asarray(raw_fc, dtype=np.float32) + bias_hat.astype(np.float32)

        out_chunks.append(
            pl.DataFrame(
                {
                    "SID": sid,
                    "analysistime_dt": at,
                    "validtime_dt": vt,
                    "analysistime": at,
                    "validtime": vt,
                    "leadtime": lt,
                    "raw_fc": np.asarray(raw_fc, dtype=np.float32),
                    LABEL_OBS: np.asarray(obs, dtype=np.float32),
                    CORR_COL: corrected,
                    "split_set": split_set,
                }
            )
        )

        xb, mb, meta = [], [], []

    needed = [
        SID_COL,
        "analysistime_dt",
        "validtime_dt",
        "dt_valid_hours",
        *FEATS,
        LABEL_OBS,
    ]

    for fp in prep_files:
        lf = (
            pl.scan_parquet(fp)
            .select(needed)
            .filter(
                (pl.col("analysistime_dt") >= pl.lit(at_min))
                & (pl.col("analysistime_dt") < pl.lit(at_max))
            )
            .filter(pl.col(TEMP_FC).is_not_null() & pl.col(LABEL_OBS).is_not_null())
        )

        if lf.head(1).collect(engine="streaming").height == 0:
            continue

        df = lf.collect(engine="streaming")

        if df.height == 0:
            continue

        x_block = transform_block(df, FEATS, mu, sd)

        sid = df[SID_COL].to_numpy()
        raw_fc = df[TEMP_FC].to_numpy().astype(np.float32)
        obs = df[LABEL_OBS].to_numpy().astype(np.float32)

        at = df["analysistime_dt"].to_list()
        vt = df["validtime_dt"].to_list()
        lt = df["leadtime"].to_numpy()

        for i in range(df.height):
            key = (str(sid[i]), at[i])
            hist = buf[key]

            if len(hist) >= 1:
                hist_arr = np.stack(hist, axis=0)

                if hist_arr.shape[0] < seq_len:
                    pad = seq_len - hist_arr.shape[0]

                    xseq = np.vstack(
                        [
                            np.zeros((pad, x_block.shape[1]), np.float32),
                            hist_arr,
                        ]
                    )

                    mseq = np.concatenate(
                        [
                            np.zeros(pad, np.float32),
                            np.ones(hist_arr.shape[0], np.float32),
                        ]
                    ).astype(np.float32)

                else:
                    xseq = hist_arr[-seq_len:]
                    mseq = np.ones(seq_len, np.float32)

                vti = vt[i]

                if (vti >= plot_vt_start) and (vti < plot_vt_end_excl):
                    split_set = split_set_from_at(
                        at[i],
                        val_start,
                        val_end,
                        test_start,
                    )

                    if split_set != "embargo":
                        xb.append(xseq)
                        mb.append(mseq)
                        meta.append(
                            (
                                str(sid[i]),
                                at[i],
                                vti,
                                int(lt[i]),
                                float(raw_fc[i]),
                                float(obs[i]),
                                split_set,
                            )
                        )

                        if len(xb) >= batch_size:
                            flush()

            hist.append(x_block[i])

    flush()

    if not out_chunks:
        return pl.DataFrame()

    return pl.concat(out_chunks, how="vertical_relaxed")


def to_plot_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert inference output into the column layout expected by plotting scripts.
    """
    drop_cols = [c for c in ["analysistime", "validtime"] if c in df.columns]

    if drop_cols:
        df = df.drop(drop_cols)

    df = df.rename(
        {
            "analysistime_dt": "analysistime",
            "validtime_dt": "validtime",
        }
    )

    df = df.with_columns(
        pl.col("analysistime").dt.strftime("%Y-%m-%d %H:%M:%S"),
        pl.col("validtime").dt.strftime("%Y-%m-%d %H:%M:%S"),
    )

    return df


# -------------------------
# Main
# -------------------------

def main():
    global DEVICE, MODEL_TAG, CORR_COL

    args = parse_args()

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    MODEL_TAG = args.model_tag
    CORR_COL = f"corrected_{MODEL_TAG}"

    prep_glob = resolve_prep_glob(args.prep_input)
    prep_files = list_files(prep_glob)

    model_dir = Path(args.model_dir)

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(model_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using run_dir:", run_dir)
    print("Prepared input:", prep_glob)
    print("Output directory:", output_dir)
    print("Device:", DEVICE)
    print("Model tag:", MODEL_TAG)
    print("Corrected column:", CORR_COL)

    cfg, mu, sd, seq_len, hidden, num_layers, dropout, model_path = load_run_artifacts(
        run_dir,
        model_tag=MODEL_TAG,
    )

    val_start_final = datetime.fromisoformat(cfg["val_start_final"])
    val_end_final = datetime.fromisoformat(cfg.get("val_end_final", cfg["test_start"]))
    test_start = datetime.fromisoformat(cfg["test_start"])

    test_end_incl = datetime.fromisoformat(args.test_end)
    test_end_excl = test_end_incl + timedelta(days=1)

    in_dim = 2 * len(FEATS) + 1

    model = BiasLSTM(
        in_dim=in_dim,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    print("Loaded model config from final_training_config.json:")
    print(f"  seq_len={seq_len} hidden={hidden} layers={num_layers} dropout={dropout}")
    print(f"Inference validtime window: {test_start} .. {test_end_incl} inclusive")

    all_df = predict_period(
        prep_files,
        model,
        mu,
        sd,
        seq_len=seq_len,
        batch_size=args.pred_batch,
        plot_vt_start=test_start,
        plot_vt_end_excl=test_end_excl,
        val_start=val_start_final,
        val_end=val_end_final,
        test_start=test_start,
        history_buffer_days=args.history_buffer_days,
    )

    if all_df.height == 0:
        print("No rows produced for the requested test period.")
        return

    years = sorted(
        all_df
        .select(pl.col("analysistime_dt").dt.year().unique())
        .to_series()
        .to_list()
    )

    for year in years:
        df_y = all_df.filter(pl.col("analysistime_dt").dt.year() == year)
        df_y = to_plot_schema(df_y)

        tag = f"{SPLIT_COLUMN}_{MODEL_TAG}_{year}"
        out_path = output_dir / f"eval_rows_{tag}_fin.parquet"

        df_y.write_parquet(out_path)

        print("Saved:", out_path)


if __name__ == "__main__":
    main()
