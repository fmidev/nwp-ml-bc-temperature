import os
import argparse


def parse_pre_args():
    """
    Parse CUDA/thread settings before importing torch.

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

import math
import random
from pathlib import Path
from datetime import timedelta
from glob import glob
from collections import defaultdict, deque
import json
import pickle
import gc
from datetime import datetime
import threading
import queue

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.utils.data import get_worker_info
import torch.multiprocessing as mp

import optuna
from optuna.exceptions import TrialPruned


# -------------------------
# Fixed columns
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
STATION_ID_COL = "SID"


# -------------------------
# Runtime globals set in main()
# -------------------------

DEVICE = None
SEED = None

RAW_GLOB = None
PREP_DIR = None
PREP_GLOB = None

MODEL_DIR = None
MODEL_TAG = None
MODEL_PATH = None
STATS_PATH = None

RUN_OPTUNA = None

BATCH_SIZE = None
MAX_EPOCHS = None
PATIENCE = None
MIN_DELTA = None
LR = None
WD = None

N_TRIALS = None
TRIAL_STEPS_PER_EPOCH = None
TRIAL_MAX_EPOCHS = None
TRIAL_PATIENCE = None

FINAL_STEPS_PER_EPOCH = None
FINAL_VAL_STEPS = None

TEST_DAYS = None
VAL_DAYS_FINAL = None
N_FOLDS = None

STATS_SAMPLE_PER_FILE = None

NUM_WORKERS_OPTUNA = None
PREFETCH_FINAL = None
NUM_WORKERS_FINAL = None

MAX_LEAD_HOURS = None
EMBARGO_TD = None


# -------------------------
# Arguments
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BiasLSTM with optional Optuna tuning using prepared parquet files."
    )

    parser.add_argument(
        "--raw-input",
        required=True,
        type=str,
        help=(
            "Raw input parquet file, directory, or glob pattern. "
            "Used when --prepare is given. "
            "Example: /path/to/ml_data_full/ml_data_full_*.parquet"
        ),
    )

    parser.add_argument(
        "--prep-dir",
        required=True,
        type=str,
        help="Directory where prepared parquet files are written/read.",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help="Directory where model run directories are saved.",
    )

    parser.add_argument(
        "--model-tag",
        default="bias_lstm_stream",
        type=str,
        help="Model tag used for saved model/stat filenames. Default: bias_lstm_stream.",
    )

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create prepared parquet files and exit.",
    )

    parser.add_argument(
        "--run-optuna",
        action="store_true",
        help="Run Optuna tuning. If omitted, fixed default hyperparameters are used.",
    )

    parser.add_argument(
        "--n-trials",
        default=30,
        type=int,
        help="Number of Optuna trials. Default: 30.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device. Default: auto.",
    )

    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        type=str,
        help="CUDA_VISIBLE_DEVICES value. Parsed before torch import.",
    )

    parser.add_argument(
        "--threads",
        default="4",
        type=str,
        help="Thread count. Parsed before torch import.",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--batch-size",
        default=16384,
        type=int,
        help="Default/final batch size if Optuna is not used. Default: 16384.",
    )

    parser.add_argument(
        "--max-epochs",
        default=200,
        type=int,
        help="Maximum final training epochs. Default: 200.",
    )

    parser.add_argument(
        "--patience",
        default=8,
        type=int,
        help="Final training early stopping patience. Default: 8.",
    )

    parser.add_argument(
        "--min-delta",
        default=1e-4,
        type=float,
        help="Minimum validation improvement for early stopping. Default: 1e-4.",
    )

    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Default learning rate if fixed params are used. Default: 3e-4.",
    )

    parser.add_argument(
        "--wd",
        default=1e-5,
        type=float,
        help="Default weight decay if fixed params are used. Default: 1e-5.",
    )

    parser.add_argument(
        "--trial-steps-per-epoch",
        default=200,
        type=int,
        help="Max training steps per Optuna epoch. Default: 200.",
    )

    parser.add_argument(
        "--trial-max-epochs",
        default=50,
        type=int,
        help="Max epochs per Optuna trial fold. Default: 50.",
    )

    parser.add_argument(
        "--trial-patience",
        default=3,
        type=int,
        help="Optuna trial early stopping patience. Default: 3.",
    )

    parser.add_argument(
        "--final-steps-per-epoch",
        default=1000,
        type=int,
        help="Max final training steps per epoch. Default: 1000.",
    )

    parser.add_argument(
        "--final-val-steps",
        default=500,
        type=int,
        help="Max final validation steps. Default: 500.",
    )

    parser.add_argument(
        "--test-days",
        default=365,
        type=int,
        help="Number of final days used as test period. Default: 365.",
    )

    parser.add_argument(
        "--val-days-final",
        default=90,
        type=int,
        help="Final validation window length in days. Default: 90.",
    )

    parser.add_argument(
        "--n-folds",
        default=3,
        type=int,
        help="Number of rolling CV folds for Optuna. Default: 3.",
    )

    parser.add_argument(
        "--stats-sample-per-file",
        default=20_000,
        type=int,
        help="Rows sampled per prepared file when fitting normalization stats. Default: 20000.",
    )

    parser.add_argument(
        "--num-workers-optuna",
        default=0,
        type=int,
        help="DataLoader workers during Optuna. Default: 0.",
    )

    parser.add_argument(
        "--num-workers-final",
        default=4,
        type=int,
        help="DataLoader workers during final training. Default: 4.",
    )

    parser.add_argument(
        "--prefetch-final",
        default=2,
        type=int,
        help="DataLoader prefetch factor for final training. Default: 2.",
    )

    parser.add_argument(
        "--max-lead-hours",
        default=240,
        type=int,
        help="Maximum lead time in hours, also used as embargo. Default: 240.",
    )

    parser.add_argument(
        "--fixed-params-json",
        default=None,
        type=str,
        help=(
            "Optional JSON file with fixed final params. "
            "If provided and --run-optuna is not used, these params override defaults."
        ),
    )

    return parser.parse_args()


def resolve_glob(path_arg: str, default_pattern: str) -> str:
    """
    Accept a file, directory, or glob pattern.
    """
    p = Path(path_arg)

    if p.is_dir():
        return str(p / default_pattern)

    return str(p)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_mp_spawn():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


# -------------------------
# Utilities
# -------------------------

def list_files(path_glob):
    files = sorted(glob(str(path_glob)))

    if not files:
        raise FileNotFoundError(f"No parquet files matched: {path_glob}")

    return files


def scan_base(fp: str):
    """
    Robust scanner that supports either:
      - analysistime_dt/validtime_dt already present
      - analysistime/validtime as strings
    """
    try:
        lf = pl.scan_parquet(fp).select(
            FEATS + [LABEL_OBS, STATION_ID_COL, "analysistime_dt", "validtime_dt"]
        )
        _ = lf.head(0).collect(engine="streaming")
        lf = lf.with_columns(pl.col("analysishour").cast(pl.Int16))
        return lf
    except Exception:
        pass

    lf = pl.scan_parquet(fp).select(
        FEATS + [LABEL_OBS, STATION_ID_COL, "analysistime", "validtime"]
    )

    lf = (
        lf.with_columns(
            pl.col("analysistime")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, strict=False)
            .alias("analysistime_dt"),

            pl.col("validtime")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, strict=False)
            .alias("validtime_dt"),

            pl.col("analysishour").cast(pl.Int16),
        )
        .drop(["analysistime", "validtime"])
    )

    _ = lf.head(0).collect(engine="streaming")

    return lf


def dedup_per_station_validtime(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort([STATION_ID_COL, "analysistime_dt", "leadtime", "validtime_dt"])
        .unique(
            subset=[STATION_ID_COL, "analysistime_dt", "leadtime", "analysishour"],
            keep="last",
        )
    )


def add_dt_hours(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort([STATION_ID_COL, "leadtime", "analysistime_dt"])
        .with_columns(
            (
                pl.col("analysistime_dt")
                .diff()
                .over([STATION_ID_COL, "leadtime"])
                .dt.total_hours()
                .fill_null(0.0)
                .clip(0.0, 24.0 * 30.0)
                .alias("dt_hours")
            )
        )
    )


def add_dt_valid_hours(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort([STATION_ID_COL, "analysistime_dt", "validtime_dt"])
        .with_columns(
            (
                pl.col("validtime_dt")
                .diff()
                .over([STATION_ID_COL, "analysistime_dt"])
                .dt.total_hours()
                .fill_null(0.0)
                .clip(0.0, 24.0 * 30.0)
                .alias("dt_valid_hours")
            )
        )
    )


def get_global_minmax_validtime(files):
    vt_min = None
    vt_max = None

    for fp in files:
        lf = scan_base(fp).select("validtime_dt")

        df = (
            lf.select(
                pl.col("validtime_dt").min().alias("mn"),
                pl.col("validtime_dt").max().alias("mx"),
            )
            .collect(engine="streaming")
        )

        mn = df["mn"][0]
        mx = df["mx"][0]

        if mn is None or mx is None:
            continue

        vt_min = mn if vt_min is None else min(vt_min, mn)
        vt_max = mx if vt_max is None else max(vt_max, mx)

    if vt_min is None or vt_max is None:
        raise RuntimeError("Could not infer validtime_dt min/max from files.")

    return vt_min, vt_max


def split_trainval_test_rolling_from_files(
    prep_files,
    test_days=365,
    n_folds=3,
    train_start_dt=None,
    embargo_td=None,
):
    """
    Rolling folds using analysistime, with embargo before validation.
    """
    if embargo_td is None:
        embargo_td = EMBARGO_TD

    _, vt_max = get_global_minmax_validtime(prep_files)
    test_start = vt_max - timedelta(days=test_days)

    latest_val_init = test_start - embargo_td

    chunks = []

    for fp in prep_files:
        lf = pl.scan_parquet(fp).select("analysistime_dt")

        if train_start_dt is None:
            lf = lf.filter(pl.col("analysistime_dt") < latest_val_init)
        else:
            lf = lf.filter(
                (pl.col("analysistime_dt") < latest_val_init)
                & (pl.col("analysistime_dt") >= train_start_dt)
            )

        df = lf.unique().collect(engine="streaming")

        if df.height:
            chunks.append(df)

    if not chunks:
        raise RuntimeError("No analysistime_dt found for train/val period.")

    inits = (
        pl.concat(chunks, how="vertical_relaxed")
        .unique()
        .sort("analysistime_dt")["analysistime_dt"]
        .to_list()
    )

    if len(inits) < n_folds + 1:
        raise ValueError("Not enough initializations for requested folds.")

    fold_edges = [
        int(round(i * len(inits) / (n_folds + 1)))
        for i in range(1, n_folds + 1)
    ]

    step = (fold_edges[1] - fold_edges[0]) if len(fold_edges) > 1 else fold_edges[0]

    folds = []

    for edge in fold_edges:
        val_start = inits[edge - 1]

        next_edge = min(edge + step, len(inits))
        val_end = inits[next_edge - 1] if next_edge > edge else inits[-1]

        train_end = val_start - embargo_td

        if train_start_dt is not None and train_end <= train_start_dt:
            continue

        if train_end >= val_start:
            continue

        folds.append((train_end, val_start, val_end))

    if not folds:
        raise ValueError("No valid folds left after applying embargo.")

    return folds, test_start


def split_final_train_val_test_from_max(
    prep_files,
    test_days=365,
    val_days=90,
    train_start_dt=None,
    embargo_td=None,
):
    """
    Final split with embargo between validation and test.
    """
    if embargo_td is None:
        embargo_td = EMBARGO_TD

    _, vt_max = get_global_minmax_validtime(prep_files)

    test_start = vt_max - timedelta(days=test_days)
    val_end = test_start - embargo_td
    val_start = val_end - timedelta(days=val_days)

    if train_start_dt is not None and val_start <= train_start_dt:
        raise ValueError("Validation window too early after applying embargo.")

    return val_start, val_end, test_start


def split_ranges_from_files(files, test_days=365, val_days=90):
    _, vt_max = get_global_minmax_validtime(files)

    test_start = vt_max - timedelta(days=test_days)
    val_start = test_start - timedelta(days=val_days)

    return val_start, test_start


# -------------------------
# Prepare step
# -------------------------

def prepare_files_once(raw_files):
    """
    Create prepared parquet files.
    """
    needed = [STATION_ID_COL, "validtime_dt", "analysistime_dt", *FEATS, LABEL_OBS]

    PREP_DIR.mkdir(parents=True, exist_ok=True)

    for fp in raw_files:
        out = PREP_DIR / (Path(fp).name.replace("ml_data_full_", "ml_data_prep_"))

        if out.exists():
            continue

        df = scan_base(fp).select(needed).collect(engine="streaming")

        if df.height == 0:
            continue

        df = df.sort([STATION_ID_COL, "analysistime_dt", "validtime_dt"])
        df = add_dt_valid_hours(df)

        if df.height == 0:
            continue

        df.write_parquet(out)
        print("Wrote", out)


# -------------------------
# Fit stats
# -------------------------

def fit_stats_strict_train(prep_files, feats, time_max, sample_per_file, seed):
    """
    Fit mean/std using train-only data.
    """
    n = {f: 0 for f in feats}
    s1 = {f: 0.0 for f in feats}
    s2 = {f: 0.0 for f in feats}

    for fp in prep_files:
        lf = (
            pl.scan_parquet(fp)
            .filter(pl.col("analysistime_dt") < time_max)
            .select(feats)
            .head(sample_per_file * 5)
        )

        df = lf.collect(engine="streaming")

        if df.height == 0:
            continue

        if df.height > sample_per_file:
            df = df.sample(n=sample_per_file, with_replacement=False, seed=seed)

        for f in feats:
            arr = df[f].to_numpy()
            arr = arr[~np.isnan(arr)]

            if arr.size:
                n[f] += arr.size
                s1[f] += float(arr.sum())
                s2[f] += float((arr * arr).sum())

    mu, sd = {}, {}

    for f in feats:
        if n[f] == 0:
            mu[f], sd[f] = 0.0, 1.0
            continue

        mean = s1[f] / n[f]
        var = max(s2[f] / n[f] - mean * mean, 1e-12)

        mu[f] = float(mean)
        sd[f] = float(np.sqrt(var)) if var > 0 else 1.0

        if sd[f] == 0.0:
            sd[f] = 1.0

    return mu, sd


# -------------------------
# Dataset
# -------------------------

class PrepStreamSeqDataset(IterableDataset):
    """
    Streaming dataset for prepared parquet files.
    """

    def __init__(
        self,
        files,
        time_min,
        time_max,
        feats,
        label_obs,
        temp_fc,
        sid_col,
        seq_len,
        mu,
        sd,
    ):
        super().__init__()

        self.files = files
        self.time_min = time_min
        self.time_max = time_max
        self.feats = feats
        self.label_obs = label_obs
        self.temp_fc = temp_fc
        self.sid_col = sid_col
        self.seq_len = seq_len
        self.mu = mu
        self.sd = sd
        self.D = 2 * len(feats) + 1

    def _transform_block(self, df: pl.DataFrame) -> np.ndarray:
        x = np.empty((df.height, self.D), dtype=np.float32)

        k = 0

        for f in self.feats:
            col = df[f].to_numpy()
            miss = np.isnan(col)

            z = np.zeros_like(col, dtype=np.float32)

            if (~miss).any():
                z[~miss] = ((col[~miss] - self.mu[f]) / self.sd[f]).astype(
                    np.float32,
                    copy=False,
                )

            x[:, k] = z
            x[:, k + 1] = miss.astype(np.float32)

            k += 2

        dt = df["dt_valid_hours"].to_numpy()
        dt = np.nan_to_num(dt, nan=0.0).astype(np.float32, copy=False)

        x[:, -1] = dt

        return x

    def __iter__(self):
        buf = defaultdict(lambda: deque(maxlen=self.seq_len))

        worker = get_worker_info()

        if worker is None:
            files = self.files
        else:
            files = self.files[worker.id::worker.num_workers]

        for fp in files:
            lf = (
                pl.scan_parquet(fp)
                .select(
                    [
                        self.sid_col,
                        "analysistime_dt",
                        "validtime_dt",
                        "dt_valid_hours",
                        *self.feats,
                        self.label_obs,
                    ]
                )
                .filter(
                    (pl.col("validtime_dt") >= self.time_min)
                    & (pl.col("validtime_dt") < self.time_max)
                )
                .filter(
                    pl.col(self.label_obs).is_not_null()
                    & pl.col(self.temp_fc).is_not_null()
                )
            )

            df = lf.collect(engine="streaming")

            if df.height == 0:
                continue

            x_block = self._transform_block(df)

            sid = df[self.sid_col].to_numpy()
            at = df["analysistime_dt"].to_list()
            y = (df[self.label_obs] - df[self.temp_fc]).to_numpy().astype(np.float32)

            for i in range(df.height):
                key = (str(sid[i]), at[i])
                hist = buf[key]

                if len(hist) >= 1:
                    hist_arr = np.stack(hist, axis=0)

                    if hist_arr.shape[0] < self.seq_len:
                        pad = self.seq_len - hist_arr.shape[0]

                        xseq = np.vstack(
                            [
                                np.zeros((pad, self.D), np.float32),
                                hist_arr,
                            ]
                        )

                        m = np.concatenate(
                            [
                                np.zeros(pad, np.float32),
                                np.ones(hist_arr.shape[0], np.float32),
                            ]
                        ).astype(np.float32)

                    else:
                        xseq = hist_arr[-self.seq_len:]
                        m = np.ones(self.seq_len, np.float32)

                    yield (
                        torch.from_numpy(xseq),
                        torch.tensor(y[i], dtype=torch.float32),
                        torch.from_numpy(m),
                    )

                hist.append(x_block[i])


def collate_batch(batch):
    x, y, m = zip(*batch)

    return torch.stack(x, 0), torch.stack(y, 0), torch.stack(m, 0)


class ThreadPrefetcher:
    """
    Wrap iterable loader and prefetch batches in a background thread.
    """

    def __init__(self, loader, max_prefetch=8):
        self.loader = loader
        self.q = queue.Queue(maxsize=max_prefetch)
        self._stop = object()

    def __iter__(self):
        it = iter(self.loader)

        def _worker():
            try:
                for batch in it:
                    self.q.put(batch)
            finally:
                self.q.put(self._stop)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while True:
            batch = self.q.get()

            if batch is self._stop:
                break

            yield batch


# -------------------------
# Model
# -------------------------

class BiasLSTM(nn.Module):
    """
    Sequence-to-scalar LSTM model for bias prediction.
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

    def forward(self, x, pad_mask=None):
        out, _ = self.lstm(x)

        lengths = pad_mask.sum(dim=1).clamp(min=1).long()
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))

        last = out.gather(1, idx).squeeze(1)

        return self.head(last).squeeze(-1)


# -------------------------
# Train / eval
# -------------------------

def global_rmse_from_loader(model, loader, device, max_steps=None):
    model.eval()

    se = 0.0
    n = 0
    steps = 0

    with torch.no_grad():
        for x, y, m in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            pred = model(x, m)

            se += ((pred - y) ** 2).sum().item()
            n += y.numel()

            steps += 1

            if max_steps is not None and steps >= max_steps:
                break

    return math.sqrt(se / (n + 1e-8))


def train_epoch_amp(model, loader, opt, device, grad_clip=2.0, max_steps=None):
    model.train()

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    se = 0.0
    n = 0
    steps = 0

    for x, y, m in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            pred = model(x, m)
            loss = ((pred - y) ** 2).mean()

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(opt)
        scaler.update()

        se += ((pred - y) ** 2).sum().item()
        n += y.numel()

        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return math.sqrt(se / (n + 1e-8))


def make_run_dir():
    run_dir = MODEL_DIR / f"bias_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


# -------------------------
# Main
# -------------------------

def main():
    global DEVICE, SEED
    global RAW_GLOB, PREP_DIR, PREP_GLOB
    global MODEL_DIR, MODEL_TAG, MODEL_PATH, STATS_PATH
    global RUN_OPTUNA
    global BATCH_SIZE, MAX_EPOCHS, PATIENCE, MIN_DELTA, LR, WD
    global N_TRIALS, TRIAL_STEPS_PER_EPOCH, TRIAL_MAX_EPOCHS, TRIAL_PATIENCE
    global FINAL_STEPS_PER_EPOCH, FINAL_VAL_STEPS
    global TEST_DAYS, VAL_DAYS_FINAL, N_FOLDS
    global STATS_SAMPLE_PER_FILE
    global NUM_WORKERS_OPTUNA, PREFETCH_FINAL, NUM_WORKERS_FINAL
    global MAX_LEAD_HOURS, EMBARGO_TD

    args = parse_args()

    SEED = args.seed
    set_seed(SEED)

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    RAW_GLOB = resolve_glob(args.raw_input, "ml_data_full_*.parquet")

    PREP_DIR = Path(args.prep_dir)
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    PREP_GLOB = PREP_DIR / "ml_data_prep_*.parquet"

    MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_TAG = args.model_tag
    MODEL_PATH = MODEL_DIR / f"{MODEL_TAG}.pt"
    STATS_PATH = MODEL_DIR / f"{MODEL_TAG}_stats.json"

    RUN_OPTUNA = args.run_optuna

    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.max_epochs
    PATIENCE = args.patience
    MIN_DELTA = args.min_delta
    LR = args.lr
    WD = args.wd

    N_TRIALS = args.n_trials
    TRIAL_STEPS_PER_EPOCH = args.trial_steps_per_epoch
    TRIAL_MAX_EPOCHS = args.trial_max_epochs
    TRIAL_PATIENCE = args.trial_patience

    FINAL_STEPS_PER_EPOCH = args.final_steps_per_epoch
    FINAL_VAL_STEPS = args.final_val_steps

    TEST_DAYS = args.test_days
    VAL_DAYS_FINAL = args.val_days_final
    N_FOLDS = args.n_folds

    STATS_SAMPLE_PER_FILE = args.stats_sample_per_file

    NUM_WORKERS_OPTUNA = args.num_workers_optuna
    PREFETCH_FINAL = args.prefetch_final
    NUM_WORKERS_FINAL = args.num_workers_final

    MAX_LEAD_HOURS = args.max_lead_hours
    EMBARGO_TD = timedelta(hours=MAX_LEAD_HOURS)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Raw input: {RAW_GLOB}")
    print(f"[INFO] Prepared dir: {PREP_DIR}")
    print(f"[INFO] Model dir: {MODEL_DIR}")
    print(f"[INFO] Model tag: {MODEL_TAG}")
    print(f"[INFO] Run Optuna: {RUN_OPTUNA}")

    raw_files = list_files(RAW_GLOB)
    _set_mp_spawn()

    # ------------------------------------------------------------
    # 0) Prepare step
    # ------------------------------------------------------------
    if args.prepare:
        print("Preparing parquet files...")
        prepare_files_once(raw_files)
        print("Done. Prepared files in:", PREP_DIR)
        return

    prep_files = list_files(PREP_GLOB)

    # ------------------------------------------------------------
    # 1) Create run directory
    # ------------------------------------------------------------
    run_dir = make_run_dir()
    print("Run dir:", run_dir)

    # ------------------------------------------------------------
    # 2) Rolling CV folds + final split
    # ------------------------------------------------------------
    folds, test_start = split_trainval_test_rolling_from_files(
        prep_files,
        test_days=TEST_DAYS,
        n_folds=N_FOLDS,
        train_start_dt=None,
        embargo_td=EMBARGO_TD,
    )

    val_start_final, val_end_final, test_start = split_final_train_val_test_from_max(
        prep_files,
        test_days=TEST_DAYS,
        val_days=VAL_DAYS_FINAL,
        train_start_dt=None,
        embargo_td=EMBARGO_TD,
    )

    print("Final split dates:")
    print("  val_start_final:", val_start_final)
    print("  val_end_final  :", val_end_final)
    print("  test_start     :", test_start)
    print("  embargo        :", EMBARGO_TD)

    # ------------------------------------------------------------
    # 3) Fit normalization stats
    # ------------------------------------------------------------
    print("Fitting normalization stats on train data sample...")

    mu, sd = fit_stats_strict_train(
        prep_files=prep_files,
        feats=FEATS,
        time_max=val_start_final,
        sample_per_file=STATS_SAMPLE_PER_FILE,
        seed=SEED,
    )

    stats_path = run_dir / f"{MODEL_TAG}_stats.json"

    with open(stats_path, "w") as f:
        json.dump({"mu": mu, "sd": sd}, f)

    print("Saved stats to:", stats_path)

    in_dim = 2 * len(FEATS) + 1
    print("LSTM input dim:", in_dim)

    # ------------------------------------------------------------
    # 4) Loader factory
    # ------------------------------------------------------------
    def make_loaders(
        batch_size: int,
        train_end,
        val_start,
        val_end,
        seq_len,
        test_start=None,
        num_workers: int = None,
        prefetch_factor: int = None,
    ):
        if num_workers is None:
            num_workers = NUM_WORKERS_FINAL

        if prefetch_factor is None:
            prefetch_factor = PREFETCH_FINAL

        ds_tr = PrepStreamSeqDataset(
            files=prep_files,
            time_min=pl.datetime(1900, 1, 1),
            time_max=train_end,
            feats=FEATS,
            label_obs=LABEL_OBS,
            temp_fc=TEMP_FC,
            sid_col=STATION_ID_COL,
            seq_len=seq_len,
            mu=mu,
            sd=sd,
        )

        ds_va = PrepStreamSeqDataset(
            files=prep_files,
            time_min=val_start,
            time_max=val_end,
            feats=FEATS,
            label_obs=LABEL_OBS,
            temp_fc=TEMP_FC,
            sid_col=STATION_ID_COL,
            seq_len=seq_len,
            mu=mu,
            sd=sd,
        )

        ds_ts = None

        if test_start is not None:
            ds_ts = PrepStreamSeqDataset(
                files=prep_files,
                time_min=test_start,
                time_max=pl.datetime(2200, 1, 1),
                feats=FEATS,
                label_obs=LABEL_OBS,
                temp_fc=TEMP_FC,
                sid_col=STATION_ID_COL,
                seq_len=seq_len,
                mu=mu,
                sd=sd,
            )

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            pin_memory=(DEVICE == "cuda"),
            num_workers=num_workers,
        )

        if num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=prefetch_factor,
                multiprocessing_context=mp.get_context("spawn"),
            )

        dl_tr = DataLoader(ds_tr, **loader_kwargs)
        dl_va = DataLoader(ds_va, **loader_kwargs)

        dl_ts = None

        if ds_ts is not None:
            dl_ts = DataLoader(ds_ts, **loader_kwargs)

        return dl_tr, dl_va, dl_ts

    # ------------------------------------------------------------
    # 5) Optuna
    # ------------------------------------------------------------
    best_params = None

    if RUN_OPTUNA:
        def objective(trial: optuna.Trial):
            hidden = trial.suggest_categorical("hidden", [160, 192, 256])
            num_layers = trial.suggest_int("num_layers", 1, 1)
            dropout = trial.suggest_float("dropout", 0.0, 0.4)
            lr = trial.suggest_float("lr", 5e-5, 8e-4, log=True)
            wd = trial.suggest_float("wd", 1e-6, 5e-3, log=True)
            grad_clip = trial.suggest_float("grad_clip", 0.5, 3.0)
            bs = trial.suggest_categorical("batch_size", [8192, 16384])
            seq_len = trial.suggest_int("seq_len", 5, 24)

            fold_bests = []
            global_step = 0

            for fold_id, (train_end, vs, ve) in enumerate(folds):
                dl_tr, dl_va, _ = make_loaders(
                    bs,
                    train_end=train_end,
                    val_start=vs,
                    val_end=ve,
                    num_workers=NUM_WORKERS_OPTUNA,
                    prefetch_factor=PREFETCH_FINAL,
                    seq_len=seq_len,
                )

                model = BiasLSTM(
                    in_dim=in_dim,
                    hidden=hidden,
                    num_layers=num_layers,
                    dropout=dropout,
                ).to(DEVICE)

                opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

                best_val = float("inf")
                bad = 0

                try:
                    for ep in range(TRIAL_MAX_EPOCHS):
                        _ = train_epoch_amp(
                            model,
                            dl_tr,
                            opt,
                            DEVICE,
                            grad_clip=grad_clip,
                            max_steps=TRIAL_STEPS_PER_EPOCH,
                        )

                        va_rmse = global_rmse_from_loader(
                            model,
                            dl_va,
                            DEVICE,
                            max_steps=TRIAL_STEPS_PER_EPOCH,
                        )

                        print(
                            f"[trial {trial.number:03d}] "
                            f"fold {fold_id} ep {ep:02d} "
                            f"val_rmse={va_rmse:.4f}"
                        )

                        if va_rmse < best_val - MIN_DELTA:
                            best_val = va_rmse
                            bad = 0
                        else:
                            bad += 1

                            if bad >= TRIAL_PATIENCE:
                                break

                        trial.report(best_val, step=global_step)
                        global_step += 1

                        if trial.should_prune():
                            raise TrialPruned()

                finally:
                    del model, opt, dl_tr, dl_va
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                fold_bests.append(best_val)

            return float(np.mean(fold_bests))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        )

        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

        best_params = dict(study.best_params)

        print("Optuna best value mean CV val RMSE:", study.best_value)
        print("Optuna best params:", best_params)

        with open(run_dir / "best_params.json", "w") as f:
            json.dump(
                {
                    "best_value": float(study.best_value),
                    "best_params": best_params,
                },
                f,
                indent=2,
            )

        with open(run_dir / "optuna_study.pkl", "wb") as f:
            pickle.dump(study, f)

        print("Saved Optuna artifacts to:", run_dir)

    # ------------------------------------------------------------
    # 6) Choose final hyperparameters
    # ------------------------------------------------------------
    if best_params is not None:
        final_hidden = int(best_params["hidden"])
        final_layers = int(best_params["num_layers"])
        final_dropout = float(best_params["dropout"])
        final_lr = float(best_params["lr"])
        final_wd = float(best_params["wd"])
        final_clip = float(best_params["grad_clip"])
        final_bs = int(best_params["batch_size"])
        final_seq_len = int(best_params["seq_len"])

    elif args.fixed_params_json is not None:
        fixed_params_path = Path(args.fixed_params_json)

        if not fixed_params_path.exists():
            raise FileNotFoundError(f"Fixed params JSON not found: {fixed_params_path}")

        with open(fixed_params_path, "r") as f:
            fixed_params = json.load(f)

        final_hidden = int(fixed_params["hidden"])
        final_layers = int(fixed_params.get("num_layers", 1))
        final_dropout = float(fixed_params["dropout"])
        final_lr = float(fixed_params["lr"])
        final_wd = float(fixed_params["wd"])
        final_clip = float(fixed_params["grad_clip"])
        final_bs = int(fixed_params.get("batch_size", BATCH_SIZE))
        final_seq_len = int(fixed_params["seq_len"])

    else:
        final_hidden = 192
        final_layers = 1
        final_dropout = 0.2447411578889518
        final_lr = 0.000727288685484562
        final_wd = 0.000628977181948703
        final_clip = 1.4159046082342293
        final_bs = BATCH_SIZE
        final_seq_len = 24

    with open(run_dir / "final_training_config.json", "w") as f:
        json.dump(
            {
                "model_tag": MODEL_TAG,
                "seq_len": final_seq_len,
                "batch_size": final_bs,
                "hidden": final_hidden,
                "num_layers": final_layers,
                "dropout": final_dropout,
                "lr": final_lr,
                "wd": final_wd,
                "grad_clip": final_clip,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "min_delta": MIN_DELTA,
                "trial_steps_per_epoch": TRIAL_STEPS_PER_EPOCH,
                "trial_max_epochs": TRIAL_MAX_EPOCHS,
                "trial_patience": TRIAL_PATIENCE,
                "num_workers_final": NUM_WORKERS_FINAL,
                "prefetch_factor": PREFETCH_FINAL,
                "val_start_final": str(val_start_final),
                "val_end_final": str(val_end_final),
                "test_start": str(test_start),
                "n_folds": N_FOLDS,
                "test_days": TEST_DAYS,
                "val_days_final": VAL_DAYS_FINAL,
                "max_lead_hours": MAX_LEAD_HOURS,
                "embargo_hours": MAX_LEAD_HOURS,
            },
            f,
            indent=2,
        )

    print("Saved final training config to:", run_dir / "final_training_config.json")

    # ------------------------------------------------------------
    # 7) Final training
    # ------------------------------------------------------------
    dl_tr, dl_va, dl_ts = make_loaders(
        final_bs,
        train_end=val_start_final,
        val_start=val_start_final,
        val_end=val_end_final,
        test_start=test_start,
        num_workers=NUM_WORKERS_FINAL,
        prefetch_factor=PREFETCH_FINAL,
        seq_len=final_seq_len,
    )

    model = BiasLSTM(
        in_dim=in_dim,
        hidden=final_hidden,
        num_layers=final_layers,
        dropout=final_dropout,
    ).to(DEVICE)

    opt = AdamW(model.parameters(), lr=final_lr, weight_decay=final_wd)

    best_val = float("inf")
    bad = 0
    best_state = None

    for ep in range(MAX_EPOCHS):
        tr_rmse = train_epoch_amp(
            model,
            dl_tr,
            opt,
            DEVICE,
            grad_clip=final_clip,
            max_steps=FINAL_STEPS_PER_EPOCH,
        )

        va_rmse = global_rmse_from_loader(
            model,
            dl_va,
            DEVICE,
            max_steps=FINAL_VAL_STEPS,
        )

        print(f"Epoch {ep:03d}: train_rmse={tr_rmse:.4f} val_rmse={va_rmse:.4f}")

        if va_rmse < best_val - MIN_DELTA:
            best_val = va_rmse
            bad = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:
            bad += 1

            if bad >= PATIENCE:
                print(f"Early stop at ep={ep:03d}, best_val_rmse={best_val:.4f}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = run_dir / f"{MODEL_TAG}.pt"
    torch.save(model.state_dict(), model_path)

    print("Saved model weights to:", model_path)

    ts_rmse_bias = global_rmse_from_loader(model, dl_ts, DEVICE)

    print(f"\nTEST global RMSE on bias obs-fc: {ts_rmse_bias:.4f}")

    with open(run_dir / "results.json", "w") as f:
        json.dump(
            {
                "best_val_rmse": float(best_val),
                "test_rmse_bias": float(ts_rmse_bias),
            },
            f,
            indent=2,
        )

    print("Saved results to:", run_dir / "results.json")
    print("[DONE] Run directory:", run_dir)


if __name__ == "__main__":
    main()
