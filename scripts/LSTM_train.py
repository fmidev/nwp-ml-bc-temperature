import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # choose GPU #1

import math
import random
import argparse
from pathlib import Path
from datetime import timedelta
from glob import glob
from collections import defaultdict, deque
import json

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.utils.data import get_worker_info

import optuna
from optuna.exceptions import TrialPruned
import pickle
import gc

import time
from datetime import datetime
import torch.multiprocessing as mp

import threading, queue



# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

HOME = Path.home()
RAW_GLOB = HOME / "thesis_project" / "data" / "ml_data_full" / "ml_data_full_*.parquet"
PREP_DIR = HOME / "thesis_project" / "data" / "ml_data_prepared"
PREP_DIR.mkdir(parents=True, exist_ok=True)
PREP_GLOB = PREP_DIR / "ml_data_prep_*.parquet"

FEATS = [
    "T2","SKT","MX2T","MN2T","D2","T_925","MSL","U10","V10","T2_M1","T_925_M1",
    "T2_ENSMEAN_MA1", "LCC", "MCC", "sin_hod","cos_hod","sin_doy","cos_doy",
    "analysishour","leadtime","lon","lat","elev"
]
LABEL_OBS = "obs_TA"
TEMP_FC   = "T2"
STATION_ID_COL = "SID"

#SEQ_LEN = 5
BATCH_SIZE = 16384          # use your VRAM
MAX_EPOCHS = 200
PATIENCE = 8
MIN_DELTA = 1e-4
LR = 3e-4
WD = 1e-5

# -------------------------
# Optuna / tuning controls
# -------------------------
RUN_OPTUNA = True
N_TRIALS = 30

# For IterableDataset we must cap work per trial, otherwise each "epoch" is huge.
TRIAL_STEPS_PER_EPOCH = 200   # e.g. 200–2000
TRIAL_MAX_EPOCHS = 50       # per trial
TRIAL_PATIENCE = 3            # per trial early stopping

FINAL_STEPS_PER_EPOCH = 1000 
FINAL_VAL_STEPS       = 500

TEST_DAYS = 365
VAL_DAYS_FINAL = 90
N_FOLDS = 3


# Stats fitting
STATS_SAMPLE_PER_FILE = 20_000

# Loader tuning (effective only after prepare step)
NUM_WORKERS_OPTUNA = 0
PREFETCH_FINAL = 2
NUM_WORKERS_FINAL = 4

# Artifacts for inference
MODEL_DIR = HOME / "thesis_project" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "bias_lstm_stream.pt"
STATS_PATH = MODEL_DIR / "bias_lstm_stream_stats.json"

MAX_LEAD_HOURS = 240
EMBARGO_TD = timedelta(hours=MAX_LEAD_HOURS)


import torch.multiprocessing as mp

def _set_mp_spawn():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # already set in this interpreter
        pass



# -------------------------
# Utilities
# -------------------------

def split_trainval_test_rolling_from_files(prep_files, test_days=365, n_folds=3, train_start_dt=None, embargo_td = EMBARGO_TD):
    """
        -Splits made using analysistime instead of validtime like in XGB/GNN to keep full forecasts intact
        - train_end = val_start - embargo_td
        -Added embargo_td to not leak information into the next fold
    Returns:
      folds = list of (train_end, val_start, val_end) as datetimes
      test_start
    """

    # Get global max validtime_dt
    _, vt_max = get_global_minmax_validtime(prep_files)
    test_start = vt_max - timedelta(days=test_days)

    latest_val_init = test_start - embargo_td

    # Gather unique analysistime_dt in TV window (streaming per file)
    chunks = []
    for fp in prep_files:
        lf = pl.scan_parquet(fp).select("analysistime_dt")

        if train_start_dt is None:
            lf = lf.filter(pl.col("analysistime_dt") < latest_val_init)
        else:
            lf = lf.filter((pl.col("analysistime_dt") < latest_val_init) & (pl.col("analysistime_dt") >= train_start_dt))

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
        raise ValueError("Not enough initializations for requested N_FOLDS after applying TRAIN_START_DT.")

    fold_edges = [int(round(i * len(inits) / (n_folds + 1))) for i in range(1, n_folds + 1)]
    step = (fold_edges[1] - fold_edges[0]) if len(fold_edges) > 1 else fold_edges[0]

    folds = []
    for edge in fold_edges:
        val_start = inits[edge - 1]

        next_edge = min(edge + step, len(inits))
        if next_edge > edge:
            val_end = inits[next_edge - 1]
        else:
            val_end = inits[-1]

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
    embargo_td=EMBARGO_TD,
):
    """
    Final split with embargo between validation and test.

    Returns:
      val_start, val_end, test_start

      train: analysistime_dt < val_start
      val:   analysistime_dt >= val_start and analysistime_dt < val_end
      test:  analysistime_dt >= test_start

      embargo region [val_end, test_start) is dropped.
    """
    _, vt_max = get_global_minmax_validtime(prep_files)
    test_start = vt_max - timedelta(days=test_days)

    val_end = test_start - embargo_td
    val_start = val_end - timedelta(days=val_days)

    if train_start_dt is not None and val_start <= train_start_dt:
        raise ValueError("Validation window too early after applying embargo.")

    return val_start, val_end, test_start





def list_files(path_glob: Path):
    files = sorted(glob(str(path_glob)))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {path_glob}")
    return files


def scan_base(fp: str):
    """
    Robust scanner that supports either:
      - analysistime_dt/validtime_dt already present in parquet, OR
      - analysistime/validtime as strings that need parsing.
    Uses a try/fallback (schema-based checks can be unreliable on older polars).
    """
    # Try dt columns first
    try:
        lf = pl.scan_parquet(fp).select(
            FEATS + [LABEL_OBS, STATION_ID_COL, "analysistime_dt", "validtime_dt"]
        )
        _ = lf.head(0).collect(engine="streaming")
        lf = lf.with_columns(pl.col("analysishour").cast(pl.Int16))
        return lf
    except Exception:
        pass

    # Fallback to string columns
    lf = pl.scan_parquet(fp).select(
        FEATS + [LABEL_OBS, STATION_ID_COL, "analysistime", "validtime"]
    )
    lf = lf.with_columns(
        pl.col("analysistime").str.strptime(pl.Datetime, strict=False).alias("analysistime_dt"),
        pl.col("validtime").str.strptime(pl.Datetime, strict=False).alias("validtime_dt"),
        pl.col("analysishour").cast(pl.Int16),
    ).drop(["analysistime", "validtime"])
    _ = lf.head(0).collect(engine="streaming")
    return lf


def dedup_per_station_validtime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Deterministic de-duplication per station and forecast cycle.

    Sorts rows by station id, analysis time, leadtime, and valid time so that a
    consistent “last row wins” rule can be applied. Then drops duplicates based on
    (station id, analysistime_dt, leadtime, analysishour), keeping only the last
    occurrence in the sorted order.

    Useful when the same forecast hour/cycle appears multiple times (e.g. reprocessing
    or overlapping loads) and you want one canonical record.
    """
    return (
        df.sort([STATION_ID_COL, "analysistime_dt", "leadtime", "validtime_dt"])
          .unique(subset=[STATION_ID_COL, "analysistime_dt", "leadtime", "analysishour"], keep="last")
    )


def add_dt_hours(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds per-station, per-leadtime time-step size (hours) between analysis times.

    Within each (station id, leadtime) group, computes the difference between consecutive
    analysistime_dt values, converts that duration to hours, and stores it as dt_hours.

    - The first row of each group has no previous timestamp, so dt_hours is filled with 0.0.
    - Negative steps (out-of-order data) are clipped to 0.0.
    - Very large gaps are capped at 30 days (720 hours) to avoid extreme outliers.

    Assumes analysistime_dt is a Polars Datetime column.
    """
    return (
        df.sort([STATION_ID_COL, "leadtime", "analysistime_dt"])
          .with_columns(
              (pl.col("analysistime_dt")
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
    """
    Add a `dt_valid_hours` column containing the time difference in hours
    between consecutive `validtime_dt` values within each station and
    `analysistime_dt` group.

    The input is first sorted by station, analysis time, and valid time so
    that differences are calculated in chronological order. The first row in
    each group gets 0.0, and all values are clipped to the range [0, 30 days].
    """
    return (
        df.sort([STATION_ID_COL, "analysistime_dt", "validtime_dt"])
          .with_columns(
              (
                  # Compute the difference between consecutive valid times
                  # for each (station, analysistime) group.
                  pl.col("validtime_dt")
                    .diff()
                    .over([STATION_ID_COL, "analysistime_dt"])
                    .dt.total_hours()
                    # The first row in each group has no previous value,
                    # so replace null with 0.0 hours.
                    .fill_null(0.0)
                    .clip(0.0, 24.0 * 30.0)
                    .alias("dt_valid_hours")
              )
          )
    )


def get_global_minmax_validtime(files):
    """
    Computes global valid-time coverage across a list of parquet inputs.

    Iterates over files, lazily scans each one and extracts min/max of validtime_dt.
    Aggregates these into a single overall (vt_min, vt_max) spanning *all* files.

    - Skips files where validtime_dt min/max cannot be inferred (e.g. empty partitions).
    - Uses streaming collect to keep memory usage low.
    - Raises RuntimeError if no usable min/max values are found in any file.

    Returns:
        (vt_min, vt_max): earliest and latest validtime_dt across all provided files.
    """
    vt_min = None
    vt_max = None
    for fp in files:
        lf = scan_base(fp).select("validtime_dt")
        df = lf.select(
            pl.col("validtime_dt").min().alias("mn"),
            pl.col("validtime_dt").max().alias("mx"),
        ).collect(engine="streaming")
        mn = df["mn"][0]
        mx = df["mx"][0]
        if mn is None or mx is None:
            continue
        vt_min = mn if vt_min is None else min(vt_min, mn)
        vt_max = mx if vt_max is None else max(vt_max, mx)
    if vt_min is None or vt_max is None:
        raise RuntimeError("Could not infer validtime_dt min/max from files.")
    return vt_min, vt_max


def split_ranges_from_files(files, test_days=365, val_days=90):
    _, vt_max = get_global_minmax_validtime(files)
    test_start = vt_max - timedelta(days=test_days)
    val_start  = test_start - timedelta(days=val_days)
    return val_start, test_start


# -------------------------
# PREPARE STEP (one-time)
# -------------------------
def prepare_files_once(raw_files):
    """
    Creates prepared parquet files:
      - select needed columns
      - ensure *_dt exist
      - dedup + dt_hours
      - sort by (SID, analysistime_dt, validtime_dt)
    Output: PREP_DIR/ml_data_prep_*.parquet
    """
    needed = [STATION_ID_COL, "validtime_dt", "analysistime_dt", *FEATS, LABEL_OBS]
    for fp in raw_files:
        out = PREP_DIR / (Path(fp).name.replace("ml_data_full_", "ml_data_prep_"))
        if out.exists():
            continue

        df = scan_base(fp).select(needed).collect(engine="streaming")
        if df.height == 0:
            continue

        """df = add_dt_hours(dedup_per_station_validtime(df))
        if df.height == 0:
            continue"""

        df = df.sort([STATION_ID_COL, "analysistime_dt", "validtime_dt"])

        df= add_dt_valid_hours(df)
        if df.height == 0:
            continue

        df.write_parquet(out)
        print("Wrote", out)


# -------------------------
# Fit stats (TRAIN only) on prepared files
# -------------------------
def fit_stats_strict_train(prep_files, feats, time_max, sample_per_file, seed):
    """
    Fits per-feature mean and standard deviation using TRAIN-only data.

    For each prepared parquet file:
      - keeps only rows with validtime_dt < time_max (i.e., strictly before the split point),
      - selects only the requested feature columns,
      - reads at most sample_per_file*5 rows (cheap cap before sampling),
      - then (if needed) uniformly samples exactly sample_per_file rows without replacement
        for a consistent per-file contribution.

    Accumulates per-feature counts, sum, and sum-of-squares while ignoring NaNs, and
    computes:
      mean = sum / n
      var  = max(E[x^2] - mean^2, 1e-12)
      sd   = sqrt(var)

    Safety defaults:
      - If a feature has no valid samples at all: mean=0.0, sd=1.0
      - If computed sd becomes 0.0: force sd=1.0 (prevents divide-by-zero in scaling)

    Returns:
        mu: dict {feature: mean}
        sd: dict {feature: std_dev}
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
# Dataset on PREP files (no more dedup/sort/dt_hours each epoch)
# -------------------------
class PrepStreamSeqDataset(IterableDataset):
    """
    Iterable, streaming dataset that reads from *prepared* parquet files and yields
    sequential model inputs without doing per-epoch preprocessing.

    Designed for pipelines where heavy steps (sorting, de-duplication, dt_hours
    computation) were already baked into PREP files, so training can focus on cheap
    filtering + normalization.

    Stores:
      - a time window [time_min, time_max) used later when scanning/filtering rows,
      - feature column names (feats),
      - label/aux columns (label_obs, temp_fc) and station id column (sid_col),
      - sequence length (seq_len),
      - normalization statistics (mu, sd) for z-scoring,
      - input dimensionality D = 2*len(feats) + 1:
          for each feature -> (z-scored value, missingness flag),
          plus one extra channel for dt_hours.
    """
    def __init__(self, files, time_min, time_max, feats, label_obs, temp_fc, sid_col, seq_len, mu, sd):
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
        """
        Converts a Polars DataFrame block into a dense float32 design matrix.

        For each feature f in self.feats:
          - pulls the column to numpy,
          - builds a missingness mask (NaN -> missing),
          - creates a z-scored value array where missing values are set to 0,
          - appends two channels to X:
              [z_value, missing_flag].

        Then appends dt_hours as the final channel:
          - reads df["dt_hours"],
          - replaces NaNs with 0.0,
          - stores it in X[:, -1].

        Output shape:
            (num_rows_in_df, D) where D = 2*len(feats) + 1
        Output dtype:
            float32
        """
        X = np.empty((df.height, self.D), dtype=np.float32)
        k = 0
        for f in self.feats:
            col = df[f].to_numpy()
            miss = np.isnan(col)
            z = np.zeros_like(col, dtype=np.float32)
            if (~miss).any():
                z[~miss] = ((col[~miss] - self.mu[f]) / self.sd[f]).astype(np.float32, copy=False)
            X[:, k] = z
            X[:, k + 1] = miss.astype(np.float32)
            k += 2

        dt = df["dt_valid_hours"].to_numpy()
        dt = np.nan_to_num(dt, nan=0.0).astype(np.float32, copy=False)
        X[:, -1] = dt
        return X

    def __iter__(self):
        """
        Streams sequences from PREP parquet files and yields (Xseq, y, mask) samples.

        Maintains a per-station rolling buffer (deque) of the most recent feature rows,
        up to seq_len. As it iterates through time-ordered rows, it uses the *history
        before the current row* as the model input sequence, and the current row as
        the target.

        Worker sharding:
        - If running under a DataLoader with multiple workers, splits the file list so
            worker k reads every num_workers-th file (files[worker.id::worker.num_workers]).
        - In single-worker mode, reads all files.

        Per file:
        - Lazily scans parquet, selecting station id, validtime_dt, dt_hours, features,
            and the observation label column.
        - Filters to the configured time window [time_min, time_max).
        - Drops rows where either the observation label or forecast temperature (temp_fc)
            is null (ensures the target is computable).
        - Collects in streaming mode for lower memory overhead.
        - Transforms the block to a float32 design matrix with z-scores + missing flags
            + dt_hours via _transform_block().
        - Builds station ids (sid) and the regression target:
            y = label_obs - temp_fc  (float32), i.e. forecast error / correction.

        Sample generation (row-by-row):
        - For station s, fetch its history buffer hist.
        - If there is at least 1 previous timestep in hist:
            * stack hist into (T, D)
            * left-pad with zeros if T < seq_len
            * create a mask m of length seq_len:
                0 for padded timesteps, 1 for real timesteps
            * yield:
                Xseq: torch.FloatTensor [seq_len, D]
                y:    torch.FloatTensor scalar (current timestep target)
                m:    torch.FloatTensor [seq_len] (padding mask)
        - Finally, append the current timestep’s features to the station history buffer
            so it becomes available for future targets.

        Notes:
        - Assumes PREP files are already sorted by (sid, analysistime_dt, validtime_dt); the optional sort
            is kept commented out as a safety fallback.
        - The model never “sees” the current timestep features when predicting y[i];
            it only uses prior history (causal setup).
        """
        buf = defaultdict(lambda: deque(maxlen=self.seq_len))

        # --- Shard file list across workers ---
        worker = get_worker_info()
        if worker is None:
            files = self.files
        else:
            # Each worker gets every nth file
            files = self.files[worker.id::worker.num_workers]

        for fp in files:
            lf = (
                pl.scan_parquet(fp)
                .select([self.sid_col, "analysistime_dt", "validtime_dt", "dt_valid_hours", *self.feats, self.label_obs])
                .filter((pl.col("validtime_dt") >= self.time_min) & (pl.col("validtime_dt") < self.time_max))
                .filter(pl.col(self.label_obs).is_not_null() & pl.col(self.temp_fc).is_not_null())
            )

            df = lf.collect(engine="streaming")
            if df.height == 0:
                continue

            # Already sorted in prep, but keep safe
            # df = df.sort([self.sid_col, "analysistime_dt", "validtime_dt"])

            X_block = self._transform_block(df)
            sid = df[self.sid_col].to_numpy()
            #at_ns = df["analysistime_ns"].to_numpy()
            at = df["analysistime_dt"].to_list()
            y = (df[self.label_obs] - df[self.temp_fc]).to_numpy().astype(np.float32)

            for i in range(df.height):
                # To only use the current forecast to correct that forecast
                key = (int(sid[i]), at[i])
                hist = buf[key]

                if len(hist) >= 1:
                    hist_arr = np.stack(hist, axis=0)
                    if hist_arr.shape[0] < self.seq_len:
                        pad = self.seq_len - hist_arr.shape[0]
                        Xseq = np.vstack([np.zeros((pad, self.D), np.float32), hist_arr])
                        m = np.concatenate(
                            [np.zeros(pad, np.float32), np.ones(hist_arr.shape[0], np.float32)]
                        ).astype(np.float32)
                    else:
                        Xseq = hist_arr[-self.seq_len:]
                        m = np.ones(self.seq_len, np.float32)

                    yield torch.from_numpy(Xseq), torch.tensor(y[i], dtype=torch.float32), torch.from_numpy(m)

                hist.append(X_block[i])



def collate_batch(batch):
    """
    Custom DataLoader collate function for sequence samples.

    Takes a list of dataset items produced by the IterableDataset, where each item is:
    (Xseq, y, mask)

    Unzips the list into separate tuples (X, y, m) and stacks each along the new
    batch dimension (dim=0), producing:

      - X: [B, seq_len, D]   batched input sequences
      - y: [B] (or [B, 1])   batched targets, depending on how y was created upstream
      - m: [B, seq_len]      batched padding masks (0 = pad, 1 = real)

    Returns:
        (X_batch, y_batch, m_batch) as torch tensors ready for model forward pass.
    """
    X, y, m = zip(*batch)
    return torch.stack(X, 0), torch.stack(y, 0), torch.stack(m, 0)


class ThreadPrefetcher:
    """
    Wrap an iterable loader and prefetch items in a background thread.

    Items from `loader` are read asynchronously and stored in a bounded
    queue so the consumer can retrieve already-prepared batches while the
    next ones are being loaded.

    Args:
        loader: Iterable producing batches.
        max_prefetch: Maximum number of prefetched batches to buffer.
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
    Sequence-to-scalar LSTM model for predicting a bias/correction from past context.

    Architecture:
      - An LSTM encoder that processes an input sequence x of shape [B, T, in_dim],
        producing hidden states out of shape [B, T, hidden].
      - A small MLP "head" that maps the selected last valid hidden state to a single
        scalar output per sample (shape [B]).

    Padding handling:
      - Expects an optional pad_mask of shape [B, T] with 1 for real timesteps and
        0 for left-padding (as produced by the dataset).
      - Computes each sample’s effective sequence length as sum(pad_mask).
      - Selects the hidden state at the last real timestep (length-1) using gather(),
        so predictions are based on the most recent available history, not padded zeros.

    Notes:
      - LSTM uses dropout only when num_layers > 1 (PyTorch LSTM behavior).
      - Output is a single regression value per batch item (squeezed to [B]).
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
    """
    Evaluates a model on a DataLoader and returns global RMSE over all seen samples.

    Runs the model in eval mode with gradients disabled, iterating batches from `loader`.
    For each batch:
      - moves inputs/targets/masks to `device`,
      - computes predictions with padding-aware forward pass (model(X, m)),
      - accumulates total squared error (sum over all elements),
      - accumulates total sample count.

    Supports early stopping via `max_steps` to limit the number of batches processed
    (useful for quick validation checks or debugging).

    Returns:
        rmse = sqrt( sum((pred - y)^2) / (N + 1e-8) )

    Notes:
      - Uses a small epsilon in the denominator to avoid divide-by-zero if N=0.
      - This computes a *global* RMSE (not averaged per-batch).
    """
    model.eval()
    se = 0.0
    n = 0
    steps = 0
    with torch.no_grad():
        for X, y, m in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            pred = model(X, m)
            se += ((pred - y) ** 2).sum().item()
            n += y.numel()

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

    return math.sqrt(se / (n + 1e-8))


def train_epoch_amp(model, loader, opt, device, grad_clip=2.0, max_steps=None):
    """
    Trains the model for one epoch using (optional) CUDA automatic mixed precision (AMP).

    Runs the model in train mode and iterates batches from `loader`. For each batch:
      - moves data to `device`,
      - clears gradients (set_to_none=True is a small performance win),
      - computes predictions and MSE loss under autocast when on CUDA,
      - backpropagates using GradScaler to prevent FP16 underflow,
      - optionally clips gradient norm to `grad_clip`,
      - performs optimizer step via the scaler and updates scaling factor.

    Also accumulates a global RMSE over the processed batches (using the same global
    sum-of-squared-error / count logic as evaluation).

    Args:
        model: torch.nn.Module
        loader: DataLoader yielding (X, y, m)
        opt: optimizer (e.g., Adam)
        device: "cuda" or "cpu"
        grad_clip: if > 0, clips global grad norm to this value (stabilizes training)
        max_steps: optional cap on number of batches processed

    Returns:
        rmse = sqrt( sum((pred - y)^2) / (N + 1e-8) )

    Notes:
      - AMP is enabled only when device == "cuda"; on CPU it behaves like standard FP32.
      - RMSE is computed from predictions produced *during training mode* (dropout on),
        so it may be slightly noisier than eval RMSE.
    """
    model.train()
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    se = 0.0
    n = 0
    steps = 0

    for X, y, m in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            pred = model(X, m)
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
    """
    Creates and returns a new timestamped directory for storing a training run’s outputs.

    Builds a folder name like:
        optuna_lstm_YYYYMMDD_HHMMSS
    under the base path MODEL_DIR, using the current time from pl.datetime_now().

    Ensures the directory exists (creates parent directories as needed) and returns
    the resulting Path object, so callers can save checkpoints, logs, Optuna trials,
    and config artifacts in a unique per-run location.
    """
    run_dir = MODEL_DIR / f"optuna_lstm_{pl.datetime_now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def objective_cv(trial, prep_files, folds, mu, sd, ctx):
    """
    Optuna objective for time-based cross-validation over PREP files.

    Samples a set of LSTM + optimizer hyperparameters from `trial`, then evaluates
    them across multiple temporal folds. Each fold trains on all data strictly
    before the fold’s validation window, and validates on a contiguous time window.

    Hyperparameters tuned (search space):
      - hidden:       LSTM hidden size (categorical list)
      - num_layers:   number of stacked LSTM layers (1..3)
      - dropout:      dropout used in LSTM (between layers) and in the MLP head
      - lr:           AdamW learning rate (log-uniform)
      - wd:           AdamW weight decay (log-uniform)
      - grad_clip:    gradient norm clipping threshold

    Data / folds:
      - `folds` is an iterable of tuples like: (train_end, val_start, val_end)
        (train_end is not used directly here; training is defined as < val_start).
      - For each fold:
          * Training dataset:  time_min=1900-01-01, time_max=val_start
          * Validation dataset: time_min=val_start, time_max=val_end
        Both datasets read from the same PREP parquet files and use the same
        normalization stats (mu, sd).

    DataLoader setup:
      - IterableDataset => shuffle=False
      - num_workers/persistent_workers/prefetch_factor control parallel reading
      - pin_memory enabled on CUDA for faster host->device transfer
      - multiprocessing_context `ctx` is passed explicitly for worker start method

    Training loop (per fold):
      - Builds a fresh BiasLSTM + AdamW optimizer.
      - Repeats for up to TRIAL_MAX_EPOCHS:
          * trains for one epoch using AMP (train_epoch_amp),
            optionally limiting work via TRIAL_STEPS_PER_EPOCH
          * evaluates global RMSE on validation loader (global_rmse_from_loader),
            also step-limited
          * tracks best validation RMSE and applies early stopping:
              - improvement must exceed MIN_DELTA
              - stop after TRIAL_PATIENCE non-improving epochs

    Optuna reporting / pruning:
      - trial.report(best_val, step=global_step) reports intermediate best RMSE
        so Optuna can plot learning curves and make pruning decisions.
      - trial.should_prune() triggers pruning via raising TrialPruned.

    Cleanup:
      - Ensures model, optimizer, loaders, and datasets are deleted each fold,
        calls gc.collect(), and empties CUDA cache to reduce GPU memory buildup
        across folds/trials.

    Objective value:
      - Returns the mean of the best validation RMSE from each fold:
          mean(best_val_fold_0, best_val_fold_1, ...)
      - Optuna will try to minimize this returned float.
    """
    hidden     = trial.suggest_categorical("hidden", [160, 192, 256])
    num_layers = trial.suggest_int("num_layers", 1, 1)
    dropout    = trial.suggest_float("dropout", 0.0, 0.4)
    lr         = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    wd         = trial.suggest_float("wd", 1e-6, 5e-3, log=True)
    grad_clip  = trial.suggest_float("grad_clip", 0.5, 5.0)
    seq_len = trial.suggest_int("seq_len", 5, 24)

    in_dim = 2 * len(FEATS) + 1
    fold_bests = []
    global_step = 0

    for fold_id, (train_end, val_start, val_end) in enumerate(folds):
        ds_tr = PrepStreamSeqDataset(
            files=prep_files,
            time_min=pl.datetime(1900, 1, 1),
            time_max=train_end,
            feats=FEATS, label_obs=LABEL_OBS, temp_fc=TEMP_FC,
            sid_col=STATION_ID_COL, seq_len=seq_len,
            mu=mu, sd=sd,
        )
        ds_va = PrepStreamSeqDataset(
            files=prep_files,
            time_min=val_start,
            time_max=val_end,
            feats=FEATS, label_obs=LABEL_OBS, temp_fc=TEMP_FC,
            sid_col=STATION_ID_COL, seq_len=seq_len,
            mu=mu, sd=sd,
        )

        dl_tr = DataLoader(
            ds_tr, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS_OPTUNA, collate_fn=collate_batch,
            pin_memory=(DEVICE == "cuda"),
            persistent_workers=(NUM_WORKERS_OPTUNA > 0),
            prefetch_factor=PREFETCH_FINAL if NUM_WORKERS_OPTUNA > 0 else 2,
            multiprocessing_context=ctx,
        )
        dl_va = DataLoader(
            ds_va, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS_OPTUNA, collate_fn=collate_batch,
            pin_memory=(DEVICE == "cuda"),
            persistent_workers=(NUM_WORKERS_OPTUNA > 0),
            prefetch_factor=PREFETCH_FINAL if NUM_WORKERS_OPTUNA > 0 else 2,
            multiprocessing_context=ctx,
        )

        dl_tr = ThreadPrefetcher(dl_tr, max_prefetch=8)
        dl_va = ThreadPrefetcher(dl_va, max_prefetch=4)

        model = BiasLSTM(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=dropout).to(DEVICE)
        opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

        best_val = float("inf")
        bad = 0

        try:
            for ep in range(TRIAL_MAX_EPOCHS):
                _ = train_epoch_amp(
                    model, dl_tr, opt, DEVICE,
                    grad_clip=grad_clip,
                    max_steps=TRIAL_STEPS_PER_EPOCH
                )
                val_rmse = global_rmse_from_loader(
                    model, dl_va, DEVICE,
                    max_steps=TRIAL_STEPS_PER_EPOCH
                )

                print(f"[trial {trial.number:03d}] fold {fold_id} ep {ep:02d} val_rmse={val_rmse:.4f}")

                if val_rmse < best_val - MIN_DELTA:
                    best_val = val_rmse
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
            del model, opt, dl_tr, dl_va, ds_tr, ds_va
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fold_bests.append(best_val)

    return float(np.mean(fold_bests))



def run_optuna_and_save(prep_files, val_start, test_start, mu, sd, out_dir: Path):
    """
    Runs Optuna hyperparameter search and saves the resulting study + best parameters.

    Creates an Optuna study configured to minimize the cross-validation objective:
      - Sampler: TPESampler with a fixed seed for reproducible suggestions.
      - Pruner:  MedianPruner (with warmup) to stop underperforming trials early.

    Optimization:
      - Calls study.optimize(...) for N_TRIALS trials.
      - Uses a lambda wrapper to pass fixed arguments into objective_cv.
        (The objective is expected to return a validation RMSE-like scalar to minimize.)
      - Enables gc_after_trial to encourage cleanup between trials.

    Persistence / artifacts:
      - Saves the full Optuna Study object to `optuna_study.pkl` (pickle) so you can
        later inspect trials, plots, and parameter importance without rerunning.
      - Extracts the best trial parameters (study.best_params) and augments them with:
          * best_value_val_rmse: the best objective value found
          * trial_steps_per_epoch: the step cap used during tuning
          * trial_max_epochs: the max epochs used during tuning
        and writes them to `best_params.json` for easy reproducibility.

    Side effects:
      - Prints the best value and parameters to stdout.
      - Assumes `out_dir` exists (or is created upstream) and is writable.

    Returns:
        best_params: dict containing best hyperparameters + metadata for the tuning run.
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    study.optimize(
        lambda t: objective_cv(t, prep_files, val_start, test_start, mu, sd),
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    # --- save study + best params BEFORE final training ---
    with open(out_dir / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    best_params = dict(study.best_params)
    best_params["best_value_val_rmse"] = float(study.best_value)
    best_params["trial_steps_per_epoch"] = TRIAL_STEPS_PER_EPOCH
    best_params["trial_max_epochs"] = TRIAL_MAX_EPOCHS

    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("Optuna best:", study.best_value, study.best_params)
    print("Saved Optuna artifacts to:", out_dir)

    return best_params




# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Create prepared parquet files (one-time).")
    args = parser.parse_args()

    raw_files = list_files(RAW_GLOB)
    _set_mp_spawn()

    # ------------------------------------------------------------
    # 0) PREPARE step (one-time)
    # ------------------------------------------------------------
    if args.prepare:
        print("Preparing parquet files (dedup + dt_hours + sort)...")
        prepare_files_once(raw_files)
        print("Done. Prepared files in:", PREP_DIR)
        return

    prep_files = list_files(PREP_GLOB)

    # ------------------------------------------------------------
    # 1) Create run directory
    # ------------------------------------------------------------
    run_name = datetime.now().strftime("bias_lstm_%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Run dir:", run_dir)

    # ------------------------------------------------------------
    # 2) Rolling CV folds (Optuna) + final holdout split
    # ------------------------------------------------------------
    folds, test_start = split_trainval_test_rolling_from_files(
        prep_files,
        test_days=TEST_DAYS,
        n_folds=N_FOLDS,
        train_start_dt=None,
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
    # 3) Fit normalization stats (TRAIN ONLY for final training) + save
    # ------------------------------------------------------------
    print("Fitting normalization stats on a sample of TRAIN data (bounded RAM)...")
    mu, sd = fit_stats_strict_train(
        prep_files=prep_files,
        feats=FEATS,
        time_max=val_start_final,
        sample_per_file=STATS_SAMPLE_PER_FILE,
        seed=SEED,
    )
    stats_path = run_dir / "bias_lstm_stream_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"mu": mu, "sd": sd}, f)
    print("Saved stats to:", stats_path)

    in_dim = 2 * len(FEATS) + 1
    print("LSTM input dim:", in_dim)

    # ------------------------------------------------------------
    # 4) Loader factory (supports per-call num_workers for Optuna stability)
    # ------------------------------------------------------------
    def make_loaders(
        batch_size: int,
        train_end,
        val_start,
        val_end,
        seq_len,
        test_start=None,
        num_workers: int = NUM_WORKERS_FINAL,
        prefetch_factor: int = PREFETCH_FINAL,
    ):
        """
        train: [1900 .. train_end)
        val:   [val_start .. val_end)
        test:  [test_start .. 2200)   (optional)
        """

        ds_tr = PrepStreamSeqDataset(
            files=prep_files,
            time_min=pl.datetime(1900, 1, 1),
            time_max=train_end,
            feats=FEATS, label_obs=LABEL_OBS, temp_fc=TEMP_FC,
            sid_col=STATION_ID_COL, seq_len=seq_len,
            mu=mu, sd=sd,
        )
        ds_va = PrepStreamSeqDataset(
            files=prep_files,
            time_min=val_start,
            time_max=val_end,
            feats=FEATS, label_obs=LABEL_OBS, temp_fc=TEMP_FC,
            sid_col=STATION_ID_COL, seq_len=seq_len,
            mu=mu, sd=sd,
        )

        ds_ts = None
        if test_start is not None:
            ds_ts = PrepStreamSeqDataset(
                files=prep_files,
                time_min=test_start,
                time_max=pl.datetime(2200, 1, 1),
                feats=FEATS, label_obs=LABEL_OBS, temp_fc=TEMP_FC,
                sid_col=STATION_ID_COL, seq_len=seq_len,
                mu=mu, sd=sd,
            )

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            pin_memory=(DEVICE == "cuda"),
            num_workers=num_workers,
        )

        # only legal when multiprocessing is enabled
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
    # 5) Optuna with rolling CV
    #    IMPORTANT: use num_workers=0 in Optuna to avoid worker abort/reset issues
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
                # num_workers=0 => stable Optuna trials
                dl_tr, dl_va, _ = make_loaders(
                    bs,
                    train_end=train_end,
                    val_start=vs,
                    val_end=ve,
                    num_workers=NUM_WORKERS_OPTUNA,
                    prefetch_factor=PREFETCH_FINAL,
                    seq_len=seq_len
                )


                model = BiasLSTM(in_dim=in_dim, hidden=hidden, num_layers=num_layers, dropout=dropout).to(DEVICE)
                opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

                best_val = float("inf")
                bad = 0

                try:
                    for ep in range(TRIAL_MAX_EPOCHS):
                        _ = train_epoch_amp(
                            model, dl_tr, opt, DEVICE,
                            grad_clip=grad_clip,
                            max_steps=TRIAL_STEPS_PER_EPOCH
                        )
                        va_rmse = global_rmse_from_loader(
                            model, dl_va, DEVICE,
                            max_steps=TRIAL_STEPS_PER_EPOCH
                        )

                        # print epochs (what you asked for)
                        print(f"[trial {trial.number:03d}] fold {fold_id} ep {ep:02d} val_rmse={va_rmse:.4f}")

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
        print("Optuna best value (mean CV val RMSE):", study.best_value)
        print("Optuna best params:", best_params)

        with open(run_dir / "best_params.json", "w") as f:
            json.dump({"best_value": float(study.best_value), "best_params": best_params}, f, indent=2)
        with open(run_dir / "optuna_study.pkl", "wb") as f:
            pickle.dump(study, f)
        print("Saved Optuna artifacts to:", run_dir)

    # ------------------------------------------------------------
    # 6) Choose final hyperparameters (Optuna or defaults)
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
    # 7) FINAL training (full, real early stopping)
    # ------------------------------------------------------------
    dl_tr, dl_va, dl_ts = make_loaders(
        final_bs,
        train_end=val_start_final,     # train: < val_start_final
        val_start=val_start_final,     # val:   [val_start_final, val_end_final)
        val_end=val_end_final,
        test_start=test_start,         # test:  [test_start, ...)
        num_workers=NUM_WORKERS_FINAL,
        prefetch_factor=PREFETCH_FINAL,
        seq_len=final_seq_len,
    )

    model = BiasLSTM(in_dim=in_dim, hidden=final_hidden, num_layers=final_layers, dropout=final_dropout).to(DEVICE)
    opt = AdamW(model.parameters(), lr=final_lr, weight_decay=final_wd)

    best_val = float("inf")
    bad = 0
    best_state = None

    for ep in range(MAX_EPOCHS):
        tr_rmse = train_epoch_amp(model, dl_tr, opt, DEVICE, grad_clip=final_clip, max_steps=FINAL_STEPS_PER_EPOCH)
        va_rmse = global_rmse_from_loader(model, dl_va, DEVICE, max_steps=FINAL_VAL_STEPS)
        print(f"Epoch {ep:03d}: train_rmse={tr_rmse:.4f} val_rmse={va_rmse:.4f}")

        if va_rmse < best_val - MIN_DELTA:
            best_val = va_rmse
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop at ep={ep:03d}, best_val_rmse={best_val:.4f}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = run_dir / "bias_lstm_stream.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved model weights to:", model_path)

    ts_rmse_bias = global_rmse_from_loader(model, dl_ts, DEVICE)
    print(f"\nTEST global RMSE on bias (obs-fc): {ts_rmse_bias:.4f}")

    with open(run_dir / "results.json", "w") as f:
        json.dump({"best_val_rmse": float(best_val), "test_rmse_bias": float(ts_rmse_bias)}, f, indent=2)
    print("Saved results to:", run_dir / "results.json")


if __name__ == "__main__":
    main()

