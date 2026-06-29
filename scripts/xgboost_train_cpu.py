import argparse
import gc
import glob
import json
import re
from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

# This training entrypoint is intentionally narrow: it expects already-cleaned
# monthly parquet files and trains CPU XGBoost models through the external-memory
# API so we do not have to materialize the full dataset in RAM at once.

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False


# -----------------------------------------------------------------------------
# Clean XGBoost dataset columns
# -----------------------------------------------------------------------------

TEMP_FC = "T2"
DEW_FC = "D2"
TMAX_FC = "MX2T"
TMIN_FC = "MN2T"

DEFAULT_TARGET_COL = "target_bias_TA"

# Map each supported learning target to the forecast field it corrects. This is
# metadata for logs/reports; the training label itself still comes from
# --target-col in the parquet files.
KNOWN_TARGET_TO_BASE_FC = {
    "target_bias": TEMP_FC,
    "target_bias_TA": TEMP_FC,
    "target_bias_TD": DEW_FC,
    "target_bias_TMAX": TMAX_FC,
    "target_bias_TMIN": TMIN_FC,
}

# Canonical output model filenames. Operational inference scripts expect these
# stable names instead of the older verbose training artifact names.
MODEL_BASENAME_BY_TARGET = {
    "target_bias": "XGB_T2",
    "target_bias_TA": "XGB_T2",
    "target_bias_TD": "XGB_TD",
    "target_bias_TMAX": "XGB_TMAX",
    "target_bias_TMIN": "XGB_TMIN",
}

weather = [
    "MSL", TEMP_FC, DEW_FC, "U10", "V10", "LCC", "MCC", "SKT",
    TMAX_FC, TMIN_FC, "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1",
]

meta = [
    "leadtime", "lon", "lat", "elev",
    "sin_hod", "cos_hod", "sin_doy", "cos_doy",
    "analysishour",
]

# Keep feature order fixed so training, validation, and later inference all feed
# XGBoost the columns in the same order.
FEATS = weather + meta

RANDOM_SEED = 42
DEFAULT_MAX_BIN = 128
DEFAULT_NUM_BOOST_ROUND = 3000
DEFAULT_EARLY_STOP = 50
DEFAULT_VALID_DAYS = 60
DEFAULT_TEST_DAYS = 365
DEFAULT_N_TRIALS = 30
DEFAULT_TUNE_NUM_BOOST_ROUND = 1000

DEFAULT_XGB_PARAMS = {
    "max_depth": 7,
    "min_child_weight": 2.0,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "reg_lambda": 43.136380049474496,
    "reg_alpha": 0.05,
    "gamma": 0.08,
    "eta": 0.02,
    "max_bin": DEFAULT_MAX_BIN,
}


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train XGBoost using CPU hist + external-memory DataIter from cleaned "
            "monthly parquet files. Supports multi-target clean datasets via --target-col."
        )
    )

    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)

    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        type=str,
        help=(
            "Target column to train on. Examples: target_bias_TA, target_bias_TD, "
            "target_bias_TMAX, target_bias_TMIN. Default: target_bias_TA. "
            "The old single-target column target_bias is also supported if present."
        ),
    )

    parser.add_argument(
        "--params-json",
        default=None,
        type=str,
        help="Optional JSON file with fixed/base XGBoost parameters.",
    )

    parser.add_argument("--num-boost-round", default=DEFAULT_NUM_BOOST_ROUND, type=int)
    parser.add_argument("--early-stop", default=DEFAULT_EARLY_STOP, type=int)
    parser.add_argument("--valid-days", default=DEFAULT_VALID_DAYS, type=int)
    parser.add_argument("--test-days", default=DEFAULT_TEST_DAYS, type=int)

    parser.add_argument(
        "--batch-size",
        default=250_000,
        type=int,
        help="Rows per iterator batch. Try 50000, 100000, 250000, or 500000.",
    )

    parser.add_argument(
        "--external-memory",
        action="store_true",
        help="Use CPU external-memory training. This script expects this flag.",
    )

    parser.add_argument(
        "--cpu-clean-input",
        action="store_true",
        help="Input parquet files already contain FEATS + target columns.",
    )

    parser.add_argument(
        "--no-valid-dmatrix",
        action="store_true",
        help="Skip validation DMatrix and early stopping. Not compatible with --tune.",
    )

    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip batched test evaluation after training.",
    )

    parser.add_argument(
        "--all-files-train",
        action="store_true",
        help="Use all input files for training. Useful for yearly/subset tests.",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="Directory for XGBoost external-memory cache. Default: <output>/xgb_cpu_cache.",
    )

    parser.add_argument(
        "--nthread",
        default=0,
        type=int,
        help="XGBoost CPU threads. 0 lets XGBoost decide. Example: 32.",
    )

    parser.add_argument(
        "--progress-every",
        default=50,
        type=int,
        help="Print DataIter progress every N yielded batches. Default: 50.",
    )

    parser.add_argument(
        "--weight-mode",
        default="off",
        choices=["off"],
        help="Only off is supported for cleaned CPU external-memory input.",
    )

    # -------------------------------------------------------------------------
    # Optuna tuning options
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Run Optuna tuning before final training. Tuning uses CPU external-memory "
            "dtrain/dvalid; it does not load the full data into RAM."
        ),
    )

    parser.add_argument(
        "--n-trials",
        default=DEFAULT_N_TRIALS,
        type=int,
        help=f"Number of Optuna trials. Default: {DEFAULT_N_TRIALS}.",
    )

    parser.add_argument(
        "--tune-timeout",
        default=None,
        type=int,
        help="Optional Optuna timeout in seconds.",
    )

    parser.add_argument(
        "--tune-num-boost-round",
        default=DEFAULT_TUNE_NUM_BOOST_ROUND,
        type=int,
        help=(
            "Maximum boosting rounds during Optuna trials. Usually lower than final "
            "--num-boost-round."
        ),
    )

    parser.add_argument(
        "--tune-early-stop",
        default=None,
        type=int,
        help="Early stopping rounds during tuning. Default: same as --early-stop.",
    )

    parser.add_argument(
        "--tune-max-depth-min",
        default=3,
        type=int,
        help="Optuna lower bound for max_depth. Default: 3.",
    )

    parser.add_argument(
        "--tune-max-depth-max",
        default=6,
        type=int,
        help="Optuna upper bound for max_depth. Default: 6.",
    )

    parser.add_argument(
        "--tune-fixed-max-bin",
        action="store_true",
        help=(
            "Keep max_bin fixed from --params-json/default. This script keeps max_bin "
            "fixed regardless because matrices are built once before tuning."
        ),
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def unique_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def safe_name(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


def list_parquet_files(input_path):
    p = Path(input_path)
    input_str = str(p)

    # Accept a directory of monthly files, a glob pattern, or a single file path.
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
    elif any(ch in input_str for ch in ["*", "?", "["]):
        files = sorted(Path(x) for x in glob.glob(input_str))
    else:
        files = [p]

    files = [x for x in files if x.exists()]

    if not files:
        raise FileNotFoundError(f"No parquet files found for input: {input_str}")

    return files


def month_from_filename(path):
    """
    Parse YYYY-MM from filenames such as:
        ml_data_clean_2024-01.parquet
        ml_data_2024-01.parquet
        ml_data_clean_202401.parquet
    """
    name = Path(path).name

    m = re.search(r"(\d{4})-(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.search(r"(\d{4})(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    raise ValueError(
        f"Could not infer YYYY-MM or YYYYMM from filename: {path}. "
        "Rename files like ml_data_clean_YYYY-MM.parquet."
    )


def get_max_month_datetime_from_files(parquet_files):
    months = [month_from_filename(p) for p in parquet_files]
    max_month = max(months)
    # Convert the last YYYY-MM seen in filenames into the final timestamp of that
    # month so validation/test cutoffs can be derived without opening all data.
    month_start = datetime.strptime(max_month + "-01", "%Y-%m-%d")
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return next_month - timedelta(seconds=1)


def split_files_by_month_from_name(parquet_files, train_end, test_start):
    train_files = []
    valid_files = []
    test_files = []

    train_end_month = train_end.strftime("%Y-%m")
    test_start_month = test_start.strftime("%Y-%m")

    for path in parquet_files:
        month = month_from_filename(path)

        # YYYY-MM strings are lexicographically sortable, so string comparison is
        # enough to bucket monthly files into train/valid/test windows.
        if month < train_end_month:
            train_files.append(path)
        elif month < test_start_month:
            valid_files.append(path)
        else:
            test_files.append(path)

    if not train_files:
        raise ValueError(
            "No train files selected by filename month split. "
            "For a small subset, use --all-files-train."
        )

    return train_files, valid_files, test_files


def get_dir_size_gib(path):
    path = Path(path)
    if not path.exists():
        return 0.0

    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    return total / 1024**3


def print_cache_size(label, cache_dir):
    try:
        print(f"{label}: cache_dir_size={get_dir_size_gib(cache_dir):.2f} GiB", flush=True)
    except Exception as exc:
        print(f"[WARN] Could not read cache size: {exc}", flush=True)


def load_fixed_params(params_json=None, nthread=0):
    if params_json is not None:
        with open(params_json, "r") as f:
            params = json.load(f)
    else:
        params = dict(DEFAULT_XGB_PARAMS)

    params = dict(params)
    # These core settings define the supported training mode for this script,
    # even when the caller supplies extra params from JSON.
    params.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "rmse",
        "seed": RANDOM_SEED,
    })

    params["max_bin"] = int(params.get("max_bin", DEFAULT_MAX_BIN))

    if int(nthread) > 0:
        params["nthread"] = int(nthread)

    # Remove GPU-only params if they accidentally appear in JSON.
    params.pop("sampling_method", None)

    return params


def make_trial_params(trial, base_params, args):
    """
    Build trial params for CPU hist external-memory tuning.

    max_bin is intentionally kept fixed because dtrain/dvalid are built once.
    Changing max_bin would require rebuilding the QuantileDMatrix for every trial,
    which would be very expensive on this dataset.
    """
    params = dict(base_params)

    params.update({
        "max_depth": trial.suggest_int(
            "max_depth",
            int(args.tune_max_depth_min),
            int(args.tune_max_depth_max),
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            5.0,
            200.0,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.4,
            1.0,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1.0,
            100.0,
            log=True,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-3,
            10.0,
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        ),
        "eta": trial.suggest_float(
            "eta",
            0.01,
            0.1,
            log=True,
        ),
    })

    # Keep fixed because matrix was already built.
    params["max_bin"] = int(base_params.get("max_bin", DEFAULT_MAX_BIN))

    return params


def check_required_columns(parquet_files, target_col):
    """
    Validate the first file schema before spending hours building matrices.
    """
    first = Path(parquet_files[0])
    schema_names = set(pq.ParquetFile(first).schema_arrow.names)

    missing_features = [c for c in FEATS if c not in schema_names]
    if missing_features:
        raise ValueError(
            f"Input file is missing feature columns needed by the model: {missing_features}\n"
            f"File checked: {first}"
        )

    if target_col not in schema_names:
        raise ValueError(
            f"Target column '{target_col}' was not found in input file: {first}\n"
            f"Available target-like columns: "
            f"{sorted([c for c in schema_names if c.startswith('target')])}"
        )


# -----------------------------------------------------------------------------
# CPU external-memory DataIter for cleaned parquet
# -----------------------------------------------------------------------------

class CleanParquetCpuIter(xgb.DataIter):
    """
    CPU external-memory iterator for cleaned monthly parquet files.

    Expected input columns:
        FEATS + [target_col]

    The iterator yields NumPy float32 arrays. It does not use CuPy, cuDF, Dask,
    or any GPU memory.
    """

    def __init__(
        self,
        parquet_files,
        feature_cols,
        target_col,
        batch_size,
        cache_prefix,
        progress_every=50,
    ):
        super().__init__(
            cache_prefix=cache_prefix,
            on_host=True,
            release_data=True,
        )

        self.parquet_files = list(parquet_files)
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.batch_size = int(batch_size)
        self.progress_every = int(progress_every)
        self.columns = unique_keep_order(self.feature_cols + [self.target_col])

        # DataIter is stateful: XGBoost will call reset() and then repeatedly
        # call next() until it returns 0.
        self._file_idx = 0
        self._batch_iter = None
        self._batch_count = 0
        self._yielded_rows = 0

    def reset(self):
        self._file_idx = 0
        self._batch_iter = None
        self._batch_count = 0
        self._yielded_rows = 0

    def _open_next_file(self):
        if self._file_idx >= len(self.parquet_files):
            return False

        path = self.parquet_files[self._file_idx]
        self._file_idx += 1

        print(
            f"[DataIter:{self.target_col}] opening file "
            f"{self._file_idx}/{len(self.parquet_files)}: {path}",
            flush=True,
        )

        pf = pq.ParquetFile(path)
        schema_names = set(pf.schema_arrow.names)

        missing = [c for c in self.columns if c not in schema_names]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        # Arrow yields row-group batches lazily, which is what makes the
        # external-memory pipeline work for large monthly parquet files.
        self._batch_iter = pf.iter_batches(
            batch_size=self.batch_size,
            columns=self.columns,
        )

        return True

    def next(self, input_data):
        while True:
            if self._batch_iter is None:
                opened = self._open_next_file()
                if not opened:
                    return 0

            try:
                batch = next(self._batch_iter)
            except StopIteration:
                self._batch_iter = None
                continue

            # CPU-side one-batch dataframe. This avoids GPU VRAM entirely.
            table = batch.to_pandas()

            # Target must be non-null. Features may contain NaN; XGBoost handles them.
            table = table.dropna(subset=[self.target_col])

            if len(table) == 0:
                del table, batch
                gc.collect()
                continue

            # Convert one batch at a time to NumPy and hand it directly to
            # XGBoost; we never accumulate multiple batches in memory here.
            X = table[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
            y = table[self.target_col].to_numpy(dtype=np.float32, copy=False)

            input_data(data=X, label=y)

            self._batch_count += 1
            self._yielded_rows += len(y)

            if self._batch_count % self.progress_every == 0:
                print(
                    f"[DataIter:{self.target_col}] yielded_batches={self._batch_count}, "
                    f"yielded_rows={self._yielded_rows:,}, "
                    f"current_file={self._file_idx}/{len(self.parquet_files)}",
                    flush=True,
                )

            del table, batch, X, y
            gc.collect()

            return 1


# -----------------------------------------------------------------------------
# Batched CPU evaluation
# -----------------------------------------------------------------------------

def evaluate_batched_cpu(bst, parquet_files, target_col, batch_size):
    sse = 0.0
    n = 0
    columns = unique_keep_order(FEATS + [target_col])

    # Mirror the training-side batching so test evaluation also stays bounded in
    # memory even for large held-out monthly files.
    for file_idx, path in enumerate(parquet_files, start=1):
        print(f"[Eval:{target_col}] opening file {file_idx}/{len(parquet_files)}: {path}", flush=True)

        pf = pq.ParquetFile(path)
        schema_names = set(pf.schema_arrow.names)
        missing = [c for c in columns if c not in schema_names]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            table = batch.to_pandas()
            table = table.dropna(subset=[target_col])

            if len(table) == 0:
                del table, batch
                gc.collect()
                continue

            X = table[FEATS].to_numpy(dtype=np.float32, copy=False)
            y = table[target_col].to_numpy(dtype=np.float32, copy=False)

            pred = bst.predict(xgb.DMatrix(X, feature_names=FEATS))
            err = y - pred

            sse += float(np.sum(err ** 2))
            n += len(y)

            del table, batch, X, y, pred, err
            gc.collect()

    if n == 0:
        return float("nan")

    return float(np.sqrt(sse / n))


# -----------------------------------------------------------------------------
# Optuna tuning on existing external-memory matrices
# -----------------------------------------------------------------------------

def tune_params_external_memory(args, base_params, dtrain, dvalid, output_dir, target_safe):
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed/importable, but --tune was set.")

    if dvalid is None:
        raise ValueError("--tune requires validation. Do not use --no-valid-dmatrix with --tune.")

    tune_early_stop = int(args.tune_early_stop or args.early_stop)

    print("Running Optuna tuning using CPU external-memory matrices...", flush=True)
    print("Target column:", args.target_col, flush=True)
    print("Optuna trials:", int(args.n_trials), flush=True)
    print("Optuna timeout:", args.tune_timeout, flush=True)
    print("Tune num_boost_round:", int(args.tune_num_boost_round), flush=True)
    print("Tune early stopping:", tune_early_stop, flush=True)
    print("NOTE: max_bin is fixed during tuning because matrices are already built.", flush=True)

    def objective(trial):
        params = make_trial_params(trial, base_params, args)

        # Reuse the already-built external-memory matrices for each trial. That
        # keeps tuning practical on large datasets, but also means max_bin must
        # stay fixed across trials.
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=int(args.tune_num_boost_round),
            evals=[(dvalid, "valid")],
            early_stopping_rounds=tune_early_stop,
            verbose_eval=False,
        )

        score = float(bst.best_score)
        trial.set_user_attr("best_iteration", int(bst.best_iteration))

        del bst
        gc.collect()

        return score

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=args.tune_timeout,
        gc_after_trial=True,
    )

    best_params = dict(base_params)
    best_params.update(study.best_trial.params)
    best_params["max_bin"] = int(base_params.get("max_bin", DEFAULT_MAX_BIN))

    best_iteration = study.best_trial.user_attrs.get("best_iteration")

    print("Best Optuna params:", study.best_trial.params, flush=True)
    print("Best Optuna valid RMSE:", study.best_value, flush=True)
    print("Best Optuna iteration:", best_iteration, flush=True)

    with open(output_dir / f"optuna_best_params_cpu_external_memory_{target_safe}.json", "w") as f:
        json.dump(best_params, f, indent=4)

    study_summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "target_col": args.target_col,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_trial_params": study.best_trial.params,
        "best_iteration": best_iteration,
        "n_trials": len(study.trials),
        "tune_num_boost_round": int(args.tune_num_boost_round),
        "tune_early_stop": tune_early_stop,
        "fixed_max_bin": int(base_params.get("max_bin", DEFAULT_MAX_BIN)),
    }

    with open(output_dir / f"optuna_summary_cpu_external_memory_{target_safe}.json", "w") as f:
        json.dump(study_summary, f, indent=4)

    return best_params, study_summary


# -----------------------------------------------------------------------------
# CPU external-memory training
# -----------------------------------------------------------------------------

def run_cpu_external_memory(args, input_path, output_dir):
    if not args.external_memory:
        raise ValueError("This script expects --external-memory for CPU external-memory training.")

    if not args.cpu_clean_input:
        raise ValueError("Use --cpu-clean-input with the cleaned parquet dataset.")

    if args.weight_mode != "off":
        raise NotImplementedError("Clean CPU external-memory path supports only --weight-mode off.")

    if args.tune and args.no_valid_dmatrix:
        raise ValueError("--tune requires validation. Remove --no-valid-dmatrix.")

    target_col = args.target_col
    target_safe = safe_name(target_col)
    model_basename = MODEL_BASENAME_BY_TARGET.get(target_col, f"XGB_{target_safe}")
    base_fc_col = KNOWN_TARGET_TO_BASE_FC.get(target_col)

    parquet_files = list_parquet_files(input_path)
    # Fail fast on schema problems before spending time building cache files and
    # QuantileDMatrices from many monthly parquet inputs.
    check_required_columns(parquet_files, target_col)

    print("Number of parquet files:", len(parquet_files), flush=True)
    print("Target column:", target_col, flush=True)
    print("Base forecast column for target:", base_fc_col, flush=True)

    max_vt = get_max_month_datetime_from_files(parquet_files)
    test_start = max_vt - timedelta(days=int(args.test_days))
    train_end = test_start - timedelta(days=int(args.valid_days))

    print("Max month approximation:", max_vt, flush=True)
    print("Validation start / train end:", train_end, flush=True)
    print("Test cutoff:", test_start, flush=True)

    # The default split is time-based by month name so evaluation stays strictly
    # out-of-sample relative to the most recent data.
    if args.all_files_train:
        train_files = parquet_files
        valid_files = []
        test_files = []
        print("Using all input files for training because --all-files-train is set.", flush=True)
        if args.tune or not args.no_valid_dmatrix:
            raise ValueError(
                "--all-files-train does not create validation files. Use --no-valid-dmatrix "
                "and do not use --tune, or run on a normal multi-month/full input split."
            )
    else:
        train_files, valid_files, test_files = split_files_by_month_from_name(
            parquet_files,
            train_end=train_end,
            test_start=test_start,
        )

    print("Train files:", len(train_files), flush=True)
    print("Valid files:", len(valid_files), flush=True)
    print("Test files:", len(test_files), flush=True)

    if train_files:
        print("First train file:", train_files[0], flush=True)
        print("Last train file:", train_files[-1], flush=True)
    if valid_files:
        print("First valid file:", valid_files[0], flush=True)
        print("Last valid file:", valid_files[-1], flush=True)
    if test_files:
        print("First test file:", test_files[0], flush=True)
        print("Last test file:", test_files[-1], flush=True)

    base_params = load_fixed_params(args.params_json, nthread=args.nthread)
    print("Base/fixed params:", base_params, flush=True)

    # XGBoost writes its external-memory cache here while constructing the
    # training and validation matrices.
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else output_dir / f"xgb_cpu_cache_{target_safe}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_iter = CleanParquetCpuIter(
        parquet_files=train_files,
        feature_cols=FEATS,
        target_col=target_col,
        batch_size=args.batch_size,
        cache_prefix=str(cache_dir / f"train_cache_{target_safe}"),
        progress_every=args.progress_every,
    )

    valid_iter = None
    if not args.no_valid_dmatrix:
        if not valid_files:
            raise ValueError(
                "Validation requested but no validation files were selected. "
                "Use --no-valid-dmatrix or adjust --valid-days."
            )

        valid_iter = CleanParquetCpuIter(
            parquet_files=valid_files,
            feature_cols=FEATS,
            target_col=target_col,
            batch_size=args.batch_size,
            cache_prefix=str(cache_dir / f"valid_cache_{target_safe}"),
            progress_every=args.progress_every,
        )

    print("Building CPU external-memory QuantileDMatrix for training...", flush=True)
    dtrain = xgb.ExtMemQuantileDMatrix(
        train_iter,
        max_bin=base_params.get("max_bin", DEFAULT_MAX_BIN),
    )

    print(
        f"dtrain shape: rows={dtrain.num_row():,}, cols={dtrain.num_col():,}",
        flush=True,
    )
    print_cache_size("After building dtrain", cache_dir)

    dvalid = None
    if valid_iter is not None:
        print("Building CPU external-memory QuantileDMatrix for validation...", flush=True)
        dvalid = xgb.ExtMemQuantileDMatrix(
            valid_iter,
            max_bin=base_params.get("max_bin", DEFAULT_MAX_BIN),
            ref=dtrain,
        )
        print(
            f"dvalid shape: rows={dvalid.num_row():,}, cols={dvalid.num_col():,}",
            flush=True,
        )
        print_cache_size("After building dvalid", cache_dir)
    else:
        print("Skipping validation DMatrix.", flush=True)

    optuna_summary = None
    final_params = dict(base_params)

    # Optional tuning happens after matrices are built so trials can reuse the
    # same cached data instead of rebuilding from parquet each time.
    if args.tune:
        final_params, optuna_summary = tune_params_external_memory(
            args=args,
            base_params=base_params,
            dtrain=dtrain,
            dvalid=dvalid,
            output_dir=output_dir,
            target_safe=target_safe,
        )

    params_filename = (
        f"fixed_params_cpu_external_memory_{target_safe}.json"
        if not args.tune
        else f"best_params_cpu_external_memory_{target_safe}.json"
    )
    model_filename = f"{model_basename}.json"

    with open(output_dir / params_filename, "w") as f:
        json.dump(final_params, f, indent=4)

    print("Training final model using CPU external memory...", flush=True)
    print("Final params:", final_params, flush=True)

    # Final training uses the tuned params if tuning ran; otherwise it uses the
    # fixed/default params loaded earlier.
    if dvalid is not None:
        bst = xgb.train(
            final_params,
            dtrain,
            num_boost_round=int(args.num_boost_round),
            evals=[(dvalid, "valid")],
            early_stopping_rounds=int(args.early_stop),
            verbose_eval=100,
        )
    else:
        bst = xgb.train(
            final_params,
            dtrain,
            num_boost_round=int(args.num_boost_round),
            evals=[],
            verbose_eval=100,
        )

    model_path = output_dir / model_filename
    bst.save_model(model_path)

    print("Saved model:", model_path, flush=True)
    print("Best iteration:", getattr(bst, "best_iteration", None), flush=True)
    print("Best validation score:", getattr(bst, "best_score", None), flush=True)

    rmse_test_bias = None
    if not args.skip_test_eval and test_files:
        print("Evaluating test files in CPU batches...", flush=True)
        rmse_test_bias = evaluate_batched_cpu(
            bst=bst,
            parquet_files=test_files,
            target_col=target_col,
            batch_size=args.batch_size,
        )
        print("TEST bias RMSE:", rmse_test_bias, flush=True)

    # Write a compact training report alongside the model so downstream runs can
    # inspect what target, split, params, and cache settings produced it.
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "external_memory": True,
        "device": "cpu",
        "clean_input": True,
        "tuning_enabled": bool(args.tune),
        "target_col": target_col,
        "base_forecast_col": base_fc_col,
        "feature_columns": FEATS,
        "weight_mode": args.weight_mode,
        "rmse_test_bias": rmse_test_bias,
        "best_iteration": getattr(bst, "best_iteration", None),
        "best_score": getattr(bst, "best_score", None),
        "model_file": model_filename,
        "generic_model_file": None,
        "params_file": params_filename,
        "max_month_approx": str(max_vt),
        "train_end": str(train_end),
        "test_cutoff": str(test_start),
        "valid_days": int(args.valid_days),
        "test_days": int(args.test_days),
        "batch_size": int(args.batch_size),
        "xgboost_version": xgb.__version__,
        "cache_dir": str(cache_dir),
        "cache_dir_size_gib": get_dir_size_gib(cache_dir),
        "no_valid_dmatrix": bool(args.no_valid_dmatrix),
        "skip_test_eval": bool(args.skip_test_eval),
        "all_files_train": bool(args.all_files_train),
        "optuna_summary": optuna_summary,
    }

    with open(output_dir / f"test_report_cpu_external_memory_{target_safe}.json", "w") as f:
        json.dump(report, f, indent=4)

    del bst, dtrain
    if dvalid is not None:
        del dvalid
    del train_iter, valid_iter
    gc.collect()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"XGBoost version: {xgb.__version__}")
    print(f"Optuna import available: {HAS_OPTUNA}")
    print(f"External memory enabled: {args.external_memory}")
    print(f"CPU clean input: {args.cpu_clean_input}")
    print(f"Tune enabled: {args.tune}")
    print(f"Target column: {args.target_col}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Validation days: {args.valid_days}")
    print(f"Test days: {args.test_days}")
    print(f"Num boost round: {args.num_boost_round}")
    print(f"Early stopping rounds: {args.early_stop}")
    print(f"No validation DMatrix: {args.no_valid_dmatrix}")
    print(f"Skip test eval: {args.skip_test_eval}")
    print(f"nthread: {args.nthread}")

    run_cpu_external_memory(args, args.input, output_dir)


if __name__ == "__main__":
    main()
