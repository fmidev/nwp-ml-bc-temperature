import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from pathlib import Path
from datetime import timedelta
import numpy as np
import polars as pl
import xgboost as xgb
import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, UTC


# Silence specific warning 
import warnings
warnings.filterwarnings(
    "ignore",
    message="The reported value is ignored because this `step`.*",
    category=UserWarning,
    module="optuna.trial._trial"
)

# Paths
PATH = Path.home() / "thesis_project" / "data" / "ml_data_full" / "ml_data_full_*.parquet"
LABEL_OBS = "obs_TA"
TEMP_FC   = "T2"
OUT = Path.home() / "thesis_project" / "figures"


# Features
weather = [
    "MSL", TEMP_FC, "D2", "U10", "V10", "LCC", "MCC", "SKT",
    "MX2T", "MN2T", "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1"
]
meta = ["leadtime", "lon", "lat", "elev", "sin_hod", "cos_hod", "sin_doy", "cos_doy", "analysishour"]
FEATS = weather + meta

# Hold out the last ~1y as final TEST
TEST_DAYS = 365
TRAIN_START_DT = None

# Inside the pre-test period, use K rolling folds
N_FOLDS = 3
NUM_BOOST_ROUND = 10000
EARLY_STOP = 50
BIN = 256

# Optuna budget
N_TRIALS = 30      
TIMEOUT  = None    # (seconds) or keep None
RANDOM_SEED = 42

# Weighting controls
WEIGHT_MODE = "off"   # "off" | "cold" | "spatial" | "both"
STATION_ID_COL = "SID"    # your station id column name
APPLY_STATION_LIST_WEIGHTS = False


STATIONS_TO_UPWEIGHT = [
    "100917","100932","100896","101044","101065","101118","101237","101268","101339",
    "101398","101430","101537","101570","101608","101725","101794","101886","101928",
    "101932","10201","102033","102035"
]

# How much to upweight them 
STATION_LIST_FACTOR = 3.0

# safety clamps
STATION_LIST_MIN_W = 0.5
STATION_LIST_MAX_W = 10.0

# Strength / clipping for station balancing
SPATIAL_BETA  = 1.0       # 0=no effect, 1=inverse-count, >1 stronger
SPATIAL_MIN_W = 0.5
SPATIAL_MAX_W = 10.0


# -----------------------------------------
# Data loading and processing functions
# -----------------------------------------

def load_dataset():
    """
    Helper function to read the dataset from the parquet file.
        - Reads the needed columns
        - Filters out the missing observed or predicted temperature values
        - Changes the analysistime to datetime type
    Returns:
        Dataframe of the data"""
    
    # Read the file and get needed columns
    lf = pl.scan_parquet(str(PATH)).select(FEATS + [LABEL_OBS, "analysistime", STATION_ID_COL, "validtime"])

    # Change analysistime to datetime type
    lf = lf.with_columns(
        pl.col("analysistime").str.strptime(pl.Datetime, strict=False).alias("analysistime_dt"),
        pl.col("validtime").str.strptime(pl.Datetime, strict=False).alias("validtime_dt")
    )

    # Change analysishour to integer
    lf = lf.with_columns(pl.col("analysishour").cast(pl.Int8))

    # Filter out the rows with missing temperature values
    df = (
        lf.filter(pl.col(LABEL_OBS).is_not_null() & pl.col(TEMP_FC).is_not_null())
          .collect(engine="streaming")
    )
    return df

def to_xy_bias(df_pl):
    """
    Helper function to get the bias in order to work with bias and not the temperatures
        - Cast the features to float32 to save memory
        - Make the bias label (observed temperature - predicted temperature) (also as float32)
        Params: 
            df_pl = Dataframe of the features and observations
        Returns: 
            The formatted features (X) and labels (y)
    """

    # Cast features to float32 to save memory
    X = df_pl.select(FEATS).to_numpy().astype(np.float32, copy=False)

    # Create the bias labels and cast to float32
    y = (df_pl[LABEL_OBS] - df_pl[TEMP_FC]).cast(pl.Float32).to_numpy()
    return X, y

def split_trainval_test(df):

    # Last ~year as test set
    max_vt = df["validtime_dt"].max()
    test_start = max_vt - timedelta(days=TEST_DAYS)

    if TRAIN_START_DT is None:
        min_tv = df["validtime_dt"].min()

        df_tv = df.filter(
            (pl.col("validtime_dt") < test_start) &
            (pl.col("validtime_dt") >= min_tv)
        )
    else:   
        # Train+Val before test, then restrict to >= 2019-01-01
        df_tv = df.filter(
            (pl.col("validtime_dt") < test_start) &
            (pl.col("validtime_dt") >= TRAIN_START_DT)
        )
    df_test = df.filter(pl.col("validtime_dt") >= test_start)

    # Rolling folds on restricted df_tv
    inits = (df_tv.select("validtime_dt")
                  .unique()
                  .sort("validtime_dt")["validtime_dt"].to_list())
    if len(inits) < N_FOLDS + 1:
        raise ValueError("Not enough initializations for requested N_FOLDS after applying TRAIN_START_DT.")

    fold_edges = [int(round(i*len(inits)/(N_FOLDS+1))) for i in range(1, N_FOLDS+1)]
    folds = []
    for i, edge in enumerate(fold_edges, start=1):
        val_start = inits[edge-1]
        next_edge = min(edge + (fold_edges[1]-fold_edges[0] if len(fold_edges)>1 else edge), len(inits))
        val_mask = pl.col("validtime_dt") >= val_start
        if next_edge > edge:
            val_end = inits[next_edge-1]
            val_mask = (pl.col("validtime_dt") >= val_start) & (pl.col("validtime_dt") <= val_end)
        df_tr = df_tv.filter(pl.col("validtime_dt") < val_start)
        df_va = df_tv.filter(val_mask)
        if len(df_tr) and len(df_va):
            folds.append((df_tr, df_va))

    return folds, df_test

# -----------------------------------------
# Weight functions
# -----------------------------------------

def _normalize_and_clip_weights(w, min_w=0.2, max_w=50.0):
    w = np.asarray(w, dtype=np.float32)
    m = np.mean(w)
    if m > 0:
        w = w / m
    return np.clip(w, min_w, max_w).astype(np.float32, copy=False)


def make_weights_by_station_list(df_pl, factor=STATION_LIST_FACTOR, default=1.0):
    """
    Assign a multiplicative weight 'factor' to the SIDs listed in STATIONS_TO_UPWEIGHT,
    and 'default' (usually 1.0) to all others. Then normalize to mean 1 and clip.
    """
    assert STATION_ID_COL in df_pl.columns, f"{STATION_ID_COL} not in dataframe"

    # Build a tiny table SID -> factor
    upw = {str(sid): float(factor) for sid in STATIONS_TO_UPWEIGHT}
    wdf = pl.DataFrame({
        STATION_ID_COL: list(upw.keys()),
        "w_list": list(upw.values()),
    })

    # Left-join onto your rows to attach per-row weight; fill missing with default
    df_sid = df_pl.select(pl.col(STATION_ID_COL).cast(pl.Utf8).alias(STATION_ID_COL))
    joined = df_sid.join(wdf, on=STATION_ID_COL, how="left").with_columns(
        pl.col("w_list").fill_null(default)
    )
    w = joined["w_list"].to_numpy().astype(np.float32, copy=False)

    return _normalize_and_clip_weights(w, STATION_LIST_MIN_W, STATION_LIST_MAX_W)



def make_weights_by_station_balance(df_pl):
    """
    Upweight under-represented stations using inverse frequency per station:
      w_i ∝ (median_count / count_station)^SPATIAL_BETA
    """
    assert STATION_ID_COL in df_pl.columns, f"{STATION_ID_COL} not in dataframe"

    # counts per station (Polars is fast)
    counts = (df_pl.group_by(STATION_ID_COL)
                    .len()
                    .rename({"len": "cnt"}))

    # attach counts back to rows (left join on SID)
    df_join = df_pl.select(STATION_ID_COL).join(counts, on=STATION_ID_COL, how="left")
    cnt = df_join["cnt"].to_numpy().astype(np.float32, copy=False)

    # inverse-frequency (power SPATIAL_BETA), normalized and clipped
    w = (np.median(cnt) / np.maximum(cnt, 1.0)) ** SPATIAL_BETA
    return _normalize_and_clip_weights(w, SPATIAL_MIN_W, SPATIAL_MAX_W)


def make_weights_by_cold_params(df_pl, threshold_K, alpha, max_w, obs_col = LABEL_OBS):
    """Compute sample weights for cold temperatures.
            Params:
                df_pl = Dataframe of the data
                threshold_K = Threshold for cold temperature (Kelvins) weights assigned for temperatures under the threshold
                alpha = Slope parameter; how quickly the weight grows
                max_w = Maximum weight; prevents weights from getting too large
                obs_col = Name of the observation column
            Returns: The weight
    """
    # Cast observations to float32 to save memory
    T = df_pl[obs_col].cast(pl.Float32).to_numpy()

    # Create the weight
    w = 1.0 + alpha * np.maximum(0.0, threshold_K - T)
    return np.clip(w, 1.0, max_w).astype(np.float32, copy=False)


def build_weights(df_pl, trial=None):
    if WEIGHT_MODE == "off" and not APPLY_STATION_LIST_WEIGHTS:
        return None

    w_parts = []

    # Manual list (optional)
    if APPLY_STATION_LIST_WEIGHTS:
        w_parts.append(make_weights_by_station_list(df_pl))

    # Existing spatial (per-station inverse-frequency) / cold logic
    if WEIGHT_MODE in ("spatial", "both"):
        w_parts.append(make_weights_by_station_balance(df_pl))

    if WEIGHT_MODE in ("cold", "both"):
        if trial is not None:
            thrK = trial.suggest_float("w_threshold_K", 255.0, 268.0)
            a    = trial.suggest_float("w_alpha", 0.02, 0.5, log=True)
            mw   = trial.suggest_float("w_max_w", 3.0, 30.0)
        else:
            thrK, a, mw = 262.0, 0.08, 10.0
        w_parts.append(make_weights_by_cold_params(df_pl, thrK, a, mw))

    if not w_parts:
        return None

    w = np.ones(len(df_pl), dtype=np.float32)
    for wi in w_parts:
        w *= wi
    return _normalize_and_clip_weights(w, 0.2, 50.0)




# -----------------------------------------
# XGBoost functions
# -----------------------------------------

def make_params(trial):
    """
    Function to create the trial parameters for the XGBoost model
        Params:
            trial = The Optuna trail object
        Returns: The parameters for the model
    """
    
    # Parameters for the XGBoost model
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",                 
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "max_bin": 256,
        "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
        "seed": RANDOM_SEED,
    }
    return params

def make_objective_with_weights(raw_folds):
    def objective(trial):
        params = make_params(trial)
        scores = []
        pruning_cb = optuna.integration.XGBoostPruningCallback(trial, "val-rmse")

        for (df_tr, X_tr, y_tr, df_va, X_va, y_va) in raw_folds:
            w_tr = build_weights(df_tr, trial=trial)
            w_va = build_weights(df_va, trial=trial)

            if w_tr is not None:
                dtrain = xgb.QuantileDMatrix(X_tr, y_tr, weight=w_tr, max_bin=params.get("max_bin", 256))
                dval   = xgb.QuantileDMatrix(X_va, y_va, weight=w_va, ref=dtrain)
            else:
                dtrain = xgb.QuantileDMatrix(X_tr, y_tr, max_bin=params.get("max_bin", 256))
                dval   = xgb.QuantileDMatrix(X_va, y_va, ref=dtrain)

            bst = xgb.train(
                params, dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=100,
                callbacks=[pruning_cb],
            )
            scores.append(bst.best_score)

        return float(np.mean(scores))
    return objective



def main():

    np.random.seed(RANDOM_SEED)
    BIN = 256

    # Load the data and split into train/validation and test sets 
    # and create folds for the train/validation set
    df = load_dataset()
    folds, df_test = split_trainval_test(df)

    # Loop through the folds 
    raw_folds = []                                                         
    for (df_tr, df_va) in folds:

        # Get the bias labels and features for train and validation set
        X_tr, y_tr = to_xy_bias(df_tr)
        X_va, y_va = to_xy_bias(df_va)

        # Append the folds
        raw_folds.append((df_tr, X_tr, y_tr, df_va, X_va, y_va))

        # Set the pruner for the Optuna trial
        pruner = optuna.pruners.HyperbandPruner(min_resource=150, reduction_factor=3)

        # Create a Optuna study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=pruner,
        )

    # Optimize the study for the folds
    study.optimize(make_objective_with_weights(raw_folds), n_trials=N_TRIALS, timeout=TIMEOUT, gc_after_trial=True)


    print("Best params:", study.best_trial.params)
    print("Best val RMSE (avg folds):", study.best_value)

    # Refit on full pre-test data (train/validation combined) with best parameters
    cutoff = df["validtime_dt"].max() - timedelta(days=TEST_DAYS)
    df_full = df.filter(pl.col("validtime_dt") < cutoff)

    # Get labels and features for the full (train/validation) dataset
    X_full, y_full = to_xy_bias(df_full)

    # Get the best parameters
    best_params = dict(study.best_trial.params)

    # XGBoost parameters as before
    best_params.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "rmse",
        "seed": RANDOM_SEED,
    })
    best_params["max_bin"] = best_params.get("max_bin", 256)

    weight_hparams = {}
    if WEIGHT_MODE in ("cold", "both"):
        for k in ("w_threshold_K", "w_alpha", "w_max_w"):
            if k in best_params:
                weight_hparams[k] = best_params.pop(k)

    # -------------------------------
    # 2) Build a metadata dict based on WEIGHT_MODE (with explicit elif for "off")
    # -------------------------------
    weight_meta = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": WEIGHT_MODE,
        "station_id_col": STATION_ID_COL,
    }

    if WEIGHT_MODE in ("cold", "both"):
        weight_meta["cold_params"] = weight_hparams  # may be empty if none popped

    elif WEIGHT_MODE == "off":
        # Nothing from cold/spatial, but we may still apply manual station list weights
        pass  # handled below by APPLY_STATION_LIST_WEIGHTS

    if WEIGHT_MODE in ("spatial", "both"):
        weight_meta["spatial_params"] = {
            "beta": float(SPATIAL_BETA),
            "min_w": float(SPATIAL_MIN_W),
            "max_w": float(SPATIAL_MAX_W),
        }

    # Manual station list weights can be applied under ANY mode, including "off"
    if APPLY_STATION_LIST_WEIGHTS:
        weight_meta["station_list"] = {
            "enabled": True,
            "factor": float(STATION_LIST_FACTOR),
            "min_w": float(STATION_LIST_MIN_W),
            "max_w": float(STATION_LIST_MAX_W),
            "stations": list(STATIONS_TO_UPWEIGHT),
        }
    else:
        weight_meta["station_list"] = {"enabled": False}

    # Persist metadata (always)
    with open("weight_meta_2019_sid.json", "w") as f:
        json.dump(weight_meta, f, indent=4)    

    # Optional: persist for reproducibility
    if weight_hparams:
        with open("weight_hparams_sid_2019.json", "w") as f:
            json.dump(weight_hparams, f, indent=4)

    # --- build refit weights based on the chosen mode ---
    def build_weights_refit(df_pl):
        w_parts = []

        if APPLY_STATION_LIST_WEIGHTS:
            w_parts.append(make_weights_by_station_list(df_pl))

        if WEIGHT_MODE == "spatial":
            w_parts.append(make_weights_by_station_balance(df_pl))
        elif WEIGHT_MODE == "cold":
            thrK = weight_hparams.get("w_threshold_K", 262.0)
            a    = weight_hparams.get("w_alpha", 0.08)
            mw   = weight_hparams.get("w_max_w", 10.0)
            w_parts.append(make_weights_by_cold_params(df_pl, thrK, a, mw))
        elif WEIGHT_MODE == "both":
            w_parts.append(make_weights_by_station_balance(df_pl))
            thrK = weight_hparams.get("w_threshold_K", 262.0)
            a    = weight_hparams.get("w_alpha", 0.08)
            mw   = weight_hparams.get("w_max_w", 10.0)
            w_parts.append(make_weights_by_cold_params(df_pl, thrK, a, mw))

        if not w_parts:
            return None
        w = np.ones(len(df_pl), dtype=np.float32)
        for wi in w_parts:
            w *= wi
        return _normalize_and_clip_weights(w, 0.2, 50.0)


    w_full = build_weights_refit(df_full)

    # --- build QuantileDMatrix with/without weights ---
    max_bin_eff = best_params.get("max_bin", 256)
    if w_full is not None:
        dfull = xgb.QuantileDMatrix(X_full, y_full, weight=w_full, max_bin=max_bin_eff)
    else:
        dfull = xgb.QuantileDMatrix(X_full, y_full, max_bin=max_bin_eff)


    with open("best_params_full_new.json", "w") as f:
        json.dump(best_params, f, indent=4)

    bst = xgb.train(
        best_params,
        dfull,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dfull, "full")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=100,
    )

    print("Refit best_iteration:", bst.best_iteration, "| best_score:", bst.best_score)

    # Save model
    bst.save_model("bias_model_tuned_full_new.json")

    # Quick test-year report (bias RMSE)
    X_test, y_test = to_xy_bias(df_test)
    yhat_test = bst.predict(xgb.DMatrix(X_test), iteration_range=(0, bst.best_iteration + 1))
    rmse_test_bias = float(np.sqrt(np.mean((y_test - yhat_test) ** 2)))
    print("TEST (last ~1y) bias RMSE:", rmse_test_bias)

    # Plot the parameter importance and the optimization history
    from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history

    fig1 = plot_param_importances(study)
    fig1.figure.savefig(OUT / "param_importances_tuned_full_new.png", dpi=200, bbox_inches="tight")

    fig2 = plot_optimization_history(study)
    fig2.figure.savefig(OUT / "optimization_history_tuned_full_new.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main() 