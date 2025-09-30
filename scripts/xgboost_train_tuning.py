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


# Silence specific warning 
import warnings
warnings.filterwarnings(
    "ignore",
    message="The reported value is ignored because this `step`.*",
    category=UserWarning,
    module="optuna.trial._trial"
)

# Paths
PATH = Path.home() / "thesis_project" / "data" / "ml_data" / "ml_data_*.parquet"
LABEL_OBS = "obs_TA"
TEMP_FC   = "T2"
OUT = Path.home() / "thesis_project" / "figures"


# Features
weather = [
    "MSL", TEMP_FC, "D2", "U10", "V10", "LCC", "MCC", "SKT",
    "MX2T", "MN2T", "T_925", "T2_ENSMEAN_MA1", "T2_M1", "T_925_M1"
]
meta = ["leadtime", "lon", "lat", "elev", "sin_hod", "cos_hod", "sin_doy", "cos_doy"]
FEATS = weather + meta

# Hold out the last ~1y as final TEST
TEST_DAYS = 365

# Inside the pre-test period, use K rolling folds
N_FOLDS = 3
NUM_BOOST_ROUND = 10000
EARLY_STOP = 50
BIN = 256

# Optuna budget
N_TRIALS = 30      
TIMEOUT  = None    # (seconds) or keep None
RANDOM_SEED = 42


def load_dataset():
    """
    Helper function to read the dataset from the parquet file.
        - Reads the needed columns
        - Filters out the missing observed or predicted temperature values
        - Changes the analysistime to datetime type
    Returns:
        Dataframe of the data"""
    
    # Read the file and get needed columns
    lf = pl.scan_parquet(str(PATH)).select(FEATS + [LABEL_OBS, "analysistime"])

    # Change analysistime to datetime type
    lf = lf.with_columns(
        pl.col("analysistime").str.strptime(pl.Datetime, strict=False).alias("analysistime_dt")
    )

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
    """
    Function to split the dataset into train, validation and test sets
        Params: 
            df = Dataframe of the dataset
        Returns: 
            folds = Data folds for the train and validation
            df_test = Dataset for the testing """

    # Split last ~year as test set
    # Max analysistime (last analysistime)
    max_at = df["analysistime_dt"].max()

    # Get the start data for test set based on the number of test days wanted
    test_start = max_at - timedelta(days=TEST_DAYS)

    # Split the dataset into Train+Validation and Test sets 
    df_tv = df.filter(pl.col("analysistime_dt") < test_start)
    df_test = df.filter(pl.col("analysistime_dt") >= test_start)

    # Rolling folds on df_tv (Train and Validation sets)
    inits = (df_tv.select("analysistime_dt").unique().sort("analysistime_dt")["analysistime_dt"].to_list())
    if len(inits) < N_FOLDS + 1:
        raise ValueError("Not enough initializations for requested N_FOLDS.")

    # Create N_FOLDS folds: each fold uses everything up to a cutoff as train,
    # and the next chunk as validation. Rough equal splits by time.
    fold_edges = [int(round(i*len(inits)/(N_FOLDS+1))) for i in range(1, N_FOLDS+1)]
    folds = []
    prev_edge = 0
    for edge in fold_edges:
        val_start = inits[edge-1]
        train_mask = df_tv["analysistime_dt"] <  val_start
        val_mask   = df_tv["analysistime_dt"] >= val_start
        # keep validation reasonably sized: cap to next segment
        next_edge = min(edge + (fold_edges[1]-fold_edges[0] if len(fold_edges)>1 else edge), len(inits))
        if next_edge > edge:
            val_end = inits[next_edge-1]
            val_mask = (df_tv["analysistime_dt"] >= val_start) & (df_tv["analysistime_dt"] <= val_end)

        df_tr = df_tv.filter(train_mask)
        df_va = df_tv.filter(val_mask)
        if len(df_tr) == 0 or len(df_va) == 0:
            continue
        folds.append((df_tr, df_va))
        prev_edge = edge

    return folds, df_test

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

def make_objective_with_cold_weights(raw_folds): 
    """
    Function to create the Optuna objective for XGBoost training with cold weighting
        Params:
            raw_folds = A list of tuples containing training and validation data 
                        (df_tr, X_tr, y_tr, df_va, X_va, y_va) for cross-validation
        Returns: 
            objective = A function that Optuna can call for each trial. It builds 
                        weight-adjusted DMatrix objects, trains the XGBoost model, 
                        and returns the average validation score across folds
    """                      
    def objective(trial):
        """
        Function to run a single Optuna trial for XGBoost with cold weighting
            Params:
                trial = The Optuna trial object, which provides the current set of 
                        hyperparameters to evaluate
            Returns: 
                float = The average validation score (RMSE) across all folds for 
                        the given trial’s parameters
        """

        # XGBoost params for the trial
        params = make_params(trial)

        # Weight hyperparameters to optimize
        threshold_K = trial.suggest_float("w_threshold_K", 255.0, 268.0)   # ~ -18°C..-5°C
        alpha       = trial.suggest_float("w_alpha", 0.02, 0.5, log=True)
        max_w       = trial.suggest_float("w_max_w", 3.0, 30.0)

        scores = []
        pruning_cb = optuna.integration.XGBoostPruningCallback(trial, "val-rmse")

        # Loop through the folds
        for (df_tr, X_tr, y_tr, df_va, X_va, y_va) in raw_folds:

            # Build weights for THIS trial 
            w_tr = make_weights_by_cold_params(df_tr, threshold_K, alpha, max_w)
            w_va = make_weights_by_cold_params(df_va, threshold_K, alpha, max_w)

            # Build the QuantileDMatrices for the train and validation fold 
            dtrain = xgb.QuantileDMatrix(X_tr, y_tr, weight=w_tr, max_bin=256)
            dval   = xgb.QuantileDMatrix(X_va, y_va, weight=w_va, ref=dtrain)

            # Train the XGBoost model
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=100,
                callbacks=[pruning_cb],
            )
            
            # Append the best score to the scores list
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
    study.optimize(make_objective_with_cold_weights(raw_folds),            
               n_trials=N_TRIALS, timeout=TIMEOUT, gc_after_trial=True)

    print("Best params:", study.best_trial.params)
    print("Best val RMSE (avg folds):", study.best_value)

    # Refit on full pre-test data (train/validation combined) with best parameters
    cutoff = df["analysistime_dt"].max() - timedelta(days=TEST_DAYS)
    df_full = df.filter(pl.col("analysistime_dt") < cutoff)

    # Get labels and features for the full (train/validation) dataset
    X_full, y_full = to_xy_bias(df_full)

    # Get the best parameters
    best_params = dict(study.best_trial.params)

    # Get the best weight parameters
    thrK = best_params.pop("w_threshold_K")                            
    a    = best_params.pop("w_alpha")                                     
    mw   = best_params.pop("w_max_w")                                      

    # Get weights for the full (train/validation) data
    w_full = make_weights_by_cold_params(df_full, thrK, a, mw)              

    # XGBoost parameters as before
    best_params.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "rmse",
        "seed": RANDOM_SEED,
    })
    best_params["max_bin"] = best_params.get("max_bin", 256)

    # Build the QuantileDMAtrix and train the model
    dfull = xgb.QuantileDMatrix(X_full, y_full, weight=w_full, max_bin=256)
    bst = xgb.train(
        best_params,
        dfull,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dfull, "full")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=50,
    )

    print("Refit best_iteration:", bst.best_iteration, "| best_score:", bst.best_score)

    # Save model
    bst.save_model("bias_model_tuned_weighted_best.json")

    # Quick test-year report (bias RMSE)
    X_test, y_test = to_xy_bias(df_test)
    yhat_test = bst.predict(xgb.DMatrix(X_test), iteration_range=(0, bst.best_iteration + 1))
    rmse_test_bias = float(np.sqrt(np.mean((y_test - yhat_test) ** 2)))
    print("TEST (last ~1y) bias RMSE:", rmse_test_bias)

    # Plot the parameter importance and the optimization history
    from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history

    fig1 = plot_param_importances(study)
    fig1.figure.savefig(OUT / "param_importances.png", dpi=200, bbox_inches="tight")

    fig2 = plot_optimization_history(study)
    fig2.figure.savefig(OUT / "optimization_history.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main() 