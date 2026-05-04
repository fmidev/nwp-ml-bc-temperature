import os
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
import joblib
import json
from sklearn.base import BaseEstimator, TransformerMixin

# ===========================================
#  Basic config
# ===========================================
# Limit the number of CPU threads used by NumPy / MKL-backed operations.
# This helps avoid accidental oversubscription on shared machines or HPC nodes.
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

# Resolve all project paths relative to the current user's home directory.
HOME = Path.home()

# Use GPU if available, otherwise fall back to CPU.
# The same DEVICE is used both for loading the model and for inference tensors.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory containing monthly/full parquet files named like:
#   ml_data_full_2024-01.parquet, ml_data_full_2024-02.parquet, ...
DATA_DIR = HOME / "thesis_project" / "data" / "ml_data_full"

# Directory containing all saved artifacts from the trained GNN run.
# These files must match the training run exactly: same graph, station mapping,
# preprocessing pipeline, architecture hyperparameters, and model weights.
RUN_DIR = HOME / "thesis_project" / "models" / "gnn_bias_correction" / "full_gnn_gat_lsm"
MODEL_PATH = RUN_DIR / "gnn_model.pt"
BEST_PARAMS_PATH = RUN_DIR / "graph_params.json"
PREPROC_PATH = RUN_DIR / "preproc.joblib"
STATIONS_PATH = RUN_DIR / "stations.parquet"
SID_MAP_PATH = RUN_DIR / "sid_to_idx.json"
EDGE_INDEX_PATH = RUN_DIR / "edge_index.pkl"
EDGE_ATTR_PATH = RUN_DIR / "edge_attr.pkl"

# Output directory for rows used by later metric scripts.
# The written parquet files contain raw forecast, observation, corrected forecast,
# and metadata columns needed for grouped metrics.
MODEL_TAG = "full_gnn_gat_lsm"
OUTDIR = HOME / "thesis_project" / "metrics" / "full_gnn_gat_lsm"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Column used to assign rows to yearly output files.
# Keeping this as a config variable makes it easy to switch to another split axis
# if needed later, but the current workflow uses validtime.
SPLIT_COLUMN = "validtime"

# ===========================================
#  Features / labels, same as training
# ===========================================
# Raw forecast temperature column. The model predicts a bias correction for this.
TEMP_FC = "T2"

# Observation column. During training/evaluation the target bias is obs_TA - T2.
LABEL_OBS = "obs_TA"

# Station identifier column. These values are mapped to graph node indices using
# sid_to_idx from the training run.
STATION_ID_COL = "SID"

# Feature columns must be in the same order and have the same names as in training.
# The fitted preprocessing pipeline expects exactly these input columns.
FEATS = [
    "T2", "SKT", "MX2T", "MN2T", "D2", "T_925", "MSL", "U10", "V10", "T2_M1", "T_925_M1",
    "LCC", "MCC", "T2_ENSMEAN_MA1", "sin_hod", "cos_hod", "sin_doy", "cos_doy",
    "analysishour", "leadtime", "lon", "lat", "elev",
]

# Metadata columns preserved in the final prediction rows.
ID = ["SID", "analysistime", "validtime", "leadtime"]

# Name of the corrected-temperature output column.
CORR_COL = f"corrected_{MODEL_TAG}"

# Glob pattern used when reading all available full-data parquet files.
PATH = DATA_DIR / "ml_data_full_*.parquet"

# ===========================================
#  Custom transformer, must match training
# ===========================================
class ZScoreFill0WithMask(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer used inside the saved preprocessing pipeline.

    The class definition must be present when loading `preproc.joblib`, because
    joblib needs to resolve the custom class that existed during training.

    For one numeric input feature, this transformer returns two output columns:
      1. the z-scored value, with missing values replaced by 0
      2. a binary missing-value indicator

    This lets the model distinguish between a true standardized value of 0 and
    an originally missing feature value that was filled with 0.
    """

    def __init__(self, feature_name):
        # Store the original feature name so get_feature_names_out can produce
        # stable output names compatible with the training pipeline.
        self.feature_name = feature_name
        self.mean_, self.std_ = 0.0, 1.0

    def fit(self, X, y=None):
        """Estimate mean and standard deviation from non-missing values only."""
        x = np.asarray(X, dtype=float).ravel()
        m = ~np.isnan(x)
        if m.any():
            mu, sd = x[m].mean(), x[m].std(ddof=0)
            # Avoid division by zero for constant-valued columns.
            self.mean_, self.std_ = float(mu), float(sd if sd != 0 else 1.0)
        return self

    def transform(self, X):
        """Transform values into [z_score, missing_mask] columns."""
        x = np.asarray(X, dtype=float).ravel()
        miss = np.isnan(x)
        z = np.zeros_like(x)
        z[~miss] = (x[~miss] - self.mean_) / self.std_
        return np.column_stack([z, miss.astype(float)])

    def get_feature_names_out(self, input_features=None):
        """Return output column names produced by this transformer."""
        return np.array([self.feature_name, f"{self.feature_name}_missing"])


# ===========================================
#  Helpers for data loading and splitting
# ===========================================
def safe_month(col):
    """
    Extract month number from a datetime-like string column.

    The input parquet files may store validtime as a string in some places, so
    this helper parses the column safely before taking `.dt.month()`.
    """
    return pl.col(col).str.strptime(pl.Datetime, strict=False, exact=False).dt.month()


def load_dataset(path, feat_cols, lab_col, fc_col, station_id_col):
    """
    Read training/evaluation-style data from parquet files.

    This mirrors the training loader closely:
      - reads only the required columns
      - parses analysistime and validtime into datetime helper columns
      - casts analysishour to a compact integer type
      - drops rows where either observation or raw forecast temperature is missing

    Returns
    -------
    pl.DataFrame
        Collected Polars DataFrame ready for splitting or snapshot creation.
    """
    lf = (
        pl.scan_parquet(str(path))
        .select(feat_cols + [lab_col, "analysistime", station_id_col, "validtime"])
        .with_columns(
            pl.col("analysistime").str.strptime(pl.Datetime, strict=False).alias("analysistime_dt"),
            pl.col("validtime").str.strptime(pl.Datetime, strict=False).alias("validtime_dt"),
            pl.col("analysishour").cast(pl.Int8),
        )
        .filter(pl.col(lab_col).is_not_null() & pl.col(fc_col).is_not_null())
    )
    return lf.collect(engine="streaming")


def split_trainval_test(df, TEST_DAYS=365, TRAIN_START_DT=None, N_FOLDS=3):
    """
    Split a time-indexed DataFrame into rolling train/validation folds and test.

    The final `TEST_DAYS` days, based on `validtime_dt`, are held out as test.
    The earlier period is split into rolling folds so that validation is always
    later than training. This function is retained because it matches the
    training code and may be useful for reproducible evaluation checks.

    Parameters
    ----------
    df : pl.DataFrame
        Input data containing `validtime_dt`.
    TEST_DAYS : int
        Number of final days reserved for the hold-out test period.
    TRAIN_START_DT : datetime-like or None
        Optional lower bound for the train/validation period.
    N_FOLDS : int
        Number of rolling validation folds to create.

    Returns
    -------
    folds : list[tuple[pl.DataFrame, pl.DataFrame]]
        List of `(train_df, val_df)` fold pairs.
    df_test : pl.DataFrame
        Final hold-out test set.
    """
    max_vt = df["validtime_dt"].max()
    test_start = max_vt - timedelta(days=TEST_DAYS)

    if TRAIN_START_DT is None:
        min_tv = df["validtime_dt"].min()
        df_tv = df.filter(
            (pl.col("validtime_dt") < test_start)
            & (pl.col("validtime_dt") >= min_tv)
        )
    else:
        df_tv = df.filter(
            (pl.col("validtime_dt") < test_start)
            & (pl.col("validtime_dt") >= TRAIN_START_DT)
        )

    df_test = df.filter(pl.col("validtime_dt") >= test_start)

    # Use unique validtime values as the time axis for fold boundaries.
    inits = (
        df_tv.select("validtime_dt")
        .unique()
        .sort("validtime_dt")["validtime_dt"]
        .to_list()
    )
    if len(inits) < N_FOLDS + 1:
        raise ValueError("Not enough initializations for requested N_FOLDS after applying TRAIN_START_DT.")

    # Fold edges are approximately equally spaced along the time axis.
    fold_edges = [int(round(i * len(inits) / (N_FOLDS + 1))) for i in range(1, N_FOLDS + 1)]
    folds = []
    for edge in fold_edges:
        val_start = inits[edge - 1]
        next_edge = min(
            edge + (fold_edges[1] - fold_edges[0] if len(fold_edges) > 1 else edge),
            len(inits),
        )

        # Validation is the contiguous block starting at val_start.
        val_mask = pl.col("validtime_dt") >= val_start
        if next_edge > edge:
            val_end = inits[next_edge - 1]
            val_mask = (pl.col("validtime_dt") >= val_start) & (pl.col("validtime_dt") <= val_end)

        # Training always uses only earlier times than the validation start.
        df_tr = df_tv.filter(pl.col("validtime_dt") < val_start)
        df_va = df_tv.filter(val_mask)
        if len(df_tr) and len(df_va):
            folds.append((df_tr, df_va))

    return folds, df_test


# ===========================================
#  Evaluation functions
# ===========================================
def masked_rmse(pred, y, m):
    """
    Compute RMSE only over valid station-node targets.

    In each graph snapshot, the graph contains all stations, but not every
    station necessarily has a valid observation at that validtime. `m` is a mask
    with 1 for valid target nodes and 0 for missing target nodes.
    """
    return torch.sqrt(((pred - y) ** 2 * m).sum() / (m.sum() + 1e-8))


@torch.no_grad()
def eval_rmse(m, ldr, dv):
    """
    Evaluate masked RMSE over a DataLoader of graph snapshots.

    Each batch contains one or more PyG `Data` snapshots. The function sends the
    batch to the chosen device, predicts node-level bias, computes masked RMSE,
    and averages the snapshot contributions by number of graph nodes.
    """
    m.eval()
    tot = 0.0
    n = 0
    for d in ldr:
        d = d.to(dv)
        p = m(d)
        l = masked_rmse(p, d.y, d.y_mask)
        tot += l.item() * d.num_nodes
        n += d.num_nodes
    return tot / n


def split_to_snapshots(df_pl, preproc, feat_cols, lab_col, fc_col, sid_to_idx, edge_index, edge_attr):
    """
    Convert a tabular Polars DataFrame into PyTorch Geometric graph snapshots.

    A snapshot corresponds to one `validtime`. Each snapshot contains the full
    station graph with N nodes, where N is the number of stations in `sid_to_idx`.
    Rows available at that validtime are inserted into the matching station-node
    positions. Nodes without a row at that time keep zero features and receive a
    target mask value of 0.

    The model target is forecast bias:
        bias = observation - raw_forecast

    During inference, the model predicts this bias and the corrected forecast is:
        corrected = raw_forecast + predicted_bias
    """
    N = len(sid_to_idx)
    node_idx_tensor = torch.arange(N, dtype=torch.long)
    data_list = []

    # Process one validtime at a time to avoid materializing a huge dense
    # [num_times, num_stations, num_features] tensor in memory.
    for _, group_df in df_pl.group_by("validtime", maintain_order=True):
        # Apply the exact fitted preprocessing pipeline from training.
        X_pd = group_df.select(feat_cols).to_pandas()
        X_np = preproc.transform(X_pd).astype(np.float32)

        # Training target for available rows: observed temperature minus raw T2.
        y_np = (group_df[lab_col] - group_df[fc_col]).to_numpy()

        # Convert station IDs into integer node positions in the fixed graph.
        sid = group_df["SID"].to_numpy()
        nodes = np.vectorize(sid_to_idx.get)(sid)

        # Dense per-snapshot arrays covering all graph nodes.
        # Only nodes present in this validtime group are filled below.
        x_t = np.zeros((N, X_np.shape[1]), np.float32)
        y_t = np.zeros(N, np.float32)
        m_t = np.zeros(N, np.float32)

        x_t[nodes] = X_np
        y_t[nodes] = y_np
        m_t[nodes] = 1.0

        # PyG Data object for one timestamp. edge_index and edge_attr are the
        # fixed station graph learned/constructed during training.
        data = Data(
            x=torch.tensor(x_t),
            y=torch.tensor(y_t),
            y_mask=torch.tensor(m_t),
            edge_index=torch.tensor(edge_index),
            edge_attr=torch.tensor(edge_attr),
            node_idx=node_idx_tensor.clone(),
        )
        data_list.append(data)

    return data_list


# ===========================================
#  GNN model, must match training definition
# ===========================================
class TempBiasGATv2(nn.Module):
    """
    GATv2-based graph neural network for station-level temperature bias.

    The model combines two sources of node information:
      1. meteorological / temporal / geographic features transformed by preproc
      2. a learned station embedding that lets the model learn station-specific
         systematic behavior

    Two GATv2Conv layers pass information over the fixed station graph using
    edge attributes. Residual connections and LayerNorm stabilize the hidden
    representation. The final linear layer outputs one scalar bias prediction
    per station node.
    """

    def __init__(self, in_channels, edge_dim, num_stations, hidden=128, heads=4, dropout=0.2):
        super().__init__()

        # Half of the hidden vector is reserved for station identity embedding;
        # the rest is produced from the meteorological feature vector.
        emb_dim = hidden // 2
        feat_dim = hidden - emb_dim

        # Learned lookup table: station index -> embedding vector.
        self.station_emb = nn.Embedding(num_stations, emb_dim)

        # Project preprocessed input features into the hidden feature subspace.
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, feat_dim),
            nn.ReLU(),
        )

        # First graph attention layer. The output dimension per head is
        # hidden // heads, and concat=True joins all heads back to `hidden`.
        self.conv1 = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

        # Second graph attention layer with the same hidden size.
        self.conv2 = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

        # LayerNorm is applied after residual addition in each block.
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Regression head: one predicted bias value per node.
        self.lin_out = nn.Linear(hidden, 1)

    def forward(self, d):
        """
        Run a forward pass for one PyG graph snapshot or mini-batch.

        Expected fields in `d`:
          - d.x          : node feature matrix
          - d.node_idx   : station indices for embedding lookup
          - d.edge_index : graph connectivity
          - d.edge_attr  : edge-level attributes
        """
        # Build initial node representation by concatenating transformed
        # meteorological features and learned station identity embeddings.
        feat = self.input_mlp(d.x)
        emb = self.station_emb(d.node_idx)
        x = torch.cat([feat, emb], dim=-1)

        # GAT block 1: message passing, nonlinearity, dropout, residual, norm.
        h = self.conv1(x, d.edge_index, d.edge_attr)
        h = self.act(h)
        h = self.drop(h)
        x = self.norm1(x + h)

        # GAT block 2: same structure, refining the node representation.
        h = self.conv2(x, d.edge_index, d.edge_attr)
        h = self.act(h)
        h = self.drop(h)
        x = self.norm2(x + h)

        # Output shape is [num_nodes]. Values represent predicted temperature bias.
        return self.lin_out(x).squeeze(-1)


# ===========================================
#  GNN inference: build snapshots & predict
# ===========================================
def gnn_predict_bias(df_pl, model, preproc, sid_to_idx, edge_index, edge_attr):
    """
    Run GNN inference for a tabular set of forecast rows.

    Parameters
    ----------
    df_pl : pl.DataFrame
        Input rows containing ID columns, FEATS, T2, and optionally obs_TA.
    model : nn.Module
        Loaded trained GNN model.
    preproc : sklearn-like transformer
        Fitted preprocessing pipeline from training.
    sid_to_idx : dict
        Mapping from station ID to graph node index.
    edge_index, edge_attr : array-like
        Fixed graph structure and edge features from training.

    Returns
    -------
    pl.DataFrame
        One row per input forecast row, containing:
          - station/time metadata
          - raw forecast temperature
          - observation, if available
          - corrected forecast temperature
    """
    # Preprocess all rows once, then slice by validtime during graph construction.
    X_pd = df_pl.select(FEATS).to_pandas()
    X_np = preproc.transform(X_pd).astype(np.float32, copy=False)

    # Convert frequently accessed columns to NumPy arrays for fast positional indexing.
    sid_all = df_pl["SID"].to_numpy()
    vt_all = df_pl["validtime"].to_numpy()
    anal_all = df_pl["analysistime"].to_numpy()
    lt_all = df_pl["leadtime"].to_numpy()
    raw_fc_all = df_pl[TEMP_FC].to_numpy()

    # Observations may be absent for pure inference. In that case, keep NaNs so
    # downstream code can still write the same schema.
    if LABEL_OBS in df_pl.columns:
        obs_all = df_pl[LABEL_OBS].to_numpy()
    else:
        obs_all = np.full_like(raw_fc_all, np.nan, dtype=np.float32)

    # Map each row's SID into the fixed graph node index used during training.
    node_idx_all = np.vectorize(sid_to_idx.get)(sid_all)
    if np.any(node_idx_all == None):
        raise ValueError("Some SIDs in inference data are not in sid_to_idx mapping from training.")
    node_idx_all = node_idx_all.astype(int)

    # Group rows by validtime using sorted indices. Each validtime becomes one
    # dense graph snapshot for the model.
    order = np.argsort(vt_all)
    vt_sorted = vt_all[order]
    ut, si = np.unique(vt_sorted, return_index=True)

    N = len(sid_to_idx)
    F = X_np.shape[1]

    # Move fixed graph tensors to DEVICE once and reuse them for all snapshots.
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=DEVICE)
    node_idx_tensor = torch.arange(N, dtype=torch.long, device=DEVICE)

    records = []

    for i, _ in enumerate(ut):
        # Indices of original rows belonging to the current validtime.
        s = si[i]
        e = si[i + 1] if i + 1 < len(si) else len(order)
        idx = order[s:e]

        # Create the full station-node feature matrix for this snapshot.
        # Only stations present at this validtime are filled with real features.
        x_t = np.zeros((N, F), np.float32)
        nodes = node_idx_all[idx]
        x_t[nodes] = X_np[idx]

        data = Data(
            x=torch.tensor(x_t, dtype=torch.float32, device=DEVICE),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            node_idx=node_idx_tensor,
        )

        # Predict bias for every station node, then select the nodes corresponding
        # to the actual rows in this validtime group.
        with torch.no_grad():
            bias_full = model(data).cpu().numpy()

        bias_rows = bias_full[nodes]
        raw_fc = raw_fc_all[idx]
        obs = obs_all[idx]

        # Store row-level outputs. Corrected temperature is raw forecast plus the
        # model-predicted bias correction.
        for row_i, b, rf, ob in zip(idx, bias_rows, raw_fc, obs):
            records.append(
                {
                    "SID": sid_all[row_i],
                    "analysistime": anal_all[row_i],
                    "validtime": vt_all[row_i],
                    "leadtime": lt_all[row_i],
                    "raw_fc": rf,
                    LABEL_OBS: ob,
                    CORR_COL: rf + b,
                }
            )

    return pl.DataFrame(records)


# ===========================================
#  Main: load artifacts & run inference
# ===========================================
def main():
    """
    Main script workflow.

    Steps:
      1. Load all training artifacts: preprocessor, graph, station mapping, model.
      2. Run GNN bias-correction inference for yearly parquet files.
      3. Save row-level evaluation parquet files for downstream metric analysis.
      4. Compute a final masked-RMSE summary for 2024-09..2025-08.
    """
    print("Loading saved artifacts...")

    # Fitted preprocessing pipeline from training. This must be loaded before
    # constructing the model because it determines the number of input channels.
    preproc = joblib.load(PREPROC_PATH)

    # Read station metadata to verify that the artifact exists and is readable.
    # The node count used by the model comes from sid_to_idx below.
    pd.read_parquet(STATIONS_PATH)

    # Station ID -> integer node index mapping. This must match the graph and the
    # embedding table from the training run.
    with open(SID_MAP_PATH, "r") as f:
        sid_to_idx = json.load(f)
    num_stations = len(sid_to_idx)

    # Fixed graph structure and edge attributes used by the trained model.
    edge_index = joblib.load(EDGE_INDEX_PATH)
    edge_attr = joblib.load(EDGE_ATTR_PATH)

    print("Loading trained GNN model...")
    with open(BEST_PARAMS_PATH, "r") as f:
        params = json.load(f)

    # Model dimensions are inferred from saved preprocessing and graph artifacts.
    in_channels = len(preproc.get_feature_names_out())
    edge_dim = edge_attr.shape[1]

    model = TempBiasGATv2(
        in_channels=in_channels,
        edge_dim=edge_dim,
        num_stations=num_stations,
        hidden=params["hidden"],
        heads=params["heads"],
        dropout=params["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ---------------------------------------------------
    # Yearly inference outputs
    # ---------------------------------------------------
    # This loop creates row-level parquet files for each year. The rows include
    # both raw and corrected forecasts, making it easy to calculate grouped
    # metrics by month, leadtime, station, season, etc. in a separate script.
    for year in range(2024, 2026):
        print(f"\nProcessing year {year}...")
        files = sorted(DATA_DIR.glob(f"ml_data_full_{year}-*.parquet"))
        accumulated = []

        # Read only columns needed for inference and later metrics.
        needed = list(set(ID + FEATS + [LABEL_OBS]))

        for f in files:
            # Lazily scan each file so filtering/deduplication can be pushed down
            # by Polars where possible.
            lf = pl.scan_parquet(str(f)).select(needed)

            # Cast split column to string before parsing, because parquet files may
            # not all store datetime columns with identical physical types.
            lf = lf.with_columns(pl.col(SPLIT_COLUMN).cast(pl.Utf8))
            lf = lf.filter(
                pl.col(SPLIT_COLUMN)
                .str.strptime(pl.Datetime, strict=False, exact=False)
                .dt.year()
                == year
            )

            # Skip files that contain no rows for the requested year after parsing.
            if lf.head(1).collect(engine="streaming").height == 0:
                continue

            # If duplicate forecast rows exist for the same station/analysis/lead,
            # keep the last row after sorting by the full time identity.
            lf = (
                lf.sort(["SID", "analysistime", "leadtime", "validtime"])
                .unique(subset=["SID", "analysistime", "leadtime"], keep="last")
            )

            df = lf.collect(engine="streaming")
            if df.height == 0:
                continue

            # Forecast temperature is required because the corrected value is
            # raw T2 plus predicted bias.
            df = df.filter(pl.col(TEMP_FC).is_not_null())
            if df.height == 0:
                continue

            df_pred = gnn_predict_bias(df, model, preproc, sid_to_idx, edge_index, edge_attr)
            accumulated.append(df_pred)

        if not accumulated:
            print(f"No rows collected for year {year}; skipping.")
            continue

        # Combine all processed chunks for the year and keep rows where both
        # observation and corrected forecast are available for evaluation.
        all_df = pl.concat(accumulated, how="vertical_relaxed")
        all_df = all_df.filter(
            pl.col(LABEL_OBS).is_not_null()
            & pl.col(CORR_COL).is_not_null()
        )

        # Add month for downstream grouped metrics.
        all_df = all_df.with_columns(month=safe_month("validtime"))

        tag = f"{SPLIT_COLUMN}_{MODEL_TAG}_{year}"
        out_path = OUTDIR / f"eval_rows_{tag}.parquet"
        all_df.write_parquet(out_path)
        print(f"Saved to: {out_path}")

    # ---------------------------------------------------
    # Final evaluation summary for the thesis test window
    # ---------------------------------------------------
    # This reproduces the graph-snapshot evaluation style: construct one snapshot
    # per validtime, evaluate raw forecast as a zero-bias baseline, and compare it
    # against the trained GNN bias model.
    print("\nBuilding evaluation DataFrame for 2024-09..2025-08...")

    needed = list(set(ID + FEATS + [LABEL_OBS]))
    lf_all = pl.scan_parquet(str(DATA_DIR / "ml_data_full_*.parquet")).select(needed)

    # Select the September 2024 through August 2025 period using a half-open
    # interval: [2024-09-01, 2025-09-01).
    lf_all = lf_all.with_columns(
        pl.col("validtime").cast(pl.Utf8)
    ).filter(
        pl.col("validtime")
        .str.strptime(pl.Datetime, strict=False, exact=False)
        .is_between(
            pl.datetime(2024, 9, 1),
            pl.datetime(2025, 9, 1),
            closed="left",
        )
    )

    # Use the same deduplication rule as the yearly inference loop and require
    # both forecast and observation for RMSE calculation.
    lf_all = (
        lf_all.sort(["SID", "analysistime", "leadtime", "validtime"])
        .unique(subset=["SID", "analysistime", "leadtime"], keep="last")
    ).filter(
        pl.col(TEMP_FC).is_not_null() & pl.col(LABEL_OBS).is_not_null()
    )

    df_eval_pl = lf_all.collect(engine="streaming")
    print("Eval rows:", df_eval_pl.height)

    if df_eval_pl.height == 0:
        print("No rows for 2024-09..2025-08, skipping RMSE calculation.")
        return

    # Convert the tabular evaluation period into graph snapshots so the metric is
    # computed in the same node/mask format as during model training.
    snaps = split_to_snapshots(
        df_eval_pl,
        preproc,
        FEATS,
        LABEL_OBS,
        TEMP_FC,
        sid_to_idx,
        edge_index,
        edge_attr,
    )

    from torch_geometric.loader import DataLoader

    eval_loader = DataLoader(snaps, batch_size=16, shuffle=False)

    # Baseline model that predicts zero bias for every station. Since corrected
    # temperature = raw forecast + predicted bias, zero bias is equivalent to the
    # uncorrected raw forecast baseline.
    class ZeroBiasModel(nn.Module):
        def forward(self, d):
            return torch.zeros(d.x.size(0), device=d.x.device)

    zero_model = ZeroBiasModel().to(DEVICE)
    rmse_raw_style = eval_rmse(zero_model, eval_loader, DEVICE)
    rmse_gnn_style = eval_rmse(model, eval_loader, DEVICE)

    print("\nRMSE on 2024-09..2025-08:")
    print(f"  Raw EC:         {rmse_raw_style:.3f}")
    print(f"  GNN corrected:  {rmse_gnn_style:.3f}")
    print(f"  Improvement:    {rmse_raw_style - rmse_gnn_style:.3f}")


if __name__ == "__main__":
    # Show more rows when Polars prints DataFrames during interactive runs.
    pl.Config.set_tbl_rows(20)
    main()

