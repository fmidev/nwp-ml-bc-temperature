# =====================================================
#   GNN Bias Correction with Optuna + Graph Search
#   - MLP input, station embeddings, residuals, norm
#   - Optuna tunes k and radius_km
#   - Saves all artifacts needed for inference
# =====================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # choose GPU #1

import math, random, numpy as np, pandas as pd, polars as pl, torch, optuna
from datetime import timedelta
from torch import nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import json, joblib, pickle
from optuna.exceptions import TrialPruned
from torch_geometric.nn import GATv2Conv
import math
import geopandas as gpd  
import gc
from shapely.strtree import STRtree


# -----------------------------------------------------
# Config
# -----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAPH_OUT = Path.home() / "thesis_project" / "data" / "graph_artifacts"
GRAPH_OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

PATH = Path.home() / "thesis_project" / "data" / "ml_data_full" / "ml_data_full_*.parquet"
FEATS = [
    "T2","SKT","MX2T","MN2T","D2","T_925","MSL","U10","V10","T2_M1","T_925_M1",
    "T2_ENSMEAN_MA1", "LCC", "MCC", "sin_hod","cos_hod","sin_doy","cos_doy","analysishour","leadtime","lon","lat","elev"
]
LABEL_OBS = "obs_TA"
TEMP_FC = "T2"
STATION_ID_COL = "SID"

MET_VARS   = ["T2","SKT","MX2T","MN2T","D2","T_925","MSL","U10","V10","T2_M1","T_925_M1", "LCC", "MCC"]
ENSMEAN    = "T2_ENSMEAN_MA1"
TIME_FEATS = ["sin_hod","cos_hod","sin_doy","cos_doy","analysishour","leadtime"]
GEO_FEATS  = ["lon","lat","elev"]

# -----------------------------------------------------
# Coastline config
# -----------------------------------------------------
COAST_SHP = (
    Path.home()
    / "thesis_project"
    / "data"
    / "maps"
    / "coastline"
    / "ne_10m_coastline.shp"
)

# Robust metric CRS (avoids PROJ/grid issues)
COAST_METRIC_CRS = "EPSG:3857"

# =====================================================
# Data loading and splitting
# =====================================================

def load_dataset(PATH, FEATS, LABEL_OBS, TEMP_FC, STATION_ID_COL):
    """
    Helper function to read the dataset from the parquet file.
        - Reads the needed columns
        - Filters out the missing observed or predicted temperature values
        - Changes the analysistime to datetime type
    Returns:
        Dataframe of the data"""

    lf = (
        pl.scan_parquet(str(PATH))
          .select(FEATS + [LABEL_OBS, "analysistime", STATION_ID_COL, "validtime"])
    )

    lf = lf.with_columns(
        pl.col("analysistime").str.strptime(pl.Datetime, strict=False).alias("analysistime_dt"),
        pl.col("validtime").str.strptime(pl.Datetime, strict=False).alias("validtime_dt"),
        pl.col("analysishour").cast(pl.Int8),
    )

    df = (
        lf.filter(pl.col(LABEL_OBS).is_not_null() & pl.col(TEMP_FC).is_not_null())
          .collect(engine="streaming")
    )

    return df

def split_trainval_test(df, TEST_DAYS=365, TRAIN_START_DT= None, N_FOLDS=3):
    """
    Split a time-indexed Polars DataFrame into:
      1) a fixed hold-out test set consisting of the last TEST_DAYS days, and
      2) multiple rolling (time-ordered) train/validation folds built from the
         remaining earlier data.
    The split is based on the `validtime_dt` column (datetime).

    Rolling folds:
    - Unique `validtime_dt` timestamps within `df_tv` are sorted and used as
      fold boundaries.
    - The timeline is divided into (N_FOLDS + 1) roughly equal segments.
    - For each fold i:
        * Training data (`df_tr`) contains all rows with `validtime_dt` strictly
          earlier than the fold’s validation start timestamp.
        * Validation data (`df_va`) contains rows in a contiguous time window
          starting at that timestamp (and ending at the next computed edge, if
          applicable).
    - Only folds with non-empty train and validation sets are kept.
    Returns:
        folds : list[tuple[pl.DataFrame, pl.DataFrame]]
            List of (train_df, val_df) pairs for rolling cross-validation.
        df_test : pl.DataFrame
        Hold-out test set containing rows with `validtime_dt >= test_start`.
    """

    # Last ~year as test set
    max_vt = df["validtime_dt"].max()
    test_start = max_vt - timedelta(days=TEST_DAYS)

    # If want to restrict the train period to be shorter than the full data
    if TRAIN_START_DT is None:
        min_tv = df["validtime_dt"].min()
        df_tv = df.filter(
            (pl.col("validtime_dt") < test_start) &
            (pl.col("validtime_dt") >= min_tv)
        )
    else:
        df_tv = df.filter(
            (pl.col("validtime_dt") < test_start) &
            (pl.col("validtime_dt") >= TRAIN_START_DT)
        )

    df_test = df.filter(pl.col("validtime_dt") >= test_start)

    # Rolling folds on restricted df_tv
    inits = (
        df_tv.select("validtime_dt")
             .unique()
             .sort("validtime_dt")["validtime_dt"]
             .to_list()
    )
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


def split_final_train_val(df_all, test_days=365, val_days=90):
    """
    Make a final train/val split inside the TRAIN+VAL period:
    - Test period = last `test_days`
    - Final val = last `val_days` before test starts
    - Final train = everything before that
    """
    max_vt = df_all["validtime_dt"].max()
    test_start = max_vt - timedelta(days=test_days)
    val_start  = test_start - timedelta(days=val_days)

    df_train_final = df_all.filter(pl.col("validtime_dt") < val_start)
    df_val_final   = df_all.filter(
        (pl.col("validtime_dt") >= val_start) & (pl.col("validtime_dt") < test_start)
    )
    df_test_final  = df_all.filter(pl.col("validtime_dt") >= test_start)

    return df_train_final, df_val_final, df_test_final


# =====================================================
# Preprocessor: z-score → fill 0 → add mask
# =====================================================

class ZScoreFill0WithMask(BaseEstimator, TransformerMixin):
    """
    Sklearn-style transformer that standardizes a single numeric feature (z-score)
    while handling missing values by:
      - replacing NaNs with 0 in the standardized output, and
      - adding an extra binary “missing” indicator column.
    """
    def __init__(self, feature_name):
        """Store the feature name and initialize mean/std defaults."""
        self.feature_name = feature_name
        self.mean_, self.std_ = 0.0, 1.0

    def fit(self, X, y=None):
        """Compute mean and std from non-missing values (std falls back to 1 if zero)."""
        x = np.asarray(X, dtype=float).ravel()
        m = ~np.isnan(x)
        if m.any():
            mu, sd = x[m].mean(), x[m].std(ddof=0)
            self.mean_, self.std_ = float(mu), float(sd if sd != 0 else 1.0)
        return self

    def transform(self, X):
        """Return a 2-column array: [z_scored_values_with_nans_as_0, missing_mask_as_0_1]."""
        x = np.asarray(X, dtype=float).ravel()
        miss = np.isnan(x)
        z = np.zeros_like(x)
        z[~miss] = (x[~miss] - self.mean_) / self.std_
        return np.column_stack([z, miss.astype(float)])

    def get_feature_names_out(self, input_features=None):
        """Output names for the two produced columns: feature and feature_missing."""
        return np.array([self.feature_name, f"{self.feature_name}_missing"])

def make_preprocessor(df_columns, MET_VARS, ENSMEAN, TIME_FEATS, GEO_FEATS):
    """
    Build a ColumnTransformer that applies ZScoreFill0WithMask to selected meteorological
    variables (and optionally the ensemble mean), while passing through time and
    geographic features unchanged.
    Returns (preprocessor, passthrough_feature_list).
    """
    transformers = []
    for col in MET_VARS:
        if col in df_columns:
            transformers.append(
                (f"zfm_{col}", Pipeline([("zfm", ZScoreFill0WithMask(col))]), [col])
            )
    if ENSMEAN in df_columns:
        transformers.append(
            ("zfm_ENS", Pipeline([("zfm", ZScoreFill0WithMask(ENSMEAN))]), [ENSMEAN])
        )
    passthrough = [c for c in TIME_FEATS + GEO_FEATS if c in df_columns]
    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return preproc, passthrough

def fit_preproc_on_train(df_tr_pl, feat_cols, MET_VARS, ENSMEAN, TIME_FEATS, GEO_FEATS,
                         max_rows_for_fit: int = 1_000_000):
    """
    Fit the preprocessor on training data, optionally subsampling to `max_rows_for_fit`
    rows to limit memory use. Returns the fitted preprocessor and the output feature names.
    """
    n_rows = df_tr_pl.height
    if n_rows > max_rows_for_fit:
        print(f"Fitting preprocessor on a sample of {max_rows_for_fit} / {n_rows} rows.")
        df_fit = df_tr_pl.sample(max_rows_for_fit, with_replacement=False, seed=SEED)
    else:
        print(f"Fitting preprocessor on all {n_rows} rows.")
        df_fit = df_tr_pl

    Xtr = df_fit.select(feat_cols).to_pandas()
    preproc, _ = make_preprocessor(Xtr.columns, MET_VARS, ENSMEAN, TIME_FEATS, GEO_FEATS)
    preproc.fit(Xtr)
    return preproc, preproc.get_feature_names_out()


# =====================================================
# Coastline helpers
# =====================================================

def add_dist_sea_to_stations_STRtree(
    stations: pd.DataFrame,
    coast_shp: Path = COAST_SHP,
    metric_crs: str = COAST_METRIC_CRS,
) -> pd.DataFrame:
    """
    Add dist_sea (km) using STRtree nearest coastline segment.
    Avoids union/sjoin issues and handles STRtree.nearest returning index OR geometry.
    """
    # Load + project coastline
    coast = gpd.read_file(coast_shp)
    coast = coast[coast.geometry.notna() & ~coast.geometry.is_empty].copy()
    coast = coast.to_crs(metric_crs)

    geoms = list(coast.geometry.values)
    if len(geoms) == 0:
        raise RuntimeError("No valid coastline geometries loaded.")

    tree = STRtree(geoms)

    # Stations -> projected points
    gdf_stn = gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    ).to_crs(metric_crs)

    pts = list(gdf_stn.geometry.values)
    dists_m = np.empty(len(pts), dtype=float)

    for i, p in enumerate(pts):
        nearest = tree.nearest(p)
        # Shapely STRtree.nearest may return geometry OR integer index
        if isinstance(nearest, (int, np.integer)):
            nearest = geoms[int(nearest)]
        dists_m[i] = p.distance(nearest)

    out = stations.copy()
    out["dist_sea"] = dists_m / 1000.0  # km
    return out


# =====================================================
# Graph building
# =====================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (in km) between two latitude/longitude points
    using the haversine formula. Supports scalars or numpy arrays (broadcastable).
    """
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def rbf(x, c, w):
    """
    Compute Gaussian RBF features for input values `x` given centers `c` and widths `w`.
    Returns an (len(x), len(c)) matrix of exp(- (x-c)^2 / (2 w^2)).
    """
    x = np.asarray(x)[:, None]
    c = np.asarray(c)[None, :]
    w = np.asarray(w)[None, :]
    return np.exp(-((x - c)**2) / (2 * w**2))


def build_graph(stations, k=10, radius_km=275.0, gamma=5.0, coast_threshold_km=20, use_land_sea_mask=True):
    """
    Build a k-NN graph between stations based on an effective distance that combines
    horizontal (haversine) distance and elevation difference.

    Steps:
    - Compute pairwise geographic distances (km) and elevation deltas.
    - Form an effective distance: sqrt(d_geo^2 + (gamma * |dz|/1000)^2).
    - For each station, connect to its k nearest neighbors by effective distance.
    - Optionally mask edges:
        * keep only edges within `radius_km`, and
        * if `use_land_sea_mask` is True, only connect stations that are both coastal
          or both non-coastal (based on `dist_sea <= coast_threshold_km`).
    - Create an undirected edge_index (both directions).
    - Build edge attributes including:
        * RBF expansions of geographic distance (with preset centers/widths),
        * raw distance,
        * signed elevation difference,
        * absolute elevation difference,
      and z-score standardize the last three continuous attributes.

    Returns:
    edge_index : (2, E) int64 array
        Source/target indices for each directed edge.
    edge_attr : (E, F) float32 array
        Edge feature matrix.
    """

    stn = stations.reset_index(drop=True)

    lat  = stn["lat"].to_numpy()
    lon  = stn["lon"].to_numpy()
    elev = stn["elev"].to_numpy()


    N = len(stn)
    latm = np.repeat(lat[:, None], N, 1)
    lonm = np.repeat(lon[:, None], N, 1)

    d_geo = haversine_km(latm, lonm, latm.T, lonm.T)
    np.fill_diagonal(d_geo, np.inf)
    dz = elev[:, None] - elev[None, :]
    d_eff = np.sqrt(d_geo**2 + (gamma * (np.abs(dz) / 1000.0))**2)

    nn_idx = np.argsort(d_eff, 1)[:, :k]
    row = np.repeat(np.arange(N), k)
    col = nn_idx.flatten()

    # ---- Edge masking ----
    if use_land_sea_mask:
        dist_sea = stn["dist_sea"].to_numpy()

        if not np.isfinite(dist_sea).all():
            raise ValueError("dist_sea contains NaNs/inf — coastline distance failed.")

        is_coast = dist_sea <= coast_threshold_km
        same_coast = is_coast[row] == is_coast[col]
        mask = (d_geo[row, col] <= radius_km) & same_coast
    else:
        mask = (d_geo[row, col] <= radius_km)

    row, col = row[mask], col[mask]

    edge_index = np.vstack([np.concatenate([row, col]), np.concatenate([col, row])])
    dist = d_geo[edge_index[0], edge_index[1]]
    edz = dz[edge_index[0], edge_index[1]]
    centers = np.array([25, 50, 100, 200])
    widths = centers / 2
    rbf_d = rbf(dist, centers, widths)

    edge_attr = np.column_stack([rbf_d, dist, edz, np.abs(edz)]).astype(np.float32)
    cont_idx = np.arange(edge_attr.shape[1] - 3, edge_attr.shape[1])
    m, std = edge_attr[:, cont_idx].mean(0), edge_attr[:, cont_idx].std(0)
    std[std == 0] = 1
    edge_attr[:, cont_idx] = (edge_attr[:, cont_idx] - m) / std

    return edge_index.astype(np.int64), edge_attr


def check_coastal_edges(stations, edge_index, coast_threshold_km=20.0):
    """
    Diagnostic helper that checks whether any edges violate the coastal mask rule
    (coastal nodes connected to non-coastal nodes). Prints the number of violating
    edges and a few examples with their dist_sea values.
    """
        
    stn = stations.reset_index(drop=True)
    dist_sea = stn["dist_sea"].to_numpy()
    is_coast = dist_sea <= coast_threshold_km

    src = edge_index[0]
    dst = edge_index[1]
    bad = is_coast[src] != is_coast[dst]

    print("Edges violating coast mask:", bad.sum())
    if bad.sum() > 0:
        idxs = np.where(bad)[0][:10]
        print("Example problematic edges (first 10):")
        for i in idxs:
            s = src[i]
            t = dst[i]
            print(
                f"  edge {i}: {s} -> {t}, "
                f"dist_sea_src={dist_sea[s]:.1f}, dist_sea_dst={dist_sea[t]:.1f}"
            )


# =====================================================
# PyG snapshots
# =====================================================

def make_node_index(stations_df):
    return dict(zip(stations_df["SID"], range(len(stations_df))))

def split_to_snapshots(df_pl, preproc, feat_cols, lab_col, fc_col,
                       sid_to_idx, edge_index, edge_attr):
    """
    Build a list of PyG Data snapshots, one per validtime, without
    ever materializing the full feature matrix in memory.
    """

    N = len(sid_to_idx)
    node_idx_tensor = torch.arange(N, dtype=torch.long)

    data_list = []

    # group_by validtime, keep chronological order of groups
    for validtime_value, group_df in df_pl.group_by("validtime", maintain_order=True):
        # group_df is a Polars DataFrame with rows for a single validtime

        # --- 1) Features (small pandas slice) ---
        X_pd = group_df.select(feat_cols).to_pandas()
        X_np = preproc.transform(X_pd).astype(np.float32)   # [num_rows_t, in_ch]

        # --- 2) Target: bias = obs - fc ---
        y_np = (group_df[lab_col] - group_df[fc_col]).to_numpy()

        # --- 3) Station indices ---
        sid   = group_df["SID"].to_numpy()
        nodes = np.vectorize(sid_to_idx.get)(sid)    # [num_rows_t]

        # --- 4) Allocate full node tensors for this snapshot ---
        x_t = np.zeros((N, X_np.shape[1]), np.float32)
        y_t = np.zeros(N, np.float32)
        m_t = np.zeros(N, np.float32)

        x_t[nodes] = X_np
        y_t[nodes] = y_np
        m_t[nodes] = 1.0


        # --- 5) Build PyG Data ---
        data = Data(
            x=torch.tensor(x_t),
            y=torch.tensor(y_t),
            y_mask=torch.tensor(m_t),
            edge_index=torch.tensor(edge_index),
            edge_attr=torch.tensor(edge_attr),
            node_idx=node_idx_tensor.clone(),  # for station embeddings
        )
        data_list.append(data)

    return data_list


# =====================================================
# Model (MLP + station embeddings + residual + norm)
# =====================================================


class TempBiasGATv2(nn.Module):
    """
    Graph attention model (GATv2) for predicting per-station temperature bias (e.g., obs - fc)
    on a fixed station graph. Combines:
      - transformed per-node input features, and
      - a learned station embedding (identity embedding),
    then applies stacked GATv2Conv blocks with residual connections + LayerNorm, and outputs
    a single scalar prediction per node.
    """

    def __init__(self, in_channels, edge_dim, num_stations,
                 hidden=128, heads=4, dropout=0.2):
        
        """
        Params:
        in_channels : int
            Number of input node feature channels (after preprocessing).
        edge_dim : int
            Number of edge feature channels (edge_attr width).
        num_stations : int
            Number of nodes/stations in the graph (for the embedding table).
        hidden : int
            Internal hidden dimension (split into feature projection + station embedding).
        heads : int
            Number of attention heads in each GATv2Conv layer.
        dropout : float
            Dropout rate used inside attention and in the block dropout.
        """

        super().__init__()

        # Split hidden into a learned station-id embedding and a projected feature part
        emb_dim  = hidden // 2
        feat_dim = hidden - emb_dim


        # Learned per-station embedding to provide node identity / station-specific offsets
        self.station_emb = nn.Embedding(num_stations, emb_dim)

        # Map raw node features -> feat_dim before concatenating with station embedding
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, feat_dim),
            nn.ReLU(),
        )

        # Message passing / attention over the station graph using edge attributes
        self.conv1 = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,         
            add_self_loops=False 
                                  
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False
        )
        

        # Post-block normalization (used with residual connections)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Final per-node regression head -> scalar bias prediction
        self.lin_out = nn.Linear(hidden, 1)

    def forward(self, d):
        """
        Forward pass on a PyG Data snapshot `d`.

        Expects `d` to contain:
          - d.x          : [N, in_channels] node features
          - d.node_idx   : [N] integer node indices for station embeddings
          - d.edge_index : [2, E] graph connectivity
          - d.edge_attr  : [E, edge_dim] edge features

        Returns
        out : [N] tensor
            Predicted bias value for each node/station.
        """

        # Combine projected features with station identity embedding
        feat = self.input_mlp(d.x)          # [N, feat_dim]
        emb  = self.station_emb(d.node_idx) # [N, emb_dim]
        x    = torch.cat([feat, emb], dim=-1)  # [N, hidden]

        # Block 1: attention message passing + residual + norm
        h = self.conv1(x, d.edge_index, d.edge_attr)
        h = self.act(h)
        h = self.drop(h)
        x = self.norm1(x + h)

        # Block 2: attention message passing + residual + norm
        h = self.conv2(x, d.edge_index, d.edge_attr)
        h = self.act(h)
        h = self.drop(h)
        x = self.norm2(x + h)      

        # Per-node scalar output
        out = self.lin_out(x).squeeze(-1)
        return out


# =====================================================
# Training / evaluation helpers
# =====================================================


def train_epoch(m, ldr, opt, dv):
    """
    Train the model for one epoch over a DataLoader of PyG snapshots.

    For each snapshot:
      - move data to device,
      - compute predictions,
      - compute a masked MSE loss using `d.y_mask` (only stations with targets),
      - backprop + gradient clipping + optimizer step.

    Returns:
    float
        Global RMSE over the whole epoch, computed as:
          sqrt( sum((pred - y)^2 * mask) / sum(mask) )
        aggregated across all snapshots and nodes.
    """

    m.train()
    se_sum = 0.0   # sum of squared errors (masked)
    m_sum  = 0.0   # sum of mask entries

    for d in ldr:
        d = d.to(dv)
        opt.zero_grad()
        p = m(d)

        # ---- loss for optimization (MSE) ----
        e2 = (p - d.y) ** 2
        loss = (e2 * d.y_mask).sum() / (d.y_mask.sum() + 1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=2.0)
        opt.step()

        # ---- accumulate for GLOBAL RMSE logging ----
        se_sum += (e2 * d.y_mask).sum().item()
        m_sum  += d.y_mask.sum().item()

    if m_sum == 0:
        return float("nan")

    return math.sqrt(se_sum / (m_sum + 1e-8))




@torch.no_grad()

def eval_rmse(m, ldr, dv):
    """
    Evaluate the model using masked global RMSE across a DataLoader of PyG snapshots.

    Computes:
      sqrt( sum((pred - y)^2 * mask) / sum(mask) )
    summed over all nodes and all snapshots, matching the inference-time metric.

    Returns:
    float
        Global masked RMSE (NaN if there are no valid masked entries).
    """
    m.eval()
    se_sum = 0.0  # sum of squared errors, masked
    m_sum  = 0.0  # sum of mask entries

    for d in ldr:
        d = d.to(dv)
        p = m(d)

        e2 = (p - d.y) ** 2               # [N]
        se_sum += (e2 * d.y_mask).sum().item()
        m_sum  += d.y_mask.sum().item()

    if m_sum == 0:
        return float("nan")

    return math.sqrt(se_sum / (m_sum + 1e-8))


# =====================================================
# Optuna CV + graph search + final training
# =====================================================


def cv_trial(trial, df_folds, stations, preproc_global, in_ch, num_stations, sid_to_idx, device):
    """
    Run one Optuna hyperparameter trial using time-based CV folds and a fixed
    TempBiasGATv2 training loop.

    What it does:
      - Samples graph, model, optimizer, and batch-size hyperparameters via Optuna.
      - Builds a station graph for this trial (edge_index/edge_attr) using the sampled
        k-nearest-neighbor and radius settings, optionally enforcing a land/sea (coastal)
        connectivity mask.
      - For each (train, val) fold:
          * Converts the fold data into PyG snapshot graphs (one per validtime) without
            materializing the full feature matrix.
          * Trains TempBiasGATv2 with AdamW using masked MSE, logging global masked RMSE.
          * Applies early stopping on validation global RMSE (with patience + min_delta).
          * Reports intermediate best validation RMSE to Optuna and supports pruning.
      - Returns the mean of the best validation RMSE across folds as the trial objective.

    Notes:
      - Uses global masked RMSE aggregated over all nodes and timesteps in the fold.
      - Explicitly frees CPU/GPU memory between folds; prunes on CUDA OOM.

    Returns:
    float
        Mean best validation global RMSE across folds (Optuna minimizes this).
    """


    print(f"\n===== Starting Optuna trial {trial.number} =====")

    global_step = 0

    # -----------------------------
    # Search space 
    # -----------------------------

    # Graph hyperparameters
    k = trial.suggest_categorical("k", [5, 8, 10, 12, 15, 20])
    radius_km = trial.suggest_categorical("radius_km", [150, 200, 250, 300, 350, 400, 500])

    # Model hyperparameters (2-layer architecture fixed in code)
    hidden = trial.suggest_categorical("hidden", [64, 96, 128, 160, 192])
    heads  = trial.suggest_categorical("heads",  [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    # Optimizer hyperparameters
    lr = trial.suggest_float("lr", 5e-5, 8e-4, log=True)
    wd = trial.suggest_float("wd", 1e-7, 5e-3, log=True)

    # Batch size (keep tunable)
    bs = trial.suggest_categorical("bs", [8, 16, 24, 32])

    # Training schedule (fixed, early stopping decides effective epochs)
    MAX_EPOCHS = 120
    PATIENCE = 10  
    MIN_DELTA = 1e-4  # small improvement threshold to avoid noise triggering resets

    try:
        # Build graph for this trial
        edge_index, edge_attr = build_graph(stations, k=k, radius_km=radius_km, use_land_sea_mask=True)
        edge_dim = edge_attr.shape[1]

        print(
            f"Trial {trial.number}: graph with "
            f"{edge_index.shape[1]} edges, edge_dim={edge_dim}"
        )

        check_coastal_edges(stations, edge_index, coast_threshold_km=20.0)

        fold_bests = []

        # ---- process folds one by one ----
        for fold_id, (df_tr, df_va) in enumerate(df_folds):
            print(f"  Preparing snapshots for fold {fold_id}...")

            tr_snaps = split_to_snapshots(
                df_tr, preproc_global, FEATS, LABEL_OBS, TEMP_FC,
                sid_to_idx, edge_index, edge_attr
            )
            va_snaps = split_to_snapshots(
                df_va, preproc_global, FEATS, LABEL_OBS, TEMP_FC,
                sid_to_idx, edge_index, edge_attr
            )

            train_loader = DataLoader(tr_snaps, batch_size=bs, shuffle=True)
            val_loader   = DataLoader(va_snaps, batch_size=bs, shuffle=False)

            model = TempBiasGATv2(
                in_channels=in_ch,
                edge_dim=edge_dim,
                num_stations=num_stations,
                hidden=hidden,
                heads=heads,
                dropout=dropout,
            ).to(device)

            opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

            best_val = float("inf")
            bad = 0

            for ep in range(MAX_EPOCHS):
                train_loss = train_epoch(model, train_loader, opt, device) 
                val_rmse   = eval_rmse(model, val_loader, device)          # global masked RMSE

                print(f"  Fold {fold_id} Ep {ep:03d}: train={train_loss:.4f} val_globalRMSE={val_rmse:.4f}")

                # early stopping on global val RMSE
                if val_rmse < (best_val - MIN_DELTA):
                    best_val = val_rmse
                    bad = 0
                else:
                    bad += 1
                    if bad > PATIENCE:
                        break

                # Report to Optuna (use global metric)
                trial.report(best_val, step=global_step)
                global_step += 1

                if trial.should_prune():
                    print("Pruning trial due to poor performance.")
                    raise TrialPruned()

            fold_bests.append(best_val)

            # free memory for this fold explicitly
            del tr_snaps, va_snaps, train_loader, val_loader, model, opt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_rmse = float(np.mean(fold_bests))
        print(f"===== Trial {trial.number}: mean CV val_globalRMSE={mean_rmse:.4f} =====")
        return mean_rmse

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA OOM encountered, pruning this trial.")
            raise TrialPruned()
        else:
            raise



def run_optuna(df_all, folds_pl, df_test_pl, in_ch, num_stations, device,
               preproc_global, stations, sid_to_idx):
    
    """
    End-to-end training pipeline that:
      1) runs Optuna hyperparameter tuning with cross-validation folds,
      2) saves the best Optuna study + parameters to a timestamped run directory,
      3) rebuilds the best graph (k, radius) and prepares PyG snapshots for a final
         train/val/test split,
      4) retrains the TempBiasGATv2 model with early stopping on validation RMSE,
      5) evaluates on the final test set, and
      6) saves the trained model and all artifacts needed for reproducible inference
         (preprocessor, stations list/order, SID→index mapping, and graph structure).

    Key behaviors:
    - Optuna objective: mean best validation global masked RMSE across CV folds.
    - Final training: early stopping based on global masked RMSE on a held-out
      validation window; best model weights are restored before test evaluation.
    - Persistence: writes Optuna results + model + preprocessing + graph metadata
      into OUT_DIR.

    Returns:
    model : nn.Module
        Trained TempBiasGATv2 model loaded with the best early-stopped weights.
    study : optuna.Study
        Completed Optuna study containing all trials and the selected best params.
    OUT_DIR : pathlib.Path
        Directory containing saved artifacts for this run.
    """

    # ---- 1 Optuna tuning ----
    study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(
        n_warmup_steps=3,   # let a few reports accumulate first
        interval_steps=1    # check every step
        ),
    )
    study.optimize(
        lambda t: cv_trial(
            t,
            df_folds=folds_pl,
            stations=stations,
            preproc_global=preproc_global,
            in_ch=in_ch,
            num_stations=num_stations,
            sid_to_idx=sid_to_idx,
            device=device,
        ),
        n_trials=30
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Best trial:", study.best_value, study.best_params)

    # ---- 2 Prepare run directory ----
    MODEL_DIR = Path.home() / "thesis_project" / "models" / "gnn_bias_correction"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    OUT_DIR = MODEL_DIR / run_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save Optuna study and best params
    with open(OUT_DIR / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved Optuna results to {OUT_DIR}")

    # ---- 3 Rebuild best graph and snapshots ----
    best_k        = study.best_params["k"]
    best_radius   = study.best_params["radius_km"]
    edge_index, edge_attr = build_graph(stations, k=best_k, radius_km=best_radius, use_land_sea_mask=True)
    edge_dim = edge_attr.shape[1]

    # ----  Final split for early stopping + final test reporting ----
    df_train_final, df_val_final, df_test_final = split_final_train_val(
        df_all, test_days=365, val_days=90
    )

    print("Final split sizes:")
    print("  train rows:", df_train_final.height)
    print("  val rows:  ", df_val_final.height)
    print("  test rows: ", df_test_final.height)

    print("Building train snapshots...")
    tr_snaps = split_to_snapshots(
        df_train_final, preproc_global, FEATS, LABEL_OBS, TEMP_FC,
        sid_to_idx, edge_index, edge_attr
    )
    print("Building val snapshots...")
    va_snaps = split_to_snapshots(
        df_val_final, preproc_global, FEATS, LABEL_OBS, TEMP_FC,
        sid_to_idx, edge_index, edge_attr
    )
    print("Building test snapshots...")
    ts_snaps = split_to_snapshots(
        df_test_final, preproc_global, FEATS, LABEL_OBS, TEMP_FC,
        sid_to_idx, edge_index, edge_attr
    )

    bs = study.best_params["bs"]
    tr_loader = DataLoader(tr_snaps, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(va_snaps, batch_size=bs, shuffle=False)
    ts_loader  = DataLoader(ts_snaps, batch_size=bs, shuffle=False)



    # ---- 4 Final model training (fixed MAX_EPOCHS + early stopping on val GLOBAL RMSE) ----
    model = TempBiasGATv2(
        in_channels=in_ch,
        edge_dim=edge_dim,
        num_stations=num_stations,
        hidden=study.best_params["hidden"],
        heads=study.best_params["heads"],
        dropout=study.best_params["dropout"],
    ).to(device)

    opt = AdamW(
        model.parameters(),
        lr=study.best_params["lr"],
        weight_decay=study.best_params["wd"],
    )

    MAX_EPOCHS = 200
    PATIENCE   = 15
    MIN_DELTA  = 1e-4

    best_val = float("inf")
    bad = 0
    best_state = None

    for ep in range(MAX_EPOCHS):
        train_rmse = train_epoch(model, tr_loader, opt, device)   # returns GLOBAL train RMSE (as you updated)
        val_rmse   = eval_rmse(model, val_loader, device)          

        print(f"Final training | Epoch {ep:03d}: train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < (best_val - MIN_DELTA):
            best_val = val_rmse
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping at epoch {ep:03d} (best val_rmse={best_val:.4f})")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best val RMSE during final training: {best_val:.4f}")


    test_rmse = eval_rmse(model, ts_loader, device)
    print(f"Final test RMSE = {test_rmse:.4f}")

    # ---- 5 Save model + artifacts ----
    torch.save(model.state_dict(), OUT_DIR / "gnn_model.pt")
    print(f"Model saved to {OUT_DIR / 'gnn_model.pt'}")

    # preprocessor
    joblib.dump(preproc_global, OUT_DIR / "preproc.joblib")
    print(f"Saved preprocessor to {OUT_DIR / 'preproc.joblib'}")

    # stations dataframe (for reproducible station list/order)
    stations.to_parquet(OUT_DIR / "stations.parquet")
    print(f"Saved stations to {OUT_DIR / 'stations.parquet'}")

    # SID -> node index mapping
    with open(OUT_DIR / "sid_to_idx.json", "w") as f:
        json.dump(sid_to_idx, f)
    print(f"Saved sid_to_idx to {OUT_DIR / 'sid_to_idx.json'}")

    # graph structure
    joblib.dump(edge_index, OUT_DIR / "edge_index.pkl")
    joblib.dump(edge_attr,  OUT_DIR / "edge_attr.pkl")
    print(f"Saved edge_index and edge_attr to {OUT_DIR}")

    # graph hyperparams
    with open(OUT_DIR / "graph_params.json", "w") as f:
        json.dump({"k": best_k, "radius_km": best_radius}, f, indent=2)
    print(f"Saved graph_params to {OUT_DIR / 'graph_params.json'}")

    return model, study, OUT_DIR



def train_without_optuna(df_train_final: pl.DataFrame, df_val_final: pl.DataFrame, df_test_final: pl.DataFrame, in_ch: int,
    num_stations: int, device: str, preproc_global, stations: pd.DataFrame, sid_to_idx: dict, OUT_NAME="full_gnn_gat_lsm",):
    
    """
    Train and evaluate the TempBiasGATv2 model once (no Optuna), using a fixed set of
    hyperparameters and early stopping on validation GLOBAL masked RMSE.

    Workflow:
    1) Define a fixed `best_params` dict (typically copied from an Optuna best trial).
    2) Build the station graph (edge_index/edge_attr) using the chosen k and radius,
       optionally enforcing the coastal land/sea mask.
    3) Convert train/val/test splits into PyG snapshots (one graph per validtime).
    4) Initialize the GATv2-based model and AdamW optimizer.
    5) Train for up to MAX_EPOCHS with early stopping on validation RMSE, restoring
       the best model state before evaluation.
    6) Compute the final test GLOBAL RMSE.
    7) Save the trained model and all inference artifacts to a named output directory
       (preprocessor, station order, SID→index mapping, and graph structure/params).

    Returns:
    model : nn.Module
        Trained TempBiasGATv2 model loaded with the best early-stopped weights.
    OUT_DIR : pathlib.Path
        Directory containing saved model + preprocessing + graph artifacts.
    """

    print("\n=== Running FINAL training (no Optuna) with early stopping on GLOBAL RMSE ===")

    # -------------------------------------------------------------------------
    # 1) Paste Optuna best_params here or run on some other chosen parameters
    # -------------------------------------------------------------------------
    best_params = {
    "k": 12,
    "radius_km": 500,
    "hidden": 192,
    "heads": 8,
    "dropout": 0.1384634953194369,
    "lr": 0.0006517770362188271,
    "wd": 0.0043741181457823495,
    "bs": 16
    }

    k = best_params["k"]
    radius_km = best_params["radius_km"]
    hidden = best_params["hidden"]
    heads = best_params["heads"]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    wd = best_params["wd"]
    bs = best_params["bs"]

    # Early stopping settings
    MAX_EPOCHS = 200
    PATIENCE = 15
    MIN_DELTA = 1e-4  # small threshold to avoid stopping on tiny noise

    # -------------------------------
    # 2) Build graph
    # -------------------------------
    edge_index, edge_attr = build_graph(stations, k=k, radius_km=radius_km, use_land_sea_mask=True)
    edge_dim = edge_attr.shape[1]
    print(f"Graph: edges={edge_index.shape[1]} edge_dim={edge_dim}")
    check_coastal_edges(stations, edge_index, coast_threshold_km=20.0)

    # -------------------------------
    # 3) Build snapshots (train/val/test)
    # -------------------------------
    print("Building train snapshots...")
    tr_snaps = split_to_snapshots(
        df_train_final, preproc_global, FEATS, LABEL_OBS, TEMP_FC,
        sid_to_idx, edge_index, edge_attr
    )
    print("Building val snapshots...")
    va_snaps = split_to_snapshots(
        df_val_final, preproc_global, FEATS, LABEL_OBS, TEMP_FC,
        sid_to_idx, edge_index, edge_attr
    )
    print("Building test snapshots...")
    ts_snaps = split_to_snapshots(
        df_test_final, preproc_global, FEATS, LABEL_OBS, TEMP_FC,
        sid_to_idx, edge_index, edge_attr
    )

    train_loader = DataLoader(tr_snaps, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(va_snaps, batch_size=bs, shuffle=False)
    test_loader  = DataLoader(ts_snaps, batch_size=bs, shuffle=False)

    # -------------------------------
    # 4) Model + optimizer
    # -------------------------------
    model = TempBiasGATv2(
        in_channels=in_ch,
        edge_dim=edge_dim,
        num_stations=num_stations,
        hidden=hidden,
        heads=heads,
        dropout=dropout,
    ).to(device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # -------------------------------
    # 5) Train with early stopping on GLOBAL RMSE (val)
    # -------------------------------
    best_val = float("inf")
    bad = 0
    best_state = None

    for ep in range(MAX_EPOCHS):
        train_rmse = train_epoch(model, train_loader, opt, device)
        val_rmse   = eval_rmse(model, val_loader, device)

        print(f"Epoch {ep:03d}: train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < best_val - MIN_DELTA:
            best_val = val_rmse
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping at epoch {ep:03d} (best val_rmse={best_val:.4f})")
                break

        # keep memory stable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------------
    # 6) Final test score (GLOBAL RMSE)
    # -------------------------------
    test_rmse = eval_rmse(model, test_loader, device)
    print(f"\n=== FINAL TEST RMSE (GLOBAL): {test_rmse:.4f} ===\n")

    # -------------------------------
    # 7) Save artifacts
    # -------------------------------
    MODEL_DIR = Path.home() / "thesis_project" / "models" / "gnn_bias_correction"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = MODEL_DIR / OUT_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), OUT_DIR / "gnn_model.pt")
    joblib.dump(preproc_global, OUT_DIR / "preproc.joblib")
    stations.to_parquet(OUT_DIR / "stations.parquet")

    with open(OUT_DIR / "sid_to_idx.json", "w") as f:
        json.dump(sid_to_idx, f)

    joblib.dump(edge_index, OUT_DIR / "edge_index.pkl")
    joblib.dump(edge_attr,  OUT_DIR / "edge_attr.pkl")

    with open(OUT_DIR / "graph_params.json", "w") as f:
        json.dump(
            {
                "k": k,
                "radius_km": radius_km,
                "hidden": hidden,
                "heads": heads,
                "dropout": dropout,
                "lr": lr,
                "wd": wd,
                "bs": bs,
                "val_days": 90,
                "early_stopping": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA},
            },
            f,
            indent=2,
        )

    print(f"Saved everything to: {OUT_DIR}")
    return model, OUT_DIR




# =====================================================
# Main
# =====================================================

def main():
    RUN_OPTUNA = False  # keep False since you already have best params

    # 1) Load all data (still streaming collect at the end)
    df_all = load_dataset(PATH, FEATS, LABEL_OBS, TEMP_FC, STATION_ID_COL)

    # 2) Final split for early stopping + final test reporting
    df_train_final, df_val_final, df_test_final = split_final_train_val(
        df_all, test_days=365, val_days=90
    )

    print("Final split sizes:")
    print("  train rows:", df_train_final.height)
    print("  val rows:  ", df_val_final.height)
    print("  test rows: ", df_test_final.height)

    # 3) Stations + dist-to-sea + canonical order
    stations = (
        df_all.select(["SID", "lat", "lon", "elev"])
              .unique(subset=["SID"])
              .to_pandas()
    )
    stations = add_dist_sea_to_stations_STRtree(stations)
    print(
        "dist_sea stats (km):",
        float(stations["dist_sea"].min()),
        float(stations["dist_sea"].median()),
        float(stations["dist_sea"].max()),
    )

    stations = stations.sort_values("SID").reset_index(drop=True)
    sid_to_idx = make_node_index(stations)
    num_stations = len(sid_to_idx)

    # IMPORTANT: JSON later will stringify keys; keep sid_to_idx keys as ints here.
    # In inference, you should do: sid_to_idx = {int(k): int(v) for k,v in sid_to_idx.items()}

    # 4) Fit preprocessor ONLY on FINAL TRAIN (not val/test)
    preproc_global, out_cols_global = fit_preproc_on_train(
        df_train_final,
        FEATS,
        MET_VARS,
        ENSMEAN,
        TIME_FEATS,
        GEO_FEATS,
    )
    in_ch = len(out_cols_global)
    print("in_channels:", in_ch)

    if RUN_OPTUNA:
        
        folds_pl, df_test_pl = split_trainval_test(df_all)
        model, study, OUT_DIR = run_optuna(
            df_all=df_all,
            folds_pl=folds_pl,
            df_test_pl=df_test_pl,
            in_ch=in_ch,
            num_stations=num_stations,
            device=DEVICE,
            preproc_global=preproc_global,
            stations=stations,
            sid_to_idx=sid_to_idx,
        )
    else:
        model, OUT_DIR = train_without_optuna(
            df_train_final=df_train_final,
            df_val_final=df_val_final,
            df_test_final=df_test_final,
            in_ch=in_ch,
            num_stations=num_stations,
            device=DEVICE,
            preproc_global=preproc_global,
            stations=stations,
            sid_to_idx=sid_to_idx,
            OUT_NAME="full_gnn_gat_lsm",
        )


if __name__ == "__main__":
    main()
