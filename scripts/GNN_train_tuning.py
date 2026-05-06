# =====================================================
#   GNN Bias Correction with Optuna + Graph Search
#   - MLP input, station embeddings, residuals, norm
#   - Optuna tunes k and radius_km
#   - Saves all artifacts needed for inference
# =====================================================

import os
import argparse


def parse_pre_args():
    """
    Parse CUDA_VISIBLE_DEVICES before importing torch.
    This matters because CUDA visibility should be set before torch is imported.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        type=str,
        help="CUDA_VISIBLE_DEVICES value, for example '0', '1', or empty string.",
    )

    args, _ = parser.parse_known_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


parse_pre_args()

import math
import random
import gc
import json
import joblib
import pickle
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import torch
import optuna
import geopandas as gpd

from torch import nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from optuna.exceptions import TrialPruned
from shapely.strtree import STRtree


# -----------------------------------------------------
# Runtime config set from command-line args
# -----------------------------------------------------

DEVICE = None
SEED = None
PATH = None
MODEL_BASE_DIR = None

COAST_SHP = None
COAST_METRIC_CRS = "EPSG:3857"
COAST_THRESHOLD_KM = 20.0


# -----------------------------------------------------
# Features / labels
# -----------------------------------------------------

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

MET_VARS = [
    "T2", "SKT", "MX2T", "MN2T", "D2", "T_925", "MSL",
    "U10", "V10", "T2_M1", "T_925_M1", "LCC", "MCC",
]

ENSMEAN = "T2_ENSMEAN_MA1"
TIME_FEATS = ["sin_hod", "cos_hod", "sin_doy", "cos_doy", "analysishour", "leadtime"]
GEO_FEATS = ["lon", "lat", "elev"]


# -----------------------------------------------------
# Arguments
# -----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GNN bias-correction model with optional Optuna graph search."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help=(
            "Input parquet file, directory, or glob pattern. "
            "Examples: /path/to/ml_data_full/ml_data_full_*.parquet or /path/to/ml_data_full"
        ),
    )

    parser.add_argument(
        "--coast-shp",
        required=True,
        type=str,
        help="Path to coastline shapefile, for example ne_10m_coastline.shp.",
    )

    parser.add_argument(
        "--model-base-dir",
        required=True,
        type=str,
        help="Base directory where trained GNN run directories will be saved.",
    )

    parser.add_argument(
        "--out-name",
        default="full_gnn_gat_lsm",
        type=str,
        help="Output run directory name when training without Optuna.",
    )

    parser.add_argument(
        "--run-optuna",
        action="store_true",
        help="Run Optuna search. If omitted, use fixed hyperparameters.",
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
        help="Device to use. Default: auto.",
    )

    parser.add_argument(
        "--threads",
        default="16",
        type=str,
        help="Thread count for OMP_NUM_THREADS and MKL_NUM_THREADS. Default: 16.",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--test-days",
        default=365,
        type=int,
        help="Number of final days reserved as test period. Default: 365.",
    )

    parser.add_argument(
        "--val-days",
        default=90,
        type=int,
        help="Number of days before test used as final validation period. Default: 90.",
    )

    parser.add_argument(
        "--n-folds",
        default=3,
        type=int,
        help="Number of rolling CV folds for Optuna. Default: 3.",
    )

    parser.add_argument(
        "--max-rows-preproc-fit",
        default=1_000_000,
        type=int,
        help="Maximum rows used to fit the preprocessing pipeline. Default: 1000000.",
    )

    parser.add_argument(
        "--coast-threshold-km",
        default=20.0,
        type=float,
        help="Distance-to-sea threshold for coastal station classification. Default: 20.",
    )

    parser.add_argument(
        "--coast-metric-crs",
        default="EPSG:3857",
        type=str,
        help="Metric CRS used for coastline distance calculation. Default: EPSG:3857.",
    )

    parser.add_argument(
        "--fixed-params-json",
        default=None,
        type=str,
        help=(
            "Optional JSON file containing fixed parameters for non-Optuna training. "
            "If omitted, the defaults inside train_without_optuna are used."
        ),
    )

    return parser.parse_args()


def resolve_input_path(input_arg: str) -> str:
    input_path = Path(input_arg)

    if input_path.is_dir():
        return str(input_path / "*.parquet")

    return str(input_path)


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# Data loading and splitting
# =====================================================

def load_dataset(path, feats, label_obs, temp_fc, station_id_col):
    """
    Read the dataset from parquet files.
    """
    lf = (
        pl.scan_parquet(str(path))
        .select(feats + [label_obs, "analysistime", station_id_col, "validtime"])
    )

    lf = lf.with_columns(
        pl.col("analysistime")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False)
        .alias("analysistime_dt"),

        pl.col("validtime")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False)
        .alias("validtime_dt"),

        pl.col("analysishour").cast(pl.Int8),
    )

    df = (
        lf.filter(pl.col(label_obs).is_not_null() & pl.col(temp_fc).is_not_null())
        .collect(engine="streaming")
    )

    return df


def split_trainval_test(df, TEST_DAYS=365, TRAIN_START_DT=None, N_FOLDS=3):
    """
    Split a time-indexed Polars DataFrame into rolling train/validation folds and test.
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

    inits = (
        df_tv.select("validtime_dt")
        .unique()
        .sort("validtime_dt")["validtime_dt"]
        .to_list()
    )

    if len(inits) < N_FOLDS + 1:
        raise ValueError("Not enough initializations for requested N_FOLDS after applying TRAIN_START_DT.")

    fold_edges = [
        int(round(i * len(inits) / (N_FOLDS + 1)))
        for i in range(1, N_FOLDS + 1)
    ]

    folds = []

    for edge in fold_edges:
        val_start = inits[edge - 1]
        next_edge = min(
            edge + (fold_edges[1] - fold_edges[0] if len(fold_edges) > 1 else edge),
            len(inits),
        )

        val_mask = pl.col("validtime_dt") >= val_start

        if next_edge > edge:
            val_end = inits[next_edge - 1]
            val_mask = (
                (pl.col("validtime_dt") >= val_start)
                & (pl.col("validtime_dt") <= val_end)
            )

        df_tr = df_tv.filter(pl.col("validtime_dt") < val_start)
        df_va = df_tv.filter(val_mask)

        if len(df_tr) and len(df_va):
            folds.append((df_tr, df_va))

    return folds, df_test


def split_final_train_val(df_all, test_days=365, val_days=90):
    """
    Make a final train/val/test split:
    - Test period = last test_days
    - Validation = last val_days before test starts
    - Train = everything before validation
    """
    max_vt = df_all["validtime_dt"].max()
    test_start = max_vt - timedelta(days=test_days)
    val_start = test_start - timedelta(days=val_days)

    df_train_final = df_all.filter(pl.col("validtime_dt") < val_start)
    df_val_final = df_all.filter(
        (pl.col("validtime_dt") >= val_start)
        & (pl.col("validtime_dt") < test_start)
    )
    df_test_final = df_all.filter(pl.col("validtime_dt") >= test_start)

    return df_train_final, df_val_final, df_test_final


# =====================================================
# Preprocessor
# =====================================================

class ZScoreFill0WithMask(BaseEstimator, TransformerMixin):
    """
    Standardize one numeric feature and add a missing-value mask.
    """

    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.mean_, self.std_ = 0.0, 1.0

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float).ravel()
        m = ~np.isnan(x)

        if m.any():
            mu, sd = x[m].mean(), x[m].std(ddof=0)
            self.mean_, self.std_ = float(mu), float(sd if sd != 0 else 1.0)

        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float).ravel()
        miss = np.isnan(x)

        z = np.zeros_like(x)
        z[~miss] = (x[~miss] - self.mean_) / self.std_

        return np.column_stack([z, miss.astype(float)])

    def get_feature_names_out(self, input_features=None):
        return np.array([self.feature_name, f"{self.feature_name}_missing"])


def make_preprocessor(df_columns, met_vars, ensmean, time_feats, geo_feats):
    transformers = []

    for col in met_vars:
        if col in df_columns:
            transformers.append(
                (f"zfm_{col}", Pipeline([("zfm", ZScoreFill0WithMask(col))]), [col])
            )

    if ensmean in df_columns:
        transformers.append(
            ("zfm_ENS", Pipeline([("zfm", ZScoreFill0WithMask(ensmean))]), [ensmean])
        )

    passthrough = [c for c in time_feats + geo_feats if c in df_columns]

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return preproc, passthrough


def fit_preproc_on_train(
    df_tr_pl,
    feat_cols,
    met_vars,
    ensmean,
    time_feats,
    geo_feats,
    max_rows_for_fit=1_000_000,
):
    n_rows = df_tr_pl.height

    if n_rows > max_rows_for_fit:
        print(f"Fitting preprocessor on a sample of {max_rows_for_fit} / {n_rows} rows.")
        df_fit = df_tr_pl.sample(max_rows_for_fit, with_replacement=False, seed=SEED)
    else:
        print(f"Fitting preprocessor on all {n_rows} rows.")
        df_fit = df_tr_pl

    x_tr = df_fit.select(feat_cols).to_pandas()

    preproc, _ = make_preprocessor(
        x_tr.columns,
        met_vars,
        ensmean,
        time_feats,
        geo_feats,
    )

    preproc.fit(x_tr)

    return preproc, preproc.get_feature_names_out()


# =====================================================
# Coastline helpers
# =====================================================

def add_dist_sea_to_stations_STRtree(
    stations: pd.DataFrame,
    coast_shp: Path,
    metric_crs: str,
) -> pd.DataFrame:
    """
    Add dist_sea in km using STRtree nearest coastline segment.
    """
    coast = gpd.read_file(coast_shp)
    coast = coast[coast.geometry.notna() & ~coast.geometry.is_empty].copy()
    coast = coast.to_crs(metric_crs)

    geoms = list(coast.geometry.values)

    if len(geoms) == 0:
        raise RuntimeError("No valid coastline geometries loaded.")

    tree = STRtree(geoms)

    gdf_stn = gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    ).to_crs(metric_crs)

    pts = list(gdf_stn.geometry.values)
    dists_m = np.empty(len(pts), dtype=float)

    for i, p in enumerate(pts):
        nearest = tree.nearest(p)

        if isinstance(nearest, (int, np.integer)):
            nearest = geoms[int(nearest)]

        dists_m[i] = p.distance(nearest)

    out = stations.copy()
    out["dist_sea"] = dists_m / 1000.0

    return out


# =====================================================
# Graph building
# =====================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2

    return 2 * R * np.arcsin(np.sqrt(a))


def rbf(x, c, w):
    x = np.asarray(x)[:, None]
    c = np.asarray(c)[None, :]
    w = np.asarray(w)[None, :]

    return np.exp(-((x - c) ** 2) / (2 * w ** 2))


def build_graph(
    stations,
    k=10,
    radius_km=275.0,
    gamma=5.0,
    coast_threshold_km=20.0,
    use_land_sea_mask=True,
):
    """
    Build station graph using k-nearest neighbors, radius filtering,
    elevation-aware effective distance, and optional coastal mask.
    """
    stn = stations.reset_index(drop=True)

    lat = stn["lat"].to_numpy()
    lon = stn["lon"].to_numpy()
    elev = stn["elev"].to_numpy()

    n = len(stn)

    latm = np.repeat(lat[:, None], n, 1)
    lonm = np.repeat(lon[:, None], n, 1)

    d_geo = haversine_km(latm, lonm, latm.T, lonm.T)
    np.fill_diagonal(d_geo, np.inf)

    dz = elev[:, None] - elev[None, :]
    d_eff = np.sqrt(d_geo ** 2 + (gamma * (np.abs(dz) / 1000.0)) ** 2)

    nn_idx = np.argsort(d_eff, 1)[:, :k]

    row = np.repeat(np.arange(n), k)
    col = nn_idx.flatten()

    if use_land_sea_mask:
        dist_sea = stn["dist_sea"].to_numpy()

        if not np.isfinite(dist_sea).all():
            raise ValueError("dist_sea contains NaNs/inf. Coastline distance failed.")

        is_coast = dist_sea <= coast_threshold_km
        same_coast = is_coast[row] == is_coast[col]
        mask = (d_geo[row, col] <= radius_km) & same_coast
    else:
        mask = d_geo[row, col] <= radius_km

    row, col = row[mask], col[mask]

    edge_index = np.vstack(
        [
            np.concatenate([row, col]),
            np.concatenate([col, row]),
        ]
    )

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
    stn = stations.reset_index(drop=True)

    dist_sea = stn["dist_sea"].to_numpy()
    is_coast = dist_sea <= coast_threshold_km

    src = edge_index[0]
    dst = edge_index[1]

    bad = is_coast[src] != is_coast[dst]

    print("Edges violating coast mask:", bad.sum())

    if bad.sum() > 0:
        idxs = np.where(bad)[0][:10]

        print("Example problematic edges:")
        for i in idxs:
            s = src[i]
            t = dst[i]
            print(
                f"  edge {i}: {s} -> {t}, "
                f"dist_sea_src={dist_sea[s]:.1f}, "
                f"dist_sea_dst={dist_sea[t]:.1f}"
            )


# =====================================================
# PyG snapshots
# =====================================================

def make_node_index(stations_df):
    return dict(zip(stations_df["SID"].astype(str), range(len(stations_df))))


def split_to_snapshots(
    df_pl,
    preproc,
    feat_cols,
    lab_col,
    fc_col,
    sid_to_idx,
    edge_index,
    edge_attr,
):
    n = len(sid_to_idx)
    node_idx_tensor = torch.arange(n, dtype=torch.long)

    data_list = []

    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    for _, group_df in df_pl.group_by("validtime", maintain_order=True):
        x_pd = group_df.select(feat_cols).to_pandas()
        x_np = preproc.transform(x_pd).astype(np.float32)

        y_np = (group_df[lab_col] - group_df[fc_col]).to_numpy()

        sid = group_df["SID"].cast(pl.Utf8).to_numpy()
        nodes = np.array([sid_to_idx.get(str(s)) for s in sid], dtype=object)

        if np.any(pd.isna(nodes)):
            missing = sorted({str(s) for s, node in zip(sid, nodes) if node is None})
            raise ValueError(f"Some SIDs are missing from sid_to_idx: {missing[:10]}")

        nodes = nodes.astype(int)

        x_t = np.zeros((n, x_np.shape[1]), np.float32)
        y_t = np.zeros(n, np.float32)
        m_t = np.zeros(n, np.float32)

        x_t[nodes] = x_np
        y_t[nodes] = y_np
        m_t[nodes] = 1.0

        data = Data(
            x=torch.tensor(x_t),
            y=torch.tensor(y_t),
            y_mask=torch.tensor(m_t),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            node_idx=node_idx_tensor.clone(),
        )

        data_list.append(data)

    return data_list


# =====================================================
# Model
# =====================================================

class TempBiasGATv2(nn.Module):
    """
    GATv2 model for station-level temperature bias correction.
    """

    def __init__(
        self,
        in_channels,
        edge_dim,
        num_stations,
        hidden=128,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        emb_dim = hidden // 2
        feat_dim = hidden - emb_dim

        self.station_emb = nn.Embedding(num_stations, emb_dim)

        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, feat_dim),
            nn.ReLU(),
        )

        self.conv1 = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden,
            out_channels=hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.lin_out = nn.Linear(hidden, 1)

    def forward(self, d):
        feat = self.input_mlp(d.x)
        emb = self.station_emb(d.node_idx)

        x = torch.cat([feat, emb], dim=-1)

        h = self.conv1(x, d.edge_index, d.edge_attr)
        h = self.act(h)
        h = self.drop(h)
        x = self.norm1(x + h)

        h = self.conv2(x, d.edge_index, d.edge_attr)
        h = self.act(h)
        h = self.drop(h)
        x = self.norm2(x + h)

        out = self.lin_out(x).squeeze(-1)

        return out


# =====================================================
# Training / evaluation helpers
# =====================================================

def train_epoch(m, ldr, opt, dv):
    m.train()

    se_sum = 0.0
    m_sum = 0.0

    for d in ldr:
        d = d.to(dv)

        opt.zero_grad()

        p = m(d)

        e2 = (p - d.y) ** 2
        loss = (e2 * d.y_mask).sum() / (d.y_mask.sum() + 1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=2.0)
        opt.step()

        se_sum += (e2 * d.y_mask).sum().item()
        m_sum += d.y_mask.sum().item()

    if m_sum == 0:
        return float("nan")

    return math.sqrt(se_sum / (m_sum + 1e-8))


@torch.no_grad()
def eval_rmse(m, ldr, dv):
    m.eval()

    se_sum = 0.0
    m_sum = 0.0

    for d in ldr:
        d = d.to(dv)

        p = m(d)
        e2 = (p - d.y) ** 2

        se_sum += (e2 * d.y_mask).sum().item()
        m_sum += d.y_mask.sum().item()

    if m_sum == 0:
        return float("nan")

    return math.sqrt(se_sum / (m_sum + 1e-8))


# =====================================================
# Optuna CV + graph search + final training
# =====================================================

def cv_trial(
    trial,
    df_folds,
    stations,
    preproc_global,
    in_ch,
    num_stations,
    sid_to_idx,
    device,
):
    print(f"\n===== Starting Optuna trial {trial.number} =====")

    global_step = 0

    k = trial.suggest_categorical("k", [5, 8, 10, 12, 15, 20])
    radius_km = trial.suggest_categorical("radius_km", [150, 200, 250, 300, 350, 400, 500])

    hidden = trial.suggest_categorical("hidden", [64, 96, 128, 160, 192])
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    lr = trial.suggest_float("lr", 5e-5, 8e-4, log=True)
    wd = trial.suggest_float("wd", 1e-7, 5e-3, log=True)

    bs = trial.suggest_categorical("bs", [8, 16, 24, 32])

    max_epochs = 120
    patience = 10
    min_delta = 1e-4

    try:
        edge_index, edge_attr = build_graph(
            stations,
            k=k,
            radius_km=radius_km,
            coast_threshold_km=COAST_THRESHOLD_KM,
            use_land_sea_mask=True,
        )

        edge_dim = edge_attr.shape[1]

        print(
            f"Trial {trial.number}: graph with "
            f"{edge_index.shape[1]} edges, edge_dim={edge_dim}"
        )

        check_coastal_edges(
            stations,
            edge_index,
            coast_threshold_km=COAST_THRESHOLD_KM,
        )

        fold_bests = []

        for fold_id, (df_tr, df_va) in enumerate(df_folds):
            print(f"  Preparing snapshots for fold {fold_id}...")

            tr_snaps = split_to_snapshots(
                df_tr,
                preproc_global,
                FEATS,
                LABEL_OBS,
                TEMP_FC,
                sid_to_idx,
                edge_index,
                edge_attr,
            )

            va_snaps = split_to_snapshots(
                df_va,
                preproc_global,
                FEATS,
                LABEL_OBS,
                TEMP_FC,
                sid_to_idx,
                edge_index,
                edge_attr,
            )

            train_loader = DataLoader(tr_snaps, batch_size=bs, shuffle=True)
            val_loader = DataLoader(va_snaps, batch_size=bs, shuffle=False)

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

            for ep in range(max_epochs):
                train_loss = train_epoch(model, train_loader, opt, device)
                val_rmse = eval_rmse(model, val_loader, device)

                print(
                    f"  Fold {fold_id} Ep {ep:03d}: "
                    f"train={train_loss:.4f} val_globalRMSE={val_rmse:.4f}"
                )

                if val_rmse < (best_val - min_delta):
                    best_val = val_rmse
                    bad = 0
                else:
                    bad += 1

                    if bad > patience:
                        break

                trial.report(best_val, step=global_step)
                global_step += 1

                if trial.should_prune():
                    print("Pruning trial due to poor performance.")
                    raise TrialPruned()

            fold_bests.append(best_val)

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

        raise


def run_optuna(
    df_all,
    folds_pl,
    df_test_pl,
    in_ch,
    num_stations,
    device,
    preproc_global,
    stations,
    sid_to_idx,
    model_base_dir: Path,
    n_trials: int,
    test_days: int,
    val_days: int,
):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=3,
            interval_steps=1,
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
        n_trials=n_trials,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Best trial:", study.best_value, study.best_params)

    model_base_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    out_dir = model_base_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    with open(out_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"Saved Optuna results to {out_dir}")

    best_k = study.best_params["k"]
    best_radius = study.best_params["radius_km"]

    edge_index, edge_attr = build_graph(
        stations,
        k=best_k,
        radius_km=best_radius,
        coast_threshold_km=COAST_THRESHOLD_KM,
        use_land_sea_mask=True,
    )

    edge_dim = edge_attr.shape[1]

    df_train_final, df_val_final, df_test_final = split_final_train_val(
        df_all,
        test_days=test_days,
        val_days=val_days,
    )

    print("Final split sizes:")
    print("  train rows:", df_train_final.height)
    print("  val rows:  ", df_val_final.height)
    print("  test rows: ", df_test_final.height)

    print("Building train snapshots...")
    tr_snaps = split_to_snapshots(
        df_train_final,
        preproc_global,
        FEATS,
        LABEL_OBS,
        TEMP_FC,
        sid_to_idx,
        edge_index,
        edge_attr,
    )

    print("Building val snapshots...")
    va_snaps = split_to_snapshots(
        df_val_final,
        preproc_global,
        FEATS,
        LABEL_OBS,
        TEMP_FC,
        sid_to_idx,
        edge_index,
        edge_attr,
    )

    print("Building test snapshots...")
    ts_snaps = split_to_snapshots(
        df_test_final,
        preproc_global,
        FEATS,
        LABEL_OBS,
        TEMP_FC,
        sid_to_idx,
        edge_index,
        edge_attr,
    )

    bs = study.best_params["bs"]

    tr_loader = DataLoader(tr_snaps, batch_size=bs, shuffle=True)
    val_loader = DataLoader(va_snaps, batch_size=bs, shuffle=False)
    ts_loader = DataLoader(ts_snaps, batch_size=bs, shuffle=False)

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

    max_epochs = 200
    patience = 15
    min_delta = 1e-4

    best_val = float("inf")
    bad = 0
    best_state = None

    for ep in range(max_epochs):
        train_rmse = train_epoch(model, tr_loader, opt, device)
        val_rmse = eval_rmse(model, val_loader, device)

        print(
            f"Final training | Epoch {ep:03d}: "
            f"train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}"
        )

        if val_rmse < (best_val - min_delta):
            best_val = val_rmse
            bad = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1

            if bad >= patience:
                print(
                    f"Early stopping at epoch {ep:03d} "
                    f"(best val_rmse={best_val:.4f})"
                )
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best val RMSE during final training: {best_val:.4f}")

    test_rmse = eval_rmse(model, ts_loader, device)

    print(f"Final test RMSE = {test_rmse:.4f}")

    torch.save(model.state_dict(), out_dir / "gnn_model.pt")
    joblib.dump(preproc_global, out_dir / "preproc.joblib")
    stations.to_parquet(out_dir / "stations.parquet")

    with open(out_dir / "sid_to_idx.json", "w") as f:
        json.dump(sid_to_idx, f)

    joblib.dump(edge_index, out_dir / "edge_index.pkl")
    joblib.dump(edge_attr, out_dir / "edge_attr.pkl")

    with open(out_dir / "graph_params.json", "w") as f:
        json.dump(
            {
                **study.best_params,
                "test_days": test_days,
                "val_days": val_days,
                "coast_threshold_km": COAST_THRESHOLD_KM,
                "early_stopping": {
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "min_delta": min_delta,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved everything to: {out_dir}")

    return model, study, out_dir


def train_without_optuna(
    df_train_final: pl.DataFrame,
    df_val_final: pl.DataFrame,
    df_test_final: pl.DataFrame,
    in_ch: int,
    num_stations: int,
    device: str,
    preproc_global,
    stations: pd.DataFrame,
    sid_to_idx: dict,
    model_base_dir: Path,
    out_name: str,
    fixed_params_json: str | None = None,
    val_days: int = 90,
):
    print("\n=== Running FINAL training without Optuna ===")

    default_params = {
        "k": 12,
        "radius_km": 500,
        "hidden": 192,
        "heads": 8,
        "dropout": 0.1384634953194369,
        "lr": 0.0006517770362188271,
        "wd": 0.0043741181457823495,
        "bs": 16,
    }

    if fixed_params_json is not None:
        fixed_params_path = Path(fixed_params_json)

        if not fixed_params_path.exists():
            raise FileNotFoundError(f"Fixed params JSON not found: {fixed_params_path}")

        with open(fixed_params_path, "r") as f:
            best_params = json.load(f)

        print(f"Loaded fixed params from: {fixed_params_path}")
    else:
        best_params = default_params
        print("Using default fixed params in script.")

    k = best_params["k"]
    radius_km = best_params["radius_km"]
    hidden = best_params["hidden"]
    heads = best_params["heads"]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    wd = best_params["wd"]
    bs = best_params["bs"]

    max_epochs = 200
    patience = 15
    min_delta = 1e-4

    edge_index, edge_attr = build_graph(
        stations,
        k=k,
        radius_km=radius_km,
        coast_threshold_km=COAST_THRESHOLD_KM,
        use_land_sea_mask=True,
    )

    edge_dim = edge_attr.shape[1]

    print(f"Graph: edges={edge_index.shape[1]} edge_dim={edge_dim}")

    check_coastal_edges(
        stations,
        edge_index,
        coast_threshold_km=COAST_THRESHOLD_KM,
    )

    print("Building train snapshots...")
    tr_snaps = split_to_snapshots(
        df_train_final,
        preproc_global,
        FEATS,
        LABEL_OBS,
        TEMP_FC,
        sid_to_idx,
        edge_index,
        edge_attr,
    )

    print("Building val snapshots...")
    va_snaps = split_to_snapshots(
        df_val_final,
        preproc_global,
        FEATS,
        LABEL_OBS,
        TEMP_FC,
        sid_to_idx,
        edge_index,
        edge_attr,
    )

    print("Building test snapshots...")
    ts_snaps = split_to_snapshots(
        df_test_final,
        preproc_global,
        FEATS,
        LABEL_OBS,
        TEMP_FC,
        sid_to_idx,
        edge_index,
        edge_attr,
    )

    train_loader = DataLoader(tr_snaps, batch_size=bs, shuffle=True)
    val_loader = DataLoader(va_snaps, batch_size=bs, shuffle=False)
    test_loader = DataLoader(ts_snaps, batch_size=bs, shuffle=False)

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
    best_state = None

    for ep in range(max_epochs):
        train_rmse = train_epoch(model, train_loader, opt, device)
        val_rmse = eval_rmse(model, val_loader, device)

        print(f"Epoch {ep:03d}: train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < best_val - min_delta:
            best_val = val_rmse
            bad = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1

            if bad >= patience:
                print(f"Early stopping at epoch {ep:03d} (best val_rmse={best_val:.4f})")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse = eval_rmse(model, test_loader, device)

    print(f"\n=== FINAL TEST RMSE GLOBAL: {test_rmse:.4f} ===\n")

    model_base_dir.mkdir(parents=True, exist_ok=True)

    out_dir = model_base_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "gnn_model.pt")
    joblib.dump(preproc_global, out_dir / "preproc.joblib")
    stations.to_parquet(out_dir / "stations.parquet")

    with open(out_dir / "sid_to_idx.json", "w") as f:
        json.dump(sid_to_idx, f)

    joblib.dump(edge_index, out_dir / "edge_index.pkl")
    joblib.dump(edge_attr, out_dir / "edge_attr.pkl")

    with open(out_dir / "graph_params.json", "w") as f:
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
                "val_days": val_days,
                "coast_threshold_km": COAST_THRESHOLD_KM,
                "early_stopping": {
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "min_delta": min_delta,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved everything to: {out_dir}")

    return model, out_dir


# =====================================================
# Main
# =====================================================

def main():
    global DEVICE, SEED, PATH, MODEL_BASE_DIR
    global COAST_SHP, COAST_METRIC_CRS, COAST_THRESHOLD_KM

    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = args.threads
    os.environ["MKL_NUM_THREADS"] = args.threads

    SEED = args.seed
    set_seed(SEED)

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    PATH = resolve_input_path(args.input)
    MODEL_BASE_DIR = Path(args.model_base_dir)

    COAST_SHP = Path(args.coast_shp)
    COAST_METRIC_CRS = args.coast_metric_crs
    COAST_THRESHOLD_KM = args.coast_threshold_km

    if not COAST_SHP.exists():
        raise FileNotFoundError(f"Coastline shapefile not found: {COAST_SHP}")

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Input data: {PATH}")
    print(f"[INFO] Model base directory: {MODEL_BASE_DIR}")
    print(f"[INFO] Coastline shapefile: {COAST_SHP}")
    print(f"[INFO] Coast metric CRS: {COAST_METRIC_CRS}")
    print(f"[INFO] Coast threshold km: {COAST_THRESHOLD_KM}")
    print(f"[INFO] Run Optuna: {args.run_optuna}")

    df_all = load_dataset(PATH, FEATS, LABEL_OBS, TEMP_FC, STATION_ID_COL)

    df_train_final, df_val_final, df_test_final = split_final_train_val(
        df_all,
        test_days=args.test_days,
        val_days=args.val_days,
    )

    print("Final split sizes:")
    print("  train rows:", df_train_final.height)
    print("  val rows:  ", df_val_final.height)
    print("  test rows: ", df_test_final.height)

    stations = (
        df_all.select(["SID", "lat", "lon", "elev"])
        .unique(subset=["SID"])
        .to_pandas()
    )

    stations["SID"] = stations["SID"].astype(str)

    stations = add_dist_sea_to_stations_STRtree(
        stations,
        coast_shp=COAST_SHP,
        metric_crs=COAST_METRIC_CRS,
    )

    print(
        "dist_sea stats km:",
        float(stations["dist_sea"].min()),
        float(stations["dist_sea"].median()),
        float(stations["dist_sea"].max()),
    )

    stations = stations.sort_values("SID").reset_index(drop=True)

    sid_to_idx = make_node_index(stations)
    num_stations = len(sid_to_idx)

    preproc_global, out_cols_global = fit_preproc_on_train(
        df_train_final,
        FEATS,
        MET_VARS,
        ENSMEAN,
        TIME_FEATS,
        GEO_FEATS,
        max_rows_for_fit=args.max_rows_preproc_fit,
    )

    in_ch = len(out_cols_global)

    print("in_channels:", in_ch)

    if args.run_optuna:
        folds_pl, df_test_pl = split_trainval_test(
            df_all,
            TEST_DAYS=args.test_days,
            TRAIN_START_DT=None,
            N_FOLDS=args.n_folds,
        )

        model, study, out_dir = run_optuna(
            df_all=df_all,
            folds_pl=folds_pl,
            df_test_pl=df_test_pl,
            in_ch=in_ch,
            num_stations=num_stations,
            device=DEVICE,
            preproc_global=preproc_global,
            stations=stations,
            sid_to_idx=sid_to_idx,
            model_base_dir=MODEL_BASE_DIR,
            n_trials=args.n_trials,
            test_days=args.test_days,
            val_days=args.val_days,
        )

    else:
        model, out_dir = train_without_optuna(
            df_train_final=df_train_final,
            df_val_final=df_val_final,
            df_test_final=df_test_final,
            in_ch=in_ch,
            num_stations=num_stations,
            device=DEVICE,
            preproc_global=preproc_global,
            stations=stations,
            sid_to_idx=sid_to_idx,
            model_base_dir=MODEL_BASE_DIR,
            out_name=args.out_name,
            fixed_params_json=args.fixed_params_json,
            val_days=args.val_days,
        )

    print(f"[DONE] Saved run artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
