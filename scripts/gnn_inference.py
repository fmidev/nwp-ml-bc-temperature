import argparse
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
#  Global config set from command-line args
# ===========================================

DEVICE = None

DATA_DIR = None
RUN_DIR = None
MODEL_PATH = None
BEST_PARAMS_PATH = None
PREPROC_PATH = None
STATIONS_PATH = None
SID_MAP_PATH = None
EDGE_INDEX_PATH = None
EDGE_ATTR_PATH = None

MODEL_TAG = None
OUTDIR = None
CORR_COL = None

SPLIT_COLUMN = "validtime"

TEMP_FC = "T2"
LABEL_OBS = "obs_TA"
STATION_ID_COL = "SID"

FEATS = [
    "T2", "SKT", "MX2T", "MN2T", "D2", "T_925", "MSL", "U10", "V10",
    "T2_M1", "T_925_M1", "LCC", "MCC", "T2_ENSMEAN_MA1",
    "sin_hod", "cos_hod", "sin_doy", "cos_doy",
    "analysishour", "leadtime", "lon", "lat", "elev",
]

ID = ["SID", "analysistime", "validtime", "leadtime"]


# ===========================================
#  Argument parsing
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained GNN bias-correction model."
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help=(
            "Directory containing input parquet files named like "
            "ml_data_full_2024-01.parquet."
        ),
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=str,
        help=(
            "Directory containing saved GNN artifacts: gnn_model.pt, "
            "graph_params.json, preproc.joblib, stations.parquet, "
            "sid_to_idx.json, edge_index.pkl, edge_attr.pkl."
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
        default="full_gnn_gat_lsm",
        type=str,
        help="Model tag used in corrected_<tag> column and output filenames.",
    )

    parser.add_argument(
        "--start-year",
        default=2024,
        type=int,
        help="First year to process, inclusive. Default: 2024.",
    )

    parser.add_argument(
        "--end-year",
        default=2025,
        type=int,
        help="Last year to process, inclusive. Default: 2025.",
    )

    parser.add_argument(
        "--eval-start",
        default="2024-09-01",
        type=str,
        help="Start date for final RMSE summary, inclusive. Default: 2024-09-01.",
    )

    parser.add_argument(
        "--eval-end",
        default="2025-09-01",
        type=str,
        help="End date for final RMSE summary, exclusive. Default: 2025-09-01.",
    )

    parser.add_argument(
        "--threads",
        default="16",
        type=str,
        help="Thread count for OMP_NUM_THREADS and MKL_NUM_THREADS. Default: 16.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference. Default: auto.",
    )

    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="Batch size for final graph-snapshot RMSE evaluation. Default: 16.",
    )

    return parser.parse_args()


# ===========================================
#  Custom transformer, must match training
# ===========================================

class ZScoreFill0WithMask(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer used inside the saved preprocessing pipeline.
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


# ===========================================
#  Helpers for data loading and splitting
# ===========================================

def safe_month(col):
    """
    Extract month number from a datetime-like column.
    """
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False, exact=False)
        .dt.month()
    )


def load_dataset(path, feat_cols, lab_col, fc_col, station_id_col):
    """
    Read training/evaluation-style data from parquet files.
    """
    lf = (
        pl.scan_parquet(str(path))
        .select(feat_cols + [lab_col, "analysistime", station_id_col, "validtime"])
        .with_columns(
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
        .filter(pl.col(lab_col).is_not_null() & pl.col(fc_col).is_not_null())
    )

    return lf.collect(engine="streaming")


def split_trainval_test(df, TEST_DAYS=365, TRAIN_START_DT=None, N_FOLDS=3):
    """
    Split a time-indexed DataFrame into rolling train/validation folds and test.
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
        raise ValueError(
            "Not enough initializations for requested N_FOLDS after applying TRAIN_START_DT."
        )

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


# ===========================================
#  Evaluation functions
# ===========================================

def masked_rmse(pred, y, m):
    """
    Compute RMSE only over valid station-node targets.
    """
    return torch.sqrt(((pred - y) ** 2 * m).sum() / (m.sum() + 1e-8))


@torch.no_grad()
def eval_rmse(m, ldr, dv):
    """
    Evaluate masked RMSE over a DataLoader of graph snapshots.
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

    return tot / n if n else float("nan")


def split_to_snapshots(df_pl, preproc, feat_cols, lab_col, fc_col, sid_to_idx, edge_index, edge_attr):
    """
    Convert a tabular Polars DataFrame into PyTorch Geometric graph snapshots.
    """
    n_nodes = len(sid_to_idx)
    node_idx_tensor = torch.arange(n_nodes, dtype=torch.long)
    data_list = []

    edge_index_np = np.asarray(edge_index, dtype=np.int64)
    edge_attr_np = np.asarray(edge_attr, dtype=np.float32)

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

        x_t = np.zeros((n_nodes, x_np.shape[1]), np.float32)
        y_t = np.zeros(n_nodes, np.float32)
        m_t = np.zeros(n_nodes, np.float32)

        x_t[nodes] = x_np
        y_t[nodes] = y_np
        m_t[nodes] = 1.0

        data = Data(
            x=torch.tensor(x_t),
            y=torch.tensor(y_t),
            y_mask=torch.tensor(m_t),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr_np, dtype=torch.float32),
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
    """

    def __init__(self, in_channels, edge_dim, num_stations, hidden=128, heads=4, dropout=0.2):
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

        return self.lin_out(x).squeeze(-1)


# ===========================================
#  GNN inference
# ===========================================

def gnn_predict_bias(df_pl, model, preproc, sid_to_idx, edge_index, edge_attr):
    """
    Run GNN inference for a tabular set of forecast rows.
    """
    x_pd = df_pl.select(FEATS).to_pandas()
    x_np = preproc.transform(x_pd).astype(np.float32, copy=False)

    sid_all = df_pl["SID"].cast(pl.Utf8).to_numpy()
    vt_all = df_pl["validtime"].to_numpy()
    anal_all = df_pl["analysistime"].to_numpy()
    lt_all = df_pl["leadtime"].to_numpy()
    raw_fc_all = df_pl[TEMP_FC].to_numpy()

    if LABEL_OBS in df_pl.columns:
        obs_all = df_pl[LABEL_OBS].to_numpy()
    else:
        obs_all = np.full_like(raw_fc_all, np.nan, dtype=np.float32)

    node_idx_all = np.array(
        [sid_to_idx.get(str(s)) for s in sid_all],
        dtype=object,
    )

    if np.any(pd.isna(node_idx_all)):
        missing = sorted({str(s) for s, node in zip(sid_all, node_idx_all) if node is None})
        raise ValueError(f"Some SIDs in inference data are missing from sid_to_idx: {missing[:10]}")

    node_idx_all = node_idx_all.astype(int)

    order = np.argsort(vt_all)
    vt_sorted = vt_all[order]

    unique_times, start_idx = np.unique(vt_sorted, return_index=True)

    n_nodes = len(sid_to_idx)
    n_features = x_np.shape[1]

    edge_index_t = torch.tensor(
        np.asarray(edge_index, dtype=np.int64),
        dtype=torch.long,
        device=DEVICE,
    )

    edge_attr_t = torch.tensor(
        np.asarray(edge_attr, dtype=np.float32),
        dtype=torch.float32,
        device=DEVICE,
    )

    node_idx_tensor = torch.arange(n_nodes, dtype=torch.long, device=DEVICE)

    records = []

    for i, _ in enumerate(unique_times):
        s = start_idx[i]
        e = start_idx[i + 1] if i + 1 < len(start_idx) else len(order)

        idx = order[s:e]

        x_t = np.zeros((n_nodes, n_features), np.float32)

        nodes = node_idx_all[idx]
        x_t[nodes] = x_np[idx]

        data = Data(
            x=torch.tensor(x_t, dtype=torch.float32, device=DEVICE),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            node_idx=node_idx_tensor,
        )

        with torch.no_grad():
            bias_full = model(data).cpu().numpy()

        bias_rows = bias_full[nodes]
        raw_fc = raw_fc_all[idx]
        obs = obs_all[idx]

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
#  Artifact loading
# ===========================================

def check_required_artifacts():
    required = [
        MODEL_PATH,
        BEST_PARAMS_PATH,
        PREPROC_PATH,
        STATIONS_PATH,
        SID_MAP_PATH,
        EDGE_INDEX_PATH,
        EDGE_ATTR_PATH,
    ]

    missing = [p for p in required if not p.exists()]

    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required artifact file(s):\n{msg}")


def load_artifacts():
    print("Loading saved artifacts...")

    check_required_artifacts()

    preproc = joblib.load(PREPROC_PATH)

    # Check that station artifact exists and is readable.
    pd.read_parquet(STATIONS_PATH)

    with open(SID_MAP_PATH, "r") as f:
        sid_to_idx = json.load(f)

    # JSON keys are strings, but make this explicit.
    sid_to_idx = {str(k): int(v) for k, v in sid_to_idx.items()}

    edge_index = joblib.load(EDGE_INDEX_PATH)
    edge_attr = joblib.load(EDGE_ATTR_PATH)

    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_attr = np.asarray(edge_attr, dtype=np.float32)

    with open(BEST_PARAMS_PATH, "r") as f:
        params = json.load(f)

    print("Loading trained GNN model...")

    num_stations = len(sid_to_idx)
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

    return model, preproc, sid_to_idx, edge_index, edge_attr


# ===========================================
#  Main
# ===========================================

def main():
    global DEVICE
    global DATA_DIR, RUN_DIR, MODEL_PATH, BEST_PARAMS_PATH, PREPROC_PATH
    global STATIONS_PATH, SID_MAP_PATH, EDGE_INDEX_PATH, EDGE_ATTR_PATH
    global MODEL_TAG, OUTDIR, CORR_COL

    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = args.threads
    os.environ["MKL_NUM_THREADS"] = args.threads

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    DATA_DIR = Path(args.data_dir)
    RUN_DIR = Path(args.run_dir)

    MODEL_PATH = RUN_DIR / "gnn_model.pt"
    BEST_PARAMS_PATH = RUN_DIR / "graph_params.json"
    PREPROC_PATH = RUN_DIR / "preproc.joblib"
    STATIONS_PATH = RUN_DIR / "stations.parquet"
    SID_MAP_PATH = RUN_DIR / "sid_to_idx.json"
    EDGE_INDEX_PATH = RUN_DIR / "edge_index.pkl"
    EDGE_ATTR_PATH = RUN_DIR / "edge_attr.pkl"

    MODEL_TAG = args.model_tag
    OUTDIR = Path(args.output_dir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    CORR_COL = f"corrected_{MODEL_TAG}"

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Run directory not found: {RUN_DIR}")

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Data directory: {DATA_DIR}")
    print(f"[INFO] Run directory: {RUN_DIR}")
    print(f"[INFO] Output directory: {OUTDIR}")
    print(f"[INFO] Model tag: {MODEL_TAG}")
    print(f"[INFO] Corrected column: {CORR_COL}")

    model, preproc, sid_to_idx, edge_index, edge_attr = load_artifacts()

    # ---------------------------------------------------
    # Yearly inference outputs
    # ---------------------------------------------------

    needed = list(set(ID + FEATS + [LABEL_OBS]))

    for year in range(args.start_year, args.end_year + 1):
        print(f"\nProcessing year {year}...")

        files = sorted(DATA_DIR.glob(f"ml_data_full_{year}-*.parquet"))

        if not files:
            print(f"No files found for year {year}; skipping.")
            continue

        accumulated = []

        for f in files:
            lf = pl.scan_parquet(str(f)).select(needed)

            lf = lf.with_columns(pl.col(SPLIT_COLUMN).cast(pl.Utf8))

            lf = lf.filter(
                pl.col(SPLIT_COLUMN)
                .str.strptime(pl.Datetime, strict=False, exact=False)
                .dt.year()
                == year
            )

            if lf.head(1).collect(engine="streaming").height == 0:
                continue

            lf = (
                lf.sort(["SID", "analysistime", "leadtime", "validtime"])
                .unique(subset=["SID", "analysistime", "leadtime"], keep="last")
            )

            df = lf.collect(engine="streaming")

            if df.height == 0:
                continue

            df = df.filter(pl.col(TEMP_FC).is_not_null())

            if df.height == 0:
                continue

            df_pred = gnn_predict_bias(
                df,
                model,
                preproc,
                sid_to_idx,
                edge_index,
                edge_attr,
            )

            accumulated.append(df_pred)

        if not accumulated:
            print(f"No rows collected for year {year}; skipping.")
            continue

        all_df = pl.concat(accumulated, how="vertical_relaxed")

        all_df = all_df.filter(
            pl.col(LABEL_OBS).is_not_null()
            & pl.col(CORR_COL).is_not_null()
        )

        all_df = all_df.with_columns(month=safe_month("validtime"))

        tag = f"{SPLIT_COLUMN}_{MODEL_TAG}_{year}"
        out_path = OUTDIR / f"eval_rows_{tag}.parquet"

        all_df.write_parquet(out_path)

        print(f"Saved to: {out_path}")

    # ---------------------------------------------------
    # Final evaluation summary
    # ---------------------------------------------------

    print(f"\nBuilding evaluation DataFrame for {args.eval_start}..{args.eval_end}...")

    lf_all = pl.scan_parquet(str(DATA_DIR / "ml_data_full_*.parquet")).select(needed)

    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)

    lf_all = (
        lf_all.with_columns(pl.col("validtime").cast(pl.Utf8))
        .filter(
            pl.col("validtime")
            .str.strptime(pl.Datetime, strict=False, exact=False)
            .is_between(
                pl.datetime(eval_start.year, eval_start.month, eval_start.day),
                pl.datetime(eval_end.year, eval_end.month, eval_end.day),
                closed="left",
            )
        )
    )

    lf_all = (
        lf_all.sort(["SID", "analysistime", "leadtime", "validtime"])
        .unique(subset=["SID", "analysistime", "leadtime"], keep="last")
    ).filter(
        pl.col(TEMP_FC).is_not_null()
        & pl.col(LABEL_OBS).is_not_null()
    )

    df_eval_pl = lf_all.collect(engine="streaming")

    print("Eval rows:", df_eval_pl.height)

    if df_eval_pl.height == 0:
        print("No rows for requested evaluation window; skipping RMSE calculation.")
        return

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

    eval_loader = DataLoader(snaps, batch_size=args.batch_size, shuffle=False)

    class ZeroBiasModel(nn.Module):
        def forward(self, d):
            return torch.zeros(d.x.size(0), device=d.x.device)

    zero_model = ZeroBiasModel().to(DEVICE)

    rmse_raw_style = eval_rmse(zero_model, eval_loader, DEVICE)
    rmse_gnn_style = eval_rmse(model, eval_loader, DEVICE)

    print(f"\nRMSE on {args.eval_start}..{args.eval_end}:")
    print(f"  Raw EC:         {rmse_raw_style:.3f}")
    print(f"  GNN corrected:  {rmse_gnn_style:.3f}")
    print(f"  Improvement:    {rmse_raw_style - rmse_gnn_style:.3f}")


if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    main()
