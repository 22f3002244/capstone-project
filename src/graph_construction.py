import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import os
import pickle
import warnings
warnings.filterwarnings("ignore")


def _ip_to_str(val) -> str:
    return str(val).strip()


def build_flow_graph(
    df: pd.DataFrame,
    edge_features: np.ndarray,
    labels: np.ndarray,
    multiclass: bool = False,
) -> Data:
    src_col = "src_ip"   if "src_ip"   in df.columns else "src_device"
    dst_col = "dst_ip"   if "dst_ip"   in df.columns else "dst_device"

    src_vals = df[src_col].apply(_ip_to_str).values
    dst_vals = df[dst_col].apply(_ip_to_str).values

    unique_nodes = sorted(set(src_vals) | set(dst_vals))
    node_to_idx  = {n: i for i, n in enumerate(unique_nodes)}
    n_nodes      = len(unique_nodes)

    n_edge_feats = edge_features.shape[1]
    x = torch.ones(n_nodes, n_edge_feats, dtype=torch.float)

    src_idx  = torch.tensor([node_to_idx[s] for s in src_vals], dtype=torch.long)
    dst_idx  = torch.tensor([node_to_idx[d] for d in dst_vals], dtype=torch.long)
    edge_idx = torch.stack([src_idx, dst_idx], dim=0)

    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    y         = torch.tensor(labels, dtype=torch.long)

    node_label_lists = [[] for _ in range(n_nodes)]
    for i, lbl in enumerate(y.tolist()):
        node_label_lists[src_idx[i].item()].append(lbl)
        node_label_lists[dst_idx[i].item()].append(lbl)
    node_y = torch.tensor(
        [Counter(lst).most_common(1)[0][0] if lst else 0 for lst in node_label_lists],
        dtype=torch.long,
    )

    data = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y)
    data.node_y      = node_y
    data.num_nodes   = n_nodes
    data.node_to_idx = node_to_idx

    print(f"[OK] Flow graph   : {n_nodes:>6} nodes | {edge_idx.shape[1]:>7} edges "
          f"| edge-feat dim={n_edge_feats}")
    return data


def build_knn_graph(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 7,
) -> Data:
    print(f"[INFO] Building k-NN graph (k={k}, n={len(features):,})...")
    knn = kneighbors_graph(features, k, mode="connectivity",
                           include_self=False, n_jobs=-1)
    rows, cols = knn.nonzero()
    edge_index  = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels,   dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    print(f"[OK] k-NN graph   : {x.shape[0]:>6} nodes | {edge_index.shape[1]:>7} edges")
    return data


def build_temporal_graph(
    features: np.ndarray,
    labels: np.ndarray,
    window: int = 50,
) -> Data:
    n = len(features)
    edges = []
    for i in range(n):
        for j in range(max(0, i - window), i):
            edges.append([j, i])
            edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels,   dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    print(f"[OK] Temporal graph: {n:>6} nodes | {edge_index.shape[1]:>7} edges (window={window})")
    return data


def build_hybrid_graph(
    df: pd.DataFrame,
    edge_features: np.ndarray,
    node_features: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> Data:
    flow_data = build_flow_graph(df, edge_features, labels)

    knn = kneighbors_graph(flow_data.x.numpy(), k, mode="connectivity",
                           include_self=False, n_jobs=-1)
    rows, cols = knn.nonzero()
    knn_edges = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

    combined   = torch.cat([flow_data.edge_index, knn_edges], dim=1)
    unique_e   = torch.unique(combined, dim=1)
    flow_data.edge_index = unique_e

    n_knn = knn_edges.shape[1]
    zero_attr = torch.zeros(n_knn, flow_data.edge_attr.shape[1])
    flow_data.edge_attr = torch.cat([flow_data.edge_attr, zero_attr], dim=0)

    print(f"[OK] Hybrid graph  : {flow_data.num_nodes:>6} nodes | {unique_e.shape[1]:>7} edges")
    return flow_data


class IoTGraphConstructor:

    def __init__(self, method: str = "flow", k_neighbors: int = 7):
        assert method in ("flow", "knn", "temporal", "hybrid"), \
            f"Unknown method '{method}'. Choose: flow | knn | temporal | hybrid"
        self.method      = method
        self.k_neighbors = k_neighbors

    def construct(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Data:
        if isinstance(features, pd.DataFrame):
            features = features.values

        if self.method == "flow":
            return build_flow_graph(df, features, labels)
        elif self.method == "knn":
            return build_knn_graph(features, labels, k=self.k_neighbors)
        elif self.method == "temporal":
            return build_temporal_graph(features, labels)
        elif self.method == "hybrid":
            return build_hybrid_graph(df, features, features, labels, k=self.k_neighbors)

    @staticmethod
    def add_masks(
        data: Data,
        split: tuple = (0.70, 0.15, 0.15),
        seed: int = 42,
    ) -> Data:
        n = data.y.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        t = int(n * split[0])
        v = int(n * split[1])

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask   = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)
        train_mask[idx[:t]]    = True
        val_mask[idx[t:t+v]]   = True
        test_mask[idx[t+v:]]   = True

        data.train_mask = train_mask
        data.val_mask   = val_mask
        data.test_mask  = test_mask

        if hasattr(data, "node_y"):
            n_nodes = data.node_y.shape[0]
            rng2 = np.random.default_rng(seed + 1)
            nidx = rng2.permutation(n_nodes)
            nt = int(n_nodes * split[0])
            nv = int(n_nodes * split[1])
            ntrain = torch.zeros(n_nodes, dtype=torch.bool)
            nval   = torch.zeros(n_nodes, dtype=torch.bool)
            ntest  = torch.zeros(n_nodes, dtype=torch.bool)
            ntrain[nidx[:nt]]    = True
            nval[nidx[nt:nt+nv]] = True
            ntest[nidx[nt+nv:]]  = True
            data.node_train_mask = ntrain
            data.node_val_mask   = nval
            data.node_test_mask  = ntest

        print(f"     Masks → train:{train_mask.sum():>6} | "
              f"val:{val_mask.sum():>5} | test:{test_mask.sum():>6}")
        return data

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"method": self.method, "k": self.k_neighbors}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.method      = state["method"]
        self.k_neighbors = state["k"]
