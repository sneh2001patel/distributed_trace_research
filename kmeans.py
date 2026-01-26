import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from networkx import random_shell_graph
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

from synthetic_graphs import generate_synthetic_graph, load_decoder


class LoadDataset(InMemoryDataset):
    """
    Loader for processed/SN_TT_data.pt produced by build_trace_graphs.py.
    Node features: [service_id, op_id, duration].
    Label: 0 for SN_Dataset, 1 for TT_Dataset.
    """

    def __init__(self, datapath="../processed/SN_data.pt") -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices

    def get(self, idx):
        return super().get(idx)


def extract_nodes(ds: InMemoryDataset):
    xs = []

    for i, graph in enumerate(ds):
        if graph.x is None:
            raise ValueError(f"Graph: {i} has no x")

        if graph.x.shape[1] != 3:
            raise ValueError(f"Expected 3 node feat got {graph.x.shape}")
        xs.append(graph.x.detach().cpu())

    return torch.cat(xs, dim=0)


def preprocess_sid_op_duration(X_real: torch.Tensor, X_syn: torch.Tensor):

    X_all = torch.cat([X_real, X_syn], dim=0)
    sid = X_all[:, 0].round().long()
    op = X_all[:, 1].round().long()
    dur = X_all[:, 2].float().unsqueeze(1)

    if (sid < 0).any() or (op < 0).any():
        raise ValueError("Found neg Ids in service_id or op_id")

    n_sid = int(sid.max().item()) + 1
    n_op = int(op.max().item()) + 1

    # one-hot encode categorical IDs
    sid_oh = torch.nn.functional.one_hot(sid, num_classes=n_sid).float()
    op_oh = torch.nn.functional.one_hot(op, num_classes=n_op).float()

    # standardize duration only (then we will standardize the whole vector too)
    dur_np = dur.numpy()
    dur_scaled = StandardScaler().fit_transform(dur_np)  # [N,1]

    # Combine
    X_feat = torch.cat(
        [sid_oh, op_oh, torch.from_numpy(dur_scaled).float()], dim=1
    ).numpy()

    origin = np.array([0] * len(X_real) + [1] * len(X_syn))  # 0=real, 1=synthetic
    return X_feat, origin


real = LoadDataset(datapath="./datasets/SN_data.pt")
synthetic = LoadDataset(datapath="./datasets/fixed_size/SN_synthetic.pt")

print(f"Real dataset: {real}")
print(f"Synthetic dataset: {synthetic}")
print(f"Num graphs real: {len(real)}")
print(f"Num graphs syn: {len(synthetic)}")

Xr = extract_nodes(real).float()
Xs = extract_nodes(synthetic).float()

print(Xr.shape, Xs.shape)
X_feat, origin = preprocess_sid_op_duration(Xr, Xs)

X_feat = StandardScaler().fit_transform(X_feat)

print(
    "Final feature matrix:",
    X_feat.shape,
    "(one-hot sid + one-hot op + scaled duration)",
)

k = 2
km = KMeans(n_clusters=k, n_init=20, random_state=42)
cluster_id = km.fit_predict(X_feat)

print("\n=== Cluster mixing (fraction synthetic per cluster) ===")
for c in range(k):
    idx = np.where(cluster_id == c)[0]
    frac_syn = origin[idx].mean() if len(idx) else float("nan")
    print(f"cluster {c:02d}: n={len(idx):6d}  frac_syn={frac_syn:.3f}")

nmi = normalized_mutual_info_score(origin, cluster_id)
print("\nNMI(origin vs cluster):", float(nmi), " (0=mixed, higher=separable)")

pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(X_feat)

# plt.figure(figsize=(7, 6))
# plt.scatter(Xp[origin == 0, 0], Xp[origin == 0, 1], s=6, alpha=0.5, label="Real nodes")
# plt.scatter(
#     Xp[origin == 1, 0], Xp[origin == 1, 1], s=6, alpha=0.5, label="Synthetic nodes"
# )
# plt.title("PCA of Node Features (sid/op/duration): Real vs Synthetic")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
# plt.tight_layout()
# plt.savefig("kmeans_SN_dataset.png", dpi=300, bbox_inches="tight")
# plt.close()
