import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


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


def preprocess_sid_op(
    X_real: torch.Tensor, X_syn: torch.Tensor, n_sid: int, n_op: int
):
    X_all = torch.cat([X_real, X_syn], dim=0)
    sid = X_all[:, 0].round().long()
    op = X_all[:, 1].round().long()

    if (sid < 0).any() or (op < 0).any():
        raise ValueError("Found neg Ids in service_id or op_id")

    # one-hot encode categorical IDs
    sid_oh = torch.nn.functional.one_hot(sid, num_classes=n_sid).float()
    op_oh = torch.nn.functional.one_hot(op, num_classes=n_op).float()

    # Combine (ignore duration)
    X_feat = torch.cat([sid_oh, op_oh], dim=1).numpy()

    origin = np.array([0] * len(X_real) + [1] * len(X_syn))  # 0=real, 1=synthetic
    return X_feat, origin


def freq_drift_report(
    X_real: torch.Tensor, X_syn: torch.Tensor, n_sid: int, n_op: int, top_k: int = 10
):
    sid_r = X_real[:, 0].round().long().numpy()
    sid_s = X_syn[:, 0].round().long().numpy()
    op_r = X_real[:, 1].round().long().numpy()
    op_s = X_syn[:, 1].round().long().numpy()

    sid_r_counts = np.bincount(sid_r, minlength=n_sid)
    sid_s_counts = np.bincount(sid_s, minlength=n_sid)
    op_r_counts = np.bincount(op_r, minlength=n_op)
    op_s_counts = np.bincount(op_s, minlength=n_op)

    sid_r_freq = sid_r_counts / max(sid_r_counts.sum(), 1)
    sid_s_freq = sid_s_counts / max(sid_s_counts.sum(), 1)
    op_r_freq = op_r_counts / max(op_r_counts.sum(), 1)
    op_s_freq = op_s_counts / max(op_s_counts.sum(), 1)

    sid_diff = np.abs(sid_r_freq - sid_s_freq)
    op_diff = np.abs(op_r_freq - op_s_freq)

    sid_top = np.argsort(sid_diff)[-top_k:][::-1]
    op_top = np.argsort(op_diff)[-top_k:][::-1]

    # joint (service_id, op_id) drift
    joint_r = sid_r * n_op + op_r
    joint_s = sid_s * n_op + op_s
    joint_size = n_sid * n_op
    joint_r_counts = np.bincount(joint_r, minlength=joint_size)
    joint_s_counts = np.bincount(joint_s, minlength=joint_size)
    joint_r_freq = joint_r_counts / max(joint_r_counts.sum(), 1)
    joint_s_freq = joint_s_counts / max(joint_s_counts.sum(), 1)
    joint_diff = np.abs(joint_r_freq - joint_s_freq)
    joint_top = np.argsort(joint_diff)[-top_k:][::-1]

    def _format_pairs(indices):
        rows = []
        for idx in indices:
            sid = idx // n_op
            op = idx % n_op
            rows.append((sid, op, joint_r_freq[idx], joint_s_freq[idx], joint_diff[idx]))
        return rows

    print("\nTop service_id frequency drifts (abs diff):")
    for sid in sid_top:
        print(
            f"  sid={sid}: real={sid_r_freq[sid]:.4f} syn={sid_s_freq[sid]:.4f} diff={sid_diff[sid]:.4f}"
        )

    print("\nTop op_id frequency drifts (abs diff):")
    for op in op_top:
        print(
            f"  op={op}: real={op_r_freq[op]:.4f} syn={op_s_freq[op]:.4f} diff={op_diff[op]:.4f}"
        )

    print("\nTop (service_id, op_id) joint frequency drifts (abs diff):")
    for sid, op, r, s, d in _format_pairs(joint_top):
        print(f"  (sid={sid}, op={op}): real={r:.4f} syn={s:.4f} diff={d:.4f}")


real = LoadDataset(datapath="./processed/TT_data.pt")
synthetic = LoadDataset(datapath="./processed/exact_replica/prop_order_TT_synthetic.pt")

real_graphs = [g.x.detach().cpu().float() for g in real]
syn_graphs = [g.x.detach().cpu().float() for g in synthetic]

print(f"Real graphs: {len(real_graphs)}, Synthetic graphs: {len(syn_graphs)}")

# graph-level split to avoid node leakage
real_idx = np.arange(len(real_graphs))
syn_idx = np.arange(len(syn_graphs))
real_train_idx, real_test_idx = train_test_split(
    real_idx, test_size=0.3, random_state=42
)
syn_train_idx, syn_test_idx = train_test_split(
    syn_idx, test_size=0.3, random_state=42
)

Xr_train = torch.cat([real_graphs[i] for i in real_train_idx], dim=0)
Xr_test = torch.cat([real_graphs[i] for i in real_test_idx], dim=0)
Xs_train = torch.cat([syn_graphs[i] for i in syn_train_idx], dim=0)
Xs_test = torch.cat([syn_graphs[i] for i in syn_test_idx], dim=0)

X_all = torch.cat(
    [torch.cat(real_graphs, dim=0), torch.cat(syn_graphs, dim=0)], dim=0
)
n_sid = int(X_all[:, 0].round().long().max().item()) + 1
n_op = int(X_all[:, 1].round().long().max().item()) + 1

X_train_feat, _ = preprocess_sid_op(Xr_train, Xs_train, n_sid, n_op)
X_test_feat, _ = preprocess_sid_op(Xr_test, Xs_test, n_sid, n_op)

freq_drift_report(Xr_train, Xs_train, n_sid, n_op, top_k=10)

X_train = np.vstack([X_train_feat[: len(Xr_train)], X_train_feat[len(Xr_train) :]])
y_train = np.concatenate(
    [
        np.zeros(len(Xr_train), dtype=np.int64),
        np.ones(len(Xs_train), dtype=np.int64),
    ]
)

X_test = np.vstack([X_test_feat[: len(Xr_test)], X_test_feat[len(Xr_test) :]])
y_test = np.concatenate(
    [
        np.zeros(len(Xr_test), dtype=np.int64),
        np.ones(len(Xs_test), dtype=np.int64),
    ]
)

# 1) Simple, conservative RF discriminator
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=42,
)

rf.fit(X_train, y_train)

# 4) Evaluate
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"RF real-vs-synthetic accuracy: {acc:.4f}")
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("\nReport:")
print(classification_report(y_test, y_pred, target_names=["Real", "Synthetic"]))
