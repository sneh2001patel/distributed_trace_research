import random
import copy

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr


class PTDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def load_graphs(datapath: str):
    dataset = PTDataset(datapath)
    graphs = []
    for i in range(len(dataset)):
        graphs.append(dataset.get(i).cpu())
    return graphs


def deduplicate_graph_edges(graphs):
    deduped = []
    for g in graphs:
        h = copy.copy(g)
        if g.edge_index.numel() > 0:
            h.edge_index = torch.unique(g.edge_index.t(), dim=0).t().contiguous()
        deduped.append(h)
    return deduped


def filter_graphs_by_min_nodes(graphs, min_nodes: int):
    if min_nodes <= 1:
        return graphs
    return [g for g in graphs if g.num_nodes >= min_nodes]


def _set_graph_label(graphs, label: int):
    y = torch.tensor(label, dtype=torch.long)
    for g in graphs:
        g.y = y


def _split_graphs(graphs, keep_ratio: float, seed: int = 42):
    if keep_ratio <= 0:
        return [], graphs
    if keep_ratio >= 1:
        return graphs, []
    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)
    keep = int(len(indices) * keep_ratio)
    keep_idx = set(indices[:keep])
    kept = [g for i, g in enumerate(graphs) if i in keep_idx]
    rest = [g for i, g in enumerate(graphs) if i not in keep_idx]
    return kept, rest


def _take_fraction(graphs, fraction: float, seed: int = 42):
    if fraction >= 1.0:
        return graphs
    if fraction <= 0:
        return []
    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)
    keep = max(1, int(len(indices) * fraction))
    keep_idx = set(indices[:keep])
    return [g for i, g in enumerate(graphs) if i in keep_idx]


def _take_count(graphs, count: int, seed: int = 42):
    if count <= 0:
        return []
    if count >= len(graphs):
        return graphs
    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)
    keep_idx = set(indices[:count])
    return [g for i, g in enumerate(graphs) if i in keep_idx]


def _split_target_by_class(normal_graphs, abnormal_graphs, target_total: int):
    total = len(normal_graphs) + len(abnormal_graphs)
    if total <= 0 or target_total <= 0:
        return 0, 0
    normal_ratio = len(normal_graphs) / total
    normal_target = min(len(normal_graphs), round(target_total * normal_ratio))
    abnormal_target = min(len(abnormal_graphs), target_total - normal_target)
    remaining = target_total - (normal_target + abnormal_target)
    if remaining > 0:
        normal_room = len(normal_graphs) - normal_target
        take_normal = min(normal_room, remaining)
        normal_target += take_normal
        remaining -= take_normal
        abnormal_target += min(len(abnormal_graphs) - abnormal_target, remaining)
    return normal_target, abnormal_target


def prepare_sn_anomaly_datasets(
    syn_normal_path: str,
    syn_abnormal_path: str,
    real_normal_path: str,
    real_abnormal_path: str,
    real_train_ratio: float = 0.1,
    real_percent: float = None,
    val_ratio: float = 0.2,
    real_holdout_ratio: float = 0.2,
    real_val_ratio: float = 0.1,
    real_test_fraction: float = 1.0,
    max_synth_per_class: int = None,
    min_nodes: int = 1,
    seed: int = 42,
):
    syn_normal = load_graphs(syn_normal_path)
    syn_abnormal = load_graphs(syn_abnormal_path)
    real_normal = load_graphs(real_normal_path)
    real_abnormal = load_graphs(real_abnormal_path)

    _set_graph_label(syn_normal, 0)
    _set_graph_label(syn_abnormal, 1)
    _set_graph_label(real_normal, 0)
    _set_graph_label(real_abnormal, 1)

    if min_nodes > 1:
        syn_normal = filter_graphs_by_min_nodes(syn_normal, min_nodes=min_nodes)
        syn_abnormal = filter_graphs_by_min_nodes(syn_abnormal, min_nodes=min_nodes)
        real_normal = filter_graphs_by_min_nodes(real_normal, min_nodes=min_nodes)
        real_abnormal = filter_graphs_by_min_nodes(real_abnormal, min_nodes=min_nodes)

    if max_synth_per_class is not None:
        syn_normal = _take_count(syn_normal, max_synth_per_class, seed=seed)
        syn_abnormal = _take_count(syn_abnormal, max_synth_per_class, seed=seed)

    syn_train_normal, syn_val_normal = _split_graphs(
        syn_normal, keep_ratio=1.0 - val_ratio, seed=seed
    )
    syn_train_abnormal, syn_val_abnormal = _split_graphs(
        syn_abnormal, keep_ratio=1.0 - val_ratio, seed=seed
    )

    if real_percent is not None:
        # For ratio experiments, keep a fixed held-out real test split and use the
        # remaining real graphs as the candidate training pool.
        real_pool_normal, real_test_normal = _split_graphs(
            real_normal, keep_ratio=1.0 - real_holdout_ratio, seed=seed
        )
        real_pool_abnormal, real_test_abnormal = _split_graphs(
            real_abnormal, keep_ratio=1.0 - real_holdout_ratio, seed=seed
        )
        real_train_normal, real_val_normal = _split_graphs(
            real_pool_normal, keep_ratio=1.0 - real_val_ratio, seed=seed + 1
        )
        real_train_abnormal, real_val_abnormal = _split_graphs(
            real_pool_abnormal, keep_ratio=1.0 - real_val_ratio, seed=seed + 1
        )
    else:
        real_train_normal, real_test_normal = _split_graphs(
            real_normal, keep_ratio=real_train_ratio, seed=seed
        )
        real_train_abnormal, real_test_abnormal = _split_graphs(
            real_abnormal, keep_ratio=real_train_ratio, seed=seed
        )
        real_val_normal, real_val_abnormal = [], []

    real_test_normal = _take_fraction(real_test_normal, real_test_fraction, seed=seed)
    real_test_abnormal = _take_fraction(
        real_test_abnormal, real_test_fraction, seed=seed
    )

    if real_percent is not None:
        if not 0.0 <= real_percent <= 100.0:
            raise ValueError("real_percent must be between 0 and 100")

        synth_percent = 100.0 - real_percent
        syn_total = len(syn_train_normal) + len(syn_train_abnormal)
        real_total = len(real_train_normal) + len(real_train_abnormal)

        if real_percent == 0.0:
            target_syn_total = syn_total
            target_real_total = 0
        elif synth_percent == 0.0:
            target_syn_total = 0
            target_real_total = real_total
        else:
            max_total_from_syn = syn_total / (synth_percent / 100.0)
            max_total_from_real = real_total / (real_percent / 100.0)
            target_total = int(min(max_total_from_syn, max_total_from_real))
            target_syn_total = int(round(target_total * synth_percent / 100.0))
            target_real_total = int(round(target_total * real_percent / 100.0))

        syn_normal_target, syn_abnormal_target = _split_target_by_class(
            syn_train_normal, syn_train_abnormal, target_syn_total
        )
        real_normal_target, real_abnormal_target = _split_target_by_class(
            real_train_normal, real_train_abnormal, target_real_total
        )

        syn_train_normal = _take_count(syn_train_normal, syn_normal_target, seed=seed)
        syn_train_abnormal = _take_count(
            syn_train_abnormal, syn_abnormal_target, seed=seed
        )
        real_train_normal = _take_count(
            real_train_normal, real_normal_target, seed=seed
        )
        real_train_abnormal = _take_count(
            real_train_abnormal, real_abnormal_target, seed=seed
        )

    train_ds = GraphDataset(
        syn_train_normal
        + syn_train_abnormal
        + real_train_normal
        + real_train_abnormal
    )
    if real_percent is not None:
        val_graphs = real_val_normal + real_val_abnormal
    else:
        val_graphs = syn_val_normal + syn_val_abnormal
    val_ds = GraphDataset(val_graphs)
    test_ds = GraphDataset(real_test_normal + real_test_abnormal)

    summary = {
        "train_normal_synth": len(syn_train_normal),
        "train_abnormal_synth": len(syn_train_abnormal),
        "train_normal_real": len(real_train_normal),
        "train_abnormal_real": len(real_train_abnormal),
        "val_normal_synth": len(syn_val_normal),
        "val_abnormal_synth": len(syn_val_abnormal),
        "val_normal_real": len(real_val_normal),
        "val_abnormal_real": len(real_val_abnormal),
        "test_normal_real": len(real_test_normal),
        "test_abnormal_real": len(real_test_abnormal),
        "train_real_total": len(real_train_normal) + len(real_train_abnormal),
        "train_synth_total": len(syn_train_normal) + len(syn_train_abnormal),
    }
    return train_ds, val_ds, test_ds, summary


def prepare_tt_anomaly_datasets(
    syn_normal_path: str,
    syn_abnormal_path: str,
    real_normal_path: str,
    real_abnormal_path: str,
    real_train_ratio: float = 0.1,
    real_percent: float = None,
    val_ratio: float = 0.2,
    real_holdout_ratio: float = 0.2,
    real_val_ratio: float = 0.1,
    real_test_fraction: float = 1.0,
    max_synth_per_class: int = None,
    min_nodes: int = 3,
    seed: int = 42,
):
    train_ds, val_ds, test_ds, summary = prepare_sn_anomaly_datasets(
        syn_normal_path=syn_normal_path,
        syn_abnormal_path=syn_abnormal_path,
        real_normal_path=real_normal_path,
        real_abnormal_path=real_abnormal_path,
        real_train_ratio=real_train_ratio,
        real_percent=real_percent,
        val_ratio=val_ratio,
        real_holdout_ratio=real_holdout_ratio,
        real_val_ratio=real_val_ratio,
        real_test_fraction=real_test_fraction,
        max_synth_per_class=max_synth_per_class,
        min_nodes=min_nodes,
        seed=seed,
    )
    train_ds.graphs = filter_graphs_by_min_nodes(
        deduplicate_graph_edges(train_ds.graphs), min_nodes=min_nodes
    )
    val_ds.graphs = filter_graphs_by_min_nodes(
        deduplicate_graph_edges(val_ds.graphs), min_nodes=min_nodes
    )
    test_ds.graphs = filter_graphs_by_min_nodes(
        deduplicate_graph_edges(test_ds.graphs), min_nodes=min_nodes
    )
    summary["min_nodes_filter"] = min_nodes
    summary["train_total_after_filter"] = len(train_ds.graphs)
    summary["val_total_after_filter"] = len(val_ds.graphs)
    summary["test_total_after_filter"] = len(test_ds.graphs)
    summary["train_normal_real_after_filter"] = sum(
        int(g.y.item()) == 0 for g in train_ds.graphs
    )
    summary["train_abnormal_real_after_filter"] = sum(
        int(g.y.item()) == 1 for g in train_ds.graphs
    )
    summary["val_normal_real_after_filter"] = sum(
        int(g.y.item()) == 0 for g in val_ds.graphs
    )
    summary["val_abnormal_real_after_filter"] = sum(
        int(g.y.item()) == 1 for g in val_ds.graphs
    )
    summary["test_normal_real_after_filter"] = sum(
        int(g.y.item()) == 0 for g in test_ds.graphs
    )
    summary["test_abnormal_real_after_filter"] = sum(
        int(g.y.item()) == 1 for g in test_ds.graphs
    )
    return train_ds, val_ds, test_ds, summary
