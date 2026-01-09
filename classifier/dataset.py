import torch
from torch.utils.data import Dataset, Subset, random_split
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
        g = dataset.get(i)
        graphs.append(g.cpu())
    return graphs


def _set_graph_label(graphs, label: int):
    y = torch.tensor(label, dtype=torch.long)
    for g in graphs:
        g.y = y


def _materialize_graphs(ds):
    if isinstance(ds, Subset):
        return [ds.dataset.graphs[i] for i in ds.indices]
    return ds.graphs


def prepare_datasets(
    syn_sn_path: str,
    syn_tt_path: str,
    real_sn_path: str,
    real_tt_path: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    syn_sn = load_graphs(syn_sn_path)
    syn_tt = load_graphs(syn_tt_path)
    _set_graph_label(syn_sn, 0)
    _set_graph_label(syn_tt, 1)

    synthetic = GraphDataset(syn_sn + syn_tt)
    if val_ratio <= 0:
        train_ds = synthetic
        val_ds = None
    else:
        val_len = int(len(synthetic) * val_ratio)
        train_len = len(synthetic) - val_len
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(
            synthetic, [train_len, val_len], generator=generator
        )

    real_sn = load_graphs(real_sn_path)
    real_tt = load_graphs(real_tt_path)
    _set_graph_label(real_sn, 0)
    _set_graph_label(real_tt, 1)
    test_ds = GraphDataset(real_sn + real_tt)

    return train_ds, val_ds, test_ds


def prepare_mixed_datasets(
    syn_sn_path: str,
    syn_tt_path: str,
    real_sn_path: str,
    real_tt_path: str,
    real_train_ratio: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    syn_sn = load_graphs(syn_sn_path)
    syn_tt = load_graphs(syn_tt_path)
    _set_graph_label(syn_sn, 0)
    _set_graph_label(syn_tt, 1)
    synthetic = GraphDataset(syn_sn + syn_tt)

    real_sn = load_graphs(real_sn_path)
    real_tt = load_graphs(real_tt_path)
    _set_graph_label(real_sn, 0)
    _set_graph_label(real_tt, 1)
    real_all = GraphDataset(real_sn + real_tt)

    generator = torch.Generator().manual_seed(seed)
    if val_ratio <= 0:
        syn_train = synthetic
        syn_val = None
    else:
        syn_val_len = int(len(synthetic) * val_ratio)
        syn_train_len = len(synthetic) - syn_val_len
        syn_train, syn_val = random_split(
            synthetic, [syn_train_len, syn_val_len], generator=generator
        )

    real_train_len = int(len(real_all) * real_train_ratio)
    real_test_len = len(real_all) - real_train_len
    real_train, real_test = random_split(
        real_all, [real_train_len, real_test_len], generator=generator
    )

    train_graphs = _materialize_graphs(syn_train)
    real_train_graphs = _materialize_graphs(real_train)
    train_ds = GraphDataset(train_graphs + real_train_graphs)

    return train_ds, syn_val, real_test
