import torch
from torch.utils.data import Dataset, random_split
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
    return [dataset.get(i) for i in range(len(dataset))]


def _set_graph_label(graphs, label: int):
    y = torch.tensor(label, dtype=torch.long)
    for g in graphs:
        g.y = y


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
