import torch
import torch.serialization
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
        self.name = "SN" if "SN" in datapath else "TT"
        self.data, self.slices = data, slices

    def get(self, idx):
        return super().get(idx)


def summarize(dataset: LoadDataset):
    total_graphs = len(dataset)
    labels = []
    total_nodes = 0
    min_nodes = float("inf")
    max_nodes = 0

    for g in dataset:
        if hasattr(g, "y") and g.y is not None:
            labels.append(int(g.y.item()))
        num_nodes = g.num_nodes
        total_nodes += g.num_nodes
        min_nodes = min(min_nodes, num_nodes)
        max_nodes = max(max_nodes, num_nodes)

    if labels:
        labels_t = torch.tensor(labels)
        unique, counts = torch.unique(labels_t, return_counts=True)
        print("Label distribution:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            pct = 100.0 * c / total_graphs
            name = dataset.name
            print(f"  {name} (label {u}): {c} graphs ({pct:.2f}%)")
    else:
        print("No labels found.")

    print(f"\nTotal graphs: {total_graphs}")
    print(f"Total nodes across all graphs: {total_nodes}")
    print(f"Average nodes per graph: {total_nodes/total_graphs:.2f}")
    print(f"Min nodes in a graph {min_nodes}")
    print(f"Max nodes in a graph {max_nodes}")

    # Feature info
    x = dataset.data.x
    print("\nNode feature columns: [service_name, operation_name, duration]")
    uniq_service = x[:, 0].unique().numel()
    uniq_op = x[:, 1].unique().numel()
    print(f"  service_name unique: {uniq_service}")
    print(f"  operation_name unique: {uniq_op}")
    print(
        f"  duration min/max:   {x[:,2].min().item():.4f} / {x[:,2].max().item():.4f}"
    )


def main():
    ds = LoadDataset()
    summarize(ds)


if __name__ == "__main__":
    main()
