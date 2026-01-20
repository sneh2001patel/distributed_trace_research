import argparse

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.utils import to_networkx


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices

    def get(self, idx):
        return super().get(idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synth-path",
        default="./processed/TT_data.pt",
        help="Path to synthetic dataset .pt file",
    )
    parser.add_argument(
        "--output",
        default="./real_graph_first.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    ds = LoadDataset(args.synth_path)
    graph = ds[0]
    G = to_networkx(graph, node_attrs=["x"])

    labels = {}
    for n in G.nodes():
        x = G.nodes[n].get("x")
        if x is None:
            labels[n] = str(n)
        else:
            x = torch.as_tensor(x).flatten()
            if x.numel() >= 3:
                labels[n] = (
                    f"{n}\nsvc={int(round(float(x[0])))} op={int(round(float(x[1])))}\n"
                    f"dur={float(x[2]):.3f}"
                )
            else:
                labels[n] = f"{n}\n{x.tolist()}"

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=7)
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=900,
        font_size=7,
    )
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
