import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

from synthetic_graphs import generate_synthetic_graph, load_tt_decoder


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_tt_decoder("./weights/tt_vae_weights.pt", device)

    graph = generate_synthetic_graph(
        decoder,
        num_nodes=10,
        duration_mean=cfg["duration_mean"],
        duration_std=cfg["duration_std"],
        device=device,
        # sample_edges=args.sample_edges,
        # sample_nodes=args.sample_nodes,
    )

    G = to_networkx(graph, to_undirected=True)
    os.makedirs("./syn_graphs", exist_ok=True)
    out_path = os.path.join("./syn_graphs", "synthetic_graph.png")

    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(G, seed=7)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10)
    # plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
