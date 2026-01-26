import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

from synthetic_graphs import generate_synthetic_graph, load_decoder


def main(weight_path="./weights/tt_vae_weights.pt"):
    """
    Generates a synthetic graph using the weights from encoder-decoder architecture.
    This only generates one, it creates a png file.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder(weight_path, device)

    graph = generate_synthetic_graph(
        decoder,
        num_nodes=10,  # creates graph with 10 nodes
        duration_mean=cfg["duration_mean"],
        duration_std=cfg["duration_std"],
        device=device,
        # sample_edges=args.sample_edges,
        # sample_nodes=args.sample_nodes,
    )

    # create a networkx graph
    G = to_networkx(graph, to_undirected=True)
    os.makedirs("./syn_graphs", exist_ok=True)
    out_path = os.path.join("./syn_graphs", "synthetic_graph.png")

    # plot the graph
    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(G, seed=7)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10)
    # plt.tight_layout()
    # save file
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    # replace tt_weight_path with sn_weight_path to create synthetic graph for sn
    tt_weight_path = "./weights/tt_vae_weights.pt"
    sn_weight_path = "./weights/sn_vae_weights.pt"
    main(weight_path=tt_weight_path)
