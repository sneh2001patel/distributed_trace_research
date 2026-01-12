import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.stats import ks_2samp
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.utils import to_networkx

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


def summary_stat(arr, avg="Average"):

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    arr = np.array(arr)

    print(f"{avg} Overall Similarity: {np.mean(arr):6.2f}%")

    # Find most and least reconstructions
    most = max(arr)
    least = min(arr)

    print(f"\nMost: ({most:.2f}% similar)")
    print(f"Least: ({least:.2f}% similar)")

    # Count how many are "good" matches
    excellent = sum(1 for s in arr if s >= 80)
    good = sum(1 for s in arr if 50 <= s < 80)
    poor = sum(1 for s in arr if s < 50)

    print(f"\nReconstruction Quality:")
    print(f"(≥80% similarity): {excellent}/{len(arr)} ({excellent/len(arr)*100:.1f}%)")
    print(f" (50-80% similarity):    {good}/{len(arr)} ({good/len(arr)*100:.1f}%)")
    print(f" (<50% similarity):      {poor}/{len(arr)} ({poor/len(arr)*100:.1f}%)")


def compute_similarity(G1, G2):
    """Compare two graphs using alignment-free statistics."""
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    e1, e2 = G1.number_of_edges(), G2.number_of_edges()
    dens1 = nx.density(G1) if n1 > 1 else 0.0
    dens2 = nx.density(G2) if n2 > 1 else 0.0
    avg_deg1 = (2 * e1 / n1) if n1 > 0 else 0.0
    avg_deg2 = (2 * e2 / n2) if n2 > 0 else 0.0
    clust1 = nx.average_clustering(G1) if n1 > 1 else 0.0
    clust2 = nx.average_clustering(G2) if n2 > 1 else 0.0

    degrees1 = [d for _, d in G1.degree()]
    degrees2 = [d for _, d in G2.degree()]
    ks_stat, _ = ks_2samp(degrees1, degrees2)

    def _score(a, b):
        denom = max(a, b, 1e-9)
        return 1.0 - min(abs(a - b) / denom, 1.0)

    scores = [
        _score(n1, n2),
        _score(e1, e2),
        1.0 - abs(dens1 - dens2),
        _score(avg_deg1, avg_deg2),
        1.0 - abs(clust1 - clust2),
        1.0 - min(ks_stat, 1.0),
    ]
    return sum(scores) / len(scores)


def main():

    # load ds
    real_path = "./processed/SN_data.pt"
    synth_path = "./processed/not_exact_replica/SN_synthetic.pt"
    real_ds = LoadDataset(real_path)
    synth_ds = LoadDataset(synth_path)

    # Compare each synthetic graph with the original graph
    count = 1
    avg_sims = []
    max_sims = []
    min_sims = []
    for synth_graph in synth_ds:
        G_synth = to_networkx(synth_graph)

        similarities = []
        for real_graph in real_ds:
            G_real = to_networkx(real_graph)
            similarity_score = compute_similarity(G_real, G_synth) * 100
            similarities.append(similarity_score)
        similarities = np.array(similarities)
        avg_sim = np.mean(similarities)
        max_sim = max(similarities)
        min_sim = min(similarities)
        print(
            f"Graph: {count} Average Similarity Score: {avg_sim:.2f}, Max: {max(similarities):.2f}, Min: {min(similarities):.2f}"
        )
        avg_sims.append(avg_sim)
        max_sims.append(max_sim)
        min_sims.append(min_sim)
        count += 1

    summary_stat(avg_sims, avg="Average")
    summary_stat(max_sims, avg="Average Max Similarity")
    summary_stat(min_sims, avg="Average Min Similarity")


main()
