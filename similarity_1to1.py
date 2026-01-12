import matplotlib.pyplot as plt
import networkx as nx
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


def create_graph(graph, output):
    # Graph
    G = to_networkx(graph)
    # real_outpath = os.path.join("./syn_graphs", "TT_real_graph.png")
    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(G, seed=7)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {output}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load ds
    real_path = "./processed/SN_data.pt"
    synth_path = "./processed/exact_replica/prop_order_SN_synthetic.pt"
    real_ds = LoadDataset(real_path)
    synth_ds = LoadDataset(synth_path)

    assert len(real_ds) == len(synth_ds)

    # Sample detailed comparison for first few graphs
    print("\nDetailed Comparison (First 3 Pairs):")
    print("=" * 80)
    for i in range(min(3, len(real_ds))):
        G_real = to_networkx(real_ds[i])
        G_synth = to_networkx(synth_ds[i])
        print(f"\nPair {i}:")
        print(
            f"  Real:   {G_real.number_of_nodes()} nodes, {G_real.number_of_edges()} edges"
        )
        print(
            f"  Synth:  {G_synth.number_of_nodes()} nodes, {G_synth.number_of_edges()} edges"
        )
        print(f"  Real degree dist:  {sorted([d for n, d in G_real.degree()])}")
        print(f"  Synth degree dist: {sorted([d for n, d in G_synth.degree()])}")
    print("=" * 80)

    print("\nSimilarity Analysis (paired by index):")
    print("-" * 60)
    similarities = []

    for i, (real_graph, synth_graph) in enumerate(zip(real_ds, synth_ds)):
        G_real = to_networkx(real_graph)
        G_synth = to_networkx(synth_graph)
        similarity_score = compute_similarity(G_real, G_synth) * 100

        similarities.append(
            {
                "index": i,
                "overall_similarity": similarity_score,
            }
        )

        print(f"Graph Pair {i:4d}: " f"Similarity Score: {similarity_score:6.2f}%")
        print("-" * 60)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    avg_similarity = sum(s["overall_similarity"] for s in similarities) / len(
        similarities
    )

    print(f"Average Overall Similarity: {avg_similarity:6.2f}%")

    # Find best and worst reconstructions
    best = max(similarities, key=lambda x: x["overall_similarity"])
    worst = min(similarities, key=lambda x: x["overall_similarity"])

    print(f"\nMost:  Graph {best['index']} ({best['overall_similarity']:.2f}% similar)")
    print(f"Least: Graph {worst['index']} ({worst['overall_similarity']:.2f}% similar)")

    # Count how many are "good" matches
    excellent = sum(1 for s in similarities if s["overall_similarity"] >= 80)
    good = sum(1 for s in similarities if 50 <= s["overall_similarity"] < 80)
    poor = sum(1 for s in similarities if s["overall_similarity"] < 50)

    print(f"\nReconstruction Quality:")
    print(
        f"  (≥80% similarity): {excellent}/{len(similarities)} ({excellent/len(similarities)*100:.1f}%)"
    )
    print(
        f" (50-80% similarity):    {good}/{len(similarities)} ({good/len(similarities)*100:.1f}%)"
    )
    print(
        f" (<50% similarity):      {poor}/{len(similarities)} ({poor/len(similarities)*100:.1f}%)"
    )

    print("\nDataset-level Distribution Similarity:")
    print("-" * 60)

    def _dataset_stats(ds):
        stats = {
            "nodes": [],
            "edges": [],
            "density": [],
            "avg_degree": [],
            "avg_clustering": [],
        }
        for g in ds:
            G = to_networkx(g)
            n = G.number_of_nodes()
            e = G.number_of_edges()
            stats["nodes"].append(n)
            stats["edges"].append(e)
            stats["density"].append(nx.density(G) if n > 1 else 0.0)
            stats["avg_degree"].append((2 * e / n) if n > 0 else 0.0)
            stats["avg_clustering"].append(nx.average_clustering(G) if n > 1 else 0.0)
        return stats

    real_stats = _dataset_stats(real_ds)
    synth_stats = _dataset_stats(synth_ds)
    for key in real_stats:
        ks_stat, p_val = ks_2samp(real_stats[key], synth_stats[key])
        print(f"{key:14s} KS={ks_stat:.3f} p={p_val:.3g}")


main()
