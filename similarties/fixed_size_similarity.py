import matplotlib.pyplot as plj
import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
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

    def __init__(self, datapath="../datasets/SN_data.pt") -> None:
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


def _to_networkx_with_attrs(graph):
    node_attrs = ["x"] if hasattr(graph, "x") and graph.x is not None else None
    edge_attrs = (
        ["edge_attr"]
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None
        else None
    )
    return to_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)


def compute_fidelity(G1, G2, duration_atol=1e-3, duration_rtol=1e-5):
    """Check graph isomorphism with and without node attributes."""

    def _node_match(a, b):
        xa, xb = a.get("x"), b.get("x")
        if xa is None or xb is None:
            return True
        xa = torch.as_tensor(xa)
        xb = torch.as_tensor(xb)
        if xa.shape != xb.shape:
            return False
        if xa.numel() >= 2:
            ids_a = xa[:2].round().to(torch.long)
            ids_b = xb[:2].round().to(torch.long)
            if not torch.equal(ids_a, ids_b):
                return False
        if xa.numel() >= 3:
            dur_a = xa[2:]
            dur_b = xb[2:]
            if not torch.allclose(
                dur_a, dur_b, atol=duration_atol, rtol=duration_rtol, equal_nan=True
            ):
                return False
        return True

    def _edge_match(a, b):
        ea, eb = a.get("edge_attr"), b.get("edge_attr")
        if ea is None or eb is None:
            return True
        ea = torch.as_tensor(ea)
        eb = torch.as_tensor(eb)
        if ea.shape != eb.shape:
            return False
        return torch.allclose(ea, eb, atol=1e-6, rtol=1e-5, equal_nan=True)

    iso_struct = nx.is_isomorphic(G1, G2)
    iso_attr = nx.is_isomorphic(G1, G2, node_match=_node_match, edge_match=_edge_match)
    return {
        "isomorphic_structure": iso_struct,
        "isomorphic_attr": iso_attr,
    }


def compute_node_attr_similarity(G1, G2, duration_eps=1e-9):
    """Soft node-attribute similarity via best bipartite matching."""

    def _score(a, b):
        denom = max(abs(a), abs(b), duration_eps)
        return 1.0 - min(abs(a - b) / denom, 1.0)

    def _node_sim(xa, xb):
        if xa is None or xb is None:
            return 0.0
        xa = torch.as_tensor(xa).flatten()
        xb = torch.as_tensor(xb).flatten()
        if xa.numel() < 3 or xb.numel() < 3:
            return 0.0
        sid_sim = 1.0 if int(round(float(xa[0]))) == int(round(float(xb[0]))) else 0.0
        op_sim = 1.0 if int(round(float(xa[1]))) == int(round(float(xb[1]))) else 0.0
        dur_sim = _score(float(xa[2]), float(xb[2]))
        return 0.4 * sid_sim + 0.4 * op_sim + 0.2 * dur_sim

    nodes1 = list(G1.nodes())
    nodes2 = list(G2.nodes())
    if not nodes1 or not nodes2:
        return 0.0

    sim = np.zeros((len(nodes1), len(nodes2)), dtype=float)
    for i, n1 in enumerate(nodes1):
        x1 = G1.nodes[n1].get("x")
        for j, n2 in enumerate(nodes2):
            x2 = G2.nodes[n2].get("x")
            sim[i, j] = _node_sim(x1, x2)

    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_sim = sim[row_ind, col_ind]
    avg_sim = float(matched_sim.mean()) if matched_sim.size else 0.0
    coverage = min(len(nodes1), len(nodes2)) / max(len(nodes1), len(nodes2))
    return avg_sim * coverage


def compute_node_attr_similarity_with_jsd(
    G1,
    G2,
    duration_eps: float = 1e-9,
    n_dur_bins: int = 32,
    jsd_weights=(0.4, 0.4, 0.2),  # (svc, op, dur) weights for JSD part
    combine_weights=(0.6, 0.4),  # (matching, jsd) weights for final score
):
    """
    Node attribute similarity = (Hungarian best-match similarity) AND (JSD distribution similarity).

    Returns a dict with:
      - match_score: alignment-based (0..1)
      - jsd_score: distribution-based (0..1)
      - combined: weighted combination (0..1)
      - details: component scores
    Assumes node attribute 'x' has at least 3 values: [svc_id, op_id, duration]
    """

    # ---------- helpers ----------
    def _ratio_score(a, b):
        denom = max(abs(a), abs(b), duration_eps)
        return 1.0 - min(abs(a - b) / denom, 1.0)

    def _node_sim(xa, xb):
        if xa is None or xb is None:
            return 0.0
        xa = torch.as_tensor(xa).flatten()
        xb = torch.as_tensor(xb).flatten()
        if xa.numel() < 3 or xb.numel() < 3:
            return 0.0
        sid_sim = 1.0 if int(round(float(xa[0]))) == int(round(float(xb[0]))) else 0.0
        op_sim = 1.0 if int(round(float(xa[1]))) == int(round(float(xb[1]))) else 0.0
        dur_sim = _ratio_score(float(xa[2]), float(xb[2]))
        return 0.4 * sid_sim + 0.4 * op_sim + 0.2 * dur_sim

    def _extract_attrs(G):
        svc, op, dur = [], [], []
        for n in G.nodes():
            x = G.nodes[n].get("x", None)
            if x is None:
                continue
            x = torch.as_tensor(x).flatten()
            if x.numel() < 3:
                continue
            svc.append(int(round(float(x[0]))))
            op.append(int(round(float(x[1]))))
            dur.append(float(x[2]))
        return np.array(svc), np.array(op), np.array(dur, dtype=float)

    def _pmf_from_ints(vals, support=None):
        if vals.size == 0:
            return None
        if support is None:
            support = np.unique(vals)
        idx = {v: i for i, v in enumerate(support)}
        counts = np.zeros(len(support), dtype=float)
        for v in vals:
            if v in idx:
                counts[idx[v]] += 1.0
        s = counts.sum()
        return counts / s if s > 0 else None

    def _pmf_from_durations(vals, bins, log_space=True):
        if vals.size == 0:
            return None
        v = vals.copy()
        # duration often heavy-tailed; log-binning helps
        if log_space:
            v = np.log10(np.maximum(v, duration_eps))
        counts, _ = np.histogram(v, bins=bins)
        counts = counts.astype(float)
        s = counts.sum()
        return counts / s if s > 0 else None

    def _jsd_sim(p, q):
        """
        Jensen-Shannon distance from scipy is in [0,1] (base=2 by default).
        Convert to similarity: 1 - JSD_distance
        """
        if p is None or q is None:
            return 0.0
        # smooth to avoid zeros causing issues in some setups
        eps = 1e-12
        p = np.asarray(p, dtype=float) + eps
        q = np.asarray(q, dtype=float) + eps
        p = p / p.sum()
        q = q / q.sum()
        d = float(jensenshannon(p, q, base=2.0))  # distance in [0,1]
        d = min(max(d, 0.0), 1.0)
        return 1.0 - d

    # ---------- empty handling ----------
    nodes1 = list(G1.nodes())
    nodes2 = list(G2.nodes())
    if not nodes1 or not nodes2:
        return {
            "match_score": 0.0,
            "jsd_score": 0.0,
            "combined": 0.0,
            "details": {},
        }

    # ---------- (A) matching-based score ----------
    sim = np.zeros((len(nodes1), len(nodes2)), dtype=float)
    for i, n1 in enumerate(nodes1):
        x1 = G1.nodes[n1].get("x")
        for j, n2 in enumerate(nodes2):
            x2 = G2.nodes[n2].get("x")
            sim[i, j] = _node_sim(x1, x2)

    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_sim = sim[row_ind, col_ind]
    avg_sim = float(matched_sim.mean()) if matched_sim.size else 0.0
    coverage = min(len(nodes1), len(nodes2)) / max(len(nodes1), len(nodes2))
    match_score = avg_sim * coverage

    # ---------- (B) JSD-based distribution similarity ----------
    svc1, op1, dur1 = _extract_attrs(G1)
    svc2, op2, dur2 = _extract_attrs(G2)

    # support union so vectors align
    svc_support = (
        np.unique(np.concatenate([svc1, svc2]))
        if svc1.size or svc2.size
        else np.array([], dtype=int)
    )
    op_support = (
        np.unique(np.concatenate([op1, op2]))
        if op1.size or op2.size
        else np.array([], dtype=int)
    )

    p_svc = _pmf_from_ints(svc1, support=svc_support) if svc_support.size else None
    q_svc = _pmf_from_ints(svc2, support=svc_support) if svc_support.size else None
    p_op = _pmf_from_ints(op1, support=op_support) if op_support.size else None
    q_op = _pmf_from_ints(op2, support=op_support) if op_support.size else None

    # shared bins for durations (use log space)
    if dur1.size and dur2.size:
        d_all = np.concatenate([dur1, dur2])
        d_all = np.log10(np.maximum(d_all, duration_eps))
        lo, hi = float(d_all.min()), float(d_all.max())
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        dur_bins = np.linspace(lo, hi, n_dur_bins + 1)
        p_dur = _pmf_from_durations(dur1, bins=dur_bins, log_space=True)
        q_dur = _pmf_from_durations(dur2, bins=dur_bins, log_space=True)
    else:
        p_dur = q_dur = None

    svc_sim = _jsd_sim(p_svc, q_svc)
    op_sim = _jsd_sim(p_op, q_op)
    dur_sim = _jsd_sim(p_dur, q_dur)

    w_svc, w_op, w_dur = jsd_weights
    w_sum = max(w_svc + w_op + w_dur, 1e-12)
    jsd_score = (w_svc * svc_sim + w_op * op_sim + w_dur * dur_sim) / w_sum

    # ---------- combine ----------
    w_match, w_jsd = combine_weights
    w2 = max(w_match + w_jsd, 1e-12)
    combined = (w_match * match_score + w_jsd * jsd_score) / w2

    return {
        "match_score": float(match_score),
        "jsd_score": float(jsd_score),
        "combined": float(combined),
        "details": {
            "svc_jsd_sim": float(svc_sim),
            "op_jsd_sim": float(op_sim),
            "dur_jsd_sim": float(dur_sim),
            "coverage": float(coverage),
            "avg_matched_sim": float(avg_sim),
            "n1": int(len(nodes1)),
            "n2": int(len(nodes2)),
        },
    }


def create_graph(graph, output):
    # Graph
    G = _to_networkx_with_attrs(graph)
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
    real_path = (
        "../datasets/SN_data.pt"  # change this to ../datasets/TT_data.pt for tt results
    )
    synth_path = "../datasets/fixed_size/prop_order_SN_synthetic.pt"  # change this to ../datasets/fixed_size/prop_order_TT_synthetic.pt for tt results

    real_ds = LoadDataset(real_path)
    synth_ds = LoadDataset(synth_path)

    assert len(real_ds) == len(synth_ds)

    # Sample detailed comparison for first few graphs
    print("\nDetailed Comparison (First 3 Pairs):")
    print("=" * 80)
    for i in range(min(3, len(real_ds))):
        G_real = _to_networkx_with_attrs(real_ds[i])
        G_synth = _to_networkx_with_attrs(synth_ds[i])
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
        G_real = _to_networkx_with_attrs(real_graph)
        G_synth = _to_networkx_with_attrs(synth_graph)
        similarity_score = compute_similarity(G_real, G_synth) * 100

        similarities.append(
            {
                "index": i,
                "overall_similarity": similarity_score,
            }
        )

        print(f"Graph Pair {i:4d}: " f"Similarity Score: {similarity_score:6.2f}% | ")
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
            G = _to_networkx_with_attrs(g)
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
