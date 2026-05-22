import argparse
import csv
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


ROOT = Path(__file__).resolve().parent


class PTDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        self.data, self.slices = torch.load(datapath, weights_only=False)


def load_graphs(path: str):
    ds = PTDataset(str(ROOT / path))
    return [ds.get(i).cpu() for i in range(len(ds))]


def node_labels(g):
    x = g.x.detach().cpu()
    return tuple(sorted((int(row[0].item()), int(row[1].item())) for row in x))


def edge_labels(g):
    x = g.x.detach().cpu()
    labels = [(int(row[0].item()), int(row[1].item())) for row in x]
    edges = []
    if g.edge_index.numel() > 0:
        for src, dst in g.edge_index.t().detach().cpu().tolist():
            if src < len(labels) and dst < len(labels):
                edges.append((labels[src], labels[dst]))
    return tuple(sorted(edges))


def signature(g):
    return (node_labels(g), edge_labels(g))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def coverage(real_items, synth_items):
    real_set = set(real_items)
    synth_set = set(synth_items)
    return len(synth_set & real_set) / len(real_set) if real_set else 0.0


def analyze_pair(system, cls, real_path, synth_path, near_threshold=0.9):
    real = load_graphs(real_path)
    synth = load_graphs(synth_path)

    real_sigs = [signature(g) for g in real]
    synth_sigs = [signature(g) for g in synth]
    real_sig_set = set(real_sigs)
    synth_sig_set = set(synth_sigs)

    real_edge_sets = [set(sig[1]) for sig in real_sigs]
    exact_dupes = sum(1 for sig in synth_sigs if sig in real_sig_set)

    near_dupes = 0
    nn_scores = []
    for sig in synth_sigs:
        synth_edges = set(sig[1])
        best = max((jaccard(synth_edges, real_edges) for real_edges in real_edge_sets), default=0.0)
        nn_scores.append(best)
        if best >= near_threshold:
            near_dupes += 1

    real_nodes = [item for sig in real_sigs for item in sig[0]]
    synth_nodes = [item for sig in synth_sigs for item in sig[0]]
    real_edges = [item for sig in real_sigs for item in sig[1]]
    synth_edges = [item for sig in synth_sigs for item in sig[1]]

    return {
        "system": system,
        "trace_class": cls,
        "real_graphs": len(real),
        "synthetic_graphs": len(synth),
        "unique_synthetic_ratio": len(synth_sig_set) / len(synth_sigs) if synth_sigs else 0.0,
        "exact_duplicate_rate": exact_dupes / len(synth_sigs) if synth_sigs else 0.0,
        "novelty_rate": 1.0 - (exact_dupes / len(synth_sigs) if synth_sigs else 0.0),
        "near_duplicate_rate": near_dupes / len(synth_sigs) if synth_sigs else 0.0,
        "mean_nearest_edge_jaccard": sum(nn_scores) / len(nn_scores) if nn_scores else 0.0,
        "service_operation_coverage": coverage(real_nodes, synth_nodes),
        "edge_pattern_coverage": coverage(real_edges, synth_edges),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default="./classifier/memorization_directed_full.csv")
    parser.add_argument("--near-threshold", type=float, default=0.9)
    args = parser.parse_args()

    jobs = [
        ("TT", "normal", "./datasets/anomaly/TT/TT_normal.pt", "./datasets/anomaly/TT/TT_normal_synthetic.pt"),
        ("TT", "abnormal", "./datasets/anomaly/TT/TT_abnormal.pt", "./datasets/anomaly/TT/TT_abnormal_synthetic.pt"),
        ("SN", "normal", "./datasets/anomaly/SN/SN_normal.pt", "./datasets/anomaly/SN/SN_normal_synthetic.pt"),
        ("SN", "abnormal", "./datasets/anomaly/SN/SN_abnormal.pt", "./datasets/anomaly/SN/SN_abnormal_synthetic.pt"),
    ]
    rows = [analyze_pair(*job, near_threshold=args.near_threshold) for job in jobs]
    out = ROOT / args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
