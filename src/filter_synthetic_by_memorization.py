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


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_graphs(path: str):
    ds = PTDataset(str(resolve(path)))
    return [ds.get(i).cpu() for i in range(len(ds))]


def save_graphs(graphs, path: str):
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data, slices = InMemoryDataset.collate(list(graphs))
    torch.save((data, slices), out)


def node_labels(g):
    x = g.x.detach().cpu()
    return tuple(sorted((int(row[0].item()), int(row[1].item())) for row in x))


def service_set(g):
    x = g.x.detach().cpu()
    return set((int(row[0].item()), int(row[1].item())) for row in x)


def edge_labels(g):
    x = g.x.detach().cpu()
    labels = [(int(row[0].item()), int(row[1].item())) for row in x]
    edges = []
    if g.edge_index.numel() > 0:
        for src, dst in g.edge_index.t().detach().cpu().tolist():
            if src < len(labels) and dst < len(labels):
                edges.append((labels[src], labels[dst]))
    return tuple(sorted(edges))


def edge_set(g):
    return set(edge_labels(g))


def signature(g):
    return (node_labels(g), edge_labels(g))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def combined_similarity(synth_graph, real_edge_sets, real_service_sets):
    s_edges = edge_set(synth_graph)
    s_services = service_set(synth_graph)
    best = 0.0
    for r_edges, r_services in zip(real_edge_sets, real_service_sets):
        score = 0.7 * jaccard(s_edges, r_edges) + 0.3 * jaccard(s_services, r_services)
        if score > best:
            best = score
    return best


def summarize(graphs, real_sig_set, real_edge_sets, real_service_sets, threshold):
    if not graphs:
        return {
            "unique_ratio": 0.0,
            "exact_duplicate_rate": 0.0,
            "near_duplicate_rate": 0.0,
            "mean_nearest_similarity": 0.0,
            "p95_nearest_similarity": 0.0,
        }
    sigs = [signature(g) for g in graphs]
    sims = [combined_similarity(g, real_edge_sets, real_service_sets) for g in graphs]
    sims_sorted = sorted(sims)
    p95_idx = min(len(sims_sorted) - 1, int(round(0.95 * (len(sims_sorted) - 1))))
    return {
        "unique_ratio": len(set(sigs)) / len(sigs),
        "exact_duplicate_rate": sum(1 for sig in sigs if sig in real_sig_set) / len(sigs),
        "near_duplicate_rate": sum(1 for sim in sims if sim >= threshold) / len(sims),
        "mean_nearest_similarity": sum(sims) / len(sims),
        "p95_nearest_similarity": sims_sorted[p95_idx],
    }


def append_csv(path: str, row: dict):
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--trace-class", required=True, choices=["normal", "abnormal"])
    parser.add_argument("--generator", required=True)
    parser.add_argument("--filter-name", required=True)
    parser.add_argument("--real-path", required=True)
    parser.add_argument("--synth-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()

    real = load_graphs(args.real_path)
    synth = load_graphs(args.synth_path)
    real_sig_set = set(signature(g) for g in real)
    real_edge_sets = [edge_set(g) for g in real]
    real_service_sets = [service_set(g) for g in real]

    kept = []
    rejected_exact = 0
    rejected_near = 0
    for graph in synth:
        sig = signature(graph)
        if sig in real_sig_set:
            rejected_exact += 1
            continue
        sim = combined_similarity(graph, real_edge_sets, real_service_sets)
        if not args.exact_only and sim >= args.threshold:
            rejected_near += 1
            continue
        kept.append(graph)

    save_graphs(kept, args.out_path)
    before = summarize(synth, real_sig_set, real_edge_sets, real_service_sets, args.threshold)
    after = summarize(kept, real_sig_set, real_edge_sets, real_service_sets, args.threshold)
    row = {
        "system": args.system,
        "trace_class": args.trace_class,
        "generator": args.generator,
        "filter": args.filter_name,
        "threshold": args.threshold,
        "exact_only": args.exact_only,
        "input_path": args.synth_path,
        "output_path": args.out_path,
        "real_graphs": len(real),
        "requested_count": len(synth),
        "candidate_count": len(synth),
        "retained_count": len(kept),
        "retention_rate": len(kept) / len(synth) if synth else 0.0,
        "rejected_exact": rejected_exact,
        "rejected_near": rejected_near,
        "unique_synthetic_ratio_before": before["unique_ratio"],
        "unique_synthetic_ratio_after": after["unique_ratio"],
        "exact_duplicate_rate_before": before["exact_duplicate_rate"],
        "exact_duplicate_rate_after": after["exact_duplicate_rate"],
        "near_duplicate_rate_before": before["near_duplicate_rate"],
        "near_duplicate_rate_after": after["near_duplicate_rate"],
        "mean_nearest_similarity_before": before["mean_nearest_similarity"],
        "mean_nearest_similarity_after": after["mean_nearest_similarity"],
        "p95_nearest_similarity_before": before["p95_nearest_similarity"],
        "p95_nearest_similarity_after": after["p95_nearest_similarity"],
    }
    append_csv(args.stats_csv, row)
    print(row)


if __name__ == "__main__":
    main()
