import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


ROOT = Path(__file__).resolve().parent


DATASET_PRESETS = {
    "fixed_size": {
        "SN": {
            "real": "./datasets/SN_data.pt",
            "synthetic": "./datasets/fixed_size/SN_synthetic.pt",
        },
        "TT": {
            "real": "./datasets/TT_data.pt",
            "synthetic": "./datasets/fixed_size/TT_synthetic.pt",
        },
    },
    "fixed_size_precise": {
        "SN": {
            "real": "./datasets/SN_data.pt",
            "synthetic": "./datasets/fixed_size/order_SN_fixed_size_precise.pt",
        },
        "TT": {
            "real": "./datasets/TT_data.pt",
            "synthetic": "./datasets/fixed_size/order_TT_fixed_size_precise.pt",
        },
    },
    "anomaly_normal": {
        "SN": {
            "real": "./datasets/anomaly/SN/SN_normal.pt",
            "synthetic": "./datasets/anomaly/SN/SN_normal_synthetic.pt",
        },
        "TT": {
            "real": "./datasets/anomaly/TT/TT_normal.pt",
            "synthetic": "./datasets/anomaly/TT/TT_normal_synthetic.pt",
        },
    },
    "anomaly_abnormal": {
        "SN": {
            "real": "./datasets/anomaly/SN/SN_abnormal.pt",
            "synthetic": "./datasets/anomaly/SN/SN_abnormal_synthetic.pt",
        },
        "TT": {
            "real": "./datasets/anomaly/TT/TT_abnormal.pt",
            "synthetic": "./datasets/anomaly/TT/TT_abnormal_synthetic.pt",
        },
    },
}


class PTDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        self.data, self.slices = torch.load(datapath, weights_only=False)


@dataclass(frozen=True)
class TraceSignature:
    node_labels: frozenset[Tuple[int, int]]
    edge_labels: frozenset[Tuple[Tuple[int, int], Tuple[int, int]]]

    @property
    def items(self) -> frozenset[Tuple[str, object]]:
        node_items = (("n", node_label) for node_label in self.node_labels)
        edge_items = (("e", edge_label) for edge_label in self.edge_labels)
        return frozenset([*node_items, *edge_items])


def load_graphs(path: str, limit: int | None, seed: int) -> List[Data]:
    ds = PTDataset(str(ROOT / path))
    graphs = [ds.get(i).cpu() for i in range(len(ds))]
    if limit is not None and len(graphs) > limit:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(graphs), size=limit, replace=False)
        graphs = [graphs[int(i)] for i in indices]
    return graphs


def graph_to_signature(graph: Data) -> TraceSignature:
    x = graph.x.detach().cpu()
    service_op = [
        (int(round(float(row[0]))), int(round(float(row[1])))) for row in x.tolist()
    ]
    node_labels = frozenset(service_op)

    edges = []
    edge_index = graph.edge_index.detach().cpu().long()
    for src, dst in edge_index.t().tolist():
        if 0 <= src < len(service_op) and 0 <= dst < len(service_op):
            edges.append((service_op[src], service_op[dst]))
    return TraceSignature(node_labels=node_labels, edge_labels=frozenset(edges))


def jaccard_distance(left: frozenset, right: frozenset) -> float:
    if not left and not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return 1.0 - (intersection / union if union else 1.0)


def nearest_neighbor_distances(
    query: Sequence[TraceSignature], reference: Sequence[TraceSignature]
) -> List[float]:
    if not query or not reference:
        return []
    reference_items = [sig.items for sig in reference]
    distances = []
    for sig in query:
        items = sig.items
        distances.append(min(jaccard_distance(items, ref) for ref in reference_items))
    return distances


def self_nearest_neighbor_distances(signatures: Sequence[TraceSignature]) -> List[float]:
    if len(signatures) < 2:
        return []
    items = [sig.items for sig in signatures]
    distances = []
    for i, left in enumerate(items):
        best = math.inf
        for j, right in enumerate(items):
            if i == j:
                continue
            best = min(best, jaccard_distance(left, right))
        distances.append(float(best))
    return distances


def duplicate_rate(signatures: Sequence[TraceSignature]) -> float:
    if not signatures:
        return 0.0
    counts = Counter(signatures)
    duplicated = sum(count for count in counts.values() if count > 1)
    return duplicated / len(signatures)


def entropy(counts: Iterable[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts:
        if count:
            p = count / total
            value -= p * math.log2(p)
    return value


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def coverage(real_patterns: set, synthetic_patterns: set) -> float:
    if not real_patterns:
        return 0.0
    return len(real_patterns & synthetic_patterns) / len(real_patterns)


def compare_signatures(
    real_graphs: Sequence[Data],
    synthetic_graphs: Sequence[Data],
    near_duplicate_threshold: float,
) -> Dict[str, float]:
    real_signatures = [graph_to_signature(graph) for graph in real_graphs]
    synthetic_signatures = [graph_to_signature(graph) for graph in synthetic_graphs]

    real_counts = Counter(real_signatures)
    synthetic_counts = Counter(synthetic_signatures)

    real_node_patterns = set().union(*(sig.node_labels for sig in real_signatures))
    synthetic_node_patterns = set().union(
        *(sig.node_labels for sig in synthetic_signatures)
    )
    real_edge_patterns = set().union(*(sig.edge_labels for sig in real_signatures))
    synthetic_edge_patterns = set().union(
        *(sig.edge_labels for sig in synthetic_signatures)
    )

    nn_distances = nearest_neighbor_distances(synthetic_signatures, real_signatures)
    synth_self_distances = self_nearest_neighbor_distances(synthetic_signatures)

    exact_matches = sum(1 for sig in synthetic_signatures if sig in real_counts)
    near_duplicates = sum(
        1 for distance in synth_self_distances if distance <= near_duplicate_threshold
    )

    return {
        "real_traces": len(real_signatures),
        "synthetic_traces": len(synthetic_signatures),
        "real_unique_trace_signatures": len(real_counts),
        "synthetic_unique_trace_signatures": len(synthetic_counts),
        "real_unique_trace_signature_rate": len(real_counts) / max(len(real_signatures), 1),
        "synthetic_unique_trace_signature_rate": len(synthetic_counts)
        / max(len(synthetic_signatures), 1),
        "real_trace_signature_entropy": entropy(real_counts.values()),
        "synthetic_trace_signature_entropy": entropy(synthetic_counts.values()),
        "duplicate_rate": duplicate_rate(synthetic_signatures),
        "near_duplicate_rate": near_duplicates / max(len(synthetic_signatures), 1),
        "near_duplicate_threshold": near_duplicate_threshold,
        "exact_train_match_rate": exact_matches / max(len(synthetic_signatures), 1),
        "novelty_rate": 1.0 - exact_matches / max(len(synthetic_signatures), 1),
        "node_pattern_coverage": coverage(real_node_patterns, synthetic_node_patterns),
        "edge_pattern_coverage": coverage(real_edge_patterns, synthetic_edge_patterns),
        "service_operation_pattern_coverage": coverage(
            real_node_patterns | real_edge_patterns,
            synthetic_node_patterns | synthetic_edge_patterns,
        ),
        "real_unique_node_patterns": len(real_node_patterns),
        "synthetic_unique_node_patterns": len(synthetic_node_patterns),
        "real_unique_edge_patterns": len(real_edge_patterns),
        "synthetic_unique_edge_patterns": len(synthetic_edge_patterns),
        "nearest_real_distance_mean": float(np.mean(nn_distances)) if nn_distances else 0.0,
        "nearest_real_distance_median": quantile(nn_distances, 0.50),
        "nearest_real_distance_p05": quantile(nn_distances, 0.05),
        "nearest_real_distance_p95": quantile(nn_distances, 0.95),
        "nearest_real_similarity_mean": 1.0
        - (float(np.mean(nn_distances)) if nn_distances else 0.0),
    }


def write_markdown(rows: Sequence[Dict[str, float]], out_path: Path) -> None:
    fields = [
        ("system", "System"),
        ("preset", "Preset"),
        ("real_traces", "Real"),
        ("synthetic_traces", "Synthetic"),
        ("nearest_real_distance_mean", "NN Dist"),
        ("nearest_real_similarity_mean", "NN Sim"),
        ("duplicate_rate", "Dup Rate"),
        ("near_duplicate_rate", "Near Dup"),
        ("node_pattern_coverage", "Node Cov"),
        ("edge_pattern_coverage", "Edge Cov"),
        ("synthetic_unique_trace_signature_rate", "Unique Rate"),
        ("novelty_rate", "Novelty"),
    ]

    def fmt(value):
        if isinstance(value, str):
            return value
        if isinstance(value, int):
            return str(value)
        return f"{float(value):.3f}"

    lines = [
        "# RQ4 Trace Signature Similarity",
        "",
        "Trace signature = set of `(service, operation)` node labels plus set of directed edge labels `(parent_service, parent_operation) -> (child_service, child_operation)`. Nearest-neighbor distance is Jaccard distance over the combined node/edge signature items.",
        "",
        "| " + " | ".join(title for _, title in fields) + " |",
        "|---" + "|---:" * (len(fields) - 1) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key, _ in fields) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ4 trace-signature similarity metrics for real vs synthetic traces."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DATASET_PRESETS),
        default="fixed_size",
        help="Dataset path preset. Defaults to the 1,244 SN/TT fixed-size hierarchical VAE datasets.",
    )
    parser.add_argument("--systems", nargs="+", choices=["SN", "TT"], default=["SN", "TT"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap per real/synthetic dataset. By default all available graphs are used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.05,
        help="Jaccard distance threshold for synthetic near-duplicates.",
    )
    parser.add_argument("--out-csv", default="./classifier/rq4_trace_signature_metrics.csv")
    parser.add_argument("--out-md", default="./classifier/rq4_trace_signature_metrics.md")
    args = parser.parse_args()

    rows = []
    for system in args.systems:
        paths = DATASET_PRESETS[args.preset][system]
        real_graphs = load_graphs(paths["real"], limit=args.limit, seed=args.seed)
        synthetic_graphs = load_graphs(
            paths["synthetic"], limit=args.limit, seed=args.seed
        )

        if len(synthetic_graphs) > len(real_graphs):
            synthetic_graphs = synthetic_graphs[: len(real_graphs)]
        elif len(real_graphs) > len(synthetic_graphs):
            real_graphs = real_graphs[: len(synthetic_graphs)]

        row = compare_signatures(
            real_graphs,
            synthetic_graphs,
            near_duplicate_threshold=args.near_duplicate_threshold,
        )
        row.update(
            {
                "system": system,
                "preset": args.preset,
                "real_path": paths["real"],
                "synthetic_path": paths["synthetic"],
            }
        )
        rows.append(row)
        print(
            f"{system} {args.preset}: "
            f"nn_dist={row['nearest_real_distance_mean']:.3f} "
            f"dup={row['duplicate_rate']:.3f} "
            f"near_dup={row['near_duplicate_rate']:.3f} "
            f"coverage={row['service_operation_pattern_coverage']:.3f} "
            f"novelty={row['novelty_rate']:.3f}"
        )

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["system", "preset", "real_path", "synthetic_path"] + [
        key
        for key in rows[0].keys()
        if key not in {"system", "preset", "real_path", "synthetic_path"}
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_md = ROOT / args.out_md
    write_markdown(rows, out_md)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
