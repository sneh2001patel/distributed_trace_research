import argparse
import csv
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


ROOT = Path(__file__).resolve().parent

SPAN_BINS = [
    ("small", 1, 5),
    ("medium", 6, 10),
    ("large", 11, 20),
    ("very_large", 21, None),
]

DATASET_PRESETS = {
    "anomaly_combined": {
        "SN": {
            "real": [
                "./datasets/anomaly/SN/SN_normal.pt",
                "./datasets/anomaly/SN/SN_abnormal.pt",
            ],
            "synthetic": [
                "./datasets/anomaly/SN/SN_normal_synthetic.pt",
                "./datasets/anomaly/SN/SN_abnormal_synthetic.pt",
            ],
        },
        "TT": {
            "real": [
                "./datasets/anomaly/TT/TT_normal.pt",
                "./datasets/anomaly/TT/TT_abnormal.pt",
            ],
            "synthetic": [
                "./datasets/anomaly/TT/TT_normal_synthetic.pt",
                "./datasets/anomaly/TT/TT_abnormal_synthetic.pt",
            ],
        },
    },
    "anomaly_normal": {
        "SN": {
            "real": ["./datasets/anomaly/SN/SN_normal.pt"],
            "synthetic": ["./datasets/anomaly/SN/SN_normal_synthetic.pt"],
        },
        "TT": {
            "real": ["./datasets/anomaly/TT/TT_normal.pt"],
            "synthetic": ["./datasets/anomaly/TT/TT_normal_synthetic.pt"],
        },
    },
    "anomaly_abnormal": {
        "SN": {
            "real": ["./datasets/anomaly/SN/SN_abnormal.pt"],
            "synthetic": ["./datasets/anomaly/SN/SN_abnormal_synthetic.pt"],
        },
        "TT": {
            "real": ["./datasets/anomaly/TT/TT_abnormal.pt"],
            "synthetic": ["./datasets/anomaly/TT/TT_abnormal_synthetic.pt"],
        },
    },
    "fixed_size": {
        "SN": {
            "real": ["./datasets/SN_data.pt"],
            "synthetic": ["./datasets/fixed_size/SN_synthetic.pt"],
        },
        "TT": {
            "real": ["./datasets/TT_data.pt"],
            "synthetic": ["./datasets/fixed_size/TT_synthetic.pt"],
        },
    },
}


class PTDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        self.data, self.slices = torch.load(datapath, weights_only=False)


def load_graphs(paths: Sequence[str]) -> List[Data]:
    graphs = []
    for path in paths:
        ds = PTDataset(str(ROOT / path))
        graphs.extend(ds.get(i).cpu() for i in range(len(ds)))
    return graphs


def span_bin(num_nodes: int) -> str:
    for name, low, high in SPAN_BINS:
        if num_nodes >= low and (high is None or num_nodes <= high):
            return name
    return "outside"


def graph_depth(graph: Data) -> int:
    n = int(graph.num_nodes)
    if n <= 1:
        return 0

    edge_index = graph.edge_index.detach().cpu().long()
    children = [[] for _ in range(n)]
    undirected = [[] for _ in range(n)]
    indegree = [0] * n
    for src, dst in edge_index.t().tolist():
        if not (0 <= src < n and 0 <= dst < n) or src == dst:
            continue
        children[src].append(dst)
        undirected[src].append(dst)
        undirected[dst].append(src)
        indegree[dst] += 1

    roots = [idx for idx, degree in enumerate(indegree) if degree == 0]
    adjacency = children if roots else undirected
    starts = roots if roots else [0]

    max_depth = 0
    queue = deque((start, 0) for start in starts)
    seen = set(starts)
    while queue:
        node, depth = queue.popleft()
        max_depth = max(max_depth, depth)
        for nxt in adjacency[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, depth + 1))
    return int(max_depth)


def graph_record(graph: Data) -> Dict[str, object]:
    x = graph.x.detach().cpu()
    services = x[:, 0].round().long().numpy()
    ops = x[:, 1].round().long().numpy()
    durations = np.log1p(x[:, 2].float().numpy().clip(0))
    node_count = int(graph.num_nodes)
    edge_count = int(graph.edge_index.size(1))
    return {
        "graph": graph,
        "bin": span_bin(node_count),
        "node_count": node_count,
        "edge_count": edge_count,
        "depth": graph_depth(graph),
        "services": services,
        "ops": ops,
        "durations": durations,
    }


def to_records(graphs: Sequence[Data]) -> List[Dict[str, object]]:
    return [graph_record(graph) for graph in graphs]


def stratified_sample(
    real_records: Sequence[Dict[str, object]],
    synthetic_records: Sequence[Dict[str, object]],
    seed: int,
    target_count: int | None,
) -> tuple[List[Dict[str, object]], Dict[str, int], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    real_by_bin = group_by_bin(real_records)
    synth_by_bin = group_by_bin(synthetic_records)

    total = target_count or len(real_records)
    bin_names = [name for name, _, _ in SPAN_BINS]
    real_counts = {name: len(real_by_bin.get(name, [])) for name in bin_names}
    raw_targets = {
        name: total * (real_counts[name] / max(len(real_records), 1)) for name in bin_names
    }
    targets = {name: int(np.floor(raw_targets[name])) for name in bin_names}
    remainder = total - sum(targets.values())
    for name in sorted(bin_names, key=lambda n: raw_targets[n] - targets[n], reverse=True):
        if remainder <= 0:
            break
        targets[name] += 1
        remainder -= 1

    sampled = []
    shortfalls = {}
    all_synth = list(synthetic_records)
    for name in bin_names:
        need = targets[name]
        candidates = synth_by_bin.get(name, [])
        if need == 0:
            shortfalls[name] = 0
            continue
        if candidates:
            replace = len(candidates) < need
            indices = rng.choice(len(candidates), size=need, replace=replace)
            sampled.extend(candidates[int(i)] for i in indices)
            shortfalls[name] = max(0, need - len(candidates))
        elif all_synth:
            indices = rng.choice(len(all_synth), size=need, replace=len(all_synth) < need)
            sampled.extend(all_synth[int(i)] for i in indices)
            shortfalls[name] = need
        else:
            shortfalls[name] = need
    return sampled, targets, shortfalls


def random_sample_records(
    records: Sequence[Dict[str, object]], sample_size: int | None, seed: int
) -> List[Dict[str, object]]:
    if sample_size is None or sample_size >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=sample_size, replace=False)
    return [records[int(index)] for index in indices]


def group_by_bin(records: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped = {name: [] for name, _, _ in SPAN_BINS}
    for record in records:
        grouped.setdefault(str(record["bin"]), []).append(record)
    return grouped


def flatten(records: Sequence[Dict[str, object]], key: str) -> np.ndarray:
    values = []
    for record in records:
        value = record[key]
        if isinstance(value, np.ndarray):
            values.extend(value.tolist())
        else:
            values.append(value)
    return np.asarray(values)


def categorical_jsd(real_values: np.ndarray, synthetic_values: np.ndarray) -> float:
    if real_values.size == 0 or synthetic_values.size == 0:
        return 0.0
    support = np.unique(np.concatenate([real_values, synthetic_values]).astype(np.int64))
    real_counts = Counter(real_values.astype(np.int64).tolist())
    synth_counts = Counter(synthetic_values.astype(np.int64).tolist())
    p = np.asarray([real_counts.get(int(value), 0) for value in support], dtype=float)
    q = np.asarray([synth_counts.get(int(value), 0) for value in support], dtype=float)
    p = (p + 1e-12) / (p.sum() + 1e-12 * len(p))
    q = (q + 1e-12) / (q.sum() + 1e-12 * len(q))
    return float(jensenshannon(p, q, base=2.0))


def numeric_wasserstein(real_values: np.ndarray, synthetic_values: np.ndarray) -> float:
    if real_values.size == 0 or synthetic_values.size == 0:
        return 0.0
    return float(wasserstein_distance(real_values.astype(float), synthetic_values.astype(float)))


def mean_abs_diff(real_values: np.ndarray, synthetic_values: np.ndarray) -> float:
    if real_values.size == 0 or synthetic_values.size == 0:
        return 0.0
    return float(abs(np.mean(real_values.astype(float)) - np.mean(synthetic_values.astype(float))))


def compare_records(
    real_records: Sequence[Dict[str, object]],
    synthetic_records: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    real_services = flatten(real_records, "services")
    synth_services = flatten(synthetic_records, "services")
    real_ops = flatten(real_records, "ops")
    synth_ops = flatten(synthetic_records, "ops")
    real_durations = flatten(real_records, "durations")
    synth_durations = flatten(synthetic_records, "durations")

    max_op = int(max(real_ops.max(initial=0), synth_ops.max(initial=0))) + 1
    real_joint = real_services.astype(np.int64) * max_op + real_ops.astype(np.int64)
    synth_joint = synth_services.astype(np.int64) * max_op + synth_ops.astype(np.int64)

    real_nodes = flatten(real_records, "node_count")
    synth_nodes = flatten(synthetic_records, "node_count")
    real_edges = flatten(real_records, "edge_count")
    synth_edges = flatten(synthetic_records, "edge_count")
    real_depth = flatten(real_records, "depth")
    synth_depth = flatten(synthetic_records, "depth")

    return {
        "real_traces": len(real_records),
        "synthetic_traces": len(synthetic_records),
        "service_js_distance": categorical_jsd(real_services, synth_services),
        "operation_js_distance": categorical_jsd(real_ops, synth_ops),
        "service_operation_js_distance": categorical_jsd(real_joint, synth_joint),
        "duration_wasserstein": numeric_wasserstein(real_durations, synth_durations),
        "duration_mean_mae": mean_abs_diff(real_durations, synth_durations),
        "node_count_mae": mean_abs_diff(real_nodes, synth_nodes),
        "edge_count_mae": mean_abs_diff(real_edges, synth_edges),
        "depth_mae": mean_abs_diff(real_depth, synth_depth),
        "edge_count_wasserstein": numeric_wasserstein(real_edges, synth_edges),
        "depth_wasserstein": numeric_wasserstein(real_depth, synth_depth),
    }


def bin_count_columns(records: Sequence[Dict[str, object]], prefix: str) -> Dict[str, int]:
    counts = Counter(str(record["bin"]) for record in records)
    return {f"{prefix}_{name}_count": counts.get(name, 0) for name, _, _ in SPAN_BINS}


def format_value(key: str, value: object) -> str:
    if isinstance(value, str):
        return value
    if key in {"real_traces", "synthetic_traces"}:
        return str(int(float(value)))
    number = float(value)
    return f"{number:.3f}"


def write_markdown(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    fields = [
        ("system", "System"),
        ("stratum", "Stratum"),
        ("real_traces", "Real"),
        ("synthetic_traces", "Synthetic"),
        ("service_js_distance", "Svc JSD"),
        ("operation_js_distance", "Op JSD"),
        ("service_operation_js_distance", "Svc-Op JSD"),
        ("duration_wasserstein", "Dur WDist (log1p)"),
        ("duration_mean_mae", "Dur MAE (log1p)"),
        ("edge_count_mae", "Edge MAE"),
        ("depth_mae", "Depth MAE"),
        ("node_count_mae", "Span MAE"),
    ]
    lines = [
        "# RQ4 Stratified Representativeness",
        "",
        "Synthetic traces are sampled to match the real trace-size bin proportions. Span bins are small `1-5`, medium `6-10`, large `11-20`, and very large `>20` spans. Lower distance and MAE values indicate closer distributional similarity.",
        "",
        "| " + " | ".join(title for _, title in fields) + " |",
        "|---" + "|---:" * (len(fields) - 1) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(format_value(key, row[key]) for key, _ in fields) + " |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def evaluate_system(
    system: str,
    preset: str,
    seed: int,
    target_count: int | None,
    real_sample_size: int | None,
) -> List[Dict[str, object]]:
    paths = DATASET_PRESETS[preset][system]
    real_records = to_records(load_graphs(paths["real"]))
    synthetic_records = to_records(load_graphs(paths["synthetic"]))
    real_total_before_sampling = len(real_records)
    real_records = random_sample_records(
        real_records, sample_size=real_sample_size, seed=seed
    )
    sampled_synthetic, targets, shortfalls = stratified_sample(
        real_records, synthetic_records, seed=seed, target_count=target_count
    )

    rows = []
    overall = compare_records(real_records, sampled_synthetic)
    overall.update(
        {
            "system": system,
            "preset": preset,
            "stratum": "all_stratified",
            "real_total_before_sampling": real_total_before_sampling,
            "real_sample_size": len(real_records),
            **bin_count_columns(real_records, "real"),
            **bin_count_columns(sampled_synthetic, "synthetic"),
            **{f"target_{name}_count": count for name, count in targets.items()},
            **{f"synthetic_{name}_shortfall": count for name, count in shortfalls.items()},
        }
    )
    rows.append(overall)

    real_by_bin = group_by_bin(real_records)
    synth_by_bin = group_by_bin(sampled_synthetic)
    for name, _, _ in SPAN_BINS:
        real_bin = real_by_bin.get(name, [])
        synth_bin = synth_by_bin.get(name, [])
        if not real_bin and not synth_bin:
            continue
        row = compare_records(real_bin, synth_bin)
        row.update(
            {
                "system": system,
                "preset": preset,
                "stratum": name,
                "real_total_before_sampling": real_total_before_sampling,
                "real_sample_size": len(real_records),
                "target_bin_count": targets.get(name, 0),
                "synthetic_bin_shortfall": shortfalls.get(name, 0),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified representativeness metrics for real vs synthetic traces."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DATASET_PRESETS),
        default="anomaly_combined",
        help="Dataset path preset. Defaults to normal+abnormal anomaly datasets per system.",
    )
    parser.add_argument("--systems", nargs="+", choices=["SN", "TT"], default=["SN", "TT"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Synthetic sample size per system. Defaults to the number of real traces.",
    )
    parser.add_argument(
        "--real-sample-size",
        type=int,
        default=None,
        help="Optional random sample size for real traces before stratifying synthetic traces.",
    )
    parser.add_argument(
        "--out-csv",
        default="./classifier/rq4_stratified_representativeness.csv",
    )
    parser.add_argument(
        "--out-md",
        default="./classifier/rq4_stratified_representativeness.md",
    )
    args = parser.parse_args()

    rows = []
    for system in args.systems:
        system_rows = evaluate_system(
            system=system,
            preset=args.preset,
            seed=args.seed,
            target_count=args.target_count,
            real_sample_size=args.real_sample_size,
        )
        rows.extend(system_rows)
        overall = system_rows[0]
        print(
            f"{system} {args.preset}: "
            f"svc_op_jsd={overall['service_operation_js_distance']:.3f} "
            f"duration_w={overall['duration_wasserstein']:.3f} "
            f"edge_mae={overall['edge_count_mae']:.3f} "
            f"depth_mae={overall['depth_mae']:.3f}"
        )

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
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
