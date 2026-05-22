import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


ROOT = Path(__file__).resolve().parent


DATASETS = {
    "TT": {
        "real": {
            "normal": "./datasets/anomaly/TT/TT_normal.pt",
            "abnormal": "./datasets/anomaly/TT/TT_abnormal.pt",
        },
        "synthetic": {
            "random_sampling": {
                "normal": "./datasets/baselines/TT_normal_random_sampling.pt",
                "abnormal": "./datasets/baselines/TT_abnormal_random_sampling.pt",
            },
            "empirical": {
                "normal": "./datasets/baselines/TT_normal_empirical_random_tree.pt",
                "abnormal": "./datasets/baselines/TT_abnormal_empirical_random_tree.pt",
            },
            "flat_vae": {
                "normal": "./datasets/baselines/TT_normal_flat_vae_synthetic.pt",
                "abnormal": "./datasets/baselines/TT_abnormal_flat_vae_synthetic.pt",
            },
            "hierarchical_vae": {
                "normal": "./datasets/anomaly/TT/TT_normal_synthetic.pt",
                "abnormal": "./datasets/anomaly/TT/TT_abnormal_synthetic.pt",
            },
        },
        "min_nodes": 3,
    },
    "SN": {
        "real": {
            "normal": "./datasets/anomaly/SN/SN_normal.pt",
            "abnormal": "./datasets/anomaly/SN/SN_abnormal.pt",
        },
        "synthetic": {
            "random_sampling": {
                "normal": "./datasets/baselines/SN_normal_random_sampling.pt",
                "abnormal": "./datasets/baselines/SN_abnormal_random_sampling.pt",
            },
            "empirical": {
                "normal": "./datasets/baselines/SN_normal_empirical_random_tree.pt",
                "abnormal": "./datasets/baselines/SN_abnormal_empirical_random_tree.pt",
            },
            "flat_vae": {
                "normal": "./datasets/baselines/SN_normal_flat_vae_synthetic.pt",
                "abnormal": "./datasets/baselines/SN_abnormal_flat_vae_synthetic.pt",
            },
            "hierarchical_vae": {
                "normal": "./datasets/anomaly/SN/SN_normal_synthetic.pt",
                "abnormal": "./datasets/anomaly/SN/SN_abnormal_synthetic.pt",
            },
        },
        "min_nodes": 1,
    },
}


class PTDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        self.data, self.slices = torch.load(datapath, weights_only=False)


def load_graphs(path: str, min_nodes: int = 1, max_graphs: int = None, seed: int = 42):
    ds = PTDataset(str(ROOT / path))
    graphs = [ds.get(i).cpu() for i in range(len(ds))]
    if min_nodes > 1:
        graphs = [g for g in graphs if g.num_nodes >= min_nodes]
    if max_graphs is not None and len(graphs) > max_graphs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(graphs), size=max_graphs, replace=False)
        graphs = [graphs[int(i)] for i in idx]
    return graphs


def graph_stats(graphs: List[Data]) -> Dict[str, np.ndarray]:
    node_count = []
    edge_count = []
    density = []
    avg_degree = []
    max_degree = []
    roots_ratio = []
    leaves_ratio = []
    duration_mean = []
    duration_std = []
    unique_services = []
    unique_ops = []
    all_degrees = []
    all_durations = []
    all_services = []
    all_ops = []
    all_joint = []

    for g in graphs:
        n = int(g.num_nodes)
        e = int(g.edge_index.size(1))
        x = g.x.detach().cpu()
        services = x[:, 0].round().long().numpy()
        ops = x[:, 1].round().long().numpy()
        durations = x[:, 2].float().numpy()

        if e > 0:
            src = g.edge_index[0].detach().cpu().long()
            dst = g.edge_index[1].detach().cpu().long()
            indeg = torch.bincount(dst, minlength=n).numpy()
            outdeg = torch.bincount(src, minlength=n).numpy()
            deg = indeg + outdeg
        else:
            indeg = np.zeros(n, dtype=np.int64)
            outdeg = np.zeros(n, dtype=np.int64)
            deg = np.zeros(n, dtype=np.int64)

        possible_directed = max(n * (n - 1), 1)
        node_count.append(n)
        edge_count.append(e)
        density.append(e / possible_directed)
        avg_degree.append(float(deg.mean()) if n else 0.0)
        max_degree.append(float(deg.max()) if n else 0.0)
        roots_ratio.append(float((indeg == 0).sum() / max(n, 1)))
        leaves_ratio.append(float((outdeg == 0).sum() / max(n, 1)))
        duration_mean.append(float(np.mean(durations)) if durations.size else 0.0)
        duration_std.append(float(np.std(durations)) if durations.size else 0.0)
        unique_services.append(len(np.unique(services)))
        unique_ops.append(len(np.unique(ops)))
        all_degrees.extend(deg.tolist())
        all_durations.extend(durations.tolist())
        all_services.extend(services.tolist())
        all_ops.extend(ops.tolist())

    return {
        "node_count": np.asarray(node_count, dtype=float),
        "edge_count": np.asarray(edge_count, dtype=float),
        "density": np.asarray(density, dtype=float),
        "avg_degree": np.asarray(avg_degree, dtype=float),
        "max_degree": np.asarray(max_degree, dtype=float),
        "roots_ratio": np.asarray(roots_ratio, dtype=float),
        "leaves_ratio": np.asarray(leaves_ratio, dtype=float),
        "duration_mean": np.asarray(duration_mean, dtype=float),
        "duration_std": np.asarray(duration_std, dtype=float),
        "unique_services": np.asarray(unique_services, dtype=float),
        "unique_ops": np.asarray(unique_ops, dtype=float),
        "all_degrees": np.asarray(all_degrees, dtype=float),
        "all_durations": np.asarray(all_durations, dtype=float),
        "all_services": np.asarray(all_services, dtype=np.int64),
        "all_ops": np.asarray(all_ops, dtype=np.int64),
    }


def ks_similarity(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    stat, p_value = ks_2samp(a, b)
    return float(1.0 - stat), float(stat)


def normalized_wasserstein_similarity(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0
    distance = float(wasserstein_distance(a, b))
    scale = max(float(np.std(np.concatenate([a, b]))), float(abs(np.mean(a))), float(abs(np.mean(b))), 1e-9)
    return float(max(0.0, 1.0 - distance / scale)), distance


def pmf(vals, support):
    counts = np.zeros(len(support), dtype=float)
    index = {int(v): i for i, v in enumerate(support)}
    for val in vals:
        i = index.get(int(val))
        if i is not None:
            counts[i] += 1.0
    if counts.sum() == 0:
        return counts
    return counts / counts.sum()


def js_similarity(real_vals, synth_vals):
    support = np.unique(np.concatenate([real_vals, synth_vals]))
    if support.size == 0:
        return 0.0
    p = pmf(real_vals, support) + 1e-12
    q = pmf(synth_vals, support) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float(1.0 - jensenshannon(p, q, base=2.0))


def service_operation_validity(real_joint, synth_joint):
    if len(synth_joint) == 0:
        return 0.0
    real_support = set(int(v) for v in real_joint)
    valid = sum(1 for v in synth_joint if int(v) in real_support)
    return valid / len(synth_joint)


def graph_feature_matrix(stats: Dict[str, np.ndarray]) -> np.ndarray:
    keys = [
        "node_count",
        "edge_count",
        "density",
        "avg_degree",
        "max_degree",
        "roots_ratio",
        "leaves_ratio",
        "duration_mean",
        "duration_std",
        "unique_services",
        "unique_ops",
    ]
    cols = []
    for key in keys:
        vals = stats[key]
        if key in {"node_count", "edge_count", "duration_mean", "duration_std"}:
            vals = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
        cols.append(vals)
    return np.stack(cols, axis=1) if cols else np.empty((0, 0))


def rbf_mmd_similarity(real_features, synth_features):
    if len(real_features) == 0 or len(synth_features) == 0:
        return 0.0, 0.0
    combined = np.vstack([real_features, synth_features])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True) + 1e-9
    x = (real_features - mean) / std
    y = (synth_features - mean) / std

    sample_cap = 800
    rng = np.random.default_rng(42)
    if len(x) > sample_cap:
        x = x[rng.choice(len(x), size=sample_cap, replace=False)]
    if len(y) > sample_cap:
        y = y[rng.choice(len(y), size=sample_cap, replace=False)]

    z = np.vstack([x, y])
    sq = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    nonzero = sq[sq > 0]
    gamma = 1.0 / (2.0 * (np.median(nonzero) if nonzero.size else 1.0))

    def kernel(a, b):
        dist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * dist)

    kxx = kernel(x, x).mean()
    kyy = kernel(y, y).mean()
    kxy = kernel(x, y).mean()
    mmd = max(float(kxx + kyy - 2.0 * kxy), 0.0)
    return float(math.exp(-mmd)), mmd


def compare(real_graphs: List[Data], synth_graphs: List[Data]) -> Dict[str, float]:
    real = graph_stats(real_graphs)
    synth = graph_stats(synth_graphs)
    max_op = int(
        max(
            real["all_ops"].max(initial=0) if real["all_ops"].size else 0,
            synth["all_ops"].max(initial=0) if synth["all_ops"].size else 0,
        )
    ) + 1
    real_joint = real["all_services"].astype(np.int64) * max_op + real["all_ops"].astype(np.int64)
    synth_joint = synth["all_services"].astype(np.int64) * max_op + synth["all_ops"].astype(np.int64)

    row = {
        "real_graphs": len(real_graphs),
        "synthetic_graphs": len(synth_graphs),
    }
    ks_keys = [
        "node_count",
        "edge_count",
        "density",
        "avg_degree",
        "max_degree",
        "roots_ratio",
        "leaves_ratio",
        "duration_mean",
        "duration_std",
        "unique_services",
        "unique_ops",
        "all_degrees",
        "all_durations",
    ]
    ks_sims = []
    for key in ks_keys:
        sim, stat = ks_similarity(real[key], synth[key])
        row[f"{key}_ks_similarity"] = sim
        row[f"{key}_ks_stat"] = stat
        ks_sims.append(sim)

    for key in ["node_count", "edge_count", "density", "avg_degree"]:
        sim, dist = normalized_wasserstein_similarity(real[key], synth[key])
        row[f"{key}_wasserstein_similarity"] = sim
        row[f"{key}_wasserstein_distance"] = dist

    row["service_js_similarity"] = js_similarity(real["all_services"], synth["all_services"])
    row["operation_js_similarity"] = js_similarity(real["all_ops"], synth["all_ops"])
    row["service_operation_js_similarity"] = js_similarity(real_joint, synth_joint)
    row["service_operation_valid_mass"] = service_operation_validity(
        real_joint, synth_joint
    )

    mmd_sim, mmd = rbf_mmd_similarity(graph_feature_matrix(real), graph_feature_matrix(synth))
    row["graph_feature_mmd_similarity"] = mmd_sim
    row["graph_feature_mmd"] = mmd

    row["structure_mean"] = float(
        np.mean(
            [
                row["node_count_ks_similarity"],
                row["edge_count_ks_similarity"],
                row["density_ks_similarity"],
                row["avg_degree_ks_similarity"],
                row["max_degree_ks_similarity"],
                row["all_degrees_ks_similarity"],
                row["graph_feature_mmd_similarity"],
            ]
        )
    )
    row["timing_mean"] = float(
        np.mean(
            [
                row["duration_mean_ks_similarity"],
                row["all_durations_ks_similarity"],
            ]
        )
    )
    row["semantic_mean"] = float(
        np.mean(
            [
                row["service_js_similarity"],
                row["operation_js_similarity"],
                row["service_operation_js_similarity"],
                row["service_operation_valid_mass"],
            ]
        )
    )
    row["overall_fidelity"] = float(
        np.mean(
            [
                row["structure_mean"],
                row["timing_mean"],
                row["semantic_mean"],
            ]
        )
    )
    return row


def write_markdown(rows: List[Dict[str, float]], out_path: Path):
    def fmt(v):
        return v if isinstance(v, str) else f"{float(v):.3f}"

    struct_cols = [
        ("node_count_ks_similarity", "Node KS"),
        ("edge_count_ks_similarity", "Edge KS"),
        ("density_ks_similarity", "Density KS"),
        ("avg_degree_ks_similarity", "AvgDeg KS"),
        ("max_degree_ks_similarity", "MaxDeg KS"),
        ("all_degrees_ks_similarity", "AllDeg KS"),
        ("graph_feature_mmd_similarity", "MMD Sim"),
        ("structure_mean", "Mean"),
    ]
    timing_cols = [
        ("duration_mean_ks_similarity", "DurMean KS"),
        ("all_durations_ks_similarity", "AllDur KS"),
        ("timing_mean", "Mean"),
    ]
    semantic_cols = [
        ("service_js_similarity", "Svc JS"),
        ("operation_js_similarity", "Op JS"),
        ("service_operation_js_similarity", "SvcOp JS"),
        ("service_operation_valid_mass", "Valid"),
        ("semantic_mean", "Mean"),
    ]

    systems = ["TT", "SN"]
    order = {"random_sampling": 0, "empirical": 1, "flat_vae": 2, "hierarchical_vae": 3}
    cls_order = {"normal": 0, "abnormal": 1, "combined": 2}

    lines = [
        "# Structural Fidelity Comparison",
        "",
        "All scores are similarities — higher is better.",
        "",
        "- **KS sim** = `1 − KS statistic` (0 = maximally different distributions, 1 = identical)",
        "- **MMD sim** = `exp(−MMD)` over 11 graph-level features (node/edge counts, density, degree stats, duration stats, service/op counts)",
        "- **JS sim** = `1 − Jensen-Shannon divergence` over discrete label distributions",
        "- **Valid** = fraction of synthetic (service, op) pairs that appear in the real data",
        "- **Structure Mean** = mean of 7 topology metrics (6 KS sims + MMD sim)",
        "- **Duration Mean** = mean of 2 duration KS sims (per-graph mean duration, all node durations)",
        "- **Semantic Mean** = mean of 4 label metrics (Svc JS, Op JS, SvcOp JS, Valid)",
        "- **Overall** = mean(Structure Mean, Duration Mean, Semantic Mean) — three equal groups",
        "",
    ]

    for system in systems:
        lines.append(f"## {system}")
        lines.append("")
        sys_rows = sorted(
            [r for r in rows if r["system"] == system],
            key=lambda r: (order.get(r["generator"], 9), cls_order.get(r["trace_class"], 9)),
        )

        lines.append("<table>")
        lines.append("<thead>")
        lines.append("<tr>")
        lines.append('  <th rowspan="2">Generator</th>')
        lines.append('  <th rowspan="2">Class</th>')
        lines.append(f'  <th colspan="{len(struct_cols)}">Structure</th>')
        lines.append(f'  <th colspan="{len(timing_cols)}">Duration</th>')
        lines.append(f'  <th colspan="{len(semantic_cols)}">Semantic</th>')
        lines.append('  <th rowspan="2">Overall</th>')
        lines.append("</tr>")
        lines.append("<tr>")
        for _, label in struct_cols:
            lines.append(f"  <th>{label}</th>")
        for _, label in timing_cols:
            lines.append(f"  <th>{label}</th>")
        for _, label in semantic_cols:
            lines.append(f"  <th>{label}</th>")
        lines.append("</tr>")
        lines.append("</thead>")
        lines.append("<tbody>")

        for row in sys_rows:
            lines.append("<tr>")
            lines.append(f'  <td>{row["generator"]}</td>')
            lines.append(f'  <td>{row["trace_class"]}</td>')
            for key, _ in struct_cols:
                val = fmt(row[key])
                tag = "<td><b>" + val + "</b></td>" if key == "structure_mean" else f"<td>{val}</td>"
                lines.append(f"  {tag}")
            for key, _ in timing_cols:
                val = fmt(row[key])
                tag = "<td><b>" + val + "</b></td>" if key == "timing_mean" else f"<td>{val}</td>"
                lines.append(f"  {tag}")
            for key, _ in semantic_cols:
                val = fmt(row[key])
                tag = "<td><b>" + val + "</b></td>" if key == "semantic_mean" else f"<td>{val}</td>"
                lines.append(f"  {tag}")
            lines.append(f'  <td><b>{fmt(row["overall_fidelity"])}</b></td>')
            lines.append("</tr>")

        lines.append("</tbody>")
        lines.append("</table>")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare structural fidelity of synthetic trace generators.")
    parser.add_argument("--systems", nargs="+", choices=["TT", "SN"], default=["TT", "SN"])
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=["random_sampling", "empirical", "flat_vae", "hierarchical_vae"],
        default=["random_sampling", "empirical", "flat_vae", "hierarchical_vae"],
    )
    parser.add_argument("--max-graphs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default="./classifier/structural_fidelity_results.csv")
    parser.add_argument("--out-md", default="./classifier/structural_fidelity_tables.md")
    args = parser.parse_args()

    rows = []
    for system in args.systems:
        cfg = DATASETS[system]
        min_nodes = cfg["min_nodes"]
        real_by_class = {
            cls: load_graphs(path, min_nodes=min_nodes, max_graphs=args.max_graphs, seed=args.seed)
            for cls, path in cfg["real"].items()
        }
        for generator in args.generators:
            synth_by_class = {
                cls: load_graphs(
                    path,
                    min_nodes=min_nodes,
                    max_graphs=args.max_graphs,
                    seed=args.seed,
                )
                for cls, path in cfg["synthetic"][generator].items()
            }
            for cls in ["normal", "abnormal"]:
                row = compare(real_by_class[cls], synth_by_class[cls])
                row.update({"system": system, "generator": generator, "trace_class": cls})
                rows.append(row)
                print(
                    f"{system} {generator} {cls}: overall={row['overall_fidelity']:.3f} "
                    f"structure={row['structure_mean']:.3f} timing={row['timing_mean']:.3f} semantic={row['semantic_mean']:.3f}"
                )
            combined_real = real_by_class["normal"] + real_by_class["abnormal"]
            combined_synth = synth_by_class["normal"] + synth_by_class["abnormal"]
            row = compare(combined_real, combined_synth)
            row.update({"system": system, "generator": generator, "trace_class": "combined"})
            rows.append(row)
            print(
                f"{system} {generator} combined: overall={row['overall_fidelity']:.3f} "
                f"structure={row['structure_mean']:.3f} timing={row['timing_mean']:.3f} semantic={row['semantic_mean']:.3f}"
            )

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["system", "generator", "trace_class"] + [
        k for k in rows[0].keys() if k not in {"system", "generator", "trace_class"}
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_md = ROOT / args.out_md
    write_markdown(rows, out_md)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
