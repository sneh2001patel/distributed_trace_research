import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr

from synthetic_graphs import generate_synthetic_graph, load_decoder


ROOT = Path(__file__).resolve().parent


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


def _load_graphs(datapath: str):
    ds = LoadDataset(datapath)
    return [ds.get(i) for i in range(len(ds))]


def _sample_node_counts(graphs, target_count: int = None, seed: int = 42):
    counts = [g.num_nodes for g in graphs]
    target = target_count or len(counts)
    rng = random.Random(seed)
    if target <= len(counts):
        return rng.sample(counts, target)
    return [rng.choice(counts) for _ in range(target)]


def _save_graphs(graphs, out_path: str):
    data, slices = InMemoryDataset.collate(graphs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(graphs)} graphs)")


def build_sn_synthetic_dataset(
    real_path: str,
    weights_path: str,
    out_path: str,
    y_label: int,
    target_count: int = None,
    seed: int = 42,
    edge_threshold: float = 0.6,
    sample_edges: bool = True,
    sample_nodes: bool = True,
    node_temperature: float = 1.25,
    edge_dropout: float = 0.05,
    duration_noise: float = 0.02,
    threshold_jitter: float = 0.02,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder(weights_path, device)

    real_graphs = _load_graphs(real_path)
    node_counts = _sample_node_counts(real_graphs, target_count=target_count, seed=seed)

    syn_graphs = []
    for n_nodes in node_counts:
        g = generate_synthetic_graph(
            decoder,
            num_nodes=n_nodes,
            duration_mean=cfg["duration_mean"],
            duration_std=cfg["duration_std"],
            device=device,
            edge_threshold=edge_threshold,
            sample_edges=sample_edges,
            sample_nodes=sample_nodes,
            node_temperature=node_temperature,
            edge_dropout=edge_dropout,
            duration_noise=duration_noise,
            threshold_jitter=threshold_jitter,
        )
        g.y = torch.tensor(y_label, dtype=torch.long)
        syn_graphs.append(g.cpu())

    _save_graphs(syn_graphs, out_path)
    return {
        "real_path": real_path,
        "weights_path": weights_path,
        "out_path": out_path,
        "graphs": len(syn_graphs),
        "node_count_min": min(node_counts) if node_counts else 0,
        "node_count_max": max(node_counts) if node_counts else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build SN normal/abnormal synthetic datasets from trained VAEs."
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "abnormal", "both"],
        default="both",
        help="Which SN synthetic dataset(s) to generate.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Number of synthetic graphs to generate per dataset. Defaults to the size of the real dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jobs = []
    if args.mode in {"normal", "both"}:
        jobs.append(
            {
                "name": "normal",
                "real_path": "./datasets/anomaly/SN/SN_normal.pt",
                "weights_path": "./weights/sn_normal_vae_weights.pt",
                "out_path": "./datasets/anomaly/SN/SN_normal_synthetic.pt",
                "y_label": 0,
            }
        )
    if args.mode in {"abnormal", "both"}:
        jobs.append(
            {
                "name": "abnormal",
                "real_path": "./datasets/anomaly/SN/SN_abnormal.pt",
                "weights_path": "./weights/sn_abnormal_vae_weights.pt",
                "out_path": "./datasets/anomaly/SN/SN_abnormal_synthetic.pt",
                "y_label": 1,
            }
        )

    summary = {}
    for job in jobs:
        print(f"\n=== BUILDING SN {job['name'].upper()} SYNTHETIC DATASET ===")
        summary[job["name"]] = build_sn_synthetic_dataset(
            real_path=job["real_path"],
            weights_path=job["weights_path"],
            out_path=job["out_path"],
            y_label=job["y_label"],
            target_count=args.target_count,
            seed=args.seed,
        )

    summary_path = ROOT / "datasets" / "anomaly" / "SN" / "synthetic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
