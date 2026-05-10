import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


ROOT = Path(__file__).resolve().parent


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


@dataclass
class RandomSamplingTraceModel:
    node_count_min: int
    node_count_max: int
    edge_count_min: int
    edge_count_max: int
    feature_min: torch.Tensor
    feature_max: torch.Tensor
    integer_feature_indices: tuple[int, ...] = (0, 1)

    @classmethod
    def fit(
        cls,
        graphs: Sequence[Data],
        integer_feature_indices: Sequence[int] = (0, 1),
    ) -> "RandomSamplingTraceModel":
        if not graphs:
            raise ValueError("Cannot fit random sampling model on an empty dataset.")

        xs = torch.cat([graph.x.detach().cpu().float() for graph in graphs], dim=0)
        node_counts = [int(graph.num_nodes) for graph in graphs]
        edge_counts = [int(graph.edge_index.size(1)) for graph in graphs]

        return cls(
            node_count_min=min(node_counts),
            node_count_max=max(node_counts),
            edge_count_min=min(edge_counts),
            edge_count_max=max(edge_counts),
            feature_min=xs.min(dim=0).values,
            feature_max=xs.max(dim=0).values,
            integer_feature_indices=tuple(int(i) for i in integer_feature_indices),
        )

    def sample_graph(self, y_label: int, rng: random.Random) -> Data:
        n_nodes = rng.randint(self.node_count_min, self.node_count_max)
        x = self._sample_features(n_nodes)
        edge_index = self._sample_edges(n_nodes, rng)
        return Data(x=x, edge_index=edge_index, y=torch.tensor([y_label], dtype=torch.long))

    def _sample_features(self, n_nodes: int) -> torch.Tensor:
        lower = self.feature_min.float()
        upper = self.feature_max.float()
        span = torch.clamp(upper - lower, min=0.0)
        x = lower + torch.rand((n_nodes, lower.numel())) * span

        for idx in self.integer_feature_indices:
            if idx < 0 or idx >= lower.numel():
                continue
            lo = int(torch.floor(lower[idx]).item())
            hi = int(torch.ceil(upper[idx]).item())
            if hi <= lo:
                x[:, idx] = lo
            else:
                x[:, idx] = torch.randint(lo, hi + 1, (n_nodes,), dtype=torch.float)
        return x

    def _sample_edges(self, n_nodes: int, rng: random.Random) -> torch.Tensor:
        if n_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        max_possible_edges = n_nodes * (n_nodes - 1)
        edge_count = rng.randint(self.edge_count_min, self.edge_count_max)
        edge_count = min(max(edge_count, 0), max_possible_edges)
        if edge_count == 0:
            return torch.empty((2, 0), dtype=torch.long)

        sampled_edges = set()
        while len(sampled_edges) < edge_count:
            src = rng.randrange(n_nodes)
            dst = rng.randrange(n_nodes - 1)
            if dst >= src:
                dst += 1
            sampled_edges.add((src, dst))
        return torch.tensor(sorted(sampled_edges), dtype=torch.long).t().contiguous()

    def summary(self) -> dict:
        return {
            "node_count_min": self.node_count_min,
            "node_count_max": self.node_count_max,
            "edge_count_min": self.edge_count_min,
            "edge_count_max": self.edge_count_max,
            "feature_min": [float(v) for v in self.feature_min.tolist()],
            "feature_max": [float(v) for v in self.feature_max.tolist()],
            "integer_feature_indices": list(self.integer_feature_indices),
        }


def load_graphs(path: str) -> List[Data]:
    ds = LoadDataset(path)
    return [ds.get(i).cpu() for i in range(len(ds))]


def save_graphs(graphs: Sequence[Data], out_path: str):
    data, slices = InMemoryDataset.collate(list(graphs))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(graphs)} graphs)", flush=True)


def build_random_sampling_dataset(
    real_path: str,
    out_path: str,
    y_label: int,
    target_count: int = None,
    seed: int = 42,
    integer_feature_indices: Sequence[int] = (0, 1),
) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    real_graphs = load_graphs(real_path)
    model = RandomSamplingTraceModel.fit(
        real_graphs,
        integer_feature_indices=integer_feature_indices,
    )
    n_graphs = target_count or len(real_graphs)
    synthetic = [model.sample_graph(y_label=y_label, rng=rng) for _ in range(n_graphs)]
    save_graphs(synthetic, out_path)

    return {
        "real_path": real_path,
        "out_path": out_path,
        "graphs": len(synthetic),
        **model.summary(),
    }


def _system_jobs(system: str):
    return {
        "SN": [
            {
                "name": "normal",
                "real_path": "./datasets/anomaly/SN/SN_normal.pt",
                "out_path": "./datasets/baselines/SN_normal_random_sampling.pt",
                "y_label": 0,
            },
            {
                "name": "abnormal",
                "real_path": "./datasets/anomaly/SN/SN_abnormal.pt",
                "out_path": "./datasets/baselines/SN_abnormal_random_sampling.pt",
                "y_label": 1,
            },
        ],
        "TT": [
            {
                "name": "normal",
                "real_path": "./datasets/anomaly/TT/TT_normal.pt",
                "out_path": "./datasets/baselines/TT_normal_random_sampling.pt",
                "y_label": 0,
            },
            {
                "name": "abnormal",
                "real_path": "./datasets/anomaly/TT/TT_abnormal.pt",
                "out_path": "./datasets/baselines/TT_abnormal_random_sampling.pt",
                "y_label": 1,
            },
        ],
    }[system]


def main():
    parser = argparse.ArgumentParser(
        description="Generate random-sampling synthetic traces from real-data feature bounds."
    )
    parser.add_argument("--real-path", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--y-label", type=int, default=None)
    parser.add_argument(
        "--system",
        choices=["SN", "TT", "both"],
        default=None,
        help="Convenience mode for building SN and/or TT normal/abnormal baselines.",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "abnormal", "both"],
        default="both",
        help="Class split to generate when using --system.",
    )
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--integer-feature-indices",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Feature columns sampled as integer IDs. Defaults to service_id and operation_id.",
    )
    args = parser.parse_args()

    if args.system is None:
        if args.real_path is None or args.out_path is None or args.y_label is None:
            parser.error(
                "Either pass --system, or pass --real-path, --out-path, and --y-label."
            )
        summary = build_random_sampling_dataset(
            real_path=args.real_path,
            out_path=args.out_path,
            y_label=args.y_label,
            target_count=args.target_count,
            seed=args.seed,
            integer_feature_indices=args.integer_feature_indices,
        )
        print(json.dumps(summary, indent=2))
        return

    systems = ["SN", "TT"] if args.system == "both" else [args.system]
    summary = {}
    for system in systems:
        summary[system] = {}
        for job in _system_jobs(system):
            if args.mode != "both" and job["name"] != args.mode:
                continue
            print(
                f"\n=== BUILDING {system} {job['name'].upper()} RANDOM SAMPLING DATASET ===",
                flush=True,
            )
            summary[system][job["name"]] = build_random_sampling_dataset(
                real_path=job["real_path"],
                out_path=job["out_path"],
                y_label=job["y_label"],
                target_count=args.target_count,
                seed=args.seed,
                integer_feature_indices=args.integer_feature_indices,
            )

    summary_path = ROOT / "datasets" / "baselines" / "random_sampling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
