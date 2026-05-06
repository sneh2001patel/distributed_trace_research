import argparse
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.utils import to_undirected


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


@dataclass
class EmpiricalTraceModel:
    node_counts: List[int]
    service_values: torch.Tensor
    service_probs: torch.Tensor
    op_by_service: Dict[int, torch.Tensor]
    op_probs_by_service: Dict[int, torch.Tensor]
    op_values: torch.Tensor
    op_probs: torch.Tensor
    durations: torch.Tensor
    edge_patterns: List[torch.Tensor]

    @classmethod
    def fit(cls, graphs: Sequence[Data]) -> "EmpiricalTraceModel":
        node_counts = [g.num_nodes for g in graphs]
        service_counter = Counter()
        op_counter = Counter()
        service_to_ops = defaultdict(Counter)
        durations = []
        edge_patterns = []

        for graph in graphs:
            x = graph.x.cpu()
            services = x[:, 0].round().long().tolist()
            ops = x[:, 1].round().long().tolist()
            service_counter.update(services)
            op_counter.update(ops)
            for service, op in zip(services, ops):
                service_to_ops[service][op] += 1
            durations.append(x[:, 2].float())
            edge_patterns.append(graph.edge_index.cpu().clone())

        service_values, service_probs = _counter_to_distribution(service_counter)
        op_values, op_probs = _counter_to_distribution(op_counter)

        op_by_service = {}
        op_probs_by_service = {}
        for service, counter in service_to_ops.items():
            values, probs = _counter_to_distribution(counter)
            op_by_service[service] = values
            op_probs_by_service[service] = probs

        return cls(
            node_counts=node_counts,
            service_values=service_values,
            service_probs=service_probs,
            op_by_service=op_by_service,
            op_probs_by_service=op_probs_by_service,
            op_values=op_values,
            op_probs=op_probs,
            durations=torch.cat(durations),
            edge_patterns=edge_patterns,
        )

    def sample_graph(
        self,
        structure: str = "random_tree",
        y_label: int = 0,
        rng: random.Random = None,
    ) -> Data:
        rng = rng or random
        n_nodes = rng.choice(self.node_counts)
        services = self.service_values[
            torch.multinomial(self.service_probs, n_nodes, replacement=True)
        ]

        ops = []
        for service in services.tolist():
            if service in self.op_by_service:
                values = self.op_by_service[service]
                probs = self.op_probs_by_service[service]
            else:
                values = self.op_values
                probs = self.op_probs
            ops.append(values[torch.multinomial(probs, 1).item()])
        ops = torch.tensor(ops, dtype=torch.long)

        dur_idx = torch.randint(0, self.durations.numel(), (n_nodes,))
        durations = self.durations[dur_idx]
        edge_index = build_structure(n_nodes, structure, self.edge_patterns, rng)

        x = torch.stack([services.float(), ops.float(), durations.float()], dim=1)
        return Data(x=x, edge_index=edge_index, y=torch.tensor([y_label], dtype=torch.long))


def _counter_to_distribution(counter: Counter):
    values = torch.tensor(sorted(counter.keys()), dtype=torch.long)
    counts = torch.tensor([counter[int(v)] for v in values], dtype=torch.float)
    probs = counts / counts.sum()
    return values, probs


def build_structure(
    n_nodes: int,
    structure: str,
    edge_patterns: Sequence[torch.Tensor],
    rng: random.Random,
) -> torch.Tensor:
    if n_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    if structure == "chain":
        src = torch.arange(0, n_nodes - 1, dtype=torch.long)
        dst = torch.arange(1, n_nodes, dtype=torch.long)
        return to_undirected(torch.stack([src, dst], dim=0), num_nodes=n_nodes)

    if structure == "random_tree":
        src, dst = [], []
        for child in range(1, n_nodes):
            src.append(rng.randrange(0, child))
            dst.append(child)
        return to_undirected(torch.tensor([src, dst], dtype=torch.long), num_nodes=n_nodes)

    if structure == "shallow_dag":
        src, dst = [], []
        width = max(1, min(4, n_nodes - 1))
        for child in range(1, n_nodes):
            parent_max = min(child, width)
            src.append(rng.randrange(0, parent_max))
            dst.append(child)
            if child > width and rng.random() < 0.25:
                src.append(rng.randrange(0, child))
                dst.append(child)
        return to_undirected(torch.tensor([src, dst], dtype=torch.long), num_nodes=n_nodes)

    if structure == "reuse":
        same_size = []
        for edge_index in edge_patterns:
            if edge_index.numel() == 0 or int(edge_index.max().item()) < n_nodes:
                same_size.append(edge_index)
        candidates = same_size or list(edge_patterns)
        if not candidates:
            return build_structure(n_nodes, "random_tree", edge_patterns, rng)
        edge_index = rng.choice(candidates).clone()
        if edge_index.numel() == 0:
            return edge_index
        mask = (edge_index[0] < n_nodes) & (edge_index[1] < n_nodes)
        return edge_index[:, mask]

    raise ValueError(f"Unknown structure '{structure}'")


def load_graphs(path: str) -> List[Data]:
    ds = LoadDataset(path)
    return [ds.get(i).cpu() for i in range(len(ds))]


def save_graphs(graphs: Sequence[Data], out_path: str):
    data, slices = InMemoryDataset.collate(list(graphs))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(graphs)} graphs)")


def build_empirical_dataset(
    real_path: str,
    out_path: str,
    y_label: int,
    target_count: int = None,
    structure: str = "random_tree",
    seed: int = 42,
):
    rng = random.Random(seed)
    real_graphs = load_graphs(real_path)
    model = EmpiricalTraceModel.fit(real_graphs)
    n_graphs = target_count or len(real_graphs)
    synthetic = [
        model.sample_graph(structure=structure, y_label=y_label, rng=rng)
        for _ in range(n_graphs)
    ]
    save_graphs(synthetic, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate non-learning empirical synthetic trace baselines."
    )
    parser.add_argument("--real-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--y-label", type=int, required=True)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument(
        "--structure",
        choices=["random_tree", "chain", "shallow_dag", "reuse"],
        default="random_tree",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_empirical_dataset(
        real_path=args.real_path,
        out_path=args.out_path,
        y_label=args.y_label,
        target_count=args.target_count,
        structure=args.structure,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
