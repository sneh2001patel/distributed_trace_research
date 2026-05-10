import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr

from synthetic_graphs import generate_synthetic_graph
from vae import GNNDecoder


ROOT = Path(__file__).resolve().parent


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


def load_decoder_from_checkpoint(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    cfg = checkpoint["config"]
    state = checkpoint["decoder_state"]

    max_nodes = cfg.get("max_nodes")
    if max_nodes is None:
        max_nodes = int(state["pos_emb.weight"].shape[0])

    head_hidden_dim = cfg.get("head_hidden_dim")
    if head_hidden_dim is None:
        head_hidden_dim = int(state["service_head.0.weight"].shape[0])

    dec = GNNDecoder(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_service_classes=cfg["num_services"],
        n_op_classes=cfg["num_ops"],
        encoder_hidden_dim=cfg.get("encoder_hidden_dim", cfg["hidden_dim"]),
        max_nodes=max_nodes,
        head_hidden_dim=head_hidden_dim,
        head_dropout=cfg.get("head_dropout", 0.1),
        enc_h_dropout=cfg.get("enc_h_dropout", 0.0),
    ).to(device)
    dec.load_state_dict(state)
    dec.eval()
    return dec, cfg


def load_graphs(datapath: str):
    ds = LoadDataset(datapath)
    return [ds.get(i) for i in range(len(ds))]


def sample_node_counts(graphs, target_count: int = None, seed: int = 42):
    counts = [g.num_nodes for g in graphs]
    target = target_count or len(counts)
    rng = random.Random(seed)
    if target <= len(counts):
        return rng.sample(counts, target)
    return [rng.choice(counts) for _ in range(target)]


def sample_edge_counts(graphs, target_count: int = None, seed: int = 42):
    counts = [int(g.edge_index.size(1)) for g in graphs]
    target = target_count or len(counts)
    rng = random.Random(seed + 17)
    if target <= len(counts):
        return rng.sample(counts, target)
    return [rng.choice(counts) for _ in range(target)]


def build_valid_ops_by_service(graphs):
    valid = {}
    for graph in graphs:
        x = graph.x.detach().cpu()
        for service_id, op_id in zip(x[:, 0].round().long(), x[:, 1].round().long()):
            valid.setdefault(int(service_id.item()), set()).add(int(op_id.item()))
    return {
        service_id: torch.tensor(sorted(op_ids), dtype=torch.long)
        for service_id, op_ids in valid.items()
    }


def save_graphs(graphs, out_path: str):
    data, slices = InMemoryDataset.collate(graphs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(graphs)} graphs)")


def build_dataset(
    *,
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
    decoder, cfg = load_decoder_from_checkpoint(weights_path, device)

    real_graphs = load_graphs(real_path)
    node_counts = sample_node_counts(real_graphs, target_count=target_count, seed=seed)
    edge_counts = sample_edge_counts(real_graphs, target_count=len(node_counts), seed=seed)
    valid_ops_by_service = build_valid_ops_by_service(real_graphs)

    syn_graphs = []
    for n_nodes, target_edges in zip(node_counts, edge_counts):
        graph = generate_synthetic_graph(
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
            target_edge_count=target_edges,
            valid_ops_by_service=valid_ops_by_service,
        )
        graph.y = torch.tensor(y_label, dtype=torch.long)
        syn_graphs.append(graph.cpu())

    save_graphs(syn_graphs, out_path)
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
        description="Generate SN normal and abnormal synthetic datasets from trained VAEs."
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
        help="Synthetic graph count per dataset. Defaults to size of the real dataset.",
    )
    parser.add_argument(
        "--normal-count",
        type=int,
        default=None,
        help="Generate this many normal synthetic graphs and scale abnormal graphs proportionally to the real SN class ratio.",
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

    proportional_counts = {}
    if args.normal_count is not None:
        normal_graphs = len(load_graphs("./datasets/anomaly/SN/SN_normal.pt"))
        abnormal_graphs = len(load_graphs("./datasets/anomaly/SN/SN_abnormal.pt"))
        abnormal_ratio = abnormal_graphs / normal_graphs if normal_graphs else 0.0
        proportional_counts["normal"] = args.normal_count
        proportional_counts["abnormal"] = max(
            1, round(args.normal_count * abnormal_ratio)
        )

    summary = {}
    for job in jobs:
        print(f"\n=== BUILDING SN {job['name'].upper()} SYNTHETIC DATASET ===")
        summary[job["name"]] = build_dataset(
            real_path=job["real_path"],
            weights_path=job["weights_path"],
            out_path=job["out_path"],
            y_label=job["y_label"],
            target_count=proportional_counts.get(job["name"], args.target_count),
            seed=args.seed,
        )

    summary_path = ROOT / "datasets" / "anomaly" / "SN" / "synthetic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
