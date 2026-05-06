import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr

from vae import _log_denormalize_duration
from vae_tt import TTGNNDecoder


ROOT = Path(__file__).resolve().parent


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


def load_graphs(datapath: str):
    ds = LoadDataset(datapath)
    return [ds.get(i) for i in range(len(ds))]


def load_decoder_from_checkpoint(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    cfg = checkpoint["config"]
    state = checkpoint["decoder_state"]

    max_nodes = int(state["pos_emb.weight"].shape[0])

    head_hidden_dim = cfg.get("head_hidden_dim")
    if head_hidden_dim is None:
        head_hidden_dim = int(state["service_head.0.weight"].shape[0])

    dec = TTGNNDecoder(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_service_classes=cfg["num_services"],
        n_op_classes=cfg["num_ops"],
        encoder_hidden_dim=cfg.get("encoder_hidden_dim", cfg["hidden_dim"]),
        max_nodes=max_nodes,
        head_hidden_dim=head_hidden_dim,
        head_dropout=cfg.get("head_dropout", 0.05),
    ).to(device)
    dec.load_state_dict(state)
    dec.eval()
    return dec, cfg


def filter_graphs_by_max_nodes(graphs, max_nodes: int):
    return [g for g in graphs if g.num_nodes <= max_nodes]


def sample_node_counts(graphs, target_count: int = None, seed: int = 42):
    counts = [g.num_nodes for g in graphs]
    target = target_count or len(counts)
    rng = random.Random(seed)
    if target <= len(counts):
        return rng.sample(counts, target)
    return [rng.choice(counts) for _ in range(target)]


def save_graphs(graphs, out_path: str):
    data, slices = InMemoryDataset.collate(graphs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(graphs)} graphs)")


@torch.no_grad()
def generate_tt_synthetic_graph(
    decoder: TTGNNDecoder,
    num_nodes: int,
    duration_mean: float,
    duration_std: float,
    device: torch.device,
    sample_nodes: bool = True,
    node_temperature: float = 1.1,
    parent_temperature: float = 1.0,
    duration_noise: float = 0.02,
) -> Data:
    z_g = torch.randn(decoder.latent_dim, device=device)
    z_n = torch.randn(num_nodes, decoder.latent_dim, device=device)
    enc_h = torch.zeros(num_nodes, decoder.encoder_hidden_dim, device=device)

    service_logits, op_logits, duration_pred, parent_logits = decoder.decode_for_training(
        z_g, z_n, enc_h
    )

    if sample_nodes:
        temp = max(float(node_temperature), 1e-6)
        service_probs = torch.softmax(service_logits / temp, dim=1)
        op_probs = torch.softmax(op_logits / temp, dim=1)
        service_ids = torch.multinomial(service_probs, 1).squeeze(1)
        op_ids = torch.multinomial(op_probs, 1).squeeze(1)
    else:
        service_ids = service_logits.argmax(dim=1)
        op_ids = op_logits.argmax(dim=1)

    duration = _log_denormalize_duration(
        duration_pred.squeeze(-1),
        torch.tensor(duration_mean, device=device),
        torch.tensor(duration_std, device=device),
    )
    if duration_noise > 0.0:
        duration = torch.clamp(
            duration + torch.randn_like(duration) * duration_noise, min=0.0
        )

    parent_temp = max(float(parent_temperature), 1e-6)
    parent_probs = torch.softmax(parent_logits / parent_temp, dim=1)
    if sample_nodes:
        parent_ids = torch.multinomial(parent_probs, 1).squeeze(1)
    else:
        parent_ids = parent_logits.argmax(dim=1)

    edges = []
    for child, parent in enumerate(parent_ids.tolist()):
        if parent < num_nodes:
            edges.append([parent, child])
    if edges:
        edge_index = torch.tensor(edges, device=device, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
    x = torch.stack([service_ids, op_ids, duration], dim=1).float()
    return Data(x=x, edge_index=edge_index)


def build_dataset(
    *,
    real_path: str,
    weights_path: str,
    out_path: str,
    y_label: int,
    target_count: int = None,
    seed: int = 42,
    trained_node_cap: int = 30,
    sample_nodes: bool = True,
    node_temperature: float = 1.1,
    parent_temperature: float = 1.0,
    duration_noise: float = 0.02,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder_from_checkpoint(weights_path, device)

    real_graphs = load_graphs(real_path)
    filtered_real_graphs = filter_graphs_by_max_nodes(real_graphs, trained_node_cap)
    if not filtered_real_graphs:
        raise RuntimeError(
            f"No TT real graphs <= {trained_node_cap} nodes found in {real_path}."
        )

    node_counts = sample_node_counts(
        filtered_real_graphs, target_count=target_count, seed=seed
    )

    syn_graphs = []
    for n_nodes in node_counts:
        graph = generate_tt_synthetic_graph(
            decoder,
            num_nodes=n_nodes,
            duration_mean=cfg["duration_mean"],
            duration_std=cfg["duration_std"],
            device=device,
            sample_nodes=sample_nodes,
            node_temperature=node_temperature,
            parent_temperature=parent_temperature,
            duration_noise=duration_noise,
        )
        graph.y = torch.tensor(y_label, dtype=torch.long)
        syn_graphs.append(graph.cpu())

    save_graphs(syn_graphs, out_path)
    return {
        "real_path": real_path,
        "weights_path": weights_path,
        "out_path": out_path,
        "graphs": len(syn_graphs),
        "real_graphs_total": len(real_graphs),
        "real_graphs_used_for_sizes": len(filtered_real_graphs),
        "trained_node_cap": trained_node_cap,
        "node_count_min": min(node_counts) if node_counts else 0,
        "node_count_max": max(node_counts) if node_counts else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate TT normal and abnormal synthetic datasets from trained TT VAEs."
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "abnormal", "both"],
        default="both",
        help="Which TT synthetic dataset(s) to generate.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Synthetic graph count per dataset. Defaults to size of the filtered real dataset.",
    )
    parser.add_argument(
        "--normal-count",
        type=int,
        default=None,
        help="Generate this many normal synthetic graphs and scale abnormal graphs proportionally to the real TT class ratio.",
    )
    parser.add_argument(
        "--trained-node-cap",
        type=int,
        default=30,
        help="Only sample node counts from real TT graphs at or below the node cap used during TT VAE training.",
    )
    parser.add_argument("--parent-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jobs = []
    if args.mode in {"normal", "both"}:
        jobs.append(
            {
                "name": "normal",
                "real_path": "./datasets/anomaly/TT/TT_normal.pt",
                "weights_path": "./weights/tt_normal_vae_weights.pt",
                "out_path": "./datasets/anomaly/TT/TT_normal_synthetic.pt",
                "y_label": 0,
            }
        )
    if args.mode in {"abnormal", "both"}:
        jobs.append(
            {
                "name": "abnormal",
                "real_path": "./datasets/anomaly/TT/TT_abnormal.pt",
                "weights_path": "./weights/tt_abnormal_vae_weights.pt",
                "out_path": "./datasets/anomaly/TT/TT_abnormal_synthetic.pt",
                "y_label": 1,
            }
        )

    proportional_counts = {}
    if args.normal_count is not None:
        normal_graphs = len(load_graphs("./datasets/anomaly/TT/TT_normal.pt"))
        abnormal_graphs = len(load_graphs("./datasets/anomaly/TT/TT_abnormal.pt"))
        abnormal_ratio = abnormal_graphs / normal_graphs if normal_graphs else 0.0
        proportional_counts["normal"] = args.normal_count
        proportional_counts["abnormal"] = max(
            1, round(args.normal_count * abnormal_ratio)
        )

    summary = {}
    for job in jobs:
        print(f"\n=== BUILDING TT {job['name'].upper()} SYNTHETIC DATASET ===")
        summary[job["name"]] = build_dataset(
            real_path=job["real_path"],
            weights_path=job["weights_path"],
            out_path=job["out_path"],
            y_label=job["y_label"],
            target_count=proportional_counts.get(job["name"], args.target_count),
            seed=args.seed,
            trained_node_cap=args.trained_node_cap,
            parent_temperature=args.parent_temperature,
        )

    summary_path = ROOT / "datasets" / "anomaly" / "TT" / "synthetic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
