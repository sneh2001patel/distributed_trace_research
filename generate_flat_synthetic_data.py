import argparse
import os
import random

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.utils import to_undirected

from flat_vae import FlatGNNDecoder, denormalize_duration


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


def load_decoder(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    decoder = FlatGNNDecoder(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_service_classes=cfg["num_services"],
        n_op_classes=cfg["num_ops"],
        max_nodes=cfg.get("max_nodes", 256),
        head_hidden_dim=cfg.get("head_hidden_dim", cfg["hidden_dim"] * 2),
        head_dropout=cfg.get("head_dropout", 0.3),
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state"])
    decoder.eval()
    return decoder, cfg


@torch.no_grad()
def sample_graph(
    decoder,
    cfg,
    n_nodes: int,
    y_label: int,
    device: torch.device,
    edge_threshold: float = 0.9,
    sample_nodes: bool = True,
    sample_edges: bool = True,
    node_temperature: float = 1.0,
):
    z = torch.randn(decoder.latent_dim, device=device)
    service_logits, op_logits, duration_pred, edge_logits = decoder.decode(z, n_nodes)

    if sample_nodes:
        temp = max(node_temperature, 1e-6)
        service_ids = torch.multinomial(
            torch.softmax(service_logits / temp, dim=1), 1
        ).squeeze(1)
        op_ids = torch.multinomial(torch.softmax(op_logits / temp, dim=1), 1).squeeze(1)
    else:
        service_ids = service_logits.argmax(dim=1)
        op_ids = op_logits.argmax(dim=1)

    durations = denormalize_duration(
        duration_pred,
        torch.tensor(cfg["duration_mean"], device=device),
        torch.tensor(cfg["duration_std"], device=device),
    )

    edge_probs = torch.sigmoid(edge_logits)
    mask = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1).bool()
    if sample_edges:
        edge_keep = (torch.rand_like(edge_probs) < edge_probs) & mask
    else:
        edge_keep = (edge_probs > edge_threshold) & mask
    edge_index = to_undirected(edge_keep.nonzero(as_tuple=False).t(), num_nodes=n_nodes)

    x = torch.stack([service_ids.float(), op_ids.float(), durations.float()], dim=1)
    return Data(
        x=x.cpu(),
        edge_index=edge_index.cpu(),
        y=torch.tensor([y_label], dtype=torch.long),
    )


def generate_dataset(args):
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder(args.weights_path, device)
    real_ds = LoadDataset(args.real_path)
    node_counts = [real_ds.get(i).num_nodes for i in range(len(real_ds))]
    target = args.target_count or len(node_counts)

    if target <= len(node_counts):
        sampled_counts = rng.sample(node_counts, target)
    else:
        sampled_counts = [rng.choice(node_counts) for _ in range(target)]

    graphs = [
        sample_graph(
            decoder,
            cfg,
            n_nodes=n,
            y_label=args.y_label,
            device=device,
            edge_threshold=args.edge_threshold,
            sample_nodes=args.sample_nodes,
            sample_edges=args.sample_edges,
            node_temperature=args.node_temperature,
        )
        for n in sampled_counts
    ]
    data, slices = InMemoryDataset.collate(graphs)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save((data, slices), args.out_path)
    print(f"saved {args.out_path} ({len(graphs)} graphs)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic traces from a one-latent VAE baseline."
    )
    parser.add_argument("--real-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--y-label", type=int, required=True)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--edge-threshold", type=float, default=0.9)
    parser.add_argument("--node-temperature", type=float, default=1.0)
    parser.add_argument("--sample-nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
