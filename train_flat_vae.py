import argparse
import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from flat_vae import FlatGNNDecoder, FlatGraphEncoderVAE, flat_vae_loss


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices
        x = data.x
        self.num_services = int(x[:, 0].max().item()) + 1
        self.num_ops = int(x[:, 1].max().item()) + 1
        log_duration = torch.log1p(torch.clamp(x[:, 2], min=0))
        self.duration_mean = log_duration.mean().item()
        self.duration_std = log_duration.std().item() or 1.0

    def get(self, idx):
        data = super().get(idx)
        data.edge_index = to_undirected(data.edge_index)
        return data


class GraphListDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def _filter_graphs_by_max_nodes(dataset: LoadDataset, max_nodes: int):
    kept = []
    dropped = 0
    for i in range(len(dataset)):
        graph = dataset.get(i)
        if graph.num_nodes <= max_nodes:
            kept.append(graph)
        else:
            dropped += 1
    return kept, dropped


def _sample_graphs(graphs, target_count: int, seed: int = 42):
    if target_count is None or len(graphs) <= target_count:
        return graphs
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(graphs), generator=generator)[:target_count].tolist()
    return [graphs[i] for i in indices]


def _duration_stats(graphs):
    durations = torch.cat([graph.x[:, 2] for graph in graphs], dim=0)
    log_duration = torch.log1p(torch.clamp(durations, min=0))
    return log_duration.mean().item(), (log_duration.std().item() or 1.0)


def train(args):
    base_dataset = LoadDataset(args.data_path)
    filtered_graphs, dropped_graphs = _filter_graphs_by_max_nodes(
        base_dataset, max_nodes=args.max_graph_nodes
    )
    if not filtered_graphs:
        raise RuntimeError(
            f"All graphs were filtered out by --max-graph-nodes {args.max_graph_nodes}."
        )
    sampled_graphs = _sample_graphs(
        filtered_graphs, target_count=args.target_graphs, seed=args.seed
    )
    duration_mean, duration_std = _duration_stats(sampled_graphs)
    dataset = GraphListDataset(sampled_graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Using {len(sampled_graphs)} sampled graphs with <= {args.max_graph_nodes} nodes "
        f"from {len(filtered_graphs)} filtered graphs (dropped {dropped_graphs} oversized graphs)."
    )

    encoder = FlatGraphEncoderVAE(
        n_services=base_dataset.num_services,
        n_ops=base_dataset.num_ops,
        duration_mean=duration_mean,
        duration_std=duration_std,
        hidden_ch=args.hidden_dim,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    decoder = FlatGNNDecoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_service_classes=base_dataset.num_services,
        n_op_classes=base_dataset.num_ops,
        max_nodes=args.max_nodes,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    ).to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    warmup = max(5, int(args.epochs * 0.2))

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = total_node = total_edge = total_kl = 0.0
        beta = args.beta_final * min(1.0, epoch / warmup)

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            mu, logvar, z = encoder(data.x, data.edge_index, data.batch)
            service_logits, op_logits, duration_pred, edge_logits = decoder.decode(
                z[0], data.num_nodes
            )
            loss, node_l, edge_l, kl_l = flat_vae_loss(
                data,
                service_logits,
                op_logits,
                duration_pred,
                edge_logits,
                mu,
                logvar,
                encoder.duration_mean,
                encoder.duration_std,
                beta=beta,
                edge_weight=args.edge_weight,
                edge_neg_weight=args.edge_neg_weight,
                op_loss_scale=args.op_loss_scale,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_node += node_l.item()
            total_edge += edge_l.item()
            total_kl += kl_l.item()

        n = len(loader)
        print(
            f"Epoch {epoch:03d} | loss {total_loss/n:.4f} | node {total_node/n:.4f} | edge {total_edge/n:.4f} | kl {total_kl/n:.4f} | beta {beta:.4f}"
        )

    os.makedirs(os.path.dirname(args.out_weights), exist_ok=True)
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "config": {
                "num_services": base_dataset.num_services,
                "num_ops": base_dataset.num_ops,
                "duration_mean": duration_mean,
                "duration_std": duration_std,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "embed_dim": args.embed_dim,
                "max_nodes": args.max_nodes,
                "head_hidden_dim": args.head_hidden_dim,
                "head_dropout": args.head_dropout,
            },
        },
        args.out_weights,
    )
    print(f"saved {args.out_weights}")


def main():
    parser = argparse.ArgumentParser(description="Train one-latent graph VAE baseline.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--target-graphs", type=int, default=1000)
    parser.add_argument("--max-graph-nodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--beta-final", type=float, default=0.02)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--edge-neg-weight", type=float, default=3.0)
    parser.add_argument("--op-loss-scale", type=float, default=1.6)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
