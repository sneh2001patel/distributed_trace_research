import argparse

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data, DataEdgeAttr
from torch_geometric.utils import to_undirected

from vae import GNNDecoder, GraphEncoderVAE, evaluate_reconstruction, run_training


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath="./datasets/anomaly/SN/SN_normal.pt") -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
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
        g = dataset.get(i)
        if g.num_nodes <= max_nodes:
            kept.append(g)
        else:
            dropped += 1
    return kept, dropped


def _duration_stats(graphs):
    durations = torch.cat([g.x[:, 2] for g in graphs], dim=0)
    log_duration = torch.log1p(torch.clamp(durations, min=0))
    return log_duration.mean().item(), (log_duration.std().item() or 1.0)


def _sample_graphs(graphs, target_count: int, seed: int = 42):
    if len(graphs) <= target_count:
        return graphs
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(graphs), generator=generator)[:target_count].tolist()
    return [graphs[i] for i in indices]


def main(
    epochs: int = 300,
    beta_final: float = 0.05,
    enc_h_dropout: float = 1.0,
):
    print("\n=== TRAINING SN NORMAL DATA ===")

    datapath = "./datasets/anomaly/SN/SN_normal.pt"
    base_dataset = LoadDataset(datapath)
    max_nodes = 30
    target_graphs = 1000
    filtered_graphs, dropped_graphs = _filter_graphs_by_max_nodes(
        base_dataset, max_nodes=max_nodes
    )
    if not filtered_graphs:
        raise RuntimeError("All SN normal graphs were filtered out.")

    sampled_graphs = _sample_graphs(filtered_graphs, target_graphs, seed=42)

    dataset = GraphListDataset(sampled_graphs)
    duration_mean, duration_std = _duration_stats(sampled_graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Using {len(sampled_graphs)} sampled SN normal graphs with <= {max_nodes} nodes "
        f"from {len(filtered_graphs)} filtered graphs (dropped {dropped_graphs} oversized graphs)."
    )

    enc = GraphEncoderVAE(
        n_services=base_dataset.num_services,
        n_ops=base_dataset.num_ops,
        duration_mean=duration_mean,
        duration_std=duration_std,
        hidden_ch=128,
        embed_dim=48,
        latent_dim=64,
    ).to(device)

    dec = GNNDecoder(
        latent_dim=64,
        hidden_dim=128,
        n_service_classes=base_dataset.num_services,
        n_op_classes=base_dataset.num_ops,
        encoder_hidden_dim=128,
        max_nodes=64,
        head_hidden_dim=256,
        head_dropout=0.1,
        enc_h_dropout=enc_h_dropout,
    ).to(device)

    run_training(
        dataset,
        enc,
        dec,
        device,
        epochs=epochs,
        batch_size=2,
        lr=3e-4,
        beta_final=beta_final,
        use_class_weights=True,
        op_loss_scale=2.0,
        edge_weight=2.0,
        edge_neg_weight=1.0,
        class_weight_max_ratio=10.0,
    )

    torch.save(
        {
            "encoder_state": enc.state_dict(),
            "decoder_state": dec.state_dict(),
            "config": {
                "num_services": base_dataset.num_services,
                "num_ops": base_dataset.num_ops,
                "duration_mean": duration_mean,
                "duration_std": duration_std,
                "latent_dim": 64,
                "hidden_dim": 128,
                "embed_dim": 48,
                "encoder_hidden_dim": 128,
                "max_nodes": 64,
                "head_hidden_dim": 256,
                "head_dropout": 0.1,
                "enc_h_dropout": enc_h_dropout,
            },
        },
        "./weights/sn_normal_vae_weights.pt",
    )

    metrics = evaluate_reconstruction(dataset, enc, dec, device, edge_threshold=0.6)
    print(metrics)
    print("\n=== Reconstruction Evaluation ===")
    print(f"Total nodes evaluated: {metrics['total_nodes']}")
    print(f"Service accuracy: {metrics['service_acc']*100:.2f}%")
    print(f"Op accuracy:  {metrics['op_acc']*100:.2f}%")
    print(f"Duration MAE: {metrics['dur_mae']:.4f}")
    print(f"Edge precision: {metrics['edge_precision']*100:.2f}%")
    print(f"Edge recall:    {metrics['edge_recall']*100:.2f}%")
    print(f"Edge F1:        {metrics['edge_f1']*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SN normal hierarchical VAE.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--beta-final", type=float, default=0.05)
    parser.add_argument("--enc-h-dropout", type=float, default=1.0)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        beta_final=args.beta_final,
        enc_h_dropout=args.enc_h_dropout,
    )
