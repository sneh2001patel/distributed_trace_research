import argparse

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data, DataEdgeAttr

from vae_tt import TTGNNDecoder, GraphEncoderVAE, evaluate_reconstruction, run_training


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath="./datasets/anomaly/TT/TT_normal.pt") -> None:
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
        return super().get(idx)


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


def _split_train_val(graphs, val_ratio: float = 0.0, seed: int = 42):
    if len(graphs) < 2:
        return graphs, []
    if val_ratio <= 0.0:
        return graphs, []
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(graphs), generator=generator).tolist()
    val_count = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:val_count])
    train_graphs = [g for i, g in enumerate(graphs) if i not in val_idx]
    val_graphs = [g for i, g in enumerate(graphs) if i in val_idx]
    return train_graphs, val_graphs


def main(
    max_nodes: int = 30,
    target_graphs: int = 1000,
    val_ratio: float = 0.0,
    epochs: int = 300,
    batch_size: int = 4,
    lr: float = 3e-4,
    beta_final: float = 0.03,
    edge_weight: float = 2.0,
    edge_rank_weight: float = 0.20,
    enc_h_dropout: float = 1.0,
    patience: int = 20,
):
    print("\n=== TRAINING TT NORMAL DATA ===")

    datapath = "./datasets/anomaly/TT/TT_normal.pt"
    base_dataset = LoadDataset(datapath)
    filtered_graphs, dropped_graphs = _filter_graphs_by_max_nodes(
        base_dataset, max_nodes=max_nodes
    )
    if not filtered_graphs:
        raise RuntimeError("All TT normal graphs were filtered out.")

    sampled_graphs = _sample_graphs(filtered_graphs, target_graphs, seed=42)

    train_graphs, val_graphs = _split_train_val(sampled_graphs, val_ratio=val_ratio, seed=42)
    dataset = GraphListDataset(train_graphs)
    val_dataset = GraphListDataset(val_graphs)
    full_dataset = GraphListDataset(sampled_graphs)
    duration_mean, duration_std = _duration_stats(train_graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Using {len(sampled_graphs)} sampled TT normal graphs with <= {max_nodes} nodes "
        f"from {len(filtered_graphs)} filtered graphs (dropped {dropped_graphs} oversized graphs)."
    )
    if len(val_graphs) > 0:
        print(f"Train/val split: {len(train_graphs)} train / {len(val_graphs)} val")
    else:
        print("Train/val split: using full sampled dataset for model selection")

    enc = GraphEncoderVAE(
        n_services=base_dataset.num_services,
        n_ops=base_dataset.num_ops,
        duration_mean=duration_mean,
        duration_std=duration_std,
        hidden_ch=96,
        embed_dim=32,
        latent_dim=48,
        dropout=0.1,
    ).to(device)

    dec = TTGNNDecoder(
        latent_dim=48,
        hidden_dim=96,
        n_service_classes=base_dataset.num_services,
        n_op_classes=base_dataset.num_ops,
        encoder_hidden_dim=96,
        max_nodes=64,
        head_hidden_dim=128,
        head_dropout=0.05,
        enc_h_dropout=enc_h_dropout,
    ).to(device)

    train_summary = run_training(
        dataset,
        enc,
        dec,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta_final=beta_final,
        use_class_weights=True,
        op_loss_scale=1.5,
        edge_weight=edge_weight,
        edge_rank_weight=edge_rank_weight,
        class_weight_max_ratio=10.0,
        val_dataset=val_dataset if len(val_graphs) > 0 else full_dataset,
        early_stopping_patience=patience,
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
                "latent_dim": 48,
                "hidden_dim": 96,
                "embed_dim": 32,
                "encoder_hidden_dim": 96,
                "max_nodes": 64,
                "head_hidden_dim": 128,
                "head_dropout": 0.05,
                "enc_h_dropout": enc_h_dropout,
            },
        },
        "./weights/tt_normal_vae_weights.pt",
    )

    val_metrics = (
        evaluate_reconstruction(val_dataset, enc, dec, device)
        if len(val_graphs) > 0
        else None
    )
    full_metrics = evaluate_reconstruction(full_dataset, enc, dec, device)
    print(train_summary)
    print("\nSelected edge decode mode: parent")
    if val_metrics is not None:
        print("\n=== Validation Reconstruction Evaluation ===")
        print(f"Total nodes evaluated: {val_metrics['total_nodes']}")
        print(f"Service accuracy: {val_metrics['service_acc']*100:.2f}%")
        print(f"Op accuracy:  {val_metrics['op_acc']*100:.2f}%")
        print(f"Duration MAE: {val_metrics['dur_mae']:.4f}")
        print(f"Edge precision: {val_metrics['edge_precision']*100:.2f}%")
        print(f"Edge recall:    {val_metrics['edge_recall']*100:.2f}%")
        print(f"Edge F1:        {val_metrics['edge_f1']*100:.2f}%")
    print("\n=== Reconstruction Evaluation ===")
    print(f"Total nodes evaluated: {full_metrics['total_nodes']}")
    print(f"Service accuracy: {full_metrics['service_acc']*100:.2f}%")
    print(f"Op accuracy:  {full_metrics['op_acc']*100:.2f}%")
    print(f"Duration MAE: {full_metrics['dur_mae']:.4f}")
    print(f"Edge precision: {full_metrics['edge_precision']*100:.2f}%")
    print(f"Edge recall:    {full_metrics['edge_recall']*100:.2f}%")
    print(f"Edge F1:        {full_metrics['edge_f1']*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TT normal parent-decoder VAE."
    )
    parser.add_argument("--max-nodes", type=int, default=30)
    parser.add_argument("--target-graphs", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta-final", type=float, default=0.03)
    parser.add_argument("--edge-weight", type=float, default=2.0)
    parser.add_argument("--edge-rank-weight", type=float, default=0.20)
    parser.add_argument("--enc-h-dropout", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    main(
        max_nodes=args.max_nodes,
        target_graphs=args.target_graphs,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_final=args.beta_final,
        edge_weight=args.edge_weight,
        edge_rank_weight=args.edge_rank_weight,
        enc_h_dropout=args.enc_h_dropout,
        patience=args.patience,
    )
