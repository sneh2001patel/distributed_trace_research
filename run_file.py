import os

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import Data, DataEdgeAttr
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from vae_improved import GNNDecoder, GraphEncoder, evaluate_reconstruction, run_training


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath="./data.pt") -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices
        x = data.x
        self.num_services = int(x[:, 0].max().item()) + 1
        self.num_operations = int(x[:, 1].max().item()) + 1
        log_duration = torch.log1p(torch.clamp(x[:, 2], min=0))
        self.duration_mean = log_duration.mean().item()
        self.duration_std = log_duration.std().item() or 1.0

    def get(self, idx):
        data = super().get(idx)
        data.edge_index = to_undirected(data.edge_index)
        return data


def main():
    datapath = "./processed/SN_data.pt"
    dataset = LoadDataset(datapath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GraphEncoder(
        n_services=dataset.num_services,
        n_operations=dataset.num_operations,
        duration_mean=dataset.duration_mean,
        duration_std=dataset.duration_std,
        hidden_ch=256,
        embed_dim=96,
        out_dim=128,
    ).to(device)

    dec = GNNDecoder(
        latent_dim=128,
        hidden_dim=256,
        n_service_classes=dataset.num_services,
        n_operation_classes=dataset.num_operations,
        encoder_hidden_dim=256,
        num_gnn_layers=4,
        max_nodes=256,
    ).to(device)

    epochs = int(os.getenv("EPOCHS", "150"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    lr = float(os.getenv("LR", "3e-4"))
    op_loss_scale = float(os.getenv("OP_LOSS_SCALE", "2.0"))
    edge_weight = float(os.getenv("EDGE_WEIGHT", "2.0"))
    edge_threshold = float(os.getenv("EDGE_THRESHOLD", "0.7"))
    use_class_weights = os.getenv("USE_CLASS_WEIGHTS", "true").lower() != "false"

    run_training(
        dataset,
        enc,
        dec,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_class_weights=use_class_weights,
        op_loss_scale=op_loss_scale,
        edge_weight=edge_weight,
    )

    metrics = evaluate_reconstruction(
        dataset, enc, dec, device, edge_threshold=edge_threshold
    )
    print(metrics)

    print("\n=== Reconstruction Evaluation ===")
    print(f"Total nodes evaluated: {metrics['total_nodes']}")
    print(f"Service name accuracy: {metrics['service_acc']*100:.2f}%")
    print(f"Operation accuracy:  {metrics['op_acc']*100:.2f}%")
    print(f"Duration MAE: {metrics['dur_mae']:.4f}")
    print(f"Edge precision: {metrics['edge_precision']*100:.2f}%")
    print(f"Edge recall:    {metrics['edge_recall']*100:.2f}%")
    print(f"Edge F1:        {metrics['edge_f1']*100:.2f}%")


if __name__ == "__main__":
    main()
