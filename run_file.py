import torch
from vae_improved import GraphEncoderVAE, GNNDecoder, run_training, evaluate_reconstruction
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from torch_geometric.data.data import Data, DataEdgeAttr


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath="./data.pt") -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices
        x = data.x
        self.num_pods = int(x[:, 0].max().item()) + 1
        self.num_ops = int(x[:, 1].max().item()) + 1
        log_duration = torch.log1p(torch.clamp(x[:, 2], min=0))
        self.duration_mean = log_duration.mean().item()
        self.duration_std = log_duration.std().item() or 1.0

    def get(self, idx):
        data = super().get(idx)
        data.edge_index = to_undirected(data.edge_index)
        return data

datapath = "./data.pt"
dataset = LoadDataset(datapath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



enc = GraphEncoderVAE(
    n_pods=dataset.num_pods,
    n_ops=dataset.num_ops,
    duration_mean=dataset.duration_mean,
    duration_std=dataset.duration_std,
    hidden_ch=128,
    embed_dim=48,
    latent_dim=64,
).to(device)

dec = GNNDecoder(
    latent_dim=64,
    hidden_dim=128,
    n_pod_classes=dataset.num_pods,
    n_op_classes=dataset.num_ops,
    encoder_hidden_dim=128,
    max_nodes=256,
).to(device)

run_training(dataset, enc, dec, device, epochs=80, batch_size=2, lr=3e-4, beta_final=0.05)
metrics = evaluate_reconstruction(dataset, enc, dec, device, edge_threshold=0.8)
print(metrics)

print("\n=== Reconstruction Evaluation ===")
print(f"Total nodes evaluated: {metrics['total_nodes']}")
print(f"Pod accuracy: {metrics['pod_acc']*100:.2f}%")
print(f"Op accuracy:  {metrics['op_acc']*100:.2f}%")
print(f"Duration MAE: {metrics['dur_mae']:.4f}")
print(f"Edge precision: {metrics['edge_precision']*100:.2f}%")
print(f"Edge recall:    {metrics['edge_recall']*100:.2f}%")
print(f"Edge F1:        {metrics['edge_f1']*100:.2f}%")
