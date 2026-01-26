import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import Data, DataEdgeAttr
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from vae import GNNDecoder, GraphEncoderVAE, evaluate_reconstruction, run_training

print("\n=== TRAINING TT DATA ===")


# Load the pt dataset to use it PyTorch
class LoadDataset(InMemoryDataset):
    def __init__(self, datapath="./datasets/TT_data.pt") -> None:
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


datapath = "./datasets/TT_data.pt"
dataset = LoadDataset(datapath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Graph encoder
enc = GraphEncoderVAE(
    n_services=dataset.num_services,
    n_ops=dataset.num_ops,
    duration_mean=dataset.duration_mean,
    duration_std=dataset.duration_std,
    hidden_ch=128,
    embed_dim=48,
    latent_dim=64,
).to(device)

# Graph Decoder
dec = GNNDecoder(
    latent_dim=64,
    hidden_dim=128,
    n_service_classes=dataset.num_services,
    n_op_classes=dataset.num_ops,
    encoder_hidden_dim=128,
    max_nodes=256,
    head_hidden_dim=256,
    head_dropout=0.3,
).to(device)

run_training(
    dataset,
    enc,
    dec,
    device,
    epochs=300,
    batch_size=2,
    lr=3e-4,
    beta_final=0.02,
    use_class_weights=False,
    op_loss_scale=1.6,
    edge_weight=1.0,
    edge_neg_weight=3.0,
    class_weight_max_ratio=10.0,
)
torch.save(
    {
        "encoder_state": enc.state_dict(),
        "decoder_state": dec.state_dict(),
        "config": {
            "num_services": dataset.num_services,
            "num_ops": dataset.num_ops,
            "duration_mean": dataset.duration_mean,
            "duration_std": dataset.duration_std,
            "latent_dim": 64,
            "hidden_dim": 128,
            "embed_dim": 48,
        },
    },
    "./weights/tt_vae_weights1.pt",
)
metrics = evaluate_reconstruction(dataset, enc, dec, device, edge_threshold=0.9)
print(metrics)
print("\n=== Reconstruction Evaluation ===")
print(f"Total nodes evaluated: {metrics['total_nodes']}")
print(f"Service accuracy: {metrics['service_acc']*100:.2f}%")
print(f"Op accuracy:  {metrics['op_acc']*100:.2f}%")
print(f"Duration MAE: {metrics['dur_mae']:.4f}")
print(f"Edge precision: {metrics['edge_precision']*100:.2f}%")
print(f"Edge recall:    {metrics['edge_recall']*100:.2f}%")
print(f"Edge F1:        {metrics['edge_f1']*100:.2f}%")
