import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.reul(x)
        # 2. Pool node embeddings -> graph embedding
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        # 3. Final classification / embedding output
        x = self.lin(x)

        return x


# Supervised training
