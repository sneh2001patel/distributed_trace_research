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


# Unsupervised training
def get_graph_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x = model.conv1(batch.x, batch.edge_index)
            x = F.relu(x)
            x = global_mean_pool(x, batch.batch)
            embeddings.append(x.cpu())
    return torch.cat(embeddings)


def train_unsupervised(model, loader, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass (node embeddings)
            x = model.conv1(batch.x, batch.edge_index)
            x = F.relu(x)
            x = model.conv2(x, batch.edge_index)
            x = F.relu(x)

            # Compute self-supervised loss
            # (example: try to reconstruct node features)
            recon = model.lin(global_mean_pool(x, batch.batch))
            loss = F.mse_loss(recon, global_mean_pool(batch.x, batch.batch))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")
