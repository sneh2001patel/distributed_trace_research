import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.reul(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        # For now use random dummy targets
        target = torch.randint(0, 2, (out.size(0),), device=device)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")


with torch.no_grad():
    model.eval()
    embeddings = []
    for batch in train_loader:
        batch = batch.to(device)
        x = model.conv1(batch.x, batch.edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch.batch)
        embeddings.append(x.cpu())
    embeddings = torch.cat(embeddings)
print("Learned graph embeddings shape:", embeddings.shape)
