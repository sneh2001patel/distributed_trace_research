import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from GCN import GCN, get_graph_embeddings, train_unsupervised

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()


# load
data = torch.load("trace_data.pt")
print(f"Loaded {len(data)} trace graphs.")
train_loader = DataLoader(data, batch_size=8, shuffle=True)

embeddings = get_graph_embeddings(model, train_loader, device)
print(f"Embeddings: {embeddings.shape}")

# train the model
train_unsupervised(model, train_loader, optimizer, device)


# torch.save(model.state_dict(), "gcn_trace_model.pt")
#
# model.load_state_dict(torch.load("gcn_trace_model.pt"))
model.eval()
