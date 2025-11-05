import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from GCN import GCN

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_channels=3, hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()


# load
data, slices = torch.load("./processed/data.pt")

# reconstruct individual graphs
num_graphs = len(slices["y"]) - 1
graphs = []

for i in range(num_graphs):
    g = Data()
    for key in data.keys():
        item, s = data[key], slices[key]
        g[key] = item[s[i] : s[i + 1]]
    graphs.append(g)

for i, g in enumerate(graphs):
    print(
        f"Graph {i:02d} — Label: {int(g.y)} — Nodes: {g.num_nodes}, Edges: {g.num_edges}"
    )
