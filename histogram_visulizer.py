import torch

from torch_geometric.data import Data, InMemoryDataset
from collections import Counter

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
pod_counter = Counter()
op_counter = Counter()


for g in dataset:
    pod_counter.update(g.x[:,0].long().tolist())
    op_counter.update(g.x[:,1].long().tolist())


print("Pods: unique", len(pod_counter))
for k, v in pod_counter.most_common():
    print(f"pods {k}: {v}")

print("Ops: unique", len(op_counter))
for k, v in op_counter.most_common():
    print(f"op {k}: {v}")

