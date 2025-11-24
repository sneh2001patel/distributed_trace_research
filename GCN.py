import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    """
    Graph classifier that treats the first two columns in `x` as categorical
    identifiers (pod + operation) and the last column as a numeric duration.
    """

    def __init__(
        self,
        num_pods: int,
        num_ops: int,
        duration_mean: float,
        duration_std: float,
        hidden_channels: int = 128,
        embed_dim: int = 48,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.pod_emb = torch.nn.Embedding(num_pods, embed_dim)
        self.op_emb = torch.nn.Embedding(num_ops, embed_dim)
        self.duration_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

        in_channels = embed_dim * 3
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, 2)  # two logits
        self.dropout = dropout

        duration_std = max(duration_std, 1e-6)
        self.register_buffer("duration_mean", torch.tensor(duration_mean))
        self.register_buffer("duration_std", torch.tensor(duration_std))

    def _build_features(self, x: torch.Tensor) -> torch.Tensor:
        pod_ids = x[:, 0].long().clamp(min=0, max=self.pod_emb.num_embeddings - 1)
        op_ids = x[:, 1].long().clamp(min=0, max=self.op_emb.num_embeddings - 1)
        duration = torch.clamp(x[:, 2], min=0)
        duration = torch.log1p(duration)
        duration = (duration - self.duration_mean) / (self.duration_std + 1e-9)
        duration = duration.unsqueeze(-1)

        pod_feat = self.pod_emb(pod_ids)
        op_feat = self.op_emb(op_ids)
        duration_feat = self.duration_encoder(duration)

        return torch.cat([pod_feat, op_feat, duration_feat], dim=-1)

    def forward(self, x, edge_index, batch):
        feats = self._build_features(x)

        h = self.conv1(feats, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = global_mean_pool(h, batch)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)
