import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


def _log_normalize_duration(raw_duration: torch.Tensor, mean, std) -> torch.Tensor:
    dur = torch.clamp(raw_duration, min=0)
    dur = torch.log1p(dur)
    if mean is None or std is None:
        return dur
    return (dur - mean) / (std + 1e-9)


class GraphClassifier(nn.Module):
    def __init__(
        self,
        n_pods: int,
        n_ops: int,
        num_classes: int = 2,
        embed_dim: int = 48,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        duration_mean: float = None,
        duration_std: float = None,
    ):
        super().__init__()
        self.pod_embeddings = nn.Embedding(n_pods, embed_dim)
        self.op_embeddings = nn.Embedding(n_ops, embed_dim)
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        in_channels = embed_dim * 3
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_ch, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.dropout = dropout

        if duration_mean is not None and duration_std is not None:
            duration_std = max(float(duration_std), 1e-6)
            self.register_buffer("duration_mean", torch.tensor(float(duration_mean)))
            self.register_buffer("duration_std", torch.tensor(duration_std))
        else:
            self.duration_mean = None
            self.duration_std = None

    def _build_feats(self, x: torch.Tensor) -> torch.Tensor:
        pod_ids = x[:, 0].long().clamp(min=0, max=self.pod_embeddings.num_embeddings - 1)
        op_ids = x[:, 1].long().clamp(min=0, max=self.op_embeddings.num_embeddings - 1)
        duration = _log_normalize_duration(
            x[:, 2],
            self.duration_mean,
            self.duration_std,
        ).unsqueeze(-1)

        pod_feat = self.pod_embeddings(pod_ids)
        op_feat = self.op_embeddings(op_ids)
        duration_feat = self.duration_encoder(duration)
        return torch.cat([pod_feat, op_feat, duration_feat], dim=1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self._build_feats(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        g = global_mean_pool(h, batch)
        return self.classifier(g)
