import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

try:
    from model import _log_normalize_duration
except ModuleNotFoundError:
    from classifier.model import _log_normalize_duration


class TTGraphClassifier(nn.Module):
    def __init__(
        self,
        n_services: int,
        n_ops: int,
        num_classes: int = 2,
        embed_dim: int = 32,
        hidden_dim: int = 96,
        num_layers: int = 3,
        dropout: float = 0.15,
        duration_mean: float = None,
        duration_std: float = None,
    ):
        super().__init__()
        self.service_embeddings = nn.Embedding(n_services, embed_dim)
        self.op_embeddings = nn.Embedding(n_ops, embed_dim)
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        in_channels = embed_dim * 3
        self.fwd_convs = nn.ModuleList()
        self.rev_convs = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            self.fwd_convs.append(SAGEConv(in_ch, hidden_dim))
            self.rev_convs.append(SAGEConv(in_ch, hidden_dim))
            self.fuse_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.stats_proj = nn.Sequential(
            nn.Linear(13, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
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
        service_ids = (
            x[:, 0].long().clamp(min=0, max=self.service_embeddings.num_embeddings - 1)
        )
        op_ids = x[:, 1].long().clamp(min=0, max=self.op_embeddings.num_embeddings - 1)
        duration = _log_normalize_duration(
            x[:, 2], self.duration_mean, self.duration_std
        ).unsqueeze(-1)

        service_feat = self.service_embeddings(service_ids)
        op_feat = self.op_embeddings(op_ids)
        duration_feat = self.duration_encoder(duration)
        return torch.cat([service_feat, op_feat, duration_feat], dim=1)

    def _graph_stats(self, edge_index: torch.Tensor, batch: torch.Tensor, num_nodes: int):
        batch_size = int(batch.max().item()) + 1 if num_nodes > 0 else 1
        node_counts = torch.bincount(batch, minlength=batch_size).float()

        if edge_index.numel() > 0:
            edge_batch = batch[edge_index[0]]
            edge_counts = torch.bincount(edge_batch, minlength=batch_size).float()
            indeg = torch.bincount(edge_index[1], minlength=num_nodes).float()
            outdeg = torch.bincount(edge_index[0], minlength=num_nodes).float()
        else:
            edge_counts = torch.zeros(batch_size, device=batch.device)
            indeg = torch.zeros(num_nodes, device=batch.device)
            outdeg = torch.zeros(num_nodes, device=batch.device)

        max_possible = torch.clamp(node_counts * torch.clamp(node_counts - 1, min=0), min=1.0)
        density = edge_counts / max_possible

        roots = global_mean_pool((indeg == 0).float().unsqueeze(1), batch).squeeze(1) * node_counts
        leaves = global_mean_pool((outdeg == 0).float().unsqueeze(1), batch).squeeze(1) * node_counts
        mean_indeg = global_mean_pool(indeg.unsqueeze(1), batch).squeeze(1)
        mean_outdeg = global_mean_pool(outdeg.unsqueeze(1), batch).squeeze(1)
        max_indeg = self._global_max_pool(indeg.unsqueeze(1), batch).squeeze(1)
        max_outdeg = self._global_max_pool(outdeg.unsqueeze(1), batch).squeeze(1)

        x_duration = self._cached_x[:, 2]
        duration_mean = global_mean_pool(x_duration.unsqueeze(1), batch).squeeze(1)
        duration_sq_mean = global_mean_pool((x_duration ** 2).unsqueeze(1), batch).squeeze(1)
        duration_std = torch.sqrt(torch.clamp(duration_sq_mean - duration_mean**2, min=0.0))
        duration_max = self._global_max_pool(x_duration.unsqueeze(1), batch).squeeze(1)

        service_ids = self._cached_x[:, 0].long()
        op_ids = self._cached_x[:, 1].long()
        uniq_services = []
        uniq_ops = []
        for graph_idx in range(batch_size):
            mask = batch == graph_idx
            uniq_services.append(torch.unique(service_ids[mask]).numel())
            uniq_ops.append(torch.unique(op_ids[mask]).numel())
        uniq_services = torch.tensor(uniq_services, device=batch.device, dtype=torch.float)
        uniq_ops = torch.tensor(uniq_ops, device=batch.device, dtype=torch.float)

        return torch.stack(
            [
                torch.log1p(node_counts),
                torch.log1p(edge_counts),
                density,
                roots / torch.clamp(node_counts, min=1.0),
                leaves / torch.clamp(node_counts, min=1.0),
                mean_indeg,
                mean_outdeg,
                max_indeg,
                max_outdeg,
                torch.log1p(uniq_services),
                torch.log1p(uniq_ops),
                torch.log1p(torch.clamp(duration_mean, min=0.0)),
                torch.log1p(torch.clamp(duration_std + duration_max, min=0.0)),
            ],
            dim=1,
        )

    def _global_max_pool(self, x: torch.Tensor, batch: torch.Tensor):
        batch_size = int(batch.max().item()) + 1 if x.size(0) > 0 else 1
        out = torch.full(
            (batch_size, x.size(1)),
            -math.inf,
            dtype=x.dtype,
            device=x.device,
        )
        index = batch.unsqueeze(1).expand_as(x)
        out.scatter_reduce_(0, index, x, reduce="amax", include_self=True)
        return torch.where(out == -math.inf, torch.zeros_like(out), out)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        self._cached_x = x

        h = self._build_feats(x)
        rev_edge_index = edge_index[[1, 0]] if edge_index.numel() > 0 else edge_index

        for fwd_conv, rev_conv, fuse, norm in zip(
            self.fwd_convs, self.rev_convs, self.fuse_layers, self.norms
        ):
            h_fwd = fwd_conv(h, edge_index)
            h_rev = rev_conv(h, rev_edge_index)
            h = fuse(torch.cat([h_fwd, h_rev], dim=1))
            h = norm(h)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        g_mean = global_mean_pool(h, batch)
        g_max = self._global_max_pool(h, batch)
        g_stats = self.stats_proj(self._graph_stats(edge_index, batch, x.size(0)))
        g = torch.cat([g_mean, g_max, g_stats], dim=1)
        return self.classifier(g)
