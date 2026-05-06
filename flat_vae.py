import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from vae import (
    _log_denormalize_duration,
    _log_normalize_duration,
    edge_recon_loss,
    kl_loss,
    node_recon_loss,
)


class FlatGraphEncoderVAE(nn.Module):
    """One-latent graph VAE encoder: encode the whole trace into z_G only."""

    def __init__(
        self,
        n_services: int,
        n_ops: int,
        duration_mean: float,
        duration_std: float,
        hidden_ch: int = 128,
        embed_dim: int = 48,
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.service_embeddings = nn.Embedding(n_services, embed_dim)
        self.op_embeddings = nn.Embedding(n_ops, embed_dim)
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = GCNConv(embed_dim * 3, hidden_ch)
        self.norm1 = nn.LayerNorm(hidden_ch)
        self.conv2 = GCNConv(hidden_ch, hidden_ch)
        self.norm2 = nn.LayerNorm(hidden_ch)
        self.lin_feat = nn.Linear(hidden_ch, hidden_ch // 2)
        self.mu = nn.Linear(hidden_ch // 2, latent_dim)
        self.logvar = nn.Linear(hidden_ch // 2, latent_dim)
        self.dropout = dropout

        self.register_buffer("duration_mean", torch.tensor(duration_mean))
        self.register_buffer("duration_std", torch.tensor(max(duration_std, 1e-6)))

    def build_feats(self, x: torch.Tensor) -> torch.Tensor:
        service_ids = (
            x[:, 0].long().clamp(min=0, max=self.service_embeddings.num_embeddings - 1)
        )
        op_ids = x[:, 1].long().clamp(min=0, max=self.op_embeddings.num_embeddings - 1)
        duration = _log_normalize_duration(
            x[:, 2], self.duration_mean, self.duration_std
        ).unsqueeze(-1)
        return torch.cat(
            [
                self.service_embeddings(service_ids),
                self.op_embeddings(op_ids),
                self.duration_encoder(duration),
            ],
            dim=1,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, batch: torch.Tensor):
        h = self.build_feats(x)
        h = F.relu(self.norm1(self.conv1(h, edge_idx)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.norm2(self.conv2(h, edge_idx)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_graph = global_mean_pool(h, batch)
        h_graph = F.relu(self.lin_feat(h_graph))
        h_graph = F.dropout(h_graph, p=self.dropout, training=self.training)
        mu = self.mu(h_graph)
        logvar = self.logvar(h_graph)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class FlatGNNDecoder(nn.Module):
    """Decode all node attributes and edges from one graph latent plus positions."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_service_classes: int,
        n_op_classes: int,
        num_gnn_layers: int = 2,
        max_nodes: int = 256,
        head_hidden_dim: int = None,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_service_classes = n_service_classes
        self.n_op_classes = n_op_classes
        head_hidden_dim = head_hidden_dim or hidden_dim

        self.graph_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_nodes, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.service_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden_dim, n_service_classes),
        )
        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden_dim, n_op_classes),
        )
        self.duration_head = nn.Linear(hidden_dim, 1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _ensure_pos_capacity(self, n_nodes: int, device: torch.device):
        if n_nodes <= self.pos_emb.num_embeddings:
            return
        new_size = max(n_nodes, self.pos_emb.num_embeddings * 2)
        new_emb = nn.Embedding(new_size, self.hidden_dim, device=device)
        with torch.no_grad():
            new_emb.weight[: self.pos_emb.num_embeddings].copy_(self.pos_emb.weight)
            nn.init.normal_(new_emb.weight[self.pos_emb.num_embeddings :], std=0.02)
        self.pos_emb = new_emb

    def _upper_edge_index(self, n_nodes: int, device: torch.device):
        row = torch.arange(n_nodes, device=device).repeat_interleave(n_nodes)
        col = torch.arange(n_nodes, device=device).repeat(n_nodes)
        mask = row < col
        return torch.stack([row[mask], col[mask]], dim=0)

    def decode(self, z_graph: torch.Tensor, n_nodes: int):
        device = z_graph.device
        self._ensure_pos_capacity(n_nodes, device)
        node_idx = torch.arange(n_nodes, device=device)
        h = self.graph_proj(z_graph).unsqueeze(0).expand(n_nodes, -1)
        h = h + self.pos_emb(node_idx)

        edge_index = self._upper_edge_index(n_nodes, device)
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h_res = h
            h = F.relu(norm(conv(h, edge_index)))
            h = h + h_res

        service_logits = self.service_head(h)
        op_logits = self.op_head(h)
        duration_pred = self.duration_head(h)

        hi = h.unsqueeze(1).expand(-1, n_nodes, -1)
        hj = h.unsqueeze(0).expand(n_nodes, -1, -1)
        edge_logits = self.edge_mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)
        diag_mask = torch.eye(n_nodes, device=device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)
        edge_logits = torch.triu(edge_logits, diagonal=1)
        return service_logits, op_logits, duration_pred, edge_logits


def flat_vae_loss(
    data,
    service_logits,
    op_logits,
    duration_pred,
    edge_logits,
    mu,
    logvar,
    duration_mean,
    duration_std,
    beta: float = 0.02,
    edge_weight: float = 1.0,
    edge_neg_weight: float = 3.0,
    op_loss_scale: float = 1.0,
):
    node_l = node_recon_loss(
        data.x,
        service_logits,
        op_logits,
        duration_pred,
        duration_mean,
        duration_std,
        op_loss_scale=op_loss_scale,
    )
    if data.num_nodes < 2:
        edge_l = torch.zeros((), device=data.x.device)
    else:
        edge_l = edge_recon_loss(
            data.edge_index,
            edge_logits,
            data.num_nodes,
            data.x.device,
            neg_weight=edge_neg_weight,
        )
    kl_l = kl_loss(mu, logvar)
    return node_l + edge_weight * edge_l + beta * kl_l, node_l, edge_l, kl_l


@torch.no_grad()
def denormalize_duration(duration_pred, duration_mean, duration_std):
    return _log_denormalize_duration(
        duration_pred.squeeze(-1), duration_mean, duration_std
    )
