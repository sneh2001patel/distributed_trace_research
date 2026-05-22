from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


def _log_normalize_duration(
    raw_duration: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Log + z-score durations so encoder/decoder see a well-scaled target."""
    dur = torch.clamp(raw_duration, min=0)
    dur = torch.log1p(dur)
    return (dur - mean) / (std + 1e-9)


def _log_denormalize_duration(
    norm_duration: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Invert `_log_normalize_duration`."""
    dur = norm_duration * (std + 1e-9) + mean
    return torch.expm1(dur).clamp(min=0)


class GraphEncoderVAE(nn.Module):
    """
    Node-conditioned graph encoder:
    - returns a graph-level latent and node-level latents (plus the pre-pooled node embeddings)
    """

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

        in_channels = embed_dim * 3
        self.conv1 = GCNConv(in_channels, hidden_ch)
        self.bn1 = nn.LayerNorm(hidden_ch)
        self.conv2 = GCNConv(hidden_ch, hidden_ch)
        self.bn2 = nn.LayerNorm(hidden_ch)

        self.lin_feat = nn.Linear(hidden_ch, hidden_ch // 2)
        self.dropout = dropout

        self.graph_mu = nn.Linear(hidden_ch // 2, latent_dim)
        self.graph_logvar = nn.Linear(hidden_ch // 2, latent_dim)

        # Node-level latent heads so the decoder can reconstruct specific nodes
        self.node_mu = nn.Linear(hidden_ch, latent_dim)
        self.node_logvar = nn.Linear(hidden_ch, latent_dim)

        duration_std = max(duration_std, 1e-6)
        self.register_buffer("duration_mean", torch.tensor(duration_mean))
        self.register_buffer("duration_std", torch.tensor(duration_std))

    def build_feats(self, x: torch.Tensor) -> torch.Tensor:
        service_ids = (
            x[:, 0].long().clamp(min=0, max=self.service_embeddings.num_embeddings - 1)
        )
        op_ids = x[:, 1].long().clamp(min=0, max=self.op_embeddings.num_embeddings - 1)

        duration = _log_normalize_duration(
            x[:, 2], self.duration_mean, self.duration_std
        )
        duration = duration.unsqueeze(-1)

        service_feat = self.service_embeddings(service_ids)
        op_feat = self.op_embeddings(op_ids)
        duration_feat = self.duration_encoder(duration)

        return torch.cat([service_feat, op_feat, duration_feat], dim=1)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, batch: torch.Tensor):
        feats = self.build_feats(x)

        h = self.conv1(feats, edge_idx)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_idx)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # graph-level latent
        h_graph = global_mean_pool(h, batch)
        h_graph = F.relu(self.lin_feat(h_graph))
        h_graph = F.dropout(h_graph, p=self.dropout, training=self.training)
        mu_g = self.graph_mu(h_graph)
        logvar_g = self.graph_logvar(h_graph)
        z_g = self._reparameterize(mu_g, logvar_g)

        # node-level latent
        mu_n = self.node_mu(h)
        logvar_n = self.node_logvar(h)
        z_n = self._reparameterize(mu_n, logvar_n)

        return {
            "graph": (mu_g, logvar_g, z_g),
            "node": (mu_n, logvar_n, z_n),
            "node_h": h,
        }


class GNNDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_service_classes: int,
        n_op_classes: int,
        encoder_hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        max_nodes: int = 256,
        head_hidden_dim: int = None,
        head_dropout: float = 0.2,
        enc_h_dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_service_classes = n_service_classes
        self.n_op_classes = n_op_classes
        self.head_dropout = head_dropout
        self.enc_h_dropout = enc_h_dropout
        head_hidden_dim = head_hidden_dim or hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        self.node_latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.graph_bias = nn.Linear(latent_dim, hidden_dim)
        self.enc_skip = nn.Linear(encoder_hidden_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_nodes, hidden_dim)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.gnn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)]
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + encoder_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
        )

        self.service_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden_dim, n_service_classes),
        )
        # input = h_cls + service context (one-hot during training, softmax at inference)
        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim + n_service_classes, head_hidden_dim),
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

    def _build_upper_edge_index(self, n_nodes: int, device: torch.device):
        row = torch.arange(n_nodes, device=device).repeat_interleave(n_nodes)
        col = torch.arange(n_nodes, device=device).repeat(n_nodes)
        mask = row < col  # upper triangle only
        return torch.stack([row[mask], col[mask]], dim=0)

    def _ensure_pos_capacity(self, n_nodes: int, device: torch.device):
        """Grow positional embeddings if a graph has more nodes than max_nodes."""
        if n_nodes <= self.pos_emb.num_embeddings:
            return
        new_size = max(n_nodes, self.pos_emb.num_embeddings * 2)
        new_emb = nn.Embedding(new_size, self.hidden_dim, device=device)
        with torch.no_grad():
            new_emb.weight[: self.pos_emb.num_embeddings].copy_(self.pos_emb.weight)
            nn.init.normal_(new_emb.weight[self.pos_emb.num_embeddings :], std=0.02)
        self.pos_emb = new_emb

    def decode_for_training(
        self,
        z_graph: torch.Tensor,
        z_nodes: torch.Tensor,
        enc_h: torch.Tensor,
        service_gt: torch.Tensor = None,
    ):
        """
        z_graph:    (latent_dim,)
        z_nodes:    (n_nodes, latent_dim)
        enc_h:      (n_nodes, encoder_hidden_dim)
        service_gt: (n_nodes,) int — ground-truth service ids for teacher forcing.
                    Pass during training; omit at inference to use predicted softmax.
        """
        device = z_graph.device
        n_nodes = z_nodes.size(0)
        if self.training and self.enc_h_dropout > 0.0:
            if self.enc_h_dropout >= 1.0:
                enc_h = torch.zeros_like(enc_h)
            else:
                keep = torch.rand_like(enc_h) > self.enc_h_dropout
                enc_h = enc_h * keep.float()

        h = self.graph_bias(z_graph).unsqueeze(0).expand(n_nodes, -1)
        h = h + self.node_latent_proj(z_nodes)
        h = h + self.enc_skip(enc_h)

        node_idx = torch.arange(n_nodes, device=device)
        self._ensure_pos_capacity(n_nodes, device)
        h = h + self.pos_emb(node_idx)

        edge_index = self._build_upper_edge_index(n_nodes, device)

        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = h + h_res

        h_cls = self.fuse(torch.cat([h, enc_h], dim=1))
        service_logits = self.service_head(h_cls)
        if service_gt is not None:
            service_context = F.one_hot(
                service_gt.clamp(0, self.n_service_classes - 1), self.n_service_classes
            ).float()
        else:
            service_context = torch.softmax(service_logits.detach(), dim=1)
        op_logits = self.op_head(torch.cat([h_cls, service_context], dim=1))
        duration_pred = self.duration_head(h_cls)

        hi = h.unsqueeze(1).expand(-1, n_nodes, -1)
        hj = h.unsqueeze(0).expand(n_nodes, -1, -1)
        edge_logits = self.edge_mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)

        # block diagonal + lower triangle
        diag_mask = torch.eye(n_nodes, device=device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)
        edge_logits = torch.triu(edge_logits, diagonal=1)

        return service_logits, op_logits, duration_pred, edge_logits


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl_per_sample.mean()


def compute_class_weights(dataset, max_ratio: float = 5.0):
    """
    Compute inverse-frequency class weights for service and op.
    We clamp extremely rare classes to avoid unstable gradients.
    Returns (service_weight, op_weight) CPU tensors sized [num_classes].
    """
    service_counter = Counter()
    op_counter = Counter()
    for g in dataset:
        x = g.x
        service_counter.update(x[:, 0].long().tolist())
        op_counter.update(x[:, 1].long().tolist())

    def _make_weight(counter):
        num_classes = max(counter.keys()) + 1
        freq = torch.ones(num_classes, dtype=torch.float)
        for k, v in counter.items():
            freq[k] = v
        inv = 1.0 / (freq + 1e-8)
        inv = inv / inv.mean()  # normalize to keep average weight ~=1
        max_w = inv.median() * max_ratio
        inv = torch.clamp(inv, max=max_w)
        return inv

    return _make_weight(service_counter), _make_weight(op_counter)


def node_recon_loss(
    real_x: torch.Tensor,
    service_logits: torch.Tensor,
    op_logits: torch.Tensor,
    duration_pred: torch.Tensor,
    duration_mean: torch.Tensor,
    duration_std: torch.Tensor,
    service_weight: torch.Tensor = None,
    op_weight: torch.Tensor = None,
    op_loss_scale: float = 1.0,
) -> torch.Tensor:
    service_real = real_x[:, 0].long()
    op_real = real_x[:, 1].long()
    duration_real = real_x[:, 2]

    service_loss = F.cross_entropy(service_logits, service_real, weight=service_weight)
    op_loss = F.cross_entropy(op_logits, op_real, weight=op_weight) * op_loss_scale

    target_duration = _log_normalize_duration(
        duration_real, duration_mean, duration_std
    )
    duration_loss = F.l1_loss(duration_pred.squeeze(-1), target_duration)

    return service_loss + op_loss + duration_loss


def edge_recon_loss(
    real_edges: torch.Tensor,
    pred_edge_logits: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    neg_weight: float = 2.0,
) -> torch.Tensor:
    mask = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1).bool()
    target = torch.zeros((n_nodes, n_nodes), device=device)
    if real_edges.numel() > 0:
        target[real_edges[0], real_edges[1]] = 1.0

    target = target[mask]
    logits = pred_edge_logits[mask]

    num_pos = target.sum()
    num_neg = target.numel() - num_pos

    if num_pos == 0:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos_weight = torch.tensor(
            min((num_neg / (num_pos + 1e-6)).sqrt().item(), 5.0), device=device
        )

    weight = torch.ones_like(target)
    weight[target == 0] = neg_weight  # penalize false positives harder

    return F.binary_cross_entropy_with_logits(
        logits, target, weight=weight, pos_weight=pos_weight
    )


def total_vae_loss(
    real_data,
    service_logits,
    op_logits,
    duration_pred,
    pred_edge_logits,
    mu_g,
    logvar_g,
    mu_n,
    logvar_n,
    duration_mean,
    duration_std,
    service_weight=None,
    op_weight=None,
    op_loss_scale: float = 1.0,
    beta: float = 0.01,
    edge_weight: float = 3.0,
    edge_neg_weight: float = 2.0,
):
    node_l = node_recon_loss(
        real_data.x,
        service_logits,
        op_logits,
        duration_pred,
        duration_mean,
        duration_std,
        service_weight=service_weight,
        op_weight=op_weight,
        op_loss_scale=op_loss_scale,
    )
    edge_l = edge_recon_loss(
        real_data.edge_index,
        pred_edge_logits,
        real_data.num_nodes,
        real_data.x.device,
        neg_weight=edge_neg_weight,
    )
    kl_g = kl_loss(mu_g, logvar_g)
    kl_n = kl_loss(mu_n, logvar_n)

    loss = node_l + edge_weight * edge_l + beta * (kl_g + 0.1 * kl_n)
    return loss, node_l, edge_l, kl_g + 0.1 * kl_n


@torch.no_grad()
def decode_and_denorm_duration(
    duration_pred: torch.Tensor, duration_mean: torch.Tensor, duration_std: torch.Tensor
) -> torch.Tensor:
    return _log_denormalize_duration(
        duration_pred.squeeze(-1), duration_mean, duration_std
    )


@torch.no_grad()
def evaluate_reconstruction(
    dataset, encoder, decoder, device, edge_threshold: float = 0.75
):
    encoder.eval()
    decoder.eval()

    total_nodes = 0
    correct_service = 0
    correct_op = 0
    duration_abs_err = 0.0
    edge_TP = edge_FP = edge_FN = 0

    for data in DataLoader(dataset, batch_size=1, shuffle=False):
        data = data.to(device)
        enc_out = encoder(data.x, data.edge_index, data.batch)
        mu_g, logvar_g, z_g = enc_out["graph"]
        mu_n, logvar_n, z_n = enc_out["node"]
        h_enc = enc_out["node_h"]

        service_logits, op_logits, duration_pred, edge_logits = (
            decoder.decode_for_training(z_g[0], z_n, h_enc)
        )

        service_pred = service_logits.argmax(dim=1)
        op_pred = op_logits.argmax(dim=1)
        dur_pred = decode_and_denorm_duration(
            duration_pred, encoder.duration_mean, encoder.duration_std
        )

        service_real = data.x[:, 0].long()
        op_real = data.x[:, 1].long()
        dur_real = data.x[:, 2]

        total_nodes += data.num_nodes
        correct_service += (service_pred == service_real).sum().item()
        correct_op += (op_pred == op_real).sum().item()
        dur_pred_log = torch.log1p(dur_pred.clamp(min=0))
        dur_real_log = torch.log1p(dur_real.clamp(min=0))
        duration_abs_err += torch.abs(dur_pred_log - dur_real_log).sum().item()

        adj_real = torch.zeros((data.num_nodes, data.num_nodes), device=device)
        if data.edge_index.numel() > 0:
            adj_real[data.edge_index[0], data.edge_index[1]] = 1.0

        adj_pred = (torch.sigmoid(edge_logits) > edge_threshold).float()
        mask = torch.triu(
            torch.ones(data.num_nodes, data.num_nodes, device=device), diagonal=1
        ).bool()

        real_vec = adj_real[mask]
        pred_vec = adj_pred[mask]

        TP = ((pred_vec == 1) & (real_vec == 1)).sum().item()
        FP = ((pred_vec == 1) & (real_vec == 0)).sum().item()
        FN = ((pred_vec == 0) & (real_vec == 1)).sum().item()

        edge_TP += TP
        edge_FP += FP
        edge_FN += FN

    service_acc = correct_service / total_nodes if total_nodes > 0 else 0.0
    op_acc = correct_op / total_nodes if total_nodes > 0 else 0.0
    dur_mae = duration_abs_err / total_nodes if total_nodes > 0 else 0.0
    precision = edge_TP / (edge_TP + edge_FP) if (edge_TP + edge_FP) > 0 else 0.0
    recall = edge_TP / (edge_TP + edge_FN) if (edge_TP + edge_FN) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "total_nodes": total_nodes,
        "service_acc": service_acc,
        "op_acc": op_acc,
        "dur_mae": dur_mae,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
    }


def _ensure_batch(data):
    if not hasattr(data, "batch"):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
    return data


def train_vae_epoch(
    encoder,
    decoder,
    loader,
    optimizer,
    device,
    beta: float,
    op_loss_scale: float = 1.0,
    edge_weight: float = 3.0,
    edge_neg_weight: float = 2.0,
):
    encoder.train()
    decoder.train()

    total_loss = total_node = total_edge = total_kl = 0.0

    for data in loader:
        data = data.to(device)
        data = _ensure_batch(data)

        optimizer.zero_grad()

        enc_out = encoder(data.x, data.edge_index, data.batch)
        mu_g, logvar_g, z_g = enc_out["graph"]
        mu_n, logvar_n, z_n = enc_out["node"]

        service_gt = data.x[:, 0].long()
        service_logits, op_logits, duration_pred, edge_logits = (
            decoder.decode_for_training(z_g[0], z_n, enc_out["node_h"], service_gt=service_gt)
        )

        loss, node_l, edge_l, kl_l = total_vae_loss(
            data,
            service_logits,
            op_logits,
            duration_pred,
            edge_logits,
            mu_g,
            logvar_g,
            mu_n,
            logvar_n,
            encoder.duration_mean,
            encoder.duration_std,
            service_weight=getattr(encoder, "service_weight", None),
            op_weight=getattr(encoder, "op_weight", None),
            op_loss_scale=op_loss_scale,
            beta=beta,
            edge_weight=edge_weight,
            edge_neg_weight=edge_neg_weight,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_node += node_l.item()
        total_edge += edge_l.item()
        total_kl += kl_l.item()

    n_batches = len(loader)
    return (
        total_loss / n_batches,
        total_node / n_batches,
        total_edge / n_batches,
        total_kl / n_batches,
    )


def run_training(
    dataset,
    encoder,
    decoder,
    device,
    epochs: int = 80,
    batch_size: int = 2,
    lr: float = 3e-4,
    beta_final: float = 0.05,
    use_class_weights: bool = True,
    op_loss_scale: float = 1.0,
    edge_weight: float = 3.0,
    edge_neg_weight: float = 2.0,
    class_weight_max_ratio: float = 5.0,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    if use_class_weights:
        service_w_cpu, op_w_cpu = compute_class_weights(
            dataset, max_ratio=class_weight_max_ratio
        )
        encoder.service_weight = service_w_cpu.to(device)
        encoder.op_weight = op_w_cpu.to(device)
    else:
        encoder.service_weight = encoder.op_weight = None

    warmup = max(5, int(epochs * 0.2))
    for epoch in range(1, epochs + 1):
        beta = beta_final * min(1.0, epoch / warmup)
        loss, node_l, edge_l, kl_l = train_vae_epoch(
            encoder,
            decoder,
            loader,
            optimizer,
            device,
            beta,
            op_loss_scale,
            edge_weight,
            edge_neg_weight,
        )
        print(
            f"Epoch {epoch:03d} | loss {loss:.4f} | node {node_l:.4f} | edge {edge_l:.4f} | kl {kl_l:.4f} | beta {beta:.4f}"
        )
