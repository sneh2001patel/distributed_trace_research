import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from collections import Counter


# -----------------------------
# Utility: duration transforms
# -----------------------------
def _log_normalize_duration(raw_duration: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    dur = torch.clamp(raw_duration, min=0)
    dur = torch.log1p(dur)
    return (dur - mean) / (std + 1e-9)


def _log_denormalize_duration(norm_duration: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    dur = norm_duration * (std + 1e-9) + mean
    return torch.expm1(dur).clamp(min=0)


# -----------------------------
# Encoder
# -----------------------------
class GraphEncoder(nn.Module):
    """
    Deterministic encoder for reconstruction. No KL; produces graph and node embeddings.
    """

    def __init__(
        self,
        n_services: int,
        n_operations: int,
        duration_mean: float,
        duration_std: float,
        hidden_ch: int = 256,
        embed_dim: int = 96,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.service_embeddings = nn.Embedding(n_services, embed_dim)
        self.operation_embeddings = nn.Embedding(n_operations, embed_dim)
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        in_channels = embed_dim * 3
        self.conv1 = GCNConv(in_channels, hidden_ch)
        self.conv2 = GATConv(hidden_ch, hidden_ch, heads=2, concat=False)
        self.conv3 = GCNConv(hidden_ch, hidden_ch)
        self.dropout = dropout

        self.graph_proj = nn.Linear(hidden_ch, out_dim)
        self.node_proj = nn.Linear(hidden_ch, out_dim)

        duration_std = max(duration_std, 1e-6)
        self.register_buffer("duration_mean", torch.tensor(duration_mean))
        self.register_buffer("duration_std", torch.tensor(duration_std))

    def build_feats(self, x: torch.Tensor) -> torch.Tensor:
        service_ids = x[:, 0].long().clamp(min=0, max=self.service_embeddings.num_embeddings - 1)
        op_ids = x[:, 1].long().clamp(min=0, max=self.operation_embeddings.num_embeddings - 1)
        duration = _log_normalize_duration(x[:, 2], self.duration_mean, self.duration_std).unsqueeze(-1)
        s = self.service_embeddings(service_ids)
        o = self.operation_embeddings(op_ids)
        d = self.duration_encoder(duration)
        return torch.cat([s, o, d], dim=1)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, batch: torch.Tensor):
        h = self.build_feats(x)
        h = F.relu(self.conv1(h, edge_idx))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.elu(self.conv2(h, edge_idx))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.relu(self.conv3(h, edge_idx))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h_graph = global_mean_pool(h, batch)
        g = self.graph_proj(h_graph)
        n = self.node_proj(h)
        return {"graph": g, "node": n, "node_h": h}


# -----------------------------
# Decoder
# -----------------------------
class GNNDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_service_classes: int,
        n_operation_classes: int,
        encoder_hidden_dim: int = None,
        num_gnn_layers: int = 3,
        max_nodes: int = 256,
    ):
        super().__init__()
        if encoder_hidden_dim is None:
            encoder_hidden_dim = hidden_dim

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_service_classes = n_service_classes
        self.n_operation_classes = n_operation_classes
        self.encoder_hidden_dim = encoder_hidden_dim

        self.node_latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.graph_bias = nn.Linear(latent_dim, hidden_dim)
        self.enc_skip = nn.Linear(encoder_hidden_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_nodes, hidden_dim)

        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) if i % 2 == 0 else GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
             for i in range(num_gnn_layers)]
        )

        self.service_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_service_classes),
        )
        self.operation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_operation_classes),
        )
        self.duration_head = nn.Linear(hidden_dim, 1)

        # stronger edge MLP with gating
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_upper_edge_index(self, n_nodes: int, device: torch.device):
        row = torch.arange(n_nodes, device=device).repeat_interleave(n_nodes)
        col = torch.arange(n_nodes, device=device).repeat(n_nodes)
        mask = row < col
        return torch.stack([row[mask], col[mask]], dim=0)

    def _ensure_pos_capacity(self, n_nodes: int, device: torch.device):
        if n_nodes <= self.pos_emb.num_embeddings:
            return
        new_size = max(n_nodes, self.pos_emb.num_embeddings * 2)
        new_emb = nn.Embedding(new_size, self.hidden_dim, device=device)
        with torch.no_grad():
            new_emb.weight[: self.pos_emb.num_embeddings].copy_(self.pos_emb.weight)
            nn.init.normal_(new_emb.weight[self.pos_emb.num_embeddings:], std=0.02)
        self.pos_emb = new_emb

    def decode_for_training(self, g_emb: torch.Tensor, n_emb: torch.Tensor, enc_h: torch.Tensor):
        device = g_emb.device
        n_nodes = n_emb.size(0)

        h = self.graph_bias(g_emb).unsqueeze(0).expand(n_nodes, -1)
        h = h + self.node_latent_proj(n_emb) + self.enc_skip(enc_h)

        node_idx = torch.arange(n_nodes, device=device)
        self._ensure_pos_capacity(n_nodes, device)
        h = h + self.pos_emb(node_idx)

        edge_index = self._build_upper_edge_index(n_nodes, device)
        for conv in self.gnn_layers:
            h = conv(h, edge_index)
            h = F.relu(h)

        service_logits = self.service_head(h)
        op_logits = self.operation_head(h)
        duration_pred = self.duration_head(h)

        hi = h.unsqueeze(1).expand(-1, n_nodes, -1)
        hj = h.unsqueeze(0).expand(n_nodes, -1, -1)
        edge_logits = self.edge_mlp(torch.cat([hi, hj], dim=-1)).squeeze(-1)

        diag_mask = torch.eye(n_nodes, device=device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(diag_mask, -1e9)
        edge_logits = torch.triu(edge_logits, diagonal=1)

        return service_logits, op_logits, duration_pred, edge_logits


# -----------------------------
# Losses
# -----------------------------
def compute_class_weights(dataset, max_ratio: float = 10.0):
    svc_counter = Counter()
    op_counter = Counter()
    for g in dataset:
        x = g.x
        svc_counter.update(x[:, 0].long().tolist())
        op_counter.update(x[:, 1].long().tolist())

    def _weights(counter):
        num_classes = max(counter.keys()) + 1
        freq = torch.ones(num_classes, dtype=torch.float)
        for k, v in counter.items():
            freq[k] = v
        inv = 1.0 / (freq + 1e-8)
        inv = inv / inv.mean()
        inv = torch.clamp(inv, max=inv.median() * max_ratio)
        return inv

    return _weights(svc_counter), _weights(op_counter)


def focal_ce(logits: torch.Tensor, target: torch.Tensor, weight=None, gamma: float = 1.5):
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def node_recon_loss(real_x, service_logits, op_logits, duration_pred, duration_mean, duration_std,
                    service_weight=None, op_weight=None, op_loss_scale: float = 1.0):
    svc = real_x[:, 0].long()
    op = real_x[:, 1].long()
    duration_real = real_x[:, 2]

    svc_loss = focal_ce(service_logits, svc, weight=service_weight, gamma=1.5)
    op_loss = focal_ce(op_logits, op, weight=op_weight, gamma=1.5) * op_loss_scale

    target_duration = _log_normalize_duration(duration_real, duration_mean, duration_std)
    dur_loss = F.l1_loss(duration_pred.squeeze(-1), target_duration)

    return svc_loss + op_loss + dur_loss


def edge_recon_loss(real_edges: torch.Tensor, pred_edge_logits: torch.Tensor, n_nodes: int, device: torch.device,
                    neg_weight: float = 1.5, pos_max_ratio: float = 10.0) -> torch.Tensor:
    if n_nodes <= 1:
        return torch.tensor(0.0, device=device)
    mask = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1).bool()
    target = torch.zeros((n_nodes, n_nodes), device=device)
    if real_edges.numel() > 0:
        target[real_edges[0], real_edges[1]] = 1.0
    target = target[mask]
    logits = pred_edge_logits[mask].clamp(min=-30.0, max=30.0)

    num_pos = target.sum()
    num_total = target.numel()
    num_neg = num_total - num_pos

    if num_total == 0:
        return torch.tensor(0.0, device=device)

    if num_pos == 0:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        imbalance = (num_neg / (num_pos + 1e-6)).item()
        pos_weight = torch.tensor(min(max(imbalance, 1.0), pos_max_ratio), device=device)

    weight = torch.ones_like(target)
    weight[target == 0] = neg_weight
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight, pos_weight=pos_weight)


def total_recon_loss(real_data,
                     service_logits, op_logits, duration_pred,
                     pred_edge_logits,
                     duration_mean, duration_std,
                     service_weight=None, op_weight=None, op_loss_scale: float = 1.0,
                     edge_weight: float = 2.0):
    node_l = node_recon_loss(real_data.x, service_logits, op_logits, duration_pred, duration_mean, duration_std,
                             service_weight=service_weight, op_weight=op_weight, op_loss_scale=op_loss_scale)
    edge_l = edge_recon_loss(real_data.edge_index, pred_edge_logits, real_data.num_nodes, real_data.x.device)
    loss = node_l + edge_weight * edge_l
    return loss, node_l, edge_l, torch.tensor(0.0, device=real_data.x.device)


@torch.no_grad()
def decode_and_denorm_duration(duration_pred: torch.Tensor, duration_mean: torch.Tensor, duration_std: torch.Tensor) -> torch.Tensor:
    return _log_denormalize_duration(duration_pred.squeeze(-1), duration_mean, duration_std)


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_reconstruction(dataset, encoder, decoder, device, edge_threshold: float = 0.7):
    encoder.eval()
    decoder.eval()

    total_nodes = correct_service = correct_op = 0
    duration_abs_err = duration_log_abs_err = duration_rel_err = 0.0
    edge_TP = edge_FP = edge_FN = 0

    for data in DataLoader(dataset, batch_size=1, shuffle=False):
        data = data.to(device)
        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

        enc_out = encoder(data.x, data.edge_index, data.batch)
        g_emb = enc_out["graph"]
        n_emb = enc_out["node"]
        h_enc = enc_out["node_h"]

        svc_logits, op_logits, dur_pred, edge_logits = decoder.decode_for_training(g_emb[0], n_emb, h_enc)

        svc_pred = svc_logits.argmax(dim=1)
        op_pred = op_logits.argmax(dim=1)
        dur_denorm = decode_and_denorm_duration(dur_pred, encoder.duration_mean, encoder.duration_std)

        svc_real = data.x[:, 0].long()
        op_real = data.x[:, 1].long()
        dur_real = data.x[:, 2]

        total_nodes += data.num_nodes
        correct_service += (svc_pred == svc_real).sum().item()
        correct_op += (op_pred == op_real).sum().item()
        duration_abs_err += torch.abs(dur_denorm - dur_real).sum().item()
        target_log = _log_normalize_duration(dur_real, encoder.duration_mean, encoder.duration_std)
        duration_log_abs_err += torch.abs(dur_pred.squeeze(-1) - target_log).sum().item()
        duration_rel_err += (torch.abs(dur_denorm - dur_real) / (torch.abs(dur_real) + 1e-6)).sum().item()

        adj_real = torch.zeros((data.num_nodes, data.num_nodes), device=device)
        if data.edge_index.numel() > 0:
            adj_real[data.edge_index[0], data.edge_index[1]] = 1.0
        adj_pred = (torch.sigmoid(edge_logits) > edge_threshold).float()
        mask = torch.triu(torch.ones(data.num_nodes, data.num_nodes, device=device), diagonal=1).bool()
        real_vec = adj_real[mask]
        pred_vec = adj_pred[mask]
        TP = ((pred_vec == 1) & (real_vec == 1)).sum().item()
        FP = ((pred_vec == 1) & (real_vec == 0)).sum().item()
        FN = ((pred_vec == 0) & (real_vec == 1)).sum().item()
        edge_TP += TP
        edge_FP += FP
        edge_FN += FN

    service_acc = correct_service / total_nodes if total_nodes else 0.0
    op_acc = correct_op / total_nodes if total_nodes else 0.0
    dur_mae = duration_abs_err / total_nodes if total_nodes else 0.0
    dur_log_mae = duration_log_abs_err / total_nodes if total_nodes else 0.0
    dur_mape = duration_rel_err / total_nodes if total_nodes else 0.0
    precision = edge_TP / (edge_TP + edge_FP) if (edge_TP + edge_FP) > 0 else 0.0
    recall = edge_TP / (edge_TP + edge_FN) if (edge_TP + edge_FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "total_nodes": total_nodes,
        "service_acc": service_acc,
        "op_acc": op_acc,
        "dur_mae": dur_mae,
        "dur_log_mae": dur_log_mae,
        "dur_mape": dur_mape,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
    }


# -----------------------------
# Training
# -----------------------------
def _ensure_batch(data):
    if not hasattr(data, "batch"):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
    return data


def train_epoch(encoder, decoder, loader, optimizer, device, op_loss_scale: float = 1.0,
                edge_weight: float = 2.0):
    encoder.train()
    decoder.train()
    total_loss = total_node = total_edge = total_kl = 0.0

    for data in loader:
        data = data.to(device)
        data = _ensure_batch(data)
        optimizer.zero_grad()

        enc_out = encoder(data.x, data.edge_index, data.batch)
        g_emb = enc_out["graph"]
        n_emb = enc_out["node"]

        svc_logits, op_logits, dur_pred, edge_logits = decoder.decode_for_training(g_emb[0], n_emb, enc_out["node_h"])
        loss, node_l, edge_l, kl_l = total_recon_loss(
            data,
            svc_logits,
            op_logits,
            dur_pred,
            edge_logits,
            encoder.duration_mean,
            encoder.duration_std,
            service_weight=getattr(encoder, "service_weight", None),
            op_weight=getattr(encoder, "op_weight", None),
            op_loss_scale=op_loss_scale,
            edge_weight=edge_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.5)
        optimizer.step()

        total_loss += loss.item()
        total_node += node_l.item()
        total_edge += edge_l.item()
        total_kl += kl_l.item()

    n = len(loader)
    return total_loss / n, total_node / n, total_edge / n, total_kl / n


def run_training(dataset, encoder, decoder, device, epochs: int = 100, batch_size: int = 4, lr: float = 3e-4,
                 use_class_weights: bool = True, op_loss_scale: float = 2.0,
                 edge_weight: float = 2.0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=5e-5)

    if use_class_weights:
        svc_w_cpu, op_w_cpu = compute_class_weights(dataset)
        encoder.service_weight = svc_w_cpu.to(device)
        encoder.op_weight = op_w_cpu.to(device)
    else:
        encoder.service_weight = encoder.op_weight = None

    for epoch in range(1, epochs + 1):
        loss, node_l, edge_l, kl_l = train_epoch(encoder, decoder, loader, optimizer, device,
                                                 op_loss_scale=op_loss_scale, edge_weight=edge_weight)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | node {node_l:.4f} | edge {edge_l:.4f} | kl {kl_l:.4f}")
