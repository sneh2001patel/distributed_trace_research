import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from vae import (
    GraphEncoderVAE,
    _ensure_batch,
    compute_class_weights,
    decode_and_denorm_duration,
    kl_loss,
    node_recon_loss,
)


class TTGNNDecoder(nn.Module):
    """
    TT-specific decoder:
    - lighter hidden state than the generic decoder
    - more discriminative edge scorer using pairwise interactions
    - intended for the smaller, sparser TT anomaly graphs
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_service_classes: int,
        n_op_classes: int,
        encoder_hidden_dim: int = 128,
        num_gnn_layers: int = 1,
        max_nodes: int = 64,
        head_hidden_dim: int = None,
        head_dropout: float = 0.1,
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
        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden_dim, n_op_classes),
        )
        self.duration_head = nn.Linear(hidden_dim, 1)

        edge_in = hidden_dim * 4
        self.parent_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.root_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_dense_directed_edge_index(self, n_nodes: int, device: torch.device):
        row = torch.arange(n_nodes, device=device).repeat_interleave(n_nodes)
        col = torch.arange(n_nodes, device=device).repeat(n_nodes)
        mask = row != col
        return torch.stack([row[mask], col[mask]], dim=0)

    def _ensure_pos_capacity(self, n_nodes: int, device: torch.device):
        if n_nodes <= self.pos_emb.num_embeddings:
            return
        new_size = max(n_nodes, self.pos_emb.num_embeddings * 2)
        new_emb = nn.Embedding(new_size, self.hidden_dim, device=device)
        with torch.no_grad():
            new_emb.weight[: self.pos_emb.num_embeddings].copy_(self.pos_emb.weight)
            nn.init.normal_(new_emb.weight[self.pos_emb.num_embeddings :], std=0.02)
        self.pos_emb = new_emb

    def decode_for_training(
        self, z_graph: torch.Tensor, z_nodes: torch.Tensor, enc_h: torch.Tensor
    ):
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

        edge_index = self._build_dense_directed_edge_index(n_nodes, device)
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = h + h_res

        h_cls = self.fuse(torch.cat([h, enc_h], dim=1))
        service_logits = self.service_head(h_cls)
        op_logits = self.op_head(h_cls)
        duration_pred = self.duration_head(h_cls)

        hi = h.unsqueeze(1).expand(-1, n_nodes, -1)
        hj = h.unsqueeze(0).expand(n_nodes, -1, -1)
        parent_features = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)
        parent_scores = self.parent_mlp(parent_features).squeeze(-1)  # [src, dst]
        parent_logits = parent_scores.transpose(0, 1).contiguous()  # [dst, src]

        diag_mask = torch.eye(n_nodes, device=device, dtype=torch.bool)
        parent_logits = parent_logits.masked_fill(diag_mask, -1e9)
        root_logits = self.root_head(h_cls).squeeze(-1).unsqueeze(1)
        parent_logits = torch.cat([parent_logits, root_logits], dim=1)
        return service_logits, op_logits, duration_pred, parent_logits


def _build_parent_targets(
    real_edges: torch.Tensor,
    n_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    target = torch.full((n_nodes,), n_nodes, dtype=torch.long, device=device)
    if real_edges.numel() == 0:
        return target

    uniq_edges = torch.unique(real_edges.t(), dim=0)
    for src, dst in uniq_edges.tolist():
        if 0 <= dst < n_nodes and target[dst].item() == n_nodes:
            target[dst] = src
    return target


def parent_recon_loss(
    real_edges: torch.Tensor,
    pred_parent_logits: torch.Tensor,
    n_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    target = _build_parent_targets(real_edges, n_nodes, device)
    return F.cross_entropy(pred_parent_logits, target)


def parent_rank_loss(
    real_edges: torch.Tensor,
    pred_parent_logits: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    max_nodes_sample: int = 64,
    margin: float = 0.2,
) -> torch.Tensor:
    target = _build_parent_targets(real_edges, n_nodes, device)
    sampled_pos = []
    sampled_neg = []

    for child in range(n_nodes):
        true_parent = int(target[child].item())
        candidates = torch.arange(n_nodes + 1, device=device)
        neg_candidates = candidates[candidates != true_parent]
        if neg_candidates.numel() == 0:
            continue
        neg_choice = neg_candidates[
            torch.randint(0, neg_candidates.numel(), (1,), device=device)
        ]
        sampled_pos.append(pred_parent_logits[child, true_parent])
        sampled_neg.append(pred_parent_logits[child, neg_choice].squeeze(0))
        if len(sampled_pos) >= max_nodes_sample:
            break

    if not sampled_pos:
        return torch.tensor(0.0, device=device)

    pos = torch.stack(sampled_pos)
    neg = torch.stack(sampled_neg)
    return F.relu(margin - (pos - neg)).mean()


def total_tt_vae_loss(
    real_data,
    service_logits,
    op_logits,
    duration_pred,
    pred_parent_logits,
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
    edge_weight: float = 2.0,
    edge_rank_weight: float = 0.0,
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
    edge_l = parent_recon_loss(
        real_data.edge_index,
        pred_parent_logits,
        real_data.num_nodes,
        real_data.x.device,
    )
    edge_rank_l = parent_rank_loss(
        real_data.edge_index,
        pred_parent_logits,
        real_data.num_nodes,
        real_data.x.device,
    )
    kl_g = kl_loss(mu_g, logvar_g)
    kl_n = kl_loss(mu_n, logvar_n)
    loss = node_l + edge_weight * edge_l + edge_rank_weight * edge_rank_l + beta * (
        kl_g + 0.1 * kl_n
    )
    return loss, node_l, edge_l, kl_g + 0.1 * kl_n


def _decode_tree_edges(edge_logits: torch.Tensor):
    n_nodes = edge_logits.size(0)
    if n_nodes <= 1:
        return torch.zeros_like(edge_logits)

    probs = torch.sigmoid(edge_logits)
    mask = ~torch.eye(n_nodes, device=edge_logits.device, dtype=torch.bool)
    candidate_indices = mask.nonzero(as_tuple=False)
    candidate_scores = probs[mask]
    order = torch.argsort(candidate_scores, descending=True)

    parent_of = [-1] * n_nodes
    uf_parent = list(range(n_nodes))

    def find(x):
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        uf_parent[rb] = ra
        return True

    chosen = []
    for idx in order.tolist():
        src = int(candidate_indices[idx, 0].item())
        dst = int(candidate_indices[idx, 1].item())
        if parent_of[dst] != -1:
            continue
        if not union(src, dst):
            continue
        parent_of[dst] = src
        chosen.append((src, dst))
        if len(chosen) == n_nodes - 1:
            break

    if len(chosen) < n_nodes - 1:
        for idx in order.tolist():
            src = int(candidate_indices[idx, 0].item())
            dst = int(candidate_indices[idx, 1].item())
            if parent_of[dst] != -1:
                continue
            parent_of[dst] = src
            chosen.append((src, dst))
            if len(chosen) == n_nodes - 1:
                break

    adj_pred = torch.zeros_like(edge_logits)
    for src, dst in chosen:
        adj_pred[src, dst] = 1.0
    return adj_pred


def _parent_logits_to_adjacency(parent_logits: torch.Tensor) -> torch.Tensor:
    n_nodes = parent_logits.size(0)
    adj_pred = torch.zeros((n_nodes, n_nodes), device=parent_logits.device)
    pred_parent = parent_logits.argmax(dim=1)
    for child, parent in enumerate(pred_parent.tolist()):
        if parent < n_nodes:
            adj_pred[parent, child] = 1.0
    return adj_pred


@torch.no_grad()
def evaluate_reconstruction(
    dataset,
    encoder,
    decoder,
    device,
    edge_threshold: float = 0.75,
    edge_decode_mode: str = "parent",
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

        service_logits, op_logits, duration_pred, parent_logits = (
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
        duration_abs_err += torch.abs(dur_pred - dur_real).sum().item()

        adj_real = torch.zeros((data.num_nodes, data.num_nodes), device=device)
        if data.edge_index.numel() > 0:
            adj_real[data.edge_index[0], data.edge_index[1]] = 1.0

        adj_pred = _parent_logits_to_adjacency(parent_logits)
        mask = ~torch.eye(data.num_nodes, data.num_nodes, device=device, dtype=torch.bool)

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
        "edge_decode_mode": "parent",
        "edge_threshold": None,
    }


def train_vae_epoch(
    encoder,
    decoder,
    loader,
    optimizer,
    device,
    beta: float,
    op_loss_scale: float = 1.0,
    edge_weight: float = 2.0,
    edge_rank_weight: float = 0.0,
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
        service_logits, op_logits, duration_pred, parent_logits = (
            decoder.decode_for_training(z_g[0], z_n, enc_out["node_h"])
        )

        loss, node_l, edge_l, kl_l = total_tt_vae_loss(
            data,
            service_logits,
            op_logits,
            duration_pred,
            parent_logits,
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
            edge_rank_weight=edge_rank_weight,
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
    batch_size: int = 4,
    lr: float = 3e-4,
    beta_final: float = 0.01,
    use_class_weights: bool = True,
    op_loss_scale: float = 1.0,
    edge_weight: float = 2.0,
    edge_rank_weight: float = 0.0,
    class_weight_max_ratio: float = 5.0,
    val_dataset=None,
    early_stopping_patience: int = None,
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
    best_encoder_state = None
    best_decoder_state = None
    best_val_f1 = None
    stale_epochs = 0
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
            edge_rank_weight,
        )
        if val_dataset is not None:
            val_metrics = evaluate_reconstruction(val_dataset, encoder, decoder, device)
            val_edge_f1 = val_metrics["edge_f1"]
            improved = best_val_f1 is None or val_edge_f1 > best_val_f1 + 1e-6
            if improved:
                best_val_f1 = val_edge_f1
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_decoder_state = copy.deepcopy(decoder.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | node {node_l:.4f} | edge {edge_l:.4f} | "
                f"kl {kl_l:.4f} | beta {beta:.4f} | val_edge_f1 {val_edge_f1:.4f}"
            )
            if (
                early_stopping_patience is not None
                and stale_epochs >= early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch:03d} after "
                    f"{early_stopping_patience} epochs without validation edge F1 improvement."
                )
                break
        else:
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | node {node_l:.4f} | edge {edge_l:.4f} | kl {kl_l:.4f} | beta {beta:.4f}"
            )

    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)

    return {
        "best_val_edge_f1": best_val_f1,
        "used_validation": val_dataset is not None,
    }


__all__ = [
    "GraphEncoderVAE",
    "TTGNNDecoder",
    "evaluate_reconstruction",
    "run_training",
]
