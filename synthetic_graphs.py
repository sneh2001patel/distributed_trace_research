import random
from typing import List, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from vae import GNNDecoder, _log_denormalize_duration


def load_decoder(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    cfg = checkpoint["config"]
    dec = GNNDecoder(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_service_classes=cfg["num_services"],
        n_op_classes=cfg["num_ops"],
        encoder_hidden_dim=cfg.get("encoder_hidden_dim", cfg["hidden_dim"]),
        max_nodes=cfg.get("max_nodes", 256),
        head_hidden_dim=cfg.get("head_hidden_dim", cfg["hidden_dim"] * 2),
        head_dropout=cfg.get("head_dropout", 0.3),
        enc_h_dropout=cfg.get("enc_h_dropout", 0.0),
    ).to(device)
    dec.load_state_dict(checkpoint["decoder_state"])
    dec.eval()
    return dec, cfg


@torch.no_grad()
def generate_synthetic_graph(
    decoder: GNNDecoder,
    num_nodes: int,
    duration_mean: float,
    duration_std: float,
    device: torch.device,
    edge_threshold: float = 0.9,
    sample_edges: bool = False,
    sample_nodes: bool = False,
    node_temperature: float = 1.0,
    edge_dropout: float = 0.0,
    duration_noise: float = 0.0,
    threshold_jitter: float = 0.0,
    target_edge_count: int = None,
    valid_ops_by_service: dict[int, torch.Tensor] = None,
) -> Data:
    z_g = torch.randn(decoder.latent_dim, device=device)
    z_n = torch.randn(num_nodes, decoder.latent_dim, device=device)
    enc_h = torch.zeros(num_nodes, decoder.encoder_hidden_dim, device=device)

    service_logits, op_logits, duration_pred, edge_logits = decoder.decode_for_training(
        z_g, z_n, enc_h
    )

    if sample_nodes:
        temp = max(float(node_temperature), 1e-6)
        service_probs = torch.softmax(service_logits / temp, dim=1)
        service_ids = torch.multinomial(service_probs, 1).squeeze(1)
    else:
        service_ids = service_logits.argmax(dim=1)

    if valid_ops_by_service:
        op_logits = op_logits.clone()
        for i, service_id in enumerate(service_ids.tolist()):
            valid_ops = valid_ops_by_service.get(int(service_id))
            if valid_ops is None or valid_ops.numel() == 0:
                continue
            mask = torch.ones(op_logits.size(1), device=device, dtype=torch.bool)
            mask[valid_ops.to(device).long()] = False
            op_logits[i, mask] = -1e9

    if sample_nodes:
        temp = max(float(node_temperature), 1e-6)
        op_probs = torch.softmax(op_logits / temp, dim=1)
        op_ids = torch.multinomial(op_probs, 1).squeeze(1)
    else:
        op_ids = op_logits.argmax(dim=1)

    duration = _log_denormalize_duration(
        duration_pred.squeeze(-1),
        torch.tensor(duration_mean, device=device),
        torch.tensor(duration_std, device=device),
    )
    if duration_noise > 0.0:
        duration = torch.clamp(
            duration + torch.randn_like(duration) * duration_noise, min=0.0
        )

    edge_probs = torch.sigmoid(edge_logits)
    if threshold_jitter > 0.0:
        edge_threshold = float(
            torch.clamp(
                torch.tensor(edge_threshold, device=device)
                + torch.randn((), device=device) * threshold_jitter,
                min=0.05,
                max=0.95,
            ).item()
        )
    mask = torch.triu(
        torch.ones(num_nodes, num_nodes, device=device), diagonal=1
    ).bool()
    if target_edge_count is not None:
        target_edge_count = max(0, int(target_edge_count))
        n_select = min(int((target_edge_count + 1) // 2), int(mask.sum().item()))
        edge_keep = torch.zeros_like(mask)
        if n_select > 0:
            scores = edge_probs.masked_fill(~mask, -1.0)
            flat_idx = torch.topk(scores.flatten(), k=n_select).indices
            edge_keep = edge_keep.flatten()
            edge_keep[flat_idx] = True
            edge_keep = edge_keep.view_as(mask)
    elif sample_edges:
        edge_keep = (torch.rand_like(edge_probs) < edge_probs) & mask
    else:
        edge_keep = (edge_probs > edge_threshold) & mask
    if edge_dropout > 0.0:
        keep_mask = torch.rand_like(edge_keep.float()) > edge_dropout
        edge_keep = edge_keep & keep_mask.bool()
    edge_index = edge_keep.nonzero(as_tuple=False).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    x = torch.stack([service_ids, op_ids, duration], dim=1).float()
    return Data(x=x, edge_index=edge_index)


def generate_synthetic_dataset(
    num_graphs: int,
    num_nodes: Tuple[int, int],
    weights_path: str = "./weights/tt_vae_weights.pt",
    device: torch.device = None,
    edge_threshold: float = 0.9,
    sample_edges: bool = False,
    sample_nodes: bool = False,
    edge_dropout: float = 0.0,
    duration_noise: float = 0.0,
    threshold_jitter: float = 0.0,
) -> List[Data]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder(weights_path, device)
    graphs: List[Data] = []
    for _ in range(num_graphs):
        if isinstance(num_nodes, tuple):
            n_nodes = random.randint(*num_nodes)
        else:
            n_nodes = num_nodes
        graphs.append(
            generate_synthetic_graph(
                decoder,
                n_nodes,
                cfg["duration_mean"],
                cfg["duration_std"],
                device,
                edge_threshold=edge_threshold,
                sample_edges=sample_edges,
                sample_nodes=sample_nodes,
                edge_dropout=edge_dropout,
                duration_noise=duration_noise,
                threshold_jitter=threshold_jitter,
            )
        )
    return graphs
