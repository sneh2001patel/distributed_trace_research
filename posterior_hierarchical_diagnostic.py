import argparse
import os
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.utils import to_undirected

from generate_tt_synthetic_data import (
    build_valid_ops_by_service as build_tt_valid_ops,
    generate_tt_synthetic_graph,
    load_decoder_from_checkpoint as load_tt_decoder,
)
from synthetic_graphs import generate_synthetic_graph
from vae import GNNDecoder, GraphEncoderVAE
from vae_tt import TTGNNDecoder


ROOT = Path(__file__).resolve().parent


class PTDataset(InMemoryDataset):
    def __init__(self, datapath: str, undirected: bool = False) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        self.data, self.slices = torch.load(datapath, weights_only=False)
        self.undirected = undirected

    def get(self, idx):
        data = super().get(idx)
        if self.undirected:
            data.edge_index = to_undirected(data.edge_index)
        return data


def load_encoder(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    cfg = checkpoint["config"]
    encoder = GraphEncoderVAE(
        n_services=cfg["num_services"],
        n_ops=cfg["num_ops"],
        duration_mean=cfg["duration_mean"],
        duration_std=cfg["duration_std"],
        hidden_ch=cfg.get("encoder_hidden_dim", cfg["hidden_dim"]),
        embed_dim=cfg.get("embed_dim", 48),
        latent_dim=cfg["latent_dim"],
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    encoder.eval()
    return encoder, cfg


def load_sn_decoder(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    cfg = checkpoint["config"]
    state = checkpoint["decoder_state"]
    max_nodes = cfg.get("max_nodes", int(state["pos_emb.weight"].shape[0]))
    head_hidden_dim = cfg.get("head_hidden_dim", int(state["service_head.0.weight"].shape[0]))
    decoder = GNNDecoder(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_service_classes=cfg["num_services"],
        n_op_classes=cfg["num_ops"],
        encoder_hidden_dim=cfg.get("encoder_hidden_dim", cfg["hidden_dim"]),
        max_nodes=max_nodes,
        head_hidden_dim=head_hidden_dim,
        head_dropout=cfg.get("head_dropout", 0.1),
        enc_h_dropout=cfg.get("enc_h_dropout", 0.0),
    ).to(device)
    decoder.load_state_dict(state)
    decoder.eval()
    return decoder, cfg


def build_valid_ops(graphs):
    valid = {}
    for graph in graphs:
        x = graph.x.detach().cpu()
        for service_id, op_id in zip(x[:, 0].round().long(), x[:, 1].round().long()):
            valid.setdefault(int(service_id.item()), set()).add(int(op_id.item()))
    return {
        service_id: torch.tensor(sorted(op_ids), dtype=torch.long)
        for service_id, op_ids in valid.items()
    }


def _sample_latent(mu, logvar, scale: float):
    if scale <= 0.0:
        return mu
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std * scale


@torch.no_grad()
def generate_sn_posterior_graph(
    decoder: GNNDecoder,
    cfg,
    real_graph,
    z_g,
    z_n,
    enc_h,
    use_enc_h: bool,
    valid_ops_by_service,
):
    num_nodes = real_graph.num_nodes
    if not use_enc_h:
        enc_h = torch.zeros_like(enc_h)
    service_logits, op_logits, duration_pred, edge_logits = decoder.decode_for_training(
        z_g, z_n, enc_h
    )
    service_ids = service_logits.argmax(dim=1)
    if valid_ops_by_service:
        op_logits = op_logits.clone()
        for i, service_id in enumerate(service_ids.tolist()):
            valid_ops = valid_ops_by_service.get(int(service_id))
            if valid_ops is None or valid_ops.numel() == 0:
                continue
            mask = torch.ones(op_logits.size(1), device=op_logits.device, dtype=torch.bool)
            mask[valid_ops.to(op_logits.device).long()] = False
            op_logits[i, mask] = -1e9
    op_ids = op_logits.argmax(dim=1)
    duration = torch.expm1(
        duration_pred.squeeze(-1) * (cfg["duration_std"] + 1e-9) + cfg["duration_mean"]
    ).clamp(min=0)

    target_edges = int(real_graph.edge_index.size(1))
    mask = torch.triu(torch.ones(num_nodes, num_nodes, device=edge_logits.device), diagonal=1).bool()
    n_select = min(int((target_edges + 1) // 2), int(mask.sum().item()))
    edge_keep = torch.zeros_like(mask)
    if n_select > 0:
        scores = torch.sigmoid(edge_logits).masked_fill(~mask, -1.0)
        idx = torch.topk(scores.flatten(), k=n_select).indices
        edge_keep = edge_keep.flatten()
        edge_keep[idx] = True
        edge_keep = edge_keep.view_as(mask)
    edge_index = to_undirected(edge_keep.nonzero(as_tuple=False).t(), num_nodes=num_nodes)
    x = torch.stack([service_ids.float(), op_ids.float(), duration.float()], dim=1)
    return Data(x=x.cpu(), edge_index=edge_index.cpu(), y=real_graph.y.cpu())


@torch.no_grad()
def generate_tt_posterior_graph(
    decoder: TTGNNDecoder,
    cfg,
    real_graph,
    z_g,
    z_n,
    enc_h,
    use_enc_h: bool,
    valid_ops_by_service,
):
    if not use_enc_h:
        enc_h = torch.zeros_like(enc_h)
    service_logits, op_logits, duration_pred, parent_logits = decoder.decode_for_training(
        z_g, z_n, enc_h
    )
    service_ids = service_logits.argmax(dim=1)
    if valid_ops_by_service:
        op_logits = op_logits.clone()
        for i, service_id in enumerate(service_ids.tolist()):
            valid_ops = valid_ops_by_service.get(int(service_id))
            if valid_ops is None or valid_ops.numel() == 0:
                continue
            mask = torch.ones(op_logits.size(1), device=op_logits.device, dtype=torch.bool)
            mask[valid_ops.to(op_logits.device).long()] = False
            op_logits[i, mask] = -1e9
    op_ids = op_logits.argmax(dim=1)
    duration = torch.expm1(
        duration_pred.squeeze(-1) * (cfg["duration_std"] + 1e-9) + cfg["duration_mean"]
    ).clamp(min=0)

    num_nodes = real_graph.num_nodes
    target_edges = min(int(real_graph.edge_index.size(1)), max(num_nodes - 1, 0))
    nonroot = parent_logits[:, :num_nodes].clone()
    nonroot[torch.arange(num_nodes, device=nonroot.device), torch.arange(num_nodes, device=nonroot.device)] = -1e9
    best_scores, best_parents = nonroot.max(dim=1)
    edges = []
    for child in torch.argsort(best_scores, descending=True).tolist():
        parent = int(best_parents[child].item())
        if parent < num_nodes and parent != child:
            edges.append([parent, child])
        if len(edges) >= target_edges:
            break
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    x = torch.stack([service_ids.float(), op_ids.float(), duration.float()], dim=1)
    return Data(x=x.cpu(), edge_index=edge_index.cpu(), y=real_graph.y.cpu())


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    undirected = args.system == "SN"
    ds = PTDataset(args.real_path, undirected=undirected)
    real_graphs = [ds.get(i).cpu() for i in range(len(ds))]
    if args.max_graphs is not None:
        real_graphs = real_graphs[: args.max_graphs]

    encoder, cfg = load_encoder(args.weights_path, device)
    if args.system == "TT":
        decoder, _ = load_tt_decoder(args.weights_path, device)
        valid_ops = build_tt_valid_ops(real_graphs)
    else:
        decoder, _ = load_sn_decoder(args.weights_path, device)
        valid_ops = build_valid_ops(real_graphs)

    generated = []
    for graph in real_graphs:
        data = graph.to(device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        enc_out = encoder(data.x, data.edge_index, batch)
        mu_g, logvar_g, _ = enc_out["graph"]
        mu_n, logvar_n, _ = enc_out["node"]
        z_g = _sample_latent(mu_g[0], logvar_g[0], args.posterior_scale)
        z_n = _sample_latent(mu_n, logvar_n, args.posterior_scale)
        if args.system == "TT":
            out = generate_tt_posterior_graph(
                decoder, cfg, graph, z_g, z_n, enc_out["node_h"], args.use_enc_h, valid_ops
            )
        else:
            out = generate_sn_posterior_graph(
                decoder, cfg, graph, z_g, z_n, enc_out["node_h"], args.use_enc_h, valid_ops
            )
        generated.append(out)

    data, slices = InMemoryDataset.collate(generated)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save((data, slices), args.out_path)
    print(f"saved {args.out_path} ({len(generated)} graphs)")


def main():
    parser = argparse.ArgumentParser(description="Posterior-sampling diagnostic for hierarchical VAE generation.")
    parser.add_argument("--system", choices=["TT", "SN"], required=True)
    parser.add_argument("--real-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--posterior-scale", type=float, default=1.0)
    parser.add_argument("--use-enc-h", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-graphs", type=int, default=1000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
