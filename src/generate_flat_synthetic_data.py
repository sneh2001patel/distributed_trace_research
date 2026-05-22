import argparse
import os
import random

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

from flat_vae import FlatGNNDecoder, denormalize_duration


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


def load_decoder(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    decoder = FlatGNNDecoder(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_service_classes=cfg["num_services"],
        n_op_classes=cfg["num_ops"],
        max_nodes=cfg.get("max_nodes", 256),
        head_hidden_dim=cfg.get("head_hidden_dim", cfg["hidden_dim"] * 2),
        head_dropout=cfg.get("head_dropout", 0.3),
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state"])
    decoder.eval()
    return decoder, cfg


def build_valid_ops_by_service(graphs):
    valid = {}
    for graph in graphs:
        x = graph.x.detach().cpu()
        for service_id, op_id in zip(x[:, 0].round().long(), x[:, 1].round().long()):
            valid.setdefault(int(service_id.item()), set()).add(int(op_id.item()))
    return {
        sid: torch.tensor(sorted(ops), dtype=torch.long)
        for sid, ops in valid.items()
    }


@torch.no_grad()
def sample_graph(
    decoder,
    cfg,
    n_nodes: int,
    y_label: int,
    device: torch.device,
    edge_threshold: float = 0.9,
    sample_nodes: bool = True,
    sample_edges: bool = True,
    node_temperature: float = 1.0,
    valid_ops_by_service: dict = None,
):
    z = torch.randn(decoder.latent_dim, device=device)
    service_logits, op_logits, duration_pred, edge_logits = decoder.decode(z, n_nodes)

    if sample_nodes:
        temp = max(node_temperature, 1e-6)
        service_ids = torch.multinomial(
            torch.softmax(service_logits / temp, dim=1), 1
        ).squeeze(1)
    else:
        service_ids = service_logits.argmax(dim=1)

    if valid_ops_by_service:
        op_logits = op_logits.clone()
        for i, sid in enumerate(service_ids.tolist()):
            valid_ops = valid_ops_by_service.get(int(sid))
            if valid_ops is None or valid_ops.numel() == 0:
                continue
            mask = torch.ones(op_logits.size(1), device=device, dtype=torch.bool)
            mask[valid_ops.to(device).long()] = False
            op_logits[i, mask] = -1e9

    if sample_nodes:
        temp = max(node_temperature, 1e-6)
        op_ids = torch.multinomial(torch.softmax(op_logits / temp, dim=1), 1).squeeze(1)
    else:
        op_ids = op_logits.argmax(dim=1)

    durations = denormalize_duration(
        duration_pred,
        torch.tensor(cfg["duration_mean"], device=device),
        torch.tensor(cfg["duration_std"], device=device),
    )

    edge_probs = torch.sigmoid(edge_logits)
    mask = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1).bool()
    if sample_edges:
        edge_keep = (torch.rand_like(edge_probs) < edge_probs) & mask
    else:
        edge_keep = (edge_probs > edge_threshold) & mask
    edge_index = edge_keep.nonzero(as_tuple=False).t().contiguous()

    x = torch.stack([service_ids.float(), op_ids.float(), durations.float()], dim=1)
    return Data(
        x=x.cpu(),
        edge_index=edge_index.cpu(),
        y=torch.tensor([y_label], dtype=torch.long),
    )


def generate_dataset(args):
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder(args.weights_path, device)
    real_ds = LoadDataset(args.real_path)
    all_graphs = [real_ds.get(i) for i in range(len(real_ds))]
    # Only sample node counts from graphs the VAE was trained on.
    real_graphs = [g for g in all_graphs if g.num_nodes <= args.max_nodes]
    if not real_graphs:
        raise RuntimeError(f"No graphs with <= {args.max_nodes} nodes found in {args.real_path}")
    node_counts = [g.num_nodes for g in real_graphs]
    valid_ops_by_service = build_valid_ops_by_service(real_graphs)
    target = args.target_count or len(node_counts)

    if target <= len(node_counts):
        sampled_counts = rng.sample(node_counts, target)
    else:
        sampled_counts = [rng.choice(node_counts) for _ in range(target)]

    graphs = [
        sample_graph(
            decoder,
            cfg,
            n_nodes=n,
            y_label=args.y_label,
            device=device,
            edge_threshold=args.edge_threshold,
            sample_nodes=args.sample_nodes,
            sample_edges=args.sample_edges,
            node_temperature=args.node_temperature,
            valid_ops_by_service=valid_ops_by_service,
        )
        for n in sampled_counts
    ]
    data, slices = InMemoryDataset.collate(graphs)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save((data, slices), args.out_path)
    print(f"saved {args.out_path} ({len(graphs)} graphs)")


_SYSTEM_JOBS = {
    "TT": [
        {
            "name": "normal",
            "real_path": "./datasets/anomaly/TT/TT_normal.pt",
            "weights_path": "./weights/tt_normal_flat_vae_weights.pt",
            "out_path": "./datasets/baselines/TT_normal_flat_vae_synthetic.pt",
            "y_label": 0,
        },
        {
            "name": "abnormal",
            "real_path": "./datasets/anomaly/TT/TT_abnormal.pt",
            "weights_path": "./weights/tt_abnormal_flat_vae_weights.pt",
            "out_path": "./datasets/baselines/TT_abnormal_flat_vae_synthetic.pt",
            "y_label": 1,
        },
    ],
    "SN": [
        {
            "name": "normal",
            "real_path": "./datasets/anomaly/SN/SN_normal.pt",
            "weights_path": "./weights/sn_normal_flat_vae_weights.pt",
            "out_path": "./datasets/baselines/SN_normal_flat_vae_synthetic.pt",
            "y_label": 0,
        },
        {
            "name": "abnormal",
            "real_path": "./datasets/anomaly/SN/SN_abnormal.pt",
            "weights_path": "./weights/sn_abnormal_flat_vae_weights.pt",
            "out_path": "./datasets/baselines/SN_abnormal_flat_vae_synthetic.pt",
            "y_label": 1,
        },
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic traces from a one-latent VAE baseline."
    )
    parser.add_argument("--system", choices=["SN", "TT", "both"], default=None,
                        help="Convenience mode: auto-fills paths for SN and/or TT.")
    parser.add_argument("--mode", choices=["normal", "abnormal", "both"], default="both",
                        help="Class split to generate when using --system.")
    parser.add_argument("--real-path", default=None)
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--y-label", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=30,
                        help="Only sample node counts from real graphs with <= this many nodes (must match --max-graph-nodes used during training).")
    parser.add_argument("--edge-threshold", type=float, default=0.9)
    parser.add_argument("--node-temperature", type=float, default=1.0)
    parser.add_argument("--sample-nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.system is None:
        if args.real_path is None or args.weights_path is None or args.out_path is None or args.y_label is None:
            parser.error("Either pass --system, or pass --real-path, --weights-path, --out-path, and --y-label.")
        generate_dataset(args)
        return

    systems = ["SN", "TT"] if args.system == "both" else [args.system]
    for system in systems:
        for job in _SYSTEM_JOBS[system]:
            if args.mode != "both" and job["name"] != args.mode:
                continue
            print(f"\n=== BUILDING {system} {job['name'].upper()} FLAT VAE DATASET ===")
            job_args = argparse.Namespace(**vars(args))
            job_args.real_path = job["real_path"]
            job_args.weights_path = job["weights_path"]
            job_args.out_path = job["out_path"]
            job_args.y_label = job["y_label"]
            generate_dataset(job_args)


if __name__ == "__main__":
    main()
