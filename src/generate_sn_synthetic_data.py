import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr

from synthetic_graphs import generate_synthetic_graph
from vae import GNNDecoder, GraphEncoderVAE


ROOT = Path(__file__).resolve().parent


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


def load_encoder_decoder_from_checkpoint(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    cfg = checkpoint["config"]
    dec_state = checkpoint["decoder_state"]

    max_nodes = cfg.get("max_nodes")
    if max_nodes is None:
        max_nodes = int(dec_state["pos_emb.weight"].shape[0])

    head_hidden_dim = cfg.get("head_hidden_dim")
    if head_hidden_dim is None:
        head_hidden_dim = int(dec_state["service_head.0.weight"].shape[0])

    dec = GNNDecoder(
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
    dec.load_state_dict(dec_state)
    dec.eval()

    enc = GraphEncoderVAE(
        n_services=cfg["num_services"],
        n_ops=cfg["num_ops"],
        duration_mean=cfg["duration_mean"],
        duration_std=cfg["duration_std"],
        hidden_ch=cfg.get("hidden_ch", 128),
        embed_dim=cfg.get("embed_dim", 48),
        latent_dim=cfg["latent_dim"],
    ).to(device)
    enc.load_state_dict(checkpoint["encoder_state"], strict=False)
    enc.eval()

    return enc, dec, cfg


# Keep old name as alias so synthetic_graphs.py callers still work
def load_decoder_from_checkpoint(weights_path: str, device: torch.device):
    _, dec, cfg = load_encoder_decoder_from_checkpoint(weights_path, device)
    return dec, cfg


@torch.no_grad()
def build_latent_pool(encoder, graphs, device: torch.device):
    """
    Encode every training graph and store per-graph posteriors, indexed by node
    count for fast lookup.  Sampling z_n from a graph with a matching node count
    preserves per-node service/op identity instead of averaging it away.
    """
    by_n = {}
    for graph in graphs:
        graph = graph.to(device)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        enc_out = encoder(graph.x, graph.edge_index, batch)
        mu_g, logvar_g, _ = enc_out["graph"]
        mu_n, logvar_n, _ = enc_out["node"]
        entry = {
            "mu_g":     mu_g.squeeze(0).cpu(),
            "logvar_g": logvar_g.squeeze(0).cpu(),
            "mu_n":     mu_n.cpu(),
            "logvar_n": logvar_n.cpu(),
        }
        by_n.setdefault(graph.num_nodes, []).append(entry)
    return by_n


def sample_from_pool(
    pool: dict,
    target_n_nodes: int,
    device: torch.device,
    noise_scale: float = 1.0,
):
    """
    Pick a training graph whose node count matches target_n_nodes (or the
    closest available count), then sample z_g and z_n from its posterior.
    This keeps each node's latent tied to a real service/op identity.
    """
    if target_n_nodes in pool:
        entries = pool[target_n_nodes]
    else:
        closest = min(pool.keys(), key=lambda k: abs(k - target_n_nodes))
        entries = pool[closest]

    entry = random.choice(entries)
    mu_g     = entry["mu_g"].to(device)
    logvar_g = entry["logvar_g"].to(device)
    mu_n     = entry["mu_n"].to(device)
    logvar_n = entry["logvar_n"].to(device)

    z_g = mu_g + noise_scale * (0.5 * logvar_g).exp() * torch.randn_like(mu_g)
    z_n = mu_n + noise_scale * (0.5 * logvar_n).exp() * torch.randn_like(mu_n)

    # resize z_n if the matched graph's node count differs from target
    actual = z_n.size(0)
    if actual > target_n_nodes:
        z_n = z_n[:target_n_nodes]
    elif actual < target_n_nodes:
        extra = target_n_nodes - actual
        idx = torch.randint(actual, (extra,), device=device)
        z_n = torch.cat([z_n, z_n[idx] + 0.1 * torch.randn(extra, z_n.size(1), device=device)], dim=0)

    return z_g, z_n


def load_graphs(datapath: str):
    ds = LoadDataset(datapath)
    return [ds.get(i) for i in range(len(ds))]


def sample_node_counts(graphs, target_count: int = None, seed: int = 42):
    counts = [g.num_nodes for g in graphs]
    target = target_count or len(counts)
    rng = random.Random(seed)
    if target <= len(counts):
        return rng.sample(counts, target)
    return [rng.choice(counts) for _ in range(target)]


def sample_edge_counts(graphs, target_count: int = None, seed: int = 42):
    counts = [int(g.edge_index.size(1)) for g in graphs]
    target = target_count or len(counts)
    rng = random.Random(seed + 17)
    if target <= len(counts):
        return rng.sample(counts, target)
    return [rng.choice(counts) for _ in range(target)]


def filter_graphs_by_node_range(graphs, min_nodes: int, max_nodes: int):
    return [g for g in graphs if min_nodes <= g.num_nodes <= max_nodes]


def sample_node_counts_stratified(graphs, target_count: int = None, seed: int = 42):
    buckets = [(1, 5), (6, 10), (11, 20), (21, 30)]
    rng = random.Random(seed)
    bucketed = {bucket: [] for bucket in buckets}
    for graph in graphs:
        for lo, hi in buckets:
            if lo <= graph.num_nodes <= hi:
                bucketed[(lo, hi)].append(graph)
                break
    non_empty = [(bucket, bucket_graphs) for bucket, bucket_graphs in bucketed.items() if bucket_graphs]
    if not non_empty:
        return sample_node_counts(graphs, target_count=target_count, seed=seed)
    target = target_count or len(graphs)
    per_bucket = target // len(non_empty)
    remainder = target % len(non_empty)
    counts = []
    for i, (_, bucket_graphs) in enumerate(non_empty):
        n_take = per_bucket + (1 if i < remainder else 0)
        bucket_counts = [g.num_nodes for g in bucket_graphs]
        counts.extend([rng.choice(bucket_counts) for _ in range(n_take)])
    rng.shuffle(counts)
    return counts


def sample_edge_counts_for_node_counts(graphs, node_counts: list, seed: int = 42):
    by_n = {}
    for graph in graphs:
        by_n.setdefault(graph.num_nodes, []).append(int(graph.edge_index.size(1)))
    all_counts = sorted(by_n.keys())
    rng = random.Random(seed + 17)
    result = []
    for n_nodes in node_counts:
        closest = min(all_counts, key=lambda k: abs(k - n_nodes))
        result.append(rng.choice(by_n[closest]))
    return result


def build_valid_ops_by_service(graphs):
    valid = {}
    for graph in graphs:
        x = graph.x.detach().cpu()
        for service_id, op_id in zip(x[:, 0].round().long(), x[:, 1].round().long()):
            valid.setdefault(int(service_id.item()), set()).add(int(op_id.item()))
    return {
        service_id: torch.tensor(sorted(op_ids), dtype=torch.long)
        for service_id, op_ids in valid.items()
    }


def save_graphs(graphs, out_path: str):
    data, slices = InMemoryDataset.collate(graphs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(graphs)} graphs)")


def build_dataset(
    *,
    real_path: str,
    weights_path: str,
    out_path: str,
    y_label: int,
    target_count: int = None,
    seed: int = 42,
    edge_threshold: float = 0.6,
    sample_edges: bool = True,
    sample_nodes: bool = True,
    node_temperature: float = 1.25,
    edge_dropout: float = 0.05,
    duration_noise: float = 0.02,
    threshold_jitter: float = 0.02,
    latent_noise_scale: float = 1.0,
    min_nodes: int = 1,
    trained_node_cap: int = 30,
    stratified: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, cfg = load_encoder_decoder_from_checkpoint(weights_path, device)

    real_graphs = load_graphs(real_path)
    eligible_graphs = filter_graphs_by_node_range(
        real_graphs,
        min_nodes=min_nodes,
        max_nodes=trained_node_cap,
    )
    if not eligible_graphs:
        raise RuntimeError(
            f"No real graphs in {real_path} matched "
            f"{min_nodes} <= num_nodes <= {trained_node_cap}."
        )
    print(
        f"Building latent pool from {len(eligible_graphs)} eligible real graphs "
        f"({min_nodes} <= nodes <= {trained_node_cap}) out of {len(real_graphs)} total."
    )
    pool = build_latent_pool(encoder, eligible_graphs, device)

    if stratified:
        print("Using stratified node-count sampling across span-count buckets.")
        node_counts = sample_node_counts_stratified(
            eligible_graphs, target_count=target_count, seed=seed
        )
    else:
        node_counts = sample_node_counts(eligible_graphs, target_count=target_count, seed=seed)
    edge_counts = sample_edge_counts_for_node_counts(eligible_graphs, node_counts, seed=seed)
    valid_ops_by_service = build_valid_ops_by_service(eligible_graphs)

    torch.manual_seed(seed)
    syn_graphs = []
    for n_nodes, target_edges in zip(node_counts, edge_counts):
        z_g, z_n = sample_from_pool(pool, n_nodes, device, noise_scale=latent_noise_scale)
        graph = generate_synthetic_graph(
            decoder,
            num_nodes=n_nodes,
            duration_mean=cfg["duration_mean"],
            duration_std=cfg["duration_std"],
            device=device,
            edge_threshold=edge_threshold,
            sample_edges=sample_edges,
            sample_nodes=sample_nodes,
            node_temperature=node_temperature,
            edge_dropout=edge_dropout,
            duration_noise=duration_noise,
            threshold_jitter=threshold_jitter,
            target_edge_count=target_edges,
            valid_ops_by_service=valid_ops_by_service,
            z_g=z_g,
            z_n=z_n,
        )
        graph.y = torch.tensor(y_label, dtype=torch.long)
        syn_graphs.append(graph.cpu())

    save_graphs(syn_graphs, out_path)
    return {
        "real_path": real_path,
        "weights_path": weights_path,
        "out_path": out_path,
        "graphs": len(syn_graphs),
        "real_graphs_total": len(real_graphs),
        "real_graphs_used_for_sizes": len(eligible_graphs),
        "min_nodes": min_nodes,
        "trained_node_cap": trained_node_cap,
        "stratified": stratified,
        "node_count_min": min(node_counts) if node_counts else 0,
        "node_count_max": max(node_counts) if node_counts else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SN normal and abnormal synthetic datasets from trained VAEs."
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "abnormal", "both"],
        default="both",
        help="Which SN synthetic dataset(s) to generate.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Synthetic graph count per dataset. Defaults to size of the real dataset.",
    )
    parser.add_argument(
        "--normal-count",
        type=int,
        default=None,
        help="Generate this many normal synthetic graphs and scale abnormal graphs proportionally to the real SN class ratio.",
    )
    parser.add_argument("--normal-out-path", default=None)
    parser.add_argument("--abnormal-out-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--min-nodes", type=int, default=1)
    parser.add_argument("--trained-node-cap", type=int, default=30)
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Sample node counts evenly across 1-5, 6-10, 11-20, and 21-30 span buckets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jobs = []
    if args.mode in {"normal", "both"}:
        jobs.append(
            {
                "name": "normal",
                "real_path": "./datasets/anomaly/SN/SN_normal.pt",
                "weights_path": "./weights/sn_normal_vae_weights.pt",
                "out_path": args.normal_out_path or "./datasets/anomaly/SN/SN_normal_synthetic.pt",
                "y_label": 0,
            }
        )
    if args.mode in {"abnormal", "both"}:
        jobs.append(
            {
                "name": "abnormal",
                "real_path": "./datasets/anomaly/SN/SN_abnormal.pt",
                "weights_path": "./weights/sn_abnormal_vae_weights.pt",
                "out_path": args.abnormal_out_path or "./datasets/anomaly/SN/SN_abnormal_synthetic.pt",
                "y_label": 1,
            }
        )

    proportional_counts = {}
    if args.normal_count is not None:
        normal_graphs = len(load_graphs("./datasets/anomaly/SN/SN_normal.pt"))
        abnormal_graphs = len(load_graphs("./datasets/anomaly/SN/SN_abnormal.pt"))
        abnormal_ratio = abnormal_graphs / normal_graphs if normal_graphs else 0.0
        proportional_counts["normal"] = args.normal_count
        proportional_counts["abnormal"] = max(
            1, round(args.normal_count * abnormal_ratio)
        )

    summary = {}
    for job in jobs:
        print(f"\n=== BUILDING SN {job['name'].upper()} SYNTHETIC DATASET ===")
        summary[job["name"]] = build_dataset(
            real_path=job["real_path"],
            weights_path=job["weights_path"],
            out_path=job["out_path"],
            y_label=job["y_label"],
            target_count=proportional_counts.get(job["name"], args.target_count),
            seed=args.seed,
            min_nodes=args.min_nodes,
            trained_node_cap=args.trained_node_cap,
            stratified=args.stratified,
        )

    summary_path = Path(args.summary_path) if args.summary_path else ROOT / "datasets" / "anomaly" / "SN" / "synthetic_summary.json"
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
