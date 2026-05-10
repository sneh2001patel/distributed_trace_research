import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from vae import GNNDecoder, GraphEncoderVAE
from vae_tt import TTGNNDecoder


ROOT = Path(__file__).resolve().parent


DEFAULT_RUNS = {
    ("SN", "normal"): {
        "data_path": "./datasets/anomaly/SN/SN_normal.pt",
        "weights_path": "./weights/sn_normal_vae_weights.pt",
        "parameter_name": "edge_threshold",
        "default_threshold": 0.6,
        "max_nodes": 30,
        "target_graphs": 1000,
    },
    ("SN", "abnormal"): {
        "data_path": "./datasets/anomaly/SN/SN_abnormal.pt",
        "weights_path": "./weights/sn_abnormal_vae_weights.pt",
        "parameter_name": "edge_threshold",
        "default_threshold": 0.6,
        "max_nodes": 30,
        "target_graphs": 1000,
    },
    ("TT", "normal"): {
        "data_path": "./datasets/anomaly/TT/TT_normal.pt",
        "weights_path": "./weights/tt_normal_vae_weights.pt",
        "parameter_name": "parent_threshold",
        "default_threshold": None,
        "max_nodes": 30,
        "target_graphs": 1000,
    },
    ("TT", "abnormal"): {
        "data_path": "./datasets/anomaly/TT/TT_abnormal.pt",
        "weights_path": "./weights/tt_abnormal_vae_weights.pt",
        "parameter_name": "parent_threshold",
        "default_threshold": None,
        "max_nodes": 30,
        "target_graphs": 1000,
    },
}


class PTDataset(InMemoryDataset):
    def __init__(self, datapath: str, make_undirected: bool = False) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices
        self.make_undirected = make_undirected

    def get(self, idx):
        data = super().get(idx)
        if self.make_undirected:
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        return data


class GraphListDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def _resolve(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(ROOT / path_obj)


def _load_graphs(path: str, system: str):
    dataset = PTDataset(path, make_undirected=(system == "SN"))
    return [dataset.get(i).cpu() for i in range(len(dataset))]


def _filter_and_sample(graphs, max_nodes: int, target_graphs: int, seed: int):
    filtered = [g for g in graphs if g.num_nodes <= max_nodes]
    if target_graphs is None or len(filtered) <= target_graphs:
        return filtered
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(filtered), generator=generator)[:target_graphs].tolist()
    return [filtered[i] for i in indices]


def _load_checkpoint(system: str, weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    cfg = checkpoint["config"]
    decoder_state = checkpoint["decoder_state"]
    max_nodes = cfg.get("max_nodes")
    if max_nodes is None or "pos_emb.weight" in decoder_state:
        max_nodes = int(decoder_state["pos_emb.weight"].shape[0])

    encoder = GraphEncoderVAE(
        n_services=cfg["num_services"],
        n_ops=cfg["num_ops"],
        duration_mean=cfg["duration_mean"],
        duration_std=cfg["duration_std"],
        hidden_ch=cfg.get("encoder_hidden_dim", cfg["hidden_dim"]),
        embed_dim=cfg.get("embed_dim", 48 if system == "SN" else 32),
        latent_dim=cfg["latent_dim"],
        dropout=cfg.get("encoder_dropout", 0.2 if system == "SN" else 0.1),
    ).to(device)

    if system == "TT":
        decoder = TTGNNDecoder(
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_service_classes=cfg["num_services"],
            n_op_classes=cfg["num_ops"],
            encoder_hidden_dim=cfg.get("encoder_hidden_dim", cfg["hidden_dim"]),
            max_nodes=max_nodes,
            head_hidden_dim=cfg.get("head_hidden_dim", 128),
            head_dropout=cfg.get("head_dropout", 0.05),
            enc_h_dropout=cfg.get("enc_h_dropout", 0.0),
        ).to(device)
    else:
        decoder = GNNDecoder(
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_service_classes=cfg["num_services"],
            n_op_classes=cfg["num_ops"],
            encoder_hidden_dim=cfg.get("encoder_hidden_dim", cfg["hidden_dim"]),
            max_nodes=max_nodes,
            head_hidden_dim=cfg.get("head_hidden_dim", cfg["hidden_dim"] * 2),
            head_dropout=cfg.get("head_dropout", 0.1),
            enc_h_dropout=cfg.get("enc_h_dropout", 0.0),
        ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@torch.no_grad()
def _collect_sn_edge_scores(dataset, encoder, decoder, device):
    scores = []
    labels = []
    for data in DataLoader(dataset, batch_size=1, shuffle=False):
        data = data.to(device)
        enc_out = encoder(data.x, data.edge_index, data.batch)
        _, _, z_g = enc_out["graph"]
        _, _, z_n = enc_out["node"]
        h_enc = enc_out["node_h"]
        _, _, _, edge_logits = decoder.decode_for_training(z_g[0], z_n, h_enc)

        adj_real = torch.zeros((data.num_nodes, data.num_nodes), device=device)
        if data.edge_index.numel() > 0:
            adj_real[data.edge_index[0], data.edge_index[1]] = 1.0

        mask = torch.triu(
            torch.ones(data.num_nodes, data.num_nodes, device=device, dtype=torch.bool),
            diagonal=1,
        )
        scores.append(torch.sigmoid(edge_logits)[mask].detach().cpu())
        labels.append(adj_real[mask].detach().cpu())
    return torch.cat(scores), torch.cat(labels)


@torch.no_grad()
def _collect_tt_edge_scores(dataset, encoder, decoder, device, parent_temperature: float):
    scores = []
    labels = []
    for data in DataLoader(dataset, batch_size=1, shuffle=False):
        data = data.to(device)
        enc_out = encoder(data.x, data.edge_index, data.batch)
        _, _, z_g = enc_out["graph"]
        _, _, z_n = enc_out["node"]
        h_enc = enc_out["node_h"]
        _, _, _, parent_logits = decoder.decode_for_training(z_g[0], z_n, h_enc)

        n_nodes = data.num_nodes
        parent_probs = F.softmax(parent_logits / max(parent_temperature, 1e-6), dim=1)
        edge_scores = parent_probs[:, :n_nodes].t().contiguous()

        adj_real = torch.zeros((n_nodes, n_nodes), device=device)
        if data.edge_index.numel() > 0:
            adj_real[data.edge_index[0], data.edge_index[1]] = 1.0

        mask = ~torch.eye(n_nodes, device=device, dtype=torch.bool)
        scores.append(edge_scores[mask].detach().cpu())
        labels.append(adj_real[mask].detach().cpu())
    return torch.cat(scores), torch.cat(labels)


def _threshold_rows(scores, labels, thresholds):
    labels = labels.bool()
    positives = int(labels.sum().item())
    negatives = int(labels.numel() - positives)
    rows = []
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int((pred & labels).sum().item())
        fp = int((pred & ~labels).sum().item())
        fn = int((~pred & labels).sum().item())
        tn = int((~pred & ~labels).sum().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "parameter_value": round(float(threshold), 6),
                "precision": round(precision, 8),
                "recall": round(recall, 8),
                "f1": round(f1, 8),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "num_positive_edges": positives,
                "num_negative_edges": negatives,
                "num_candidate_edges": int(labels.numel()),
            }
        )
    return rows


def _make_thresholds(start: float, stop: float, step: float):
    count = int(round((stop - start) / step)) + 1
    return [start + i * step for i in range(count)]


def main():
    parser = argparse.ArgumentParser(
        description="Export VAE edge precision-recall curve points by sweeping decoder thresholds."
    )
    parser.add_argument("--systems", nargs="+", choices=["SN", "TT"], default=["SN", "TT"])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["normal", "abnormal"],
        default=["normal", "abnormal"],
        help="Which separately trained VAE checkpoints/datasets to evaluate.",
    )
    parser.add_argument(
        "--out-csv",
        default="./classifier/vae_pr_curve_results.csv",
        help="CSV path for PR curve rows.",
    )
    parser.add_argument("--threshold-start", type=float, default=0.01)
    parser.add_argument("--threshold-stop", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--parent-temperature",
        type=float,
        default=1.0,
        help="TT parent softmax temperature. Keep fixed for the main PR curve.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-graphs",
        type=int,
        default=None,
        help="Override sampled graph count per run. Defaults are checkpoint-family specific.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Override max node filter per run. Defaults are checkpoint-family specific.",
    )
    args = parser.parse_args()

    thresholds = _make_thresholds(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_rows = []
    for system in args.systems:
        for split in args.splits:
            cfg = DEFAULT_RUNS[(system, split)]
            data_path = _resolve(cfg["data_path"])
            weights_path = _resolve(cfg["weights_path"])
            max_nodes = args.max_nodes if args.max_nodes is not None else cfg["max_nodes"]
            target_graphs = (
                args.target_graphs
                if args.target_graphs is not None
                else cfg["target_graphs"]
            )

            graphs = _filter_and_sample(
                _load_graphs(data_path, system),
                max_nodes=max_nodes,
                target_graphs=target_graphs,
                seed=args.seed,
            )
            if not graphs:
                raise RuntimeError(f"No graphs left after filtering for {system} {split}.")

            dataset = GraphListDataset(graphs)
            encoder, decoder = _load_checkpoint(system, weights_path, device)
            if system == "TT":
                scores, labels = _collect_tt_edge_scores(
                    dataset, encoder, decoder, device, args.parent_temperature
                )
            else:
                scores, labels = _collect_sn_edge_scores(dataset, encoder, decoder, device)

            for row in _threshold_rows(scores, labels, thresholds):
                row.update(
                    {
                        "system": system,
                        "split": split,
                        "parameter_name": cfg["parameter_name"],
                        "default_threshold": cfg["default_threshold"],
                        "weights_path": cfg["weights_path"],
                        "data_path": cfg["data_path"],
                        "graphs_evaluated": len(graphs),
                        "max_nodes": max_nodes,
                        "target_graphs": target_graphs,
                        "parent_temperature": args.parent_temperature
                        if system == "TT"
                        else "",
                    }
                )
                output_rows.append(row)

    fieldnames = [
        "system",
        "split",
        "parameter_name",
        "parameter_value",
        "default_threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "num_positive_edges",
        "num_negative_edges",
        "num_candidate_edges",
        "graphs_evaluated",
        "max_nodes",
        "target_graphs",
        "parent_temperature",
        "weights_path",
        "data_path",
    ]
    os.makedirs(os.path.dirname(_resolve(args.out_csv)) or ".", exist_ok=True)
    with open(_resolve(args.out_csv), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved {len(output_rows)} PR curve rows to {args.out_csv}")


if __name__ == "__main__":
    main()
