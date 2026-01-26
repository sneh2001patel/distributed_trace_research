import argparse
import os
import random

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr

from synthetic_graphs import generate_synthetic_graph, load_decoder


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        super().__init__(".")
        data, slices = torch.load(datapath, weights_only=False)
        self.data, self.slices = data, slices


def build_synthetic_precise_dataset(
    real_path: str,
    weights_path: str,
    out_path: str,
    y_label: int,
    target_count: int = None,
    edge_threshold: float = 0.9,
    sample_edges: bool = True,
    sample_nodes: bool = True,
    node_temperature: float = 1.0,
    match_id_distribution: bool = False,
    edge_dropout: float = 0.0,
    duration_noise: float = 0.0,
    threshold_jitter: float = 0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder(weights_path, device)

    real_ds = LoadDataset(real_path)
    node_counts = [real_ds.get(i).num_nodes for i in range(len(real_ds))]
    target = target_count or len(node_counts)
    syn_graphs = []
    joint_probs = None
    n_sid = None
    n_op = None
    if match_id_distribution:
        xs = torch.cat([real_ds.get(i).x for i in range(len(real_ds))], dim=0)
        sid = xs[:, 0].round().long()
        op = xs[:, 1].round().long()
        n_sid = int(sid.max().item()) + 1
        n_op = int(op.max().item()) + 1
        joint = sid * n_op + op
        counts = torch.bincount(joint, minlength=n_sid * n_op).float()
        joint_probs = counts / counts.sum()
    # if target <= len(node_counts):
    #     sampled_counts = random.sample(node_counts, target)
    # else:
    #     sampled_counts = random.choices(node_counts, k=target)

    for n_nodes in node_counts:
        syn_graph = generate_synthetic_graph(
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
        )
        if match_id_distribution:
            probs = joint_probs.to(syn_graph.x.device)
            joint_idx = torch.multinomial(probs, n_nodes, replacement=True)
            sid = (joint_idx // n_op).float()
            op = (joint_idx % n_op).float()
            syn_graph.x[:, 0] = sid
            syn_graph.x[:, 1] = op
        syn_graph.y = torch.tensor(y_label)
        syn_graphs.append(syn_graph)

    data, slices = InMemoryDataset.collate(syn_graphs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(syn_graphs)} graphs)")


def build_synthetic_dataset(
    real_path: str,
    weights_path: str,
    out_path: str,
    y_label: int,
    target_count: int = None,
    edge_threshold: float = 0.9,
    sample_edges: bool = False,
    sample_nodes: bool = False,
    order: bool = False,
    edge_dropout: float = 0.0,
    duration_noise: float = 0.0,
    threshold_jitter: float = 0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder(weights_path, device)

    real_ds = LoadDataset(real_path)
    node_counts = [real_ds.get(i).num_nodes for i in range(len(real_ds))]
    syn_graphs = []
    if order:
        target = target_count or len(node_counts)
        if target <= len(node_counts):
            sampled_counts = random.sample(node_counts, target)
        else:
            sampled_counts = random.choices(node_counts, k=target)
        node_counts = sampled_counts
    for n_nodes in node_counts:
        syn_graph = generate_synthetic_graph(
            decoder,
            num_nodes=n_nodes,
            duration_mean=cfg["duration_mean"],
            duration_std=cfg["duration_std"],
            device=device,
            edge_threshold=edge_threshold,
            sample_edges=sample_edges,
            sample_nodes=sample_nodes,
            edge_dropout=edge_dropout,
            duration_noise=duration_noise,
            threshold_jitter=threshold_jitter,
        )
        syn_graph.y = torch.tensor(y_label)
        syn_graphs.append(syn_graph)

    data, slices = InMemoryDataset.collate(syn_graphs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(syn_graphs)} graphs)")


def test():
    real_ds1 = LoadDataset("./datasets/TT_data.pt")
    real_ds2 = LoadDataset("./datasets/SN_data.pt")

    print("TT DATA")
    for i in range(len(real_ds1)):
        r_graph = real_ds1.get(i)
        print(r_graph.y.item())

    print("SN DATA")
    for j in range(len(real_ds2)):
        r_graph = real_ds2.get(j)
        print(r_graph.y.item())


def build_synthetic_dataset_variable_size(
    weights_path,
    out_path,
    y_label,
    range_n_nodes,
    num_graphs: int = 1244,
    edge_threshold=0.9,
    sample_edges=False,
    sample_nodes=False,
    node_temperature: float = 1.0,
    edge_dropout: float = 0.0,
    duration_noise: float = 0.0,
    threshold_jitter: float = 0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, cfg = load_decoder(weights_path, device)
    syn_graphs = []

    min_nodes, max_nodes = range_n_nodes
    for _ in range(num_graphs):
        num_nodes = random.randint(min_nodes, max_nodes)
        syn_graph = generate_synthetic_graph(
            decoder,
            num_nodes=num_nodes,  # sample in [min_nodes, max_nodes]
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
        )
        syn_graph.y = torch.tensor(y_label)
        syn_graphs.append(syn_graph)

    data, slices = InMemoryDataset.collate(syn_graphs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save((data, slices), out_path)
    print(f"saved {out_path} ({len(syn_graphs)} graphs)")


def main():
    precise_aug = {
        "sample_edges": True,
        "sample_nodes": True,
        "node_temperature": 2.0,
        "match_id_distribution": True,
        "edge_dropout": 0.1,
        "duration_noise": 0.05,
        "threshold_jitter": 0.05,
    }

    fixed_aug = {
        "sample_edges": True,
        "sample_nodes": False,
        "edge_dropout": 0.1,
        "duration_noise": 0.05,
        "threshold_jitter": 0.05,
    }

    variable_aug = {
        "sample_edges": True,
        "sample_nodes": False,
        "edge_dropout": 0.1,
        "duration_noise": 0.05,
        "threshold_jitter": 0.05,
    }

    tt_count = len(LoadDataset("./datasets/TT_data.pt"))
    sn_count = len(LoadDataset("./datasets/SN_data.pt"))
    total = tt_count + sn_count
    target_tt = int(total * 0.5)
    target_sn = total - target_tt

    # Generate Fixed-Size Dataset

    # TT
    build_synthetic_dataset(
        "./datasets/TT_data.pt",
        "./weights/tt_vae_weights1.pt",
        "./datasets/fixed_size/TT_synthetic.pt",
        y_label=1,
        target_count=None,
        order=False,
        edge_threshold=0.9,
        **fixed_aug,
    )
    # # SN
    build_synthetic_dataset(
        "./datasets/SN_data.pt",
        "./weights/sn_vae_weights1.pt",
        "./datasets/fixed_size/SN_synthetic.pt",
        y_label=0,
        target_count=None,
        order=False,
        edge_threshold=0.9,
        **fixed_aug,
    )
    # TT
    build_synthetic_precise_dataset(
        "./datasets/TT_data.pt",
        "./weights/tt_vae_weights1.pt",
        "./datasets/fixed_size/order_TT_fixed_size_precise.pt",
        y_label=0,
        target_count=None,
        edge_threshold=0.9,
        **precise_aug,
    )

    # SN
    build_synthetic_precise_dataset(
        "./datasets/SN_data.pt",
        "./weights/sn_vae_weights1.pt",
        "./datasets/fixed_size/order_SN_fixed_size_precise.pt",
        y_label=0,
        target_count=None,
        edge_threshold=0.9,
        **precise_aug,
    )
    # Generate Dataset with liberty
    # TT
    build_synthetic_dataset_variable_size(
        "./weights/tt_vae_weights1.pt",
        "./datasets/variable_size/TT_synthetic.pt",
        y_label=1,
        range_n_nodes=(1, 23),  # min max gotten from summarize_ds script
        edge_threshold=0.9,
        **variable_aug,
    )

    # SN
    build_synthetic_dataset_variable_size(
        "./weights/sn_vae_weights1.pt",
        "./datasets/variable_size/SN_synthetic.pt",
        y_label=0,
        range_n_nodes=(3, 30),  # min max gotten from summarize_ds script
        edge_threshold=0.9,
        **variable_aug,
    )


if __name__ == "__main__":
    main()
