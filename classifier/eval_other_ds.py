import os

import torch
from dataset import load_graphs, prepare_mixed_dataset
from model import GraphClassifier
from torch_geometric.loader import DataLoader


def _max_ids(graphs):
    services = []
    ops = []
    for g in graphs:
        x = g.x
        services.append(x[:, 0])
        ops.append(x[:, 1])
    service_max = int(torch.cat(services).max().item()) if services else 0
    op_max = int(torch.cat(ops).max().item()) if ops else 0
    return service_max + 1, op_max + 1


def _duration_stats(graphs):
    durations = []
    for g in graphs:
        durations.append(g.x[:, 2].detach().cpu())
    d = torch.cat(durations) if durations else torch.tensor([0.0])
    log_d = torch.log1p(torch.clamp(d, min=0))
    return log_d.mean().item(), (log_d.std().item() or 1.0)


def _load_training_stats(processed_dir):
    exact_dir = os.path.join(processed_dir, "exact_replica")
    syn_sn = os.path.join(exact_dir, "SN_synthetic.pt")
    syn_tt = os.path.join(exact_dir, "TT_synthetic.pt")
    if not os.path.exists(syn_sn) or not os.path.exists(syn_tt):
        syn_sn = os.path.join(processed_dir, "SN_synthetic.pt")
        syn_tt = os.path.join(processed_dir, "TT_synthetic.pt")

    train_graphs = []
    if os.path.exists(syn_sn) and os.path.exists(syn_tt):
        train_ds, _, _ = prepare_mixed_dataset(
            syn_sn_path=syn_sn,
            syn_tt_path=syn_tt,
            real_sn_path=os.path.join(processed_dir, "SN_data.pt"),
            real_tt_path=os.path.join(processed_dir, "TT_data.pt"),
            real_train_ratio=0.1,
            val_ratio=0.2,
            seed=42,
        )
        train_graphs = (
            train_ds.dataset.graphs if hasattr(train_ds, "dataset") else train_ds.graphs
        )

    real_sn_graphs = load_graphs(os.path.join(processed_dir, "SN_data.pt"))
    real_tt_graphs = load_graphs(os.path.join(processed_dir, "TT_data.pt"))
    real_graphs = real_sn_graphs + real_tt_graphs

    n_services, n_ops = _max_ids(real_graphs)
    dur_mean, dur_std = _duration_stats(train_graphs + real_graphs)
    return n_services, n_ops, dur_mean, dur_std


@torch.no_grad()
def _predict_labels(model, device, graphs, name, acc, ground_truth):
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    model.eval()
    wrong = 0
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        logits = model(batch)
        pred = int(logits.argmax(dim=1).item())
        if pred == ground_truth:
            acc.append(1)
        else:
            wrong += 1
            print(name)
            print(f"{name}[{idx}] -> {pred}/{ground_truth}")
    return wrong


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(base_dir, "..", "processed"))

    weights_path = os.path.join(base_dir, "classifier_weights_final.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights at {weights_path}")

    n_services, n_ops, dur_mean, dur_std = _load_training_stats(processed_dir)

    model = GraphClassifier(
        n_services=n_services,
        n_ops=n_ops,
        num_classes=2,
        embed_dim=48,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        duration_mean=dur_mean,
        duration_std=dur_std,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    sn_path1 = os.path.join(processed_dir, "other_ds", "SN", "SN_data1.pt")
    tt_path1 = os.path.join(processed_dir, "other_ds", "TT", "TT_data1.pt")

    sn_path2 = os.path.join(processed_dir, "other_ds", "SN", "SN_data2.pt")
    tt_path2 = os.path.join(processed_dir, "other_ds", "TT", "TT_data2.pt")
    sn_path3 = os.path.join(processed_dir, "other_ds", "SN", "SN_data3.pt")
    tt_path3 = os.path.join(processed_dir, "other_ds", "TT", "TT_data3.pt")

    sn_graphs1 = load_graphs(sn_path1)
    tt_graphs1 = load_graphs(tt_path1)

    sn_graphs2 = load_graphs(sn_path2)
    tt_graphs2 = load_graphs(tt_path2)

    sn_graphs3 = load_graphs(sn_path3)
    tt_graphs3 = load_graphs(tt_path3)

    print("Predicted labels (0=SN, 1=TT):")

    sn_acc1 = []
    sn_wrongs1 = _predict_labels(model, device, sn_graphs1, "SN_data1", sn_acc1, 0)

    sn_acc2 = []
    sn_wrongs2 = _predict_labels(model, device, sn_graphs2, "SN_data2", sn_acc2, 0)

    sn_acc3 = []
    sn_wrongs3 = _predict_labels(model, device, sn_graphs3, "SN_data3", sn_acc3, 0)

    tt_acc1 = []
    tt_wrongs1 = _predict_labels(model, device, tt_graphs1, "TT_data1", tt_acc1, 1)

    tt_acc2 = []
    tt_wrongs2 = _predict_labels(model, device, tt_graphs2, "TT_data2", tt_acc2, 1)

    tt_acc3 = []
    tt_wrongs3 = _predict_labels(model, device, tt_graphs3, "TT_data3", tt_acc3, 1)

    print(
        f"SN Data 1: {len(sn_acc1) / len(sn_graphs1)}, Number of Wrongs: {sn_wrongs1}"
    )
    print(
        f"SN Data 2: {len(sn_acc2) / len(sn_graphs2)}, Number of Wrongs: {sn_wrongs2}"
    )
    print(
        f"SN Data 3: {len(sn_acc3) / len(sn_graphs3)}, Number of Wrongs: {sn_wrongs3}"
    )

    print(
        f"TT Data 1: {len(tt_acc1) / len(tt_graphs1)}, Number of Wrongs: {tt_wrongs1}"
    )
    print(
        f"TT Data 2: {len(tt_acc2) / len(tt_graphs2)}, Number of Wrongs: {tt_wrongs2}"
    )
    print(
        f"TT Data 3: {len(tt_acc3) / len(tt_graphs3)}, Number of Wrongs: {tt_wrongs3}"
    )


if __name__ == "__main__":
    main()
