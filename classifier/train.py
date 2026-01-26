import os
import random

import torch
import torch.nn.functional as F
from model import GraphClassifier
from prepare_datasets import load_graphs, prepare_datasets, prepare_mixed_dataset
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


def _class_weights(graphs, num_classes=2):
    counts = torch.zeros(num_classes, dtype=torch.float)
    for g in graphs:
        y = int(g.y.item()) if hasattr(g, "y") else 0
        counts[y] += 1
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights


def _run_epoch(model, loader, device, optimizer=None, class_weights=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.view(-1).long()
        loss = F.cross_entropy(logits, y, weight=class_weights)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc


@torch.no_grad()
def _confusion_matrix(model, loader, device, num_classes=2):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.view(-1).long()
        pred = logits.argmax(dim=1)
        for t, p in zip(y, pred):
            cm[t.item(), p.item()] += 1
    return cm


def _per_class_accuracy(cm: torch.Tensor):
    acc = []
    for i in range(cm.size(0)):
        total = cm[i].sum().item()
        correct = cm[i, i].item()
        acc.append((correct / total) if total else 0.0)
    return acc


def _per_class_prf(cm: torch.Tensor):
    metrics = []
    for i in range(cm.size(0)):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        metrics.append((precision, recall, f1))
    return metrics


# def test():
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     processed_dir = os.path.abspath(os.path.join(base_dir, "..", "processed"))
#     exact_dir = os.path.join(processed_dir, "exact_replica")
#     not_exact_dir = os.path.join(processed_dir, "exact_replica")
#
#     syn_sn = os.path.join(exact_dir, "SN_synthetic.pt")
#     syn_tt = os.path.join(exact_dir, "TT_synthetic.pt")
#     if not os.path.exists(syn_sn) or not os.path.exists(syn_tt):
#         syn_sn = os.path.join(processed_dir, "SN_synthetic.pt")
#         syn_tt = os.path.join(processed_dir, "TT_synthetic.pt")
#
#     train_ds, val_ds, test_ds = prepare_mixed_dataset(
#         syn_sn_path=syn_sn,
#         syn_tt_path=syn_tt,
#         real_sn_path=os.path.join(processed_dir, "SN_data.pt"),
#         real_tt_path=os.path.join(processed_dir, "TT_data.pt"),
#         real_train_ratio=0.5,
#         val_ratio=0.2,
#         seed=42,
#     )
#
#     print(
#         f"Training Data: {len(train_ds)}, Validate Data: {len(val_ds)}, Testing Data: {len(test_ds)}"
#     )
#


def main(weights_path):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(base_dir, "..", "datasets"))
    fixed_size_dir = os.path.join(processed_dir, "fixed_size")
    variable_size_dir = os.path.join(processed_dir, "variable_size")

    syn_sn = os.path.join(variable_size_dir, "SN_synthetic.pt")
    syn_tt = os.path.join(variable_size_dir, "TT_synthetic.pt")

    train_ds, val_ds, test_ds = prepare_mixed_dataset(
        syn_sn_path=syn_sn,
        syn_tt_path=syn_tt,
        real_sn_path=os.path.join(processed_dir, "SN_data.pt"),
        real_tt_path=os.path.join(processed_dir, "TT_data.pt"),
        real_train_ratio=0.1,
        val_ratio=0.2,
        seed=42,
    )

    # train_ds, val_ds, test_ds = prepare_datasets(
    #     syn_sn_path=syn_sn,
    #     syn_tt_path=syn_tt,
    #     real_sn_path=os.path.join(processed_dir, "SN_data.pt"),
    #     real_tt_path=os.path.join(processed_dir, "TT_data.pt"),
    #     val_ratio=0.2,
    #     seed=42,
    # )

    train_graphs = (
        train_ds.dataset.graphs if hasattr(train_ds, "dataset") else train_ds.graphs
    )
    real_sn_graphs = load_graphs(os.path.join(processed_dir, "SN_data.pt"))
    real_tt_graphs = load_graphs(os.path.join(processed_dir, "TT_data.pt"))
    real_graphs = real_sn_graphs + real_tt_graphs

    n_services, n_ops = _max_ids(real_graphs)
    dur_mean, dur_std = _duration_stats(train_graphs + real_graphs)

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

    class_weights = None
    use_class_weights = True
    if use_class_weights:
        class_weights = _class_weights(train_graphs).to(device)

    lr = 3e-4

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = None
    epochs = 50
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, device, optimizer, class_weights
        )
        if val_loader:
            val_loss, val_acc = _run_epoch(model, val_loader, device)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "./classifier_weights.pt")
            print(
                f"Epoch {epoch:03d} | train {train_loss:.4f}/{train_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train {train_loss:.4f}/{train_acc:.3f}")

    # final_weights_path = "./classifier_weights_final.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved final weights to {weights_path}")

    # if val_loader and os.path.exists("./classifier_weights.pt"):
    #     model.load_state_dict(
    #         torch.load("./classifier_weights.pt", map_location=device)
    # )

    test_loss, test_acc = _run_epoch(model, test_loader, device)
    print(f"Test {test_loss:.4f}/{test_acc:.3f}")
    cm = _confusion_matrix(model, test_loader, device, num_classes=2)
    per_class = _per_class_accuracy(cm)
    prf = _per_class_prf(cm)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"Per-class accuracy: SN={per_class[0]:.3f}, TT={per_class[1]:.3f}")
    print(
        "Per-class PRF: "
        f"SN Precision={prf[0][0]:.3f} Recall={prf[0][1]:.3f} F1={prf[0][2]:.3f} | "
        f"TT Precision={prf[1][0]:.3f} Recall={prf[1][1]:.3f} F1={prf[1][2]:.3f}"
    )


if __name__ == "__main__":
    main("./classifier_weights.pt")
