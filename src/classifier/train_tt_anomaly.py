import os
import random
import argparse
import csv

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
try:
    from model import GraphClassifier
    from prepare_anomaly_datasets import load_graphs, prepare_tt_anomaly_datasets
except ModuleNotFoundError:
    from classifier.model import GraphClassifier
    from classifier.prepare_anomaly_datasets import load_graphs, prepare_tt_anomaly_datasets
from torch_geometric.loader import DataLoader


def _max_ids(graphs):
    services = []
    ops = []
    for g in graphs:
        services.append(g.x[:, 0])
        ops.append(g.x[:, 1])
    service_max = int(torch.cat(services).max().item()) if services else 0
    op_max = int(torch.cat(ops).max().item()) if ops else 0
    return service_max + 1, op_max + 1


def _duration_stats(graphs):
    durations = [g.x[:, 2].detach().cpu() for g in graphs]
    d = torch.cat(durations) if durations else torch.tensor([0.0])
    log_d = torch.log1p(torch.clamp(d, min=0))
    return log_d.mean().item(), (log_d.std().item() or 1.0)


def _class_weights(graphs, num_classes=2):
    counts = torch.zeros(num_classes, dtype=torch.float)
    for g in graphs:
        counts[int(g.y.item())] += 1
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
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total if total else 0.0, correct / total if total else 0.0


@torch.no_grad()
def _confusion_matrix(model, loader, device, num_classes=2):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).argmax(dim=1)
        y = batch.y.view(-1).long()
        for t, p in zip(y, pred):
            cm[t.item(), p.item()] += 1
    return cm


@torch.no_grad()
def _prediction_scores(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    for batch in loader:
        batch = batch.to(device)
        probs = F.softmax(model(batch), dim=1)
        y_true.extend(batch.y.view(-1).long().detach().cpu().tolist())
        y_score.extend(probs[:, 1].detach().cpu().tolist())
    return y_true, y_score


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


def _weighted_prf(cm: torch.Tensor):
    per_class = _per_class_prf(cm)
    supports = [cm[i, :].sum().item() for i in range(cm.size(0))]
    total = sum(supports)
    if total == 0:
        return 0.0, 0.0, 0.0
    weighted_precision = sum(
        per_class[i][0] * supports[i] for i in range(len(per_class))
    ) / total
    weighted_recall = sum(
        per_class[i][1] * supports[i] for i in range(len(per_class))
    ) / total
    weighted_f1 = sum(per_class[i][2] * supports[i] for i in range(len(per_class))) / total
    return weighted_precision, weighted_recall, weighted_f1


def _append_results_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _validate_zero_real_setup(summary: dict, real_percent):
    if real_percent != 0.0:
        return
    if (
        summary["train_real_total"] != 0
        or summary["train_normal_real"] != 0
        or summary["train_abnormal_real"] != 0
    ):
        raise RuntimeError(f"--real 0 produced real training data: {summary}")
    if summary.get("val_source") != "synthetic":
        raise RuntimeError(f"--real 0 must use synthetic validation: {summary}")


def main(
    weights_path="./tt_anomaly_classifier_weights.pt",
    syn_normal_path=None,
    syn_abnormal_path=None,
    generator_name="hierarchical_vae",
    epochs=120,
    real_train_ratio=0.1,
    real_percent=None,
    val_ratio=0.2,
    real_holdout_ratio=0.2,
    real_val_ratio=0.1,
    min_nodes=3,
    max_synth_per_class=None,
    results_csv="./classifier/tt_anomaly_results.csv",
    seed=42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    base_dir = os.path.dirname(os.path.abspath(__file__))
    anomaly_dir = os.path.abspath(os.path.join(base_dir, "..", "datasets", "anomaly", "TT"))
    syn_normal_path = syn_normal_path or os.path.join(anomaly_dir, "TT_normal_synthetic.pt")
    syn_abnormal_path = syn_abnormal_path or os.path.join(anomaly_dir, "TT_abnormal_synthetic.pt")
    if generator_name == "random_sampling":
        for label, path in {
            "normal": syn_normal_path,
            "abnormal": syn_abnormal_path,
        }.items():
            if "random_sampling" not in os.path.basename(path):
                raise ValueError(
                    f"random_sampling generator received non-random {label} path: {path}"
                )

    train_ds, val_ds, test_ds, summary = prepare_tt_anomaly_datasets(
        syn_normal_path=syn_normal_path,
        syn_abnormal_path=syn_abnormal_path,
        real_normal_path=os.path.join(anomaly_dir, "TT_normal.pt"),
        real_abnormal_path=os.path.join(anomaly_dir, "TT_abnormal.pt"),
        real_train_ratio=real_train_ratio,
        real_percent=real_percent,
        val_ratio=val_ratio,
        real_holdout_ratio=real_holdout_ratio,
        real_val_ratio=real_val_ratio,
        max_synth_per_class=max_synth_per_class,
        min_nodes=min_nodes,
        seed=seed,
    )
    _validate_zero_real_setup(summary, real_percent)
    print("Dataset summary:", summary)
    train_total = summary["train_real_total"] + summary["train_synth_total"]
    if train_total > 0:
        actual_real_pct = 100.0 * summary["train_real_total"] / train_total
        actual_synth_pct = 100.0 * summary["train_synth_total"] / train_total
        print(
            f"Training mix: real={actual_real_pct:.1f}% synth={actual_synth_pct:.1f}%"
        )

    train_graphs = train_ds.graphs
    real_graphs = load_graphs(os.path.join(anomaly_dir, "TT_normal.pt")) + load_graphs(
        os.path.join(anomaly_dir, "TT_abnormal.pt")
    )

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

    class_weights = _class_weights(train_graphs).to(device)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    test_loader = DataLoader(test_ds, batch_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val = None
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, device, optimizer, class_weights
        )
        val_loss, val_acc = _run_epoch(model, val_loader, device)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), weights_path)
        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f}/{train_acc:.3f} | "
            f"val {val_loss:.4f}/{val_acc:.3f}"
        )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_loss, test_acc = _run_epoch(model, test_loader, device)
    print(f"Test {test_loss:.4f}/{test_acc:.3f}")
    cm = _confusion_matrix(model, test_loader, device, num_classes=2)
    prf = _per_class_prf(cm)
    weighted_precision, weighted_recall, weighted_f1 = _weighted_prf(cm)
    y_true, y_score = _prediction_scores(model, test_loader, device)
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = 0.0
    tn = int(cm[0, 0].item())
    fp = int(cm[0, 1].item())
    fn = int(cm[1, 0].item())
    tp = int(cm[1, 1].item())
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(
        "Per-class PRF: "
        f"Normal Precision={prf[0][0]:.3f} Recall={prf[0][1]:.3f} F1={prf[0][2]:.3f} | "
        f"Abnormal Precision={prf[1][0]:.3f} Recall={prf[1][1]:.3f} F1={prf[1][2]:.3f}"
    )
    print(
        "Weighted Overall: "
        f"Precision={weighted_precision:.3f} "
        f"Recall={weighted_recall:.3f} "
        f"F1={weighted_f1:.3f}"
    )

    train_total = summary["train_real_total"] + summary["train_synth_total"]
    actual_real_pct = 100.0 * summary["train_real_total"] / train_total if train_total else 0.0
    actual_synth_pct = 100.0 * summary["train_synth_total"] / train_total if train_total else 0.0
    result_row = {
        "mode_version": "v5_tt_real_validation",
        "seed": seed,
        "generator": generator_name,
        "syn_normal_path": syn_normal_path,
        "syn_abnormal_path": syn_abnormal_path,
        "real_percent_requested": "" if real_percent is None else real_percent,
        "real_percent_actual": round(actual_real_pct, 3),
        "synth_percent_actual": round(actual_synth_pct, 3),
        "real_holdout_ratio": real_holdout_ratio,
        "real_val_ratio": real_val_ratio,
        "min_nodes_filter": min_nodes,
        "max_synth_per_class": "" if max_synth_per_class is None else max_synth_per_class,
        "train_real_total": summary["train_real_total"],
        "train_synth_total": summary["train_synth_total"],
        "train_normal_real": summary["train_normal_real"],
        "train_abnormal_real": summary["train_abnormal_real"],
        "train_normal_synth": summary["train_normal_synth"],
        "train_abnormal_synth": summary["train_abnormal_synth"],
        "val_normal_synth": summary["val_normal_synth"],
        "val_abnormal_synth": summary["val_abnormal_synth"],
        "val_normal_real": summary["val_normal_real"],
        "val_abnormal_real": summary["val_abnormal_real"],
        "test_normal_real": summary["test_normal_real"],
        "test_abnormal_real": summary["test_abnormal_real"],
        "train_total_after_filter": summary["train_total_after_filter"],
        "val_total_after_filter": summary["val_total_after_filter"],
        "test_total_after_filter": summary["test_total_after_filter"],
        "test_normal_real_after_filter": summary["test_normal_real_after_filter"],
        "test_abnormal_real_after_filter": summary["test_abnormal_real_after_filter"],
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_acc, 6),
        "weighted_precision": round(weighted_precision, 6),
        "weighted_recall": round(weighted_recall, 6),
        "weighted_f1": round(weighted_f1, 6),
        "normal_precision": round(prf[0][0], 6),
        "normal_recall": round(prf[0][1], 6),
        "normal_f1": round(prf[0][2], 6),
        "abnormal_precision": round(prf[1][0], 6),
        "abnormal_recall": round(prf[1][1], 6),
        "abnormal_f1": round(prf[1][2], 6),
        "pr_auc": round(pr_auc, 6),
        "false_alarm_rate": round(false_alarm_rate, 6),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    _append_results_csv(results_csv, result_row)
    print(f"Saved results row to {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TT normal-vs-abnormal graph classifier."
    )
    parser.add_argument(
        "--weights-path",
        default="./tt_anomaly_classifier_weights.pt",
        help="Where to save the best classifier weights.",
    )
    parser.add_argument(
        "--syn-normal-path",
        default=None,
        help="Synthetic normal .pt dataset. Defaults to hierarchical VAE TT path.",
    )
    parser.add_argument(
        "--syn-abnormal-path",
        default=None,
        help="Synthetic abnormal .pt dataset. Defaults to hierarchical VAE TT path.",
    )
    parser.add_argument(
        "--generator-name",
        default="hierarchical_vae",
        help="Name recorded in the result CSV for this generator.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum classifier training epochs.",
    )
    parser.add_argument(
        "--real-train-ratio",
        type=float,
        default=0.1,
        help="Legacy mode: fraction of real normal and real abnormal graphs to include in training.",
    )
    parser.add_argument(
        "--real",
        type=float,
        default=None,
        help="Percent of training data to draw from real graphs; synthetic uses the remainder.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of synthetic graphs to reserve for validation.",
    )
    parser.add_argument(
        "--real-holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of real normal and abnormal graphs to hold out for testing when using --real.",
    )
    parser.add_argument(
        "--real-val-ratio",
        type=float,
        default=0.1,
        help="Fraction of the non-test real pool to reserve for validation when using --real.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=3,
        help="Filter TT traces below this node count before training/validation/testing.",
    )
    parser.add_argument(
        "--max-synth-per-class",
        type=int,
        default=None,
        help="Cap synthetic normal and synthetic abnormal graphs before splitting.",
    )
    parser.add_argument(
        "--results-csv",
        default="./classifier/tt_anomaly_results.csv",
        help="CSV file to append run metrics to.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        weights_path=args.weights_path,
        syn_normal_path=args.syn_normal_path,
        syn_abnormal_path=args.syn_abnormal_path,
        generator_name=args.generator_name,
        epochs=args.epochs,
        real_train_ratio=args.real_train_ratio,
        real_percent=args.real,
        val_ratio=args.val_ratio,
        real_holdout_ratio=args.real_holdout_ratio,
        real_val_ratio=args.real_val_ratio,
        min_nodes=args.min_nodes,
        max_synth_per_class=args.max_synth_per_class,
        results_csv=args.results_csv,
        seed=args.seed,
    )
