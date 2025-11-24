import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from GCN import GCN


class LoadDataset(InMemoryDataset):
    def __init__(self, datapath="./data.pt") -> None:
        super().__init__(".")
        data, slices = torch.load(datapath)
        self.data, self.slices = data, slices
        x = data.x
        self.num_pods = int(x[:, 0].max().item()) + 1
        self.num_ops = int(x[:, 1].max().item()) + 1
        log_duration = torch.log1p(torch.clamp(x[:, 2], min=0))
        self.duration_mean = log_duration.mean().item()
        self.duration_std = log_duration.std().item() or 1.0

    def get(self, idx):
        data = super().get(idx)
        data.edge_index = to_undirected(data.edge_index)
        return data


def summarize_dataset(dataset):

    # num_graphs = data["y"].size(0)
    print("=== DATASET SUMMARY ===")
    print(f"Total graphs: {len(dataset)}")

    labels = []
    total_nodes = 0

    for data in dataset:
        if hasattr(data, "y") and data.y is not None:
            if data.y.numel() == 1:
                labels.append(int(data.y.item()))
            else:
                labels.extend(data.y.tolist())

        total_nodes += data.num_nodes

    if len(labels) == 0:
        print("No Labels found in dataset")
        return

    y = torch.tensor(labels)
    unique_labels, counts = torch.unique(y, return_counts=True)

    print("\nLabel distribution:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        pct = 100.0 * count / len(y)
        print(f"  Label {label}: {count} graphs ({pct:.1f}%)")

    print(f"\nNumber of unique labels: {len(unique_labels)}")
    print(f"Total nodes across all graphs: {total_nodes}")
    print(f"Average nodes per graph: {total_nodes / len(dataset):.2f}")


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Safely split a dataset into train and validation sets.
    Returns (train_dataset, val_dataset)
    """
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Dataset must contain at least two graphs to split.")

    train_len = int(n_total * train_ratio)
    val_len = n_total - train_len  # ensures total matches exactly

    # Fix edge cases
    if train_len == 0:
        train_len = 1
        val_len = n_total - 1
    elif val_len == 0:
        val_len = 1
        train_len = n_total - 1

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_len, val_len], generator=generator
    )

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Dataloaders ready - batch size: {batch_size}")
    return train_loader, val_loader


def train(model, train_loader, optimizer, device):

    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)

        if out.shape[-1] == 1:
            out = out.view(-1)
            loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        else:
            loss = F.cross_entropy(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)

        if logits.shape[-1] == 1:
            logits = logits.view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())
            preds = (torch.sigmoid(logits) > 0.5).long()

        else:
            loss = F.cross_entropy(logits, batch.y)
            preds = logits.argmax(dim=1)

        total_loss += loss.item()
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0
    # print(f"Val Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}%")
    return avg_loss, acc


def run_training(model, train_loader, val_loader, device, epochs=50, lr=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model = model.to(device)

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_loss = train(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Val Loss:{val_loss:.4f} Accuracy: {val_acc*100:.2f}")
    print("\n✅ Training complete.")


def main():
    datapath = "./data.pt"
    dataset = LoadDataset(datapath)
    summarize_dataset(dataset)
    print(f"Unique Pods: {dataset.num_pods}, Unique Operations: {dataset.num_ops}")

    train_ds, val_ds = split_dataset(dataset)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds)
    print(f"Total dataset: {len(dataset)}")
    print(f"Total train dataset: {len(train_ds)}")
    print(f"Total validate dataset: {len(val_ds)}")
    print(f"Total sum: {len(val_ds) + len(train_ds)}")

    print("=== Training Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(
        num_pods=dataset.num_pods,
        num_ops=dataset.num_ops,
        duration_mean=dataset.duration_mean,
        duration_std=dataset.duration_std,
    ).to(device)
    run_training(model, train_loader, val_loader, device, lr=1e-3)


if __name__ == "__main__":
    main()
