import argparse
import csv
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


ROOT = Path(__file__).resolve().parent


class PTDataset(InMemoryDataset):
    def __init__(self, datapath: str) -> None:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".")
        self.data, self.slices = torch.load(datapath, weights_only=False)


def load_graphs(path: str):
    ds = PTDataset(str(ROOT / path))
    return [ds.get(i).cpu() for i in range(len(ds))]


def summarize(system, cls, path, thresholds):
    graphs = load_graphs(path)
    total = len(graphs)
    rows = []
    for threshold in thresholds:
        kept = sum(1 for g in graphs if g.num_nodes <= threshold)
        rows.append(
            {
                "system": system,
                "trace_class": cls,
                "threshold": threshold,
                "total_graphs": total,
                "kept_graphs": kept,
                "filtered_graphs": total - kept,
                "kept_percent": 100.0 * kept / total if total else 0.0,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, default=[30, 50])
    parser.add_argument("--out-csv", default="./classifier/filtering_sensitivity.csv")
    args = parser.parse_args()

    jobs = [
        ("TT", "normal", "./datasets/anomaly/TT/TT_normal.pt"),
        ("TT", "abnormal", "./datasets/anomaly/TT/TT_abnormal.pt"),
        ("SN", "normal", "./datasets/anomaly/SN/SN_normal.pt"),
        ("SN", "abnormal", "./datasets/anomaly/SN/SN_abnormal.pt"),
    ]
    rows = []
    for job in jobs:
        rows.extend(summarize(*job, thresholds=args.thresholds))
    out = ROOT / args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
