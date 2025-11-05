import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def parse_trace_csv_to_graph(trace_csv, label):
    df = pd.read_csv(trace_csv)

    # Normalize column names (just in case of case differences)
    df.columns = [c.strip().lower() for c in df.columns]

    # Expecting these columns based on your schema
    required = [
        "spanid",
        "parentid",
        "duration",
        "starttimeunixnano",
        "endtimeunixnano",
    ]
    for r in required:
        if r not in df.columns:
            raise KeyError(f"Missing required column '{r}' in {trace_csv}")

    # Build node mapping
    node_ids = df["spanid"].astype(str).tolist()
    id_to_idx = {sid: i for i, sid in enumerate(node_ids)}

    # Build edges (parent → child)
    edge_src, edge_dst = [], []
    for _, row in df.iterrows():
        parent = str(row["parentid"])
        if parent != "root" and parent in id_to_idx:
            edge_src.append(id_to_idx[parent])
            edge_dst.append(id_to_idx[str(row["spanid"])])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    # Build node features (Duration + relative timing)
    start = df["starttimeunixnano"].to_numpy(dtype=float)
    end = df["endtimeunixnano"].to_numpy(dtype=float)
    dur = df["duration"].to_numpy(dtype=float)

    # normalize to seconds
    start_rel = (start - start.min()) / 1e9
    end_rel = (end - start.min()) / 1e9
    dur_norm = (dur - dur.min()) / (dur.max() - dur.min() + 1e-9)

    # Combine into node feature matrix [start_time, end_time, duration]
    x = torch.tensor(list(zip(start_rel, end_rel, dur_norm)), dtype=torch.float)

    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


class TraceGraphDataset(InMemoryDataset):
    def __init__(
        self, normal_trace_path, abnormal_trace_path, transform=None, pre_transform=None
    ):
        self.normal_trace_path = normal_trace_path
        self.abnormal_trace_path = abnormal_trace_path
        super().__init__(".", transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        data_list = []

        def load_traces(root, label):
            for date in os.listdir(root):
                trace_dir = os.path.join(root, date, "trace")
                if not os.path.isdir(trace_dir):
                    continue
                for file in os.listdir(trace_dir):
                    if file.endswith(".csv"):
                        path = os.path.join(trace_dir, file)
                        try:
                            g = parse_trace_csv_to_graph(path, label)
                            data_list.append(g)
                        except Exception as e:
                            print(f"⚠️ Skipping {path}: {e}")

        load_traces(self.normal_trace_path, 0)  # 0 -> Normal
        load_traces(self.abnormal_trace_path, 1)  # 1 -> Abnormal

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


dataset = TraceGraphDataset(
    normal_trace_path="./normal_trace/", abnormal_trace_path="./abnormal_trace/"
)
print(len(dataset))
