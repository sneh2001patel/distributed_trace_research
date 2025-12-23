import os
import random
from typing import List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


def _load_first_csv(path: str) -> pd.DataFrame:
    csv_files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_encoders(dfs: List[pd.DataFrame]) -> Tuple[dict, dict]:
    all_services = sorted({s for df in dfs for s in df["service_name"].astype(str)})
    all_ops = sorted({o for df in dfs for o in df["operation_name"].astype(str)})
    service_to_id = {s: i for i, s in enumerate(all_services)}
    op_to_id = {o: i for i, o in enumerate(all_ops)}
    return service_to_id, op_to_id


def df_to_graphs(
    df: pd.DataFrame, label: int, service_to_id: dict, op_to_id: dict
) -> List[Data]:
    required = [
        "trace_id",
        "span_id",
        "parent_id",
        "service_name",
        "operation_name",
        "duration",
    ]
    for r in required:
        if r not in df.columns:
            raise KeyError(f"Missing required column '{r}'")

    graphs = []
    for trace_id, gdf in df.groupby("trace_id"):
        node_ids = gdf["span_id"].astype(str).tolist()
        id_to_idx = {sid: i for i, sid in enumerate(node_ids)}

        edge_src, edge_dst = [], []
        for _, row in gdf.iterrows():
            parent = str(row["parent_id"])
            if (
                parent
                and parent.lower() not in {"", "root", "none", "nan"}
                and parent in id_to_idx
            ):
                edge_src.append(id_to_idx[parent])
                edge_dst.append(id_to_idx[str(row["span_id"])])

        if edge_src and edge_dst:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        service_ids = [
            service_to_id[str(s)] for s in gdf["service_name"].astype(str).tolist()
        ]
        op_ids = [op_to_id[str(o)] for o in gdf["operation_name"].astype(str).tolist()]
        durations = gdf["duration"].astype(float).to_numpy()

        x = torch.tensor(list(zip(service_ids, op_ids, durations)), dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


class TraceGraphDataset(InMemoryDataset):
    """
    Builds graphs from specified CSV files in each dataset folder.
    One graph per trace_id. Features: [service_id, op_id, duration].
    Labels: dataset index (0 for first dataset, 1 for second, ...).
    Limits to 1244 graphs total.
    """

    def __init__(
        self,
        base_dir: str,
        dataset_names: List[str],
        trace_files: dict[str, str],
        max_graphs: int = 1244,
        transform=None,
        pre_transform=None,
    ):
        self.base_dir = base_dir
        self.dataset_names = dataset_names
        self.trace_files = trace_files
        self.max_graphs = max_graphs
        torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        super().__init__(".", transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ["TT_data.pt"]

    def process(self):
        dfs = []
        for name in self.dataset_names:
            trace_dir = os.path.join(self.base_dir, name, "trace")
            if not os.path.isdir(trace_dir):
                print(f"⚠️ Traces directory not found: {trace_dir}, skipping.")
                continue

            if name not in self.trace_files:
                print(f"⚠️ No trace file specified for dataset {name}, skipping.")
                continue

            try:
                file_path = os.path.join(trace_dir, self.trace_files[name])
                df = pd.read_csv(file_path)
                df.columns = [
                    c.strip().lower() for c in df.columns
                ]  # Normalize columns
                dfs.append(df)
                print(f"✓ Loaded {self.trace_files[name]} from {name}")
            except Exception as e:
                print(f"⚠️ Skipping dataset {name}: {e}")

        if not dfs:
            raise RuntimeError(
                f"No datasets loaded. Checked: {[os.path.join(self.base_dir, n, 'trace') for n in self.dataset_names]}"
            )

        service_to_id, op_to_id = build_encoders(dfs)

        # Convert each dataset to graphs
        dataset_graphs = []
        for label, df in enumerate(dfs):
            try:
                graphs = df_to_graphs(df, label, service_to_id, op_to_id)
                dataset_graphs.append(graphs)
                print(f"✓ Dataset {self.dataset_names[label]}: {len(graphs)} graphs")
            except Exception as e:
                print(f"⚠️ Failed to convert dataset {self.dataset_names[label]}: {e}")
                dataset_graphs.append([])

        # Limit to max_graphs
        data_list = []
        for graphs in dataset_graphs:
            data_list.extend(graphs)

        if len(data_list) > self.max_graphs:
            print(
                f"⚠️ Limiting dataset from {len(data_list)} to {self.max_graphs} graphs"
            )
            random.seed(42)  # For reproducibility
            data_list = random.sample(data_list, self.max_graphs)

        print(f"✓ Final dataset: {len(data_list)} graphs")

        if not data_list:
            raise RuntimeError("No graphs were built after processing trace files.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = TraceGraphDataset(
        base_dir="/home/snehpatel/research/parsed_output",
        dataset_names=["TT_Dataset"],
        trace_files={
            "TT_Dataset": "TT.2022-04-21T153246D2022-04-21T174753_trace.csv",
        },
        max_graphs=1244,  # Limit to 1244 graphs
    )
    print(f"Built dataset with {len(dataset)} graphs")

