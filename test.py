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


# Load csv
def load_trace_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["TraceID", "SpanID"])
    return df


# Encode categorical feat into numeric
def encode(df):
    pod = LabelEncoder()
    op = LabelEncoder()
    df["pod"] = pod.fit_transform(df["PodName"].astype(str))
    df["op"] = op.fit_transform(df["OperationName"].astype(str))
    return df, pod, op


# build graphs
def build_graphs(df):
    graphs = []

    for (
        t_id,
        g_df,
    ) in tqdm(df.groupby("TraceID"), desc="Building graphs"):
        G = nx.DiGraph()

        for _, row in g_df.iterrows():
            node_id = row["SpanID"]
            G.add_node(
                node_id,
                PodName=row["PodName"],
                OperationName=row["OperationName"],
                pod=row["pod"],
                op=row["op"],
                duration=row["Duration"],
            )

            if row["ParentID"] != "root" and row["ParentID"] in g_df["SpanID"].values:
                G.add_edge(row["ParentID"], node_id)

        if len(G) > 1:
            graphs.append((t_id, G))

    return graphs


# convert it to a input for pytorch
def graph_to_data(t_id, G):
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    edge_index = torch.tensor(list(G.edges())).t().contiguous()

    features = []
    for _, attr in G.nodes(data=True):
        features.append([attr["pod"], attr["op"], attr["duration"]])
    x = torch.tensor(features, dtype=torch.float)
    y = torch.zeros(x.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.trace_id = t_id

    return data


def display_graph(t_id, G):

    # Create readable labels: service + operation
    labels = {}
    for node, data in G.nodes(data=True):
        service = data["PodName"].split("-")[0]
        op_name = data["OperationName"].split("/")[-1]
        labels[node] = f"{service.title()}\n{op_name.title()}"

    # Hierarchical layout (top-down tree)
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos, with_labels=False, node_size=3000, node_color="skyblue", arrows=True
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title(f"Trace DAG (Tree Layout) — TraceID={t_id[:8]}...")
    plt.axis("off")
    plt.show()


def parse_csv_to_data(trace_path, label):
    df = load_trace_data(trace_path)
    df, pod, op = encode(df)
    graphs = build_graphs(df)

    print(f"Trace file: {trace_path}")
    data_list = []
    for i, (t_id, G) in enumerate(graphs, 1):
        d = graph_to_data(t_id, G)
        d.y = torch.tensor([label], dtype=torch.long)
        data_list.append(d)
        print(f"Graph {i}: nodes={d.x.size(0)}, edges={d.edge_index.size(1)}")

    print(f"✅ Total graphs created: {len(data_list)}")
    return data_list


dirs = [
    "2022-08-22",
    "2022-08-23",
    "2023-01-29",
    "2023-01-30",
]


dataset = []
for d in dirs:
    normal_path = f"./normal_trace/{d}/trace/"
    abnormal_path = f"./abnormal_trace/{d}/trace/"

    normal_traces = [
        os.path.join(normal_path, f)
        for f in os.listdir(normal_path)
        if f.endswith(".csv")
    ]
    abnormal_traces = [
        os.path.join(abnormal_path, f)
        for f in os.listdir(abnormal_path)
        if f.endswith(".csv")
    ]

    # --- Process all normal traces ---
    for trace_file in normal_traces:
        dataset.extend(parse_csv_to_data(trace_file, label=0))

    # --- Process all abnormal traces ---
    # for trace_file in abnormal_traces:
    #     dataset.extend(parse_csv_to_data(trace_file, label=1))

    if abnormal_traces:
        dataset.extend(parse_csv_to_data(abnormal_traces[0], label=1))


print(f"Total Data objects (graphs): {len(dataset)}")

data, slices = InMemoryDataset.collate(dataset)
torch.save((data, slices), "data.pt")

print("✅ Saved in PyG InMemoryDataset format.")
