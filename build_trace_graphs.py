import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from tqdm import tqdm

PATH = "./03_53_trace.csv"


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


df = load_trace_data(PATH)
df, pod, op = encode(df)

graphs = build_graphs(df)  # To use for visualization
data = [graph_to_data(t_id, G) for t_id, G in graphs]  # To use for models

print(f"\n✅ Created {len(data)} GNN-ready graphs.")
print("Example graph:")
print(data[0])
print(f"Total data: {len(data)}")

torch.save(data, "trace_data.pt")
t_1, G_1 = graphs[0]
display_graph(t_1, G_1)
