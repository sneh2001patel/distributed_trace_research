import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from build_trace_graphs import build_encoders, df_to_graphs


ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = ROOT / "preprocessed"
OUTPUT_DIR = ROOT / "datasets" / "anomaly"


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "source_run" not in df.columns:
        raise KeyError(f"Missing source_run column in {path}")
    # Trace IDs can repeat across different runs, especially in no-fault archives.
    # Use a composite trace key so each execution becomes its own graph.
    df["trace_id"] = df["source_run"].astype(str) + "::" + df["trace_id"].astype(str)
    return df


def _save_graphs(graphs: List[Data], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data, slices = InMemoryDataset.collate(graphs)
    torch.save((data, slices), out_path)


def _build_system_graphs(system: str) -> Tuple[Dict[str, List[Data]], dict]:
    normal_df = _load_csv(PREPROCESSED_DIR / f"{system}_normal_spans.csv")
    abnormal_df = _load_csv(PREPROCESSED_DIR / f"{system}_abnormal_spans.csv")

    service_to_id, op_to_id = build_encoders([normal_df, abnormal_df])

    normal_graphs = df_to_graphs(
        normal_df, label=0, service_to_id=service_to_id, op_to_id=op_to_id
    )
    abnormal_graphs = df_to_graphs(
        abnormal_df, label=1, service_to_id=service_to_id, op_to_id=op_to_id
    )

    graphs = {
        "normal": normal_graphs,
        "abnormal": abnormal_graphs,
        "combined": normal_graphs + abnormal_graphs,
    }
    metadata = {
        "system": system,
        "num_services": len(service_to_id),
        "num_operations": len(op_to_id),
        "normal_graphs": len(normal_graphs),
        "abnormal_graphs": len(abnormal_graphs),
        "service_to_id": service_to_id,
        "op_to_id": op_to_id,
    }
    return graphs, metadata


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for system in ["SN", "TT"]:
        graphs, metadata = _build_system_graphs(system)
        system_dir = OUTPUT_DIR / system

        normal_path = system_dir / f"{system}_normal.pt"
        abnormal_path = system_dir / f"{system}_abnormal.pt"
        combined_path = system_dir / f"{system}_normal_abnormal.pt"
        metadata_path = system_dir / f"{system}_metadata.json"

        _save_graphs(graphs["normal"], normal_path)
        _save_graphs(graphs["abnormal"], abnormal_path)
        _save_graphs(graphs["combined"], combined_path)
        metadata_path.write_text(json.dumps(metadata, indent=2))

        summary[system] = {
            "normal_dataset": str(normal_path.relative_to(ROOT)),
            "abnormal_dataset": str(abnormal_path.relative_to(ROOT)),
            "combined_dataset": str(combined_path.relative_to(ROOT)),
            "metadata": str(metadata_path.relative_to(ROOT)),
            "normal_graphs": metadata["normal_graphs"],
            "abnormal_graphs": metadata["abnormal_graphs"],
            "num_services": metadata["num_services"],
            "num_operations": metadata["num_operations"],
        }

        print(f"\n{system}")
        print(f"  normal graphs:   {metadata['normal_graphs']}")
        print(f"  abnormal graphs: {metadata['abnormal_graphs']}")
        print(f"  services: {metadata['num_services']} | operations: {metadata['num_operations']}")
        print(f"  wrote {normal_path.relative_to(ROOT)}")
        print(f"  wrote {abnormal_path.relative_to(ROOT)}")
        print(f"  wrote {combined_path.relative_to(ROOT)}")

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
