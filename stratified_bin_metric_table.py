import argparse
import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from rq4_stratified_representativeness import (
    DATASET_PRESETS,
    ROOT,
    SPAN_BINS,
    categorical_jsd,
    flatten,
    group_by_bin,
    load_graphs,
    mean_abs_diff,
    numeric_wasserstein,
    random_sample_records,
    stratified_sample,
    to_records,
)


BIN_LABELS = {
    "small": "1-5",
    "medium": "6-10",
    "large": "11-20",
    "very_large": ">20",
}


def graph_branching_factor(record: Dict[str, object]) -> float:
    graph = record["graph"]
    n = int(graph.num_nodes)
    if n == 0:
        return 0.0
    edge_index = graph.edge_index.detach().cpu().long()
    if edge_index.numel() == 0:
        return 0.0
    out_degree = np.bincount(edge_index[0].numpy(), minlength=n)
    parent_degrees = out_degree[out_degree > 0]
    if parent_degrees.size == 0:
        return 0.0
    return float(parent_degrees.mean())


def service_operation_joint(records: Sequence[Dict[str, object]]) -> np.ndarray:
    services = flatten(records, "services")
    ops = flatten(records, "ops")
    max_op = int(ops.max(initial=0)) + 1
    return services.astype(np.int64) * max_op + ops.astype(np.int64)


def bin_metrics(
    real_records: Sequence[Dict[str, object]],
    synthetic_records: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    real_branching = np.asarray([graph_branching_factor(record) for record in real_records])
    synth_branching = np.asarray(
        [graph_branching_factor(record) for record in synthetic_records]
    )
    return {
        "Service Dist. (JSD)": categorical_jsd(
            flatten(real_records, "services"), flatten(synthetic_records, "services")
        ),
        "Operation Dist. (JSD)": categorical_jsd(
            flatten(real_records, "ops"), flatten(synthetic_records, "ops")
        ),
        "Service-Operation Dist. (JSD)": categorical_jsd(
            service_operation_joint(real_records),
            service_operation_joint(synthetic_records),
        ),
        "Duration Dist. (Wass. log1p)": numeric_wasserstein(
            flatten(real_records, "durations"), flatten(synthetic_records, "durations")
        ),
        "Duration Mean (MAE log1p)": mean_abs_diff(
            flatten(real_records, "durations"), flatten(synthetic_records, "durations")
        ),
        "Edge Count (MAE)": mean_abs_diff(
            flatten(real_records, "edge_count"), flatten(synthetic_records, "edge_count")
        ),
        "Graph Depth (MAE)": mean_abs_diff(
            flatten(real_records, "depth"), flatten(synthetic_records, "depth")
        ),
        "Branching Factor (MAE)": mean_abs_diff(real_branching, synth_branching),
    }


def build_system_table(
    system: str,
    preset: str,
    seed: int,
    real_sample_size: int | None,
    target_count: int | None,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, int], Dict[str, int]]:
    paths = DATASET_PRESETS[preset][system]
    real_records = to_records(load_graphs(paths["real"]))
    synthetic_records = to_records(load_graphs(paths["synthetic"]))
    real_records = random_sample_records(real_records, real_sample_size, seed)
    sampled_synthetic, targets, shortfalls = stratified_sample(
        real_records, synthetic_records, seed=seed, target_count=target_count
    )

    real_by_bin = group_by_bin(real_records)
    synth_by_bin = group_by_bin(sampled_synthetic)
    table = {}
    for name, _, _ in SPAN_BINS:
        table[name] = bin_metrics(real_by_bin.get(name, []), synth_by_bin.get(name, []))
    return table, targets, shortfalls


def fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.1f}"
    return f"{value:.3f}"


def write_markdown(
    system_tables: Dict[str, Dict[str, Dict[str, float]]],
    system_targets: Dict[str, Dict[str, int]],
    system_shortfalls: Dict[str, Dict[str, int]],
    out_path: Path,
) -> None:
    metric_names = list(next(iter(next(iter(system_tables.values())).values())).keys())
    bin_names = [name for name, _, _ in SPAN_BINS]

    lines = [
        "# RQ4 Stratified Bin Metric Table",
        "",
        "Real traces are randomly sampled first, then synthetic traces are sampled to match the real span-bin proportions. Lower JSD, Wasserstein distance, and MAE values indicate better representativeness within each stratum.",
        "",
    ]
    for system, table in system_tables.items():
        lines.append(f"## {system}")
        lines.append("")
        target_text = ", ".join(
            f"{BIN_LABELS[name]}={system_targets[system].get(name, 0)}"
            for name in bin_names
        )
        shortfall_text = ", ".join(
            f"{BIN_LABELS[name]}={system_shortfalls[system].get(name, 0)}"
            for name in bin_names
        )
        lines.append(f"Sampled synthetic bin counts: {target_text}.")
        lines.append(f"Synthetic bin shortfalls filled with replacement/fallback: {shortfall_text}.")
        lines.append("")
        lines.append("| Metric | " + " | ".join(BIN_LABELS[name] for name in bin_names) + " |")
        lines.append("|---" + "|---:" * len(bin_names) + "|")
        for metric in metric_names:
            values = [fmt(table[name][metric]) for name in bin_names]
            lines.append("| " + metric + " | " + " | ".join(values) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines))


def write_csv(system_tables: Dict[str, Dict[str, Dict[str, float]]], out_path: Path) -> None:
    bin_names = [name for name, _, _ in SPAN_BINS]
    metric_names = list(next(iter(next(iter(system_tables.values())).values())).keys())
    fieldnames = ["system", "metric"] + [BIN_LABELS[name] for name in bin_names]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system, table in system_tables.items():
            for metric in metric_names:
                row = {"system": system, "metric": metric}
                row.update({BIN_LABELS[name]: table[name][metric] for name in bin_names})
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified metric tables with metrics as rows and span bins as columns."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DATASET_PRESETS),
        default="anomaly_combined",
    )
    parser.add_argument("--systems", nargs="+", choices=["SN", "TT"], default=["SN", "TT"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--real-sample-size",
        type=int,
        default=1000,
        help="Random sample size for real traces before synthetic stratified sampling.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Synthetic sample size per system. Defaults to sampled real trace count.",
    )
    parser.add_argument(
        "--out-csv",
        default="./classifier/rq4_stratified_bin_metric_table.csv",
    )
    parser.add_argument(
        "--out-md",
        default="./classifier/rq4_stratified_bin_metric_table.md",
    )
    args = parser.parse_args()

    system_tables = {}
    system_targets = {}
    system_shortfalls = {}
    for system in args.systems:
        table, targets, shortfalls = build_system_table(
            system=system,
            preset=args.preset,
            seed=args.seed,
            real_sample_size=args.real_sample_size,
            target_count=args.target_count,
        )
        system_tables[system] = table
        system_targets[system] = targets
        system_shortfalls[system] = shortfalls
        print(
            f"{system}: "
            + ", ".join(
                f"{BIN_LABELS[name]}={targets.get(name, 0)}"
                for name, _, _ in SPAN_BINS
            )
        )

    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(system_tables, out_csv)

    out_md = ROOT / args.out_md
    write_markdown(system_tables, system_targets, system_shortfalls, out_md)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
