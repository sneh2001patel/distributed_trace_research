import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


DATA_PATHS = {
    ("TT", "normal"): "./datasets/anomaly/TT/TT_normal.pt",
    ("TT", "abnormal"): "./datasets/anomaly/TT/TT_abnormal.pt",
    ("SN", "normal"): "./datasets/anomaly/SN/SN_normal.pt",
    ("SN", "abnormal"): "./datasets/anomaly/SN/SN_abnormal.pt",
}


HIERARCHICAL_TRAIN_SCRIPTS = {
    ("TT", "normal"): "./train_tt_normal_data.py",
    ("TT", "abnormal"): "./train_tt_abnormal_data.py",
    ("SN", "normal"): "./train_sn_normal_data.py",
    ("SN", "abnormal"): "./train_sn_abnormal_data.py",
}


FLAT_WEIGHT_PATHS = {
    ("TT", "normal"): "./weights/tt_normal_flat_vae_weights.pt",
    ("TT", "abnormal"): "./weights/tt_abnormal_flat_vae_weights.pt",
    ("SN", "normal"): "./weights/sn_normal_flat_vae_weights.pt",
    ("SN", "abnormal"): "./weights/sn_abnormal_flat_vae_weights.pt",
}


SUMMARY_FIELDNAMES = [
    "run_order",
    "model",
    "system",
    "trace_class",
    "weights_path",
    "output_log",
    "total_nodes_evaluated",
    "service_accuracy_pct",
    "op_accuracy_pct",
    "duration_mae",
    "edge_precision_pct",
    "edge_recall_pct",
    "edge_f1_pct",
    "resource_summary_csv",
    "resource_timeseries_csv",
    "return_code",
    "wall_seconds",
    "process_tree_cpu_seconds",
    "process_tree_avg_cpu_pct",
    "process_tree_peak_rss_mb",
    "process_tree_peak_vms_mb",
    "gpu_peak_mem_used_total_mb",
    "process_tree_gpu_peak_mem_mb",
    "gpu_peak_util_pct",
    "samples",
    "command",
]


def _job_slug(run_order, model, system, trace_class):
    return f"run_{run_order:03d}_{system}_{trace_class}_{model}"


def _resource_paths(resource_dir: Path, slug: str):
    return {
        "output_log": resource_dir / f"{slug}_training.log",
        "summary_csv": resource_dir / f"{slug}_resource_summary.csv",
        "summary_json": resource_dir / f"{slug}_resource_summary.json",
        "timeseries_csv": resource_dir / f"{slug}_resource_timeseries.csv",
    }


def _read_resource_summary(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No resource summary row was written to {path}")
    return rows[-1]


def _parse_reconstruction_metrics(log_path: Path):
    text = log_path.read_text(errors="replace")
    marker = "=== Reconstruction Evaluation ==="
    marker_idx = text.rfind(marker)
    section = text[marker_idx:] if marker_idx != -1 else text

    patterns = {
        "total_nodes_evaluated": r"Total nodes evaluated:\s*([0-9]+)",
        "service_accuracy_pct": r"Service accuracy:\s*([0-9.]+)%",
        "op_accuracy_pct": r"Op accuracy:\s*([0-9.]+)%",
        "duration_mae": r"Duration MAE:\s*([0-9.eE+-]+)",
        "edge_precision_pct": r"Edge precision:\s*([0-9.]+)%",
        "edge_recall_pct": r"Edge recall:\s*([0-9.]+)%",
        "edge_f1_pct": r"Edge F1:\s*([0-9.]+)%",
    }

    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, section)
        metrics[key] = match.group(1) if match else ""
    return metrics


def _run_with_log(command, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _build_hierarchical_command(system, trace_class, epochs):
    command = [sys.executable, HIERARCHICAL_TRAIN_SCRIPTS[(system, trace_class)]]
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    return command


def _build_flat_command(
    system,
    trace_class,
    epochs,
    target_graphs,
    max_graph_nodes,
    eval_edge_threshold,
):
    command = [
        sys.executable,
        "./train_flat_vae.py",
        "--data-path",
        DATA_PATHS[(system, trace_class)],
        "--out-weights",
        FLAT_WEIGHT_PATHS[(system, trace_class)],
    ]
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if target_graphs is not None:
        command.extend(["--target-graphs", str(target_graphs)])
    if max_graph_nodes is not None:
        command.extend(["--max-graph-nodes", str(max_graph_nodes)])
    if eval_edge_threshold is not None:
        command.extend(["--eval-edge-threshold", str(eval_edge_threshold)])
    return command


def _build_command(args, model, system, trace_class):
    if model == "hierarchical_vae":
        return _build_hierarchical_command(system, trace_class, args.epochs)
    if model == "flat_vae":
        return _build_flat_command(
            system,
            trace_class,
            args.epochs,
            args.flat_target_graphs,
            args.flat_max_graph_nodes,
            args.flat_eval_edge_threshold,
        )
    raise ValueError(f"Unsupported model: {model}")


def _weights_path(model, system, trace_class):
    if model == "flat_vae":
        return FLAT_WEIGHT_PATHS[(system, trace_class)]
    if model == "hierarchical_vae":
        prefix = system.lower()
        return f"./weights/{prefix}_{trace_class}_vae_weights.pt"
    raise ValueError(f"Unsupported model: {model}")


def _wrap_with_monitor(command, sample_interval, paths):
    return [
        sys.executable,
        "monitor_resources.py",
        "--sample-interval",
        str(sample_interval),
        "--summary-csv",
        str(paths["summary_csv"]),
        "--timeseries-csv",
        str(paths["timeseries_csv"]),
        "--summary-json",
        str(paths["summary_json"]),
        "--",
        *command,
    ]


def _write_summary(path: Path, rows, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = append and path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_summary_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _next_run_order(rows):
    run_orders = []
    for row in rows:
        try:
            run_orders.append(int(row.get("run_order", "")))
        except ValueError:
            continue
    return max(run_orders, default=0) + 1


def _fmt(value):
    if value in ("", None):
        return ""
    return f"{float(value):.3f}"


def _write_markdown(path: Path, rows):
    lines = [
        "# VAE Training Resource Usage",
        "",
        "One row is recorded for each VAE training job. Timelines are written as per-run CSV files.",
        "",
        "| Model | System | Class | Service Acc. % | Op Acc. % | Duration MAE | Edge Precision % | Edge Recall % | Edge F1 % | Wall sec | Avg CPU % | Peak RAM MB | Process GPU MB | Total GPU MB | Peak GPU % | Log | Timeline |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["system"],
                    row["trace_class"],
                    _fmt(row["service_accuracy_pct"]),
                    _fmt(row["op_accuracy_pct"]),
                    _fmt(row["duration_mae"]),
                    _fmt(row["edge_precision_pct"]),
                    _fmt(row["edge_recall_pct"]),
                    _fmt(row["edge_f1_pct"]),
                    _fmt(row["wall_seconds"]),
                    _fmt(row["process_tree_avg_cpu_pct"]),
                    _fmt(row["process_tree_peak_rss_mb"]),
                    _fmt(row["process_tree_gpu_peak_mem_mb"]),
                    _fmt(row["gpu_peak_mem_used_total_mb"]),
                    _fmt(row["gpu_peak_util_pct"]),
                    row["output_log"],
                    row["resource_timeseries_csv"],
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train hierarchical and flat VAEs while recording CPU, RAM, and GPU usage."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["hierarchical_vae", "flat_vae"],
        default=["hierarchical_vae", "flat_vae"],
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["TT", "SN"],
        default=["TT", "SN"],
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=["normal", "abnormal"],
        default=["normal", "abnormal"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override VAE training epochs. Defaults to each script's default.",
    )
    parser.add_argument("--flat-target-graphs", type=int, default=None)
    parser.add_argument("--flat-max-graph-nodes", type=int, default=None)
    parser.add_argument(
        "--flat-eval-edge-threshold",
        type=float,
        default=None,
        help="Override flat VAE reconstruction edge threshold. Defaults to train_flat_vae.py's value.",
    )
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument(
        "--resource-dir",
        default="./classifier/vae_training_resource_by_run",
    )
    parser.add_argument(
        "--summary-csv",
        default="./classifier/vae_training_resource_summary.csv",
    )
    parser.add_argument(
        "--summary-md",
        default="./classifier/vae_training_resource_tables.md",
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    resource_dir = ROOT / args.resource_dir
    summary_csv = ROOT / args.summary_csv
    summary_md = ROOT / args.summary_md
    if not args.dry_run:
        resource_dir.mkdir(parents=True, exist_ok=True)
        if not args.append:
            summary_csv.unlink(missing_ok=True)
            summary_md.unlink(missing_ok=True)

    existing_rows = _read_summary_rows(summary_csv) if args.append else []
    next_run_order = _next_run_order(existing_rows)
    jobs = []
    for model in args.models:
        for system in args.systems:
            for trace_class in args.classes:
                run_order = next_run_order + len(jobs)
                slug = _job_slug(run_order, model, system, trace_class)
                paths = _resource_paths(resource_dir, slug)
                command = _build_command(args, model, system, trace_class)
                monitored_command = _wrap_with_monitor(
                    command,
                    args.sample_interval,
                    paths,
                )
                jobs.append(
                    {
                        "run_order": run_order,
                        "model": model,
                        "system": system,
                        "trace_class": trace_class,
                        "weights_path": _weights_path(model, system, trace_class),
                        "paths": paths,
                        "command": command,
                        "monitored_command": monitored_command,
                    }
                )

    print(f"Prepared {len(jobs)} VAE training jobs.")
    rows = []
    for idx, job in enumerate(jobs, start=1):
        print(f"\n[{idx}/{len(jobs)}] {' '.join(job['command'])}")
        print(
            "Resource tracking: "
            f"{job['paths']['summary_csv']} and {job['paths']['timeseries_csv']}"
        )
        if args.dry_run:
            continue

        _run_with_log(job["monitored_command"], job["paths"]["output_log"])
        summary = _read_resource_summary(job["paths"]["summary_csv"])
        metrics = _parse_reconstruction_metrics(job["paths"]["output_log"])
        row = {
            "run_order": job["run_order"],
            "model": job["model"],
            "system": job["system"],
            "trace_class": job["trace_class"],
            "weights_path": job["weights_path"],
            "output_log": str(job["paths"]["output_log"]),
            **metrics,
            "resource_summary_csv": str(job["paths"]["summary_csv"]),
            "resource_timeseries_csv": str(job["paths"]["timeseries_csv"]),
            "return_code": summary.get("return_code", ""),
            "wall_seconds": summary.get("wall_seconds", ""),
            "process_tree_cpu_seconds": summary.get("process_tree_cpu_seconds", ""),
            "process_tree_avg_cpu_pct": summary.get("process_tree_avg_cpu_pct", ""),
            "process_tree_peak_rss_mb": summary.get("process_tree_peak_rss_mb", ""),
            "process_tree_peak_vms_mb": summary.get("process_tree_peak_vms_mb", ""),
            "gpu_peak_mem_used_total_mb": summary.get("gpu_peak_mem_used_total_mb", ""),
            "process_tree_gpu_peak_mem_mb": summary.get(
                "process_tree_gpu_peak_mem_mb",
                "",
            ),
            "gpu_peak_util_pct": summary.get("gpu_peak_util_pct", ""),
            "samples": summary.get("samples", ""),
            "command": " ".join(job["command"]),
        }
        rows.append(row)
        _write_summary(summary_csv, [row], append=True)

    if args.dry_run:
        print("\nDry run only. No training or resource files were written.")
        return

    _write_markdown(summary_md, existing_rows + rows)
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote per-run timelines under {resource_dir}")


if __name__ == "__main__":
    main()
