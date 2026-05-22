import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


GENERATOR_PATHS = {
    "TT": {
        "random_sampling": {
            "normal": "./datasets/baselines/TT_normal_random_sampling.pt",
            "abnormal": "./datasets/baselines/TT_abnormal_random_sampling.pt",
        },
        "empirical": {
            "normal": "./datasets/baselines/TT_normal_empirical_random_tree.pt",
            "abnormal": "./datasets/baselines/TT_abnormal_empirical_random_tree.pt",
        },
        "flat_vae": {
            "normal": "./datasets/baselines/TT_normal_flat_vae_synthetic.pt",
            "abnormal": "./datasets/baselines/TT_abnormal_flat_vae_synthetic.pt",
        },
        "hierarchical_vae": {
            "normal": "./datasets/anomaly/TT/TT_normal_synthetic.pt",
            "abnormal": "./datasets/anomaly/TT/TT_abnormal_synthetic.pt",
        },
    },
    "SN": {
        "random_sampling": {
            "normal": "./datasets/baselines/SN_normal_random_sampling.pt",
            "abnormal": "./datasets/baselines/SN_abnormal_random_sampling.pt",
        },
        "empirical": {
            "normal": "./datasets/baselines/SN_normal_empirical_random_tree.pt",
            "abnormal": "./datasets/baselines/SN_abnormal_empirical_random_tree.pt",
        },
        "flat_vae": {
            "normal": "./datasets/baselines/SN_normal_flat_vae_synthetic.pt",
            "abnormal": "./datasets/baselines/SN_abnormal_flat_vae_synthetic.pt",
        },
        "hierarchical_vae": {
            "normal": "./datasets/anomaly/SN/SN_normal_synthetic.pt",
            "abnormal": "./datasets/anomaly/SN/SN_abnormal_synthetic.pt",
        },
    },
}


TRAIN_SCRIPT = {
    "TT": "./classifier/train_tt_anomaly.py",
    "SN": "./classifier/train_sn_anomaly.py",
}


CONSOLIDATED_FIELDNAMES = [
    "run_order",
    "system",
    "generator",
    "real_percent_requested",
    "synthetic_percent_requested",
    "real_percent_actual",
    "synth_percent_actual",
    "mode_version",
    "syn_normal_path",
    "syn_abnormal_path",
    "real_holdout_ratio",
    "real_val_ratio",
    "min_nodes_filter",
    "max_synth_per_class",
    "train_real_total",
    "train_synth_total",
    "train_normal_real",
    "train_abnormal_real",
    "train_normal_synth",
    "train_abnormal_synth",
    "val_normal_synth",
    "val_abnormal_synth",
    "val_normal_real",
    "val_abnormal_real",
    "test_normal_real",
    "test_abnormal_real",
    "train_total_after_filter",
    "val_total_after_filter",
    "test_total_after_filter",
    "test_normal_real_after_filter",
    "test_abnormal_real_after_filter",
    "test_loss",
    "test_accuracy",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "normal_precision",
    "normal_recall",
    "normal_f1",
    "abnormal_precision",
    "abnormal_recall",
    "abnormal_f1",
    "resource_summary_csv",
    "resource_timeseries_csv",
    "resource_return_code",
    "resource_wall_seconds",
    "resource_process_tree_cpu_seconds",
    "resource_process_tree_avg_cpu_pct",
    "resource_process_tree_peak_rss_mb",
    "resource_process_tree_peak_vms_mb",
    "resource_gpu_peak_mem_used_total_mb",
    "resource_process_tree_gpu_peak_mem_mb",
    "resource_gpu_peak_util_pct",
    "resource_samples",
    "seed",
    "pr_auc",
    "false_alarm_rate",
    "tn",
    "fp",
    "fn",
    "tp",
]


TABLE_GENERATORS = {
    "our_approach": "hierarchical_vae",
    "baseline_1": "random_sampling",
    "baseline_2": "flat_vae",
}


def _check_inputs(systems, generators):
    missing = []
    for system in systems:
        for generator in generators:
            paths = GENERATOR_PATHS[system][generator]
            for label, path in paths.items():
                if not (ROOT / path).exists():
                    missing.append(f"{system} {generator} {label}: {path}")
    if missing:
        joined = "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(f"Missing synthetic dataset files:\n{joined}")


def _build_command(system, generator, real_percent, results_csv, weights_dir, seed):
    paths = GENERATOR_PATHS[system][generator]
    if generator == "random_sampling":
        for label, path in paths.items():
            if "random_sampling" not in Path(path).name:
                raise ValueError(
                    f"{system} random_sampling {label} path is not a random-sampling file: {path}"
                )
    weights_path = (
        Path(weights_dir)
        / f"{system.lower()}_{generator}_real{int(real_percent)}_classifier_weights.pt"
    )
    return [
        sys.executable,
        TRAIN_SCRIPT[system],
        "--generator-name",
        generator,
        "--syn-normal-path",
        paths["normal"],
        "--syn-abnormal-path",
        paths["abnormal"],
        "--weights-path",
        str(weights_path),
        "--results-csv",
        results_csv,
        "--real",
        str(real_percent),
        "--seed",
        str(seed),
    ]


def _format_percent(value) -> str:
    value = float(value)
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:g}%"


def _format_metric(value) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.3f}"


def _read_single_result(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No result row was written to {path}")
    return rows[-1]


def _read_resource_summary(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No resource summary row was written to {path}")
    row = rows[-1]
    return {
        "resource_return_code": row.get("return_code", ""),
        "resource_wall_seconds": row.get("wall_seconds", ""),
        "resource_process_tree_cpu_seconds": row.get(
            "process_tree_cpu_seconds",
            "",
        ),
        "resource_process_tree_avg_cpu_pct": row.get(
            "process_tree_avg_cpu_pct",
            "",
        ),
        "resource_process_tree_peak_rss_mb": row.get(
            "process_tree_peak_rss_mb",
            "",
        ),
        "resource_process_tree_peak_vms_mb": row.get(
            "process_tree_peak_vms_mb",
            "",
        ),
        "resource_gpu_peak_mem_used_total_mb": row.get(
            "gpu_peak_mem_used_total_mb",
            "",
        ),
        "resource_process_tree_gpu_peak_mem_mb": row.get(
            "process_tree_gpu_peak_mem_mb",
            "",
        ),
        "resource_gpu_peak_util_pct": row.get("gpu_peak_util_pct", ""),
        "resource_samples": row.get("samples", ""),
    }


def _append_consolidated_result(path: Path, row: dict, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = append and path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CONSOLIDATED_FIELDNAMES,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _latest_rows_by_key(rows):
    latest = {}
    for row in rows:
        key = (
            row["system"],
            row["generator"],
            float(row["real_percent_requested"]),
        )
        latest[key] = row
    return latest


def _write_summary_tables(results_csv: Path, summary_md: Path, systems, real_values):
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    latest = _latest_rows_by_key(rows)
    try:
        source_path = results_csv.relative_to(ROOT)
    except ValueError:
        source_path = results_csv

    lines = [
        "# Ordered Baseline Comparison Tables",
        "",
        f"Source: `{source_path}`.",
        "",
        "Metrics are weighted precision, weighted recall, and weighted F1.",
        "Hierarchical VAE, random sampling, and flat VAE are compared below.",
        "",
    ]

    system_names = {
        "TT": "TrainTicket",
        "SN": "SocialNetwork",
    }
    header = (
        "| Real data | Synthetic data | Hierarchical VAE precision | Hierarchical VAE recall | "
        "Hierarchical VAE F1 score | Random sampling precision | Random sampling recall | "
        "Random sampling F1 score | Flat VAE precision | Flat VAE recall | "
        "Flat VAE F1 score |"
    )
    separator = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    for system in systems:
        lines.extend([f"## {system_names[system]}", "", header, separator])
        for real_percent in real_values:
            real = float(real_percent)
            synth = 100.0 - real
            cells = [_format_percent(real), _format_percent(synth)]
            for generator in TABLE_GENERATORS.values():
                row = latest.get((system, generator, real))
                if row is None:
                    cells.extend(["", "", ""])
                    continue
                cells.extend(
                    [
                        _format_metric(row.get("weighted_precision")),
                        _format_metric(row.get("weighted_recall")),
                        _format_metric(row.get("weighted_f1")),
                    ]
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if any(row.get("resource_wall_seconds") for row in rows):
        resource_fields = [
            ("resource_wall_seconds", "Wall sec"),
            ("resource_process_tree_avg_cpu_pct", "Avg CPU %"),
            ("resource_process_tree_peak_rss_mb", "Peak RAM MB"),
            ("resource_process_tree_gpu_peak_mem_mb", "Process GPU MB"),
            ("resource_gpu_peak_mem_used_total_mb", "Total GPU MB"),
            ("resource_gpu_peak_util_pct", "Peak GPU %"),
        ]
        resource_header = (
            "| System | Generator | Real data | Synthetic data | "
            + " | ".join(title for _, title in resource_fields)
            + " |"
        )
        resource_sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        lines.extend(["## Resource Usage", "", resource_header, resource_sep])
        for system in systems:
            for generator in TABLE_GENERATORS.values():
                for real_percent in real_values:
                    real = float(real_percent)
                    row = latest.get((system, generator, real))
                    if row is None or not row.get("resource_wall_seconds"):
                        continue
                    synth = 100.0 - real
                    cells = [
                        system,
                        generator,
                        _format_percent(real),
                        _format_percent(synth),
                    ]
                    cells.extend(_format_metric(row.get(key)) for key, _ in resource_fields)
                    lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(lines))


def _resource_paths(resource_dir: Path, run_order, system, generator, real_percent):
    stem = f"run_{run_order:03d}_{system}_{generator}_real{int(real_percent)}"
    return {
        "summary_csv": resource_dir / f"{stem}_resource_summary.csv",
        "summary_json": resource_dir / f"{stem}_resource_summary.json",
        "timeseries_csv": resource_dir / f"{stem}_resource_timeseries.csv",
    }


def _wrap_with_resource_monitor(command, sample_interval, paths):
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


def main():
    parser = argparse.ArgumentParser(
        description="Run ordered random sampling, flat VAE, and hierarchical VAE anomaly classifier comparisons."
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["TT", "SN"],
        default=["TT", "SN"],
        help="Which systems to run.",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=["random_sampling", "empirical", "flat_vae", "hierarchical_vae"],
        default=["random_sampling", "flat_vae", "hierarchical_vae"],
        help="Which synthetic generators to evaluate.",
    )
    parser.add_argument(
        "--real-values",
        nargs="+",
        type=float,
        default=[0, 10, 30, 50, 70, 100],
        help="Real-data percentages to run. Synthetic percentage is 100 - real.",
    )
    parser.add_argument(
        "--results-csv",
        default="./classifier/baseline_comparison_ordered_results.csv",
        help="Consolidated CSV file for all run metrics.",
    )
    parser.add_argument(
        "--summary-md",
        default="./classifier/baseline_comparison_ordered_tables.md",
        help="Markdown file for the final comparison tables.",
    )
    parser.add_argument(
        "--weights-dir",
        default="./classifier/baseline_weights_ordered",
        help="Directory for per-run classifier weights.",
    )
    parser.add_argument(
        "--resource-dir",
        default="./classifier/baseline_resource_by_run",
        help="Directory for per-run resource summaries and timelines.",
    )
    parser.add_argument(
        "--resource-sample-interval",
        type=float,
        default=2.0,
        help="Seconds between CPU/RAM/GPU resource samples.",
    )
    parser.add_argument(
        "--no-resource-tracking",
        action="store_true",
        help="Run experiments without per-run resource tracking.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the consolidated CSV instead of starting a new file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override classifier epochs for every run. Defaults to each trainer's CLI default.",
    )
    parser.add_argument(
        "--max-synth-per-class",
        type=int,
        default=None,
        help="Optional cap for synthetic normal and abnormal graphs before validation split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.weights_dir).mkdir(parents=True, exist_ok=True)
    resource_dir = ROOT / args.resource_dir
    if not args.no_resource_tracking:
        resource_dir.mkdir(parents=True, exist_ok=True)
    _check_inputs(args.systems, args.generators)

    results_csv = ROOT / args.results_csv
    summary_md = ROOT / args.summary_md
    if not args.dry_run and not args.append:
        results_csv.unlink(missing_ok=True)
        summary_md.unlink(missing_ok=True)

    commands = []
    with tempfile.TemporaryDirectory(prefix="baseline_runs_", dir=ROOT) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for generator in args.generators:
            for real_percent in args.real_values:
                for system in args.systems:
                    run_order = len(commands) + 1
                    run_csv = (
                        tmp_dir
                        / f"run_{run_order:03d}_{system}_{generator}_real{int(real_percent)}.csv"
                    )
                    command = _build_command(
                        system=system,
                        generator=generator,
                        real_percent=real_percent,
                        results_csv=str(run_csv),
                        weights_dir=args.weights_dir,
                        seed=args.seed,
                    )
                    if args.epochs is not None:
                        command.extend(["--epochs", str(args.epochs)])
                    if (
                        args.max_synth_per_class is not None
                        and args.max_synth_per_class >= 0
                    ):
                        command.extend(
                            ["--max-synth-per-class", str(args.max_synth_per_class)]
                        )
                    resource_paths = _resource_paths(
                        resource_dir,
                        run_order,
                        system,
                        generator,
                        real_percent,
                    )
                    run_command = command
                    if not args.no_resource_tracking:
                        run_command = _wrap_with_resource_monitor(
                            command,
                            args.resource_sample_interval,
                            resource_paths,
                        )
                    commands.append(
                        (
                            run_order,
                            system,
                            generator,
                            real_percent,
                            run_csv,
                            command,
                            run_command,
                            resource_paths,
                        )
                    )

        print(f"Prepared {len(commands)} runs.")
        for idx, (
            run_order,
            system,
            generator,
            real_percent,
            run_csv,
            command,
            run_command,
            resource_paths,
        ) in enumerate(commands, start=1):
            print(f"\n[{idx}/{len(commands)}] {' '.join(command)}")
            if not args.no_resource_tracking:
                print(
                    "Resource tracking: "
                    f"{resource_paths['summary_csv']} and "
                    f"{resource_paths['timeseries_csv']}"
                )
            if args.dry_run:
                continue

            subprocess.run(run_command, cwd=ROOT, check=True)
            row = _read_single_result(run_csv)
            row.update(
                {
                    "run_order": run_order,
                    "system": system,
                    "generator": generator,
                    "real_percent_requested": real_percent,
                    "synthetic_percent_requested": 100.0 - float(real_percent),
                }
            )
            if not args.no_resource_tracking:
                row.update(
                    {
                        "resource_summary_csv": str(resource_paths["summary_csv"]),
                        "resource_timeseries_csv": str(
                            resource_paths["timeseries_csv"]
                        ),
                        **_read_resource_summary(resource_paths["summary_csv"]),
                    }
                )
            _append_consolidated_result(results_csv, row, append=True)

    if args.dry_run:
        print("\nDry run only. No CSV or table was written.")
        return

    _write_summary_tables(
        results_csv=results_csv,
        summary_md=summary_md,
        systems=args.systems,
        real_values=args.real_values,
    )

    print(f"\nDone. Results written to {results_csv}")
    print(f"Final tables written to {summary_md}")


if __name__ == "__main__":
    main()
