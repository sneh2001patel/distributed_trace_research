import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


GENERATOR_PATHS = {
    "TT": {
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


def _build_command(system, generator, real_percent, results_csv, weights_dir):
    paths = GENERATOR_PATHS[system][generator]
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
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run empirical, flat VAE, and hierarchical VAE anomaly classifier comparisons."
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
        choices=["empirical", "flat_vae", "hierarchical_vae"],
        default=["empirical", "flat_vae", "hierarchical_vae"],
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
        default="./classifier/baseline_comparison_results.csv",
        help="CSV file to append all run metrics to.",
    )
    parser.add_argument(
        "--weights-dir",
        default="./classifier/baseline_weights",
        help="Directory for per-run classifier weights.",
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
        help="Override classifier epochs for every run. Defaults: TT=120, SN=50.",
    )
    parser.add_argument(
        "--max-synth-per-class",
        type=int,
        default=500,
        help="Cap eligible synthetic normal and abnormal graphs before validation split. Use -1 to disable.",
    )
    args = parser.parse_args()

    Path(args.weights_dir).mkdir(parents=True, exist_ok=True)
    _check_inputs(args.systems, args.generators)

    commands = []
    for system in args.systems:
        for generator in args.generators:
            for real_percent in args.real_values:
                command = _build_command(
                    system=system,
                    generator=generator,
                    real_percent=real_percent,
                    results_csv=args.results_csv,
                    weights_dir=args.weights_dir,
                )
                if args.epochs is not None:
                    command.extend(["--epochs", str(args.epochs)])
                if args.max_synth_per_class is not None and args.max_synth_per_class >= 0:
                    command.extend(
                        ["--max-synth-per-class", str(args.max_synth_per_class)]
                    )
                commands.append(command)

    print(f"Prepared {len(commands)} runs.")
    for idx, command in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    print(f"\nDone. Results appended to {args.results_csv}")


if __name__ == "__main__":
    main()
