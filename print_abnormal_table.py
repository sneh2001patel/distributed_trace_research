"""
Read a baseline comparison CSV and print/save the abnormal P/R/F1 table
for both TT and SN, all three generators, at real% = 0, 10, 30, 50, 70, 100.

Usage:
    python print_abnormal_table.py --csv classifier/baseline_comparison_final.csv
    python print_abnormal_table.py --csv classifier/baseline_comparison_final.csv --out results/abnormal_table.txt
"""

import argparse
import csv
from pathlib import Path


REAL_PERCENTS = [0.0, 10.0, 30.0, 50.0, 70.0, 100.0]
GENERATORS = ["hierarchical_vae", "flat_vae", "random_sampling"]
SYSTEMS = ["TT", "SN"]

GEN_LABEL = {
    "hierarchical_vae": "Hier VAE",
    "flat_vae":         "Flat VAE",
    "random_sampling":  "Random",
}


def load_rows(csv_path: str) -> dict:
    rows = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            system = r.get("system", "")
            if not system:
                path = r.get("syn_normal_path", "")
                system = "SN" if "SN" in path else "TT"
            rp = r.get("real_percent_requested", "")
            if not rp:
                continue
            key = (system, r["generator"], float(rp))
            rows[key] = r
    return rows


def build_table(rows: dict) -> str:
    lines = []
    for system in SYSTEMS:
        lines.append("=== %s ===" % system)
        lines.append("%6s  %12s  %8s  %8s  %8s" % ("Real%", "Method", "Abn P", "Abn R", "Abn F1"))
        lines.append("-" * 50)
        for rp in REAL_PERCENTS:
            block = []
            for gen in GENERATORS:
                r = rows.get((system, gen, rp))
                if not r:
                    continue
                try:
                    ap  = float(r["abnormal_precision"])
                    ar  = float(r["abnormal_recall"])
                    af1 = float(r["abnormal_f1"])
                except (KeyError, TypeError, ValueError):
                    continue
                block.append(
                    "%6.0f  %12s  %8.3f  %8.3f  %8.3f"
                    % (rp, GEN_LABEL[gen], ap, ar, af1)
                )
            if block:
                lines.extend(block)
                lines.append("")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="./classifier/baseline_comparison_final.csv",
        help="Path to the baseline comparison CSV.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save the table as a text file.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    table = build_table(rows)

    print(table)

    out_path = args.out or Path(args.csv).parent / (Path(args.csv).stem + "_abnormal_table.txt")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(table)
    print("Saved to %s" % out_path)


if __name__ == "__main__":
    main()
