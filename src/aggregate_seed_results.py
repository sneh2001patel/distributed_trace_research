import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    with open(args.in_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(list)
    for row in rows:
        key = (row["system"], row["generator"], float(row["real_percent_requested"]))
        grouped[key].append(row)

    metrics = [
        "weighted_f1",
        "abnormal_f1",
        "abnormal_recall",
        "pr_auc",
        "false_alarm_rate",
    ]
    out_rows = []
    for (system, generator, real_percent), group in sorted(grouped.items()):
        out = {
            "system": system,
            "generator": generator,
            "real_percent": real_percent,
            "runs": len(group),
        }
        for metric in metrics:
            vals = [float(row[metric]) for row in group if row.get(metric) not in ("", None)]
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_sd"] = stdev(vals)
        out_rows.append(out)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    for row in out_rows:
        print(row)


if __name__ == "__main__":
    main()
