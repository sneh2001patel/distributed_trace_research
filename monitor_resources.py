import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _read_proc_stat(pid: int):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None

    end = text.rfind(")")
    if end == -1:
        return None
    fields = text[end + 2 :].split()
    try:
        return {
            "utime": int(fields[11]),
            "stime": int(fields[12]),
            "vsize_bytes": int(fields[20]),
            "rss_pages": int(fields[21]),
        }
    except (IndexError, ValueError):
        return None


def _parent_pid(pid: int):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _process_tree(root_pid: int):
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        ppid = _parent_pid(pid)
        if ppid is not None:
            parents[pid] = ppid

    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if pid not in tree and ppid in tree:
                tree.add(pid)
                changed = True
    return tree


def _sample_process_tree(root_pid: int, clock_ticks: int, page_size: int):
    pids = _process_tree(root_pid)
    total_ticks = 0
    rss_bytes = 0
    vms_bytes = 0
    live_pids = []

    for pid in pids:
        stat = _read_proc_stat(pid)
        if stat is None:
            continue
        live_pids.append(pid)
        total_ticks += stat["utime"] + stat["stime"]
        rss_bytes += stat["rss_pages"] * page_size
        vms_bytes += stat["vsize_bytes"]

    return {
        "pids": live_pids,
        "pid_count": len(live_pids),
        "cpu_seconds": total_ticks / clock_ticks,
        "rss_mb": rss_bytes / (1024 * 1024),
        "vms_mb": vms_bytes / (1024 * 1024),
    }


def _sample_system_cpu(prev):
    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    except OSError:
        return None, prev

    values = [int(v) for v in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    current = (idle, total)
    if prev is None:
        return None, current

    idle_delta = idle - prev[0]
    total_delta = total - prev[1]
    if total_delta <= 0:
        return None, current
    return 100.0 * (1.0 - idle_delta / total_delta), current


def _sample_gpu(process_pids):
    def parse_float(value):
        try:
            return float(value)
        except ValueError:
            return None

    def empty_gpu_sample():
        return {
            "gpu_count": 0,
            "gpu_util_max_pct": "",
            "gpu_util_avg_pct": "",
            "gpu_mem_used_max_mb": "",
            "gpu_mem_used_total_mb": "",
            "gpu_mem_total_mb": "",
            "gpu_power_draw_total_w": "",
            "process_tree_gpu_mem_mb": "",
            "process_tree_gpu_process_count": "",
        }

    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return empty_gpu_sample()

    util = []
    mem_used = []
    mem_total = []
    power = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpu_util = parse_float(parts[1])
        gpu_mem_used = parse_float(parts[2])
        gpu_mem_total = parse_float(parts[3])
        gpu_power = parse_float(parts[4])
        if gpu_util is None or gpu_mem_used is None or gpu_mem_total is None:
            continue
        util.append(gpu_util)
        mem_used.append(gpu_mem_used)
        mem_total.append(gpu_mem_total)
        if gpu_power is not None:
            power.append(gpu_power)

    if not util:
        return empty_gpu_sample()

    process_tree_gpu_mem_mb = ""
    process_tree_gpu_process_count = ""
    app_command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        apps = subprocess.run(
            app_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.SubprocessError:
        apps = None
    if apps is not None:
        process_pid_set = set(process_pids)
        process_tree_gpu_mem_mb = 0.0
        process_tree_gpu_process_count = 0
        for line in apps.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                used_memory = float(parts[1])
            except ValueError:
                continue
            if pid in process_pid_set:
                process_tree_gpu_mem_mb += used_memory
                process_tree_gpu_process_count += 1

    return {
        "gpu_count": len(util),
        "gpu_util_max_pct": max(util),
        "gpu_util_avg_pct": sum(util) / len(util),
        "gpu_mem_used_max_mb": max(mem_used),
        "gpu_mem_used_total_mb": sum(mem_used),
        "gpu_mem_total_mb": sum(mem_total),
        "gpu_power_draw_total_w": sum(power) if power else "",
        "process_tree_gpu_mem_mb": process_tree_gpu_mem_mb,
        "process_tree_gpu_process_count": process_tree_gpu_process_count,
    }


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value


def run(command, sample_interval, timeseries_csv, summary_csv, summary_json):
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    page_size = os.sysconf("SC_PAGE_SIZE")
    start_wall = time.time()

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    rows = []
    prev_tree_cpu = None
    prev_tree_time = None
    prev_system_cpu = None
    peak_rss_mb = 0.0
    peak_vms_mb = 0.0
    peak_gpu_mem_mb = 0.0
    peak_process_gpu_mem_mb = 0.0
    peak_gpu_util_pct = 0.0
    total_tree_cpu_seconds = 0.0
    return_code = None

    try:
        while True:
            now = time.time()
            elapsed = now - start_wall
            tree = _sample_process_tree(process.pid, clock_ticks, page_size)
            gpu = _sample_gpu(tree["pids"])
            system_cpu_pct, prev_system_cpu = _sample_system_cpu(prev_system_cpu)

            tree_cpu_pct = None
            if prev_tree_cpu is not None and prev_tree_time is not None:
                dt = now - prev_tree_time
                dcpu = tree["cpu_seconds"] - prev_tree_cpu
                if dt > 0 and dcpu >= 0:
                    tree_cpu_pct = 100.0 * dcpu / dt

            prev_tree_cpu = tree["cpu_seconds"]
            prev_tree_time = now
            total_tree_cpu_seconds = max(total_tree_cpu_seconds, tree["cpu_seconds"])
            peak_rss_mb = max(peak_rss_mb, tree["rss_mb"])
            peak_vms_mb = max(peak_vms_mb, tree["vms_mb"])

            if gpu["gpu_mem_used_total_mb"] != "":
                peak_gpu_mem_mb = max(peak_gpu_mem_mb, gpu["gpu_mem_used_total_mb"])
            if gpu["process_tree_gpu_mem_mb"] != "":
                peak_process_gpu_mem_mb = max(
                    peak_process_gpu_mem_mb,
                    gpu["process_tree_gpu_mem_mb"],
                )
            if gpu["gpu_util_max_pct"] != "":
                peak_gpu_util_pct = max(peak_gpu_util_pct, gpu["gpu_util_max_pct"])

            row = {
                "elapsed_seconds": elapsed,
                "pid_count": tree["pid_count"],
                "process_tree_cpu_pct": "" if tree_cpu_pct is None else tree_cpu_pct,
                "system_cpu_pct": "" if system_cpu_pct is None else system_cpu_pct,
                "process_tree_cpu_seconds": tree["cpu_seconds"],
                "process_tree_rss_mb": tree["rss_mb"],
                "process_tree_vms_mb": tree["vms_mb"],
                **gpu,
            }
            rows.append({key: _fmt(value) for key, value in row.items()})

            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(sample_interval)
    except KeyboardInterrupt:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        else:
            process.send_signal(signal.SIGINT)
        return_code = process.wait()
        raise
    finally:
        if return_code is None:
            return_code = process.poll()
        end_wall = time.time()
        wall_seconds = end_wall - start_wall
        avg_cpu_pct = (
            100.0 * total_tree_cpu_seconds / wall_seconds if wall_seconds > 0 else 0.0
        )
        summary = {
            "command": " ".join(command),
            "return_code": return_code,
            "wall_seconds": round(wall_seconds, 6),
            "process_tree_cpu_seconds": round(total_tree_cpu_seconds, 6),
            "process_tree_avg_cpu_pct": round(avg_cpu_pct, 6),
            "process_tree_peak_rss_mb": round(peak_rss_mb, 6),
            "process_tree_peak_vms_mb": round(peak_vms_mb, 6),
            "gpu_peak_mem_used_total_mb": round(peak_gpu_mem_mb, 6),
            "process_tree_gpu_peak_mem_mb": round(peak_process_gpu_mem_mb, 6),
            "gpu_peak_util_pct": round(peak_gpu_util_pct, 6),
            "sample_interval_seconds": sample_interval,
            "samples": len(rows),
            "timeseries_csv": str(timeseries_csv),
        }

        fieldnames = [
            "elapsed_seconds",
            "pid_count",
            "process_tree_cpu_pct",
            "system_cpu_pct",
            "process_tree_cpu_seconds",
            "process_tree_rss_mb",
            "process_tree_vms_mb",
            "gpu_count",
            "gpu_util_max_pct",
            "gpu_util_avg_pct",
            "gpu_mem_used_max_mb",
            "gpu_mem_used_total_mb",
            "gpu_mem_total_mb",
            "gpu_power_draw_total_w",
            "process_tree_gpu_mem_mb",
            "process_tree_gpu_process_count",
        ]
        _write_csv(timeseries_csv, rows, fieldnames)
        _write_csv(summary_csv, [summary], list(summary.keys()))
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n")

        print(f"\nResource summary written to {summary_csv}")
        print(f"Resource timeline written to {timeseries_csv}")
        print(f"Resource summary JSON written to {summary_json}")
        print(
            "Summary: "
            f"wall={wall_seconds:.1f}s, "
            f"avg_cpu={avg_cpu_pct:.1f}%, "
            f"peak_ram={peak_rss_mb:.1f} MB, "
            f"peak_process_gpu_mem={peak_process_gpu_mem_mb:.1f} MB, "
            f"peak_total_gpu_mem={peak_gpu_mem_mb:.1f} MB, "
            f"peak_gpu_util={peak_gpu_util_pct:.1f}%"
        )

    return return_code


def main():
    parser = argparse.ArgumentParser(
        description="Run a command and record process-tree CPU, RAM, and GPU usage."
    )
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument(
        "--timeseries-csv",
        default="./classifier/baseline_resource_usage_timeseries.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="./classifier/baseline_resource_usage_summary.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="./classifier/baseline_resource_usage_summary.json",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Pass the command to monitor after --")

    return_code = run(
        command=command,
        sample_interval=args.sample_interval,
        timeseries_csv=ROOT / args.timeseries_csv,
        summary_csv=ROOT / args.summary_csv,
        summary_json=ROOT / args.summary_json,
    )
    sys.exit(return_code if return_code is not None else 1)


if __name__ == "__main__":
    main()
