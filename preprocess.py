import json
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
PARSED_DIR = ROOT / "parsed_output"
OUTPUT_DIR = ROOT / "preprocessed"


def _extract_status_code(tags: Iterable[dict]) -> int:
    for tag in tags or []:
        if tag.get("key") == "http.status_code":
            return int(tag.get("value", 0))
    return 0


def _extract_parent_id(references: Iterable[dict]) -> str:
    for ref in references or []:
        if ref.get("refType") == "CHILD_OF":
            return str(ref.get("spanID", ""))
    return ""


def _load_fault_windows(path: Path) -> List[dict]:
    meta = json.loads(path.read_text())
    return meta.get("faults", [])


def _label_roots_by_fault_window(
    roots: pd.DataFrame, fault_windows: List[dict]
) -> pd.DataFrame:
    roots = roots.copy()
    roots["start_sec"] = roots["timestamp"].astype(float)
    roots["end_sec"] = roots["start_sec"] + roots["duration"].astype(float) / 1_000_000.0

    def classify(row: pd.Series) -> Tuple[str, str]:
        for fault in fault_windows:
            start = float(fault["start"])
            end = float(fault["end"])
            # Label a trace abnormal if its root span overlaps an injected fault window.
            if not (row["end_sec"] < start or row["start_sec"] > end):
                return "abnormal", str(fault["fault"])
        return "normal", "none"

    labels = roots.apply(classify, axis=1, result_type="expand")
    roots["trace_label"] = labels[0]
    roots["fault_type"] = labels[1]
    return roots[["trace_id", "trace_label", "fault_type"]]


def _normalize_fault_run(trace_csv: Path, fault_json: Path, system: str) -> pd.DataFrame:
    df = pd.read_csv(trace_csv)
    roots = df[df["span_id"].astype(str) == df["trace_id"].astype(str)]
    trace_labels = _label_roots_by_fault_window(roots, _load_fault_windows(fault_json))

    merged = df.merge(trace_labels, on="trace_id", how="left")
    merged["system"] = system
    merged["source_run"] = trace_csv.stem.replace("_trace", "")
    merged["source_kind"] = "fault_run"
    return merged


def _normalize_no_fault_archive(archive_path: Path, system: str) -> pd.DataFrame:
    with tarfile.open(archive_path, "r:*") as tf:
        span_member = next(name for name in tf.getnames() if name.endswith("spans.json"))
        traces = json.load(tf.extractfile(span_member))

    rows: List[dict] = []
    for trace in traces:
        process_map = trace.get("processes", {})
        span_process: Dict[str, str] = {}
        span_rows: List[dict] = []

        for span in trace.get("spans", []):
            service_name = process_map.get(span.get("processID", ""), {}).get(
                "serviceName", "unknown_service"
            )
            span_process[str(span.get("spanID"))] = service_name

        for span in trace.get("spans", []):
            span_id = str(span.get("spanID"))
            parent_id = _extract_parent_id(span.get("references", []))
            parent_service = span_process.get(parent_id, "no_parent") if parent_id else "no_parent"
            row = {
                "timestamp": int(span.get("startTime", 0) // 1_000_000),
                "service_name": span_process.get(span_id, "unknown_service"),
                "span_id": span_id,
                "parent_id": parent_id,
                "trace_id": str(trace.get("traceID")),
                "duration": float(span.get("duration", 0)),
                "status_code": _extract_status_code(span.get("tags", [])),
                "operation_name": span.get("operationName", "unknown_operation"),
                "parent_service_name": parent_service,
                "trace_label": "normal",
                "fault_type": "none",
                "system": system,
                "source_run": archive_path.stem.replace(".tar", ""),
                "source_kind": "no_fault_archive",
            }
            span_rows.append(row)
        rows.extend(span_rows)

    return pd.DataFrame(rows)


def _list_no_fault_run_names(archives: List[Path]) -> set[str]:
    run_names: set[str] = set()
    for archive in archives:
        with tarfile.open(archive, "r:*") as tf:
            span_members = [name for name in tf.getnames() if name.endswith("spans.json")]
            for member in span_members:
                run_names.add(Path(member).parent.name)
    return run_names


def _build_system_dataframe(
    system: str,
    parsed_trace_dir: Path,
    parsed_fault_dir: Path,
    no_fault_archives: List[Path],
) -> pd.DataFrame:
    no_fault_run_names = _list_no_fault_run_names(no_fault_archives)

    dfs: List[pd.DataFrame] = []
    for trace_csv in sorted(parsed_trace_dir.glob("*_trace.csv")):
        run_name = trace_csv.stem.replace("_trace", "")
        if run_name in no_fault_run_names:
            # Prefer rebuilding no-fault runs from the raw spans.json archive.
            continue

        fault_json = parsed_fault_dir / trace_csv.name.replace("_trace.csv", ".json").replace(
            f"{system}.", f"{system}.fault-"
        )
        if not fault_json.exists():
            raise FileNotFoundError(f"Missing fault metadata for {trace_csv.name}: {fault_json}")
        dfs.append(_normalize_fault_run(trace_csv, fault_json, system))

    for archive in sorted(no_fault_archives):
        dfs.append(_normalize_no_fault_archive(archive, system))

    if not dfs:
        raise RuntimeError(f"No data found for {system}")
    return pd.concat(dfs, ignore_index=True)


def _write_outputs(df: pd.DataFrame, system: str) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_path = OUTPUT_DIR / f"{system}_labeled_spans.csv"
    normal_path = OUTPUT_DIR / f"{system}_normal_spans.csv"
    abnormal_path = OUTPUT_DIR / f"{system}_abnormal_spans.csv"
    trace_path = OUTPUT_DIR / f"{system}_trace_labels.csv"

    df.to_csv(all_path, index=False)
    df[df["trace_label"] == "normal"].to_csv(normal_path, index=False)
    df[df["trace_label"] == "abnormal"].to_csv(abnormal_path, index=False)

    trace_labels = (
        df[df["span_id"].astype(str) == df["trace_id"].astype(str)][
            ["trace_id", "timestamp", "duration", "trace_label", "fault_type", "source_run", "source_kind"]
        ]
        .drop_duplicates("trace_id")
        .sort_values(["source_run", "timestamp", "trace_id"])
    )
    trace_labels.to_csv(trace_path, index=False)

    summary = {
        "system": system,
        "span_rows": int(len(df)),
        "trace_count": int(trace_labels["trace_id"].nunique()),
        "normal_trace_count": int((trace_labels["trace_label"] == "normal").sum()),
        "abnormal_trace_count": int((trace_labels["trace_label"] == "abnormal").sum()),
        "fault_type_counts": trace_labels["fault_type"].value_counts().to_dict(),
        "source_kind_counts": trace_labels["source_kind"].value_counts().to_dict(),
        "files": {
            "all_spans": str(all_path.relative_to(ROOT)),
            "normal_spans": str(normal_path.relative_to(ROOT)),
            "abnormal_spans": str(abnormal_path.relative_to(ROOT)),
            "trace_labels": str(trace_path.relative_to(ROOT)),
        },
    }
    return summary


def main() -> None:
    sn_df = _build_system_dataframe(
        system="SN",
        parsed_trace_dir=PARSED_DIR / "SN_Dataset" / "trace",
        parsed_fault_dir=PARSED_DIR / "SN_Dataset" / "faults",
        no_fault_archives=sorted((ROOT / "SN Dataset" / "no fault").glob("*.tar.xz")),
    )
    tt_df = _build_system_dataframe(
        system="TT",
        parsed_trace_dir=PARSED_DIR / "TT_Dataset" / "trace",
        parsed_fault_dir=PARSED_DIR / "TT_Dataset" / "faults",
        no_fault_archives=sorted((ROOT / "TT Dataset" / "no fault").glob("*.tar.xz"))
        + sorted((ROOT / "TT Dataset" / "no fault").glob("*.txz")),
    )

    summaries = {
        "SN": _write_outputs(sn_df, "SN"),
        "TT": _write_outputs(tt_df, "TT"),
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))

    for system, summary in summaries.items():
        print(f"\n{system}")
        print(
            f"  traces: {summary['trace_count']} | normal: {summary['normal_trace_count']} | "
            f"abnormal: {summary['abnormal_trace_count']}"
        )
        print(f"  fault types: {summary['fault_type_counts']}")
        print(f"  wrote: {summary['files']}")

    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
