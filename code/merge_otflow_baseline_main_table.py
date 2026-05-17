from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from diffusion_flow_time_reparameterization import LOCKED_TEST_PHASE, ROW_RECORD_FIELDS, _aggregate_main_table, _candidate_rows_by_phase
from otflow_train_val import save_json


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_row_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_RECORD_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_RECORD_FIELDS})


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge per-dataset baseline-only OTFlow main-table outputs.")
    parser.add_argument("--in_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="")
    return parser


def merge_baseline_outputs(args: argparse.Namespace) -> Dict[str, Any]:
    in_root = Path(str(args.in_root)).resolve()
    out_root = Path(str(args.out_root)).resolve() if str(args.out_root).strip() else in_root / "merged"
    out_root.mkdir(parents=True, exist_ok=True)

    rows_by_key: Dict[str, Dict[str, Any]] = {}
    source_files = sorted(path for path in in_root.glob("*/rows.jsonl") if path.is_file())
    for source_file in source_files:
        for row in _load_jsonl(source_file):
            key = json.dumps(
                [
                    row.get("benchmark_family"),
                    row.get("split_phase"),
                    row.get("seed"),
                    row.get("dataset"),
                    row.get("target_nfe"),
                    row.get("solver_key"),
                    row.get("scheduler_key"),
                    row.get("row_signature"),
                ],
                sort_keys=True,
            )
            rows_by_key[key] = row

    rows = list(rows_by_key.values())
    locked_rows = _candidate_rows_by_phase(rows, LOCKED_TEST_PHASE)
    main_table_summary = _aggregate_main_table(locked_rows)
    seed_summaries = main_table_summary.pop("seed_summaries")

    merged_jsonl = out_root / "rows.jsonl"
    with merged_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    _write_row_csv(out_root / "rows.csv", rows)
    save_json({"seed_summaries": seed_summaries}, str(out_root / "locked_test_seed_summary.json"))
    save_json(dict(main_table_summary), str(out_root / "main_table_summary.json"))

    complete_locked = [row for row in locked_rows if str(row.get("row_status", "")) == "complete"]
    combined = {
        "source_root": str(in_root),
        "source_files": [str(path) for path in source_files],
        "row_count": int(len(rows)),
        "locked_test_row_count": int(len(locked_rows)),
        "complete_locked_test_row_count": int(len(complete_locked)),
        "locked_test_seed_summary": {"seed_summaries": seed_summaries},
        "main_table_summary": dict(main_table_summary),
    }
    save_json(dict(combined), str(out_root / "combined_summary.json"))
    return combined


def main() -> None:
    merge_baseline_outputs(build_argparser().parse_args())


if __name__ == "__main__":
    main()
