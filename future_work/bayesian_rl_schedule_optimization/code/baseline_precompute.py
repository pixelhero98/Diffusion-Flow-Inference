from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from forecast_bo_runner import (
    _IndexSubset,
    _build_reference_schedule,
    _core_imports,
    _load_checkpoint,
    _run_final_comparison,
    deterministic_validation_partition,
    indices_hash,
    load_json,
    parse_comparison_schedules,
    parse_csv,
    parse_int_csv,
    resolve_workspace_path,
    schedule_hash,
    selected_indices_from_pool,
    write_csv,
    write_json,
)


DEFAULT_BASELINE_SCHEDULES: Tuple[str, ...] = (
    "uniform",
    "late_power_3",
    "ays",
    "gits",
    "ots",
    "ser_ptg_reference",
)
GENERATED_OPTIMIZER_SCHEDULES = {"bo_best", "joint_progression_ppo_best"}


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def discover_source_cache_roots(workspace_root: str | Path, explicit_roots: Sequence[str | Path] = ()) -> List[Path]:
    root = Path(workspace_root)
    candidates: List[Path] = []
    for item in explicit_roots:
        path = Path(item)
        candidates.append(path if path.is_absolute() else root / path)
    return _unique_paths(candidates)


def summarize_global_final_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["dataset"]), int(row["target_nfe"]), str(row["solver_key"]), str(row["schedule_key"])),
            [],
        ).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, target_nfe, solver_key, schedule_key), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "dataset": dataset,
            "target_nfe": int(target_nfe),
            "solver_key": solver_key,
            "schedule_key": schedule_key,
            "n_seeds": int(len(group)),
            "seed_values": sorted(int(row["seed"]) for row in group),
            "schedule_grid": list(group[0].get("schedule_grid", [])),
        }
        for metric in (
            "crps",
            "mase",
            "relative_crps_ratio",
            "relative_mase_ratio",
            "avg_relative_ratio",
            "kl_to_reference",
        ):
            values = np.asarray([float(row[metric]) for row in group if row.get(metric) is not None], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean()) if values.size else None
            item[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else (0.0 if values.size == 1 else None)
        out.append(item)
    return out


def _final_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in load_json(path).get("rows", [])]


def _row_key(row: Mapping[str, Any]) -> Tuple[str, int, str, str, int]:
    return (
        str(row.get("dataset")),
        int(row.get("target_nfe", -1)),
        str(row.get("solver_key")),
        str(row.get("schedule_key")),
        int(row.get("seed", -1)),
    )


def _expected_keys(
    *,
    datasets: Sequence[str],
    target_nfes: Sequence[int],
    solvers: Sequence[str],
    schedules: Sequence[str],
    seeds: Sequence[int],
) -> List[Tuple[str, int, str, str, int]]:
    return [
        (str(dataset), int(target_nfe), str(solver), str(schedule), int(seed))
        for dataset in datasets
        for target_nfe in target_nfes
        for solver in solvers
        for schedule in schedules
        for seed in seeds
    ]


def build_baseline_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str],
    target_nfes: Sequence[int],
    solvers: Sequence[str],
    schedules: Sequence[str],
    seeds: Sequence[int],
    out_root: str | Path,
) -> Dict[str, Any]:
    expected = _expected_keys(
        datasets=datasets,
        target_nfes=target_nfes,
        solvers=solvers,
        schedules=schedules,
        seeds=seeds,
    )
    by_key = {_row_key(row): dict(row) for row in rows}
    missing = [
        {
            "dataset": dataset,
            "target_nfe": int(target_nfe),
            "solver_key": solver,
            "schedule_key": schedule,
            "seed": int(seed),
        }
        for dataset, target_nfe, solver, schedule, seed in expected
        if (dataset, target_nfe, solver, schedule, seed) not in by_key
    ]
    references: List[Dict[str, Any]] = []
    for dataset in datasets:
        for target_nfe in target_nfes:
            for solver in solvers:
                path = Path(out_root) / str(dataset) / f"nfe_{int(target_nfe)}" / str(solver) / "reference_schedule.json"
                item: Dict[str, Any] = {
                    "dataset": str(dataset),
                    "target_nfe": int(target_nfe),
                    "solver_key": str(solver),
                    "path": str(path),
                    "exists": bool(path.exists()),
                }
                if path.exists():
                    reference = load_json(path)
                    item["schedule_hash"] = str(reference.get("schedule_hash") or schedule_hash(reference["schedule_grid"]))
                    item["reference_macro_factor"] = reference.get("reference_macro_factor")
                    item["calibration_indices_hash"] = reference.get("calibration_indices_hash") or indices_hash(
                        [int(idx) for idx in reference.get("calibration_indices", [])]
                    )
                references.append(item)
    return {
        "artifact": "forecast_baseline_cache_manifest_v1",
        "expected_rows": int(len(expected)),
        "present_rows": int(len(expected) - len(missing)),
        "missing_rows": int(len(missing)),
        "complete": not missing and all(item["exists"] for item in references),
        "datasets": list(datasets),
        "target_nfes": [int(value) for value in target_nfes],
        "solvers": list(solvers),
        "schedules": list(schedules),
        "seeds": [int(value) for value in seeds],
        "missing": missing,
        "references": references,
    }


def _namespace_for_cell(
    args: argparse.Namespace,
    *,
    dataset: str,
    target_nfe: int,
    source_cache_roots: Sequence[Path],
) -> argparse.Namespace:
    cell_args = argparse.Namespace(**vars(args))
    cell_args.dataset = str(dataset)
    cell_args.target_nfe = int(target_nfe)
    cell_args.comparison_schedules = ",".join(parse_comparison_schedules(str(args.schedules)))
    cell_args.baseline_cache_roots = ",".join(str(path) for path in source_cache_roots)
    return cell_args


def _write_validation_split(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    solver_idx: int,
    solver_out: Path,
) -> Tuple[List[int], List[int]]:
    calibration_pool, selector_pool = deterministic_validation_partition(
        len(checkpoint["splits"]["val"]),
        calibration_fraction=float(args.calibration_fraction),
        seed=int(args.bo_seed) + 1_000 * int(solver_idx),
    )
    calibration_indices = selected_indices_from_pool(
        calibration_pool,
        int(args.calibration_windows),
        seed=int(args.bo_seed) + 2_000 * int(solver_idx) + 11,
    )
    payload = {
        "artifact": "forecast_baseline_cache_validation_split_v1",
        "calibration_fraction": float(args.calibration_fraction),
        "calibration_pool_size": int(len(calibration_pool)),
        "selector_pool_size": int(len(selector_pool)),
        "calibration_indices": calibration_indices,
        "selector_pool_indices": selector_pool,
        "calibration_indices_hash": indices_hash(calibration_indices),
    }
    write_json(payload, solver_out / "validation_split.json")
    return calibration_indices, selector_pool


def _solver_results_for_cell(
    *,
    args: argparse.Namespace,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    dataset: str,
    target_nfe: int,
    solvers: Sequence[str],
    out_root: Path,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for solver_idx, solver_key in enumerate(solvers):
        solver_out = out_root / str(dataset) / f"nfe_{int(target_nfe)}" / str(solver_key)
        solver_out.mkdir(parents=True, exist_ok=True)
        calibration_indices, _ = _write_validation_split(
            args=args,
            checkpoint=checkpoint,
            solver_idx=int(solver_idx),
            solver_out=solver_out,
        )
        calibration_subset = _IndexSubset(checkpoint["splits"]["val"], calibration_indices)
        reference = _build_reference_schedule(
            args=args,
            solver_key=str(solver_key),
            solver_idx=int(solver_idx),
            checkpoint=checkpoint,
            calibration_subset=calibration_subset,
            solver_out=solver_out,
            resume=bool(args.resume),
        )
        best_payload = {
            "artifact": "forecast_baseline_cache_reference_best_placeholder_v1",
            "best_observation": {
                "candidate_id": "ser_ptg_reference",
                "source": "ser_ptg_reference",
                "schedule_grid": [float(x) for x in reference["schedule_grid"]],
                "kl_to_reference": 0.0,
            },
        }
        results[str(solver_key)] = {"reference": reference, "best": best_payload, "solver_out": solver_out}
    return results


def _validate_schedules_for_precompute(schedules: Sequence[str]) -> None:
    disallowed = [key for key in schedules if key in GENERATED_OPTIMIZER_SCHEDULES]
    if disallowed:
        raise ValueError(f"precompute-forecast-baselines cannot precompute optimizer schedules: {disallowed}.")
    if "uniform" not in schedules:
        raise ValueError("precompute schedules must include uniform so relative metrics can be computed.")
    if "ser_ptg_reference" not in schedules:
        raise ValueError("precompute schedules must include ser_ptg_reference for the reference cache.")


def run_precompute_forecast_baselines(args: argparse.Namespace) -> Dict[str, Any]:
    core = _core_imports()
    import torch

    workspace_root = resolve_workspace_path(str(args.workspace_root), Path.cwd())
    out_root = resolve_workspace_path(str(args.out_root), workspace_root)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv(str(args.datasets))
    target_nfes = parse_int_csv(str(args.target_nfes))
    solvers = parse_csv(str(args.solvers))
    schedules = parse_comparison_schedules(str(args.schedules))
    _validate_schedules_for_precompute(schedules)
    seeds = parse_int_csv(str(args.final_test_seeds))
    explicit_sources = parse_csv(str(args.source_cache_roots))
    source_cache_roots = discover_source_cache_roots(workspace_root, explicit_sources)

    run_config = {
        "artifact": "forecast_baseline_cache_run_config_v1",
        "datasets": datasets,
        "target_nfes": target_nfes,
        "solvers": solvers,
        "schedules": schedules,
        "otflow_train_steps": int(args.otflow_train_steps),
        "calibration_fraction": float(args.calibration_fraction),
        "calibration_windows": int(args.calibration_windows),
        "reference_macro_factor": float(args.reference_macro_factor),
        "calibration_trace_samples": int(args.calibration_trace_samples),
        "density_floor_eta": float(args.density_floor_eta),
        "num_eval_samples": int(args.num_eval_samples),
        "final_test_seeds": seeds,
        "final_test_windows": int(args.final_test_windows),
    }
    write_json(run_config, out_root / "run_config.json")

    device = torch.device(str(args.device))
    rows: List[Dict[str, Any]] = _final_rows(out_root / "final_comparison_rows.json") if bool(args.resume) else []
    for dataset in datasets:
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = str(dataset)
        checkpoint = _load_checkpoint(dataset_args, workspace_root, device)
        for target_nfe in target_nfes:
            cell_args = _namespace_for_cell(
                args,
                dataset=str(dataset),
                target_nfe=int(target_nfe),
                source_cache_roots=source_cache_roots,
            )
            solver_results = _solver_results_for_cell(
                args=cell_args,
                core=core,
                checkpoint=checkpoint,
                dataset=str(dataset),
                target_nfe=int(target_nfe),
                solvers=solvers,
                out_root=out_root,
            )
            rows = _run_final_comparison(
                args=cell_args,
                core=core,
                checkpoint=checkpoint,
                solver_results=solver_results,
                out_root=out_root,
                resume=bool(args.resume),
            )
            summary = summarize_global_final_rows(rows)
            write_json({"artifact": "forecast_baseline_cache_final_summary_v1", "summaries": summary}, out_root / "final_summary.json")
            write_csv(summary, out_root / "final_summary.csv")
            manifest = build_baseline_manifest(
                rows,
                datasets=datasets,
                target_nfes=target_nfes,
                solvers=solvers,
                schedules=schedules,
                seeds=seeds,
                out_root=out_root,
            )
            write_json(manifest, out_root / "baseline_cache_manifest.json")

    rows = _final_rows(out_root / "final_comparison_rows.json")
    summary = summarize_global_final_rows(rows)
    manifest = build_baseline_manifest(
        rows,
        datasets=datasets,
        target_nfes=target_nfes,
        solvers=solvers,
        schedules=schedules,
        seeds=seeds,
        out_root=out_root,
    )
    write_json({"artifact": "forecast_baseline_cache_final_summary_v1", "summaries": summary}, out_root / "final_summary.json")
    write_csv(summary, out_root / "final_summary.csv")
    write_json(manifest, out_root / "baseline_cache_manifest.json")
    return {
        "artifact": "forecast_baseline_cache_result_v1",
        "run_config": run_config,
        "row_count": int(len(rows)),
        "manifest": manifest,
    }


def add_precompute_forecast_baselines_parser(subparsers: Any) -> None:
    run = subparsers.add_parser("precompute-forecast-baselines", help="Precompute reusable forecast schedule baselines.")
    run.add_argument("--workspace-root", type=Path, default=Path.cwd())
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument("--datasets", type=str, required=True)
    run.add_argument("--target-nfes", type=str, required=True)
    run.add_argument("--solvers", type=str, default="euler,heun,midpoint_rk2,dpmpp2m")
    run.add_argument("--schedules", type=str, default=",".join(DEFAULT_BASELINE_SCHEDULES))
    run.add_argument("--otflow-train-steps", type=int, default=20000)
    run.add_argument("--calibration-fraction", type=float, default=0.7)
    run.add_argument("--calibration-windows", type=int, default=64)
    run.add_argument("--reference-macro-factor", type=float, default=16.0)
    run.add_argument("--calibration-trace-samples", type=int, default=1)
    run.add_argument("--density-floor-eta", type=float, default=0.05)
    run.add_argument("--num-eval-samples", type=int, default=5)
    run.add_argument("--final-test-seeds", type=str, default="0,1,2")
    run.add_argument("--final-test-windows", type=int, default=0)
    run.add_argument("--source-cache-roots", type=str, default="")
    run.add_argument("--bo-seed", type=int, default=0)
    run.add_argument("--device", type=str, default="cuda")
    run.add_argument("--resume", action="store_true", default=True)
    run.add_argument("--no-resume", dest="resume", action="store_false")
