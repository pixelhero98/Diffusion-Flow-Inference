from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_flow_inference.diagnostics.adaptive_noise_sampler_followup import _choose_valid_windows
from diffusion_flow_inference.schedules.diffusion_flow import (
    BASELINE_SCHEDULE_KEYS,
    FLOW_TIME_SCHEDULE_KEYS,
    TRANSFER_SCHEDULE_KEYS,
    build_schedule_grid,
    fixed_schedule_shape_statistics,
    run_fixed_schedule_variant,
    schedule_display_name,
    schedule_time_alignment,
)
from diffusion_flow_inference.evaluation.support import (
    ALL_SOLVER_ORDER,
    DEFAULT_FORECAST_DATASETS,
    DEFAULT_LOB_DATASETS,
    DEFAULT_SHARED_BACKBONE_ROOT,
    FORECAST_FAMILY,
    LOCKED_TEST_PHASE,
    LOB_FAMILY,
    SOLVER_RUNTIME_NAMES,
    UNIFORM_SCHEDULER_KEY,
    VALIDATION_PHASE,
    evaluate_forecast_schedule,
    load_forecast_checkpoint_splits,
    load_lob_checkpoint_splits,
    parse_csv,
    parse_forecast_datasets,
    parse_int_csv,
    parse_lob_datasets,
    resolved_eval_horizon,
    resolved_eval_windows,
    selection_metric_for_family,
    solver_eval_multiplier,
    solver_experiment_scope,
    solver_macro_steps,
    validate_execution_preflight,
)
from diffusion_flow_inference.schedules.paper_registry import METHOD_KEY
from diffusion_flow_inference.evaluation.paper_tables import augment_rows_with_relative_metrics
from diffusion_flow_inference.common.paths import (
    default_backbone_manifest_path,
    default_cryptos_data_path,
    default_es_mbp_10_data_path,
    default_sleep_edf_data_path,
    project_outputs_root,
    project_paper_dataset_root,
    project_root,
    resolve_project_path,
)
from diffusion_flow_inference.datasets.bundles import ensure_processed_dataset_bundle
from diffusion_flow_inference.backbones.training.train_val import save_json

RUNNER_SIGNATURE_VERSION = "diffusion_flow_time_reparameterization_v1"
DEFAULT_OUT_ROOT = project_outputs_root() / "diffusion_flow_time_reparameterization"
DEFAULT_DATASET_BUNDLE_EXTRACT_ROOT = project_outputs_root() / "dataset_bundles" / "extracted"
DEFAULT_TARGET_NFE_VALUES: Tuple[int, ...] = (10, 12, 16)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_SCHEDULES: Tuple[str, ...] = BASELINE_SCHEDULE_KEYS

ROW_RECORD_FIELDS: Tuple[str, ...] = (
    "benchmark_family",
    "split_phase",
    "seed",
    "dataset",
    "checkpoint_id",
    "checkpoint_path",
    "backbone_name",
    "train_steps",
    "train_budget_label",
    "target_nfe",
    "runtime_nfe",
    "solver_key",
    "solver_name",
    "scheduler_key",
    "scheduler_variant_key",
    "scheduler_variant_name",
    "schedule_name",
    "row_signature",
    "signal_trace_key",
    "signal_validation_spearman",
    "info_growth_scale",
    "reference_macro_factor",
    "paper_duplicate_count",
    "experiment_scope",
    "selection_metric",
    "selection_metric_value",
    "reference_macro_steps",
    "reference_time_alignment",
    "runtime_grid_q25",
    "runtime_grid_q50",
    "runtime_grid_q75",
    "crps",
    "mse",
    "mase",
    "score_main",
    "conditional_w1",
    "tstr_macro_f1",
    "relative_crps_gain_vs_uniform",
    "relative_mase_gain_vs_uniform",
    "relative_score_gain_vs_uniform",
    "realized_nfe",
    "latency_ms_per_sample",
    "num_eval_samples",
    "eval_examples",
    "eval_windows",
    "row_status",
)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cast):
        return None
    return cast


def _mean(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _std(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else None
    return float(np.std(arr, ddof=1))


def _safe_relative_gain(value: Any, baseline_value: Any) -> Optional[float]:
    v = _optional_float(value)
    b = _optional_float(baseline_value)
    if v is None or b is None or abs(float(b)) <= 1e-12:
        return None
    return float(100.0 * (1.0 - float(v) / float(b)))


def _parse_schedule_names(text: str) -> List[str]:
    names = [name.strip().lower() for name in parse_csv(text)]
    unknown = [name for name in names if name not in BASELINE_SCHEDULE_KEYS]
    if unknown:
        raise ValueError(f"Unknown active diffusion-flow schedules: {unknown}")
    return names


def _path_matches_project_default(path: Path, default_path: Path) -> bool:
    try:
        return path.resolve() == default_path.resolve()
    except OSError:
        return False


def _selected_lob_data_paths(cli_args: argparse.Namespace) -> Dict[str, Path]:
    selected = set(parse_lob_datasets(str(cli_args.lob_datasets)))
    paths: Dict[str, Path] = {}
    cryptos_path = str(getattr(cli_args, "cryptos_path", "") or "").strip()
    es_path = str(getattr(cli_args, "es_path", "") or "").strip()
    sleep_edf_path = str(getattr(cli_args, "sleep_edf_path", "") or "").strip()
    if "cryptos" in selected:
        paths["cryptos"] = resolve_project_path(cryptos_path) if cryptos_path else Path(default_cryptos_data_path())
    if "es_mbp_10" in selected:
        paths["es_mbp_10"] = resolve_project_path(es_path) if es_path else Path(default_es_mbp_10_data_path())
    if "sleep_edf" in selected:
        paths["sleep_edf"] = resolve_project_path(sleep_edf_path) if sleep_edf_path else Path(default_sleep_edf_data_path())
    return paths


def _apply_dataset_bundle(cli_args: argparse.Namespace) -> argparse.Namespace:
    bundle_zip = str(getattr(cli_args, "dataset_bundle_zip", "") or "").strip()
    mode = str(getattr(cli_args, "dataset_bundle_mode", "auto") or "auto").strip().lower()
    if mode == "none" or not bundle_zip:
        setattr(cli_args, "_dataset_bundle_resolution", {"mode": mode, "used": False, "zip_path": bundle_zip or None})
        return cli_args

    lob_paths = _selected_lob_data_paths(cli_args)
    original_dataset_root = resolve_project_path(str(cli_args.dataset_root))
    result = ensure_processed_dataset_bundle(
        bundle_zip=bundle_zip,
        extract_root=str(getattr(cli_args, "dataset_bundle_extract_root", DEFAULT_DATASET_BUNDLE_EXTRACT_ROOT)),
        mode=mode,
        dataset_root=original_dataset_root,
        data_paths=list(lob_paths.values()),
    )
    if result is None:
        setattr(cli_args, "_dataset_bundle_resolution", {"mode": mode, "used": False, "zip_path": str(resolve_project_path(bundle_zip))})
        return cli_args

    paper_root = Path(result["paper_dataset_root"])
    data_root = Path(result["data_root"])
    if mode == "extract" or not original_dataset_root.exists() or _path_matches_project_default(original_dataset_root, project_paper_dataset_root()):
        cli_args.dataset_root = str(paper_root)
    if mode == "extract" or ("cryptos" in lob_paths and not lob_paths["cryptos"].exists()):
        cli_args.cryptos_path = str(data_root / "cryptos_binance_spot_monthly_1s_l10.npz")
    if mode == "extract" or ("es_mbp_10" in lob_paths and not lob_paths["es_mbp_10"].exists()):
        cli_args.es_path = str(data_root / "es_mbp_10.npz")
    if mode == "extract" or ("sleep_edf" in lob_paths and not lob_paths["sleep_edf"].exists()):
        cli_args.sleep_edf_path = str(data_root / "sleep_edf_3ch_100hz_stage_conditioned.npz")
    setattr(
        cli_args,
        "_dataset_bundle_resolution",
        {
            "mode": mode,
            "used": True,
            "extracted": bool(result.get("extracted", False)),
            "zip_path": str(result["bundle_zip"]),
            "extract_root": str(result["extract_root"]),
            "data_root": str(result["data_root"]),
            "paper_dataset_root": str(result["paper_dataset_root"]),
            "file_count": int(result["manifest"]["file_count"]),
        },
    )
    return cli_args


def _realized_nfe_for_solver(solver_key: str, runtime_nfe: int) -> int:
    return int(runtime_nfe) * int(solver_eval_multiplier(str(solver_key)))


def _row_signature(*, dataset: str, split_phase: str, seed: int, target_nfe: int, solver_key: str, scheduler_key: str, checkpoint_id: str) -> str:
    return "|".join(
        [str(dataset), str(split_phase), str(seed), str(target_nfe), str(solver_key), str(scheduler_key), str(checkpoint_id)]
    )


def _row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("benchmark_family"),
        row.get("split_phase"),
        int(row.get("seed", -1)),
        row.get("dataset"),
        int(row.get("target_nfe", -1)),
        row.get("solver_key"),
        row.get("scheduler_key"),
        row.get("row_signature"),
    )


def _write_row_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_RECORD_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_RECORD_FIELDS})


def _load_rows(jsonl_path: Path) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[_row_key(row)] = row
    return rows


def _init_row_recorder(out_root: Path, cli_args: argparse.Namespace) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / str(getattr(cli_args, "row_jsonl_name", "rows.jsonl"))
    csv_path = out_root / str(getattr(cli_args, "row_csv_name", "rows.csv"))
    rows_by_key = _load_rows(jsonl_path) if bool(getattr(cli_args, "resume", True)) else {}
    fh = jsonl_path.open("a", encoding="utf-8")
    save_json({"runner_signature": RUNNER_SIGNATURE_VERSION, "method_key": METHOD_KEY, "args": vars(cli_args)}, str(out_root / "run_config.json"))
    if rows_by_key:
        _write_row_csv(csv_path, list(rows_by_key.values()))
    return {"out_root": out_root, "jsonl_path": jsonl_path, "csv_path": csv_path, "fh": fh, "rows_by_key": rows_by_key}


def _append_row_record(row_recorder: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    row_dict = dict(row)
    key = _row_key(row_dict)
    row_recorder["rows_by_key"][key] = row_dict
    row_recorder["fh"].write(json.dumps(row_dict, sort_keys=True) + "\n")
    row_recorder["fh"].flush()
    _write_row_csv(Path(row_recorder["csv_path"]), list(row_recorder["rows_by_key"].values()))


def _existing_complete_row(row_recorder: Mapping[str, Any], row_key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    row = row_recorder["rows_by_key"].get(row_key)
    if row is not None and str(row.get("row_status")) == "complete":
        return dict(row)
    return None


def _pending_scheduler_cases(row_recorder: Mapping[str, Any], *, benchmark_family: str, split_phase: str, seed: int, dataset: str, checkpoint_id: str, target_nfe: int, solver_key: str, scheduler_cases: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    existing: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for case in scheduler_cases:
        scheduler_key = str(case["scheduler_key"])
        signature = _row_signature(dataset=dataset, split_phase=split_phase, seed=seed, target_nfe=target_nfe, solver_key=solver_key, scheduler_key=scheduler_key, checkpoint_id=checkpoint_id)
        key = (benchmark_family, split_phase, int(seed), dataset, int(target_nfe), solver_key, scheduler_key, signature)
        row = _existing_complete_row(row_recorder, key)
        if row is None:
            pending.append(dict(case, row_signature=signature))
        else:
            existing.append(row)
    return existing, pending


def _fixed_schedule_details(scheduler_key: str, runtime_nfe: int) -> Dict[str, Any]:
    fixed_grid = build_schedule_grid(str(scheduler_key), int(runtime_nfe))
    if fixed_grid is None:
        raise ValueError(f"Unable to build fixed grid for scheduler={scheduler_key}")
    details: Dict[str, Any] = {
        "time_grid": [float(x) for x in fixed_grid],
        "reference_time_alignment": schedule_time_alignment(str(scheduler_key)),
        "paper_duplicate_count": 0,
        "reference_macro_steps": int(runtime_nfe),
    }
    details.update(fixed_schedule_shape_statistics(fixed_grid))
    return details


def _build_row(*, benchmark_family: str, split_phase: str, seed: int, dataset: str, checkpoint: Mapping[str, Any], target_nfe: int, runtime_nfe: int, solver_key: str, scheduler_key: str, details: Mapping[str, Any], metrics: Mapping[str, Any], row_signature: str) -> Dict[str, Any]:
    selection_metric = selection_metric_for_family(str(benchmark_family))
    realized_nfe = metrics.get("realized_nfe")
    if realized_nfe is None:
        realized_nfe = _realized_nfe_for_solver(str(solver_key), int(runtime_nfe))
    return {
        "benchmark_family": str(benchmark_family),
        "split_phase": str(split_phase),
        "seed": int(seed),
        "dataset": str(dataset),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "backbone_name": str(checkpoint.get("backbone_name", "otflow")),
        "train_steps": int(checkpoint["train_steps"]),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "solver_key": str(solver_key),
        "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
        "scheduler_key": str(scheduler_key),
        "scheduler_variant_key": str(scheduler_key),
        "scheduler_variant_name": schedule_display_name(str(scheduler_key)),
        "schedule_name": schedule_display_name(str(scheduler_key)),
        "row_signature": str(row_signature),
        "signal_trace_key": None,
        "signal_validation_spearman": None,
        "info_growth_scale": None,
        "reference_macro_factor": None,
        "paper_duplicate_count": int(details.get("paper_duplicate_count", 0) or 0),
        "experiment_scope": solver_experiment_scope(str(solver_key)),
        "selection_metric": str(selection_metric),
        "selection_metric_value": metrics.get(selection_metric),
        "reference_macro_steps": int(details.get("reference_macro_steps", runtime_nfe)),
        "reference_time_alignment": str(details.get("reference_time_alignment", schedule_time_alignment(str(scheduler_key)))),
        "runtime_grid_q25": details.get("runtime_grid_q25"),
        "runtime_grid_q50": details.get("runtime_grid_q50"),
        "runtime_grid_q75": details.get("runtime_grid_q75"),
        "crps": metrics.get("crps"),
        "mse": metrics.get("mse"),
        "mase": metrics.get("mase"),
        "score_main": metrics.get("score_main"),
        "conditional_w1": metrics.get("conditional_w1"),
        "tstr_macro_f1": metrics.get("tstr_macro_f1"),
        "relative_crps_gain_vs_uniform": metrics.get("relative_crps_gain_vs_uniform"),
        "relative_mase_gain_vs_uniform": metrics.get("relative_mase_gain_vs_uniform"),
        "relative_score_gain_vs_uniform": metrics.get("relative_score_gain_vs_uniform"),
        "realized_nfe": int(realized_nfe),
        "latency_ms_per_sample": metrics.get("latency_ms_per_sample", metrics.get("efficiency_ms_per_sample")),
        "num_eval_samples": metrics.get("num_eval_samples"),
        "eval_examples": metrics.get("eval_examples"),
        "eval_windows": metrics.get("eval_windows"),
        "row_status": "complete",
    }


def _scheduler_cases_for_datasets(cli_args: argparse.Namespace, datasets: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    schedule_names = _parse_schedule_names(str(cli_args.baseline_scheduler_names))
    return {str(dataset): [{"scheduler_key": key} for key in schedule_names] for dataset in datasets}


def _run_forecast_phase(cli_args: argparse.Namespace, *, row_recorder: Mapping[str, Any], split_phase: str, seeds: Sequence[int], scheduler_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    dataset_root = Path(str(cli_args.dataset_root)).resolve()
    shared_backbone_root = Path(str(cli_args.shared_backbone_root)).resolve()
    device = torch.device(str(cli_args.device))
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    for dataset_idx, dataset in enumerate(datasets):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = load_forecast_checkpoint_splits(cli_args=cli_args, dataset_root=dataset_root, shared_backbone_root=shared_backbone_root, dataset=dataset, device=device)
        checkpoint = dataset_cache[dataset]
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        eval_ds = splits["val"] if str(split_phase) == VALIDATION_PHASE else splits["test"]
        for seed in seeds:
            for target_idx, target_nfe in enumerate(parse_int_csv(str(cli_args.target_nfe_values))):
                for solver_idx, solver_key in enumerate(parse_csv(str(cli_args.solver_names))):
                    runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                    scheduler_cases = list(scheduler_cases_by_dataset[str(dataset)])
                    existing_rows, pending_cases = _pending_scheduler_cases(row_recorder, benchmark_family=FORECAST_FAMILY, split_phase=str(split_phase), seed=int(seed), dataset=str(dataset), checkpoint_id=str(checkpoint["checkpoint_id"]), target_nfe=int(target_nfe), solver_key=str(solver_key), scheduler_cases=scheduler_cases)
                    rows.extend(existing_rows)
                    cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                    for existing_row in existing_rows:
                        if str(existing_row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY:
                            cell_uniform_metrics = existing_row
                    for case in pending_cases:
                        scheduler_key = str(case["scheduler_key"])
                        details = _fixed_schedule_details(scheduler_key, int(runtime_nfe))
                        eval_seed = int(seed) + 100_000 * dataset_idx + 1_000 * target_idx + solver_idx
                        metrics = evaluate_forecast_schedule(model, eval_ds, cfg, solver_name=str(SOLVER_RUNTIME_NAMES[str(solver_key)]), runtime_nfe=int(runtime_nfe), time_grid=details["time_grid"], num_eval_samples=int(cli_args.num_eval_samples), seed=int(eval_seed))
                        if scheduler_key != UNIFORM_SCHEDULER_KEY and cell_uniform_metrics is not None:
                            metrics = dict(metrics)
                            metrics["relative_crps_gain_vs_uniform"] = _safe_relative_gain(metrics.get("crps"), cell_uniform_metrics.get("crps"))
                            metrics["relative_mase_gain_vs_uniform"] = _safe_relative_gain(metrics.get("mase"), cell_uniform_metrics.get("mase"))
                        row = _build_row(benchmark_family=FORECAST_FAMILY, split_phase=str(split_phase), seed=int(seed), dataset=str(dataset), checkpoint=checkpoint, target_nfe=int(target_nfe), runtime_nfe=int(runtime_nfe), solver_key=str(solver_key), scheduler_key=scheduler_key, details=details, metrics=metrics, row_signature=str(case["row_signature"]))
                        _append_row_record(row_recorder, row)
                        rows.append(row)
                        if scheduler_key == UNIFORM_SCHEDULER_KEY:
                            cell_uniform_metrics = row
    return rows


def _run_lob_phase(cli_args: argparse.Namespace, *, row_recorder: Mapping[str, Any], split_phase: str, seeds: Sequence[int], scheduler_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    shared_backbone_root = Path(str(cli_args.shared_backbone_root)).resolve()
    device = torch.device(str(cli_args.device))
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    datasets = parse_lob_datasets(str(cli_args.lob_datasets))
    for dataset_idx, dataset in enumerate(datasets):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = load_lob_checkpoint_splits(cli_args=cli_args, shared_backbone_root=shared_backbone_root, dataset=dataset, device=device)
        checkpoint = dataset_cache[dataset]
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        eval_ds = splits["val"] if str(split_phase) == VALIDATION_PHASE else splits["test"]
        eval_horizon = resolved_eval_horizon(cli_args, str(dataset))
        eval_windows = resolved_eval_windows(cli_args, str(dataset), "val" if str(split_phase) == VALIDATION_PHASE else "test")
        for seed in seeds:
            chosen_eval_t0s = np.asarray(_choose_valid_windows(eval_ds, horizon=int(eval_horizon), n_windows=int(eval_windows), seed=int(seed) + 1_000 * dataset_idx), dtype=np.int64)
            for target_idx, target_nfe in enumerate(parse_int_csv(str(cli_args.target_nfe_values))):
                for solver_idx, solver_key in enumerate(parse_csv(str(cli_args.solver_names))):
                    runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                    existing_rows, pending_cases = _pending_scheduler_cases(row_recorder, benchmark_family=LOB_FAMILY, split_phase=str(split_phase), seed=int(seed), dataset=str(dataset), checkpoint_id=str(checkpoint["checkpoint_id"]), target_nfe=int(target_nfe), solver_key=str(solver_key), scheduler_cases=list(scheduler_cases_by_dataset[str(dataset)]))
                    rows.extend(existing_rows)
                    cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                    for existing_row in existing_rows:
                        if str(existing_row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY:
                            cell_uniform_metrics = existing_row
                    for case in pending_cases:
                        scheduler_key = str(case["scheduler_key"])
                        details = _fixed_schedule_details(scheduler_key, int(runtime_nfe))
                        grid_spec = {"grid_name": scheduler_key, "grid_kind": "fixed_diffusion_flow_time_grid", "selection_group": scheduler_key, "comparison_role": "transferred" if scheduler_key in TRANSFER_SCHEDULE_KEYS else "baseline", "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]), "nfe": int(runtime_nfe), "time_grid": details["time_grid"]}
                        metrics_seed = int(seed) + 1_000_000 * dataset_idx + 10_000 * target_idx + solver_idx
                        result_row = run_fixed_schedule_variant(model=model, ds=eval_ds, cfg=cfg, eval_horizon=int(eval_horizon), eval_windows=int(len(chosen_eval_t0s)), grid_spec=grid_spec, chosen_t0s=chosen_eval_t0s, generation_seed_base=int(metrics_seed), metrics_seed=int(metrics_seed), score_main_only=False)
                        metrics = {"score_main": result_row.get("score_main"), "conditional_w1": result_row.get("conditional_w1"), "tstr_macro_f1": result_row.get("tstr_macro_f1"), "efficiency_ms_per_sample": result_row.get("efficiency_ms_per_sample"), "eval_windows": int(len(chosen_eval_t0s)), "realized_nfe": _realized_nfe_for_solver(str(solver_key), int(runtime_nfe))}
                        if scheduler_key != UNIFORM_SCHEDULER_KEY and cell_uniform_metrics is not None:
                            metrics["relative_score_gain_vs_uniform"] = _safe_relative_gain(metrics.get("score_main"), cell_uniform_metrics.get("score_main"))
                        row = _build_row(benchmark_family=LOB_FAMILY, split_phase=str(split_phase), seed=int(seed), dataset=str(dataset), checkpoint=checkpoint, target_nfe=int(target_nfe), runtime_nfe=int(runtime_nfe), solver_key=str(solver_key), scheduler_key=scheduler_key, details=details, metrics=metrics, row_signature=str(case["row_signature"]))
                        _append_row_record(row_recorder, row)
                        rows.append(row)
                        if scheduler_key == UNIFORM_SCHEDULER_KEY:
                            cell_uniform_metrics = row
    return rows


def _candidate_rows_by_phase(rows: Sequence[Mapping[str, Any]], split_phase: str, solver_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    solver_filter = None if solver_names is None else {str(x) for x in solver_names}
    out = []
    for row in rows:
        if str(row.get("split_phase")) != str(split_phase):
            continue
        if str(row.get("row_status")) != "complete":
            continue
        if solver_filter is not None and str(row.get("solver_key")) not in solver_filter:
            continue
        out.append(dict(row))
    return out


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row.get("benchmark_family"), row.get("dataset"), row.get("target_nfe"), row.get("solver_key"), row.get("scheduler_key"), row.get("train_budget_label"))
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    metric_names = ("crps", "mse", "mase", "score_main", "conditional_w1", "tstr_macro_f1", "relative_crps_gain_vs_uniform", "relative_mase_gain_vs_uniform", "relative_score_gain_vs_uniform", "realized_nfe", "latency_ms_per_sample")
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        family, dataset, target_nfe, solver_key, scheduler_key, budget = key
        summary: Dict[str, Any] = {"benchmark_family": family, "dataset": dataset, "target_nfe": int(target_nfe), "solver_key": solver_key, "scheduler_key": scheduler_key, "schedule_name": schedule_display_name(str(scheduler_key)), "train_budget_label": budget, "n_seeds": int(len(group)), "seed_values": sorted(int(row.get("seed", 0)) for row in group)}
        for metric in metric_names:
            vals = [_optional_float(row.get(metric)) for row in group]
            vals = [float(v) for v in vals if v is not None]
            summary[f"{metric}_mean"] = _mean(vals)
            summary[f"{metric}_std"] = _std(vals)
        summaries.append(summary)
    return summaries


def _aggregate_main_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    seed_summaries = _aggregate_seed_rows(rows)
    augmented = augment_rows_with_relative_metrics(seed_summaries)
    return {"method_key": METHOD_KEY, "row_count": int(len(rows)), "summary_row_count": int(len(augmented)), "schedule_keys": sorted({str(row.get("scheduler_key")) for row in rows}), "flow_time_schedule_keys": list(FLOW_TIME_SCHEDULE_KEYS), "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS), "seed_summaries": augmented}


def _prep_summary(cli_args: argparse.Namespace) -> Dict[str, Any]:
    schedules = _parse_schedule_names(str(cli_args.baseline_scheduler_names))
    solvers = parse_csv(str(cli_args.solver_names))
    nfes = parse_int_csv(str(cli_args.target_nfe_values))
    manifest_path = resolve_project_path(str(cli_args.backbone_manifest)) if str(cli_args.backbone_manifest).strip() else None
    manifest_summary: Dict[str, Any] = {"path": None, "ready_count": None, "missing_count": None}
    if manifest_path is not None:
        resolved = manifest_path
        manifest_summary["path"] = str(resolved)
        if resolved.exists():
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            manifest_summary["ready_count"] = int(payload.get("ready_count", 0))
            manifest_summary["missing_count"] = int(payload.get("missing_count", 0))
    return {"runner_mode": "diffusion_flow_time_reparameterization", "runner_signature": RUNNER_SIGNATURE_VERSION, "method_key": METHOD_KEY, "baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS), "flow_time_schedule_keys": list(FLOW_TIME_SCHEDULE_KEYS), "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS), "scheduled_evaluation_keys": schedules, "solver_names": solvers, "target_nfe_values": nfes, "forecast_datasets": parse_forecast_datasets(str(cli_args.forecast_datasets)), "lob_datasets": parse_lob_datasets(str(cli_args.lob_datasets)), "backbone_manifest": manifest_summary, "dataset_bundle": getattr(cli_args, "_dataset_bundle_resolution", None), "allow_execute": bool(getattr(cli_args, "allow_execute", False))}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run diffusion-flow time reparameterization fixed-schedule evaluations.")
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--dataset_root", type=str, default=str(project_paper_dataset_root()))
    ap.add_argument("--dataset_bundle_zip", type=str, default="")
    ap.add_argument("--dataset_bundle_extract_root", type=str, default=str(DEFAULT_DATASET_BUNDLE_EXTRACT_ROOT))
    ap.add_argument("--dataset_bundle_mode", type=str, choices=("auto", "extract", "none"), default="auto")
    ap.add_argument("--shared_backbone_root", type=str, default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    ap.add_argument("--backbone_manifest", type=str, default=str(default_backbone_manifest_path()))
    ap.add_argument("--otflow_train_steps", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--forecast_datasets", type=str, default=",".join(DEFAULT_FORECAST_DATASETS))
    ap.add_argument("--lob_datasets", type=str, default=",".join(DEFAULT_LOB_DATASETS))
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--sleep_edf_path", type=str, default="")
    ap.add_argument("--solver_names", type=str, default=",".join(ALL_SOLVER_ORDER))
    ap.add_argument("--target_nfe_values", type=str, default=",".join(str(x) for x in DEFAULT_TARGET_NFE_VALUES))
    ap.add_argument("--baseline_scheduler_names", type=str, default=",".join(DEFAULT_SCHEDULES))
    ap.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--num_eval_samples", type=int, default=5)
    ap.add_argument("--calibration_trace_samples", type=int, default=1)
    ap.add_argument("--eval_horizon", type=int, default=0)
    ap.add_argument("--eval_windows_val", type=int, default=0)
    ap.add_argument("--eval_windows_test", type=int, default=0)
    ap.add_argument("--sigma_eps", type=float, default=1e-6)
    ap.add_argument("--hidden_dim", type=int, default=160)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--rollout_mode", type=str, default="non_ar")
    ap.add_argument("--future_block_len", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--row_jsonl_name", type=str, default="rows.jsonl")
    ap.add_argument("--row_csv_name", type=str, default="rows.csv")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--diagnose_locked_forecast_only", action="store_true", default=False)
    ap.add_argument("--allow_execute", action="store_true", default=False)
    return ap


def run_diffusion_flow_time_reparameterization(cli_args: argparse.Namespace) -> Dict[str, Any]:
    cli_args = _apply_dataset_bundle(cli_args)
    out_root = Path(str(cli_args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    prep_payload = _prep_summary(cli_args)
    if bool(getattr(cli_args, "diagnose_locked_forecast_only", False)):
        rows = list(_load_rows(out_root / str(getattr(cli_args, "row_jsonl_name", "rows.jsonl"))).values())
        locked = _candidate_rows_by_phase(rows, LOCKED_TEST_PHASE)
        payload = {"runner_mode": "diagnose_locked_forecast_only", "row_count": int(len(rows)), "locked_row_count": int(len(locked)), "main_table_summary": _aggregate_main_table(locked)}
        save_json(dict(payload), str(out_root / "combined_summary.json"))
        return payload
    if not bool(cli_args.allow_execute):
        save_json(dict(prep_payload), str(out_root / "combined_summary.json"))
        return dict(prep_payload)

    validate_execution_preflight(cli_args)
    row_recorder = _init_row_recorder(out_root, cli_args)
    locked_seeds = parse_int_csv(str(cli_args.seeds))
    forecast_datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    lob_datasets = parse_lob_datasets(str(cli_args.lob_datasets))
    scheduler_cases = _scheduler_cases_for_datasets(cli_args, list(forecast_datasets) + list(lob_datasets))
    try:
        _run_forecast_phase(cli_args, row_recorder=row_recorder, split_phase=LOCKED_TEST_PHASE, seeds=locked_seeds, scheduler_cases_by_dataset={dataset: scheduler_cases[dataset] for dataset in forecast_datasets})
        _run_lob_phase(cli_args, row_recorder=row_recorder, split_phase=LOCKED_TEST_PHASE, seeds=locked_seeds, scheduler_cases_by_dataset={dataset: scheduler_cases[dataset] for dataset in lob_datasets})
    finally:
        row_recorder["fh"].close()

    locked_rows = _candidate_rows_by_phase(list(row_recorder["rows_by_key"].values()), LOCKED_TEST_PHASE)
    main_table_summary = _aggregate_main_table(locked_rows)
    seed_summaries = main_table_summary.pop("seed_summaries")
    save_json({"seed_summaries": seed_summaries}, str(out_root / "locked_test_seed_summary.json"))
    save_json(dict(main_table_summary), str(out_root / "main_table_summary.json"))
    schedule_selection = {"method_key": METHOD_KEY, "baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS), "flow_time_schedule_keys": list(FLOW_TIME_SCHEDULE_KEYS), "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS), "scheduled_evaluation_keys": _parse_schedule_names(str(cli_args.baseline_scheduler_names))}
    save_json(dict(schedule_selection), str(out_root / "schedule_selection_summary.json"))
    combined = {"prep": dict(prep_payload), "schedule_selection_summary": dict(schedule_selection), "locked_test_seed_summary": {"seed_summaries": seed_summaries}, "main_table_summary": dict(main_table_summary)}
    save_json(dict(combined), str(out_root / "combined_summary.json"))
    return combined


def main() -> None:
    run_diffusion_flow_time_reparameterization(build_argparser().parse_args())


if __name__ == "__main__":
    main()
