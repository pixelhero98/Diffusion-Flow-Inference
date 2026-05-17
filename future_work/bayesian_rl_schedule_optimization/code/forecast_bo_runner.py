from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from botorch_bo import suggest_bo_batch
from reference_schedule import build_reference_from_cell, local_defect_from_oracle
from residual_parameterization import (
    DEFAULT_BASIS_KIND,
    FORECAST_AVG_RELATIVE_OBJECTIVE,
    build_residual_basis,
    generate_initial_perturbations,
    normalize_observations,
    observation_objective,
    theta_to_schedule_record,
    validate_n_initial,
)


DEFAULT_COMPARISON_SCHEDULE_KEYS: Tuple[str, ...] = ("uniform", "gits", "ser_ptg_reference", "bo_best")
COMPARISON_SCHEDULE_KEYS: Tuple[str, ...] = DEFAULT_COMPARISON_SCHEDULE_KEYS
DETERMINISTIC_COMPARISON_SCHEDULE_KEYS: Tuple[str, ...] = (
    "uniform",
    "late_power_3",
    "flowts_power_sampling",
    "ays",
    "gits",
    "ots",
)
GENERATED_COMPARISON_SCHEDULE_KEYS: Tuple[str, ...] = ("ser_ptg_reference", "bo_best")
REFERENCE_CANDIDATE_ID = "reference_center"
TEST_BASELINE_REUSE_KEYS: Tuple[str, ...] = (
    "uniform",
    "late_power_3",
    "ays",
    "gits",
    "ots",
    "ser_ptg_reference",
)


class _IndexSubset:
    def __init__(self, base: Any, indices: Sequence[int]):
        self.base = base
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Any:
        return self.base[self.indices[int(item)]]

    def mase_denom(self, example_idx: int) -> float:
        return float(self.base.mase_denom(self.indices[int(example_idx)]))

    def denormalize_block(self, values: Any, example_idx: int) -> Any:
        return self.base.denormalize_block(values, self.indices[int(example_idx)])


def parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in parse_csv(text)]


def parse_comparison_schedules(text: str) -> List[str]:
    schedules = parse_csv(text)
    if not schedules:
        schedules = list(DEFAULT_COMPARISON_SCHEDULE_KEYS)
    duplicate = sorted({key for key in schedules if schedules.count(key) > 1})
    if duplicate:
        raise ValueError(f"comparison_schedules contains duplicates: {duplicate}.")
    allowed = set(DETERMINISTIC_COMPARISON_SCHEDULE_KEYS) | set(GENERATED_COMPARISON_SCHEDULE_KEYS)
    unknown = [key for key in schedules if key not in allowed]
    if unknown:
        raise ValueError(f"Unknown comparison schedule keys: {unknown}.")
    if "uniform" not in schedules:
        raise ValueError("comparison_schedules must include uniform so relative metrics can be computed.")
    return schedules


def resolve_workspace_path(path: str | Path, workspace_root: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(workspace_root).resolve() / candidate).resolve()


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_indices(length: int, limit: int, *, seed: int) -> List[int]:
    n = int(length)
    if n <= 0:
        raise ValueError("Cannot select indices from an empty dataset split.")
    cap = int(limit)
    if cap <= 0 or cap >= n:
        return list(range(n))
    rng = np.random.default_rng(int(seed))
    return sorted(int(idx) for idx in rng.choice(n, size=cap, replace=False).tolist())


def selected_indices_from_pool(pool: Sequence[int], limit: int, *, seed: int) -> List[int]:
    values = [int(idx) for idx in pool]
    if not values:
        raise ValueError("Cannot select indices from an empty index pool.")
    cap = int(limit)
    if cap <= 0 or cap >= len(values):
        return sorted(values)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.asarray(values, dtype=np.int64), size=cap, replace=False)
    return sorted(int(idx) for idx in selected.tolist())


def deterministic_validation_partition(
    length: int,
    *,
    calibration_fraction: float = 0.7,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    n = int(length)
    if n < 2:
        raise ValueError("Validation partition requires at least two examples.")
    frac = float(calibration_fraction)
    if frac <= 0.0 or frac >= 1.0:
        raise ValueError(f"calibration_fraction must be in (0, 1), got {calibration_fraction}.")
    rng = np.random.default_rng(int(seed))
    permuted = rng.permutation(n).astype(np.int64).tolist()
    n_calibration = int(round(frac * n))
    n_calibration = min(max(1, n_calibration), n - 1)
    calibration = sorted(int(idx) for idx in permuted[:n_calibration])
    bo_validation = sorted(int(idx) for idx in permuted[n_calibration:])
    if set(calibration).intersection(bo_validation):
        raise ValueError("Calibration and BO-validation partitions must be disjoint.")
    return calibration, bo_validation


def schedule_hash(schedule_grid: Sequence[float]) -> str:
    rounded = [round(float(x), 12) for x in schedule_grid]
    payload = json.dumps(rounded, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def indices_hash(indices: Sequence[int]) -> str:
    payload = json.dumps([int(idx) for idx in indices], separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def candidate_budget_breakdown(bo_budget: int, *, n_initial: int = 16) -> Dict[str, int]:
    budget = int(bo_budget)
    if budget <= 0:
        raise ValueError(f"bo_budget must be positive, got {bo_budget}.")
    initial = min(max(0, budget - 1), validate_n_initial(n_initial))
    bo = max(0, budget - 1 - initial)
    return {"reference": 1, "initial": int(initial), "bo": int(bo), "total": int(1 + initial + bo)}


def observed_candidate_ids(observations_payload: Mapping[str, Any]) -> set[str]:
    return {str(row.get("candidate_id")) for row in observations_payload.get("observations", []) if row.get("candidate_id")}


def observed_schedule_hashes(observations_payload: Mapping[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for row in observations_payload.get("observations", []):
        row_hash = row.get("schedule_hash")
        if row_hash is None and row.get("schedule_grid") is not None:
            row_hash = schedule_hash(row["schedule_grid"])
        if row_hash is not None:
            hashes.add(str(row_hash))
    return hashes


def pending_candidate_records(
    candidates: Sequence[Mapping[str, Any]],
    observations_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    seen_ids = observed_candidate_ids(observations_payload)
    seen_hashes = observed_schedule_hashes(observations_payload)
    pending: List[Dict[str, Any]] = []
    for row in candidates:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id in seen_ids:
            continue
        row_hash = row.get("schedule_hash")
        if row_hash is None and row.get("schedule_grid") is not None:
            row_hash = schedule_hash(row["schedule_grid"])
        if row_hash is not None and str(row_hash) in seen_hashes:
            continue
        pending.append(dict(row))
        seen_ids.add(candidate_id)
        if row_hash is not None:
            seen_hashes.add(str(row_hash))
    return pending


def make_observation_payload(
    *,
    dataset: str,
    solver_key: str,
    target_nfe: int,
    runtime_nfe: int,
    uniform_metrics: Mapping[str, Any],
    observations: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    crps = float(uniform_metrics["crps"])
    mase = float(uniform_metrics["mase"])
    if crps <= 0.0 or mase <= 0.0 or not np.isfinite(crps) or not np.isfinite(mase):
        raise ValueError("Uniform CRPS and MASE baselines must be finite and positive.")
    return {
        "artifact": "forecast_bo_observations_v1",
        "objective_type": FORECAST_AVG_RELATIVE_OBJECTIVE,
        "dataset": str(dataset),
        "solver_key": str(solver_key),
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "uniform_baseline": {"crps": crps, "mase": mase},
        "uniform_validation_metrics": dict(uniform_metrics),
        "observations": [dict(row) for row in (observations or [])],
    }


def select_best_observation(observations_payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [dict(row) for row in observations_payload.get("observations", [])]
    if not rows:
        raise ValueError("Cannot select a best observation from an empty observation payload.")
    return max(rows, key=lambda row: (float(row["objective_value"]), -float(row.get("metric_val", 0.0))))


def select_top_observations(observations_payload: Mapping[str, Any], *, top_k: int) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in observations_payload.get("observations", [])]
    if not rows:
        raise ValueError("Cannot select top observations from an empty observation payload.")
    limit = max(1, int(top_k))
    return sorted(rows, key=lambda row: (float(row["objective_value"]), -float(row.get("metric_val", 0.0))), reverse=True)[:limit]


def validate_final_comparison_coverage(
    rows: Sequence[Mapping[str, Any]],
    solvers: Sequence[str],
    comparison_schedules: Sequence[str] = DEFAULT_COMPARISON_SCHEDULE_KEYS,
) -> None:
    missing: List[Tuple[str, str]] = []
    for solver in solvers:
        for schedule in comparison_schedules:
            if not any(str(row.get("solver_key")) == str(solver) and str(row.get("schedule_key")) == schedule for row in rows):
                missing.append((str(solver), str(schedule)))
    if missing:
        raise ValueError(f"Missing final comparison rows for {missing}.")


def summarize_final_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["solver_key"]), str(row["schedule_key"])), []).append(row)
    summaries: List[Dict[str, Any]] = []
    for (solver_key, schedule_key), group in sorted(groups.items()):
        summary: Dict[str, Any] = {
            "solver_key": solver_key,
            "schedule_key": schedule_key,
            "n_seeds": int(len(group)),
            "seed_values": sorted(int(row["seed"]) for row in group),
            "schedule_grid": list(group[0].get("schedule_grid", [])),
        }
        for metric in ("crps", "mase", "relative_crps_ratio", "relative_mase_ratio", "avg_relative_ratio", "kl_to_reference"):
            values = np.asarray([float(row[metric]) for row in group if row.get(metric) is not None], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean()) if values.size else None
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else (0.0 if values.size == 1 else None)
        summaries.append(summary)
    return summaries


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in keys})


def _core_imports() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    core_code_dir = project_root / "code"
    if core_code_dir.exists() and str(core_code_dir) not in sys.path:
        sys.path.insert(0, str(core_code_dir))
    from otflow_evaluation_support import (
        SOLVER_RUNTIME_NAMES,
        collect_forecast_calibration,
        evaluate_forecast_schedule,
        load_forecast_checkpoint_splits,
        solver_macro_steps,
    )
    from diffusion_flow_schedules import build_schedule_grid

    return {
        "SOLVER_RUNTIME_NAMES": SOLVER_RUNTIME_NAMES,
        "build_schedule_grid": build_schedule_grid,
        "collect_forecast_calibration": collect_forecast_calibration,
        "evaluate_forecast_schedule": evaluate_forecast_schedule,
        "load_forecast_checkpoint_splits": load_forecast_checkpoint_splits,
        "solver_macro_steps": solver_macro_steps,
    }


def _load_checkpoint(args: argparse.Namespace, workspace_root: Path, device: Any) -> Dict[str, Any]:
    core = _core_imports()
    loader_args = argparse.Namespace(
        backbone_manifest=str(workspace_root / "outputs" / "backbone_matrix" / "backbone_manifest.json"),
        shared_backbone_root=str(workspace_root / "outputs" / "shared_backbones" / "otflow_fullhorizon_seed0"),
        otflow_train_steps=int(args.otflow_train_steps),
        device=str(args.device),
        dataset_seed=0,
    )
    return core["load_forecast_checkpoint_splits"](
        cli_args=loader_args,
        dataset_root=workspace_root / "paper_datasets",
        shared_backbone_root=Path(str(loader_args.shared_backbone_root)),
        dataset=str(args.dataset),
        device=device,
    )


def _reference_checkpoint_id(reference: Mapping[str, Any]) -> Optional[str]:
    if reference.get("checkpoint_id") is not None:
        return str(reference["checkpoint_id"])
    if reference.get("source_checkpoint_id") is not None:
        return str(reference["source_checkpoint_id"])
    return None


def _reference_matches_current(
    reference: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    solver_key: str,
    runtime_nfe: int,
    calibration_indices: Sequence[int],
) -> bool:
    if str(reference.get("dataset")) != str(args.dataset):
        return False
    if str(reference.get("solver_key")) != str(solver_key):
        return False
    if int(reference.get("target_nfe", -1)) != int(args.target_nfe):
        return False
    if int(reference.get("runtime_nfe", -1)) != int(runtime_nfe):
        return False
    checkpoint_id = _reference_checkpoint_id(reference)
    if checkpoint_id is None or checkpoint_id != str(checkpoint["checkpoint_id"]):
        return False
    if reference.get("train_steps") is not None and int(reference["train_steps"]) != int(checkpoint["train_steps"]):
        return False
    if reference.get("reference_macro_factor") is None:
        return False
    if abs(float(reference["reference_macro_factor"]) - float(args.reference_macro_factor)) > 1e-12:
        return False
    if reference.get("density_floor_eta") is None:
        return False
    if abs(float(reference["density_floor_eta"]) - float(args.density_floor_eta)) > 1e-12:
        return False
    if reference.get("calibration_trace_samples") is not None and int(reference["calibration_trace_samples"]) != int(args.calibration_trace_samples):
        return False
    if [int(idx) for idx in reference.get("calibration_indices", [])] != [int(idx) for idx in calibration_indices]:
        return False
    if reference.get("schedule_grid") is None:
        return False
    expected_hash = schedule_hash(reference["schedule_grid"])
    if reference.get("schedule_hash") is not None and str(reference["schedule_hash"]) != expected_hash:
        return False
    return True


def _build_reference_schedule(
    *,
    args: argparse.Namespace,
    solver_key: str,
    solver_idx: int,
    checkpoint: Mapping[str, Any],
    calibration_subset: _IndexSubset,
    solver_out: Path,
    resume: bool,
) -> Dict[str, Any]:
    core = _core_imports()
    reference_path = solver_out / "reference_schedule.json"
    runtime_nfe = int(core["solver_macro_steps"](str(solver_key), int(args.target_nfe)))
    if resume and reference_path.exists():
        reference = load_json(reference_path)
        reference.setdefault("schedule_hash", schedule_hash(reference["schedule_grid"]))
        if _reference_matches_current(
            reference,
            args=args,
            checkpoint=checkpoint,
            solver_key=str(solver_key),
            runtime_nfe=int(runtime_nfe),
            calibration_indices=calibration_subset.indices,
        ):
            return reference

    reference_macro_steps = max(32, int(round(float(args.reference_macro_factor) * float(runtime_nfe))))
    runtime_solver = core["SOLVER_RUNTIME_NAMES"][str(solver_key)]
    calibration = core["collect_forecast_calibration"](
        checkpoint["model"],
        calibration_subset,
        checkpoint["cfg"],
        macro_steps=int(reference_macro_steps),
        solver_name=str(runtime_solver),
        seed=int(args.bo_seed) + 100_000 * int(solver_idx),
        calibration_trace_samples=int(args.calibration_trace_samples),
    )
    solver_order = 1.0 if str(solver_key) == "euler" else 2.0
    local_defect = local_defect_from_oracle(
        calibration["oracle_local_error_by_step"],
        calibration["reference_time_grid"],
        solver_order=float(solver_order),
    )
    cell = {
        "dataset": str(args.dataset),
        "benchmark_family": "forecast_extrapolation",
        "solver_key": str(solver_key),
        "target_nfe": int(args.target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "reference_macro_steps": int(reference_macro_steps),
        "reference_time_grid": calibration["reference_time_grid"],
        "validation_info_growth_trace": calibration["info_growth_hardness_by_step"],
        "validation_oracle_local_error_trace": calibration["oracle_local_error_by_step"],
        "validation_local_defect_trace": [float(x) for x in local_defect.tolist()],
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "train_steps": int(checkpoint["train_steps"]),
        "train_budget_label": str(checkpoint["train_budget_label"]),
    }
    reference = build_reference_from_cell(cell, eta=float(args.density_floor_eta))
    reference.update(
        {
            "reference_macro_factor": float(args.reference_macro_factor),
            "reference_macro_steps": int(reference_macro_steps),
            "calibration_indices": list(calibration_subset.indices),
            "calibration_indices_hash": indices_hash(calibration_subset.indices),
            "calibration_windows": int(len(calibration_subset)),
            "calibration_trace_samples": int(args.calibration_trace_samples),
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "train_steps": int(checkpoint["train_steps"]),
            "train_budget_label": str(checkpoint["train_budget_label"]),
            "schedule_hash": schedule_hash(reference["schedule_grid"]),
        }
    )
    write_json(reference, reference_path)
    return reference


def _reference_candidate(reference: Mapping[str, Any]) -> Dict[str, Any]:
    q_ref = np.asarray(reference["q_ref"], dtype=np.float64)
    basis = build_residual_basis(q_ref.size, q_ref=q_ref)
    record = theta_to_schedule_record(q_ref, np.zeros(int(basis.shape[1]), dtype=np.float64), basis=basis)
    record.update({"candidate_id": REFERENCE_CANDIDATE_ID, "source": "ser_ptg_reference_center", "basis_kind": DEFAULT_BASIS_KIND})
    return record


def _initial_candidates(reference: Mapping[str, Any], *, n_initial: int, seed: int) -> List[Dict[str, Any]]:
    payload = generate_initial_perturbations(reference, n_initial=int(n_initial), seed=int(seed), use_sobol=True)
    rows = []
    for row in payload["candidates"]:
        item = dict(row)
        item["source"] = "initial_sobol_kl_perturbation"
        rows.append(item)
    return rows


def _observations_match_default_basis(payload: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    q_ref = np.asarray(reference["q_ref"], dtype=np.float64)
    expected_dim = int(build_residual_basis(q_ref.size, q_ref=q_ref).shape[1])
    payload_basis = payload.get("basis_kind")
    if payload_basis is not None and str(payload_basis) != DEFAULT_BASIS_KIND:
        return False
    payload_dim = payload.get("basis_dim")
    if payload_dim is not None and int(payload_dim) != expected_dim:
        return False
    for row in payload.get("observations", []):
        row_basis = row.get("basis_kind")
        if row_basis is not None and str(row_basis) != DEFAULT_BASIS_KIND:
            return False
        row_dim = row.get("basis_dim")
        if row_dim is not None and int(row_dim) != expected_dim:
            return False
        theta = row.get("theta")
        if theta is not None and len(theta) != expected_dim:
            return False
    return True


def _bo_observation_run_metadata(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    reference: Mapping[str, Any],
    bo_val_subset: _IndexSubset,
    solver_key: str,
) -> Dict[str, Any]:
    return {
        "artifact": "forecast_bo_observation_run_metadata_v1",
        "dataset": str(args.dataset),
        "solver_key": str(solver_key),
        "target_nfe": int(args.target_nfe),
        "runtime_nfe": int(reference["runtime_nfe"]),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "train_steps": int(checkpoint["train_steps"]),
        "reference_schedule_hash": str(reference.get("schedule_hash") or schedule_hash(reference["schedule_grid"])),
        "reference_macro_factor": float(args.reference_macro_factor),
        "calibration_fraction": float(args.calibration_fraction),
        "bo_validation_indices_hash": indices_hash(bo_val_subset.indices),
        "bo_validation_windows": int(len(bo_val_subset)),
        "num_eval_samples": int(args.num_eval_samples),
        "bo_budget": int(args.bo_budget),
        "n_initial": validate_n_initial(args.n_initial),
        "bo_batch_size": int(args.bo_batch_size),
        "lambda_kl": float(args.lambda_kl),
        "bo_seed": int(args.bo_seed),
        "theta_bound": float(args.theta_bound),
        "raw_samples": int(args.raw_samples),
        "num_restarts": int(args.num_restarts),
        "mc_samples": int(args.mc_samples),
        "density_floor_eta": float(args.density_floor_eta),
        "calibration_trace_samples": int(args.calibration_trace_samples),
        "basis_kind": DEFAULT_BASIS_KIND,
        "basis_dim": int(build_residual_basis(len(reference["q_ref"]), q_ref=reference["q_ref"]).shape[1]),
        "device": str(args.device),
    }


def _payload_matches_run_metadata(payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    return str(payload.get("run_fingerprint", "")) == stable_fingerprint(metadata) and payload.get("run_metadata") == dict(metadata)


def _source_candidate_hashes(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    hashes: List[str] = []
    for row in rows:
        row_hash = row.get("schedule_hash")
        if row_hash is None and row.get("schedule_grid") is not None:
            row_hash = schedule_hash(row["schedule_grid"])
        hashes.append(str(row_hash))
    return hashes


def _bo_confirmation_run_metadata(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    reference: Mapping[str, Any],
    observations: Mapping[str, Any],
    confirm_subset: _IndexSubset,
    solver_key: str,
    top_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "artifact": "forecast_bo_confirmation_run_metadata_v1",
        "dataset": str(args.dataset),
        "solver_key": str(solver_key),
        "target_nfe": int(args.target_nfe),
        "runtime_nfe": int(reference["runtime_nfe"]),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "reference_schedule_hash": str(reference.get("schedule_hash") or schedule_hash(reference["schedule_grid"])),
        "observation_run_fingerprint": str(observations.get("run_fingerprint", "")),
        "source_candidate_ids": [str(row.get("candidate_id", row.get("observation_id", ""))) for row in top_rows],
        "source_candidate_hashes": _source_candidate_hashes(top_rows),
        "selection_rule": f"top_{int(args.confirm_top_k)}_by_bo_validation_objective",
        "confirmation_indices_hash": indices_hash(confirm_subset.indices),
        "confirmation_windows": int(len(confirm_subset)),
        "num_eval_samples": int(args.num_eval_samples),
        "lambda_kl": float(args.lambda_kl),
        "bo_seed": int(args.bo_seed),
        "basis_kind": DEFAULT_BASIS_KIND,
    }


def _evaluate_schedule(core: Mapping[str, Any], checkpoint: Mapping[str, Any], ds: Any, *, solver_key: str, runtime_nfe: int, time_grid: Sequence[float], num_eval_samples: int, seed: int) -> Dict[str, Any]:
    return core["evaluate_forecast_schedule"](
        checkpoint["model"],
        ds,
        checkpoint["cfg"],
        solver_name=str(core["SOLVER_RUNTIME_NAMES"][str(solver_key)]),
        runtime_nfe=int(runtime_nfe),
        time_grid=[float(x) for x in time_grid],
        num_eval_samples=int(num_eval_samples),
        seed=int(seed),
    )


def _normalize_candidate_observation(reference: Mapping[str, Any], payload: Mapping[str, Any], candidate: Mapping[str, Any], metrics: Mapping[str, Any], *, lambda_kl: float) -> Dict[str, Any]:
    row = dict(candidate)
    for key in (
        "crps",
        "mase",
        "mse",
        "latency_ms_per_sample",
        "eval_examples",
        "num_eval_samples",
        "uniform_crps",
        "uniform_mase",
        "relative_crps_ratio",
        "relative_mase_ratio",
        "metric_val",
        "objective_value",
        "objective_type",
        "observation_id",
    ):
        row.pop(key, None)
    row.update(
        {
            "crps": float(metrics["crps"]),
            "mase": float(metrics["mase"]),
            "mse": float(metrics.get("mse", float("nan"))),
            "latency_ms_per_sample": float(metrics.get("latency_ms_per_sample", float("nan"))),
            "eval_examples": int(metrics.get("eval_examples", 0)),
            "num_eval_samples": int(metrics.get("num_eval_samples", 0)),
        }
    )
    normalized = normalize_observations(
        {"uniform_baseline": payload["uniform_baseline"], "observations": [row]},
        q_ref=reference["q_ref"],
        lambda_kl=float(lambda_kl),
    )
    out = normalized[0]
    out["candidate_id"] = str(candidate["candidate_id"])
    out["source"] = str(candidate.get("source", "unknown"))
    out["schedule_hash"] = schedule_hash(out["schedule_grid"])
    return out


def _load_or_create_observations(
    *,
    args: argparse.Namespace,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    bo_val_subset: _IndexSubset,
    reference: Mapping[str, Any],
    solver_key: str,
    solver_out: Path,
    resume: bool,
) -> Dict[str, Any]:
    observations_path = solver_out / "observations.json"
    run_metadata = _bo_observation_run_metadata(
        args=args,
        checkpoint=checkpoint,
        reference=reference,
        bo_val_subset=bo_val_subset,
        solver_key=str(solver_key),
    )
    if resume and observations_path.exists():
        payload = load_json(observations_path)
        if _observations_match_default_basis(payload, reference) and _payload_matches_run_metadata(payload, run_metadata):
            return payload

    uniform_grid = core["build_schedule_grid"]("uniform", int(reference["runtime_nfe"]))
    uniform_metrics = _evaluate_schedule(
        core,
        checkpoint,
        bo_val_subset,
        solver_key=str(solver_key),
        runtime_nfe=int(reference["runtime_nfe"]),
        time_grid=uniform_grid,
        num_eval_samples=int(args.num_eval_samples),
        seed=int(args.bo_seed) + 17_000,
    )
    payload = make_observation_payload(
        dataset=str(args.dataset),
        solver_key=str(solver_key),
        target_nfe=int(args.target_nfe),
        runtime_nfe=int(reference["runtime_nfe"]),
        uniform_metrics=uniform_metrics,
    )
    payload.update(
        {
            "validation_protocol": "calibration_bo_validation_split",
            "bo_validation_indices": list(bo_val_subset.indices),
            "bo_validation_windows": int(len(bo_val_subset)),
            "basis_kind": DEFAULT_BASIS_KIND,
            "basis_dim": int(build_residual_basis(len(reference["q_ref"]), q_ref=reference["q_ref"]).shape[1]),
            "run_metadata": run_metadata,
            "run_fingerprint": stable_fingerprint(run_metadata),
        }
    )
    write_json(payload, observations_path)
    return payload


def _append_observation(payload: Dict[str, Any], row: Mapping[str, Any], path: Path) -> None:
    payload.setdefault("observations", []).append(dict(row))
    write_json(payload, path)


def _confirmation_best_observation(
    *,
    args: argparse.Namespace,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    solver_key: str,
    reference: Mapping[str, Any],
    observations: Mapping[str, Any],
    confirm_subset: _IndexSubset,
    solver_out: Path,
    resume: bool,
) -> Dict[str, Any]:
    confirm_path = solver_out / "confirmation_rows.json"
    top_rows = select_top_observations(observations, top_k=int(args.confirm_top_k))
    run_metadata = _bo_confirmation_run_metadata(
        args=args,
        checkpoint=checkpoint,
        solver_key=str(solver_key),
        reference=reference,
        observations=observations,
        confirm_subset=confirm_subset,
        top_rows=top_rows,
    )
    if resume and confirm_path.exists():
        payload = load_json(confirm_path)
        rows = payload.get("rows", [])
        if rows and _payload_matches_run_metadata(payload, run_metadata):
            return max(rows, key=lambda row: (float(row["objective_value"]), -float(row.get("metric_val", 0.0))))

    uniform_grid = core["build_schedule_grid"]("uniform", int(reference["runtime_nfe"]))
    uniform_metrics = _evaluate_schedule(
        core,
        checkpoint,
        confirm_subset,
        solver_key=str(solver_key),
        runtime_nfe=int(reference["runtime_nfe"]),
        time_grid=uniform_grid,
        num_eval_samples=int(args.num_eval_samples),
        seed=int(args.bo_seed) + 37_000,
    )
    confirm_payload = make_observation_payload(
        dataset=str(args.dataset),
        solver_key=str(solver_key),
        target_nfe=int(args.target_nfe),
        runtime_nfe=int(reference["runtime_nfe"]),
        uniform_metrics=uniform_metrics,
    )
    confirm_payload.update(
        {
            "artifact": "forecast_bo_confirmation_rows_v1",
            "selection_rule": f"top_{int(args.confirm_top_k)}_by_bo_validation_objective",
            "confirmation_indices": list(confirm_subset.indices),
            "confirmation_windows": int(len(confirm_subset)),
            "basis_kind": DEFAULT_BASIS_KIND,
            "run_metadata": run_metadata,
            "run_fingerprint": stable_fingerprint(run_metadata),
        }
    )
    rows: List[Dict[str, Any]] = []
    for candidate in top_rows:
        metrics = _evaluate_schedule(
            core,
            checkpoint,
            confirm_subset,
            solver_key=str(solver_key),
            runtime_nfe=int(reference["runtime_nfe"]),
            time_grid=candidate["schedule_grid"],
            num_eval_samples=int(args.num_eval_samples),
            seed=int(args.bo_seed) + 37_000,
        )
        row = _normalize_candidate_observation(reference, confirm_payload, candidate, metrics, lambda_kl=float(args.lambda_kl))
        row["confirmation_source_objective_value"] = float(candidate["objective_value"])
        row["confirmation_source_metric_val"] = float(candidate["metric_val"])
        rows.append(row)
    confirm_payload["rows"] = rows
    write_json(confirm_payload, confirm_path)
    return max(rows, key=lambda row: (float(row["objective_value"]), -float(row.get("metric_val", 0.0))))


def _run_solver_bo(
    *,
    args: argparse.Namespace,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    solver_key: str,
    solver_idx: int,
    out_root: Path,
    resume: bool,
) -> Dict[str, Any]:
    solver_out = out_root / str(solver_key)
    solver_out.mkdir(parents=True, exist_ok=True)
    calibration_pool, bo_validation_pool = deterministic_validation_partition(
        len(checkpoint["splits"]["val"]),
        calibration_fraction=float(args.calibration_fraction),
        seed=int(args.bo_seed) + 1_000 * int(solver_idx),
    )
    calibration_indices = selected_indices_from_pool(
        calibration_pool,
        int(args.calibration_windows),
        seed=int(args.bo_seed) + 2_000 * int(solver_idx) + 11,
    )
    bo_val_indices = selected_indices_from_pool(
        bo_validation_pool,
        int(args.bo_val_windows),
        seed=int(args.bo_seed) + 2_000 * int(solver_idx) + 17,
    )
    confirm_val_indices = selected_indices_from_pool(
        bo_validation_pool,
        int(args.confirm_val_windows),
        seed=int(args.bo_seed) + 2_000 * int(solver_idx) + 23,
    )
    split_payload = {
        "artifact": "forecast_bo_validation_split_v1",
        "calibration_fraction": float(args.calibration_fraction),
        "calibration_pool_size": int(len(calibration_pool)),
        "bo_validation_pool_size": int(len(bo_validation_pool)),
        "calibration_indices": calibration_indices,
        "bo_validation_indices": bo_val_indices,
        "confirmation_indices": confirm_val_indices,
    }
    write_json(split_payload, solver_out / "validation_split.json")
    calibration_subset = _IndexSubset(checkpoint["splits"]["val"], calibration_indices)
    bo_val_subset = _IndexSubset(checkpoint["splits"]["val"], bo_val_indices)
    confirm_subset = _IndexSubset(checkpoint["splits"]["val"], confirm_val_indices)
    reference = _build_reference_schedule(
        args=args,
        solver_key=str(solver_key),
        solver_idx=int(solver_idx),
        checkpoint=checkpoint,
        calibration_subset=calibration_subset,
        solver_out=solver_out,
        resume=resume,
    )
    observations_path = solver_out / "observations.json"
    observations = _load_or_create_observations(
        args=args,
        core=core,
        checkpoint=checkpoint,
        bo_val_subset=bo_val_subset,
        reference=reference,
        solver_key=str(solver_key),
        solver_out=solver_out,
        resume=resume,
    )

    budget = int(args.bo_budget)
    breakdown = candidate_budget_breakdown(budget, n_initial=int(args.n_initial))
    planned = [_reference_candidate(reference)] + _initial_candidates(
        reference,
        n_initial=int(args.n_initial),
        seed=int(args.bo_seed) + 10_000 * int(solver_idx),
    )[: breakdown["initial"]]
    for candidate in pending_candidate_records(planned, observations):
        if len(observations.get("observations", [])) >= budget:
            break
        metrics = _evaluate_schedule(
            core,
            checkpoint,
            bo_val_subset,
            solver_key=str(solver_key),
            runtime_nfe=int(reference["runtime_nfe"]),
            time_grid=candidate["schedule_grid"],
            num_eval_samples=int(args.num_eval_samples),
            seed=int(args.bo_seed) + 23_000,
        )
        row = _normalize_candidate_observation(reference, observations, candidate, metrics, lambda_kl=float(args.lambda_kl))
        _append_observation(observations, row, observations_path)

    while len(observations.get("observations", [])) < budget:
        remaining = budget - len(observations.get("observations", []))
        batch = suggest_bo_batch(
            reference,
            observations,
            batch_size=min(int(args.bo_batch_size), int(remaining)),
            lambda_kl=float(args.lambda_kl),
            theta_bound=float(args.theta_bound),
            raw_samples=int(args.raw_samples),
            num_restarts=int(args.num_restarts),
            n_mc_samples=int(args.mc_samples),
            seed=int(args.bo_seed) + len(observations.get("observations", [])),
        )
        seen_hashes = observed_schedule_hashes(observations)
        accepted_in_batch: set[str] = set()
        before_batch_count = len(observations.get("observations", []))
        for candidate in batch["candidates"]:
            if len(observations.get("observations", [])) >= budget:
                break
            candidate_hash = candidate.get("schedule_hash")
            if candidate_hash is None and candidate.get("schedule_grid") is not None:
                candidate_hash = schedule_hash(candidate["schedule_grid"])
            if candidate_hash is not None and (str(candidate_hash) in seen_hashes or str(candidate_hash) in accepted_in_batch):
                continue
            seq = len(observations.get("observations", []))
            item = dict(candidate)
            item["candidate_id"] = f"bo_{seq:03d}"
            item["source"] = "qLogNoisyExpectedImprovement"
            metrics = _evaluate_schedule(
                core,
                checkpoint,
                bo_val_subset,
                solver_key=str(solver_key),
                runtime_nfe=int(reference["runtime_nfe"]),
                time_grid=item["schedule_grid"],
                num_eval_samples=int(args.num_eval_samples),
                seed=int(args.bo_seed) + 23_000,
            )
            row = _normalize_candidate_observation(reference, observations, item, metrics, lambda_kl=float(args.lambda_kl))
            _append_observation(observations, row, observations_path)
            accepted_in_batch.add(str(row["schedule_hash"]))
            seen_hashes.add(str(row["schedule_hash"]))
        if len(observations.get("observations", [])) == before_batch_count:
            raise RuntimeError("BO candidate suggestion produced only duplicate schedule hashes.")

    best = _confirmation_best_observation(
        args=args,
        core=core,
        checkpoint=checkpoint,
        solver_key=str(solver_key),
        reference=reference,
        observations=observations,
        confirm_subset=confirm_subset,
        solver_out=solver_out,
        resume=resume,
    )
    best_payload = {
        "artifact": "forecast_bo_best_schedule_v1",
        "dataset": str(args.dataset),
        "solver_key": str(solver_key),
        "target_nfe": int(args.target_nfe),
        "runtime_nfe": int(reference["runtime_nfe"]),
        "selected_by": f"max_confirmation_objective_value_after_top_{int(args.confirm_top_k)}_bo_validation_rescore",
        "best_observation": best,
        "reference_schedule_path": str(solver_out / "reference_schedule.json"),
        "observations_path": str(observations_path),
        "confirmation_rows_path": str(solver_out / "confirmation_rows.json"),
        "validation_split_path": str(solver_out / "validation_split.json"),
    }
    write_json(best_payload, solver_out / "best_schedule.json")
    return {"reference": reference, "observations": observations, "best": best_payload, "solver_out": solver_out}


def _final_row_key(row: Mapping[str, Any]) -> Tuple[str, int, str, str, int]:
    return (
        str(row["dataset"]),
        int(row["target_nfe"]),
        str(row["solver_key"]),
        str(row["schedule_key"]),
        int(row["seed"]),
    )


def _load_final_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    return [dict(row) for row in payload.get("rows", [])]


def _load_cached_final_rows(cache_roots: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in cache_roots:
        final_path = Path(root) / "final_comparison_rows.json"
        if not final_path.exists():
            continue
        run_config_path = Path(root) / "run_config.json"
        run_config = load_json(run_config_path) if run_config_path.exists() else {}
        for row in _load_final_rows(final_path):
            item = dict(row)
            item["_cache_run_config"] = run_config
            rows.append(item)
    return rows


def _cache_roots(args: argparse.Namespace, workspace_root: Path) -> List[Path]:
    roots: List[Path] = []
    for item in parse_csv(str(getattr(args, "baseline_cache_roots", ""))):
        roots.append(resolve_workspace_path(item, workspace_root))
    return roots


def _row_schedule_hash(row: Mapping[str, Any]) -> Optional[str]:
    if row.get("schedule_hash") is not None:
        return str(row["schedule_hash"])
    return None


def _row_checkpoint_id(row: Mapping[str, Any]) -> Optional[str]:
    if row.get("checkpoint_id") is not None:
        return str(row["checkpoint_id"])
    run_config = row.get("_cache_run_config", {})
    if isinstance(run_config, Mapping) and run_config.get("checkpoint_id") is not None:
        return str(run_config["checkpoint_id"])
    return None


def _matching_cached_final_row(
    cached_rows: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    solver_key: str,
    schedule_key: str,
    seed: int,
    runtime_nfe: int,
    schedule_grid: Sequence[float],
    expected_eval_examples: int,
    expected_test_indices_hash: str,
) -> Optional[Dict[str, Any]]:
    if schedule_key not in TEST_BASELINE_REUSE_KEYS:
        return None
    expected_hash = schedule_hash(schedule_grid)
    for row in cached_rows:
        if str(row.get("dataset")) != str(args.dataset):
            continue
        if str(row.get("solver_key")) != str(solver_key):
            continue
        if str(row.get("schedule_key")) != str(schedule_key):
            continue
        if int(row.get("target_nfe", -1)) != int(args.target_nfe):
            continue
        if int(row.get("runtime_nfe", -1)) != int(runtime_nfe):
            continue
        if int(row.get("seed", -1)) != int(seed):
            continue
        if int(row.get("num_eval_samples", -1)) != int(args.num_eval_samples):
            continue
        if int(row.get("eval_examples", -1)) != int(expected_eval_examples):
            continue
        checkpoint_id = _row_checkpoint_id(row)
        if checkpoint_id is None or checkpoint_id != str(checkpoint["checkpoint_id"]):
            continue
        if _row_test_indices_hash(row) != str(expected_test_indices_hash):
            continue
        if _row_schedule_hash(row) != expected_hash:
            continue
        reused = {key: value for key, value in dict(row).items() if not str(key).startswith("_cache_")}
        reused["reused_from_cache"] = True
        reused["schedule_hash"] = expected_hash
        reused["checkpoint_id"] = str(checkpoint["checkpoint_id"])
        if checkpoint.get("train_steps") is not None:
            reused["train_steps"] = int(checkpoint["train_steps"])
        if checkpoint.get("train_budget_label") is not None:
            reused["train_budget_label"] = str(checkpoint["train_budget_label"])
        reused["test_indices_hash"] = str(expected_test_indices_hash)
        return reused
    return None


def _final_row_matches_current(
    row: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    runtime_nfe: int,
    schedule_grid: Sequence[float],
    num_eval_samples: int,
    expected_eval_examples: int,
    expected_test_indices_hash: str,
) -> bool:
    if str(row.get("checkpoint_id", "")) != str(checkpoint["checkpoint_id"]):
        return False
    if int(row.get("runtime_nfe", -1)) != int(runtime_nfe):
        return False
    if int(row.get("num_eval_samples", -1)) != int(num_eval_samples):
        return False
    if int(row.get("eval_examples", -1)) != int(expected_eval_examples):
        return False
    if _row_test_indices_hash(row) != str(expected_test_indices_hash):
        return False
    return _row_schedule_hash(row) == schedule_hash(schedule_grid)


def _row_test_indices_hash(row: Mapping[str, Any]) -> str:
    if row.get("test_indices_hash") is not None:
        return str(row["test_indices_hash"])
    if row.get("test_indices") is not None:
        return indices_hash([int(idx) for idx in row["test_indices"]])
    return ""


def _final_row_base_payload(
    *,
    checkpoint: Mapping[str, Any],
    test_indices: Sequence[int],
    test_indices_hash: str,
    num_eval_samples: int,
) -> Dict[str, Any]:
    return {
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "train_steps": int(checkpoint["train_steps"]),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "num_eval_samples": int(num_eval_samples),
        "test_indices": [int(idx) for idx in test_indices],
        "test_indices_hash": str(test_indices_hash),
    }


def _run_final_comparison(
    *,
    args: argparse.Namespace,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    solver_results: Mapping[str, Mapping[str, Any]],
    out_root: Path,
    resume: bool,
) -> List[Dict[str, Any]]:
    final_path = out_root / "final_comparison_rows.json"
    rows = _load_final_rows(final_path) if resume else []
    comparison_schedules = parse_comparison_schedules(str(args.comparison_schedules))
    final_seeds = parse_int_csv(str(args.final_test_seeds))
    expected_keys = {
        (str(args.dataset), int(args.target_nfe), str(solver_key), str(schedule_key), int(seed))
        for solver_key in solver_results
        for schedule_key in comparison_schedules
        for seed in final_seeds
    }
    rows = [row for row in rows if _final_row_key(row) in expected_keys]
    by_key = {_final_row_key(row): dict(row) for row in rows}
    for solver_key, result in solver_results.items():
        reference = result["reference"]
        best = result["best"]["best_observation"]
        runtime_nfe = int(reference["runtime_nfe"])
        schedules: Dict[str, List[float]] = {}
        kl_values: Dict[str, Optional[float]] = {}
        for schedule_key in comparison_schedules:
            if schedule_key == "ser_ptg_reference":
                schedules[schedule_key] = list(reference["schedule_grid"])
                kl_values[schedule_key] = 0.0
            elif schedule_key == "bo_best":
                schedules[schedule_key] = list(best["schedule_grid"])
                kl_values[schedule_key] = float(best["kl_to_reference"])
            else:
                schedules[schedule_key] = list(core["build_schedule_grid"](str(schedule_key), runtime_nfe))
                kl_values[schedule_key] = None
        for seed in final_seeds:
            test_indices = selected_indices(
                len(checkpoint["splits"]["test"]),
                int(args.final_test_windows),
                seed=int(args.bo_seed) + 50_000 + int(seed),
            )
            test_hash = indices_hash(test_indices)
            test_ds = _IndexSubset(checkpoint["splits"]["test"], test_indices)
            uniform_key = (str(args.dataset), int(args.target_nfe), str(solver_key), "uniform", int(seed))
            if uniform_key in by_key and not _final_row_matches_current(
                by_key[uniform_key],
                checkpoint=checkpoint,
                runtime_nfe=runtime_nfe,
                schedule_grid=schedules["uniform"],
                num_eval_samples=int(args.num_eval_samples),
                expected_eval_examples=len(test_ds),
                expected_test_indices_hash=test_hash,
            ):
                by_key.pop(uniform_key, None)
            if uniform_key not in by_key:
                metrics = _evaluate_schedule(
                    core,
                    checkpoint,
                    test_ds,
                    solver_key=str(solver_key),
                    runtime_nfe=runtime_nfe,
                    time_grid=schedules["uniform"],
                    num_eval_samples=int(args.num_eval_samples),
                    seed=int(seed),
                )
                by_key[uniform_key] = {
                    "dataset": str(args.dataset),
                    "solver_key": str(solver_key),
                    "target_nfe": int(args.target_nfe),
                    "runtime_nfe": runtime_nfe,
                    "seed": int(seed),
                    "schedule_key": "uniform",
                    "schedule_grid": [float(x) for x in schedules["uniform"]],
                    "schedule_hash": schedule_hash(schedules["uniform"]),
                    **_final_row_base_payload(
                        checkpoint=checkpoint,
                        test_indices=test_indices,
                        test_indices_hash=test_hash,
                        num_eval_samples=int(args.num_eval_samples),
                    ),
                    "crps": float(metrics["crps"]),
                    "mase": float(metrics["mase"]),
                    "mse": float(metrics.get("mse", float("nan"))),
                    "relative_crps_ratio": 1.0,
                    "relative_mase_ratio": 1.0,
                    "avg_relative_ratio": 1.0,
                    "kl_to_reference": None,
                    "eval_examples": int(metrics.get("eval_examples", 0)),
                }
                write_json({"artifact": "forecast_bo_final_comparison_rows_v1", "rows": list(by_key.values())}, final_path)
            uniform = by_key[uniform_key]
            for schedule_key in [key for key in comparison_schedules if key != "uniform"]:
                key = (str(args.dataset), int(args.target_nfe), str(solver_key), schedule_key, int(seed))
                if key in by_key and not _final_row_matches_current(
                    by_key[key],
                    checkpoint=checkpoint,
                    runtime_nfe=runtime_nfe,
                    schedule_grid=schedules[schedule_key],
                    num_eval_samples=int(args.num_eval_samples),
                    expected_eval_examples=len(test_ds),
                    expected_test_indices_hash=test_hash,
                ):
                    by_key.pop(key, None)
                if key in by_key:
                    continue
                metrics = _evaluate_schedule(
                    core,
                    checkpoint,
                    test_ds,
                    solver_key=str(solver_key),
                    runtime_nfe=runtime_nfe,
                    time_grid=schedules[schedule_key],
                    num_eval_samples=int(args.num_eval_samples),
                    seed=int(seed),
                )
                r_crps = float(metrics["crps"]) / float(uniform["crps"])
                r_mase = float(metrics["mase"]) / float(uniform["mase"])
                by_key[key] = {
                    "dataset": str(args.dataset),
                    "solver_key": str(solver_key),
                    "target_nfe": int(args.target_nfe),
                    "runtime_nfe": runtime_nfe,
                    "seed": int(seed),
                    "schedule_key": str(schedule_key),
                    "schedule_grid": [float(x) for x in schedules[schedule_key]],
                    "schedule_hash": schedule_hash(schedules[schedule_key]),
                    **_final_row_base_payload(
                        checkpoint=checkpoint,
                        test_indices=test_indices,
                        test_indices_hash=test_hash,
                        num_eval_samples=int(args.num_eval_samples),
                    ),
                    "crps": float(metrics["crps"]),
                    "mase": float(metrics["mase"]),
                    "mse": float(metrics.get("mse", float("nan"))),
                    "relative_crps_ratio": r_crps,
                    "relative_mase_ratio": r_mase,
                    "avg_relative_ratio": float(0.5 * (r_crps + r_mase)),
                    "kl_to_reference": kl_values[schedule_key],
                    "eval_examples": int(metrics.get("eval_examples", 0)),
                }
                write_json({"artifact": "forecast_bo_final_comparison_rows_v1", "rows": list(by_key.values())}, final_path)
    rows = list(by_key.values())
    validate_final_comparison_coverage(rows, list(solver_results.keys()), comparison_schedules)
    write_json({"artifact": "forecast_bo_final_comparison_rows_v1", "rows": rows}, final_path)
    write_csv(rows, out_root / "final_comparison_rows.csv")
    summaries = summarize_final_rows(rows)
    write_json({"artifact": "forecast_bo_final_summary_v1", "summaries": summaries}, out_root / "final_summary.json")
    write_csv(summaries, out_root / "final_summary.csv")
    return rows


def run_forecast_bo(args: argparse.Namespace) -> Dict[str, Any]:
    core = _core_imports()
    import torch

    workspace_root = resolve_workspace_path(str(args.workspace_root), Path.cwd())
    out_root = resolve_workspace_path(str(args.out_root), workspace_root)
    out_root.mkdir(parents=True, exist_ok=True)
    solvers = parse_csv(str(args.solvers))
    comparison_schedules = parse_comparison_schedules(str(args.comparison_schedules))
    validate_n_initial(args.n_initial)
    device = torch.device(str(args.device))
    checkpoint = _load_checkpoint(args, workspace_root, device)
    run_config = {
        "artifact": "forecast_bo_run_config_v1",
        "dataset": str(args.dataset),
        "solvers": solvers,
        "comparison_schedules": comparison_schedules,
        "target_nfe": int(args.target_nfe),
        "otflow_train_steps": int(args.otflow_train_steps),
        "bo_budget": int(args.bo_budget),
        "n_initial": int(args.n_initial),
        "bo_batch_size": int(args.bo_batch_size),
        "lambda_kl": float(args.lambda_kl),
        "bo_seed": int(args.bo_seed),
        "theta_bound": float(args.theta_bound),
        "raw_samples": int(args.raw_samples),
        "num_restarts": int(args.num_restarts),
        "mc_samples": int(args.mc_samples),
        "density_floor_eta": float(args.density_floor_eta),
        "basis_kind": DEFAULT_BASIS_KIND,
        "basis_dim": 5,
        "calibration_fraction": float(args.calibration_fraction),
        "calibration_windows": int(args.calibration_windows),
        "bo_val_windows": int(args.bo_val_windows),
        "confirm_top_k": int(args.confirm_top_k),
        "confirm_val_windows": int(args.confirm_val_windows),
        "num_eval_samples": int(args.num_eval_samples),
        "reference_macro_factor": float(args.reference_macro_factor),
        "calibration_trace_samples": int(args.calibration_trace_samples),
        "final_test_seeds": parse_int_csv(str(args.final_test_seeds)),
        "final_test_windows": int(args.final_test_windows),
        "device": str(args.device),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "train_steps": int(checkpoint["train_steps"]),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "checkpoint_metadata": {
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "train_steps": int(checkpoint["train_steps"]),
            "train_budget_label": str(checkpoint["train_budget_label"]),
        },
    }
    write_json(run_config, out_root / "run_config.json")
    solver_results: Dict[str, Mapping[str, Any]] = {}
    for solver_idx, solver_key in enumerate(solvers):
        solver_results[str(solver_key)] = _run_solver_bo(
            args=args,
            core=core,
            checkpoint=checkpoint,
            solver_key=str(solver_key),
            solver_idx=int(solver_idx),
            out_root=out_root,
            resume=bool(args.resume),
        )
    final_rows = _run_final_comparison(
        args=args,
        core=core,
        checkpoint=checkpoint,
        solver_results=solver_results,
        out_root=out_root,
        resume=bool(args.resume),
    )
    payload = {
        "artifact": "forecast_bo_run_summary_v1",
        "run_config": run_config,
        "final_row_count": int(len(final_rows)),
        "solver_observation_counts": {
            solver: int(len(result["observations"].get("observations", []))) for solver, result in solver_results.items()
        },
    }
    write_json(payload, out_root / "run_summary.json")
    return payload


def add_run_forecast_bo_parser(subparsers: Any) -> None:
    run = subparsers.add_parser("run-forecast-bo", help="Run closed-loop forecast BO schedule search.")
    run.add_argument("--workspace-root", type=Path, default=Path.cwd())
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument("--dataset", type=str, required=True)
    run.add_argument("--target-nfe", type=int, required=True)
    run.add_argument("--solvers", type=str, default="euler,dpmpp2m")
    run.add_argument("--otflow-train-steps", type=int, default=20000)
    run.add_argument("--bo-budget", type=int, default=100)
    run.add_argument("--calibration-fraction", type=float, default=0.7)
    run.add_argument("--calibration-windows", type=int, default=64)
    run.add_argument("--bo-val-windows", type=int, default=64)
    run.add_argument("--confirm-top-k", type=int, default=5)
    run.add_argument("--confirm-val-windows", type=int, default=0)
    run.add_argument("--num-eval-samples", type=int, default=5)
    run.add_argument("--reference-macro-factor", type=float, default=16.0)
    run.add_argument("--calibration-trace-samples", type=int, default=1)
    run.add_argument("--final-test-seeds", type=str, default="0,1,2")
    run.add_argument("--final-test-windows", type=int, default=0)
    run.add_argument("--comparison-schedules", type=str, default=",".join(DEFAULT_COMPARISON_SCHEDULE_KEYS))
    run.add_argument("--n-initial", type=int, default=16)
    run.add_argument("--bo-batch-size", type=int, default=2)
    run.add_argument("--lambda-kl", type=float, default=0.05)
    run.add_argument("--theta-bound", type=float, default=3.0)
    run.add_argument("--raw-samples", type=int, default=128)
    run.add_argument("--num-restarts", type=int, default=16)
    run.add_argument("--mc-samples", type=int, default=128)
    run.add_argument("--bo-seed", type=int, default=0)
    run.add_argument("--density-floor-eta", type=float, default=0.05)
    run.add_argument("--device", type=str, default="cuda")
    run.add_argument("--resume", action="store_true", default=True)
    run.add_argument("--no-resume", dest="resume", action="store_false")
