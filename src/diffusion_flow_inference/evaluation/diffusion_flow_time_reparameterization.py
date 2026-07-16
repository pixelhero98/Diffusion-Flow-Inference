from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_flow_inference.data.otflow_medical_constants import sleep_edf_metadata_path_for_npz
from diffusion_flow_inference.data.otflow_monash_datasets import (
    monash_manifest_path,
    monash_source_dir,
)
from diffusion_flow_inference.data.otflow_paths import (
    backbone_manifest_path,
    cryptos_data_path,
    es_mbp_10_data_path,
    project_outputs_root,
    project_paper_dataset_root,
    project_root,
    resolve_project_path,
    sleep_edf_data_path,
)
from diffusion_flow_inference.evaluation.otflow_evaluation_support import (
    ALL_SOLVER_ORDER,
    CONDITIONAL_GENERATION_FAMILY,
    FORECAST_FAMILY,
    LOCKED_TEST_PHASE,
    PAPER_CONDITIONAL_GENERATION_DATASETS,
    PAPER_FORECAST_DATASETS,
    SHARED_BACKBONE_ROOT,
    SOLVER_RUNTIME_NAMES,
    UNIFORM_SCHEDULE_KEY,
    VALIDATION_PHASE,
    choose_forecast_example_indices,
    evaluate_forecast_schedule,
    load_conditional_generation_checkpoint_splits,
    load_forecast_checkpoint_splits,
    parse_conditional_generation_datasets,
    parse_csv,
    parse_forecast_datasets,
    parse_int_csv,
    resolved_eval_horizon,
    resolved_eval_windows,
    selection_metric_for_family,
    solver_eval_multiplier,
    solver_experiment_scope,
    solver_macro_steps,
    validate_execution_preflight,
)
from diffusion_flow_inference.evaluation.otflow_sampling_support import _choose_valid_windows
from diffusion_flow_inference.schedule_transfer.diffusion_flow_schedules import (
    SCHEDULE_KEYS,
    TRANSFER_SCHEDULE_KEYS,
    build_schedule_grid,
    fixed_schedule_shape_statistics,
    run_fixed_schedule_variant,
    schedule_display_name,
    schedule_time_alignment,
)
from diffusion_flow_inference.schedule_transfer.result_tables import (
    augment_rows_with_relative_metrics,
)

RUNNER_PROTOCOL = "diffusion_flow_time_reparameterization"
RUNNER_OUTPUT_ROOT = project_outputs_root() / "diffusion_flow_time_reparameterization"
RUNNER_TARGET_NFE_VALUES: Tuple[int, ...] = (10, 12, 16)
RUNNER_SEEDS: Tuple[int, ...] = (0, 1, 2)
RUNNER_SCHEDULE_KEYS: Tuple[str, ...] = SCHEDULE_KEYS

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
    "schedule_key",
    "schedule_name",
    "row_signature",
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
    "disc_auc",
    "disc_auc_gap",
    "unconditional_w1",
    "conditional_w1",
    "tstr_macro_f1",
    "u_l1",
    "c_l1",
    "spread_specific_error",
    "imbalance_specific_error",
    "ret_vol_acf_error",
    "impact_response_error",
    "stage_mismatch_rate",
    "stage_classifier_real_macro_f1",
    "sleep_signal_mae",
    "sleep_spectral_mae",
    "sleep_stage_mismatch_rate",
    "sleep_stage_classifier_real_macro_f1",
    "relative_crps_gain_vs_uniform",
    "relative_mase_gain_vs_uniform",
    "relative_score_gain_vs_uniform",
    "realized_nfe",
    "latency_ms_per_sample",
    "num_eval_samples",
    "eval_examples",
    "eval_windows",
    "eval_horizon",
    "evaluation_protocol_hash",
    "chosen_t0s_hash",
    "chosen_examples_hash",
    "stage_counts_json",
    "schedule_grid_hash",
    "protocol_hash",
    "row_status",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: Any, *, indent: Optional[int] = None, sort_keys: bool = False) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        indent=indent,
        sort_keys=sort_keys,
        separators=(",", ":") if indent is None else None,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant {value!r} is not allowed.")


def _json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_json_constant)


def _save_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    arr = np.asarray(
        [float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64
    )
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _std(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray(
        [float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64
    )
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else None
    return float(np.std(arr, ddof=1))


def _safe_relative_gain(value: Any, baseline_value: Any) -> Optional[float]:
    v = _optional_float(value)
    b = _optional_float(baseline_value)
    if v is None or b is None or abs(float(b)) <= 1e-12:
        return None
    return float(1.0 - float(v) / float(b))


def _parse_schedule_names(text: str) -> List[str]:
    names = [name.strip().lower() for name in parse_csv(text)]
    unknown = [name for name in names if name not in SCHEDULE_KEYS]
    if unknown:
        raise ValueError(f"Unknown active diffusion-flow schedules: {unknown}")
    return names


def _file_content_hash(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), int(size)


def _directory_content_fingerprint(path: Path, *, suffix: Optional[str] = None) -> Dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    candidates = sorted(
        (
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and (suffix is None or candidate.suffix == suffix)
        ),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    )
    for candidate in candidates:
        relative = candidate.relative_to(path).as_posix()
        content_hash, size = _file_content_hash(candidate)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\0")
        file_count += 1
        total_bytes += int(size)
    return {
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "file_count": int(file_count),
        "size_bytes": int(total_bytes),
    }


def _path_fingerprint(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_project_path(str(path))
    if resolved.is_file():
        content_hash, size = _file_content_hash(resolved)
        return {"kind": "file", "sha256": content_hash, "size_bytes": int(size)}
    if resolved.is_dir():
        return _directory_content_fingerprint(resolved)
    return {"kind": "missing"}


def _source_fingerprint() -> Dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    payload = _directory_content_fingerprint(package_root, suffix=".py")
    payload["kind"] = "python_source_tree"
    payload["schedule_catalog"] = _path_fingerprint(
        package_root / "schedule_transfer" / "otflow_external_schedule_catalog.json"
    )
    return payload


def _runtime_environment_fingerprint() -> Dict[str, str]:
    try:
        scipy_version = metadata.version("scipy")
    except metadata.PackageNotFoundError:
        scipy_version = "not-installed"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "scipy": str(scipy_version),
        "torch": str(torch.__version__),
    }


def _logical_artifact_path(path: str | Path) -> str:
    resolved = resolve_project_path(str(path))
    root = project_root().resolve()
    try:
        return str(resolved.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(resolved.name)


def _data_path_fingerprints(cli_args: argparse.Namespace) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    dataset_root = resolve_project_path(str(cli_args.dataset_root))
    for dataset in parse_forecast_datasets(str(cli_args.forecast_datasets)):
        manifest = monash_manifest_path(dataset_root, str(dataset))
        source_dir = monash_source_dir(dataset_root, str(dataset))
        tsf_candidates = sorted(source_dir.rglob("*.tsf")) if source_dir.is_dir() else []
        selected[f"forecast:{dataset}:manifest"] = _path_fingerprint(manifest)
        selected[f"forecast:{dataset}:tsf"] = (
            _path_fingerprint(tsf_candidates[0]) if tsf_candidates else {"kind": "missing"}
        )

    conditional_paths = {
        "cryptos": str(cli_args.cryptos_path).strip() or cryptos_data_path(),
        "es_mbp_10": str(cli_args.es_path).strip() or es_mbp_10_data_path(),
        "sleep_edf": str(cli_args.sleep_edf_path).strip() or sleep_edf_data_path(),
    }
    for dataset in parse_conditional_generation_datasets(
        str(cli_args.conditional_generation_datasets)
    ):
        data_path = resolve_project_path(str(conditional_paths[str(dataset)]))
        selected[f"conditional_generation:{dataset}:data"] = _path_fingerprint(data_path)
        if str(dataset) == "sleep_edf":
            selected[f"conditional_generation:{dataset}:metadata"] = _path_fingerprint(
                sleep_edf_metadata_path_for_npz(data_path)
            )
    return selected


def _selected_backbone_fingerprints(cli_args: argparse.Namespace) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    forecast_datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    conditional_datasets = parse_conditional_generation_datasets(
        str(cli_args.conditional_generation_datasets)
    )
    requested = [
        *[(FORECAST_FAMILY, str(dataset)) for dataset in forecast_datasets],
        *[(CONDITIONAL_GENERATION_FAMILY, str(dataset)) for dataset in conditional_datasets],
    ]
    manifest_text = str(cli_args.backbone_manifest).strip()
    manifest_path = resolve_project_path(manifest_text) if manifest_text else None
    selected["manifest"] = _path_fingerprint(manifest_path) if manifest_path is not None else None

    if manifest_path is not None and manifest_path.is_file():
        manifest_payload = _json_loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = (
            list(manifest_payload.get("artifacts", []))
            if isinstance(manifest_payload, dict)
            else []
        )
        for benchmark_family, dataset in requested:
            matches = [
                artifact
                for artifact in artifacts
                if str(artifact.get("backbone_name", "")) == "otflow"
                and str(artifact.get("benchmark_family", "")) == benchmark_family
                and str(artifact.get("dataset_key", "")) == dataset
                and int(artifact.get("train_steps", -1)) == int(cli_args.otflow_train_steps)
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Backbone manifest contains multiple selected artifacts for {benchmark_family}/{dataset}."
                )
            key = f"{benchmark_family}:{dataset}"
            if not matches:
                selected[key] = {"status": "missing"}
                continue
            artifact = dict(matches[0])
            checkpoint_text = str(artifact.get("checkpoint_path", "")).strip()
            checkpoint_path = (
                (manifest_path.parent / checkpoint_text).resolve()
                if checkpoint_text and not Path(checkpoint_text).is_absolute()
                else Path(checkpoint_text).resolve()
                if checkpoint_text
                else manifest_path.parent / "missing-checkpoint"
            )
            metadata_text = str(artifact.get("metadata_path", "")).strip()
            metadata_path = (
                (manifest_path.parent / metadata_text).resolve()
                if metadata_text and not Path(metadata_text).is_absolute()
                else Path(metadata_text).resolve()
                if metadata_text
                else checkpoint_path.with_name("checkpoint_metadata.json")
            )
            selected[key] = {
                "status": str(artifact.get("status", "")),
                "checkpoint": _path_fingerprint(checkpoint_path),
                "metadata": _path_fingerprint(metadata_path),
            }
        return selected

    shared_root = resolve_project_path(str(cli_args.shared_backbone_root))
    for benchmark_family, dataset in requested:
        if benchmark_family == FORECAST_FAMILY:
            artifact_root = shared_root / "forecast" / dataset
        else:
            artifact_root = shared_root / "conditional_generation" / dataset / "transformer"
        selected[f"{benchmark_family}:{dataset}"] = {
            "checkpoint": _path_fingerprint(artifact_root / "model.pt"),
            "metadata": _path_fingerprint(artifact_root / "checkpoint_metadata.json"),
        }
    return selected


def _selected_input_fingerprints(cli_args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "datasets": _data_path_fingerprints(cli_args),
        "backbones": _selected_backbone_fingerprints(cli_args),
    }


def _sanitized_cli_args(cli_args: argparse.Namespace) -> Dict[str, Any]:
    path_fields = {
        "out_root",
        "dataset_root",
        "shared_backbone_root",
        "backbone_manifest",
        "cryptos_path",
        "es_path",
        "sleep_edf_path",
    }
    payload: Dict[str, Any] = {}
    for key, value in vars(cli_args).items():
        if key in path_fields:
            continue
        payload[key] = value
    return payload


def _schedule_fingerprints(
    *,
    schedules: Sequence[str],
) -> Dict[str, str]:
    schedule_root = Path(__file__).resolve().parents[1] / "schedule_transfer"
    implementation = {
        "source": _path_fingerprint(schedule_root / "diffusion_flow_schedules.py"),
        "catalog": _path_fingerprint(schedule_root / "otflow_external_schedule_catalog.json"),
    }
    return {
        str(schedule_key): hashlib.sha256(
            _json_dumps(
                {"schedule_key": str(schedule_key), **implementation}, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        for schedule_key in schedules
    }


def _protocol_config_payload(
    cli_args: argparse.Namespace,
    *,
    input_fingerprints: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    forecast_datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    conditional_datasets = parse_conditional_generation_datasets(
        str(cli_args.conditional_generation_datasets)
    )
    seeds = parse_int_csv(str(cli_args.seeds))
    target_nfes = parse_int_csv(str(cli_args.target_nfe_values))
    solvers = parse_csv(str(cli_args.solver_names))
    schedules = _parse_schedule_names(str(cli_args.schedule_names))
    selected_inputs = dict(input_fingerprints or _selected_input_fingerprints(cli_args))
    return {
        "runner_protocol": RUNNER_PROTOCOL,
        "forecast_datasets": forecast_datasets,
        "conditional_generation_datasets": conditional_datasets,
        "seeds": seeds,
        "target_nfe_values": target_nfes,
        "solver_names": solvers,
        "schedule_names": schedules,
        "otflow_train_steps": int(cli_args.otflow_train_steps),
        "dataset_seed": int(cli_args.dataset_seed),
        "num_eval_samples": int(cli_args.num_eval_samples),
        "eval_horizon": int(cli_args.eval_horizon),
        "eval_windows_val": int(cli_args.eval_windows_val),
        "eval_windows_test": int(cli_args.eval_windows_test),
        "execution_config": {
            "device": str(cli_args.device),
            "steps": int(cli_args.steps),
            "hidden_dim": int(cli_args.hidden_dim),
            "fu_net_layers": int(cli_args.fu_net_layers),
            "fu_net_heads": int(cli_args.fu_net_heads),
            "rollout_mode": str(cli_args.rollout_mode),
            "future_block_len": int(cli_args.future_block_len),
            "lr": float(cli_args.lr),
            "weight_decay": float(cli_args.weight_decay),
            "grad_clip": float(cli_args.grad_clip),
            "runtime_environment": _runtime_environment_fingerprint(),
        },
        "input_fingerprints": selected_inputs,
        "source_fingerprint": _source_fingerprint(),
        "schedule_fingerprints": _schedule_fingerprints(
            schedules=schedules,
        ),
    }


def _protocol_config_fingerprint(cli_args: argparse.Namespace) -> str:
    encoded = _json_dumps(_protocol_config_payload(cli_args), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _realized_nfe_for_solver(solver_key: str, runtime_nfe: int) -> int:
    return int(runtime_nfe) * int(solver_eval_multiplier(str(solver_key)))


def _row_signature(
    *,
    dataset: str,
    split_phase: str,
    seed: int,
    target_nfe: int,
    solver_key: str,
    schedule_key: str,
    checkpoint_id: str,
) -> str:
    return "|".join(
        [
            str(dataset),
            str(split_phase),
            str(seed),
            str(target_nfe),
            str(solver_key),
            str(schedule_key),
            str(checkpoint_id),
        ]
    )


def _row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("protocol_hash"),
        row.get("benchmark_family"),
        row.get("split_phase"),
        int(row.get("seed", -1)),
        row.get("dataset"),
        int(row.get("target_nfe", -1)),
        row.get("solver_key"),
        row.get("schedule_key"),
        row.get("row_signature"),
    )


def _write_row_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_RECORD_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_RECORD_FIELDS})


def _load_rows(jsonl_path: Path, *, protocol_hash: str) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return rows
    lines = jsonl_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = _json_loads(line)
        except json.JSONDecodeError as exc:
            is_truncated_final_line = line_index == len(lines) - 1 and not raw_line.endswith(
                ("\n", "\r")
            )
            if is_truncated_final_line:
                break
            raise ValueError(f"Invalid JSONL record at {jsonl_path}:{line_index + 1}.") from exc
        except ValueError as exc:
            raise ValueError(f"Invalid JSONL record at {jsonl_path}:{line_index + 1}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL record at {jsonl_path}:{line_index + 1} must be an object.")
        if str(row.get("protocol_hash", "")) != str(protocol_hash):
            continue
        key = _row_key(row)
        if key in rows:
            raise ValueError(
                f"Duplicate resumable row at {jsonl_path}:{line_index + 1} for key {key}."
            )
        rows[key] = row
    return rows


def _init_row_recorder(out_root: Path, cli_args: argparse.Namespace) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / str(getattr(cli_args, "row_jsonl_name", "rows.jsonl"))
    csv_path = out_root / str(getattr(cli_args, "row_csv_name", "rows.csv"))
    input_fingerprints = _selected_input_fingerprints(cli_args)
    protocol_payload = _protocol_config_payload(cli_args, input_fingerprints=input_fingerprints)
    protocol_hash = hashlib.sha256(
        _json_dumps(protocol_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_config_path = out_root / "run_config.json"
    previous_config = (
        _json_loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    )
    can_resume = (
        bool(getattr(cli_args, "resume", True))
        and str(previous_config.get("protocol_hash", "")) == protocol_hash
    )
    rows_by_key = _load_rows(jsonl_path, protocol_hash=str(protocol_hash)) if can_resume else {}
    fh = jsonl_path.open("a" if can_resume else "w", encoding="utf-8")
    _save_json(
        {
            "runner_protocol": RUNNER_PROTOCOL,
            "method_key": RUNNER_PROTOCOL,
            "protocol_hash": protocol_hash,
            "args": _sanitized_cli_args(cli_args),
            "protocol_inputs": protocol_payload,
        },
        run_config_path,
    )
    if rows_by_key:
        _write_row_csv(csv_path, list(rows_by_key.values()))
    return {
        "out_root": out_root,
        "jsonl_path": jsonl_path,
        "csv_path": csv_path,
        "fh": fh,
        "rows_by_key": rows_by_key,
        "protocol_hash": protocol_hash,
    }


def _append_row_record(row_recorder: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    row_dict = dict(row)
    key = _row_key(row_dict)
    if key in row_recorder["rows_by_key"]:
        raise ValueError(f"Refusing to append duplicate row for key {key}.")
    serialized = _json_dumps(row_dict, sort_keys=True)
    row_recorder["rows_by_key"][key] = row_dict
    row_recorder["fh"].write(serialized + "\n")
    row_recorder["fh"].flush()
    _write_row_csv(Path(row_recorder["csv_path"]), list(row_recorder["rows_by_key"].values()))


def _existing_complete_row(
    row_recorder: Mapping[str, Any], row_key: Tuple[Any, ...]
) -> Optional[Dict[str, Any]]:
    row = row_recorder["rows_by_key"].get(row_key)
    if row is not None and str(row.get("row_status")) == "complete":
        return dict(row)
    return None


def _pending_schedule_cases(
    row_recorder: Mapping[str, Any],
    *,
    benchmark_family: str,
    split_phase: str,
    seed: int,
    dataset: str,
    checkpoint_id: str,
    target_nfe: int,
    solver_key: str,
    schedule_cases: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    existing: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for case in schedule_cases:
        schedule_key = str(case["schedule_key"])
        signature = _row_signature(
            dataset=dataset,
            split_phase=split_phase,
            seed=seed,
            target_nfe=target_nfe,
            solver_key=solver_key,
            schedule_key=schedule_key,
            checkpoint_id=checkpoint_id,
        )
        key = (
            row_recorder["protocol_hash"],
            benchmark_family,
            split_phase,
            int(seed),
            dataset,
            int(target_nfe),
            solver_key,
            schedule_key,
            signature,
        )
        row = _existing_complete_row(row_recorder, key)
        if row is None:
            pending.append(dict(case, row_signature=signature))
        else:
            existing.append(row)
    return existing, pending


def _fixed_schedule_details(schedule_key: str, runtime_nfe: int) -> Dict[str, Any]:
    fixed_grid = build_schedule_grid(str(schedule_key), int(runtime_nfe))
    schedule_grid_hash = hashlib.sha256(
        _json_dumps([float(x) for x in fixed_grid]).encode("utf-8")
    ).hexdigest()
    details: Dict[str, Any] = {
        "time_grid": [float(x) for x in fixed_grid],
        "schedule_grid_hash": str(schedule_grid_hash),
        "reference_time_alignment": schedule_time_alignment(str(schedule_key)),
        "reference_macro_steps": int(runtime_nfe),
    }
    details.update(fixed_schedule_shape_statistics(fixed_grid))
    return details


def _evaluation_protocol_fields(
    result_row: Mapping[str, Any], *, eval_horizon: int
) -> Dict[str, Any]:
    protocol = dict(result_row.get("evaluation_protocol", {}) or {})
    encoded = _json_dumps(protocol, sort_keys=True)
    return {
        "eval_horizon": int(eval_horizon),
        "evaluation_protocol_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "chosen_t0s_hash": str(protocol.get("chosen_t0s_hash", "")),
        "stage_counts_json": _json_dumps(
            dict(protocol.get("stage_counts", {}) or {}), sort_keys=True
        ),
    }


def _build_row(
    *,
    benchmark_family: str,
    split_phase: str,
    seed: int,
    dataset: str,
    checkpoint: Mapping[str, Any],
    target_nfe: int,
    runtime_nfe: int,
    solver_key: str,
    schedule_key: str,
    details: Mapping[str, Any],
    metrics: Mapping[str, Any],
    row_signature: str,
    protocol_hash: str,
) -> Dict[str, Any]:
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
        "checkpoint_path": _logical_artifact_path(str(checkpoint["checkpoint_path"])),
        "backbone_name": str(checkpoint.get("backbone_name", "otflow")),
        "train_steps": int(checkpoint["train_steps"]),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "solver_key": str(solver_key),
        "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
        "schedule_key": str(schedule_key),
        "schedule_name": schedule_display_name(str(schedule_key)),
        "row_signature": str(row_signature),
        "experiment_scope": solver_experiment_scope(str(solver_key)),
        "selection_metric": str(selection_metric),
        "selection_metric_value": metrics.get(selection_metric),
        "reference_macro_steps": int(details.get("reference_macro_steps", runtime_nfe)),
        "reference_time_alignment": str(
            details.get("reference_time_alignment", schedule_time_alignment(str(schedule_key)))
        ),
        "runtime_grid_q25": details.get("runtime_grid_q25"),
        "runtime_grid_q50": details.get("runtime_grid_q50"),
        "runtime_grid_q75": details.get("runtime_grid_q75"),
        "crps": metrics.get("crps"),
        "mse": metrics.get("mse"),
        "mase": metrics.get("mase"),
        "score_main": metrics.get("score_main"),
        "disc_auc": metrics.get("disc_auc"),
        "disc_auc_gap": metrics.get("disc_auc_gap"),
        "unconditional_w1": metrics.get("unconditional_w1"),
        "conditional_w1": metrics.get("conditional_w1"),
        "tstr_macro_f1": metrics.get("tstr_macro_f1"),
        "u_l1": metrics.get("u_l1"),
        "c_l1": metrics.get("c_l1"),
        "spread_specific_error": metrics.get("spread_specific_error"),
        "imbalance_specific_error": metrics.get("imbalance_specific_error"),
        "ret_vol_acf_error": metrics.get("ret_vol_acf_error"),
        "impact_response_error": metrics.get("impact_response_error"),
        "stage_mismatch_rate": metrics.get("stage_mismatch_rate"),
        "stage_classifier_real_macro_f1": metrics.get("stage_classifier_real_macro_f1"),
        "sleep_signal_mae": metrics.get("sleep_signal_mae"),
        "sleep_spectral_mae": metrics.get("sleep_spectral_mae"),
        "sleep_stage_mismatch_rate": metrics.get("sleep_stage_mismatch_rate"),
        "sleep_stage_classifier_real_macro_f1": metrics.get("sleep_stage_classifier_real_macro_f1"),
        "relative_crps_gain_vs_uniform": metrics.get("relative_crps_gain_vs_uniform"),
        "relative_mase_gain_vs_uniform": metrics.get("relative_mase_gain_vs_uniform"),
        "relative_score_gain_vs_uniform": metrics.get("relative_score_gain_vs_uniform"),
        "realized_nfe": int(realized_nfe),
        "latency_ms_per_sample": metrics.get(
            "latency_ms_per_sample", metrics.get("efficiency_ms_per_sample")
        ),
        "num_eval_samples": metrics.get("num_eval_samples"),
        "eval_examples": metrics.get("eval_examples"),
        "eval_windows": metrics.get("eval_windows"),
        "eval_horizon": metrics.get("eval_horizon"),
        "evaluation_protocol_hash": metrics.get("evaluation_protocol_hash"),
        "chosen_t0s_hash": metrics.get("chosen_t0s_hash"),
        "chosen_examples_hash": metrics.get("chosen_examples_hash"),
        "stage_counts_json": metrics.get("stage_counts_json"),
        "schedule_grid_hash": details.get("schedule_grid_hash"),
        "protocol_hash": str(protocol_hash),
        "row_status": "complete",
    }


def _schedule_cases_for_datasets(
    cli_args: argparse.Namespace, datasets: Iterable[str]
) -> Dict[str, List[Dict[str, Any]]]:
    schedule_names = _parse_schedule_names(str(cli_args.schedule_names))
    if UNIFORM_SCHEDULE_KEY in schedule_names:
        schedule_names = [UNIFORM_SCHEDULE_KEY] + [
            key for key in schedule_names if key != UNIFORM_SCHEDULE_KEY
        ]
    return {str(dataset): [{"schedule_key": key} for key in schedule_names] for dataset in datasets}


def _run_forecast_phase(
    cli_args: argparse.Namespace,
    *,
    row_recorder: Mapping[str, Any],
    split_phase: str,
    seeds: Sequence[int],
    schedule_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    dataset_root = resolve_project_path(str(cli_args.dataset_root))
    shared_backbone_root = resolve_project_path(str(cli_args.shared_backbone_root))
    device = torch.device(str(cli_args.device))
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    for dataset_idx, dataset in enumerate(datasets):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = load_forecast_checkpoint_splits(
                cli_args=cli_args,
                dataset_root=dataset_root,
                shared_backbone_root=shared_backbone_root,
                dataset=dataset,
                device=device,
            )
        checkpoint = dataset_cache[dataset]
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        eval_ds = splits["val"] if str(split_phase) == VALIDATION_PHASE else splits["test"]
        eval_window_count = int(
            cli_args.eval_windows_val
            if str(split_phase) == VALIDATION_PHASE
            else cli_args.eval_windows_test
        )
        for seed in seeds:
            chosen_examples = choose_forecast_example_indices(
                eval_ds,
                n_examples=int(eval_window_count),
                seed=int(seed) + 1_000 * dataset_idx,
            )
            for target_idx, target_nfe in enumerate(parse_int_csv(str(cli_args.target_nfe_values))):
                for solver_idx, solver_key in enumerate(parse_csv(str(cli_args.solver_names))):
                    runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                    schedule_cases = list(schedule_cases_by_dataset[str(dataset)])
                    existing_rows, pending_cases = _pending_schedule_cases(
                        row_recorder,
                        benchmark_family=FORECAST_FAMILY,
                        split_phase=str(split_phase),
                        seed=int(seed),
                        dataset=str(dataset),
                        checkpoint_id=str(checkpoint["checkpoint_id"]),
                        target_nfe=int(target_nfe),
                        solver_key=str(solver_key),
                        schedule_cases=schedule_cases,
                    )
                    rows.extend(existing_rows)
                    cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                    for existing_row in existing_rows:
                        if str(existing_row.get("schedule_key")) == UNIFORM_SCHEDULE_KEY:
                            cell_uniform_metrics = existing_row
                    for case in pending_cases:
                        schedule_key = str(case["schedule_key"])
                        details = _fixed_schedule_details(schedule_key, int(runtime_nfe))
                        eval_seed = (
                            int(seed) + 100_000 * dataset_idx + 1_000 * target_idx + solver_idx
                        )
                        metrics = evaluate_forecast_schedule(
                            model,
                            eval_ds,
                            cfg,
                            solver_name=str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
                            runtime_nfe=int(runtime_nfe),
                            time_grid=details["time_grid"],
                            num_eval_samples=int(cli_args.num_eval_samples),
                            seed=int(eval_seed),
                            example_indices=chosen_examples,
                        )
                        if (
                            schedule_key != UNIFORM_SCHEDULE_KEY
                            and cell_uniform_metrics is not None
                        ):
                            metrics = dict(metrics)
                            metrics["relative_crps_gain_vs_uniform"] = _safe_relative_gain(
                                metrics.get("crps"), cell_uniform_metrics.get("crps")
                            )
                            metrics["relative_mase_gain_vs_uniform"] = _safe_relative_gain(
                                metrics.get("mase"), cell_uniform_metrics.get("mase")
                            )
                        row = _build_row(
                            benchmark_family=FORECAST_FAMILY,
                            split_phase=str(split_phase),
                            seed=int(seed),
                            dataset=str(dataset),
                            checkpoint=checkpoint,
                            target_nfe=int(target_nfe),
                            runtime_nfe=int(runtime_nfe),
                            solver_key=str(solver_key),
                            schedule_key=schedule_key,
                            details=details,
                            metrics=metrics,
                            row_signature=str(case["row_signature"]),
                            protocol_hash=str(row_recorder["protocol_hash"]),
                        )
                        _append_row_record(row_recorder, row)
                        rows.append(row)
                        if schedule_key == UNIFORM_SCHEDULE_KEY:
                            cell_uniform_metrics = row
    return rows


def _run_conditional_generation_phase(
    cli_args: argparse.Namespace,
    *,
    row_recorder: Mapping[str, Any],
    split_phase: str,
    seeds: Sequence[int],
    schedule_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    shared_backbone_root = resolve_project_path(str(cli_args.shared_backbone_root))
    device = torch.device(str(cli_args.device))
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    datasets = parse_conditional_generation_datasets(str(cli_args.conditional_generation_datasets))
    for dataset_idx, dataset in enumerate(datasets):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = load_conditional_generation_checkpoint_splits(
                cli_args=cli_args,
                shared_backbone_root=shared_backbone_root,
                dataset=dataset,
                device=device,
            )
        checkpoint = dataset_cache[dataset]
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        eval_ds = splits["val"] if str(split_phase) == VALIDATION_PHASE else splits["test"]
        eval_horizon = resolved_eval_horizon(cli_args, str(dataset))
        eval_windows = resolved_eval_windows(
            cli_args, str(dataset), "val" if str(split_phase) == VALIDATION_PHASE else "test"
        )
        for seed in seeds:
            chosen_eval_t0s = np.asarray(
                _choose_valid_windows(
                    eval_ds,
                    horizon=int(eval_horizon),
                    n_windows=int(eval_windows),
                    seed=int(seed) + 1_000 * dataset_idx,
                ),
                dtype=np.int64,
            )
            for target_idx, target_nfe in enumerate(parse_int_csv(str(cli_args.target_nfe_values))):
                for solver_idx, solver_key in enumerate(parse_csv(str(cli_args.solver_names))):
                    runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                    existing_rows, pending_cases = _pending_schedule_cases(
                        row_recorder,
                        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                        split_phase=str(split_phase),
                        seed=int(seed),
                        dataset=str(dataset),
                        checkpoint_id=str(checkpoint["checkpoint_id"]),
                        target_nfe=int(target_nfe),
                        solver_key=str(solver_key),
                        schedule_cases=list(schedule_cases_by_dataset[str(dataset)]),
                    )
                    rows.extend(existing_rows)
                    cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                    for existing_row in existing_rows:
                        if str(existing_row.get("schedule_key")) == UNIFORM_SCHEDULE_KEY:
                            cell_uniform_metrics = existing_row
                    for case in pending_cases:
                        schedule_key = str(case["schedule_key"])
                        details = _fixed_schedule_details(schedule_key, int(runtime_nfe))
                        grid_spec = {
                            "grid_name": schedule_key,
                            "grid_kind": "fixed_diffusion_flow_time_grid",
                            "selection_group": schedule_key,
                            "comparison_role": "transferred"
                            if schedule_key in TRANSFER_SCHEDULE_KEYS
                            else "baseline",
                            "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
                            "nfe": int(runtime_nfe),
                            "time_grid": details["time_grid"],
                        }
                        metrics_seed = (
                            int(seed) + 1_000_000 * dataset_idx + 10_000 * target_idx + solver_idx
                        )
                        result_row = run_fixed_schedule_variant(
                            model=model,
                            ds=eval_ds,
                            cfg=cfg,
                            eval_horizon=int(eval_horizon),
                            eval_windows=int(len(chosen_eval_t0s)),
                            grid_spec=grid_spec,
                            chosen_t0s=chosen_eval_t0s,
                            generation_seed_base=int(metrics_seed),
                            metrics_seed=int(metrics_seed),
                            score_main_only=False,
                        )
                        metrics = {
                            "score_main": result_row.get("score_main"),
                            "tstr_macro_f1": result_row.get("tstr_macro_f1"),
                            "disc_auc": result_row.get("disc_auc"),
                            "disc_auc_gap": result_row.get("disc_auc_gap"),
                            "unconditional_w1": result_row.get("unconditional_w1"),
                            "conditional_w1": result_row.get("conditional_w1"),
                            "u_l1": result_row.get("u_l1"),
                            "c_l1": result_row.get("c_l1"),
                            "spread_specific_error": result_row.get("spread_specific_error"),
                            "imbalance_specific_error": result_row.get("imbalance_specific_error"),
                            "ret_vol_acf_error": result_row.get("ret_vol_acf_error"),
                            "impact_response_error": result_row.get("impact_response_error"),
                            "stage_mismatch_rate": result_row.get("stage_mismatch_rate"),
                            "stage_classifier_real_macro_f1": result_row.get(
                                "stage_classifier_real_macro_f1"
                            ),
                            "sleep_signal_mae": result_row.get("spread_specific_error")
                            if str(dataset) == "sleep_edf"
                            else None,
                            "sleep_spectral_mae": result_row.get("imbalance_specific_error")
                            if str(dataset) == "sleep_edf"
                            else None,
                            "sleep_stage_mismatch_rate": result_row.get("stage_mismatch_rate")
                            if str(dataset) == "sleep_edf"
                            else None,
                            "sleep_stage_classifier_real_macro_f1": result_row.get(
                                "stage_classifier_real_macro_f1"
                            )
                            if str(dataset) == "sleep_edf"
                            else None,
                            "efficiency_ms_per_sample": result_row.get("efficiency_ms_per_sample"),
                            "eval_windows": int(len(chosen_eval_t0s)),
                            "realized_nfe": _realized_nfe_for_solver(
                                str(solver_key), int(runtime_nfe)
                            ),
                            **_evaluation_protocol_fields(
                                result_row, eval_horizon=int(eval_horizon)
                            ),
                        }
                        if (
                            schedule_key != UNIFORM_SCHEDULE_KEY
                            and cell_uniform_metrics is not None
                        ):
                            metrics["relative_score_gain_vs_uniform"] = _safe_relative_gain(
                                metrics.get("score_main"), cell_uniform_metrics.get("score_main")
                            )
                        row = _build_row(
                            benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                            split_phase=str(split_phase),
                            seed=int(seed),
                            dataset=str(dataset),
                            checkpoint=checkpoint,
                            target_nfe=int(target_nfe),
                            runtime_nfe=int(runtime_nfe),
                            solver_key=str(solver_key),
                            schedule_key=schedule_key,
                            details=details,
                            metrics=metrics,
                            row_signature=str(case["row_signature"]),
                            protocol_hash=str(row_recorder["protocol_hash"]),
                        )
                        _append_row_record(row_recorder, row)
                        rows.append(row)
                        if schedule_key == UNIFORM_SCHEDULE_KEY:
                            cell_uniform_metrics = row
    return rows


def _candidate_rows_by_phase(
    rows: Sequence[Mapping[str, Any]],
    split_phase: str,
    solver_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
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
        key = (
            row.get("benchmark_family"),
            row.get("dataset"),
            row.get("target_nfe"),
            row.get("solver_key"),
            row.get("schedule_key"),
            row.get("train_budget_label"),
        )
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    metric_names = (
        "crps",
        "mse",
        "mase",
        "score_main",
        "tstr_macro_f1",
        "disc_auc",
        "disc_auc_gap",
        "unconditional_w1",
        "conditional_w1",
        "u_l1",
        "c_l1",
        "spread_specific_error",
        "imbalance_specific_error",
        "ret_vol_acf_error",
        "impact_response_error",
        "stage_mismatch_rate",
        "stage_classifier_real_macro_f1",
        "sleep_signal_mae",
        "sleep_spectral_mae",
        "sleep_stage_mismatch_rate",
        "sleep_stage_classifier_real_macro_f1",
        "relative_crps_gain_vs_uniform",
        "relative_mase_gain_vs_uniform",
        "relative_score_gain_vs_uniform",
        "realized_nfe",
        "latency_ms_per_sample",
    )
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        family, dataset, target_nfe, solver_key, schedule_key, budget = key
        summary: Dict[str, Any] = {
            "benchmark_family": family,
            "dataset": dataset,
            "target_nfe": int(target_nfe),
            "solver_key": solver_key,
            "schedule_key": schedule_key,
            "schedule_name": schedule_display_name(str(schedule_key)),
            "train_budget_label": budget,
            "n_seeds": int(len(group)),
            "seed_values": sorted(int(row.get("seed", 0)) for row in group),
        }
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
    return {
        "method_key": RUNNER_PROTOCOL,
        "row_count": int(len(rows)),
        "summary_row_count": int(len(augmented)),
        "schedule_keys": sorted({str(row.get("schedule_key")) for row in rows}),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "seed_summaries": augmented,
    }


def _prep_summary(cli_args: argparse.Namespace) -> Dict[str, Any]:
    schedules = _parse_schedule_names(str(cli_args.schedule_names))
    solvers = parse_csv(str(cli_args.solver_names))
    nfes = parse_int_csv(str(cli_args.target_nfe_values))
    manifest_path = (
        resolve_project_path(str(cli_args.backbone_manifest))
        if str(cli_args.backbone_manifest).strip()
        else None
    )
    manifest_summary: Dict[str, Any] = {"path": None, "ready_count": None, "missing_count": None}
    if manifest_path is not None:
        resolved = manifest_path
        manifest_summary["path"] = _logical_artifact_path(resolved)
        if resolved.exists():
            payload = _json_loads(resolved.read_text(encoding="utf-8"))
            manifest_summary["ready_count"] = int(payload.get("ready_count", 0))
            manifest_summary["missing_count"] = int(payload.get("missing_count", 0))
    return {
        "runner_mode": "diffusion_flow_time_reparameterization",
        "runner_protocol": RUNNER_PROTOCOL,
        "method_key": RUNNER_PROTOCOL,
        "schedule_keys": list(SCHEDULE_KEYS),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "scheduled_evaluation_keys": schedules,
        "solver_names": solvers,
        "target_nfe_values": nfes,
        "forecast_datasets": parse_forecast_datasets(str(cli_args.forecast_datasets)),
        "conditional_generation_datasets": parse_conditional_generation_datasets(
            str(cli_args.conditional_generation_datasets)
        ),
        "backbone_manifest": manifest_summary,
        "allow_execute": bool(getattr(cli_args, "allow_execute", False)),
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run diffusion-flow time reparameterization fixed-schedule evaluations."
    )
    ap.add_argument("--out_root", type=str, default=str(RUNNER_OUTPUT_ROOT))
    ap.add_argument("--dataset_root", type=str, default=str(project_paper_dataset_root()))
    ap.add_argument("--shared_backbone_root", type=str, default=str(SHARED_BACKBONE_ROOT))
    ap.add_argument("--backbone_manifest", type=str, default=str(backbone_manifest_path()))
    ap.add_argument("--otflow_train_steps", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--forecast_datasets", type=str, default=",".join(PAPER_FORECAST_DATASETS))
    ap.add_argument(
        "--conditional_generation_datasets",
        type=str,
        default=",".join(PAPER_CONDITIONAL_GENERATION_DATASETS),
    )
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--sleep_edf_path", type=str, default="")
    ap.add_argument("--solver_names", type=str, default=",".join(ALL_SOLVER_ORDER))
    ap.add_argument(
        "--target_nfe_values", type=str, default=",".join(str(x) for x in RUNNER_TARGET_NFE_VALUES)
    )
    ap.add_argument(
        "--schedule-names", dest="schedule_names", type=str, default=",".join(RUNNER_SCHEDULE_KEYS)
    )
    ap.add_argument("--seeds", type=str, default=",".join(str(x) for x in RUNNER_SEEDS))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--num_eval_samples", type=int, default=5)
    ap.add_argument("--eval_horizon", type=int, default=0)
    ap.add_argument("--eval_windows_val", type=int, default=0)
    ap.add_argument("--eval_windows_test", type=int, default=0)
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
    out_root = resolve_project_path(str(cli_args.out_root))
    out_root.mkdir(parents=True, exist_ok=True)
    prep_payload = _prep_summary(cli_args)
    if bool(getattr(cli_args, "diagnose_locked_forecast_only", False)):
        rows = list(
            _load_rows(
                out_root / str(getattr(cli_args, "row_jsonl_name", "rows.jsonl")),
                protocol_hash=_protocol_config_fingerprint(cli_args),
            ).values()
        )
        selected_seeds = set(parse_int_csv(str(cli_args.seeds)))
        locked = [
            row
            for row in _candidate_rows_by_phase(rows, LOCKED_TEST_PHASE)
            if str(row.get("benchmark_family", "")) == FORECAST_FAMILY
            and int(row.get("seed", -1)) in selected_seeds
        ]
        payload = {
            "runner_mode": "diagnose_locked_forecast_only",
            "row_count": int(len(rows)),
            "locked_row_count": int(len(locked)),
            "main_table_summary": _aggregate_main_table(locked),
        }
        _save_json(dict(payload), out_root / "combined_summary.json")
        return payload
    if not bool(cli_args.allow_execute):
        _save_json(dict(prep_payload), out_root / "combined_summary.json")
        return dict(prep_payload)

    validate_execution_preflight(cli_args)
    row_recorder = _init_row_recorder(out_root, cli_args)
    locked_seeds = parse_int_csv(str(cli_args.seeds))
    forecast_datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    conditional_generation_datasets = parse_conditional_generation_datasets(
        str(cli_args.conditional_generation_datasets)
    )
    schedule_cases = _schedule_cases_for_datasets(
        cli_args,
        list(forecast_datasets) + list(conditional_generation_datasets),
    )
    try:
        _run_forecast_phase(
            cli_args,
            row_recorder=row_recorder,
            split_phase=LOCKED_TEST_PHASE,
            seeds=locked_seeds,
            schedule_cases_by_dataset={
                dataset: schedule_cases[dataset] for dataset in forecast_datasets
            },
        )
        _run_conditional_generation_phase(
            cli_args,
            row_recorder=row_recorder,
            split_phase=LOCKED_TEST_PHASE,
            seeds=locked_seeds,
            schedule_cases_by_dataset={
                dataset: schedule_cases[dataset] for dataset in conditional_generation_datasets
            },
        )
    finally:
        row_recorder["fh"].close()

    locked_seed_set = set(int(seed) for seed in locked_seeds)
    locked_rows = [
        row
        for row in _candidate_rows_by_phase(
            list(row_recorder["rows_by_key"].values()), LOCKED_TEST_PHASE
        )
        if int(row.get("seed", -1)) in locked_seed_set
    ]
    main_table_summary = _aggregate_main_table(locked_rows)
    seed_summaries = main_table_summary.pop("seed_summaries")
    _save_json({"seed_summaries": seed_summaries}, out_root / "locked_test_seed_summary.json")
    _save_json(dict(main_table_summary), out_root / "main_table_summary.json")
    schedule_selection = {
        "method_key": RUNNER_PROTOCOL,
        "schedule_keys": list(SCHEDULE_KEYS),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "scheduled_evaluation_keys": _parse_schedule_names(str(cli_args.schedule_names)),
    }
    _save_json(dict(schedule_selection), out_root / "schedule_selection_summary.json")
    combined = {
        "prep": dict(prep_payload),
        "schedule_selection_summary": dict(schedule_selection),
        "locked_test_seed_summary": {"seed_summaries": seed_summaries},
        "main_table_summary": dict(main_table_summary),
    }
    _save_json(dict(combined), out_root / "combined_summary.json")
    return combined


def main() -> None:
    run_diffusion_flow_time_reparameterization(build_argparser().parse_args())


if __name__ == "__main__":
    main()
