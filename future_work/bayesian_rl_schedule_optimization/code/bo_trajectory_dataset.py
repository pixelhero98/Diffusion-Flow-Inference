from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from residual_parameterization import normalize_kl_to_reference
from reward import DEFAULT_LAMBDA_BAD, forecast_log_ratio_reward
from schedule_param import build_default_basis_for_reference, theta_to_checked_schedule


DEFAULT_ELITE_FRACTION = 0.20
DEFAULT_W_MAX = 10.0
DEFAULT_STD_CLIP = (0.03, 0.20)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _valid_bo_rows(observations_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in observations_payload.get("observations", []):
        if row.get("theta") is None:
            continue
        required = ("crps", "mase", "uniform_crps", "uniform_mase", "kl_to_reference")
        if any(_finite_float(row.get(key)) is None for key in required):
            continue
        rows.append(dict(row))
    if not rows:
        raise ValueError("BO observations do not contain any usable theta/metric rows.")
    return rows


def _base_reward(row: Mapping[str, Any], *, lambda_bad: float) -> Dict[str, float]:
    kl_to_reference = normalize_kl_to_reference(row.get("kl_to_reference", 0.0))
    return forecast_log_ratio_reward(
        crps=row["crps"],
        mase=row["mase"],
        uniform_crps=row["uniform_crps"],
        uniform_mase=row["uniform_mase"],
        kl_to_reference=kl_to_reference,
        beta_ref=0.0,
        lambda_bad=lambda_bad,
    )


def select_bo_elites(
    observations_payload: Mapping[str, Any],
    *,
    elite_fraction: float = DEFAULT_ELITE_FRACTION,
    lambda_bad: float = DEFAULT_LAMBDA_BAD,
) -> List[Dict[str, Any]]:
    rows = []
    for row in _valid_bo_rows(observations_payload):
        reward_payload = _base_reward(row, lambda_bad=lambda_bad)
        item = dict(row)
        item.update({f"base_{key}": value for key, value in reward_payload.items()})
        rows.append(item)
    rows.sort(key=lambda item: float(item["base_reward"]), reverse=True)
    n_elite = max(1, int(math.ceil(float(elite_fraction) * len(rows))))
    return rows[:n_elite]


def _reward_iqr(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    return float(q75 - q25)


def _advantage_weights(rewards: Sequence[float], *, w_max: float = DEFAULT_W_MAX) -> np.ndarray:
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute weights from an empty reward list.")
    tau = float(np.std(arr))
    if tau <= 1e-12:
        return np.full(arr.size, 1.0 / float(arr.size), dtype=np.float64)
    advantages = arr - float(np.median(arr))
    weights = np.clip(np.exp(advantages / tau), 0.0, float(w_max))
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(arr.size, 1.0 / float(arr.size), dtype=np.float64)
    return weights / total


def calibrate_from_bo_elites(
    reference: Mapping[str, Any],
    observations_payload: Mapping[str, Any],
    *,
    elite_fraction: float = DEFAULT_ELITE_FRACTION,
    lambda_bad: float = DEFAULT_LAMBDA_BAD,
    w_max: float = DEFAULT_W_MAX,
    std_min: float = DEFAULT_STD_CLIP[0],
    std_max: float = DEFAULT_STD_CLIP[1],
) -> Dict[str, Any]:
    q_ref = np.asarray(reference["q_ref"], dtype=np.float64)
    basis = build_default_basis_for_reference(q_ref)
    elites = select_bo_elites(observations_payload, elite_fraction=elite_fraction, lambda_bad=lambda_bad)
    base_rewards = [float(row["base_reward"]) for row in elites]
    elite_kls = [normalize_kl_to_reference(row.get("kl_to_reference", 0.0)) for row in elites]
    median_elite_kl = float(np.median(np.asarray(elite_kls, dtype=np.float64))) if elite_kls else 0.0
    reward_iqr = _reward_iqr(base_rewards)
    if reward_iqr <= 1e-12 or median_elite_kl <= 1e-12:
        beta_ref = 0.05
        beta_ref_source = "minimum_beta_ref"
    else:
        beta_ref = float(np.clip(0.25 * reward_iqr / max(median_elite_kl, 1e-6), 0.01, 0.20))
        beta_ref_source = "elite_reward_iqr_over_median_kl"
    hard_kl_cap = float(max(2.0 * median_elite_kl, 0.10))

    enriched_elites: List[Dict[str, Any]] = []
    final_rewards: List[float] = []
    for row in elites:
        reward_payload = forecast_log_ratio_reward(
            crps=row["crps"],
            mase=row["mase"],
            uniform_crps=row["uniform_crps"],
            uniform_mase=row["uniform_mase"],
            kl_to_reference=row.get("kl_to_reference", 0.0),
            beta_ref=beta_ref,
            lambda_bad=lambda_bad,
        )
        item = dict(row)
        item.update(reward_payload)
        enriched_elites.append(item)
        final_rewards.append(float(reward_payload["reward"]))

    weights = _advantage_weights(final_rewards, w_max=w_max)
    theta = np.asarray([row["theta"] for row in enriched_elites], dtype=np.float64)
    mu = np.sum(weights[:, None] * theta, axis=0)
    centered = theta - mu[None, :]
    variance = np.sum(weights[:, None] * centered * centered, axis=0)
    std = np.clip(np.sqrt(np.maximum(variance, 0.0)), float(std_min), float(std_max))
    try:
        theta_to_checked_schedule(q_ref, mu, basis=basis, hard_kl_cap=hard_kl_cap)
    except ValueError:
        best = max(enriched_elites, key=lambda row: float(row["reward"]))
        mu = np.asarray(best["theta"], dtype=np.float64)
    return {
        "artifact": "joint_progression_ppo_bo_elite_calibration_v1",
        "dataset": reference.get("dataset"),
        "solver_key": reference.get("solver_key"),
        "target_nfe": reference.get("target_nfe"),
        "runtime_nfe": reference.get("runtime_nfe"),
        "elite_fraction": float(elite_fraction),
        "n_observations": int(len(_valid_bo_rows(observations_payload))),
        "n_elites": int(len(enriched_elites)),
        "lambda_bad": float(lambda_bad),
        "reward_iqr": float(reward_iqr),
        "median_elite_kl": float(median_elite_kl),
        "beta_ref": float(beta_ref),
        "beta_ref_source": beta_ref_source,
        "hard_kl_cap": hard_kl_cap,
        "policy_mu": [float(x) for x in mu.tolist()],
        "policy_std": [float(x) for x in std.tolist()],
        "elite_candidate_ids": [str(row.get("candidate_id", row.get("observation_id", ""))) for row in enriched_elites],
        "elite_rewards": [float(row["reward"]) for row in enriched_elites],
        "elite_weights": [float(x) for x in weights.tolist()],
    }


def load_bo_artifacts(solver_dir: str | Path) -> Dict[str, Any]:
    path = Path(solver_dir)
    required = {
        "reference": path / "reference_schedule.json",
        "observations": path / "observations.json",
        "best": path / "best_schedule.json",
        "validation_split": path / "validation_split.json",
    }
    missing = [str(item) for item in required.values() if not item.exists()]
    if missing:
        raise ValueError(f"Missing BO artifacts: {missing}")
    return {key: load_json(value) for key, value in required.items()} | {"solver_dir": path}


def find_matching_bo_solver_dir(
    *,
    workspace_root: str | Path,
    dataset: str,
    target_nfe: int,
    solver_key: str,
    cache_roots: Sequence[str | Path] = (),
    checkpoint_id: Optional[str] = None,
    train_steps: Optional[int] = None,
    bo_budget: Optional[int] = None,
    calibration_fraction: Optional[float] = None,
    reference_macro_factor: Optional[float] = None,
    num_eval_samples: Optional[int] = None,
    basis_kind: Optional[str] = None,
    basis_dim: Optional[int] = None,
    calibration_windows: Optional[int] = None,
    bo_val_windows: Optional[int] = None,
    confirm_val_windows: Optional[int] = None,
    confirm_top_k: Optional[int] = None,
    calibration_trace_samples: Optional[int] = None,
    bo_seed: Optional[int] = None,
    n_initial: Optional[int] = None,
    bo_batch_size: Optional[int] = None,
    lambda_kl: Optional[float] = None,
    theta_bound: Optional[float] = None,
    raw_samples: Optional[int] = None,
    num_restarts: Optional[int] = None,
    mc_samples: Optional[int] = None,
    density_floor_eta: Optional[float] = None,
) -> Optional[Path]:
    root = Path(workspace_root)
    candidates = [Path(item) for item in cache_roots]
    for candidate in candidates:
        run_root = candidate if candidate.is_absolute() else root / candidate
        solver_dir = run_root / str(solver_key)
        if not solver_dir.exists():
            continue
        run_config_path = run_root / "run_config.json"
        reference_path = solver_dir / "reference_schedule.json"
        observations_path = solver_dir / "observations.json"
        best_path = solver_dir / "best_schedule.json"
        validation_split_path = solver_dir / "validation_split.json"
        confirmation_path = solver_dir / "confirmation_rows.json"
        if not (
            run_config_path.exists()
            and reference_path.exists()
            and observations_path.exists()
            and best_path.exists()
            and validation_split_path.exists()
            and confirmation_path.exists()
        ):
            continue
        run_config = load_json(run_config_path)
        reference = load_json(reference_path)
        observations = load_json(observations_path)
        validation_split = load_json(validation_split_path)
        confirmation = load_json(confirmation_path)
        observation_metadata = observations.get("run_metadata", {}) if isinstance(observations, Mapping) else {}
        confirmation_metadata = confirmation.get("run_metadata", {}) if isinstance(confirmation, Mapping) else {}
        if not observations.get("run_fingerprint") or not confirmation.get("run_fingerprint"):
            continue
        if str(confirmation_metadata.get("observation_run_fingerprint", "")) != str(observations["run_fingerprint"]):
            continue
        config_dataset = run_config.get("dataset", reference.get("dataset"))
        config_nfe = run_config.get("target_nfe", reference.get("target_nfe"))
        if str(config_dataset) != str(dataset):
            continue
        if int(config_nfe) != int(target_nfe):
            continue
        if str(reference.get("solver_key")) != str(solver_key):
            continue
        if checkpoint_id is not None:
            config_checkpoint = run_config.get("checkpoint_id")
            if config_checkpoint is None or str(config_checkpoint) != str(checkpoint_id):
                continue
        if train_steps is not None:
            config_steps = run_config.get("otflow_train_steps", run_config.get("train_steps"))
            if config_steps is None or int(config_steps) != int(train_steps):
                continue
        if bo_budget is not None:
            config_budget = run_config.get("bo_budget", observation_metadata.get("bo_budget"))
            if config_budget is None or int(config_budget) != int(bo_budget):
                continue
            if len(observations.get("observations", [])) < int(bo_budget):
                continue
        if calibration_fraction is not None:
            config_fraction = run_config.get("calibration_fraction", observation_metadata.get("calibration_fraction"))
            if config_fraction is None or abs(float(config_fraction) - float(calibration_fraction)) > 1e-12:
                continue
        if reference_macro_factor is not None:
            config_factor = run_config.get("reference_macro_factor", observation_metadata.get("reference_macro_factor", reference.get("reference_macro_factor")))
            if config_factor is None or abs(float(config_factor) - float(reference_macro_factor)) > 1e-12:
                continue
        if num_eval_samples is not None:
            config_samples = run_config.get(
                "num_eval_samples",
                observation_metadata.get("num_eval_samples", observations.get("uniform_validation_metrics", {}).get("num_eval_samples")),
            )
            if config_samples is None or int(config_samples) != int(num_eval_samples):
                continue
        optional_ints = {
            "calibration_windows": calibration_windows,
            "bo_val_windows": bo_val_windows,
            "confirm_val_windows": confirm_val_windows,
            "confirm_top_k": confirm_top_k,
            "calibration_trace_samples": calibration_trace_samples,
            "bo_seed": bo_seed,
            "n_initial": n_initial,
            "bo_batch_size": bo_batch_size,
            "raw_samples": raw_samples,
            "num_restarts": num_restarts,
            "mc_samples": mc_samples,
        }
        rejected = False
        for key, expected in optional_ints.items():
            if expected is None:
                continue
            actual = run_config.get(key, observation_metadata.get(key, confirmation_metadata.get(key)))
            if actual is None or int(actual) != int(expected):
                rejected = True
                break
        if rejected:
            continue
        optional_floats = {
            "lambda_kl": lambda_kl,
            "theta_bound": theta_bound,
            "density_floor_eta": density_floor_eta,
        }
        for key, expected in optional_floats.items():
            if expected is None:
                continue
            actual = run_config.get(key, observation_metadata.get(key, confirmation_metadata.get(key)))
            if actual is None or abs(float(actual) - float(expected)) > 1e-12:
                rejected = True
                break
        if rejected:
            continue
        if calibration_windows is not None and int(calibration_windows) > 0 and len(validation_split.get("calibration_indices", [])) != int(calibration_windows):
            continue
        if bo_val_windows is not None and int(bo_val_windows) > 0 and len(validation_split.get("bo_validation_indices", [])) != int(bo_val_windows):
            continue
        if confirm_val_windows is not None and int(confirm_val_windows) > 0 and len(validation_split.get("confirmation_indices", [])) != int(confirm_val_windows):
            continue
        if confirmation_metadata.get("observation_run_fingerprint") is not None and observations.get("run_fingerprint") is not None:
            if str(confirmation_metadata["observation_run_fingerprint"]) != str(observations["run_fingerprint"]):
                continue
        if basis_kind is not None:
            config_basis = run_config.get("basis_kind", observations.get("basis_kind"))
            if config_basis is None or str(config_basis) != str(basis_kind):
                continue
        if basis_dim is not None:
            expected_dim = int(basis_dim)
            config_dim = run_config.get("basis_dim", observations.get("basis_dim"))
            if config_dim is not None and int(config_dim) != expected_dim:
                continue
            mismatched_theta = False
            for row in observations.get("observations", []):
                row_dim = row.get("basis_dim")
                if row_dim is not None and int(row_dim) != expected_dim:
                    mismatched_theta = True
                    break
                theta = row.get("theta")
                if theta is not None and len(theta) != expected_dim:
                    mismatched_theta = True
                    break
            if mismatched_theta:
                continue
        return solver_dir
    return None


def all_final_cache_roots(workspace_root: str | Path, extra_roots: Sequence[str | Path] = ()) -> List[Path]:
    root = Path(workspace_root)
    out: List[Path] = []
    for item in extra_roots:
        path = Path(item)
        out.append(path if path.is_absolute() else root / path)
    unique: List[Path] = []
    seen = set()
    for path in out:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique
