from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SMALL_KL_BAND: Tuple[float, float] = (0.005, 0.02)
DEFAULT_MEDIUM_KL_BAND: Tuple[float, float] = (0.02, 0.08)
LEGACY_BASIS_BUMPS: Tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
DEFAULT_BASIS_BUMPS: Tuple[float, ...] = (0.25, 0.50, 0.75)
LEGACY_BASIS_KIND = "tilt_curvature_two_local_bumps"
DEFAULT_BASIS_KIND = "tilt_curvature_three_local_bumps"
FORECAST_AVG_RELATIVE_OBJECTIVE = "forecast_avg_relative_crps_mase"
PROVIDED_METRIC_OBJECTIVE = "provided_metric_val"
FORECAST_METRIC_FIELDS: Tuple[str, ...] = ("crps", "mase", "uniform_crps", "uniform_mase")
FORECAST_CANDIDATE_METRIC_FIELDS: Tuple[str, ...] = ("crps", "mase")
FORECAST_BASELINE_FIELDS: Tuple[str, ...] = ("uniform_crps", "uniform_mase")
BASELINE_MATCH_TOL = 1e-10


def validate_interval_probabilities(q: Sequence[float], *, name: str = "q") -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two intervals.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    arr = arr / total
    return arr


def softmax(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    shifted = x - float(np.max(x))
    exp_x = np.exp(shifted)
    return exp_x / float(np.sum(exp_x))


def _standardize_column(col: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = col - float(np.sum(weights * col))
    scale = float(np.sqrt(np.sum(weights * centered * centered)))
    if scale <= 1e-12:
        return centered
    return centered / scale


def basis_bump_centers_for_dim(theta_dim: int) -> Tuple[float, ...]:
    dim = int(theta_dim)
    if dim == 4:
        return LEGACY_BASIS_BUMPS
    if dim == 5:
        return DEFAULT_BASIS_BUMPS
    raise ValueError(f"Unsupported residual theta dimension {theta_dim}; expected 4 or 5.")


def basis_kind_for_dim(theta_dim: int) -> str:
    dim = int(theta_dim)
    if dim == 4:
        return LEGACY_BASIS_KIND
    if dim == 5:
        return DEFAULT_BASIS_KIND
    raise ValueError(f"Unsupported residual theta dimension {theta_dim}; expected 4 or 5.")


def build_residual_basis(
    n_intervals: int,
    *,
    bump_centers: Sequence[float] = DEFAULT_BASIS_BUMPS,
    bump_width: float = 0.18,
    q_ref: Optional[Sequence[float]] = None,
) -> np.ndarray:
    n = int(n_intervals)
    if n < 2:
        raise ValueError(f"n_intervals must be at least 2, got {n_intervals}.")
    weights = validate_interval_probabilities(q_ref, name="q_ref") if q_ref is not None else np.full(n, 1.0 / n)
    if weights.size != n:
        raise ValueError(f"q_ref must have {n} intervals, got {weights.size}.")
    x = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    columns = [
        2.0 * x - 1.0,
        (2.0 * x - 1.0) ** 2,
    ]
    for center in bump_centers:
        c = float(center)
        if c <= 0.0 or c >= 1.0:
            raise ValueError(f"bump center must lie in (0, 1), got {center}.")
        columns.append(np.exp(-0.5 * ((x - c) / float(bump_width)) ** 2))
    basis = np.stack([_standardize_column(col, weights) for col in columns], axis=1)
    return basis.astype(np.float64)


def candidate_from_theta(
    q_ref: Sequence[float],
    theta: Sequence[float],
    *,
    basis: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    q0 = validate_interval_probabilities(q_ref, name="q_ref")
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1:
        raise ValueError("theta must be one-dimensional.")
    if basis is None:
        B = build_residual_basis(q0.size, bump_centers=basis_bump_centers_for_dim(theta_arr.size), q_ref=q0)
    else:
        B = np.asarray(basis, dtype=np.float64)
    if B.shape != (q0.size, theta_arr.size):
        raise ValueError(f"basis shape must be {(q0.size, theta_arr.size)}, got {B.shape}.")
    q = softmax(np.log(q0) + B @ theta_arr)
    grid = np.concatenate(([0.0], np.cumsum(q)))
    grid[-1] = 1.0
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("Candidate schedule grid is not strictly increasing.")
    return q, grid


def kl_divergence(q: Sequence[float], q_ref: Sequence[float]) -> float:
    q_arr = validate_interval_probabilities(q, name="q")
    q0 = validate_interval_probabilities(q_ref, name="q_ref")
    if q_arr.size != q0.size:
        raise ValueError(f"q and q_ref must have the same length, got {q_arr.size} and {q0.size}.")
    return float(np.sum(q_arr * (np.log(q_arr) - np.log(q0))))


def theta_to_schedule_record(
    q_ref: Sequence[float],
    theta: Sequence[float],
    *,
    basis: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    q, grid = candidate_from_theta(q_ref, theta, basis=basis)
    return {
        "theta": [float(x) for x in np.asarray(theta, dtype=np.float64).tolist()],
        "q": [float(x) for x in q.tolist()],
        "schedule_grid": [float(x) for x in grid.tolist()],
        "kl_to_reference": float(kl_divergence(q, q_ref)),
    }


def _sobol_unit_samples(n_samples: int, dim: int, *, seed: int) -> Optional[np.ndarray]:
    try:
        import torch
    except Exception:
        return None
    engine = torch.quasirandom.SobolEngine(dimension=int(dim), scramble=True, seed=int(seed))
    return engine.draw(int(n_samples)).detach().cpu().numpy().astype(np.float64)


def _direction_samples(n_samples: int, dim: int, *, seed: int, use_sobol: bool) -> np.ndarray:
    if bool(use_sobol):
        sobol = _sobol_unit_samples(n_samples, dim, seed=seed)
        if sobol is not None:
            centered = 2.0 * sobol - 1.0
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            return centered / np.maximum(norms, 1e-12)
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(int(n_samples), int(dim)))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-12)


def _scale_direction_to_kl(
    direction: np.ndarray,
    q_ref: np.ndarray,
    basis: np.ndarray,
    target_kl: float,
) -> np.ndarray:
    target = float(target_kl)
    if target <= 0.0:
        raise ValueError(f"target_kl must be positive, got {target_kl}.")
    low = 0.0
    high = 1.0
    for _ in range(64):
        q, _ = candidate_from_theta(q_ref, high * direction, basis=basis)
        if kl_divergence(q, q_ref) >= target:
            break
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        q, _ = candidate_from_theta(q_ref, mid * direction, basis=basis)
        if kl_divergence(q, q_ref) < target:
            low = mid
        else:
            high = mid
    return high * direction


def _target_kl_values(
    n_initial: int,
    *,
    small_kl: Tuple[float, float],
    medium_kl: Tuple[float, float],
    seed: int,
) -> List[Tuple[str, float]]:
    n = int(n_initial)
    if n < 8 or n > 16:
        raise ValueError(f"n_initial must be in [8, 16], got {n_initial}.")
    rng = np.random.default_rng(int(seed) + 97)
    n_small = n // 2
    n_medium = n - n_small
    small = rng.uniform(float(small_kl[0]), float(small_kl[1]), size=n_small)
    medium = rng.uniform(float(medium_kl[0]), float(medium_kl[1]), size=n_medium)
    return [("small", float(x)) for x in small] + [("medium", float(x)) for x in medium]


def generate_initial_perturbations(
    reference: Mapping[str, Any],
    *,
    n_initial: int = 12,
    small_kl: Tuple[float, float] = DEFAULT_SMALL_KL_BAND,
    medium_kl: Tuple[float, float] = DEFAULT_MEDIUM_KL_BAND,
    seed: int = 0,
    use_sobol: bool = True,
) -> Dict[str, Any]:
    q_ref = validate_interval_probabilities(reference["q_ref"], name="q_ref")
    basis = build_residual_basis(q_ref.size, q_ref=q_ref)
    directions = _direction_samples(int(n_initial), basis.shape[1], seed=int(seed), use_sobol=bool(use_sobol))
    targets = _target_kl_values(int(n_initial), small_kl=small_kl, medium_kl=medium_kl, seed=int(seed))
    candidates: List[Dict[str, Any]] = []
    for idx, (direction, (band, target_kl)) in enumerate(zip(directions, targets)):
        theta = _scale_direction_to_kl(direction, q_ref, basis, target_kl)
        record = theta_to_schedule_record(q_ref, theta, basis=basis)
        record.update({"candidate_id": f"init_{idx:03d}", "kl_band": band, "target_kl": float(target_kl)})
        candidates.append(record)
    return {
        "artifact": "bo_initial_perturbations_v1",
        "reference_artifact": reference.get("artifact"),
        "dataset": reference.get("dataset"),
        "solver_key": reference.get("solver_key"),
        "target_nfe": reference.get("target_nfe"),
        "runtime_nfe": reference.get("runtime_nfe"),
        "basis_kind": DEFAULT_BASIS_KIND,
        "basis_dim": int(basis.shape[1]),
        "n_initial": int(n_initial),
        "small_kl_band": [float(x) for x in small_kl],
        "medium_kl_band": [float(x) for x in medium_kl],
        "seed": int(seed),
        "used_sobol": bool(use_sobol and _sobol_unit_samples(1, basis.shape[1], seed=seed) is not None),
        "candidates": candidates,
    }


def observation_objective(metric_val: float, kl_to_reference: float, *, lambda_kl: float = 0.05) -> float:
    return float(-float(metric_val) - float(lambda_kl) * float(kl_to_reference))


def _finite_float(value: Any, *, name: str) -> float:
    try:
        cast = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(cast):
        raise ValueError(f"{name} must be finite.")
    return cast


def _uniform_baseline_pair(
    uniform_crps_value: Any,
    uniform_mase_value: Any,
    *,
    crps_name: str = "uniform_crps",
    mase_name: str = "uniform_mase",
) -> Dict[str, float]:
    uniform_crps = _finite_float(uniform_crps_value, name=crps_name)
    uniform_mase = _finite_float(uniform_mase_value, name=mase_name)
    if uniform_crps <= 0.0:
        raise ValueError(f"{crps_name} must be positive, got {uniform_crps}.")
    if uniform_mase <= 0.0:
        raise ValueError(f"{mase_name} must be positive, got {uniform_mase}.")
    return {"uniform_crps": uniform_crps, "uniform_mase": uniform_mase}


def uniform_baseline_from_payload(observations_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Optional[Dict[str, float]]:
    if not isinstance(observations_payload, Mapping) or observations_payload.get("uniform_baseline") is None:
        return None
    baseline = observations_payload["uniform_baseline"]
    if not isinstance(baseline, Mapping):
        raise ValueError("uniform_baseline must be an object with crps and mase fields.")
    missing = [field for field in ("crps", "mase") if baseline.get(field) is None]
    if missing:
        raise ValueError(f"uniform_baseline requires crps and mase fields; missing {missing}.")
    return _uniform_baseline_pair(
        baseline["crps"],
        baseline["mase"],
        crps_name="uniform_baseline.crps",
        mase_name="uniform_baseline.mase",
    )


def _row_uniform_baseline(row: Mapping[str, Any]) -> Dict[str, float]:
    missing = [field for field in FORECAST_BASELINE_FIELDS if row.get(field) is None]
    if missing:
        raise ValueError(f"Forecast observations require uniform_crps and uniform_mase baselines; missing {missing}.")
    return _uniform_baseline_pair(row["uniform_crps"], row["uniform_mase"])


def _require_matching_uniform_baseline(
    row_baseline: Mapping[str, float],
    expected_baseline: Mapping[str, float],
    *,
    context: str,
) -> None:
    for field in FORECAST_BASELINE_FIELDS:
        actual = float(row_baseline[field])
        expected = float(expected_baseline[field])
        if not np.isclose(actual, expected, rtol=BASELINE_MATCH_TOL, atol=BASELINE_MATCH_TOL):
            raise ValueError(f"{context} {field} must match session {field}; got {actual}, expected {expected}.")


def forecast_average_relative_metric(
    row: Mapping[str, Any],
    *,
    uniform_baseline: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    missing = [field for field in FORECAST_CANDIDATE_METRIC_FIELDS if row.get(field) is None]
    if missing:
        raise ValueError(f"Forecast observations require crps and mase fields; missing {missing}.")
    crps = _finite_float(row["crps"], name="crps")
    mase = _finite_float(row["mase"], name="mase")
    if crps < 0.0:
        raise ValueError(f"crps must be nonnegative, got {crps}.")
    if mase < 0.0:
        raise ValueError(f"mase must be nonnegative, got {mase}.")
    if uniform_baseline is None:
        baseline = _row_uniform_baseline(row)
    else:
        missing_baseline = [field for field in FORECAST_BASELINE_FIELDS if uniform_baseline.get(field) is None]
        if missing_baseline:
            raise ValueError(f"uniform_baseline requires uniform_crps and uniform_mase fields; missing {missing_baseline}.")
        baseline = _uniform_baseline_pair(uniform_baseline["uniform_crps"], uniform_baseline["uniform_mase"])
        if any(row.get(field) is not None for field in FORECAST_BASELINE_FIELDS):
            _require_matching_uniform_baseline(_row_uniform_baseline(row), baseline, context="row-level")
    uniform_crps = float(baseline["uniform_crps"])
    uniform_mase = float(baseline["uniform_mase"])
    relative_crps_ratio = float(crps / uniform_crps)
    relative_mase_ratio = float(mase / uniform_mase)
    metric_val = float(0.5 * (relative_crps_ratio + relative_mase_ratio))
    return {
        "uniform_crps": uniform_crps,
        "uniform_mase": uniform_mase,
        "relative_crps_ratio": relative_crps_ratio,
        "relative_mase_ratio": relative_mase_ratio,
        "metric_val": metric_val,
    }


def metric_payload_from_observation(
    row: Mapping[str, Any],
    *,
    uniform_baseline: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    if uniform_baseline is not None and any(row.get(field) is not None for field in FORECAST_CANDIDATE_METRIC_FIELDS):
        metric_payload = forecast_average_relative_metric(row, uniform_baseline=uniform_baseline)
        metric_payload["objective_type"] = FORECAST_AVG_RELATIVE_OBJECTIVE
        return metric_payload
    if row.get("metric_val") is not None:
        return {
            "metric_val": _finite_float(row["metric_val"], name="metric_val"),
            "objective_type": str(row.get("objective_type", PROVIDED_METRIC_OBJECTIVE)),
        }
    if uniform_baseline is not None or any(row.get(field) is not None for field in FORECAST_METRIC_FIELDS):
        metric_payload = forecast_average_relative_metric(row, uniform_baseline=uniform_baseline)
        metric_payload["objective_type"] = FORECAST_AVG_RELATIVE_OBJECTIVE
        return metric_payload
    raise ValueError("Observation must include either metric_val or forecast crps/mase with a session or row uniform baseline.")


def normalize_observations(
    observations_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    q_ref: Sequence[float],
    basis: Optional[np.ndarray] = None,
    lambda_kl: float = 0.05,
) -> List[Dict[str, Any]]:
    session_uniform_baseline = uniform_baseline_from_payload(observations_payload)
    rows = observations_payload.get("observations", observations_payload) if isinstance(observations_payload, Mapping) else observations_payload
    out: List[Dict[str, Any]] = []
    legacy_uniform_baseline: Optional[Dict[str, float]] = None
    for idx, row in enumerate(rows):
        payload = dict(row)
        theta = np.asarray(payload["theta"], dtype=np.float64)
        record = theta_to_schedule_record(q_ref, theta, basis=basis)
        kl = float(payload.get("kl_to_reference", record["kl_to_reference"]))
        metric_payload = metric_payload_from_observation(payload, uniform_baseline=session_uniform_baseline)
        if metric_payload.get("objective_type") == FORECAST_AVG_RELATIVE_OBJECTIVE:
            row_baseline = {field: float(metric_payload[field]) for field in FORECAST_BASELINE_FIELDS}
            if session_uniform_baseline is None:
                if legacy_uniform_baseline is None:
                    legacy_uniform_baseline = row_baseline
                else:
                    _require_matching_uniform_baseline(row_baseline, legacy_uniform_baseline, context="legacy row-level")
        metric_val = float(metric_payload["metric_val"])
        objective = float(payload.get("objective_value", observation_objective(metric_val, kl, lambda_kl=lambda_kl)))
        payload.update(record)
        payload.update(metric_payload)
        payload.update(
            {
                "observation_id": str(payload.get("observation_id", f"obs_{idx:03d}")),
                "metric_val": metric_val,
                "kl_to_reference": kl,
                "objective_value": objective,
            }
        )
        out.append(payload)
    return out
