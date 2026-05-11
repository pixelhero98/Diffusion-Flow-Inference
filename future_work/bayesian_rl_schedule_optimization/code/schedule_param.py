from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from residual_parameterization import (
    build_residual_basis,
    candidate_from_theta,
    kl_divergence,
    theta_to_schedule_record,
    validate_interval_probabilities,
)


def interval_probabilities_from_grid(time_grid: Sequence[float]) -> np.ndarray:
    grid = np.asarray(time_grid, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional.")
    if grid.size < 2:
        raise ValueError("time_grid must contain at least two nodes.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("time_grid contains non-finite values.")
    if abs(float(grid[0])) > 1e-10:
        raise ValueError("time_grid must start at 0.0.")
    if abs(float(grid[-1]) - 1.0) > 1e-10:
        raise ValueError("time_grid must end at 1.0.")
    q = np.diff(grid)
    if np.any(q <= 0.0):
        raise ValueError("time_grid must be strictly increasing.")
    return validate_interval_probabilities(q, name="q")


def schedule_smoothness(q: Sequence[float]) -> float:
    q_arr = validate_interval_probabilities(q, name="q")
    if q_arr.size < 3:
        return 0.0
    log_q = np.log(q_arr)
    second = np.diff(log_q, n=2)
    return float(np.mean(second * second))


def schedule_diagnostics(time_grid: Sequence[float], q_ref: Sequence[float]) -> Dict[str, float]:
    q = interval_probabilities_from_grid(time_grid)
    return {
        "min_dt": float(np.min(q)),
        "smoothness": schedule_smoothness(q),
        "kl_to_reference": float(kl_divergence(q, q_ref)),
    }


def theta_to_checked_schedule(
    q_ref: Sequence[float],
    theta: Sequence[float],
    *,
    basis: Optional[np.ndarray] = None,
    hard_kl_cap: Optional[float] = None,
) -> Dict[str, Any]:
    q0 = validate_interval_probabilities(q_ref, name="q_ref")
    B = build_residual_basis(q0.size, q_ref=q0) if basis is None else np.asarray(basis, dtype=np.float64)
    record = theta_to_schedule_record(q0, theta, basis=B)
    diagnostics = schedule_diagnostics(record["schedule_grid"], q0)
    if hard_kl_cap is not None and diagnostics["kl_to_reference"] > float(hard_kl_cap):
        raise ValueError(
            f"Candidate KL {diagnostics['kl_to_reference']:.6g} exceeds hard_kl_cap {float(hard_kl_cap):.6g}."
        )
    record.update(diagnostics)
    return record


def schedule_record_from_mapping(
    row: Mapping[str, Any],
    *,
    q_ref: Sequence[float],
    basis: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if row.get("theta") is not None:
        return theta_to_checked_schedule(q_ref, row["theta"], basis=basis)
    if row.get("schedule_grid") is None:
        raise ValueError("Schedule row requires either theta or schedule_grid.")
    grid = [float(x) for x in row["schedule_grid"]]
    diagnostics = schedule_diagnostics(grid, q_ref)
    return {
        "theta": row.get("theta"),
        "schedule_grid": grid,
        "q": [float(x) for x in interval_probabilities_from_grid(grid).tolist()],
        **diagnostics,
    }


def build_default_basis_for_reference(q_ref: Sequence[float]) -> np.ndarray:
    q0 = validate_interval_probabilities(q_ref, name="q_ref")
    return build_residual_basis(q0.size, q_ref=q0)


def theta_to_grid(theta: Sequence[float], q_ref: Sequence[float], basis: np.ndarray) -> tuple[list[float], list[float]]:
    q, grid = candidate_from_theta(q_ref, theta, basis=basis)
    return [float(x) for x in grid.tolist()], [float(x) for x in q.tolist()]
