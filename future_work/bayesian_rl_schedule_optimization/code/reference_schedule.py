from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from diffusion_flow_inference.solvers.settings import solver_macro_steps, solver_order_p
except Exception:  # pragma: no cover - lets standalone future-work tests import this file.
    try:
        from otflow_evaluation_support import solver_macro_steps  # type: ignore
    except Exception:  # pragma: no cover - standalone future-work tests may not have core code.
        solver_macro_steps = None
    solver_order_p = None


LOCAL_DEFECT_KEY = "validation_local_defect_trace"
ORACLE_LOCAL_ERROR_KEY = "validation_oracle_local_error_trace"
INFO_GROWTH_KEY = "validation_info_growth_trace"
NATIVE_INFO_GROWTH_KEY = "info_growth_hardness_by_step"
DEFAULT_DENSITY_FLOOR_ETA = 0.05
DEFAULT_DEFECT_EPS = 1e-12


def validate_time_grid(grid: Sequence[float], *, name: str = "time_grid") -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two nodes.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if abs(float(arr[0])) > 1e-10:
        raise ValueError(f"{name} must start at 0.0.")
    if abs(float(arr[-1]) - 1.0) > 1e-10:
        raise ValueError(f"{name} must end at 1.0.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    arr[0] = 0.0
    arr[-1] = 1.0
    return arr


def _finite_interval_trace(values: Sequence[float], *, name: str, expected_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size != int(expected_len):
        raise ValueError(f"{name} must have {expected_len} intervals, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    arr = np.clip(arr, 0.0, None)
    if float(np.sum(arr)) <= 0.0:
        raise ValueError(f"{name} must contain positive mass.")
    return arr


def _solver_order(solver_key: str) -> float:
    if solver_order_p is not None:
        return float(solver_order_p(str(solver_key)))
    key = str(solver_key)
    if key == "euler":
        return 1.0
    if key in {"heun", "midpoint_rk2", "dpmpp2m"}:
        return 2.0
    raise ValueError(f"Unsupported solver_key={solver_key}")


def _runtime_nfe(cell: Mapping[str, Any]) -> int:
    if cell.get("runtime_nfe") is not None:
        return int(cell["runtime_nfe"])
    if solver_macro_steps is not None:
        return int(solver_macro_steps(str(cell["solver_key"]), int(cell["target_nfe"])))
    return int(cell["target_nfe"])


def local_defect_from_oracle(
    oracle_local_error: Sequence[float],
    reference_time_grid: Sequence[float],
    *,
    solver_order: float,
    eps: float = DEFAULT_DEFECT_EPS,
) -> np.ndarray:
    grid = validate_time_grid(reference_time_grid, name="reference_time_grid")
    oracle = _finite_interval_trace(oracle_local_error, name="oracle_local_error", expected_len=grid.size - 1)
    p = float(solver_order)
    if p <= 0.0:
        raise ValueError(f"solver_order must be positive, got {solver_order}.")
    widths = np.diff(grid)
    return oracle / (np.power(widths, p + 1.0) + float(eps))


def choose_defect_trace(cell: Mapping[str, Any], *, reference_time_grid: Sequence[float]) -> Tuple[np.ndarray, str]:
    grid = validate_time_grid(reference_time_grid, name="reference_time_grid")
    expected_len = int(grid.size - 1)

    def usable(key: str) -> Optional[np.ndarray]:
        if key not in cell or cell.get(key) is None:
            return None
        try:
            return _finite_interval_trace(cell[key], name=key, expected_len=expected_len)
        except ValueError:
            return None

    local = usable(LOCAL_DEFECT_KEY)
    if local is not None:
        return local, "local_defect"

    if cell.get(ORACLE_LOCAL_ERROR_KEY) is not None:
        try:
            solver_p = _solver_order(str(cell["solver_key"]))
            oracle_defect = local_defect_from_oracle(
                cell[ORACLE_LOCAL_ERROR_KEY],
                grid,
                solver_order=float(solver_p),
            )
            return oracle_defect, "oracle_local_error"
        except ValueError:
            pass

    info = usable(INFO_GROWTH_KEY)
    if info is None:
        info = usable(NATIVE_INFO_GROWTH_KEY)
    if info is not None:
        return info, "info_growth_fallback"
    raise ValueError("No usable local-defect, oracle-local-error, or Info-growth trace is available.")


def reference_density_from_defect(
    defect_trace: Sequence[float],
    reference_time_grid: Sequence[float],
    *,
    solver_order: float,
    eta: float = DEFAULT_DENSITY_FLOOR_ETA,
    eps: float = DEFAULT_DEFECT_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    grid = validate_time_grid(reference_time_grid, name="reference_time_grid")
    defect = _finite_interval_trace(defect_trace, name="defect_trace", expected_len=grid.size - 1)
    p = float(solver_order)
    if p <= 0.0:
        raise ValueError(f"solver_order must be positive, got {solver_order}.")
    eta_value = float(eta)
    if eta_value < 0.0 or eta_value > 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}.")
    widths = np.diff(grid)
    raw = np.power(defect + float(eps), 1.0 / (p + 1.0))
    integral = float(np.sum(widths * raw))
    if integral <= 0.0:
        raise ValueError("Reference density integral must be positive.")
    normalized = raw / integral
    blended = (1.0 - eta_value) * normalized + eta_value
    blended_integral = float(np.sum(widths * blended))
    if blended_integral <= 0.0:
        raise ValueError("Blended reference density integral must be positive.")
    return raw, blended / blended_integral


def runtime_grid_from_density(
    reference_time_grid: Sequence[float],
    density: Sequence[float],
    *,
    runtime_nfe: int,
) -> np.ndarray:
    grid = validate_time_grid(reference_time_grid, name="reference_time_grid")
    rho = _finite_interval_trace(density, name="density", expected_len=grid.size - 1)
    n = int(runtime_nfe)
    if n <= 0:
        raise ValueError(f"runtime_nfe must be positive, got {runtime_nfe}.")
    widths = np.diff(grid)
    masses = widths * rho
    total = float(np.sum(masses))
    if total <= 0.0:
        raise ValueError("Density mass must be positive.")
    cdf = np.concatenate(([0.0], np.cumsum(masses / total)))
    cdf[-1] = 1.0
    targets = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    out = np.interp(targets, cdf, grid)
    out[0] = 0.0
    out[-1] = 1.0
    if np.any(np.diff(out) <= 0.0):
        raise ValueError("Generated runtime schedule is not strictly increasing.")
    return out


def find_payload_cell(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    solver: str,
    target_nfe: int,
) -> Mapping[str, Any]:
    matches = []
    for cell in payload.get("cells", []):
        if (
            str(cell.get("dataset")) == str(dataset)
            and str(cell.get("solver_key")) == str(solver)
            and int(cell.get("target_nfe")) == int(target_nfe)
        ):
            matches.append(cell)
    if not matches:
        raise ValueError(f"No PTG payload cell for dataset={dataset}, solver={solver}, target_nfe={target_nfe}.")
    if len(matches) > 1:
        raise ValueError(f"Expected one matching PTG payload cell, found {len(matches)}.")
    return matches[0]


def build_reference_from_cell(
    cell: Mapping[str, Any],
    *,
    eta: float = DEFAULT_DENSITY_FLOOR_ETA,
    eps: float = DEFAULT_DEFECT_EPS,
) -> Dict[str, Any]:
    reference_grid = validate_time_grid(cell["reference_time_grid"], name="reference_time_grid")
    defect, trace_source = choose_defect_trace(cell, reference_time_grid=reference_grid)
    solver_key = str(cell["solver_key"])
    p = _solver_order(solver_key)
    runtime_nfe = _runtime_nfe(cell)
    raw_density, density = reference_density_from_defect(defect, reference_grid, solver_order=p, eta=eta, eps=eps)
    runtime_grid = runtime_grid_from_density(reference_grid, density, runtime_nfe=runtime_nfe)
    q_ref = np.diff(runtime_grid)
    return {
        "artifact": "bo_reference_schedule_v1",
        "reference_generator": "ser_ptg_local_defect_primary",
        "fallback_strategy": "info_growth_ser_ptg",
        "trace_source": trace_source,
        "dataset": str(cell["dataset"]),
        "solver_key": solver_key,
        "target_nfe": int(cell["target_nfe"]),
        "runtime_nfe": int(runtime_nfe),
        "solver_order_p": float(p),
        "reference_time_grid": [float(x) for x in reference_grid.tolist()],
        "defect_trace": [float(x) for x in defect.tolist()],
        "density_floor_eta": float(eta),
        "defect_eps": float(eps),
        "raw_density": [float(x) for x in raw_density.tolist()],
        "density": [float(x) for x in density.tolist()],
        "schedule_grid": [float(x) for x in runtime_grid.tolist()],
        "q_ref": [float(x) for x in q_ref.tolist()],
        "metric_name": "score_main" if str(cell.get("benchmark_family")) == "lob_conditional_generation" else "crps",
        "source_checkpoint_id": None if cell.get("checkpoint_id") is None else str(cell["checkpoint_id"]),
    }


def build_reference_from_payload(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    solver: str,
    target_nfe: int,
    eta: float = DEFAULT_DENSITY_FLOOR_ETA,
    eps: float = DEFAULT_DEFECT_EPS,
) -> Dict[str, Any]:
    cell = find_payload_cell(payload, dataset=dataset, solver=solver, target_nfe=target_nfe)
    return build_reference_from_cell(cell, eta=eta, eps=eps)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
