from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from residual_parameterization import (
    basis_bump_centers_for_dim,
    basis_kind_for_dim,
    FORECAST_AVG_RELATIVE_OBJECTIVE,
    build_residual_basis,
    normalize_observations,
    theta_to_schedule_record,
    validate_interval_probabilities,
)


INSTALL_MESSAGE = (
    "BoTorch is required for suggest-bo-batch. Install future-work dependencies with "
    "`python -m pip install -r future_work/bayesian_rl_schedule_optimization/requirements.txt`."
)


def _require_botorch():
    try:
        import torch
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except Exception as exc:  # pragma: no cover - exercised only when optional deps are absent.
        raise RuntimeError(INSTALL_MESSAGE) from exc
    return {
        "torch": torch,
        "SingleTaskGP": SingleTaskGP,
        "Standardize": Standardize,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "SobolQMCNormalSampler": SobolQMCNormalSampler,
        "qLogNoisyExpectedImprovement": qLogNoisyExpectedImprovement,
        "optimize_acqf": optimize_acqf,
    }


def botorch_available() -> bool:
    try:
        _require_botorch()
    except RuntimeError:
        return False
    return True


def _sampler(factory: Mapping[str, Any], *, n_mc_samples: int, seed: int):
    torch = factory["torch"]
    sampler_cls = factory["SobolQMCNormalSampler"]
    try:
        return sampler_cls(sample_shape=torch.Size([int(n_mc_samples)]), seed=int(seed))
    except TypeError:
        return sampler_cls(num_samples=int(n_mc_samples), seed=int(seed))


def _uniform_baseline_summary(observations: Sequence[Mapping[str, Any]]) -> Dict[str, float] | None:
    forecast_rows = [row for row in observations if row.get("objective_type") == FORECAST_AVG_RELATIVE_OBJECTIVE]
    if not forecast_rows:
        return None
    first = forecast_rows[0]
    if first.get("uniform_crps") is None or first.get("uniform_mase") is None:
        return None
    return {"crps": float(first["uniform_crps"]), "mase": float(first["uniform_mase"])}


def _observation_rows(observations_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    if isinstance(observations_payload, Mapping):
        return observations_payload.get("observations", [])
    return observations_payload


def _theta_dim_from_observations(observations_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> int:
    for row in _observation_rows(observations_payload):
        if row.get("theta") is not None:
            return len(row["theta"])
    return 5


def suggest_bo_batch(
    reference: Mapping[str, Any],
    observations_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 2,
    lambda_kl: float = 0.05,
    theta_bound: float = 3.0,
    raw_samples: int = 128,
    num_restarts: int = 16,
    n_mc_samples: int = 128,
    seed: int = 0,
) -> Dict[str, Any]:
    q_ref = validate_interval_probabilities(reference["q_ref"], name="q_ref")
    theta_dim = _theta_dim_from_observations(observations_payload)
    basis = build_residual_basis(q_ref.size, bump_centers=basis_bump_centers_for_dim(theta_dim), q_ref=q_ref)
    observations = normalize_observations(observations_payload, q_ref=q_ref, basis=basis, lambda_kl=float(lambda_kl))
    if len(observations) < 2:
        raise ValueError("At least two observations are required to fit the noisy BO surrogate.")
    objective_types = sorted({str(row.get("objective_type", "unknown")) for row in observations})
    objective_type = objective_types[0] if len(objective_types) == 1 else "mixed"
    uniform_baseline = _uniform_baseline_summary(observations)
    dim = int(basis.shape[1])
    factory = _require_botorch()
    torch = factory["torch"]
    torch.manual_seed(int(seed))
    train_x = torch.tensor([row["theta"] for row in observations], dtype=torch.double)
    train_y = torch.tensor([[float(row["objective_value"])] for row in observations], dtype=torch.double)
    model = factory["SingleTaskGP"](train_x, train_y, outcome_transform=factory["Standardize"](m=1))
    mll = factory["ExactMarginalLogLikelihood"](model.likelihood, model)
    factory["fit_gpytorch_mll"](mll)
    sampler = _sampler(factory, n_mc_samples=int(n_mc_samples), seed=int(seed))
    acquisition = factory["qLogNoisyExpectedImprovement"](
        model=model,
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
    )
    bounds = torch.stack(
        [
            torch.full((dim,), -float(theta_bound), dtype=torch.double),
            torch.full((dim,), float(theta_bound), dtype=torch.double),
        ]
    )
    candidates, acq_value = factory["optimize_acqf"](
        acq_function=acquisition,
        bounds=bounds,
        q=int(batch_size),
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
        options={"batch_limit": 5, "maxiter": 200},
    )
    rows = []
    for idx, theta in enumerate(candidates.detach().cpu().numpy().astype(np.float64)):
        record = theta_to_schedule_record(q_ref, theta, basis=basis)
        record.update({"candidate_id": f"bo_{idx:03d}", "source": "qLogNoisyExpectedImprovement"})
        rows.append(record)
    return {
        "artifact": "bo_qlognei_batch_v1",
        "acquisition": "qLogNoisyExpectedImprovement",
        "surrogate": "SingleTaskGP",
        "dataset": reference.get("dataset"),
        "solver_key": reference.get("solver_key"),
        "target_nfe": reference.get("target_nfe"),
        "runtime_nfe": reference.get("runtime_nfe"),
        "basis_kind": basis_kind_for_dim(dim),
        "basis_dim": dim,
        "objective_type": objective_type,
        "observation_objective_types": objective_types,
        "uniform_baseline": uniform_baseline,
        "batch_size": int(batch_size),
        "lambda_kl": float(lambda_kl),
        "theta_bound": float(theta_bound),
        "n_observations": int(len(observations)),
        "acquisition_value": float(acq_value.detach().cpu().item()),
        "candidates": rows,
    }
