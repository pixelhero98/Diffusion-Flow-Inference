from __future__ import annotations

import math
from typing import Any, Dict, Mapping


DEFAULT_LAMBDA_BAD = 3.0


def _positive_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value}.")
    return out


def forecast_log_ratio_reward(
    *,
    crps: Any,
    mase: Any,
    uniform_crps: Any,
    uniform_mase: Any,
    kl_to_reference: Any,
    beta_ref: float,
    lambda_bad: float = DEFAULT_LAMBDA_BAD,
) -> Dict[str, float]:
    crps_value = _positive_float(crps, name="crps")
    mase_value = _positive_float(mase, name="mase")
    uniform_crps_value = _positive_float(uniform_crps, name="uniform_crps")
    uniform_mase_value = _positive_float(uniform_mase, name="uniform_mase")
    beta_value = float(beta_ref)
    lambda_bad_value = float(lambda_bad)
    if not math.isfinite(beta_value) or beta_value < 0.0:
        raise ValueError(f"beta_ref must be finite and nonnegative, got {beta_ref}.")
    if not math.isfinite(lambda_bad_value) or lambda_bad_value < 0.0:
        raise ValueError(f"lambda_bad must be finite and nonnegative, got {lambda_bad}.")
    kl_value = float(kl_to_reference)
    if not math.isfinite(kl_value) or kl_value < 0.0:
        raise ValueError(f"kl_to_reference must be finite and nonnegative, got {kl_to_reference}.")

    relative_crps_ratio = float(crps_value / uniform_crps_value)
    relative_mase_ratio = float(mase_value / uniform_mase_value)
    log_rcrps = float(math.log(relative_crps_ratio))
    log_rmase = float(math.log(relative_mase_ratio))
    metric_reward = float(-0.5 * (log_rcrps + log_rmase))
    bad_penalty = float(max(0.0, log_rcrps) + max(0.0, log_rmase))
    kl_penalty = float(beta_value * kl_value)
    reward = float(metric_reward - lambda_bad_value * bad_penalty - kl_penalty)
    return {
        "relative_crps_ratio": relative_crps_ratio,
        "relative_mase_ratio": relative_mase_ratio,
        "avg_relative_ratio": float(0.5 * (relative_crps_ratio + relative_mase_ratio)),
        "log_rcrps": log_rcrps,
        "log_rmase": log_rmase,
        "avg_log_ratio": float(0.5 * (log_rcrps + log_rmase)),
        "metric_reward": metric_reward,
        "bad_penalty": bad_penalty,
        "kl_penalty": kl_penalty,
        "reward": reward,
        "lambda_bad": lambda_bad_value,
        "beta_ref": beta_value,
        "smoothness_penalty": 0.0,
        "guard_penalty": 0.0,
    }


def reward_from_metrics(
    metrics: Mapping[str, Any],
    *,
    uniform_baseline: Mapping[str, Any],
    kl_to_reference: Any,
    beta_ref: float,
    lambda_bad: float = DEFAULT_LAMBDA_BAD,
) -> Dict[str, float]:
    return forecast_log_ratio_reward(
        crps=metrics["crps"],
        mase=metrics["mase"],
        uniform_crps=uniform_baseline["crps"],
        uniform_mase=uniform_baseline["mase"],
        kl_to_reference=kl_to_reference,
        beta_ref=beta_ref,
        lambda_bad=lambda_bad,
    )


def reward_from_observation(
    row: Mapping[str, Any],
    *,
    beta_ref: float,
    lambda_bad: float = DEFAULT_LAMBDA_BAD,
) -> Dict[str, float]:
    return forecast_log_ratio_reward(
        crps=row["crps"],
        mase=row["mase"],
        uniform_crps=row["uniform_crps"],
        uniform_mase=row["uniform_mase"],
        kl_to_reference=row.get("kl_to_reference", 0.0),
        beta_ref=beta_ref,
        lambda_bad=lambda_bad,
    )


def selector_rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["reward"]),
        -float(row.get("avg_log_ratio", 0.0)),
        -max(float(row.get("log_rcrps", 0.0)), float(row.get("log_rmase", 0.0))),
        -float(row.get("kl_to_reference", 0.0)),
    )
