from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Mapping

import numpy as np
import torch

from diffusion_flow_inference.models.config import OTFlowConfig, SampleConfig
from diffusion_flow_inference.models.otflow_train_val import select_eval_window_starts

PRIMARY_METRICS = (
    "score_main",
    "tstr_macro_f1",
    "disc_auc",
    "disc_auc_gap",
    "unconditional_w1",
    "conditional_w1",
)

EXTRA_METRICS = (
    "u_l1",
    "c_l1",
    "spread_specific_error",
    "imbalance_specific_error",
    "ret_vol_acf_error",
    "impact_response_error",
    "efficiency_ms_per_sample",
)

ALL_METRICS = PRIMARY_METRICS + EXTRA_METRICS


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    if metric == "score_main":
        return float(result["cmp"]["score_main"]["mean"])
    if metric in PRIMARY_METRICS:
        return float(result["cmp"]["main"][metric]["mean"])
    return float(result["cmp"]["extra"][metric]["mean"])


def _metric_bundle(result: Mapping[str, Any]) -> Dict[str, float]:
    return {metric: float(_metric_value(result, metric)) for metric in ALL_METRICS}


def _choose_valid_windows(ds, horizon: int, n_windows: int, seed: int) -> np.ndarray:
    return select_eval_window_starts(
        ds, horizon=int(horizon), n_windows=int(n_windows), seed=int(seed)
    )


@contextmanager
def temporary_sample_config(
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    **overrides: Any,
):
    """Temporarily replace explicit sample sections and restore them on every exit path."""
    model_cfg = getattr(model, "cfg", None)
    owners = [cfg]
    if model_cfg is not None and model_cfg is not cfg:
        owners.append(model_cfg)

    valid_fields = {field.name for field in fields(SampleConfig)}
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(clean) - valid_fields)
    if unknown:
        raise TypeError(f"Unknown sample config fields: {unknown}")

    snapshots = []
    for owner in owners:
        sample = getattr(owner, "sample", None)
        if not isinstance(sample, SampleConfig):
            raise TypeError(
                "temporary_sample_config requires OTFlowConfig.sample to be a SampleConfig."
            )
        snapshots.append((owner, copy.deepcopy(sample)))

    try:
        for owner, snapshot in snapshots:
            replacement = copy.deepcopy(snapshot)
            for key, value in clean.items():
                setattr(replacement, key, value)
            owner.sample = replacement
        yield
    finally:
        for owner, snapshot in snapshots:
            owner.sample = snapshot


__all__ = [
    "ALL_METRICS",
    "EXTRA_METRICS",
    "PRIMARY_METRICS",
    "_choose_valid_windows",
    "_metric_bundle",
    "_metric_value",
    "temporary_sample_config",
]
