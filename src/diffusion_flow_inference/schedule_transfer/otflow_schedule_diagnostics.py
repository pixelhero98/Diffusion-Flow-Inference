from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_flow_inference.evaluation.otflow_sampling_support import _choose_valid_windows
from diffusion_flow_inference.models.otflow_train_val import (
    _future_time_context_seq,
    _get_dataset_item_by_t,
    _parse_batch,
    _temporary_eval_seed,
    crop_history_window,
    resolve_context_length,
)


def _prediction_horizon(model) -> int:
    model_cfg = getattr(model, "cfg", None)
    prediction_horizon = int(getattr(model_cfg, "prediction_horizon", 1))
    if prediction_horizon <= 0:
        raise ValueError(f"prediction_horizon must be positive, got {prediction_horizon}.")
    return prediction_horizon


def _sample_eval_trace(
    model,
    hist_t: torch.Tensor,
    *,
    cond_t: Optional[torch.Tensor],
    steps: int,
    solver: str,
    oracle_local_error: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any], int]:
    prediction_horizon = _prediction_horizon(model)
    if prediction_horizon > 1:
        if not hasattr(model, "sample_future_trace"):
            raise RuntimeError("A multi-step model must implement sample_future_trace(...).")
        x_block, trace = model.sample_future_trace(
            hist_t,
            cond=cond_t,
            steps=int(steps),
            solver=solver,
            oracle_local_error=oracle_local_error,
        )
        return x_block, trace, int(x_block.shape[1])

    x_next, trace = model.sample_trace(
        hist_t,
        cond=cond_t,
        steps=int(steps),
        solver=solver,
        oracle_local_error=oracle_local_error,
    )
    return x_next[:, None, :], trace, 1


def _append_rollout_context_features(
    block: torch.Tensor,
    *,
    x_hist: torch.Tensor,
    future_context_seq: Optional[torch.Tensor],
    cursor: int,
    take: int,
) -> torch.Tensor:
    if block.ndim != 3 or x_hist.ndim != 3:
        raise ValueError(
            "block and x_hist must both have shape [batch, time, features]; "
            f"got block={tuple(block.shape)} and x_hist={tuple(x_hist.shape)}."
        )
    if int(block.shape[0]) != int(x_hist.shape[0]) or int(block.shape[1]) != int(take):
        raise ValueError(
            "Rollout block must match the history batch and requested take length; "
            f"got block={tuple(block.shape)}, history={tuple(x_hist.shape)}, take={int(take)}."
        )
    target_dim = int(x_hist.shape[-1])
    block_dim = int(block.shape[-1])
    if block_dim == target_dim:
        return block
    if block_dim > target_dim:
        raise ValueError(
            f"Rollout block feature width {block_dim} exceeds history feature width {target_dim}."
        )

    extra_dim = int(target_dim - block_dim)
    if future_context_seq is None:
        raise ValueError(
            f"Rollout history requires {extra_dim} future context features, but none were provided."
        )
    if future_context_seq.ndim != 3:
        raise ValueError(
            "future_context_seq must have shape [batch, time, features], "
            f"got {tuple(future_context_seq.shape)}."
        )
    expected_end = int(cursor) + int(take)
    if (
        int(future_context_seq.shape[0]) != int(block.shape[0])
        or int(future_context_seq.shape[1]) < expected_end
        or int(future_context_seq.shape[2]) != extra_dim
    ):
        raise ValueError(
            "future_context_seq must match the rollout batch, cover the requested time slice, "
            f"and have feature width {extra_dim}; got {tuple(future_context_seq.shape)} "
            f"for cursor={int(cursor)}, take={int(take)}."
        )
    extra = future_context_seq[:, int(cursor) : int(cursor) + int(take), :].to(
        device=block.device,
        dtype=block.dtype,
    )
    return torch.cat([block, extra], dim=-1)


def _collect_rollout_diagnostics(
    model,
    ds,
    cfg,
    *,
    horizon: int,
    macro_steps: int,
    n_windows: int,
    seed: int,
    solver: str,
    chosen_t0s: Optional[Sequence[int]] = None,
    generation_seed_base: Optional[int] = None,
) -> Dict[str, Any]:
    if chosen_t0s is None:
        chosen = _choose_valid_windows(ds, horizon=horizon, n_windows=n_windows, seed=seed)
    else:
        chosen = np.asarray([int(t0) for t0 in chosen_t0s], dtype=np.int64)
    if chosen.ndim != 1 or chosen.size == 0:
        raise ValueError("chosen_t0s must be a non-empty 1D sequence of valid window starts.")
    seed_base = int(seed if generation_seed_base is None else generation_seed_base)
    field_eval_rows = []
    d_rows = []
    sample_total_evals = []

    for window_idx, t0 in enumerate(chosen.tolist()):
        batch = _get_dataset_item_by_t(ds, int(t0))
        hist, _, _, _, _ = _parse_batch(batch)
        hist_t = hist[None, :, :].to(cfg.device).float()
        context_len = resolve_context_length(hist_t.shape[1], horizon=horizon, cfg=cfg)
        cond_seq = None
        if ds.cond is not None:
            cond_seq = (
                torch.from_numpy(ds.cond[int(t0) : int(t0) + int(horizon)])
                .to(cfg.device)
                .float()[None, :, :]
            )
        future_context_seq = None
        future_context = _future_time_context_seq(ds, int(t0), int(horizon))
        if future_context is not None:
            future_context_seq = future_context.to(cfg.device).float()[None, :, :]

        x_hist = crop_history_window(hist_t, context_len).clone()
        cursor = 0
        while cursor < int(horizon):
            cond_t = cond_seq[:, cursor, :] if cond_seq is not None else None
            call_seed = seed_base + int(window_idx) * int(horizon) + int(cursor)
            with _temporary_eval_seed(call_seed):
                x_block, trace, block_len = _sample_eval_trace(
                    model,
                    x_hist,
                    cond_t=cond_t,
                    steps=int(macro_steps),
                    solver=solver,
                )
            if int(block_len) <= 0:
                raise ValueError(f"Sampled rollout block length must be positive, got {block_len}.")
            field_eval_rows.append(trace["field_evals_by_step"].cpu().numpy()[0])
            d_rows.append(trace["disagreement"].cpu().numpy()[0])
            sample_total_evals.append(float(trace["mean_total_field_evals_per_rollout"]))
            take = min(int(block_len), int(horizon) - int(cursor))
            hist_block = _append_rollout_context_features(
                x_block[:, :take, :],
                x_hist=x_hist,
                future_context_seq=future_context_seq,
                cursor=int(cursor),
                take=int(take),
            )
            x_hist = torch.cat([x_hist, hist_block], dim=1)
            x_hist = crop_history_window(x_hist, context_len)
            cursor += int(take)

    field_evals = np.asarray(field_eval_rows, dtype=np.float32)
    d_vals = np.asarray(d_rows, dtype=np.float32)

    return {
        "n_rollout_calls": int(field_evals.shape[0]),
        "macro_steps": int(macro_steps),
        "field_evals_by_step": [float(x) for x in field_evals.mean(axis=0)],
        "disagreement_by_step": [float(x) for x in d_vals.mean(axis=0)],
        "mean_field_evals_per_step": float(field_evals.mean()),
        "mean_total_field_evals_per_rollout": float(np.mean(sample_total_evals)),
    }


__all__ = [
    "_append_rollout_context_features",
    "_collect_rollout_diagnostics",
    "_sample_eval_trace",
]
