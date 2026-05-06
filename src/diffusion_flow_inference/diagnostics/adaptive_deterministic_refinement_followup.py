#!/usr/bin/env python3
"""Disagreement-guided deterministic refinement pilot for OTFlow."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_PLOT = True
except Exception:  # pragma: no cover - plotting is optional in smoke environments
    plt = None
    _HAS_PLOT = False

from diffusion_flow_inference.diagnostics.adaptive_noise_sampler_followup import (
    ALL_METRICS,
    _apply_sample_overrides,
    _choose_valid_windows,
    _make_args,
    _metric_value,
    _restore_sample_overrides,
    _save_checkpoint,
)
from diffusion_flow_inference.datasets.experiment_common import build_cfg_from_args, build_dataset_splits, mkdir
from diffusion_flow_inference.diagnostics.signal_traces import (
    NATIVE_INFO_GROWTH_ROW_KEY,
    NATIVE_INFO_GROWTH_TRACE_KEY,
    MODEL_SIGNAL_SPECS,
    compute_info_growth_hardness_numpy,
    resolved_info_growth_scale,
)
from diffusion_flow_inference.backbones.training.train_val import (
    _future_context_seq_for_window,
    _temporary_eval_seed,
    _get_dataset_item_by_t,
    _parse_batch,
    crop_history_window,
    eval_many_windows,
    resolve_context_length,
    save_json,
    seed_all,
    train_loop,
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the deterministic refinement sampler pilot on cryptos.")
    ap.add_argument("--dataset", type=str, default="cryptos", choices=("cryptos",))
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--out_root", type=str, default="results_adaptive_deterministic_refinement_followup_cryptos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_horizon", type=int, default=200)
    ap.add_argument("--eval_windows_val", type=int, default=30)
    ap.add_argument("--eval_windows_test", type=int, default=30)
    ap.add_argument("--budget_families", type=str, default="8,16")
    ap.add_argument("--percentile_values", type=str, default="0.85,0.90,0.95")
    ap.add_argument("--fixed_last_k_values", type=str, default="1,2")
    ap.add_argument("--refine_beta", type=float, default=0.9)
    ap.add_argument("--refine_sigma_eps", type=float, default=1e-6)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    return ap


def _parse_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_floats(text: str) -> List[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    if x.size < 2 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return {"n": int(x.size), "pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    return {"n": int(x.size), "pearson": pearson, "spearman": spearman}


def _metric_bundle(result: Mapping[str, Any]) -> Dict[str, float]:
    return {metric: float(_metric_value(result, metric)) for metric in ALL_METRICS}


def _variant_label(row: Mapping[str, Any]) -> str:
    solver_family = str(row["solver_family"])
    budget = int(row["budget_family"])
    if solver_family == "euler":
        return f"Euler-B{budget}"
    if solver_family == "heun":
        return f"Heun-B{budget}"
    if solver_family == "euler_refine_half_percentile":
        return f"Adaptive-p{100.0 * float(row['trigger_percentile']):.0f}-B{budget}"
    if solver_family == "euler_refine_half_fixed_late":
        return f"FixedLate-k{int(row['fixed_last_k'])}-B{budget}"
    return f"{solver_family}-B{budget}"


def _step_arrays(rows: Sequence[Mapping[str, Any]], macro_steps: int) -> Dict[str, List[float]]:
    mu = []
    sigma = []
    disagreement_by_step = []
    oracle_by_step = []
    signal_specs = [(row_key, out_key) for row_key, out_key in MODEL_SIGNAL_SPECS if out_key != "disagreement_by_step"]
    if rows and NATIVE_INFO_GROWTH_ROW_KEY in rows[0]:
        signal_specs.append((NATIVE_INFO_GROWTH_ROW_KEY, NATIVE_INFO_GROWTH_TRACE_KEY))
    signal_means: Dict[str, List[float]] = {out_key: [] for _, out_key in signal_specs}
    for step_idx in range(int(macro_steps)):
        d_vals = np.asarray(
            [float(row["disagreement"]) for row in rows if int(row["step_index"]) == step_idx],
            dtype=np.float64,
        )
        e_vals = np.asarray(
            [float(row["oracle_local_error"]) for row in rows if int(row["step_index"]) == step_idx],
            dtype=np.float64,
        )
        mu.append(float(np.mean(d_vals)) if d_vals.size > 0 else 0.0)
        sigma.append(float(np.std(d_vals)) if d_vals.size > 0 else 0.0)
        disagreement_by_step.append(float(np.mean(d_vals)) if d_vals.size > 0 else float("nan"))
        oracle_by_step.append(float(np.mean(e_vals)) if e_vals.size > 0 else float("nan"))
        for row_key, out_key in signal_specs:
            vals = np.asarray(
                [float(row[row_key]) for row in rows if int(row["step_index"]) == step_idx],
                dtype=np.float64,
            )
            signal_means[out_key].append(float(np.mean(vals)) if vals.size > 0 else float("nan"))
    payload = {
        "step_mu": mu,
        "step_sigma": sigma,
        "disagreement_by_step": disagreement_by_step,
        "oracle_local_error_by_step": oracle_by_step,
    }
    payload.update(signal_means)
    return payload


def _prediction_horizon(model) -> int:
    model_cfg = getattr(model, "cfg", None)
    return int(max(1, int(getattr(model_cfg, "prediction_horizon", 1))))


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
    if prediction_horizon > 1 and hasattr(model, "sample_future_trace"):
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
    target_dim = int(x_hist.shape[-1])
    block_dim = int(block.shape[-1])
    if block_dim == target_dim:
        return block
    if block_dim > target_dim:
        raise ValueError(f"Generated block dim {block_dim} exceeds history dim {target_dim}.")

    extra_dim = int(target_dim - block_dim)
    if future_context_seq is None:
        raise ValueError(
            "future_context_seq is required because generated rollout blocks do not include "
            f"the {extra_dim} extra history context feature(s)."
        )
    extra = future_context_seq[:, int(cursor) : int(cursor) + int(take), :].to(
        device=block.device,
        dtype=block.dtype,
    )
    expected_shape = (int(block.shape[0]), int(take), int(extra_dim))
    if tuple(extra.shape) != expected_shape:
        raise ValueError(f"future_context_seq slice has shape {tuple(extra.shape)}, expected {expected_shape}.")
    return torch.cat([block, extra], dim=-1)


def _step_thresholds_from_percentile(rows: Sequence[Mapping[str, Any]], macro_steps: int, q: float) -> List[float]:
    thresholds: List[float] = []
    for step_idx in range(int(macro_steps)):
        d_vals = np.asarray(
            [float(row["disagreement"]) for row in rows if int(row["step_index"]) == step_idx],
            dtype=np.float64,
        )
        thresholds.append(_safe_percentile(d_vals, float(q)) if d_vals.size > 0 else float("inf"))
    return thresholds


def _collect_calibration(
    model,
    ds_val,
    cfg,
    *,
    horizon: int,
    macro_steps: int,
    n_windows: int,
    seed: int,
    sigma_eps: float,
    solver: str = "euler",
    chosen_t0s: Optional[Sequence[int]] = None,
    generation_seed_base: Optional[int] = None,
) -> Dict[str, Any]:
    if chosen_t0s is None:
        chosen = _choose_valid_windows(ds_val, horizon=horizon, n_windows=n_windows, seed=seed)
    else:
        chosen = np.asarray([int(t0) for t0 in chosen_t0s], dtype=np.int64)
    seed_base = int(seed if generation_seed_base is None else generation_seed_base)
    rows: List[Dict[str, Any]] = []
    reference_time_grid: Optional[np.ndarray] = None

    for window_idx, t0 in enumerate(chosen.tolist()):
        batch = _get_dataset_item_by_t(ds_val, int(t0))
        hist, _, _, cond, _ = _parse_batch(batch)
        hist_t = hist[None, :, :].to(cfg.device).float()
        cond_t = cond[None, :].to(cfg.device).float() if cond is not None else None
        with _temporary_eval_seed(seed_base + int(window_idx)):
            _, trace, _ = _sample_eval_trace(
                model,
                hist_t,
                cond_t=cond_t,
                steps=int(macro_steps),
                solver=str(solver),
                oracle_local_error=True,
            )
        rollout_time_grid = trace["time_grid"].cpu().numpy()
        if reference_time_grid is None:
            reference_time_grid = np.asarray(rollout_time_grid, dtype=np.float64)
        elif not np.allclose(reference_time_grid, rollout_time_grid, atol=1e-8, rtol=1e-8):
            raise ValueError("Calibration trace time grids must match across validation rollouts.")
        step_left_times = reference_time_grid[:-1]
        disagreement = trace["disagreement"][0].cpu().numpy()
        velocity = trace["velocity_norm"][0].cpu().numpy()
        trace_signal_arrays = {
            row_key: trace[row_key][0].cpu().numpy()
            for row_key, _ in MODEL_SIGNAL_SPECS
            if row_key != "disagreement"
        }
        oracle = trace["oracle_local_error"][0].cpu().numpy()
        for step_idx in range(int(macro_steps)):
            row = {
                "window_index": int(window_idx),
                "t0": int(t0),
                "step_index": int(step_idx),
                "time": float(step_left_times[step_idx]),
                "disagreement": float(disagreement[step_idx]),
                "velocity_norm": float(velocity[step_idx]),
                "oracle_local_error": float(oracle[step_idx]),
            }
            for row_key, _ in MODEL_SIGNAL_SPECS:
                if row_key == "disagreement":
                    continue
                row[row_key] = float(trace_signal_arrays[row_key][step_idx])
            rows.append(row)

    residual_values = np.asarray([float(row["residual_norm"]) for row in rows], dtype=np.float64)
    info_growth_scale = resolved_info_growth_scale(residual_values)
    for row in rows:
        row[NATIVE_INFO_GROWTH_ROW_KEY] = float(
            compute_info_growth_hardness_numpy(
                np.asarray([float(row["residual_norm"])], dtype=np.float64),
                np.asarray([float(row["disagreement"])], dtype=np.float64),
                scale=float(info_growth_scale),
            )[0]
        )

    step_stats = _step_arrays(rows, macro_steps)
    for row in rows:
        step_idx = int(row["step_index"])
        sigma = max(float(step_stats["step_sigma"][step_idx]), float(sigma_eps))
        row["normalized_disagreement"] = float(
            (float(row["disagreement"]) - float(step_stats["step_mu"][step_idx])) / sigma
        )

    corr_rows = [row for row in rows if int(row["step_index"]) > 0]
    d_arr = np.asarray([float(row["disagreement"]) for row in corr_rows], dtype=np.float64)
    z_arr = np.asarray([float(row["normalized_disagreement"]) for row in corr_rows], dtype=np.float64)
    e_arr = np.asarray([float(row["oracle_local_error"]) for row in corr_rows], dtype=np.float64)
    signal_specs = list(MODEL_SIGNAL_SPECS) + [
        (NATIVE_INFO_GROWTH_ROW_KEY, NATIVE_INFO_GROWTH_TRACE_KEY),
    ]
    signal_correlations_vs_oracle = {
        out_key: _safe_corr(
            np.asarray([float(row[row_key]) for row in corr_rows], dtype=np.float64),
            e_arr,
        )
        for row_key, out_key in signal_specs
    }
    if reference_time_grid is None:
        reference_time_grid = np.linspace(0.0, 1.0, int(macro_steps) + 1, dtype=np.float64)

    payload = {
        "macro_steps": int(macro_steps),
        "solver": str(solver),
        "n_windows": int(len(chosen)),
        "exclude_step0_for_correlation": True,
        "reference_time_grid": [float(x) for x in reference_time_grid.tolist()],
        "reference_time_alignment": "left_endpoint",
        "info_growth_scale": float(info_growth_scale),
        "rows": rows,
        "step_mu": step_stats["step_mu"],
        "step_sigma": step_stats["step_sigma"],
        "disagreement_by_step": step_stats["disagreement_by_step"],
        "oracle_local_error_by_step": step_stats["oracle_local_error_by_step"],
        "disagreement_stats": {
            "mean": float(np.mean(d_arr)) if d_arr.size > 0 else float("nan"),
            "std": float(np.std(d_arr)) if d_arr.size > 0 else float("nan"),
            "p50": _safe_percentile(d_arr, 0.50),
            "p85": _safe_percentile(d_arr, 0.85),
            "p95": _safe_percentile(d_arr, 0.95),
        },
        "oracle_local_error_stats": {
            "mean": float(np.mean(e_arr)) if e_arr.size > 0 else float("nan"),
            "std": float(np.std(e_arr)) if e_arr.size > 0 else float("nan"),
            "p50": _safe_percentile(e_arr, 0.50),
            "p85": _safe_percentile(e_arr, 0.85),
            "p95": _safe_percentile(e_arr, 0.95),
        },
        "correlation_disagreement_vs_oracle": _safe_corr(d_arr, e_arr),
        "correlation_normalized_disagreement_vs_oracle": _safe_corr(z_arr, e_arr),
        "signal_correlations_vs_oracle": signal_correlations_vs_oracle,
    }
    for _, out_key in signal_specs:
        if out_key == "disagreement_by_step":
            continue
        payload[out_key] = step_stats[out_key]
    return payload


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
    seed_base = int(seed if generation_seed_base is None else generation_seed_base)
    fired_rows = []
    eligible_rows = []
    field_eval_rows = []
    z_rows = []
    d_rows = []
    sample_total_evals = []

    for window_idx, t0 in enumerate(chosen.tolist()):
        batch = _get_dataset_item_by_t(ds, int(t0))
        hist, _, _, _, _ = _parse_batch(batch)
        hist_t = hist[None, :, :].to(cfg.device).float()
        context_len = resolve_context_length(hist_t.shape[1], horizon=horizon, cfg=cfg)
        cond_seq = None
        if ds.cond is not None:
            cond_seq = torch.from_numpy(ds.cond[int(t0) : int(t0) + int(horizon)]).to(cfg.device).float()[None, :, :]
        future_context_seq = _future_context_seq_for_window(
            ds,
            int(t0),
            int(horizon),
            hist=hist_t,
            model=model,
            device=cfg.device,
            dtype=hist_t.dtype,
        )

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
            fired_rows.append(trace["fired"].to(dtype=torch.float32).cpu().numpy()[0])
            eligible_rows.append(trace["trigger_eligible"].to(dtype=torch.float32).cpu().numpy()[0])
            field_eval_rows.append(trace["field_evals_by_step"].cpu().numpy()[0])
            z_rows.append(trace["normalized_disagreement"].cpu().numpy()[0])
            d_rows.append(trace["disagreement"].cpu().numpy()[0])
            sample_total_evals.append(float(trace["mean_total_field_evals_per_rollout"]))
            take = min(int(block_len), int(horizon) - int(cursor))
            hist_block = x_block[:, :take, :]
            hist_block = _append_rollout_context_features(
                hist_block,
                x_hist=x_hist,
                future_context_seq=future_context_seq,
                cursor=int(cursor),
                take=int(take),
            )
            x_hist = torch.cat([x_hist, hist_block], dim=1)
            x_hist = crop_history_window(x_hist, context_len)
            cursor += int(take)

    fired = np.asarray(fired_rows, dtype=np.float32)
    eligible = np.asarray(eligible_rows, dtype=np.float32)
    field_evals = np.asarray(field_eval_rows, dtype=np.float32)
    z_vals = np.asarray(z_rows, dtype=np.float32)
    d_vals = np.asarray(d_rows, dtype=np.float32)
    eligible_mask = eligible > 0.5

    trigger_rate = float(fired[eligible_mask].mean()) if np.any(eligible_mask) else 0.0
    return {
        "n_rollout_calls": int(fired.shape[0]),
        "macro_steps": int(macro_steps),
        "trigger_rate": trigger_rate,
        "trigger_by_step": [float(x) for x in fired.mean(axis=0)],
        "eligible_by_step": [float(x) for x in eligible.mean(axis=0)],
        "field_evals_by_step": [float(x) for x in field_evals.mean(axis=0)],
        "disagreement_by_step": [float(x) for x in d_vals.mean(axis=0)],
        "normalized_disagreement_by_step": [float(x) for x in z_vals.mean(axis=0)],
        "mean_field_evals_per_step": float(field_evals.mean()),
        "mean_total_field_evals_per_rollout": float(np.mean(sample_total_evals)),
    }


def _select_budget_matched(rows: Sequence[Mapping[str, Any]], budget_family: int) -> Optional[Mapping[str, Any]]:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            abs(float(row["mean_total_field_evals_per_rollout"]) - float(budget_family)),
            float(row["conditional_w1"]),
            float(row["score_main"]),
        ),
    )[0]


def _pick_winner(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not rows:
        return None
    best_score = min(float(row["score_main"]) for row in rows)
    band = [row for row in rows if float(row["score_main"]) <= best_score * 1.01]
    return sorted(
        band,
        key=lambda row: (
            float(row["conditional_w1"]),
            float(row["mean_total_field_evals_per_rollout"]),
            float(row["efficiency_ms_per_sample"]),
        ),
    )[0]


def _diagnose_budget(
    *,
    budget_family: int,
    calibration: Mapping[str, Any],
    baseline: Mapping[str, Any],
    heun: Mapping[str, Any],
    adaptive_rows: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    best_adaptive = _pick_winner(adaptive_rows)
    best_fixed = _pick_winner(fixed_rows)
    adaptive_beats_euler = bool(best_adaptive is not None and float(best_adaptive["score_main"]) < float(baseline["score_main"]))
    fixed_beats_euler = bool(best_fixed is not None and float(best_fixed["score_main"]) < float(baseline["score_main"]))
    adaptive_beats_fixed = bool(
        best_adaptive is not None and best_fixed is not None and float(best_adaptive["score_main"]) < float(best_fixed["score_main"])
    )
    heun_beats_euler = bool(float(heun["score_main"]) < float(baseline["score_main"]))
    raw_corr = float(calibration["correlation_disagreement_vs_oracle"]["spearman"])
    corr_positive = bool(np.isfinite(raw_corr) and raw_corr > 0.1)
    if corr_positive and adaptive_beats_euler and adaptive_beats_fixed:
        diagnosis = "signal_validated"
    elif fixed_beats_euler and not adaptive_beats_fixed:
        diagnosis = "late_step_baseline_sufficient"
    elif heun_beats_euler and not adaptive_beats_euler:
        diagnosis = "better_solver_not_good_allocator"
    elif (not heun_beats_euler) and (not adaptive_beats_euler):
        diagnosis = "euler_already_robust"
    else:
        diagnosis = "mixed"
    return {
        "budget_family": int(budget_family),
        "raw_vs_oracle_spearman": raw_corr,
        "raw_vs_oracle_pearson": float(calibration["correlation_disagreement_vs_oracle"]["pearson"]),
        "correlation_gate_spearman_gt": 0.1,
        "correlation_positive": corr_positive,
        "adaptive_beats_euler": adaptive_beats_euler,
        "fixed_beats_euler": fixed_beats_euler,
        "adaptive_beats_fixed": adaptive_beats_fixed,
        "heun_beats_euler": heun_beats_euler,
        "best_adaptive_solver_family": None if best_adaptive is None else str(best_adaptive["solver_family"]),
        "best_adaptive_score_main": None if best_adaptive is None else float(best_adaptive["score_main"]),
        "best_fixed_solver_family": None if best_fixed is None else str(best_fixed["solver_family"]),
        "best_fixed_score_main": None if best_fixed is None else float(best_fixed["score_main"]),
        "baseline_score_main": float(baseline["score_main"]),
        "heun_score_main": float(heun["score_main"]),
        "diagnosis": diagnosis,
    }


def _plot_disagreement_histogram(path: str, calibration_by_budget: Mapping[int, Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for budget_family, payload in sorted(calibration_by_budget.items()):
        vals = np.asarray(
            [float(row["disagreement"]) for row in payload["rows"] if int(row["step_index"]) > 0],
            dtype=np.float64,
        )
        macro_steps = int(payload["macro_steps"])
        axes[0].hist(vals, bins=24, alpha=0.45, label=f"budget={budget_family} (steps={macro_steps})")
        q_grid = np.linspace(0.0, 1.0, 101)
        axes[1].plot(q_grid, np.quantile(vals, q_grid), linewidth=2.0, label=f"budget={budget_family}")
    axes[0].set_title("Disagreement histogram")
    axes[0].set_xlabel("disagreement")
    axes[0].set_ylabel("count")
    axes[1].set_title("Disagreement percentiles")
    axes[1].set_xlabel("quantile")
    axes[1].set_ylabel("disagreement")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_step_stats(path: str, calibration_by_budget: Mapping[int, Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for budget_family, payload in sorted(calibration_by_budget.items()):
        step_idx = list(range(int(payload["macro_steps"])))
        axes[0].plot(step_idx, payload["step_mu"], marker="o", linewidth=1.8, label=f"budget={budget_family}")
        axes[1].plot(step_idx, payload["step_sigma"], marker="o", linewidth=1.8, label=f"budget={budget_family}")
    axes[0].set_title("Per-step disagreement mean")
    axes[0].set_xlabel("solver step")
    axes[0].set_ylabel("mu_n")
    axes[1].set_title("Per-step disagreement std")
    axes[1].set_xlabel("solver step")
    axes[1].set_ylabel("sigma_n")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_oracle_correlation(path: str, calibration_by_budget: Mapping[int, Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    palette = ["#2e8b57", "#c03d3d", "#1f77b4", "#8c564b", "#9467bd"]
    colors = {
        int(budget): palette[idx % len(palette)]
        for idx, budget in enumerate(sorted(calibration_by_budget.keys()))
    }
    for budget_family, payload in sorted(calibration_by_budget.items()):
        rows = [row for row in payload["rows"] if int(row["step_index"]) > 0]
        d_vals = np.asarray([float(row["disagreement"]) for row in rows], dtype=np.float64)
        z_vals = np.asarray([float(row["normalized_disagreement"]) for row in rows], dtype=np.float64)
        e_vals = np.asarray([float(row["oracle_local_error"]) for row in rows], dtype=np.float64)
        color = colors.get(int(budget_family), None)
        axes[0].scatter(d_vals, e_vals, s=16, alpha=0.45, color=color, label=f"budget={budget_family}")
        axes[1].scatter(z_vals, e_vals, s=16, alpha=0.45, color=color, label=f"budget={budget_family}")
    axes[0].set_title("Raw disagreement vs oracle local error")
    axes[0].set_xlabel("d_n")
    axes[0].set_ylabel("e_n")
    axes[1].set_title("Normalized disagreement vs oracle local error")
    axes[1].set_xlabel("z_n")
    axes[1].set_ylabel("e_n")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_vs_score(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    color_map = {
        "euler": "black",
        "heun": "#1f77b4",
        "euler_refine_half_percentile": "#2e8b57",
        "euler_refine_half_fixed_late": "#ff7f0e",
    }
    marker_cycle = ["o", "s", "^", "D", "P"]
    marker_map = {
        int(budget): marker_cycle[idx % len(marker_cycle)]
        for idx, budget in enumerate(sorted({int(row["budget_family"]) for row in rows}))
    }
    for row in rows:
        color = color_map[str(row["solver_family"])]
        marker = marker_map[int(row["budget_family"])]
        ax.scatter(
            float(row["efficiency_ms_per_sample"]),
            float(row["score_main"]),
            color=color,
            marker=marker,
            s=70,
            alpha=0.9,
        )
        ax.annotate(
            _variant_label(row),
            (float(row["efficiency_ms_per_sample"]), float(row["score_main"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Latency vs score_main")
    ax.set_xlabel("efficiency_ms_per_sample")
    ax.set_ylabel("score_main")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_field_evals_vs_score(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    color_map = {
        "euler": "black",
        "heun": "#1f77b4",
        "euler_refine_half_percentile": "#2e8b57",
        "euler_refine_half_fixed_late": "#ff7f0e",
    }
    marker_cycle = ["o", "s", "^", "D", "P"]
    marker_map = {
        int(budget): marker_cycle[idx % len(marker_cycle)]
        for idx, budget in enumerate(sorted({int(row["budget_family"]) for row in rows}))
    }
    for row in rows:
        color = color_map[str(row["solver_family"])]
        marker = marker_map[int(row["budget_family"])]
        ax.scatter(
            float(row["mean_total_field_evals_per_rollout"]),
            float(row["score_main"]),
            color=color,
            marker=marker,
            s=70,
            alpha=0.9,
        )
        ax.annotate(
            _variant_label(row),
            (float(row["mean_total_field_evals_per_rollout"]), float(row["score_main"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Realized field evaluations vs score_main")
    ax.set_xlabel("mean_total_field_evals_per_rollout")
    ax.set_ylabel("score_main")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_trigger_rate_by_step(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    adaptive_rows = [row for row in rows if str(row["solver_family"]).startswith("euler_refine_half")]
    budgets = sorted({int(row["budget_family"]) for row in adaptive_rows})
    if not budgets:
        return
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.2 * len(budgets), 4.5), squeeze=False)
    for ax, budget_family in zip(axes[0], budgets):
        subset = [row for row in adaptive_rows if int(row["budget_family"]) == budget_family]
        for row in subset:
            ax.plot(
                range(int(row["macro_steps"])),
                row["diag"]["trigger_by_step"],
                marker="o",
                linewidth=1.8,
                label=_variant_label(row),
            )
        ax.set_title(f"Trigger rate by solver step (budget={budget_family})")
        ax.set_xlabel("solver step")
        ax.set_ylabel("trigger rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.2)
        if subset:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_realized_budget(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    labels = [_variant_label(row) for row in rows]
    values = [float(row["mean_total_field_evals_per_rollout"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(rows)), 4.6))
    ax.bar(range(len(rows)), values, color="#4c78a8", alpha=0.85)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mean_total_field_evals_per_rollout")
    ax.set_title("Realized field-evaluation budget by method")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_variant(
    *,
    model,
    ds_test,
    cfg,
    eval_horizon: int,
    eval_windows_test: int,
    budget_family: int,
    solver_family: str,
    solver_override: str,
    macro_steps: int,
    trigger_mode: Optional[str],
    trigger_percentile: Optional[float],
    trigger_threshold_z: Optional[float],
    trigger_threshold_raw: Optional[float],
    step_threshold: Sequence[float],
    fixed_last_k: Optional[int],
    refine_beta: float,
    refine_sigma_eps: float,
    step_mu: Sequence[float],
    step_sigma: Sequence[float],
    chosen_t0s: Sequence[int],
    generation_seed_base: int,
    metrics_seed: int,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "solver": solver_override,
        "refine_beta": float(refine_beta),
        "refine_sigma_eps": float(refine_sigma_eps),
        "refine_disallow_final_step": True,
    }
    if trigger_mode is not None:
        overrides.update(
            {
                "refine_trigger_mode": str(trigger_mode),
                "refine_threshold_z": None if trigger_threshold_z is None else float(trigger_threshold_z),
                "refine_threshold_raw": None if trigger_threshold_raw is None else float(trigger_threshold_raw),
                "refine_step_mu": tuple(float(x) for x in step_mu),
                "refine_step_sigma": tuple(float(x) for x in step_sigma),
                "refine_step_threshold": tuple(float(x) for x in step_threshold),
                "refine_fixed_last_k": None if fixed_last_k is None else int(fixed_last_k),
            }
        )

    backup = _apply_sample_overrides(model, cfg, **overrides)
    try:
        t0 = time.time()
        result = eval_many_windows(
            ds_test,
            model,
            cfg,
            horizon=int(eval_horizon),
            nfe=int(macro_steps),
            n_windows=int(eval_windows_test),
            seed=int(metrics_seed),
            horizons_eval=[int(eval_horizon)],
            chosen_t0s=chosen_t0s,
            generation_seed_base=int(generation_seed_base),
            metrics_seed=int(metrics_seed),
        )
        eval_seconds = float(time.time() - t0)
        diag = _collect_rollout_diagnostics(
            model,
            ds_test,
            cfg,
            horizon=int(eval_horizon),
            macro_steps=int(macro_steps),
            n_windows=int(eval_windows_test),
            seed=int(metrics_seed),
            solver=solver_override,
            chosen_t0s=chosen_t0s,
            generation_seed_base=int(generation_seed_base),
        )
    finally:
        _restore_sample_overrides(model, cfg, backup)

    row = {
        "budget_family": int(budget_family),
        "solver_family": str(solver_family),
        "solver_override": str(solver_override),
        "macro_steps": int(macro_steps),
        "trigger_mode": None if trigger_mode is None else str(trigger_mode),
        "trigger_percentile": None if trigger_percentile is None else float(trigger_percentile),
        "trigger_threshold_z": None if trigger_threshold_z is None else float(trigger_threshold_z),
        "trigger_threshold_raw": None if trigger_threshold_raw is None else float(trigger_threshold_raw),
        "fixed_last_k": None if fixed_last_k is None else int(fixed_last_k),
        "eval_seconds": eval_seconds,
        "mean_field_evals_per_step": float(diag["mean_field_evals_per_step"]),
        "mean_total_field_evals_per_rollout": float(diag["mean_total_field_evals_per_rollout"]),
        "trigger_rate": float(diag["trigger_rate"]),
        "diag": diag,
        "evaluation_protocol": {
            "chosen_t0s": [int(t0) for t0 in result["meta"]["chosen_t0s"]],
            "generation_seed_base": None
            if result["meta"]["generation_seed_base"] is None
            else int(result["meta"]["generation_seed_base"]),
            "metrics_seed": int(result["meta"]["metrics_seed"]),
        },
    }
    row.update(_metric_bundle(result))
    return row


def main() -> None:
    cli_args = build_argparser().parse_args()
    seed = int(cli_args.seed)
    seed_all(seed)
    mkdir(cli_args.out_root)

    args = _make_args(cli_args)
    cfg = build_cfg_from_args(args)
    splits = build_dataset_splits(args, cfg)
    model = train_loop(
        splits["train"],
        cfg,
        model_name="otflow",
        steps=int(cli_args.steps),
        log_every=int(cli_args.log_every),
    )
    checkpoint_path = os.path.join(cli_args.out_root, f"model_seed{int(cli_args.seed)}.pt")
    _save_checkpoint(checkpoint_path, model, cfg, splits["train"])

    eval_horizon = int(cli_args.eval_horizon)
    budget_families = _parse_ints(cli_args.budget_families)
    percentile_values = _parse_floats(cli_args.percentile_values)
    fixed_last_k_values = _parse_ints(cli_args.fixed_last_k_values)
    val_window_seed = seed + 17
    test_window_seed = seed + 101
    generation_seed_base = seed + 1001
    metrics_seed = seed + 2001
    shared_val_t0s = _choose_valid_windows(
        splits["val"],
        horizon=eval_horizon,
        n_windows=int(cli_args.eval_windows_val),
        seed=val_window_seed,
    ).tolist()
    shared_test_t0s = _choose_valid_windows(
        splits["test"],
        horizon=eval_horizon,
        n_windows=int(cli_args.eval_windows_test),
        seed=test_window_seed,
    ).tolist()
    evaluation_protocol = {
        "val_window_seed": int(val_window_seed),
        "test_window_seed": int(test_window_seed),
        "generation_seed_base": int(generation_seed_base),
        "metrics_seed": int(metrics_seed),
        "validation_chosen_t0s": [int(t0) for t0 in shared_val_t0s],
        "test_chosen_t0s": [int(t0) for t0 in shared_test_t0s],
    }

    calibration_by_budget: Dict[int, Dict[str, Any]] = {}
    for budget_family in budget_families:
        macro_steps = max(1, int(budget_family) // 2)
        calibration_by_budget[int(budget_family)] = _collect_calibration(
            model,
            splits["val"],
            cfg,
            horizon=eval_horizon,
            macro_steps=macro_steps,
            n_windows=int(cli_args.eval_windows_val),
            seed=val_window_seed,
            sigma_eps=float(cli_args.refine_sigma_eps),
            chosen_t0s=shared_val_t0s,
            generation_seed_base=int(generation_seed_base),
        )

    trace_calibration = {
        "dataset": "cryptos",
        "seed": seed,
        "eval_horizon": eval_horizon,
        "evaluation_protocol": evaluation_protocol,
        "budget_families": {
            str(budget_family): {
                "macro_steps": int(payload["macro_steps"]),
                "n_windows": int(payload["n_windows"]),
                "step_mu": payload["step_mu"],
                "step_sigma": payload["step_sigma"],
                "step_percentile_thresholds": {
                    f"p{int(round(100.0 * float(q)))}": _step_thresholds_from_percentile(payload["rows"], int(payload["macro_steps"]), float(q))
                    for q in percentile_values
                },
                "disagreement_stats": payload["disagreement_stats"],
            }
            for budget_family, payload in calibration_by_budget.items()
        },
    }
    oracle_error_analysis = {
        "dataset": "cryptos",
        "seed": seed,
        "eval_horizon": eval_horizon,
        "evaluation_protocol": evaluation_protocol,
        "budget_families": {
            str(budget_family): {
                "macro_steps": int(payload["macro_steps"]),
                "correlation_disagreement_vs_oracle": payload["correlation_disagreement_vs_oracle"],
                "correlation_normalized_disagreement_vs_oracle": payload["correlation_normalized_disagreement_vs_oracle"],
                "oracle_local_error_stats": payload["oracle_local_error_stats"],
            }
            for budget_family, payload in calibration_by_budget.items()
        },
    }
    save_json(trace_calibration, os.path.join(cli_args.out_root, "trace_calibration.json"))
    save_json(oracle_error_analysis, os.path.join(cli_args.out_root, "oracle_error_analysis.json"))

    _plot_disagreement_histogram(os.path.join(cli_args.out_root, "trace_calibration.png"), calibration_by_budget)
    _plot_step_stats(os.path.join(cli_args.out_root, "trace_step_stats.png"), calibration_by_budget)
    _plot_oracle_correlation(os.path.join(cli_args.out_root, "oracle_error_correlation.png"), calibration_by_budget)

    stage_b_all_rows: List[Dict[str, Any]] = []
    stage_b_selected_rows: List[Dict[str, Any]] = []
    acceptance_rows: List[Dict[str, Any]] = []

    for budget_family in budget_families:
        macro_steps_adaptive = max(1, int(budget_family) // 2)
        payload = calibration_by_budget[int(budget_family)]
        step_mu = payload["step_mu"]
        step_sigma = payload["step_sigma"]
        step_rows = payload["rows"]

        euler_row = _run_variant(
            model=model,
            ds_test=splits["test"],
            cfg=cfg,
            eval_horizon=eval_horizon,
            eval_windows_test=int(cli_args.eval_windows_test),
            budget_family=int(budget_family),
            solver_family="euler",
            solver_override="euler",
            macro_steps=int(budget_family),
            trigger_mode=None,
            trigger_percentile=None,
            trigger_threshold_z=None,
            trigger_threshold_raw=None,
            step_threshold=(),
            fixed_last_k=None,
            refine_beta=float(cli_args.refine_beta),
            refine_sigma_eps=float(cli_args.refine_sigma_eps),
            step_mu=(),
            step_sigma=(),
            chosen_t0s=shared_test_t0s,
            generation_seed_base=int(generation_seed_base),
            metrics_seed=int(metrics_seed),
        )
        heun_row = _run_variant(
            model=model,
            ds_test=splits["test"],
            cfg=cfg,
            eval_horizon=eval_horizon,
            eval_windows_test=int(cli_args.eval_windows_test),
            budget_family=int(budget_family),
            solver_family="heun",
            solver_override="heun",
            macro_steps=int(macro_steps_adaptive),
            trigger_mode=None,
            trigger_percentile=None,
            trigger_threshold_z=None,
            trigger_threshold_raw=None,
            step_threshold=(),
            fixed_last_k=None,
            refine_beta=float(cli_args.refine_beta),
            refine_sigma_eps=float(cli_args.refine_sigma_eps),
            step_mu=(),
            step_sigma=(),
            chosen_t0s=shared_test_t0s,
            generation_seed_base=int(generation_seed_base),
            metrics_seed=int(metrics_seed),
        )
        candidate_rows = [euler_row, heun_row]

        adaptive_rows = []
        fixed_rows = []
        for percentile in percentile_values:
            adaptive_rows.append(
                _run_variant(
                    model=model,
                    ds_test=splits["test"],
                    cfg=cfg,
                    eval_horizon=eval_horizon,
                    eval_windows_test=int(cli_args.eval_windows_test),
                    budget_family=int(budget_family),
                    solver_family="euler_refine_half_percentile",
                    solver_override="euler_refine_half",
                    macro_steps=int(macro_steps_adaptive),
                    trigger_mode="raw_step",
                    trigger_percentile=float(percentile),
                    trigger_threshold_z=None,
                    trigger_threshold_raw=None,
                    step_threshold=_step_thresholds_from_percentile(step_rows, int(macro_steps_adaptive), float(percentile)),
                    fixed_last_k=None,
                    refine_beta=float(cli_args.refine_beta),
                    refine_sigma_eps=float(cli_args.refine_sigma_eps),
                    step_mu=step_mu,
                    step_sigma=step_sigma,
                    chosen_t0s=shared_test_t0s,
                    generation_seed_base=int(generation_seed_base),
                    metrics_seed=int(metrics_seed),
                )
            )
        for fixed_last_k in fixed_last_k_values:
            fixed_rows.append(
                _run_variant(
                    model=model,
                    ds_test=splits["test"],
                    cfg=cfg,
                    eval_horizon=eval_horizon,
                    eval_windows_test=int(cli_args.eval_windows_test),
                    budget_family=int(budget_family),
                    solver_family="euler_refine_half_fixed_late",
                    solver_override="euler_refine_half",
                    macro_steps=int(macro_steps_adaptive),
                    trigger_mode="fixed_last_k",
                    trigger_percentile=None,
                    trigger_threshold_z=None,
                    trigger_threshold_raw=None,
                    step_threshold=(),
                    fixed_last_k=int(fixed_last_k),
                    refine_beta=float(cli_args.refine_beta),
                    refine_sigma_eps=float(cli_args.refine_sigma_eps),
                    step_mu=step_mu,
                    step_sigma=step_sigma,
                    chosen_t0s=shared_test_t0s,
                    generation_seed_base=int(generation_seed_base),
                    metrics_seed=int(metrics_seed),
                )
            )
        candidate_rows.extend(adaptive_rows)
        candidate_rows.extend(fixed_rows)
        stage_b_all_rows.extend(candidate_rows)

        selected_adaptive = _select_budget_matched(adaptive_rows, int(budget_family))
        selected_fixed = _select_budget_matched(fixed_rows, int(budget_family))
        comparison_rows = [euler_row, heun_row]
        if selected_adaptive is not None:
            comparison_rows.append(dict(selected_adaptive))
        if selected_fixed is not None:
            comparison_rows.append(dict(selected_fixed))
        winner = _pick_winner(comparison_rows)
        selected_rows = []
        for row in comparison_rows:
            tagged = dict(row)
            tagged["selected_for_budget_comparison"] = True
            tagged["winner_for_budget_family"] = bool(winner is not None and row is winner)
            selected_rows.append(tagged)
        stage_b_selected_rows.extend(selected_rows)

        acceptance_rows.append(
            _diagnose_budget(
                budget_family=int(budget_family),
                calibration=payload,
                baseline=euler_row,
                heun=heun_row,
                adaptive_rows=adaptive_rows,
                fixed_rows=fixed_rows,
            )
        )

    _plot_latency_vs_score(os.path.join(cli_args.out_root, "stage_b_latency_vs_score.png"), stage_b_selected_rows)
    _plot_field_evals_vs_score(os.path.join(cli_args.out_root, "stage_b_field_evals_vs_score.png"), stage_b_selected_rows)
    _plot_trigger_rate_by_step(os.path.join(cli_args.out_root, "stage_b_trigger_rate_by_step.png"), stage_b_all_rows)
    _plot_realized_budget(os.path.join(cli_args.out_root, "stage_b_realized_budget.png"), stage_b_selected_rows)

    stage_b_summary = {
        "dataset": "cryptos",
        "seed": seed,
        "eval_horizon": eval_horizon,
        "evaluation_protocol": evaluation_protocol,
        "budget_families": [int(x) for x in budget_families],
        "percentile_values": [float(x) for x in percentile_values],
        "fixed_last_k_values": [int(x) for x in fixed_last_k_values],
        "rows": stage_b_all_rows,
        "selected_rows": stage_b_selected_rows,
        "acceptance": acceptance_rows,
    }
    save_json(stage_b_summary, os.path.join(cli_args.out_root, "stage_b_summary.json"))


if __name__ == "__main__":
    main()
