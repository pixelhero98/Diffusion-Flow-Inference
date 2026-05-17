#!/usr/bin/env python3
"""Adaptive stochastic corrector sampler pilot for OTFlow."""

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

from benchmark_otflow_suite import DATASET_PLANS
from experiment_common import build_cfg_from_args, build_dataset_splits, get_otflow_dataset_preset, mkdir
from otflow_train_val import (
    _get_dataset_item_by_t,
    _parse_batch,
    crop_history_window,
    eval_many_windows,
    resolve_context_length,
    select_eval_window_starts,
    save_json,
    seed_all,
    train_loop,
)


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


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the adaptive-noise sampler pilot on cryptos.")
    ap.add_argument("--dataset", type=str, default="cryptos", choices=("cryptos",))
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--out_root", type=str, default="results_adaptive_noise_sampler_followup_cryptos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_horizon", type=int, default=200)
    ap.add_argument("--eval_windows_val", type=int, default=30)
    ap.add_argument("--eval_windows_test", type=int, default=30)
    ap.add_argument("--diag_windows", type=int, default=8)
    ap.add_argument("--nfe_list", type=str, default="8,16")
    ap.add_argument("--gamma_values", type=str, default="0.01,0.03,0.05,0.08")
    ap.add_argument("--fixed_gamma", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--tau_quantile", type=float, default=0.85)
    ap.add_argument("--kappa", type=float, default=12.0)
    ap.add_argument("--disable_noise_frac", type=float, default=0.1)
    ap.add_argument("--cooldown_values", type=str, default="1,2")
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


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    if metric == "score_main":
        return float(result["cmp"]["score_main"]["mean"])
    if metric in PRIMARY_METRICS:
        return float(result["cmp"]["main"][metric]["mean"])
    return float(result["cmp"]["extra"][metric]["mean"])


def _make_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    plan = DATASET_PLANS["cryptos"]
    preset = get_otflow_dataset_preset("cryptos", variant="quality")
    return argparse.Namespace(
        dataset="cryptos",
        data_path=str(cli_args.cryptos_path),
        synthetic_length=int(plan.synthetic_length),
        seed=int(cli_args.dataset_seed),
        device=str(cli_args.device),
        train_frac=plan.train_frac,
        val_frac=plan.val_frac,
        test_frac=plan.test_frac,
        stride_train=plan.stride_train,
        stride_eval=plan.stride_eval,
        levels=int(preset["levels"]),
        history_len=int(preset["history_len"]),
        batch_size=int(plan.batch_size),
        lr=float(cli_args.lr),
        weight_decay=float(cli_args.weight_decay),
        grad_clip=float(cli_args.grad_clip),
        standardize=True,
        use_cond_features=False,
        cond_standardize=True,
        hidden_dim=int(cli_args.hidden_dim),
        otflow_variant="quality",
        ctx_encoder=str(preset["ctx_encoder"]),
        ctx_causal=bool(preset["ctx_causal"]),
        ctx_local_kernel=int(preset["ctx_local_kernel"]),
        ctx_pool_scales=str(preset["ctx_pool_scales"]),
        fu_net_type="transformer",
        fu_net_layers=int(cli_args.fu_net_layers),
        fu_net_heads=int(cli_args.fu_net_heads),
        adaptive_context=False,
        adaptive_context_ratio=None,
        adaptive_context_min=None,
        adaptive_context_max=None,
        train_variable_context=False,
        train_context_min=None,
        train_context_max=None,
        use_minibatch_ot=True,
    )


def _save_checkpoint(path: str, model: torch.nn.Module, cfg, ds_train) -> None:
    state = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": cfg.to_dict(),
        "params_mean": None if ds_train.params_mean is None else np.asarray(ds_train.params_mean, dtype=np.float32),
        "params_std": None if ds_train.params_std is None else np.asarray(ds_train.params_std, dtype=np.float32),
    }
    torch.save(state, path)


def _choose_valid_windows(ds, horizon: int, n_windows: int, seed: int) -> np.ndarray:
    return select_eval_window_starts(ds, horizon=int(horizon), n_windows=int(n_windows), seed=int(seed))


def _sample_cfg_snapshot(cfg) -> Dict[str, Any]:
    return dict(cfg.to_dict()["sample"])


def _apply_sample_overrides(model: torch.nn.Module, cfg, **overrides: Any) -> Dict[str, Any]:
    backup = _sample_cfg_snapshot(cfg)
    clean = {key: value for key, value in overrides.items() if value is not None}
    if clean:
        cfg.apply_overrides(**clean)
        if getattr(model, "cfg", None) is not cfg:
            model.cfg.apply_overrides(**clean)
    return backup


def _restore_sample_overrides(model: torch.nn.Module, cfg, backup: Mapping[str, Any]) -> None:
    cfg.apply_overrides(**dict(backup))
    if getattr(model, "cfg", None) is not cfg:
        model.cfg.apply_overrides(**dict(backup))


def _one_step_inputs(ds, cfg, t0: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    batch = _get_dataset_item_by_t(ds, int(t0))
    hist, _, _, _, _ = _parse_batch(batch)
    hist_t = hist[None, :, :].to(cfg.device).float()
    cond_t = None
    if ds.cond is not None:
        cond_t = torch.from_numpy(ds.cond[int(t0)]).to(cfg.device).float()[None, :]
    return hist_t, cond_t


def _collect_calibration(model, ds_val, cfg, *, horizon: int, nfe: int, n_windows: int, seed: int, tau_quantile: float) -> Dict[str, Any]:
    chosen = _choose_valid_windows(ds_val, horizon=horizon, n_windows=n_windows, seed=seed)
    rows: List[Dict[str, Any]] = []
    disagreements = []
    velocity_norms = []

    for window_idx, t0 in enumerate(chosen.tolist()):
        hist_t, cond_t = _one_step_inputs(ds_val, cfg, int(t0))
        _, trace = model.sample_trace(hist_t, cond=cond_t, steps=int(nfe), solver="euler")
        d = trace["disagreement"][0].numpy()
        v = trace["velocity_norm"][0].numpy()
        for step_idx in range(int(nfe)):
            row = {
                "window_index": int(window_idx),
                "t0": int(t0),
                "step_index": int(step_idx),
                "time": float(trace["time"][step_idx].item()),
                "disagreement": float(d[step_idx]),
                "velocity_norm": float(v[step_idx]),
            }
            rows.append(row)
            if step_idx > 0:
                disagreements.append(float(d[step_idx]))
                velocity_norms.append(float(v[step_idx]))

    d_arr = np.asarray(disagreements, dtype=np.float64)
    v_arr = np.asarray(velocity_norms, dtype=np.float64)
    tau = float(np.quantile(d_arr, tau_quantile)) if d_arr.size > 0 else 0.0
    return {
        "nfe": int(nfe),
        "tau": tau,
        "tau_quantile": float(tau_quantile),
        "exclude_step0_for_tau": True,
        "rows": rows,
        "disagreement_stats": {
            "mean": float(np.mean(d_arr)) if d_arr.size > 0 else float("nan"),
            "std": float(np.std(d_arr)) if d_arr.size > 0 else float("nan"),
            "p50": float(np.quantile(d_arr, 0.50)) if d_arr.size > 0 else float("nan"),
            "p85": float(np.quantile(d_arr, 0.85)) if d_arr.size > 0 else float("nan"),
            "p95": float(np.quantile(d_arr, 0.95)) if d_arr.size > 0 else float("nan"),
        },
        "velocity_norm_stats": {
            "mean": float(np.mean(v_arr)) if v_arr.size > 0 else float("nan"),
            "std": float(np.std(v_arr)) if v_arr.size > 0 else float("nan"),
        },
    }


def _collect_rollout_diagnostics(
    model,
    ds,
    cfg,
    *,
    horizon: int,
    nfe: int,
    n_windows: int,
    seed: int,
) -> Dict[str, Any]:
    chosen = _choose_valid_windows(ds, horizon=horizon, n_windows=n_windows, seed=seed)
    fire_rows = []
    gamma_rows = []
    disagreement_rows = []
    noise_rows = []

    for t0 in chosen.tolist():
        batch = _get_dataset_item_by_t(ds, int(t0))
        hist, _, _, _, _ = _parse_batch(batch)
        hist_t = hist[None, :, :].to(cfg.device).float()
        context_len = resolve_context_length(hist_t.shape[1], horizon=horizon, cfg=cfg)
        cond_seq = None
        if ds.cond is not None:
            cond_seq = torch.from_numpy(ds.cond[int(t0) : int(t0) + int(horizon)]).to(cfg.device).float()[None, :, :]

        x_hist = crop_history_window(hist_t, context_len).clone()
        for step_idx in range(int(horizon)):
            cond_t = cond_seq[:, step_idx, :] if cond_seq is not None else None
            x_next, trace = model.sample_trace(x_hist, cond=cond_t, steps=int(nfe))
            fire_rows.append(trace["fired"].to(dtype=torch.float32).numpy()[0])
            gamma_rows.append(trace["gamma"].numpy()[0])
            disagreement_rows.append(trace["disagreement"].numpy()[0])
            noise_rows.append(trace["noise_norm"].numpy()[0])
            x_hist = torch.cat([x_hist, x_next[:, None, :]], dim=1)
            x_hist = crop_history_window(x_hist, context_len)

    fired = np.asarray(fire_rows, dtype=np.float32)
    gamma = np.asarray(gamma_rows, dtype=np.float32)
    disagreement = np.asarray(disagreement_rows, dtype=np.float32)
    noise_norm = np.asarray(noise_rows, dtype=np.float32)
    gate = np.asarray(trace["time_gate"].numpy(), dtype=np.float32)
    active = gate[None, :] > 0.0
    active_broadcast = np.broadcast_to(active, fired.shape)
    fire_rate_active = float(fired[active_broadcast].mean()) if np.any(active_broadcast) else 0.0

    return {
        "n_rollout_calls": int(fired.shape[0]),
        "solver_steps": int(nfe),
        "fire_rate_active": fire_rate_active,
        "mean_gamma_active": float(gamma[active_broadcast].mean()) if np.any(active_broadcast) else 0.0,
        "mean_disagreement": float(np.mean(disagreement)),
        "p95_disagreement": float(np.quantile(disagreement, 0.95)),
        "mean_noise_norm": float(np.mean(noise_norm)),
        "fire_by_step": [float(x) for x in fired.mean(axis=0)],
        "gamma_by_step": [float(x) for x in gamma.mean(axis=0)],
        "disagreement_by_step": [float(x) for x in disagreement.mean(axis=0)],
        "active_gate_by_step": [bool(x) for x in gate > 0.0],
    }


def _variant_label(row: Mapping[str, Any]) -> str:
    family = str(row["family"])
    if family == "baseline":
        return f"baseline-nfe{row['nfe']}"
    return f"{family}-{row['noise_mode']}-g{row['gamma_max']:.2f}-nfe{row['nfe']}"


def _select_best(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not rows:
        return None
    best_score = min(float(row["score_main"]) for row in rows)
    band = [row for row in rows if float(row["score_main"]) <= best_score * 1.01]
    return sorted(
        band,
        key=lambda row: (
            float(row["conditional_w1"]),
            float(row.get("fire_rate_active", 0.0)),
            float(row["efficiency_ms_per_sample"]),
        ),
    )[0]


def _pass_stage_c_gate(best_adaptive: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    score_gain = (float(baseline["score_main"]) - float(best_adaptive["score_main"])) / max(float(baseline["score_main"]), 1e-12)
    u_gain = (float(baseline["unconditional_w1"]) - float(best_adaptive["unconditional_w1"])) / max(float(baseline["unconditional_w1"]), 1e-12)
    c_gain = (float(baseline["conditional_w1"]) - float(best_adaptive["conditional_w1"])) / max(float(baseline["conditional_w1"]), 1e-12)
    passed = bool(score_gain >= 0.01 or (u_gain >= 0.03 and c_gain >= 0.03))
    return {
        "passed": passed,
        "score_gain_frac": float(score_gain),
        "unconditional_w1_gain_frac": float(u_gain),
        "conditional_w1_gain_frac": float(c_gain),
    }


def _plot_calibration(path: str, calibration: Mapping[int, Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for nfe, payload in sorted(calibration.items()):
        vals = np.asarray(
            [float(row["disagreement"]) for row in payload["rows"] if int(row["step_index"]) > 0],
            dtype=np.float64,
        )
        tau = float(payload["tau"])
        axes[0].hist(vals, bins=30, alpha=0.45, label=f"NFE={nfe}")
        axes[0].axvline(tau, linestyle="--", linewidth=1.5)
        q_grid = np.linspace(0.0, 1.0, 101)
        axes[1].plot(q_grid, np.quantile(vals, q_grid), linewidth=2.0, label=f"NFE={nfe}")
        axes[1].axhline(tau, linestyle="--", linewidth=1.0)
    axes[0].set_title("Disagreement histogram")
    axes[0].set_xlabel("1 - cosine(v, EMA(v))")
    axes[0].set_ylabel("count")
    axes[1].set_title("Disagreement percentile curve")
    axes[1].set_xlabel("quantile")
    axes[1].set_ylabel("disagreement")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_fire_rate_by_step(path: str, rows: Sequence[Mapping[str, Any]], nfe_list: Sequence[int]) -> None:
    if not _HAS_PLOT:
        return
    fig, axes = plt.subplots(1, len(nfe_list), figsize=(6.2 * len(nfe_list), 4.2), squeeze=False)
    for ax, nfe in zip(axes[0], nfe_list):
        subset = [row for row in rows if int(row["nfe"]) == int(nfe) and str(row["family"]) == "adaptive"]
        for row in subset:
            label = f"{row['noise_mode']}, g={row['gamma_max']:.2f}"
            ax.plot(range(int(nfe)), row["diag"]["fire_by_step"], marker="o", linewidth=1.8, label=label)
        ax.set_title(f"Adaptive fire rate by solver step (NFE={nfe})")
        ax.set_xlabel("solver step")
        ax.set_ylabel("fire rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.2)
        if subset:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_score_vs_gamma(path: str, rows: Sequence[Mapping[str, Any]], nfe_list: Sequence[int]) -> None:
    if not _HAS_PLOT:
        return
    fig, axes = plt.subplots(1, len(nfe_list), figsize=(6.2 * len(nfe_list), 4.2), squeeze=False)
    for ax, nfe in zip(axes[0], nfe_list):
        subset = [row for row in rows if int(row["nfe"]) == int(nfe)]
        base = next((row for row in subset if str(row["family"]) == "baseline"), None)
        if base is not None:
            ax.axhline(float(base["score_main"]), color="black", linestyle="--", linewidth=1.4, label="baseline")
        for family, linestyle in (("adaptive", "-"), ("fixed", ":")):
            for noise_mode, color in (("orthogonal", "#2e8b57"), ("isotropic", "#c03d3d")):
                rows_fg = [row for row in subset if str(row["family"]) == family and str(row["noise_mode"]) == noise_mode]
                if not rows_fg:
                    continue
                rows_fg = sorted(rows_fg, key=lambda row: float(row["gamma_max"]))
                ax.plot(
                    [float(row["gamma_max"]) for row in rows_fg],
                    [float(row["score_main"]) for row in rows_fg],
                    marker="o",
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.8,
                    label=f"{family}-{noise_mode}",
                )
        ax.set_title(f"score_main vs gamma (NFE={nfe})")
        ax.set_xlabel("gamma_max")
        ax.set_ylabel("score_main")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_vs_score(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    family_style = {
        "baseline": ("black", "o"),
        "fixed": ("#c03d3d", "s"),
        "adaptive": ("#2e8b57", "^"),
    }
    for row in rows:
        color, marker = family_style[str(row["family"])]
        ax.scatter(
            float(row["efficiency_ms_per_sample"]),
            float(row["score_main"]),
            color=color,
            marker=marker,
            s=70,
            alpha=0.9,
        )
        ax.annotate(_variant_label(row), (float(row["efficiency_ms_per_sample"]), float(row["score_main"])), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Latency vs score_main")
    ax.set_xlabel("efficiency_ms_per_sample")
    ax.set_ylabel("score_main")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli_args = build_argparser().parse_args()
    seed_all(int(cli_args.seed))
    mkdir(cli_args.out_root)

    args = _make_args(cli_args)
    cfg = build_cfg_from_args(args)
    preset = get_otflow_dataset_preset("cryptos", variant="quality")
    cfg.apply_overrides(
        adaptive_beta=float(cli_args.beta),
        adaptive_kappa=float(cli_args.kappa),
        adaptive_disable_noise_frac=float(cli_args.disable_noise_frac),
    )

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
    nfe_list = _parse_ints(cli_args.nfe_list)
    gamma_values = _parse_floats(cli_args.gamma_values)
    cooldown_values = _parse_ints(cli_args.cooldown_values)

    calibration: Dict[int, Dict[str, Any]] = {}
    for nfe in nfe_list:
        calibration[int(nfe)] = _collect_calibration(
            model,
            splits["val"],
            cfg,
            horizon=eval_horizon,
            nfe=int(nfe),
            n_windows=int(cli_args.eval_windows_val),
            seed=int(cli_args.seed) + 17 + int(nfe),
            tau_quantile=float(cli_args.tau_quantile),
        )
    trace_calibration = {
        "dataset": "cryptos",
        "seed": int(cli_args.seed),
        "eval_horizon": eval_horizon,
        "calibration": calibration,
    }
    save_json(trace_calibration, os.path.join(cli_args.out_root, "trace_calibration.json"))
    _plot_calibration(os.path.join(cli_args.out_root, "trace_calibration.png"), calibration)

    stage_b_rows: List[Dict[str, Any]] = []
    for nfe in nfe_list:
        tau = float(calibration[int(nfe)]["tau"])
        variants = [
            {
                "family": "baseline",
                "noise_mode": "none",
                "trigger_mode": "none",
                "gamma_max": 0.0,
                "cooldown_steps": 0,
                "solver": "euler",
            },
            {
                "family": "fixed",
                "noise_mode": "isotropic",
                "trigger_mode": "always",
                "gamma_max": float(cli_args.fixed_gamma),
                "cooldown_steps": 0,
                "solver": "euler_adaptive",
            },
            {
                "family": "fixed",
                "noise_mode": "orthogonal",
                "trigger_mode": "always",
                "gamma_max": float(cli_args.fixed_gamma),
                "cooldown_steps": 0,
                "solver": "euler_adaptive",
            },
        ]
        for gamma in gamma_values:
            variants.append(
                {
                    "family": "adaptive",
                    "noise_mode": "isotropic",
                    "trigger_mode": "adaptive",
                    "gamma_max": float(gamma),
                    "cooldown_steps": 0,
                    "solver": "euler_adaptive",
                }
            )
            variants.append(
                {
                    "family": "adaptive",
                    "noise_mode": "orthogonal",
                    "trigger_mode": "adaptive",
                    "gamma_max": float(gamma),
                    "cooldown_steps": 0,
                    "solver": "euler_adaptive",
                }
            )

        for variant in variants:
            backup = _apply_sample_overrides(
                model,
                cfg,
                solver=str(variant["solver"]),
                adaptive_tau=tau,
                adaptive_gamma_max=float(variant["gamma_max"]),
                adaptive_noise_mode=str(variant["noise_mode"]),
                adaptive_trigger_mode=str(variant["trigger_mode"]),
                adaptive_cooldown_steps=int(variant["cooldown_steps"]),
                adaptive_beta=float(cli_args.beta),
                adaptive_kappa=float(cli_args.kappa),
                adaptive_disable_noise_frac=float(cli_args.disable_noise_frac),
            )
            try:
                t0 = time.time()
                result = eval_many_windows(
                    splits["test"],
                    model,
                    cfg,
                    horizon=eval_horizon,
                    nfe=int(nfe),
                    n_windows=int(cli_args.eval_windows_test),
                    seed=int(cli_args.seed) + 101 + int(nfe),
                    horizons_eval=[int(eval_horizon)],
                )
                eval_seconds = float(time.time() - t0)
                diag = _collect_rollout_diagnostics(
                    model,
                    splits["test"],
                    cfg,
                    horizon=eval_horizon,
                    nfe=int(nfe),
                    n_windows=int(cli_args.diag_windows),
                    seed=int(cli_args.seed) + 701 + int(nfe),
                )
            finally:
                _restore_sample_overrides(model, cfg, backup)

            row = {
                "nfe": int(nfe),
                "tau": float(tau),
                "family": str(variant["family"]),
                "noise_mode": str(variant["noise_mode"]),
                "trigger_mode": str(variant["trigger_mode"]),
                "gamma_max": float(variant["gamma_max"]),
                "cooldown_steps": int(variant["cooldown_steps"]),
                "solver": str(variant["solver"]),
                "eval_seconds": float(eval_seconds),
                "diag": diag,
            }
            for metric in ALL_METRICS:
                row[metric] = _metric_value(result, metric)
            row["variant_label"] = _variant_label(row)
            row["fire_rate_active"] = float(diag["fire_rate_active"])
            stage_b_rows.append(row)

    baselines_by_nfe = {
        int(nfe): next(row for row in stage_b_rows if int(row["nfe"]) == int(nfe) and str(row["family"]) == "baseline")
        for nfe in nfe_list
    }
    best_overall = _select_best(stage_b_rows)
    adaptive_rows = [row for row in stage_b_rows if str(row["family"]) == "adaptive"]
    best_adaptive = _select_best(adaptive_rows)
    gate = None
    if best_adaptive is not None:
        gate = _pass_stage_c_gate(best_adaptive, baselines_by_nfe[int(best_adaptive["nfe"])])
    acceptance = {
        "disagreement_not_flat": bool(
            any(float(calibration[int(nfe)]["disagreement_stats"]["std"]) > 1e-6 for nfe in nfe_list)
        ),
        "best_adaptive_fire_rate_sparse": bool(
            best_adaptive is not None and 0.05 <= float(best_adaptive["fire_rate_active"]) <= 0.40
        ),
        "best_adaptive_not_dominated_by_fixed": bool(
            best_adaptive is not None
            and float(best_adaptive["score_main"])
            <= min(
                float(row["score_main"])
                for row in stage_b_rows
                if int(row["nfe"]) == int(best_adaptive["nfe"]) and str(row["family"]) in {"fixed", "adaptive"}
            )
        ),
    }
    stage_b_summary = {
        "dataset": "cryptos",
        "seed": int(cli_args.seed),
        "eval_horizon": eval_horizon,
        "nfe_list": [int(x) for x in nfe_list],
        "rows": stage_b_rows,
        "best_overall": best_overall,
        "best_adaptive": best_adaptive,
        "baselines_by_nfe": baselines_by_nfe,
        "gate": gate,
        "acceptance": acceptance,
    }
    save_json(stage_b_summary, os.path.join(cli_args.out_root, "stage_b_summary.json"))
    _plot_fire_rate_by_step(os.path.join(cli_args.out_root, "stage_b_fire_rate_by_step.png"), stage_b_rows, nfe_list)
    _plot_score_vs_gamma(os.path.join(cli_args.out_root, "stage_b_score_vs_gamma.png"), stage_b_rows, nfe_list)
    _plot_latency_vs_score(os.path.join(cli_args.out_root, "stage_b_latency_vs_score.png"), stage_b_rows)

    if gate is None or not bool(gate["passed"]):
        return

    stage_c_rows: List[Dict[str, Any]] = []
    assert best_adaptive is not None
    for cooldown in cooldown_values:
        backup = _apply_sample_overrides(
            model,
            cfg,
            solver="euler_adaptive",
            adaptive_tau=float(best_adaptive["tau"]),
            adaptive_gamma_max=float(best_adaptive["gamma_max"]),
            adaptive_noise_mode=str(best_adaptive["noise_mode"]),
            adaptive_trigger_mode="adaptive",
            adaptive_cooldown_steps=int(cooldown),
            adaptive_beta=float(cli_args.beta),
            adaptive_kappa=float(cli_args.kappa),
            adaptive_disable_noise_frac=float(cli_args.disable_noise_frac),
        )
        try:
            t0 = time.time()
            result = eval_many_windows(
                splits["test"],
                model,
                cfg,
                horizon=eval_horizon,
                nfe=int(best_adaptive["nfe"]),
                n_windows=int(cli_args.eval_windows_test),
                seed=int(cli_args.seed) + 1301 + int(cooldown),
                horizons_eval=[int(eval_horizon)],
            )
            eval_seconds = float(time.time() - t0)
            diag = _collect_rollout_diagnostics(
                model,
                splits["test"],
                cfg,
                horizon=eval_horizon,
                nfe=int(best_adaptive["nfe"]),
                n_windows=int(cli_args.diag_windows),
                seed=int(cli_args.seed) + 1701 + int(cooldown),
            )
        finally:
            _restore_sample_overrides(model, cfg, backup)

        row = {
            "nfe": int(best_adaptive["nfe"]),
            "tau": float(best_adaptive["tau"]),
            "family": "adaptive_cooldown",
            "noise_mode": str(best_adaptive["noise_mode"]),
            "trigger_mode": "adaptive",
            "gamma_max": float(best_adaptive["gamma_max"]),
            "cooldown_steps": int(cooldown),
            "solver": "euler_adaptive",
            "eval_seconds": float(eval_seconds),
            "diag": diag,
        }
        for metric in ALL_METRICS:
            row[metric] = _metric_value(result, metric)
        row["variant_label"] = f"adaptive-{row['noise_mode']}-g{row['gamma_max']:.2f}-cd{cooldown}-nfe{row['nfe']}"
        row["fire_rate_active"] = float(diag["fire_rate_active"])
        stage_c_rows.append(row)

    stage_c_summary = {
        "dataset": "cryptos",
        "seed": int(cli_args.seed),
        "base_adaptive_row": best_adaptive,
        "rows": stage_c_rows,
        "best_row": _select_best(stage_c_rows),
    }
    save_json(stage_c_summary, os.path.join(cli_args.out_root, "stage_c_summary.json"))


if __name__ == "__main__":
    main()
