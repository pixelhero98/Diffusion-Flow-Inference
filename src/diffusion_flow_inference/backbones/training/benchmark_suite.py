#!/usr/bin/env python3
"""Tune and benchmark the current OTFlow baseline across the main datasets.

This runner implements the confirmed plan:
- datasets: synthetic, optiver, cryptos, es_mbp_10
- use the current winning dataset-specific OTFlow defaults
- include `NFE=1` speed variants for the datasets whose quality default uses
  more than one sampling step
- evaluate the selected configuration on the confirmed horizon triplet
- aggregate the 4 primary metrics + 7 extra metrics across multiple seeds
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from diffusion_flow_inference.datasets.experiment_common import (
    build_cfg_from_args,
    build_dataset_splits,
    get_otflow_dataset_preset,
    mkdir,
    parse_int_list,
)
from diffusion_flow_inference.datasets.medical_datasets import SLEEP_EDF_DATASET_KEY, default_sleep_edf_data_path
from diffusion_flow_inference.backbones.training.train_val import eval_many_windows, save_json, seed_all, train_loop


PRIMARY_METRICS = (
    "score_main",
    "tstr_macro_f1",
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


@dataclass(frozen=True)
class DatasetPlan:
    name: str
    dataset: str
    levels: int
    horizons: Tuple[int, int, int]
    history_options: Tuple[int, ...]
    nfe_options: Tuple[int, ...]
    train_steps_tune: int
    train_steps_final: int
    eval_windows_tune: int
    eval_windows_final: int
    data_path: str = ""
    synthetic_length: int = 2_000_000
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2
    stride_train: int = 1
    stride_eval: int = 1
    batch_size: int = 64


DATASET_PLANS: Mapping[str, DatasetPlan] = {
    "synthetic": DatasetPlan(
        name="synthetic",
        dataset="synthetic",
        levels=10,
        horizons=(32, 128, 512),
        history_options=(128,),
        nfe_options=(1, 2),
        train_steps_tune=4000,
        train_steps_final=12000,
        eval_windows_tune=12,
        eval_windows_final=24,
    ),
    "optiver": DatasetPlan(
        name="optiver",
        dataset="optiver",
        levels=2,
        horizons=(10, 30, 60),
        history_options=(128,),
        nfe_options=(1, 4),
        train_steps_tune=4000,
        train_steps_final=8000,
        eval_windows_tune=12,
        eval_windows_final=24,
    ),
    "cryptos": DatasetPlan(
        name="cryptos",
        dataset="cryptos",
        levels=10,
        horizons=(60, 300, 900),
        history_options=(256,),
        nfe_options=(1, 2),
        train_steps_tune=6000,
        train_steps_final=12000,
        eval_windows_tune=10,
        eval_windows_final=20,
    ),
    "es_mbp_10": DatasetPlan(
        name="es_mbp_10",
        dataset="es_mbp_10",
        levels=10,
        horizons=(60, 300, 900),
        history_options=(256,),
        nfe_options=(1,),
        train_steps_tune=6000,
        train_steps_final=12000,
        eval_windows_tune=10,
        eval_windows_final=20,
    ),
    SLEEP_EDF_DATASET_KEY: DatasetPlan(
        name=SLEEP_EDF_DATASET_KEY,
        dataset=SLEEP_EDF_DATASET_KEY,
        levels=1,
        horizons=(1500, 1500, 1500),
        history_options=(6000,),
        nfe_options=(1,),
        train_steps_tune=4000,
        train_steps_final=12000,
        eval_windows_tune=6,
        eval_windows_final=12,
        data_path=default_sleep_edf_data_path(),
        stride_train=3000,
        stride_eval=3000,
        batch_size=2,
    ),
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Tune and benchmark the current OTFlow baseline.")
    ap.add_argument("--datasets", type=str, default=f"synthetic,optiver,cryptos,es_mbp_10,{SLEEP_EDF_DATASET_KEY}")
    ap.add_argument("--out_root", type=str, default="results_benchmark_otflow_suite")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=str, default="0,1,2", help="Model/training seeds.")
    ap.add_argument("--dataset_seed", type=int, default=0, help="Synthetic data seed; fixed across model seeds.")
    ap.add_argument("--tune_seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--ctx_encoder", type=str, default=None)
    ap.add_argument("--ctx_causal", action="store_true", default=None)
    ap.add_argument("--no-ctx_causal", dest="ctx_causal", action="store_false")
    ap.add_argument("--ctx_local_kernel", type=int, default=None)
    ap.add_argument("--ctx_pool_scales", type=str, default=None)
    ap.add_argument("--solver", type=str, default=None, choices=["euler", "dpmpp2m"])
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--synthetic_length", type=int, default=2_000_000)
    ap.add_argument("--optiver_path", type=str, default="")
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--sleep_edf_path", type=str, default=default_sleep_edf_data_path())
    ap.add_argument("--skip_tuning", action="store_true", default=False)
    ap.add_argument("--final_only", action="store_true", default=False, help="Reuse existing tuned_config.json per dataset.")
    return ap


def _parse_dataset_list(text: str) -> List[str]:
    datasets = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in datasets if name not in DATASET_PLANS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    return datasets


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    if metric == "score_main":
        return float(result["cmp"]["score_main"]["mean"])
    if metric in PRIMARY_METRICS:
        return float(result["cmp"]["main"][metric]["mean"])
    return float(result["cmp"]["extra"][metric]["mean"])


def _aggregate_values(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "mean": float(np.mean(finite)) if finite.size > 0 else float("nan"),
        "std": float(np.std(finite)) if finite.size > 0 else float("nan"),
        "n": int(arr.size),
        "n_valid": int(finite.size),
    }


def _make_base_args(cli_args: argparse.Namespace, plan: DatasetPlan, history_len: int) -> argparse.Namespace:
    preset = get_otflow_dataset_preset(plan.dataset, variant="quality")
    data_path = plan.data_path
    if plan.name == "optiver" and cli_args.optiver_path:
        data_path = cli_args.optiver_path
    elif plan.name == "cryptos" and cli_args.cryptos_path:
        data_path = cli_args.cryptos_path
    elif plan.name == "es_mbp_10" and cli_args.es_path:
        data_path = cli_args.es_path
    elif plan.name == SLEEP_EDF_DATASET_KEY and cli_args.sleep_edf_path:
        data_path = cli_args.sleep_edf_path

    return argparse.Namespace(
        dataset=plan.dataset,
        data_path=data_path,
        synthetic_length=int(cli_args.synthetic_length if plan.dataset == "synthetic" else plan.synthetic_length),
        seed=int(cli_args.dataset_seed),
        device=cli_args.device,
        train_frac=plan.train_frac,
        val_frac=plan.val_frac,
        test_frac=plan.test_frac,
        stride_train=plan.stride_train,
        stride_eval=plan.stride_eval,
        levels=plan.levels,
        history_len=int(history_len),
        batch_size=plan.batch_size,
        lr=float(cli_args.lr),
        weight_decay=float(cli_args.weight_decay),
        grad_clip=float(cli_args.grad_clip),
        standardize=True,
        use_cond_features=False,
        cond_standardize=True,
        hidden_dim=int(cli_args.hidden_dim),
        ctx_encoder=str(cli_args.ctx_encoder if cli_args.ctx_encoder is not None else preset["ctx_encoder"]),
        ctx_causal=bool(cli_args.ctx_causal if cli_args.ctx_causal is not None else preset["ctx_causal"]),
        ctx_local_kernel=int(cli_args.ctx_local_kernel if cli_args.ctx_local_kernel is not None else preset["ctx_local_kernel"]),
        ctx_pool_scales=str(cli_args.ctx_pool_scales if cli_args.ctx_pool_scales is not None else preset["ctx_pool_scales"]),
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
        cond_depths="",
        cond_vol_window=None,
        cfg_scale=1.0,
        solver=str(cli_args.solver if cli_args.solver is not None else preset["solver"]),
    )


def _build_cfg_and_splits(cli_args: argparse.Namespace, plan: DatasetPlan, history_len: int):
    args = _make_base_args(cli_args, plan, history_len)
    cfg = build_cfg_from_args(args)
    splits = build_dataset_splits(args, cfg)
    return cfg, splits


def _evaluate_horizon_set(
    ds,
    model,
    cfg,
    *,
    horizons: Sequence[int],
    nfe: int,
    n_windows: int,
    seed_base: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for horizon in horizons:
        out[str(int(horizon))] = eval_many_windows(
            ds,
            model,
            cfg,
            horizon=int(horizon),
            nfe=int(nfe),
            n_windows=int(n_windows),
            seed=int(seed_base + 1000 * int(horizon)),
            horizons_eval=horizons,
        )
    return out


def _score_across_horizons(results_by_horizon: Mapping[str, Mapping[str, Any]]) -> float:
    vals = np.asarray([float(payload["cmp"]["score_main"]["mean"]) for payload in results_by_horizon.values()], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    return float(np.mean(finite)) if finite.size > 0 else float("nan")


def _summarize_seed_runs(seed_runs: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> Dict[str, Any]:
    horizon_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for horizon in horizons:
        h_key = str(int(horizon))
        metric_values: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
        for run in seed_runs:
            payload = run["results"][h_key]
            for metric in ALL_METRICS:
                metric_values[metric].append(_metric_value(payload, metric))
        horizon_summary[h_key] = {metric: _aggregate_values(vals) for metric, vals in metric_values.items()}

    macro_metric_values: Dict[str, List[float]] = {metric: [] for metric in ALL_METRICS}
    for run in seed_runs:
        for metric in ALL_METRICS:
            vals = np.asarray([_metric_value(run["results"][str(int(h))], metric) for h in horizons], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            macro_metric_values[metric].append(float(np.mean(finite)) if finite.size > 0 else float("nan"))
    macro_summary = {metric: _aggregate_values(vals) for metric, vals in macro_metric_values.items()}

    return {
        "by_horizon": horizon_summary,
        "macro_over_horizons": macro_summary,
    }


def _tune_dataset(
    cli_args: argparse.Namespace,
    plan: DatasetPlan,
    dataset_out_dir: str,
) -> Dict[str, Any]:
    tuning_dir = os.path.join(dataset_out_dir, "tuning")
    mkdir(tuning_dir)

    candidates = [
        {"history_len": int(history_len), "eval_nfe": int(eval_nfe)}
        for history_len in plan.history_options
        for eval_nfe in plan.nfe_options
    ]

    runs: List[Dict[str, Any]] = []
    split_cache: MutableMapping[int, Tuple[Any, Any]] = {}

    for idx, candidate in enumerate(candidates):
        history_len = int(candidate["history_len"])
        eval_nfe = int(candidate["eval_nfe"])
        run_name = f"h{history_len}_nfe{eval_nfe}"
        run_dir = os.path.join(tuning_dir, run_name)
        mkdir(run_dir)

        if history_len not in split_cache:
            cfg, splits = _build_cfg_and_splits(cli_args, plan, history_len)
            split_cache[history_len] = (cfg, splits)
        cfg_base, splits = split_cache[history_len]
        cfg = copy.deepcopy(cfg_base)

        seed_all(int(cli_args.tune_seed))
        t0 = time.time()
        model = train_loop(
            splits["train"],
            cfg,
            model_name="otflow",
            steps=int(plan.train_steps_tune),
            log_every=int(cli_args.log_every),
        )
        runtime_sec = float(time.time() - t0)

        val_results = _evaluate_horizon_set(
            splits["val"],
            model,
            cfg,
            horizons=plan.horizons,
            nfe=eval_nfe,
            n_windows=plan.eval_windows_tune,
            seed_base=int(cli_args.tune_seed + idx * 100_000),
        )
        mean_score = _score_across_horizons(val_results)

        record = {
            "name": run_name,
            "candidate": candidate,
            "train_steps": int(plan.train_steps_tune),
            "runtime_sec": runtime_sec,
            "score_main_val_macro": mean_score,
            "val_results": val_results,
        }
        runs.append(record)
        save_json(record, os.path.join(run_dir, "tune_result.json"))

    runs.sort(key=lambda item: item["score_main_val_macro"])
    summary = {
        "dataset": plan.name,
        "horizons": [int(h) for h in plan.horizons],
        "tune_seed": int(cli_args.tune_seed),
        "candidates": runs,
        "winner": runs[0],
    }
    save_json(summary, os.path.join(dataset_out_dir, "tuned_config.json"))
    return summary


def _load_tuned_config(dataset_out_dir: str) -> Dict[str, Any]:
    path = os.path.join(dataset_out_dir, "tuned_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing tuned config at {path}")
    import json
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _final_evaluate_dataset(
    cli_args: argparse.Namespace,
    plan: DatasetPlan,
    tuned: Mapping[str, Any],
    dataset_out_dir: str,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    final_dir = os.path.join(dataset_out_dir, "final")
    mkdir(final_dir)

    winner = tuned["winner"]["candidate"]
    history_len = int(winner["history_len"])
    eval_nfe = int(winner["eval_nfe"])
    cfg, splits = _build_cfg_and_splits(cli_args, plan, history_len)

    seed_runs: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_out_dir = os.path.join(final_dir, f"seed{int(seed)}")
        mkdir(seed_out_dir)

        seed_all(int(seed))
        model = train_loop(
            splits["train"],
            cfg,
            model_name="otflow",
            steps=int(plan.train_steps_final),
            log_every=int(cli_args.log_every),
        )

        test_results = _evaluate_horizon_set(
            splits["test"],
            model,
            cfg,
            horizons=plan.horizons,
            nfe=eval_nfe,
            n_windows=plan.eval_windows_final,
            seed_base=int(seed * 100_000),
        )
        record = {
            "seed": int(seed),
            "history_len": history_len,
            "eval_nfe": eval_nfe,
            "train_steps": int(plan.train_steps_final),
            "results": test_results,
        }
        seed_runs.append(record)
        save_json(record, os.path.join(seed_out_dir, "test_results.json"))

    aggregate = _summarize_seed_runs(seed_runs, plan.horizons)
    summary = {
        "dataset": plan.name,
        "selected_config": {
            "history_len": history_len,
            "eval_nfe": eval_nfe,
            "train_steps": int(plan.train_steps_final),
        },
        "horizons": [int(h) for h in plan.horizons],
        "seeds": [int(seed) for seed in seeds],
        "seed_runs": seed_runs,
        "aggregate": aggregate,
    }
    save_json(summary, os.path.join(dataset_out_dir, "final_summary.json"))
    return summary


def main() -> None:
    cli_args = build_argparser().parse_args()
    datasets = _parse_dataset_list(cli_args.datasets)
    seeds = parse_int_list(cli_args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")

    mkdir(cli_args.out_root)
    overall: Dict[str, Any] = {
        "datasets": {},
        "seeds": [int(seed) for seed in seeds],
        "dataset_seed": int(cli_args.dataset_seed),
    }

    for dataset_name in datasets:
        plan = DATASET_PLANS[dataset_name]
        dataset_out_dir = os.path.join(cli_args.out_root, dataset_name)
        mkdir(dataset_out_dir)

        if cli_args.final_only:
            tuned = _load_tuned_config(dataset_out_dir)
        elif cli_args.skip_tuning:
            tuned = {
                "dataset": plan.name,
                "horizons": [int(h) for h in plan.horizons],
                "winner": {
                    "candidate": {
                        "history_len": int(plan.history_options[-1]),
                        "eval_nfe": int(plan.nfe_options[-1]),
                    }
                },
            }
            save_json(tuned, os.path.join(dataset_out_dir, "tuned_config.json"))
        else:
            tuned = _tune_dataset(cli_args, plan, dataset_out_dir)

        final_summary = _final_evaluate_dataset(cli_args, plan, tuned, dataset_out_dir, seeds)
        overall["datasets"][dataset_name] = {
            "tuned_config": tuned["winner"]["candidate"],
            "aggregate": final_summary["aggregate"],
        }

    save_json(overall, os.path.join(cli_args.out_root, "overall_summary.json"))


if __name__ == "__main__":
    main()
