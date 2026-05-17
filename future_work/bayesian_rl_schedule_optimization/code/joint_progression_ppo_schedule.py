from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bo_trajectory_dataset import (
    calibrate_from_bo_elites,
    load_bo_artifacts,
)
from eval_cache import JsonlEvalCache
from forecast_bo_runner import (
    _IndexSubset,
    _build_reference_schedule,
    _evaluate_schedule,
    _load_checkpoint,
    _load_final_rows,
    _run_solver_bo,
    _core_imports,
    deterministic_validation_partition,
    load_json,
    parse_csv,
    parse_int_csv,
    resolve_workspace_path,
    schedule_hash,
    selected_indices,
    selected_indices_from_pool,
    summarize_final_rows,
    write_csv,
    write_json,
)
from reward import DEFAULT_LAMBDA_BAD, reward_from_metrics, selector_rank_key
from residual_parameterization import DEFAULT_BASIS_KIND, validate_n_initial
from schedule_param import build_default_basis_for_reference, schedule_diagnostics, theta_to_checked_schedule


JOINT_PROGRESSION_PPO_BEST_KEY = "joint_progression_ppo_best"

PPO_COMPARISON_SCHEDULES: Tuple[str, ...] = (
    "uniform",
    "late_power_3",
    "ays",
    "gits",
    "ots",
    "ser_ptg_reference",
    "bo_best",
    JOINT_PROGRESSION_PPO_BEST_KEY,
)
PPO_CACHE_REUSE_KEYS: Tuple[str, ...] = (
    "uniform",
    "late_power_3",
    "ays",
    "gits",
    "ots",
    "ser_ptg_reference",
    "bo_best",
    JOINT_PROGRESSION_PPO_BEST_KEY,
)
def ppo_budget_total(batch_size: int, updates: int) -> int:
    batch = int(batch_size)
    n_updates = int(updates)
    if batch <= 0:
        raise ValueError(f"ppo_batch_size must be positive, got {batch_size}.")
    if n_updates <= 0:
        raise ValueError(f"ppo_updates must be positive, got {updates}.")
    return int(batch * n_updates)


def cell_dir(out_root: str | Path, dataset: str, target_nfe: int, solver_key: str) -> Path:
    return Path(out_root) / str(dataset) / f"nfe_{int(target_nfe)}" / str(solver_key)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _indices_hash(indices: Sequence[int]) -> str:
    return _stable_hash([int(idx) for idx in indices])


def _train_rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "candidate_id": str(row.get("candidate_id", "")),
            "update": int(row.get("update", -1)),
            "sample_id": int(row.get("sample_id", -1)),
            "schedule_hash": str(row.get("schedule_hash", "")),
            "reward": round(float(row.get("reward", 0.0)), 12),
        }
        for row in rows
        if str(row.get("split")) == "ppo_train_70pct"
    ]
    return _stable_hash(payload)


def _complete_train_rows_for_resume(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    run_fingerprint: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    by_update: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("split")) != "ppo_train_70pct":
            continue
        if run_fingerprint is not None and str(row.get("run_fingerprint", "")) != str(run_fingerprint):
            continue
        update = int(row.get("update", -1))
        if update < 0:
            continue
        by_update.setdefault(update, []).append(dict(row))
    kept: List[Dict[str, Any]] = []
    completed = 0
    for update in range(max(by_update.keys(), default=-1) + 1):
        group = sorted(by_update.get(update, []), key=lambda row: int(row.get("sample_id", -1)))
        sample_ids = [int(row.get("sample_id", -1)) for row in group]
        if len(group) != int(batch_size) or sample_ids != list(range(int(batch_size))):
            break
        kept.extend(group)
        completed += 1
    return kept, completed


def _selector_row_matches_candidate(
    row: Mapping[str, Any],
    *,
    candidate_id: str,
    schedule_record: Mapping[str, Any],
    expected_eval_examples: int,
    num_eval_samples: int,
    selector_indices_hash: str,
    run_fingerprint: str,
) -> bool:
    if str(row.get("split")) != "ppo_selector_30pct":
        return False
    if str(row.get("candidate_id")) != str(candidate_id):
        return False
    row_hash = row.get("schedule_hash")
    if row_hash is None and row.get("schedule_grid") is not None:
        row_hash = schedule_hash(row["schedule_grid"])
    if str(row_hash) != str(schedule_hash(schedule_record["schedule_grid"])):
        return False
    if int(row.get("eval_examples", -1)) != int(expected_eval_examples):
        return False
    if int(row.get("num_eval_samples", -1)) != int(num_eval_samples):
        return False
    if str(row.get("selector_indices_hash", "")) != str(selector_indices_hash):
        return False
    if str(row.get("run_fingerprint", "")) != str(run_fingerprint):
        return False
    return True


def _policy_run_fingerprint(
    args: argparse.Namespace,
    *,
    checkpoint: Mapping[str, Any],
    dataset: str,
    target_nfe: int,
    solver_key: str,
    reference: Mapping[str, Any],
    split_payload: Mapping[str, Any],
    calibration: Optional[Mapping[str, Any]] = None,
    bo_observation_fingerprint: Optional[str] = None,
) -> str:
    calibration_identity = None
    if calibration is not None:
        calibration_identity = {
            "beta_ref": float(calibration.get("beta_ref", 0.0)),
            "hard_kl_cap": float(calibration.get("hard_kl_cap", 0.0)),
            "policy_mu": [round(float(x), 12) for x in calibration.get("policy_mu", [])],
            "policy_std": [round(float(x), 12) for x in calibration.get("policy_std", [])],
            "elite_candidate_ids": [str(x) for x in calibration.get("elite_candidate_ids", [])],
        }
    return _stable_hash(
        {
            "artifact": "joint_progression_ppo_policy_run_fingerprint_v1",
            "dataset": str(dataset),
            "solver_key": str(solver_key),
            "target_nfe": int(target_nfe),
            "runtime_nfe": int(reference["runtime_nfe"]),
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "train_steps": int(checkpoint["train_steps"]),
            "reference_schedule_hash": str(reference.get("schedule_hash") or schedule_hash(reference["schedule_grid"])),
            "basis_kind": DEFAULT_BASIS_KIND,
            "ppo_batch_size": int(args.ppo_batch_size),
            "ppo_updates": int(args.ppo_updates),
            "ppo_epochs": int(args.ppo_epochs),
            "ppo_seed": int(args.ppo_seed),
            "clip_eps": float(args.clip_eps),
            "lr_actor": float(args.lr_actor),
            "lr_value": float(args.lr_value),
            "entropy_coef_start": float(args.entropy_coef_start),
            "entropy_coef_end": float(args.entropy_coef_end),
            "target_policy_kl": float(args.target_policy_kl),
            "max_policy_kl": float(args.max_policy_kl),
            "lambda_bad": float(args.lambda_bad),
            "elite_fraction": float(args.elite_fraction),
            "awr_w_max": float(args.awr_w_max),
            "init_std_min": float(args.init_std_min),
            "init_std_max": float(args.init_std_max),
            "selector_top_k": int(args.selector_top_k),
            "calibration_windows": int(split_payload["calibration_windows"]),
            "selector_windows": int(split_payload["selector_windows"]),
            "calibration_indices_hash": _indices_hash(split_payload["calibration_indices"]),
            "selector_indices_hash": _indices_hash(split_payload["selector_indices"]),
            "num_eval_samples": int(args.num_eval_samples),
            "reference_macro_factor": float(args.reference_macro_factor),
            "bo_observation_fingerprint": str(bo_observation_fingerprint or ""),
            "bo_budget": int(args.bo_budget),
            "n_initial": int(args.n_initial),
            "bo_batch_size": int(args.bo_batch_size),
            "lambda_kl": float(args.lambda_kl),
            "theta_bound": float(args.theta_bound),
            "raw_samples": int(args.raw_samples),
            "num_restarts": int(args.num_restarts),
            "mc_samples": int(args.mc_samples),
            "bo_seed": int(args.bo_seed),
            "bo_val_windows": int(args.bo_val_windows),
            "confirm_top_k": int(args.confirm_top_k),
            "calibration_trace_samples": int(args.calibration_trace_samples),
            "density_floor_eta": float(args.density_floor_eta),
            "calibration_identity": calibration_identity,
        }
    )


def _write_csv_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in keys})


def _uniform_baseline(metrics: Mapping[str, Any]) -> Dict[str, float]:
    crps = float(metrics["crps"])
    mase = float(metrics["mase"])
    if crps <= 0.0 or mase <= 0.0 or not np.isfinite(crps) or not np.isfinite(mase):
        raise ValueError("Uniform baseline CRPS and MASE must be finite and positive.")
    return {"crps": crps, "mase": mase}


def _candidate_row_from_metrics(
    *,
    dataset: str,
    solver_key: str,
    target_nfe: int,
    runtime_nfe: int,
    candidate_id: str,
    source: str,
    split: str,
    theta: Sequence[float],
    schedule_record: Mapping[str, Any],
    metrics: Mapping[str, Any],
    uniform_baseline: Mapping[str, Any],
    beta_ref: float,
    lambda_bad: float,
    update: Optional[int] = None,
    sample_id: Optional[int] = None,
    policy_logprob: Optional[float] = None,
) -> Dict[str, Any]:
    reward_payload = reward_from_metrics(
        metrics,
        uniform_baseline=uniform_baseline,
        kl_to_reference=schedule_record["kl_to_reference"],
        beta_ref=float(beta_ref),
        lambda_bad=float(lambda_bad),
    )
    row: Dict[str, Any] = {
        "dataset": str(dataset),
        "solver_key": str(solver_key),
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "candidate_id": str(candidate_id),
        "source": str(source),
        "split": str(split),
        "theta": [float(x) for x in theta],
        "time_grid": [float(x) for x in schedule_record["schedule_grid"]],
        "schedule_grid": [float(x) for x in schedule_record["schedule_grid"]],
        "q": [float(x) for x in schedule_record["q"]],
        "schedule_hash": schedule_hash(schedule_record["schedule_grid"]),
        "kl_to_reference": float(schedule_record["kl_to_reference"]),
        "min_dt": float(schedule_record["min_dt"]),
        "smoothness": float(schedule_record["smoothness"]),
        "crps": float(metrics["crps"]),
        "mase": float(metrics["mase"]),
        "mse": float(metrics.get("mse", float("nan"))),
        "latency_ms_per_sample": float(metrics.get("latency_ms_per_sample", float("nan"))),
        "eval_examples": int(metrics.get("eval_examples", 0)),
        "num_eval_samples": int(metrics.get("num_eval_samples", 0)),
        "uniform_crps": float(uniform_baseline["crps"]),
        "uniform_mase": float(uniform_baseline["mase"]),
        **reward_payload,
    }
    if update is not None:
        row["update"] = int(update)
    if sample_id is not None:
        row["sample_id"] = int(sample_id)
    if policy_logprob is not None:
        row["policy_logprob"] = float(policy_logprob)
    return row


def _evaluate_cached(
    *,
    cache: JsonlEvalCache,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    ds: Any,
    split_id: str,
    dataset: str,
    target_nfe: int,
    solver_key: str,
    runtime_nfe: int,
    schedule_grid: Sequence[float],
    num_eval_samples: int,
    seed: int,
) -> Dict[str, Any]:
    split_indices_hash = _indices_hash(getattr(ds, "indices", [])) if hasattr(ds, "indices") else None
    key = JsonlEvalCache.make_key(
        split_id=str(split_id),
        schedule_hash=schedule_hash(schedule_grid),
        seed=int(seed),
        num_eval_samples=int(num_eval_samples),
        dataset=str(dataset),
        target_nfe=int(target_nfe),
        solver_key=str(solver_key),
        runtime_nfe=int(runtime_nfe),
        checkpoint_id=str(checkpoint["checkpoint_id"]),
        split_indices_hash=split_indices_hash,
        eval_examples=len(ds) if hasattr(ds, "__len__") else None,
    )
    cached = cache.get(key)
    if cached is not None:
        return {k: v for k, v in cached.items() if k != "cache_key"}
    metrics = _evaluate_schedule(
        core,
        checkpoint,
        ds,
        solver_key=str(solver_key),
        runtime_nfe=int(runtime_nfe),
        time_grid=schedule_grid,
        num_eval_samples=int(num_eval_samples),
        seed=int(seed),
    )
    return cache.put(key, metrics)


class _DiagonalGaussianPolicy:
    def __init__(self, mu: Sequence[float], std: Sequence[float], *, device: str = "cpu"):
        import torch

        self.torch = torch
        self.mu = torch.nn.Parameter(torch.as_tensor(mu, dtype=torch.float64, device=device))
        self.log_std = torch.nn.Parameter(torch.log(torch.as_tensor(std, dtype=torch.float64, device=device)))
        self.value = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device=device))

    def parameters(self) -> List[Any]:
        return [self.mu, self.log_std, self.value]

    def distribution(self) -> Any:
        return self.torch.distributions.Normal(self.mu, self.torch.exp(self.log_std))

    def sample(self, *, generator: Any = None) -> Tuple[np.ndarray, float]:
        dist = self.distribution()
        if generator is None:
            theta = dist.rsample()
        else:
            eps = self.torch.randn(self.mu.shape, dtype=self.mu.dtype, device=self.mu.device, generator=generator)
            theta = self.mu + self.torch.exp(self.log_std) * eps
        logprob = dist.log_prob(theta).sum()
        return theta.detach().cpu().numpy().astype(np.float64), float(logprob.detach().cpu().item())

    def log_prob(self, theta_batch: Any) -> Any:
        return self.distribution().log_prob(theta_batch).sum(dim=-1)

    def entropy(self) -> Any:
        return self.distribution().entropy().sum()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mu": self.mu.detach().cpu(),
            "log_std": self.log_std.detach().cpu(),
            "value": self.value.detach().cpu(),
        }

    def load_state_dict(self, state: Mapping[str, Any], *, device: str = "cpu") -> None:
        torch = self.torch
        self.mu.data = torch.as_tensor(state["mu"], dtype=torch.float64, device=device)
        self.log_std.data = torch.as_tensor(state["log_std"], dtype=torch.float64, device=device)
        self.value.data = torch.as_tensor(state.get("value", 0.0), dtype=torch.float64, device=device)


def diagonal_policy_kl(old_mu: Sequence[float], old_std: Sequence[float], new_mu: Sequence[float], new_std: Sequence[float]) -> float:
    old_m = np.asarray(old_mu, dtype=np.float64)
    old_s = np.asarray(old_std, dtype=np.float64)
    new_m = np.asarray(new_mu, dtype=np.float64)
    new_s = np.asarray(new_std, dtype=np.float64)
    if np.any(old_s <= 0.0) or np.any(new_s <= 0.0):
        raise ValueError("Policy standard deviations must be positive.")
    return float(np.sum(np.log(new_s / old_s) + (old_s * old_s + (old_m - new_m) ** 2) / (2.0 * new_s * new_s) - 0.5))


def ppo_update(
    policy: _DiagonalGaussianPolicy,
    *,
    theta_batch: Sequence[Sequence[float]],
    old_logprobs: Sequence[float],
    rewards: Sequence[float],
    ppo_epochs: int,
    clip_eps: float,
    lr_actor: float,
    lr_value: float,
    entropy_coef: float,
    target_policy_kl: float,
    max_policy_kl: float,
) -> Dict[str, float]:
    torch = policy.torch
    theta_t = torch.as_tensor(theta_batch, dtype=torch.float64, device=policy.mu.device)
    old_logp = torch.as_tensor(old_logprobs, dtype=torch.float64, device=policy.mu.device)
    reward_t = torch.as_tensor(rewards, dtype=torch.float64, device=policy.mu.device)
    opt = torch.optim.Adam(
        [
            {"params": [policy.mu, policy.log_std], "lr": float(lr_actor)},
            {"params": [policy.value], "lr": float(lr_value)},
        ]
    )
    old_mu = policy.mu.detach().cpu().numpy().copy()
    old_std = torch.exp(policy.log_std.detach()).cpu().numpy().copy()
    advantages = reward_t - policy.value.detach()
    if theta_t.shape[0] > 1:
        adv_std = torch.std(advantages)
        if float(adv_std.detach().cpu().item()) > 1e-12:
            advantages = (advantages - torch.mean(advantages)) / (adv_std + 1e-12)
    last_loss = 0.0
    last_policy_kl = 0.0
    stopped_by_target_kl = False
    restored_by_max_kl = False
    for _ in range(int(ppo_epochs)):
        previous_state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        opt.zero_grad()
        logp = policy.log_prob(theta_t)
        ratio = torch.exp(logp - old_logp)
        clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
        policy_loss = -torch.mean(torch.minimum(ratio * advantages, clipped_ratio * advantages))
        value_loss = torch.mean((policy.value - reward_t) ** 2)
        loss = policy_loss + 0.5 * value_loss - float(entropy_coef) * policy.entropy()
        loss.backward()
        opt.step()
        policy.log_std.data.clamp_(math.log(0.005), math.log(1.0))
        new_mu = policy.mu.detach().cpu().numpy()
        new_std = torch.exp(policy.log_std.detach()).cpu().numpy()
        last_policy_kl = diagonal_policy_kl(old_mu, old_std, new_mu, new_std)
        last_loss = float(loss.detach().cpu().item())
        if last_policy_kl > float(max_policy_kl):
            policy.load_state_dict(previous_state, device=str(policy.mu.device))
            restored_by_max_kl = True
            new_mu = policy.mu.detach().cpu().numpy()
            new_std = torch.exp(policy.log_std.detach()).cpu().numpy()
            last_policy_kl = diagonal_policy_kl(old_mu, old_std, new_mu, new_std)
            break
        if last_policy_kl > float(target_policy_kl):
            stopped_by_target_kl = True
            break
    return {
        "loss": float(last_loss),
        "policy_kl": float(last_policy_kl),
        "stopped_by_target_policy_kl": bool(stopped_by_target_kl),
        "restored_by_max_policy_kl": bool(restored_by_max_kl),
        "value": float(policy.value.detach().cpu().item()),
        "policy_std_mean": float(policy.torch.exp(policy.log_std.detach()).mean().cpu().item()),
    }


def _sample_valid_batch(
    policy: _DiagonalGaussianPolicy,
    *,
    q_ref: Sequence[float],
    basis: np.ndarray,
    hard_kl_cap: float,
    batch_size: int,
    seed: int,
    max_attempts: int = 10_000,
) -> List[Dict[str, Any]]:
    torch = policy.torch
    generator = torch.Generator(device=policy.mu.device)
    generator.manual_seed(int(seed))
    rows: List[Dict[str, Any]] = []
    attempts = 0
    while len(rows) < int(batch_size) and attempts < int(max_attempts):
        attempts += 1
        theta, logprob = policy.sample(generator=generator)
        try:
            record = theta_to_checked_schedule(q_ref, theta, basis=basis, hard_kl_cap=float(hard_kl_cap))
        except ValueError:
            continue
        rows.append({"theta": [float(x) for x in theta.tolist()], "old_logprob": float(logprob), "schedule": record})
    if len(rows) < int(batch_size):
        raise RuntimeError(f"Unable to sample {batch_size} valid schedules under hard_kl_cap={hard_kl_cap}.")
    return rows


def _completed_training_updates(trials: Sequence[Mapping[str, Any]], batch_size: int) -> int:
    train_rows = [row for row in trials if str(row.get("split")) == "ppo_train_70pct"]
    if not train_rows:
        return 0
    completed = 0
    by_update: Dict[int, int] = {}
    for row in train_rows:
        by_update[int(row.get("update", -1))] = by_update.get(int(row.get("update", -1)), 0) + 1
    while by_update.get(completed, 0) >= int(batch_size):
        completed += 1
    return completed


def _torch_device_for_policy(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def _save_policy(path: Path, policy: _DiagonalGaussianPolicy, metadata: Mapping[str, Any]) -> None:
    torch = policy.torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": policy.state_dict(), "metadata": dict(metadata)}, path)


def _load_policy(path: Path, policy: _DiagonalGaussianPolicy, *, device: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    torch = policy.torch
    payload = torch.load(path, map_location=device)
    policy.load_state_dict(payload["state"], device=device)
    return dict(payload.get("metadata", {}))


def _cell_args(args: argparse.Namespace, *, dataset: str, target_nfe: int, solver_key: str, out_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        workspace_root=args.workspace_root,
        out_root=out_root,
        dataset=str(dataset),
        target_nfe=int(target_nfe),
        solvers=str(solver_key),
        otflow_train_steps=int(args.otflow_train_steps),
        bo_budget=int(args.bo_budget),
        calibration_fraction=float(args.calibration_fraction),
        calibration_windows=int(args.calibration_windows),
        bo_val_windows=int(args.bo_val_windows),
        confirm_top_k=int(args.confirm_top_k),
        confirm_val_windows=int(args.selector_windows),
        num_eval_samples=int(args.num_eval_samples),
        reference_macro_factor=float(args.reference_macro_factor),
        calibration_trace_samples=int(args.calibration_trace_samples),
        final_test_seeds=str(args.final_test_seeds),
        final_test_windows=int(args.final_test_windows),
        comparison_schedules="uniform,gits,ser_ptg_reference,bo_best",
        n_initial=int(args.n_initial),
        bo_batch_size=int(args.bo_batch_size),
        lambda_kl=float(args.lambda_kl),
        theta_bound=float(args.theta_bound),
        raw_samples=int(args.raw_samples),
        num_restarts=int(args.num_restarts),
        mc_samples=int(args.mc_samples),
        bo_seed=int(args.bo_seed),
        density_floor_eta=float(args.density_floor_eta),
        baseline_cache_roots="",
        device=str(args.device),
        resume=bool(args.resume),
    )


def _ensure_bo_warmstart(
    *,
    args: argparse.Namespace,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    dataset: str,
    target_nfe: int,
    solver_key: str,
    solver_idx: int,
    cell_out: Path,
) -> Path:
    bo_root = cell_out / "bo_warmstart"
    bo_args = _cell_args(args, dataset=dataset, target_nfe=int(target_nfe), solver_key=str(solver_key), out_root=bo_root)
    _run_solver_bo(
        args=bo_args,
        core=core,
        checkpoint=checkpoint,
        solver_key=str(solver_key),
        solver_idx=int(solver_idx),
        out_root=bo_root,
        resume=bool(args.resume),
    )
    return bo_root / str(solver_key)


def _split_indices_from_bo(
    split_payload: Mapping[str, Any],
    *,
    calibration_windows: int,
    selector_windows: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    calibration_indices = selected_indices_from_pool(
        [int(x) for x in split_payload["calibration_indices"]],
        int(calibration_windows),
        seed=int(seed) + 101,
    )
    selector_source = split_payload.get("confirmation_indices") or split_payload.get("bo_validation_indices")
    if not selector_source:
        raise ValueError("BO validation split must contain confirmation_indices or bo_validation_indices.")
    selector_indices = selected_indices_from_pool(selector_source, int(selector_windows), seed=int(seed))
    return calibration_indices, selector_indices


def _evaluate_uniform_baseline(
    *,
    cache: JsonlEvalCache,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    ds: Any,
    split_id: str,
    dataset: str,
    target_nfe: int,
    solver_key: str,
    runtime_nfe: int,
    num_eval_samples: int,
    seed: int,
) -> Dict[str, float]:
    uniform_grid = core["build_schedule_grid"]("uniform", int(runtime_nfe))
    metrics = _evaluate_cached(
        cache=cache,
        core=core,
        checkpoint=checkpoint,
        ds=ds,
        split_id=split_id,
        dataset=str(dataset),
        target_nfe=int(target_nfe),
        solver_key=str(solver_key),
        runtime_nfe=int(runtime_nfe),
        schedule_grid=uniform_grid,
        num_eval_samples=int(num_eval_samples),
        seed=int(seed),
    )
    return _uniform_baseline(metrics)


def _run_ppo_training_for_cell(
    *,
    args: argparse.Namespace,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    dataset: str,
    target_nfe: int,
    solver_key: str,
    solver_idx: int,
    cell_out: Path,
    bo_artifacts: Mapping[str, Any],
) -> Dict[str, Any]:
    import torch

    reference = bo_artifacts["reference"]
    observations = bo_artifacts["observations"]
    validation_split = bo_artifacts["validation_split"]
    q_ref = np.asarray(reference["q_ref"], dtype=np.float64)
    basis = build_default_basis_for_reference(q_ref)
    calibration = calibrate_from_bo_elites(
        reference,
        observations,
        elite_fraction=float(args.elite_fraction),
        lambda_bad=float(args.lambda_bad),
        w_max=float(args.awr_w_max),
        std_min=float(args.init_std_min),
        std_max=float(args.init_std_max),
    )
    write_json(calibration, cell_out / "joint_progression_ppo_calibration.json")
    write_json(reference, cell_out / "reference_schedule.json")
    write_json(bo_artifacts["best"], cell_out / "bo_best_schedule.json")

    train_indices, selector_indices = _split_indices_from_bo(
        validation_split,
        calibration_windows=int(args.calibration_windows),
        selector_windows=int(args.selector_windows),
        seed=int(args.ppo_seed) + 17_000 + int(solver_idx),
    )
    split_payload = {
        "artifact": "joint_progression_ppo_validation_split_v1",
        "calibration_indices": train_indices,
        "selector_indices": selector_indices,
        "calibration_windows": int(len(train_indices)),
        "selector_windows": int(len(selector_indices)),
    }
    write_json(split_payload, cell_out / "joint_progression_ppo_split.json")
    train_ds = _IndexSubset(checkpoint["splits"]["val"], train_indices)
    selector_ds = _IndexSubset(checkpoint["splits"]["val"], selector_indices)
    runtime_nfe = int(reference["runtime_nfe"])
    cache = JsonlEvalCache(cell_out / "eval_cache.jsonl")
    train_uniform = _evaluate_uniform_baseline(
        cache=cache,
        core=core,
        checkpoint=checkpoint,
        ds=train_ds,
        split_id="ppo_train_70pct_uniform",
        dataset=str(dataset),
        target_nfe=int(target_nfe),
        solver_key=str(solver_key),
        runtime_nfe=runtime_nfe,
        num_eval_samples=int(args.num_eval_samples),
        seed=int(args.ppo_seed) + 70_000,
    )
    selector_uniform = _evaluate_uniform_baseline(
        cache=cache,
        core=core,
        checkpoint=checkpoint,
        ds=selector_ds,
        split_id="ppo_selector_30pct_uniform",
        dataset=str(dataset),
        target_nfe=int(target_nfe),
        solver_key=str(solver_key),
        runtime_nfe=runtime_nfe,
        num_eval_samples=int(args.num_eval_samples),
        seed=int(args.ppo_seed) + 71_000,
    )
    write_json(
        {
            "artifact": "joint_progression_ppo_uniform_baselines_v1",
            "train_uniform": train_uniform,
            "selector_uniform": selector_uniform,
        },
        cell_out / "joint_progression_ppo_uniform_baselines.json",
    )

    policy_device = _torch_device_for_policy(str(args.device))
    policy = _DiagonalGaussianPolicy(calibration["policy_mu"], calibration["policy_std"], device=policy_device)
    policy_path = cell_out / "joint_progression_ppo_policy.pt"
    trials_path = cell_out / "joint_progression_ppo_trials.jsonl"
    total_updates = int(args.ppo_updates)
    run_fingerprint = _policy_run_fingerprint(
        args,
        checkpoint=checkpoint,
        dataset=str(dataset),
        target_nfe=int(target_nfe),
        solver_key=str(solver_key),
        reference=reference,
        split_payload=split_payload,
        calibration=calibration,
        bo_observation_fingerprint=str(observations.get("run_fingerprint", "")),
    )
    if bool(args.resume):
        existing_trials = _load_jsonl(trials_path)
        existing_selector_rows = [
            dict(row)
            for row in existing_trials
            if str(row.get("split")) == "ppo_selector_30pct" and str(row.get("run_fingerprint", "")) == run_fingerprint
        ]
        train_rows, completed_updates = _complete_train_rows_for_resume(
            existing_trials,
            batch_size=int(args.ppo_batch_size),
            run_fingerprint=run_fingerprint,
        )
        clean_rows = list(train_rows)
        if completed_updates >= total_updates:
            clean_rows.extend(existing_selector_rows)
        if len(clean_rows) != len(existing_trials):
            _write_jsonl(trials_path, clean_rows)
    else:
        existing_trials = []
        existing_selector_rows = []
        train_rows = []
        completed_updates = 0
        if trials_path.exists():
            _write_jsonl(trials_path, [])
    if completed_updates > 0:
        metadata = _load_policy(policy_path, policy, device=policy_device)
        metadata_ok = (
            bool(metadata)
            and str(metadata.get("run_fingerprint", "")) == run_fingerprint
            and str(metadata.get("train_rows_hash", "")) == _train_rows_hash(train_rows)
            and int(metadata.get("last_update", -1)) == int(completed_updates - 1)
        )
        if not metadata_ok:
            train_rows = []
            completed_updates = 0
            existing_selector_rows = []
            policy = _DiagonalGaussianPolicy(calibration["policy_mu"], calibration["policy_std"], device=policy_device)
            _write_jsonl(trials_path, [])
    for update in range(completed_updates, total_updates):
        entropy_coef = float(args.entropy_coef_end)
        if total_updates > 1:
            frac = float(update) / float(total_updates - 1)
            entropy_coef = float(args.entropy_coef_start) * (1.0 - frac) + float(args.entropy_coef_end) * frac
        samples = _sample_valid_batch(
            policy,
            q_ref=q_ref,
            basis=basis,
            hard_kl_cap=float(calibration["hard_kl_cap"]),
            batch_size=int(args.ppo_batch_size),
            seed=int(args.ppo_seed) + 100_000 * int(solver_idx) + int(update),
        )
        theta_batch: List[List[float]] = []
        old_logprobs: List[float] = []
        rewards: List[float] = []
        for sample_id, sample in enumerate(samples):
            candidate_id = f"ppo_u{update:03d}_s{sample_id:03d}"
            metrics = _evaluate_cached(
                cache=cache,
                core=core,
                checkpoint=checkpoint,
                ds=train_ds,
                split_id="ppo_train_70pct",
                dataset=str(dataset),
                target_nfe=int(target_nfe),
                solver_key=str(solver_key),
                runtime_nfe=runtime_nfe,
                schedule_grid=sample["schedule"]["schedule_grid"],
                num_eval_samples=int(args.num_eval_samples),
                seed=int(args.ppo_seed) + 72_000,
            )
            row = _candidate_row_from_metrics(
                dataset=str(dataset),
                solver_key=str(solver_key),
                target_nfe=int(target_nfe),
                runtime_nfe=runtime_nfe,
                candidate_id=candidate_id,
                source="joint_progression_ppo_policy",
                split="ppo_train_70pct",
                theta=sample["theta"],
                schedule_record=sample["schedule"],
                metrics=metrics,
                uniform_baseline=train_uniform,
                beta_ref=float(calibration["beta_ref"]),
                lambda_bad=float(args.lambda_bad),
                update=update,
                sample_id=sample_id,
                policy_logprob=float(sample["old_logprob"]),
            )
            row["run_fingerprint"] = run_fingerprint
            _append_jsonl(trials_path, row)
            train_rows.append(row)
            theta_batch.append(sample["theta"])
            old_logprobs.append(float(sample["old_logprob"]))
            rewards.append(float(row["reward"]))
        update_payload = ppo_update(
            policy,
            theta_batch=theta_batch,
            old_logprobs=old_logprobs,
            rewards=rewards,
            ppo_epochs=int(args.ppo_epochs),
            clip_eps=float(args.clip_eps),
            lr_actor=float(args.lr_actor),
            lr_value=float(args.lr_value),
            entropy_coef=entropy_coef,
            target_policy_kl=float(args.target_policy_kl),
            max_policy_kl=float(args.max_policy_kl),
        )
        _save_policy(
            policy_path,
            policy,
            {
                "artifact": "joint_progression_ppo_policy_v1",
                "dataset": str(dataset),
                "solver_key": str(solver_key),
                "target_nfe": int(target_nfe),
                "last_update": int(update),
                "run_fingerprint": run_fingerprint,
                "train_rows_hash": _train_rows_hash(train_rows),
                **update_payload,
            },
        )

    best_path = cell_out / "joint_progression_ppo_best_selector.json"
    selector_metadata = {
        "artifact": "joint_progression_ppo_selector_metadata_v1",
        "run_fingerprint": run_fingerprint,
        "train_rows_hash": _train_rows_hash(train_rows),
        "selector_indices_hash": _indices_hash(selector_indices),
        "selector_windows": int(len(selector_indices)),
        "selector_top_k": int(args.selector_top_k),
        "num_eval_samples": int(args.num_eval_samples),
        "beta_ref": float(calibration["beta_ref"]),
        "lambda_bad": float(args.lambda_bad),
        "hard_kl_cap": float(calibration["hard_kl_cap"]),
    }
    best_selector: Optional[Dict[str, Any]] = None
    if bool(args.resume) and best_path.exists():
        payload = load_json(best_path)
        if payload.get("selector_metadata") == selector_metadata:
            best_selector = payload
    if best_selector is None:
        candidate_pool = sorted(train_rows, key=selector_rank_key, reverse=True)[: max(1, int(args.selector_top_k))]
        try:
            current_mean = theta_to_checked_schedule(
                q_ref,
                policy.mu.detach().cpu().numpy(),
                basis=basis,
                hard_kl_cap=float(calibration["hard_kl_cap"]),
            )
            candidate_pool.append(
                {
                    "candidate_id": "ppo_policy_mean",
                    "theta": [float(x) for x in policy.mu.detach().cpu().numpy().tolist()],
                    "schedule_grid": current_mean["schedule_grid"],
                    "q": current_mean["q"],
                    "kl_to_reference": current_mean["kl_to_reference"],
                    "min_dt": current_mean["min_dt"],
                    "smoothness": current_mean["smoothness"],
                }
            )
        except ValueError:
            pass
        selector_indices_hash = _indices_hash(selector_indices)
        existing_selector_by_key = {
            (str(row.get("candidate_id")), str(row.get("schedule_hash") or (schedule_hash(row["schedule_grid"]) if row.get("schedule_grid") is not None else ""))): dict(row)
            for row in _load_jsonl(trials_path)
            if str(row.get("split")) == "ppo_selector_30pct"
        }
        selector_rows: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(candidate_pool):
            try:
                schedule_record = theta_to_checked_schedule(
                    q_ref,
                    candidate["theta"],
                    basis=basis,
                    hard_kl_cap=float(calibration["hard_kl_cap"]),
                )
            except ValueError:
                continue
            candidate_id = str(candidate.get("candidate_id", f"selector_{idx:03d}"))
            selector_key = (candidate_id, schedule_hash(schedule_record["schedule_grid"]))
            existing_selector = existing_selector_by_key.get(selector_key)
            if existing_selector is not None and _selector_row_matches_candidate(
                existing_selector,
                candidate_id=candidate_id,
                schedule_record=schedule_record,
                expected_eval_examples=len(selector_ds),
                num_eval_samples=int(args.num_eval_samples),
                selector_indices_hash=selector_indices_hash,
                run_fingerprint=run_fingerprint,
            ):
                selector_rows.append(existing_selector)
                continue
            print(
                f"[joint-progression-ppo] selector eval dataset={dataset} nfe={target_nfe} solver={solver_key} "
                f"candidate={candidate_id} {idx + 1}/{len(candidate_pool)} examples={len(selector_ds)}",
                flush=True,
            )
            metrics = _evaluate_cached(
                cache=cache,
                core=core,
                checkpoint=checkpoint,
                ds=selector_ds,
                split_id="ppo_selector_30pct",
                dataset=str(dataset),
                target_nfe=int(target_nfe),
                solver_key=str(solver_key),
                runtime_nfe=runtime_nfe,
                schedule_grid=schedule_record["schedule_grid"],
                num_eval_samples=int(args.num_eval_samples),
                seed=int(args.ppo_seed) + 73_000,
            )
            row = _candidate_row_from_metrics(
                dataset=str(dataset),
                solver_key=str(solver_key),
                target_nfe=int(target_nfe),
                runtime_nfe=runtime_nfe,
                candidate_id=candidate_id,
                source="joint_progression_ppo_selector_confirmation",
                split="ppo_selector_30pct",
                theta=candidate["theta"],
                schedule_record=schedule_record,
                metrics=metrics,
                uniform_baseline=selector_uniform,
                beta_ref=float(calibration["beta_ref"]),
                lambda_bad=float(args.lambda_bad),
            )
            row["selector_indices_hash"] = selector_indices_hash
            row["run_fingerprint"] = run_fingerprint
            selector_rows.append(row)
            _append_jsonl(trials_path, row)
        if not selector_rows:
            raise RuntimeError("No selector candidates survived hard_kl_cap filtering.")
        best_row = max(selector_rows, key=selector_rank_key)
        best_selector = {
            "artifact": "joint_progression_ppo_best_selector_v1",
            "selected_by": "max_selector_reward",
            "selector_metadata": selector_metadata,
            "selector_rows": selector_rows,
            "best_observation": best_row,
        }
        write_json(best_selector, best_path)

    summary_rows = summarize_ppo_trials(_load_jsonl(trials_path), calibration=calibration)
    _write_csv_rows(summary_rows, cell_out / "joint_progression_ppo_summary.csv")
    return {
        "cell_out": cell_out,
        "reference": reference,
        "bo_best": bo_artifacts["best"],
        "ppo_best": best_selector["best_observation"],
        "calibration": calibration,
        "train_rows": len([row for row in _load_jsonl(trials_path) if str(row.get("split")) == "ppo_train_70pct"]),
        "selector_rows": len(best_selector.get("selector_rows", [])),
    }


def summarize_ppo_trials(trials: Sequence[Mapping[str, Any]], *, calibration: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in trials:
        grouped.setdefault(str(row.get("split", "unknown")), []).append(row)
    for split, items in sorted(grouped.items()):
        rewards = np.asarray([float(row["reward"]) for row in items if row.get("reward") is not None], dtype=np.float64)
        avg_ratios = np.asarray([float(row["avg_relative_ratio"]) for row in items if row.get("avg_relative_ratio") is not None], dtype=np.float64)
        kls = np.asarray([float(row["kl_to_reference"]) for row in items if row.get("kl_to_reference") is not None], dtype=np.float64)
        rows.append(
            {
                "split": split,
                "n_rows": int(len(items)),
                "reward_max": float(np.max(rewards)) if rewards.size else None,
                "reward_mean": float(np.mean(rewards)) if rewards.size else None,
                "avg_relative_ratio_min": float(np.min(avg_ratios)) if avg_ratios.size else None,
                "avg_relative_ratio_mean": float(np.mean(avg_ratios)) if avg_ratios.size else None,
                "kl_max": float(np.max(kls)) if kls.size else None,
                "kl_mean": float(np.mean(kls)) if kls.size else None,
                "beta_ref": float(calibration["beta_ref"]),
                "hard_kl_cap": float(calibration["hard_kl_cap"]),
            }
        )
    return rows


def _matching_cached_row_for_ppo(
    cached_rows: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    dataset: str,
    target_nfe: int,
    solver_key: str,
    schedule_key: str,
    seed: int,
    runtime_nfe: int,
    schedule_grid: Sequence[float],
    expected_eval_examples: int,
    expected_test_indices_hash: str,
) -> Optional[Dict[str, Any]]:
    if schedule_key not in PPO_CACHE_REUSE_KEYS:
        return None
    expected_hash = schedule_hash(schedule_grid)
    for row in cached_rows:
        if str(row.get("dataset")) != str(dataset):
            continue
        if int(row.get("target_nfe", -1)) != int(target_nfe):
            continue
        if str(row.get("solver_key")) != str(solver_key):
            continue
        if str(row.get("schedule_key")) != str(schedule_key):
            continue
        if int(row.get("runtime_nfe", -1)) != int(runtime_nfe):
            continue
        if int(row.get("seed", -1)) != int(seed):
            continue
        if int(row.get("num_eval_samples", -1)) != int(args.num_eval_samples):
            continue
        if int(row.get("eval_examples", -1)) != int(expected_eval_examples):
            continue
        checkpoint_id = row.get("checkpoint_id")
        if checkpoint_id is None:
            run_config = row.get("_cache_run_config", {})
            if isinstance(run_config, Mapping):
                checkpoint_id = run_config.get("checkpoint_id")
        if checkpoint_id is None or str(checkpoint_id) != str(checkpoint["checkpoint_id"]):
            continue
        if _row_test_indices_hash(row) != str(expected_test_indices_hash):
            continue
        if row.get("schedule_hash") is None or str(row["schedule_hash"]) != expected_hash:
            continue
        reused = {key: value for key, value in dict(row).items() if not str(key).startswith("_cache_")}
        reused["reused_from_cache"] = True
        reused["schedule_key"] = str(schedule_key)
        reused["schedule_hash"] = expected_hash
        reused["checkpoint_id"] = str(checkpoint["checkpoint_id"])
        reused["train_steps"] = int(checkpoint["train_steps"])
        reused["train_budget_label"] = str(checkpoint["train_budget_label"])
        reused["test_indices_hash"] = str(expected_test_indices_hash)
        return reused
    return None


def _global_final_key(row: Mapping[str, Any]) -> Tuple[str, int, str, str, int]:
    return (
        str(row["dataset"]),
        int(row["target_nfe"]),
        str(row["solver_key"]),
        str(row["schedule_key"]),
        int(row["seed"]),
    )


def _merge_global_final_rows(
    existing: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, int, str, str, int], Dict[str, Any]] = {
        _global_final_key(row): dict(row) for row in existing
    }
    for row in new_rows:
        merged[_global_final_key(row)] = dict(row)
    return list(merged.values())


def _final_row_matches_current(
    row: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    runtime_nfe: int,
    schedule_grid: Sequence[float],
    num_eval_samples: int,
    expected_eval_examples: int,
    expected_test_indices_hash: str,
) -> bool:
    if str(row.get("checkpoint_id", "")) != str(checkpoint["checkpoint_id"]):
        return False
    if int(row.get("runtime_nfe", -1)) != int(runtime_nfe):
        return False
    if int(row.get("num_eval_samples", -1)) != int(num_eval_samples):
        return False
    if int(row.get("eval_examples", -1)) != int(expected_eval_examples):
        return False
    if _row_test_indices_hash(row) != str(expected_test_indices_hash):
        return False
    return row.get("schedule_hash") is not None and str(row["schedule_hash"]) == schedule_hash(schedule_grid)


def _row_test_indices_hash(row: Mapping[str, Any]) -> str:
    if row.get("test_indices_hash") is not None:
        return str(row["test_indices_hash"])
    if row.get("test_indices") is not None:
        return _indices_hash([int(idx) for idx in row["test_indices"]])
    return ""


def _final_row_base_payload(
    *,
    checkpoint: Mapping[str, Any],
    test_indices: Sequence[int],
    test_indices_hash: str,
    num_eval_samples: int,
) -> Dict[str, Any]:
    return {
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "train_steps": int(checkpoint["train_steps"]),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "num_eval_samples": int(num_eval_samples),
        "test_indices": [int(idx) for idx in test_indices],
        "test_indices_hash": str(test_indices_hash),
    }


def _final_rows_for_cell(
    *,
    args: argparse.Namespace,
    core: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    cell_result: Mapping[str, Any],
    dataset: str,
    target_nfe: int,
    solver_key: str,
    final_path: Path,
) -> List[Dict[str, Any]]:
    reference = cell_result["reference"]
    runtime_nfe = int(reference["runtime_nfe"])
    bo_best = cell_result["bo_best"]["best_observation"]
    ppo_best = cell_result["ppo_best"]
    schedules: Dict[str, List[float]] = {}
    kl_values: Dict[str, Optional[float]] = {}
    for schedule_key in PPO_COMPARISON_SCHEDULES:
        if schedule_key == "ser_ptg_reference":
            schedules[schedule_key] = [float(x) for x in reference["schedule_grid"]]
            kl_values[schedule_key] = 0.0
        elif schedule_key == "bo_best":
            schedules[schedule_key] = [float(x) for x in bo_best["schedule_grid"]]
            kl_values[schedule_key] = float(bo_best["kl_to_reference"])
        elif schedule_key == JOINT_PROGRESSION_PPO_BEST_KEY:
            schedules[schedule_key] = [float(x) for x in ppo_best["schedule_grid"]]
            kl_values[schedule_key] = float(ppo_best["kl_to_reference"])
        else:
            schedules[schedule_key] = [float(x) for x in core["build_schedule_grid"](str(schedule_key), runtime_nfe)]
            kl_values[schedule_key] = None
    rows = _load_final_rows(final_path) if bool(args.resume) and final_path.exists() else []
    final_seeds = parse_int_csv(str(args.final_test_seeds))
    expected_keys = {
        (str(dataset), int(target_nfe), str(solver_key), str(schedule_key), int(seed))
        for schedule_key in PPO_COMPARISON_SCHEDULES
        for seed in final_seeds
    }
    rows = [row for row in rows if _global_final_key(row) in expected_keys]
    by_key = {_global_final_key(row): dict(row) for row in rows}
    for seed in final_seeds:
        test_indices = selected_indices(
            len(checkpoint["splits"]["test"]),
            int(args.final_test_windows),
            seed=int(args.ppo_seed) + 50_000 + int(seed),
        )
        test_hash = _indices_hash(test_indices)
        test_ds = _IndexSubset(checkpoint["splits"]["test"], test_indices)
        uniform_key = (str(dataset), int(target_nfe), str(solver_key), "uniform", int(seed))
        if uniform_key in by_key and not _final_row_matches_current(
            by_key[uniform_key],
            checkpoint=checkpoint,
            runtime_nfe=runtime_nfe,
            schedule_grid=schedules["uniform"],
            num_eval_samples=int(args.num_eval_samples),
            expected_eval_examples=len(test_ds),
            expected_test_indices_hash=test_hash,
        ):
            by_key.pop(uniform_key, None)
        if uniform_key not in by_key:
            print(
                f"[joint-progression-ppo] final eval dataset={dataset} nfe={target_nfe} solver={solver_key} "
                f"schedule=uniform seed={seed} examples={len(test_ds)}",
                flush=True,
            )
            metrics = _evaluate_schedule(
                core,
                checkpoint,
                test_ds,
                solver_key=str(solver_key),
                runtime_nfe=runtime_nfe,
                time_grid=schedules["uniform"],
                num_eval_samples=int(args.num_eval_samples),
                seed=int(seed),
            )
            by_key[uniform_key] = {
                "dataset": str(dataset),
                "solver_key": str(solver_key),
                "target_nfe": int(target_nfe),
                "runtime_nfe": runtime_nfe,
                "seed": int(seed),
                "schedule_key": "uniform",
                "schedule_grid": schedules["uniform"],
                "schedule_hash": schedule_hash(schedules["uniform"]),
                **_final_row_base_payload(
                    checkpoint=checkpoint,
                    test_indices=test_indices,
                    test_indices_hash=test_hash,
                    num_eval_samples=int(args.num_eval_samples),
                ),
                "crps": float(metrics["crps"]),
                "mase": float(metrics["mase"]),
                "mse": float(metrics.get("mse", float("nan"))),
                "relative_crps_ratio": 1.0,
                "relative_mase_ratio": 1.0,
                "avg_relative_ratio": 1.0,
                "kl_to_reference": None,
                "eval_examples": int(metrics.get("eval_examples", 0)),
            }
            write_json({"artifact": "joint_progression_ppo_final_comparison_rows_v1", "rows": list(by_key.values())}, final_path)
        uniform = by_key[uniform_key]
        for schedule_key in [key for key in PPO_COMPARISON_SCHEDULES if key != "uniform"]:
            key = (str(dataset), int(target_nfe), str(solver_key), str(schedule_key), int(seed))
            if key in by_key and not _final_row_matches_current(
                by_key[key],
                checkpoint=checkpoint,
                runtime_nfe=runtime_nfe,
                schedule_grid=schedules[schedule_key],
                num_eval_samples=int(args.num_eval_samples),
                expected_eval_examples=len(test_ds),
                expected_test_indices_hash=test_hash,
            ):
                by_key.pop(key, None)
            if key in by_key:
                continue
            print(
                f"[joint-progression-ppo] final eval dataset={dataset} nfe={target_nfe} solver={solver_key} "
                f"schedule={schedule_key} seed={seed} examples={len(test_ds)}",
                flush=True,
            )
            metrics = _evaluate_schedule(
                core,
                checkpoint,
                test_ds,
                solver_key=str(solver_key),
                runtime_nfe=runtime_nfe,
                time_grid=schedules[schedule_key],
                num_eval_samples=int(args.num_eval_samples),
                seed=int(seed),
            )
            r_crps = float(metrics["crps"]) / float(uniform["crps"])
            r_mase = float(metrics["mase"]) / float(uniform["mase"])
            by_key[key] = {
                "dataset": str(dataset),
                "solver_key": str(solver_key),
                "target_nfe": int(target_nfe),
                "runtime_nfe": runtime_nfe,
                "seed": int(seed),
                "schedule_key": str(schedule_key),
                "schedule_grid": schedules[schedule_key],
                "schedule_hash": schedule_hash(schedules[schedule_key]),
                **_final_row_base_payload(
                    checkpoint=checkpoint,
                    test_indices=test_indices,
                    test_indices_hash=test_hash,
                    num_eval_samples=int(args.num_eval_samples),
                ),
                "crps": float(metrics["crps"]),
                "mase": float(metrics["mase"]),
                "mse": float(metrics.get("mse", float("nan"))),
                "relative_crps_ratio": r_crps,
                "relative_mase_ratio": r_mase,
                "avg_relative_ratio": float(0.5 * (r_crps + r_mase)),
                "kl_to_reference": kl_values[schedule_key],
                "eval_examples": int(metrics.get("eval_examples", 0)),
            }
            write_json({"artifact": "joint_progression_ppo_final_comparison_rows_v1", "rows": list(by_key.values())}, final_path)
    out = list(by_key.values())
    write_json({"artifact": "joint_progression_ppo_final_comparison_rows_v1", "rows": out}, final_path)
    return out


def _summarize_global_final_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["dataset"]), int(row["target_nfe"]), str(row["solver_key"]), str(row["schedule_key"])),
            [],
        ).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, target_nfe, solver_key, schedule_key), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "dataset": dataset,
            "target_nfe": int(target_nfe),
            "solver_key": solver_key,
            "schedule_key": schedule_key,
            "n_seeds": int(len(group)),
            "seed_values": sorted(int(row["seed"]) for row in group),
            "schedule_grid": list(group[0].get("schedule_grid", [])),
        }
        for metric in ("crps", "mase", "relative_crps_ratio", "relative_mase_ratio", "avg_relative_ratio", "kl_to_reference"):
            values = np.asarray([float(row[metric]) for row in group if row.get(metric) is not None], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean()) if values.size else None
            item[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else (0.0 if values.size == 1 else None)
        out.append(item)
    return out


def run_forecast_joint_progression_ppo(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    core = _core_imports()
    workspace_root = resolve_workspace_path(str(args.workspace_root), Path.cwd())
    out_root = resolve_workspace_path(str(args.out_root), workspace_root)
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = parse_csv(str(args.datasets))
    target_nfes = parse_int_csv(str(args.target_nfes))
    solvers = parse_csv(str(args.solvers))
    validate_n_initial(args.n_initial)
    run_config = {
        "artifact": "joint_progression_ppo_run_config_v1",
        "datasets": datasets,
        "target_nfes": target_nfes,
        "solvers": solvers,
        "basis_kind": DEFAULT_BASIS_KIND,
        "basis_dim": 5,
        "ppo_batch_size": int(args.ppo_batch_size),
        "ppo_updates": int(args.ppo_updates),
        "ppo_epochs": int(args.ppo_epochs),
        "ppo_budget_per_cell": ppo_budget_total(int(args.ppo_batch_size), int(args.ppo_updates)),
        "ppo_seed": int(args.ppo_seed),
        "clip_eps": float(args.clip_eps),
        "lr_actor": float(args.lr_actor),
        "lr_value": float(args.lr_value),
        "entropy_coef_start": float(args.entropy_coef_start),
        "entropy_coef_end": float(args.entropy_coef_end),
        "target_policy_kl": float(args.target_policy_kl),
        "max_policy_kl": float(args.max_policy_kl),
        "lambda_bad": float(args.lambda_bad),
        "elite_fraction": float(args.elite_fraction),
        "awr_w_max": float(args.awr_w_max),
        "init_std_min": float(args.init_std_min),
        "init_std_max": float(args.init_std_max),
        "selector_top_k": int(args.selector_top_k),
        "calibration_fraction": float(args.calibration_fraction),
        "calibration_windows": int(args.calibration_windows),
        "selector_windows": int(args.selector_windows),
        "bo_val_windows": int(args.bo_val_windows),
        "confirm_top_k": int(args.confirm_top_k),
        "reference_macro_factor": float(args.reference_macro_factor),
        "calibration_trace_samples": int(args.calibration_trace_samples),
        "num_eval_samples": int(args.num_eval_samples),
        "final_test_seeds": parse_int_csv(str(args.final_test_seeds)),
        "final_test_windows": int(args.final_test_windows),
        "bo_budget": int(args.bo_budget),
        "n_initial": int(args.n_initial),
        "bo_batch_size": int(args.bo_batch_size),
        "lambda_kl": float(args.lambda_kl),
        "theta_bound": float(args.theta_bound),
        "raw_samples": int(args.raw_samples),
        "num_restarts": int(args.num_restarts),
        "mc_samples": int(args.mc_samples),
        "bo_seed": int(args.bo_seed),
        "density_floor_eta": float(args.density_floor_eta),
        "device": str(args.device),
        "comparison_schedules": list(PPO_COMPARISON_SCHEDULES),
    }
    write_json(run_config, out_root / "run_config.json")

    device = torch.device(str(args.device))
    cell_summaries: List[Dict[str, Any]] = []
    all_final_rows: List[Dict[str, Any]] = []
    checkpoints: Dict[str, Mapping[str, Any]] = {}
    final_path = out_root / "final_comparison_rows.json"
    for dataset in datasets:
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = str(dataset)
        checkpoints[str(dataset)] = _load_checkpoint(dataset_args, workspace_root, device)
        checkpoint = checkpoints[str(dataset)]
        for target_nfe in target_nfes:
            for solver_idx, solver_key in enumerate(solvers):
                started = time.time()
                cdir = cell_dir(out_root, str(dataset), int(target_nfe), str(solver_key))
                cdir.mkdir(parents=True, exist_ok=True)
                bo_solver_dir = _ensure_bo_warmstart(
                    args=args,
                    core=core,
                    checkpoint=checkpoint,
                    dataset=str(dataset),
                    target_nfe=int(target_nfe),
                    solver_key=str(solver_key),
                    solver_idx=int(solver_idx),
                    cell_out=cdir,
                )
                bo_artifacts = load_bo_artifacts(bo_solver_dir)
                result = _run_ppo_training_for_cell(
                    args=args,
                    core=core,
                    checkpoint=checkpoint,
                    dataset=str(dataset),
                    target_nfe=int(target_nfe),
                    solver_key=str(solver_key),
                    solver_idx=int(solver_idx),
                    cell_out=cdir,
                    bo_artifacts=bo_artifacts,
                )
                cell_final_rows = _final_rows_for_cell(
                    args=args,
                    core=core,
                    checkpoint=checkpoint,
                    cell_result=result,
                    dataset=str(dataset),
                    target_nfe=int(target_nfe),
                    solver_key=str(solver_key),
                    final_path=final_path,
                )
                all_final_rows = _merge_global_final_rows(all_final_rows, cell_final_rows)
                write_json({"artifact": "joint_progression_ppo_final_comparison_rows_v1", "rows": all_final_rows}, final_path)
                summary = {
                    "dataset": str(dataset),
                    "target_nfe": int(target_nfe),
                    "solver_key": str(solver_key),
                    "train_rows": int(result["train_rows"]),
                    "selector_rows": int(result["selector_rows"]),
                    "best_selector_reward": float(result["ppo_best"]["reward"]),
                    "best_selector_avg_relative_ratio": float(result["ppo_best"]["avg_relative_ratio"]),
                    "best_selector_kl_to_reference": float(result["ppo_best"]["kl_to_reference"]),
                    "elapsed_seconds": float(time.time() - started),
                }
                cell_summaries.append(summary)
                write_json({"artifact": "joint_progression_ppo_run_summary_v1", "cells": cell_summaries}, out_root / "run_summary.json")
                _write_csv_rows(cell_summaries, out_root / "combined_summary.csv")
    if all_final_rows:
        final_summary = _summarize_global_final_rows(all_final_rows)
        write_json({"artifact": "joint_progression_ppo_final_summary_v1", "summaries": final_summary}, out_root / "final_summary.json")
        write_csv(all_final_rows, out_root / "final_comparison_rows.csv")
        _write_csv_rows(final_summary, out_root / "final_summary.csv")
    payload = {
        "artifact": "joint_progression_ppo_run_summary_v1",
        "run_config": run_config,
        "cell_count": int(len(cell_summaries)),
        "cells": cell_summaries,
        "final_row_count": int(len(all_final_rows)),
    }
    write_json(payload, out_root / "run_summary.json")
    return payload


def add_run_forecast_joint_progression_ppo_parser(subparsers: Any) -> None:
    run = subparsers.add_parser("run-forecast-joint-progression-ppo", help="Run forecast joint progression KL-PPO schedule search.")
    run.add_argument("--workspace-root", type=Path, default=Path.cwd())
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument("--datasets", type=str, required=True)
    run.add_argument("--target-nfes", type=str, required=True)
    run.add_argument("--solvers", type=str, default="euler,heun,midpoint_rk2,dpmpp2m")
    run.add_argument("--otflow-train-steps", type=int, default=20000)
    run.add_argument("--ppo-batch-size", type=int, default=8)
    run.add_argument("--ppo-updates", type=int, default=20)
    run.add_argument("--ppo-epochs", type=int, default=3)
    run.add_argument("--clip-eps", type=float, default=0.10)
    run.add_argument("--lr-actor", type=float, default=3e-4)
    run.add_argument("--lr-value", type=float, default=1e-3)
    run.add_argument("--entropy-coef-start", type=float, default=1e-3)
    run.add_argument("--entropy-coef-end", type=float, default=1e-4)
    run.add_argument("--target-policy-kl", type=float, default=0.01)
    run.add_argument("--max-policy-kl", type=float, default=0.03)
    run.add_argument("--lambda-bad", type=float, default=DEFAULT_LAMBDA_BAD)
    run.add_argument("--elite-fraction", type=float, default=0.20)
    run.add_argument("--awr-w-max", type=float, default=10.0)
    run.add_argument("--init-std-min", type=float, default=0.03)
    run.add_argument("--init-std-max", type=float, default=0.20)
    run.add_argument("--selector-top-k", type=int, default=5)
    run.add_argument("--calibration-fraction", type=float, default=0.7)
    run.add_argument("--calibration-windows", type=int, default=64)
    run.add_argument("--selector-windows", type=int, default=0)
    run.add_argument("--bo-val-windows", type=int, default=64)
    run.add_argument("--confirm-top-k", type=int, default=5)
    run.add_argument("--num-eval-samples", type=int, default=5)
    run.add_argument("--reference-macro-factor", type=float, default=16.0)
    run.add_argument("--calibration-trace-samples", type=int, default=1)
    run.add_argument("--final-test-seeds", type=str, default="0,1,2")
    run.add_argument("--final-test-windows", type=int, default=0)
    run.add_argument("--bo-budget", type=int, default=100)
    run.add_argument("--n-initial", type=int, default=16)
    run.add_argument("--bo-batch-size", type=int, default=2)
    run.add_argument("--lambda-kl", type=float, default=0.05)
    run.add_argument("--theta-bound", type=float, default=3.0)
    run.add_argument("--raw-samples", type=int, default=128)
    run.add_argument("--num-restarts", type=int, default=16)
    run.add_argument("--mc-samples", type=int, default=128)
    run.add_argument("--bo-seed", type=int, default=0)
    run.add_argument("--ppo-seed", type=int, default=0)
    run.add_argument("--density-floor-eta", type=float, default=0.05)
    run.add_argument("--device", type=str, default="cuda")
    run.add_argument("--resume", action="store_true", default=True)
    run.add_argument("--no-resume", dest="resume", action="store_false")
