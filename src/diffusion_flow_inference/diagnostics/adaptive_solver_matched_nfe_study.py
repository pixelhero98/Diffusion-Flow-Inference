from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from diffusion_flow_inference.schedules.diffusion_flow import FLOW_TIME_SCHEDULE_KEYS, TRANSFER_SCHEDULE_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "adaptive_solver_matched_nfe"
DEFAULT_LOB_RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "adaptive_solver_matched_nfe_lob"
DEFAULT_ZIP_PATH = DEFAULT_OUTPUT_ROOT / "20k.zip"
DEFAULT_BACKBONE_MANIFEST = DEFAULT_OUTPUT_ROOT / "backbone_matrix" / "backbone_manifest.json"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_ROOT / "figures"
DEFAULT_LOB_BASELINE_ROWS_PATH = DEFAULT_OUTPUT_ROOT / "lob_baseline_main" / "rows.jsonl"
DEFAULT_PLOT_POINTS_CSV = DEFAULT_RESULTS_DIR / "adaptive_matched_nfe_plot_points.csv"
DEFAULT_PLOT_DIAGNOSTICS_JSON = DEFAULT_RESULTS_DIR / "adaptive_matched_nfe_plot_diagnostics.json"
DEFAULT_PLOT_PNG = DEFAULT_FIGURE_DIR / "adaptive_matched_nfe_vs_target_performance_600dpi.png"
DEFAULT_PLOT_PDF = DEFAULT_FIGURE_DIR / "adaptive_matched_nfe_vs_target_performance_600dpi.pdf"
DEFAULT_LOB_PLOT_POINTS_CSV = DEFAULT_LOB_RESULTS_DIR / "adaptive_lob_matched_nfe_plot_points.csv"
DEFAULT_LOB_PLOT_DIAGNOSTICS_JSON = DEFAULT_LOB_RESULTS_DIR / "adaptive_lob_matched_nfe_plot_diagnostics.json"
DEFAULT_COMBINED_PLOT_PNG = DEFAULT_FIGURE_DIR / "adaptive_matched_nfe_extrapolation_conditional_1x2_600dpi.png"
DEFAULT_COMBINED_PLOT_PDF = DEFAULT_FIGURE_DIR / "adaptive_matched_nfe_extrapolation_conditional_1x2_600dpi.pdf"

RAW_STATS_MEMBER = "20k/seed_stats/forecast_baseline_raw_seed_stats.csv"
RELATIVE_STATS_MEMBER = "20k/seed_stats/forecast_baseline_relative_seed_stats.csv"
DATASETS = (
    "electricity",
    "london_smart_meters_wo_missing",
    "san_francisco_traffic",
    "solar_energy_10m",
    "wind_farms_wo_missing",
)
LOB_DATASETS = ("cryptos", "es_mbp_10", "sleep_edf")
TARGET_NFES = (10, 12, 16)
ADAPTIVE_SOLVERS = ("rk45_adaptive", "dopri5_adaptive")
TRANSFER_SCHEDULES = TRANSFER_SCHEDULE_KEYS
FLOW_TIME_SCHEDULES = FLOW_TIME_SCHEDULE_KEYS
FLOW_TIME_EXCLUSION_LABEL = ",".join(sorted(FLOW_TIME_SCHEDULES))
RTOLS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)
SEEDS = (0, 1, 2, 3, 4)
LOB_SEEDS = (0, 1, 2)
DATASET_LABELS = {
    "electricity": "Electricity",
    "london_smart_meters_wo_missing": "London SM",
    "san_francisco_traffic": "SF traffic",
    "solar_energy_10m": "Solar",
    "wind_farms_wo_missing": "Wind",
    "cryptos": "Cryptos",
    "es_mbp_10": "ES-MBP",
    "sleep_edf": "Sleep-EDF",
}
DATASET_MARKERS = {
    "electricity": "o",
    "london_smart_meters_wo_missing": "s",
    "san_francisco_traffic": "^",
    "solar_energy_10m": "D",
    "wind_farms_wo_missing": "P",
}
SOLVER_LABELS = {
    "rk45_adaptive": "RK45",
    "dopri5_adaptive": "Dopri5",
}
SOLVER_COLORS = {
    "rk45_adaptive": "#0072B2",
    "dopri5_adaptive": "#D55E00",
}
TARGET_NFE_SIZES = {
    10: 24.0,
    12: 38.0,
    16: 56.0,
}
TARGET_NFE_MARKERS = {
    10: "o",
    12: "s",
    16: "^",
}


def parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in parse_csv(text)]


def parse_float_csv(text: str) -> List[float]:
    return [float(part) for part in parse_csv(text)]


def fmt_float_key(value: float) -> str:
    return f"{float(value):.8g}".replace(".", "p").replace("-", "m")


def adaptive_atol_for_rtol(rtol: float) -> float:
    return float(max(1e-6, 1e-3 * float(rtol)))


def read_zip_csv(zip_path: Path, member: str) -> List[Dict[str, str]]:
    with zipfile.ZipFile(Path(zip_path)) as zf:
        text = zf.read(str(member)).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def write_csv_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def append_jsonl(row: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row: Mapping[str, Any]) -> Tuple[str, int, str, str]:
    return (
        str(row["dataset"]),
        int(row["seed"]),
        str(row["solver_key"]),
        fmt_float_key(float(row["rtol"])),
    )


def dedup_complete_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    by_key: Dict[Tuple[str, int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("row_status", "")) != "complete":
            continue
        by_key[row_key(row)] = row
    return [by_key[key] for key in sorted(by_key)]


def fixed_realized_nfe(solver_key: str, target_nfe: int) -> int:
    return int(target_nfe)


def extract_fixed_targets(zip_path: Path) -> List[Dict[str, Any]]:
    raw_rows = read_zip_csv(zip_path, RAW_STATS_MEMBER)
    rel_rows = read_zip_csv(zip_path, RELATIVE_STATS_MEMBER)
    raw_by_key = {
        (str(row["dataset"]), int(float(row["target_nfe"])), str(row["solver_key"]), str(row["schedule_key"])): row
        for row in raw_rows
    }
    targets: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rel_rows:
        schedule_key = str(row["schedule_key"])
        if schedule_key not in TRANSFER_SCHEDULES:
            continue
        item = dict(row)
        item["target_nfe"] = int(float(item["target_nfe"]))
        item["relative_crps_vs_uniform_mean"] = float(item["relative_crps_vs_uniform_mean"])
        item["relative_mase_vs_uniform_mean"] = float(item["relative_mase_vs_uniform_mean"])
        item["fixed_avg_relative_score"] = 0.5 * (
            float(item["relative_crps_vs_uniform_mean"]) + float(item["relative_mase_vs_uniform_mean"])
        )
        grouped[(str(item["dataset"]), int(item["target_nfe"]))].append(item)
    for key in sorted(grouped):
        dataset, target_nfe = key
        best = min(
            grouped[key],
            key=lambda row: (
                float(row["fixed_avg_relative_score"]),
                str(row["solver_key"]),
                str(row["schedule_key"]),
            ),
        )
        fixed_raw = raw_by_key[(dataset, target_nfe, str(best["solver_key"]), str(best["schedule_key"]))]
        uniform_raw = raw_by_key[(dataset, target_nfe, str(best["solver_key"]), "uniform")]
        targets.append(
            {
                "dataset": dataset,
                "target_nfe": int(target_nfe),
                "fixed_solver_key": str(best["solver_key"]),
                "fixed_solver_name": str(best["solver_name"]),
                "fixed_schedule_key": str(best["schedule_key"]),
                "fixed_schedule_name": str(best["schedule_name"]),
                "fixed_realized_nfe": fixed_realized_nfe(str(best["solver_key"]), int(target_nfe)),
                "fixed_relative_crps": float(best["relative_crps_vs_uniform_mean"]),
                "fixed_relative_mase": float(best["relative_mase_vs_uniform_mean"]),
                "fixed_avg_relative_score": float(best["fixed_avg_relative_score"]),
                "fixed_crps_mean": float(fixed_raw["crps_mean"]),
                "fixed_mase_mean": float(fixed_raw["mase_mean"]),
                "uniform_solver_key": str(best["solver_key"]),
                "uniform_crps_mean": float(uniform_raw["crps_mean"]),
                "uniform_mase_mean": float(uniform_raw["mase_mean"]),
                "selection_scope": "best_ays_gits_ots_per_dataset_nfe",
                "excluded_schedule_keys": FLOW_TIME_EXCLUSION_LABEL,
            }
        )
    expected = len(DATASETS) * len(TARGET_NFES)
    if len(targets) != expected:
        raise ValueError(f"Expected {expected} fixed targets, got {len(targets)}")
    return targets


def _safe_positive_ratio(numerator: Any, denominator: Any) -> float:
    numerator_f = float(numerator)
    denominator_f = float(denominator)
    if denominator_f <= 0.0 or not math.isfinite(denominator_f):
        raise ValueError(f"Positive denominator required for relative score, got {denominator!r}")
    if numerator_f <= 0.0 or not math.isfinite(numerator_f):
        raise ValueError(f"Positive numerator required for relative score, got {numerator!r}")
    return float(numerator_f / denominator_f)


def _higher_is_better_relative_ratio(uniform_value: Any, value: Any) -> float:
    uniform_f = float(uniform_value)
    value_f = float(value)
    if uniform_f < 0.0 or value_f < 0.0 or not math.isfinite(uniform_f) or not math.isfinite(value_f):
        raise ValueError(f"Nonnegative finite values required for higher-is-better relative score, got {uniform_value!r}, {value!r}")
    eps = 1e-12
    if uniform_f <= eps and value_f <= eps:
        return 1.0
    return float(max(uniform_f, eps) / max(value_f, eps))


def lob_average_relative_score(
    *,
    conditional_w1: Any,
    tstr_macro_f1: Any,
    uniform_conditional_w1: Any,
    uniform_tstr_macro_f1: Any,
) -> Tuple[float, float, float]:
    relative_cw1 = _safe_positive_ratio(conditional_w1, uniform_conditional_w1)
    relative_tstr = _higher_is_better_relative_ratio(uniform_tstr_macro_f1, tstr_macro_f1)
    return float(relative_cw1), float(relative_tstr), float(0.5 * (relative_cw1 + relative_tstr))


def _resolve_lob_baseline_rows_path(rows_path: Path) -> Path:
    return Path(rows_path)


def extract_lob_fixed_targets(rows_path: Path = DEFAULT_LOB_BASELINE_ROWS_PATH) -> List[Dict[str, Any]]:
    rows = read_jsonl(_resolve_lob_baseline_rows_path(Path(rows_path)))
    locked_rows = [
        dict(row)
        for row in rows
        if str(row.get("row_status")) == "complete"
        and str(row.get("benchmark_family")) == "lob_conditional_generation"
        and str(row.get("split_phase")) == "locked_test"
        and str(row.get("dataset")) in set(LOB_DATASETS)
        and int(row.get("target_nfe")) in set(TARGET_NFES)
        and int(row.get("seed")) in set(LOB_SEEDS)
    ]
    uniform_by_key: Dict[Tuple[str, int, str, int], Mapping[str, Any]] = {}
    for row in locked_rows:
        if str(row.get("scheduler_key")) != "uniform":
            continue
        uniform_by_key[(str(row["dataset"]), int(row["seed"]), str(row["solver_key"]), int(row["target_nfe"]))] = row

    grouped: Dict[Tuple[str, int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in locked_rows:
        schedule_key = str(row.get("scheduler_key"))
        if schedule_key not in TRANSFER_SCHEDULES:
            continue
        key = (str(row["dataset"]), int(row["seed"]), str(row["solver_key"]), int(row["target_nfe"]))
        uniform = uniform_by_key.get(key)
        if uniform is None:
            raise ValueError(f"Missing uniform LOB baseline for {key}")
        relative_cw1, relative_tstr, avg_score = lob_average_relative_score(
            conditional_w1=row["conditional_w1"],
            tstr_macro_f1=row["tstr_macro_f1"],
            uniform_conditional_w1=uniform["conditional_w1"],
            uniform_tstr_macro_f1=uniform["tstr_macro_f1"],
        )
        item = dict(row)
        item["relative_cw1_vs_uniform"] = relative_cw1
        item["relative_tstr_f1_vs_uniform"] = relative_tstr
        item["fixed_avg_relative_score"] = avg_score
        item["uniform_conditional_w1"] = float(uniform["conditional_w1"])
        item["uniform_tstr_macro_f1"] = float(uniform["tstr_macro_f1"])
        grouped[(str(row["dataset"]), int(row["target_nfe"]), str(row["solver_key"]), schedule_key)].append(item)

    candidate_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        dataset, target_nfe, solver_key, schedule_key = key
        group = grouped[key]
        if sorted(int(row["seed"]) for row in group) != list(LOB_SEEDS):
            raise ValueError(f"LOB target candidate {key} does not contain seeds {LOB_SEEDS}")
        candidate_rows.append(
            {
                "dataset": dataset,
                "target_nfe": int(target_nfe),
                "fixed_solver_key": solver_key,
                "fixed_solver_name": str(group[0].get("solver_name", solver_key)),
                "fixed_schedule_key": schedule_key,
                "fixed_schedule_name": str(group[0].get("schedule_name") or group[0].get("scheduler_variant_name") or schedule_key),
                "fixed_realized_nfe": int(round(float(np.mean([float(row.get("realized_nfe", target_nfe)) for row in group])))),
                "fixed_relative_cw1": float(np.mean([float(row["relative_cw1_vs_uniform"]) for row in group])),
                "fixed_relative_tstr_f1": float(np.mean([float(row["relative_tstr_f1_vs_uniform"]) for row in group])),
                "fixed_avg_relative_score": float(np.mean([float(row["fixed_avg_relative_score"]) for row in group])),
                "fixed_conditional_w1_mean": float(np.mean([float(row["conditional_w1"]) for row in group])),
                "fixed_tstr_macro_f1_mean": float(np.mean([float(row["tstr_macro_f1"]) for row in group])),
                "uniform_conditional_w1_mean": float(np.mean([float(row["uniform_conditional_w1"]) for row in group])),
                "uniform_tstr_macro_f1_mean": float(np.mean([float(row["uniform_tstr_macro_f1"]) for row in group])),
                "n_seeds": int(len(group)),
                "seed_values": json.dumps(sorted(int(row["seed"]) for row in group)),
                "selection_scope": "best_ays_gits_ots_per_lob_dataset_nfe",
                "excluded_schedule_keys": f"{FLOW_TIME_EXCLUSION_LABEL},uniform",
            }
        )

    targets: List[Dict[str, Any]] = []
    by_target: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_target[(str(row["dataset"]), int(row["target_nfe"]))].append(row)
    for key in sorted(by_target):
        targets.append(
            min(
                by_target[key],
                key=lambda row: (
                    float(row["fixed_avg_relative_score"]),
                    str(row["fixed_solver_key"]),
                    str(row["fixed_schedule_key"]),
                ),
            )
        )
    expected = len(LOB_DATASETS) * len(TARGET_NFES)
    if len(targets) != expected:
        raise ValueError(f"Expected {expected} LOB fixed targets, got {len(targets)}")
    return targets


def expected_adaptive_row_count(
    datasets: Sequence[str] = DATASETS,
    solvers: Sequence[str] = ADAPTIVE_SOLVERS,
    rtols: Sequence[float] = RTOLS,
    seeds: Sequence[int] = SEEDS,
) -> int:
    return int(len(datasets) * len(solvers) * len(rtols) * len(seeds))


def expected_lob_adaptive_row_count(
    datasets: Sequence[str] = LOB_DATASETS,
    solvers: Sequence[str] = ADAPTIVE_SOLVERS,
    rtols: Sequence[float] = RTOLS,
    seeds: Sequence[int] = LOB_SEEDS,
) -> int:
    return expected_adaptive_row_count(datasets=datasets, solvers=solvers, rtols=rtols, seeds=seeds)


def empirical_crps(samples: np.ndarray, target: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    term1 = np.mean(np.abs(samples - target[None, :]), axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :])
    term2 = 0.5 * np.mean(pairwise, axis=(0, 1))
    return float(np.mean(term1 - term2))


def point_mase(pred: np.ndarray, target: np.ndarray, denom: float) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))) / max(float(denom), 1e-8))


class IndexSubset:
    def __init__(self, base: Any, indices: Sequence[int]):
        self.base = base
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        return self.base[self.indices[int(idx)]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def denormalize_block(self, block: np.ndarray, idx: int) -> np.ndarray:
        return self.base.denormalize_block(block, self.indices[int(idx)])

    def mase_denom(self, idx: int) -> float:
        return float(self.base.mase_denom(self.indices[int(idx)]))


def maybe_cap_dataset(ds: Any, *, test_windows: int, seed: int) -> Tuple[Any, int]:
    if int(test_windows) <= 0 or int(test_windows) >= len(ds):
        return ds, int(len(ds))
    rng = np.random.default_rng(int(seed))
    indices = sorted(int(x) for x in rng.choice(np.arange(len(ds)), size=int(test_windows), replace=False))
    return IndexSubset(ds, indices), int(len(indices))


def evaluate_adaptive_forecast_row(
    model: Any,
    ds: Any,
    cfg: Any,
    *,
    solver_key: str,
    seed: int,
    rtol: float,
    adaptive_initial_steps: int,
    adaptive_max_nfe: int,
    adaptive_safety: float,
    adaptive_min_step: float,
    num_eval_samples: int,
) -> Dict[str, Any]:
    import torch
    from diffusion_flow_inference.diagnostics.adaptive_noise_sampler_followup import _apply_sample_overrides, _restore_sample_overrides
    from diffusion_flow_inference.backbones.training.train_val import seed_all

    device = cfg.train.device
    crps_values: List[float] = []
    mse_values: List[float] = []
    mase_values: List[float] = []
    latencies: List[float] = []
    used_nfes: List[float] = []
    accepted_steps: List[float] = []
    rejected_steps: List[float] = []
    hit_max_count = 0
    atol = adaptive_atol_for_rtol(float(rtol))
    backup = _apply_sample_overrides(
        model,
        cfg,
        solver=str(solver_key),
        adaptive_rtol=float(rtol),
        adaptive_atol=float(atol),
        adaptive_safety=float(adaptive_safety),
        adaptive_min_step=float(adaptive_min_step),
        adaptive_max_nfe=int(adaptive_max_nfe),
        time_grid=(),
    )
    try:
        for example_idx in range(len(ds)):
            hist_t, tgt_t, fut_t, _ = ds[int(example_idx)]
            hist = hist_t[None].to(device).float()
            true_block_norm = torch.cat([tgt_t[None, :], fut_t], dim=0).cpu().numpy()
            true_block_raw = ds.denormalize_block(true_block_norm, int(example_idx)).reshape(-1)
            draws: List[np.ndarray] = []
            for sample_idx in range(int(num_eval_samples)):
                seed_all(int(seed) + 1000 * int(example_idx) + int(sample_idx))
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                pred_norm = model.sample_future(hist, steps=int(adaptive_initial_steps), solver=str(solver_key))
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                latencies.append(time.perf_counter() - start)
                stats = dict(getattr(model, "_last_sample_stats", {}) or {})
                if "mean_total_field_evals_per_rollout" not in stats:
                    raise RuntimeError(f"Adaptive solver {solver_key} did not publish sample NFE stats.")
                used_nfes.append(float(stats["mean_total_field_evals_per_rollout"]))
                accepted_steps.append(float(stats.get("accepted_steps", float("nan"))))
                rejected_steps.append(float(stats.get("rejected_steps", float("nan"))))
                hit_max_count += int(bool(stats.get("hit_max_nfe", False)))
                pred_raw = ds.denormalize_block(pred_norm[0].detach().cpu().numpy(), int(example_idx)).reshape(-1)
                draws.append(pred_raw.astype(np.float32))
            samples = np.stack(draws, axis=0)
            pred_mean = samples.mean(axis=0)
            mse_values.append(float(np.mean((pred_mean - true_block_raw) ** 2)))
            crps_values.append(empirical_crps(samples, true_block_raw))
            mase_values.append(point_mase(pred_mean, true_block_raw, ds.mase_denom(int(example_idx))))
    finally:
        _restore_sample_overrides(model, cfg, backup)
    latency_arr = np.asarray(latencies, dtype=np.float64)
    used_arr = np.asarray(used_nfes, dtype=np.float64)
    return {
        "crps": float(np.mean(np.asarray(crps_values, dtype=np.float64))),
        "mse": float(np.mean(np.asarray(mse_values, dtype=np.float64))),
        "mase": float(np.mean(np.asarray(mase_values, dtype=np.float64))),
        "latency_ms_per_sample": float(1000.0 * latency_arr.mean()) if latency_arr.size else float("nan"),
        "used_nfe_mean": float(used_arr.mean()) if used_arr.size else float("nan"),
        "used_nfe_std": float(used_arr.std(ddof=0)) if used_arr.size else float("nan"),
        "accepted_steps_mean": float(np.nanmean(np.asarray(accepted_steps, dtype=np.float64))),
        "rejected_steps_mean": float(np.nanmean(np.asarray(rejected_steps, dtype=np.float64))),
        "hit_max_nfe_count": int(hit_max_count),
        "eval_examples": int(len(ds)),
        "num_eval_samples": int(num_eval_samples),
        "adaptive_atol": float(atol),
    }


def runner_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    from diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization import build_argparser

    argv = [
        "--out_root",
        str(Path(args.rows_jsonl).resolve().parent / runner_context),
        "--forecast_datasets",
        ",".join(parse_csv(str(args.datasets))),
        "--lob_datasets",
        "",
        "--solver_names",
        "euler",
        "--target_nfe_values",
        "10",
        "--seeds",
        ",".join(str(seed) for seed in parse_int_csv(str(args.seeds))),
        "--num_eval_samples",
        str(int(args.num_eval_samples)),
    ]
    if str(getattr(args, "backbone_manifest", "")).strip():
        argv.extend(["--backbone_manifest", str(args.backbone_manifest)])
    return build_argparser().parse_args(argv)


def runner_lob_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    from diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization import build_argparser

    argv = [
        "--out_root",
        str(Path(args.rows_jsonl).resolve().parent / runner_context),
        "--forecast_datasets",
        "",
        "--lob_datasets",
        ",".join(parse_csv(str(args.datasets))),
        "--solver_names",
        "euler",
        "--target_nfe_values",
        "10",
        "--seeds",
        ",".join(str(seed) for seed in parse_int_csv(str(args.seeds))),
    ]
    if str(getattr(args, "backbone_manifest", "")).strip():
        argv.extend(["--backbone_manifest", str(args.backbone_manifest)])
    return build_argparser().parse_args(argv)


def collect_adaptive_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    import torch
    from diffusion_flow_inference.evaluation.support import load_forecast_checkpoint_splits
    from diffusion_flow_inference.common.paths import project_paper_dataset_root

    rows_jsonl = Path(args.rows_jsonl)
    existing = {row_key(row) for row in read_jsonl(rows_jsonl) if str(row.get("row_status", "")) == "complete"}
    rows = read_jsonl(rows_jsonl)
    datasets = parse_csv(str(args.datasets))
    solvers = parse_csv(str(args.solvers))
    seeds = parse_int_csv(str(args.seeds))
    rtols = parse_float_csv(str(args.rtols))
    cli_args = runner_cli_args(args)
    dataset_root = project_paper_dataset_root()
    device = torch.device(str(getattr(cli_args, "device", "cuda" if torch.cuda.is_available() else "cpu")))
    completed_now = 0
    for dataset in datasets:
        checkpoint = load_forecast_checkpoint_splits(
            cli_args=cli_args,
            dataset_root=dataset_root,
            shared_backbone_root=Path(str(cli_args.shared_backbone_root)).resolve(),
            dataset=str(dataset),
            device=device,
        )
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        test_ds, eval_examples = maybe_cap_dataset(
            splits["test"],
            test_windows=int(args.test_windows),
            seed=17_000 + DATASETS.index(str(dataset)),
        )
        for seed in seeds:
            evaluation_seed = 10000 + int(seed)
            for solver_key in solvers:
                for rtol in rtols:
                    candidate = {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "solver_key": str(solver_key),
                        "rtol": float(rtol),
                    }
                    if row_key(candidate) in existing:
                        continue
                    start = time.perf_counter()
                    try:
                        metrics = evaluate_adaptive_forecast_row(
                            model,
                            test_ds,
                            cfg,
                            solver_key=str(solver_key),
                            seed=int(evaluation_seed),
                            rtol=float(rtol),
                            adaptive_initial_steps=int(args.adaptive_initial_steps),
                            adaptive_max_nfe=int(args.adaptive_max_nfe),
                            adaptive_safety=float(args.adaptive_safety),
                            adaptive_min_step=float(args.adaptive_min_step),
                            num_eval_samples=int(args.num_eval_samples),
                        )
                        row_status = "complete"
                        error_message = ""
                    except Exception as exc:
                        metrics = {}
                        row_status = "failed"
                        error_message = f"{type(exc).__name__}: {exc}"
                    row = {
                        "artifact": "adaptive_solver_matched_nfe_row",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset": str(dataset),
                        "split_phase": "locked_test",
                        "checkpoint_id": str(checkpoint.get("checkpoint_id", "")),
                        "checkpoint_path": str(checkpoint.get("checkpoint_path", "")),
                        "backbone_name": "otflow",
                        "train_budget_label": "20k",
                        "train_steps": 20000,
                        "seed": int(seed),
                        "evaluation_seed": int(evaluation_seed),
                        "solver_key": str(solver_key),
                        "solver_name": "RK45 adaptive" if solver_key == "rk45_adaptive" else "Dopri5 adaptive",
                        "rtol": float(rtol),
                        "rtol_key": fmt_float_key(float(rtol)),
                        "adaptive_atol": metrics.get("adaptive_atol", adaptive_atol_for_rtol(float(rtol))),
                        "adaptive_initial_steps": int(args.adaptive_initial_steps),
                        "adaptive_max_nfe": int(args.adaptive_max_nfe),
                        "adaptive_safety": float(args.adaptive_safety),
                        "adaptive_min_step": float(args.adaptive_min_step),
                        "num_eval_samples": int(args.num_eval_samples),
                        "eval_examples": int(metrics.get("eval_examples", eval_examples)),
                        "test_windows": int(args.test_windows),
                        "crps": metrics.get("crps"),
                        "mse": metrics.get("mse"),
                        "mase": metrics.get("mase"),
                        "latency_ms_per_sample": metrics.get("latency_ms_per_sample"),
                        "used_nfe_mean": metrics.get("used_nfe_mean"),
                        "used_nfe_std": metrics.get("used_nfe_std"),
                        "accepted_steps_mean": metrics.get("accepted_steps_mean"),
                        "rejected_steps_mean": metrics.get("rejected_steps_mean"),
                        "hit_max_nfe_count": metrics.get("hit_max_nfe_count"),
                        "row_seconds": float(time.perf_counter() - start),
                        "row_status": row_status,
                        "error_message": error_message,
                    }
                    rows.append(row)
                    append_jsonl(row, rows_jsonl)
                    current_stats = aggregate_adaptive_seed_stats(rows)
                    if current_stats:
                        write_csv_rows(current_stats, Path(args.seed_stats_csv))
                    completed_now += int(row_status == "complete")
                    print(json.dumps({"row_status": row_status, "dataset": dataset, "seed": seed, "solver": solver_key, "rtol": rtol, "completed_now": completed_now}), flush=True)
                    if int(args.max_rows) > 0 and completed_now >= int(args.max_rows):
                        return rows
        del model
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def collect_adaptive_lob_nfe_diagnostics(
    model: Any,
    ds: Any,
    cfg: Any,
    *,
    horizon: int,
    macro_steps: int,
    chosen_t0s: Sequence[int],
    seed: int,
    solver: str,
) -> Dict[str, Any]:
    import torch
    from diffusion_flow_inference.diagnostics.adaptive_deterministic_refinement_followup import _append_rollout_context_features, _sample_eval_trace
    from diffusion_flow_inference.backbones.training.train_val import (
        _future_context_seq_for_window,
        _get_dataset_item_by_t,
        _parse_batch,
        _temporary_eval_seed,
        crop_history_window,
        resolve_context_length,
    )

    total_evals: List[float] = []
    accepted_steps: List[float] = []
    rejected_steps: List[float] = []
    trial_steps: List[float] = []
    hit_max_count = 0
    chosen = [int(t0) for t0 in chosen_t0s]
    for window_idx, t0 in enumerate(chosen):
        batch = _get_dataset_item_by_t(ds, int(t0))
        hist, _, _, _, _ = _parse_batch(batch)
        hist_t = hist[None, :, :].to(cfg.device).float()
        context_len = resolve_context_length(hist_t.shape[1], horizon=int(horizon), cfg=cfg)
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
            call_seed = int(seed) + int(window_idx) * int(horizon) + int(cursor)
            with _temporary_eval_seed(call_seed):
                x_block, trace, block_len = _sample_eval_trace(
                    model,
                    x_hist,
                    cond_t=cond_t,
                    steps=int(macro_steps),
                    solver=str(solver),
                )
            stats = dict(trace or {})
            if "mean_total_field_evals_per_rollout" not in stats:
                stats.update(dict(getattr(model, "_last_sample_stats", {}) or {}))
            if "mean_total_field_evals_per_rollout" not in stats:
                raise RuntimeError(f"Adaptive LOB solver {solver} did not publish NFE diagnostics.")
            total_evals.append(float(stats["mean_total_field_evals_per_rollout"]))
            accepted_steps.append(float(stats.get("accepted_steps", float("nan"))))
            rejected_steps.append(float(stats.get("rejected_steps", float("nan"))))
            trial_steps.append(float(stats.get("steps", stats.get("trial_steps", float("nan")))))
            hit_max_count += int(bool(stats.get("hit_max_nfe", False)))
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
    total_arr = np.asarray(total_evals, dtype=np.float64)
    return {
        "n_rollout_calls": int(total_arr.size),
        "mean_total_field_evals_per_rollout": float(total_arr.mean()) if total_arr.size else float("nan"),
        "std_total_field_evals_per_rollout": float(total_arr.std(ddof=0)) if total_arr.size else float("nan"),
        "accepted_steps_mean": float(np.nanmean(np.asarray(accepted_steps, dtype=np.float64))) if accepted_steps else float("nan"),
        "rejected_steps_mean": float(np.nanmean(np.asarray(rejected_steps, dtype=np.float64))) if rejected_steps else float("nan"),
        "trial_steps_mean": float(np.nanmean(np.asarray(trial_steps, dtype=np.float64))) if trial_steps else float("nan"),
        "hit_max_nfe_count": int(hit_max_count),
    }


def evaluate_adaptive_lob_row(
    model: Any,
    ds: Any,
    cfg: Any,
    *,
    eval_horizon: int,
    eval_windows: int,
    chosen_t0s: Sequence[int],
    solver_key: str,
    seed: int,
    rtol: float,
    adaptive_initial_steps: int,
    adaptive_max_nfe: int,
    adaptive_safety: float,
    adaptive_min_step: float,
) -> Dict[str, Any]:
    from diffusion_flow_inference.diagnostics.adaptive_deterministic_refinement_followup import _metric_bundle
    from diffusion_flow_inference.diagnostics.adaptive_noise_sampler_followup import _apply_sample_overrides, _restore_sample_overrides
    from diffusion_flow_inference.backbones.training.train_val import eval_many_windows

    time_grid = tuple(float(x) for x in np.linspace(0.0, 1.0, int(adaptive_initial_steps) + 1))
    adaptive_atol = adaptive_atol_for_rtol(float(rtol))
    backup = _apply_sample_overrides(
        model,
        cfg,
        solver=str(solver_key),
        time_grid=time_grid,
        adaptive_rtol=float(rtol),
        adaptive_atol=float(adaptive_atol),
        adaptive_max_nfe=int(adaptive_max_nfe),
        adaptive_safety=float(adaptive_safety),
        adaptive_min_step=float(adaptive_min_step),
    )
    try:
        start = time.perf_counter()
        result = eval_many_windows(
            ds,
            model,
            cfg,
            horizon=int(eval_horizon),
            nfe=int(adaptive_initial_steps),
            n_windows=int(eval_windows),
            seed=int(seed),
            horizons_eval=[int(eval_horizon)],
            chosen_t0s=chosen_t0s,
            generation_seed_base=int(seed),
            metrics_seed=int(seed),
            main_metrics_only=False,
        )
        eval_seconds = float(time.perf_counter() - start)
        diag = collect_adaptive_lob_nfe_diagnostics(
            model,
            ds,
            cfg,
            horizon=int(eval_horizon),
            macro_steps=int(adaptive_initial_steps),
            chosen_t0s=chosen_t0s,
            seed=int(seed),
            solver=str(solver_key),
        )
    finally:
        _restore_sample_overrides(model, cfg, backup)
    metric_bundle = _metric_bundle(result)
    return {
        **metric_bundle,
        "eval_seconds": float(eval_seconds),
        "eval_windows": int(eval_windows),
        "adaptive_atol": float(adaptive_atol),
        "used_nfe_mean": float(diag["mean_total_field_evals_per_rollout"]),
        "used_nfe_std": float(diag["std_total_field_evals_per_rollout"]),
        "accepted_steps_mean": float(diag["accepted_steps_mean"]),
        "rejected_steps_mean": float(diag["rejected_steps_mean"]),
        "hit_max_nfe_count": int(diag["hit_max_nfe_count"]),
        "diag": diag,
    }


def collect_adaptive_lob_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    import torch
    from diffusion_flow_inference.diagnostics.adaptive_noise_sampler_followup import _choose_valid_windows
    from diffusion_flow_inference.evaluation.support import load_lob_checkpoint_splits, resolved_eval_horizon, resolved_eval_windows

    rows_jsonl = Path(args.rows_jsonl)
    existing = {row_key(row) for row in read_jsonl(rows_jsonl) if str(row.get("row_status", "")) == "complete"}
    rows = read_jsonl(rows_jsonl)
    datasets = parse_csv(str(args.datasets))
    solvers = parse_csv(str(args.solvers))
    seeds = parse_int_csv(str(args.seeds))
    rtols = parse_float_csv(str(args.rtols))
    cli_args = runner_lob_cli_args(args)
    shared_backbone_root = Path(str(cli_args.shared_backbone_root)).resolve()
    device = torch.device(str(getattr(cli_args, "device", "cuda" if torch.cuda.is_available() else "cpu")))
    completed_now = 0
    for dataset_idx, dataset in enumerate(datasets):
        checkpoint = load_lob_checkpoint_splits(
            cli_args=cli_args,
            shared_backbone_root=shared_backbone_root,
            dataset=str(dataset),
            device=device,
        )
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        test_ds = splits["test"]
        eval_horizon = resolved_eval_horizon(cli_args, str(dataset))
        eval_windows = resolved_eval_windows(cli_args, str(dataset), "test")
        for seed in seeds:
            chosen_t0s = np.asarray(
                _choose_valid_windows(
                    test_ds,
                    horizon=int(eval_horizon),
                    n_windows=int(eval_windows),
                    seed=int(seed) + 1_000 * int(dataset_idx),
                ),
                dtype=np.int64,
            )
            evaluation_seed = 10000 + int(seed)
            for solver_idx, solver_key in enumerate(solvers):
                for rtol in rtols:
                    candidate = {
                        "dataset": str(dataset),
                        "seed": int(seed),
                        "solver_key": str(solver_key),
                        "rtol": float(rtol),
                    }
                    if row_key(candidate) in existing:
                        continue
                    start = time.perf_counter()
                    metrics_seed = int(evaluation_seed) + 1_000_000 * int(dataset_idx) + 10_000 * int(solver_idx)
                    try:
                        metrics = evaluate_adaptive_lob_row(
                            model,
                            test_ds,
                            cfg,
                            eval_horizon=int(eval_horizon),
                            eval_windows=int(len(chosen_t0s)),
                            chosen_t0s=chosen_t0s,
                            solver_key=str(solver_key),
                            seed=int(metrics_seed),
                            rtol=float(rtol),
                            adaptive_initial_steps=int(args.adaptive_initial_steps),
                            adaptive_max_nfe=int(args.adaptive_max_nfe),
                            adaptive_safety=float(args.adaptive_safety),
                            adaptive_min_step=float(args.adaptive_min_step),
                        )
                        row_status = "complete"
                        error_message = ""
                    except Exception as exc:
                        metrics = {}
                        row_status = "failed"
                        error_message = f"{type(exc).__name__}: {exc}"
                    row = {
                        "artifact": "adaptive_lob_solver_matched_nfe_row",
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset": str(dataset),
                        "split_phase": "locked_test",
                        "checkpoint_id": str(checkpoint.get("checkpoint_id", "")),
                        "checkpoint_path": str(checkpoint.get("checkpoint_path", "")),
                        "benchmark_family": "lob_conditional_generation",
                        "backbone_name": "otflow",
                        "train_budget_label": "20k",
                        "train_steps": 20000,
                        "seed": int(seed),
                        "evaluation_seed": int(evaluation_seed),
                        "metrics_seed": int(metrics_seed),
                        "solver_key": str(solver_key),
                        "solver_name": "RK45 adaptive" if solver_key == "rk45_adaptive" else "Dopri5 adaptive",
                        "rtol": float(rtol),
                        "rtol_key": fmt_float_key(float(rtol)),
                        "adaptive_atol": metrics.get("adaptive_atol", adaptive_atol_for_rtol(float(rtol))),
                        "adaptive_initial_steps": int(args.adaptive_initial_steps),
                        "adaptive_max_nfe": int(args.adaptive_max_nfe),
                        "adaptive_safety": float(args.adaptive_safety),
                        "adaptive_min_step": float(args.adaptive_min_step),
                        "eval_horizon": int(eval_horizon),
                        "eval_windows": int(metrics.get("eval_windows", len(chosen_t0s))),
                        "conditional_w1": metrics.get("conditional_w1"),
                        "tstr_macro_f1": metrics.get("tstr_macro_f1"),
                        "score_main": metrics.get("score_main"),
                        "efficiency_ms_per_sample": metrics.get("efficiency_ms_per_sample"),
                        "eval_seconds": metrics.get("eval_seconds"),
                        "used_nfe_mean": metrics.get("used_nfe_mean"),
                        "used_nfe_std": metrics.get("used_nfe_std"),
                        "accepted_steps_mean": metrics.get("accepted_steps_mean"),
                        "rejected_steps_mean": metrics.get("rejected_steps_mean"),
                        "hit_max_nfe_count": metrics.get("hit_max_nfe_count"),
                        "row_seconds": float(time.perf_counter() - start),
                        "row_status": row_status,
                        "error_message": error_message,
                    }
                    rows.append(row)
                    append_jsonl(row, rows_jsonl)
                    current_stats = aggregate_adaptive_lob_seed_stats(rows)
                    if current_stats:
                        write_csv_rows(current_stats, Path(args.seed_stats_csv))
                    completed_now += int(row_status == "complete")
                    print(
                        json.dumps(
                            {
                                "row_status": row_status,
                                "dataset": dataset,
                                "seed": seed,
                                "solver": solver_key,
                                "rtol": rtol,
                                "completed_now": completed_now,
                            }
                        ),
                        flush=True,
                    )
                    if int(args.max_rows) > 0 and completed_now >= int(args.max_rows):
                        return rows
        del model
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def aggregate_adaptive_seed_stats(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in dedup_complete_rows(rows):
        groups[(str(row["dataset"]), str(row["solver_key"]), fmt_float_key(float(row["rtol"])))].append(row)
    stats: List[Dict[str, Any]] = []
    for key in sorted(groups):
        dataset, solver_key, rtol_key = key
        group = groups[key]
        rtol = float(group[0]["rtol"])
        out: Dict[str, Any] = {
            "dataset": dataset,
            "solver_key": solver_key,
            "solver_name": str(group[0]["solver_name"]),
            "rtol": rtol,
            "rtol_key": rtol_key,
            "adaptive_atol": float(group[0]["adaptive_atol"]),
            "n_seeds": int(len(group)),
            "seed_values": json.dumps(sorted(int(row["seed"]) for row in group)),
            "eval_examples": int(group[0]["eval_examples"]),
        }
        for metric in ("crps", "mase", "mse", "latency_ms_per_sample", "used_nfe_mean", "accepted_steps_mean", "rejected_steps_mean"):
            values = np.asarray([float(row[metric]) for row in group if row.get(metric) not in (None, "")], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            out[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        out["hit_max_nfe_count"] = int(sum(int(row.get("hit_max_nfe_count") or 0) for row in group))
        stats.append(out)
    return stats


def aggregate_adaptive_lob_seed_stats(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in dedup_complete_rows(rows):
        groups[(str(row["dataset"]), str(row["solver_key"]), fmt_float_key(float(row["rtol"])))].append(row)
    stats: List[Dict[str, Any]] = []
    for key in sorted(groups):
        dataset, solver_key, rtol_key = key
        group = groups[key]
        rtol = float(group[0]["rtol"])
        out: Dict[str, Any] = {
            "dataset": dataset,
            "solver_key": solver_key,
            "solver_name": str(group[0]["solver_name"]),
            "rtol": rtol,
            "rtol_key": rtol_key,
            "adaptive_atol": float(group[0]["adaptive_atol"]),
            "n_seeds": int(len(group)),
            "seed_values": json.dumps(sorted(int(row["seed"]) for row in group)),
            "eval_horizon": int(group[0]["eval_horizon"]),
            "eval_windows": int(group[0]["eval_windows"]),
        }
        for metric in (
            "conditional_w1",
            "tstr_macro_f1",
            "score_main",
            "efficiency_ms_per_sample",
            "eval_seconds",
            "used_nfe_mean",
            "accepted_steps_mean",
            "rejected_steps_mean",
        ):
            values = np.asarray([float(row[metric]) for row in group if row.get(metric) not in (None, "")], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            out[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else float("nan")
        out["hit_max_nfe_count"] = int(sum(int(row.get("hit_max_nfe_count") or 0) for row in group))
        stats.append(out)
    return stats


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def summarize_matches(target_rows: Sequence[Mapping[str, Any]], adaptive_stats_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    adaptive_by_dataset_solver: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in adaptive_stats_rows:
        adaptive_by_dataset_solver[(str(row["dataset"]), str(row["solver_key"]))].append(dict(row))
    summary: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "artifact": "adaptive_solver_matched_nfe_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "match_metric": "avg_relative_rcrps_rmase",
        "target_count": len(target_rows),
        "adaptive_stat_rows": len(adaptive_stats_rows),
        "unmatched": [],
        "brackets": [],
    }
    for target in sorted(target_rows, key=lambda row: (str(row["dataset"]), int(row["target_nfe"]))):
        base: Dict[str, Any] = {
            "dataset": str(target["dataset"]),
            "target_nfe": int(target["target_nfe"]),
            "best_fixed_solver": str(target["fixed_solver_key"]),
            "best_fixed_schedule": str(target["fixed_schedule_key"]),
            "fixed_avg_relative_score": float(target["fixed_avg_relative_score"]),
            "fixed_realized_nfe": int(target["fixed_realized_nfe"]),
        }
        target_score = float(target["fixed_avg_relative_score"])
        for solver_key in ADAPTIVE_SOLVERS:
            candidates: List[Dict[str, Any]] = []
            for row in adaptive_by_dataset_solver.get((str(target["dataset"]), solver_key), []):
                adaptive_rcrps = float(row["crps_mean"]) / float(target["uniform_crps_mean"])
                adaptive_rmase = float(row["mase_mean"]) / float(target["uniform_mase_mean"])
                scored = dict(row)
                scored["adaptive_relative_crps"] = adaptive_rcrps
                scored["adaptive_relative_mase"] = adaptive_rmase
                scored["adaptive_avg_relative_score"] = 0.5 * (adaptive_rcrps + adaptive_rmase)
                candidates.append(scored)
            candidates.sort(key=lambda row: float(row["used_nfe_mean_mean"]))
            matches = [row for row in candidates if float(row["adaptive_avg_relative_score"]) <= target_score]
            prefix = "rk45" if solver_key == "rk45_adaptive" else "dopri5"
            if matches:
                best = min(matches, key=lambda row: (float(row["used_nfe_mean_mean"]), float(row["adaptive_avg_relative_score"])))
                base[f"{prefix}_matched"] = True
                base[f"{prefix}_matched_used_nfe"] = float(best["used_nfe_mean_mean"])
                base[f"{prefix}_matched_used_nfe_std"] = float(best["used_nfe_mean_std"])
                base[f"{prefix}_matched_rtol"] = float(best["rtol"])
                base[f"{prefix}_matched_atol"] = float(best["adaptive_atol"])
                base[f"{prefix}_matched_avg_relative_score"] = float(best["adaptive_avg_relative_score"])
                base[f"{prefix}_matched_relative_crps"] = float(best["adaptive_relative_crps"])
                base[f"{prefix}_matched_relative_mase"] = float(best["adaptive_relative_mase"])
            else:
                base[f"{prefix}_matched"] = False
                base[f"{prefix}_matched_used_nfe"] = ""
                base[f"{prefix}_matched_used_nfe_std"] = ""
                base[f"{prefix}_matched_rtol"] = ""
                base[f"{prefix}_matched_atol"] = ""
                base[f"{prefix}_matched_avg_relative_score"] = ""
                base[f"{prefix}_matched_relative_crps"] = ""
                base[f"{prefix}_matched_relative_mase"] = ""
                diagnostics["unmatched"].append(
                    {"dataset": str(target["dataset"]), "target_nfe": int(target["target_nfe"]), "solver_key": solver_key}
                )
            bracket = bracket_for_target(candidates, target_score)
            diagnostics["brackets"].append(
                {
                    "dataset": str(target["dataset"]),
                    "target_nfe": int(target["target_nfe"]),
                    "solver_key": solver_key,
                    "target_score": target_score,
                    **bracket,
                }
            )
        summary.append(base)
    diagnostics["matched_cells"] = int(
        sum(int(bool(row.get("rk45_matched"))) + int(bool(row.get("dopri5_matched"))) for row in summary)
    )
    diagnostics["total_cells"] = int(len(summary) * len(ADAPTIVE_SOLVERS))
    return summary, diagnostics


def summarize_lob_matches(target_rows: Sequence[Mapping[str, Any]], adaptive_stats_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    adaptive_by_dataset_solver: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in adaptive_stats_rows:
        adaptive_by_dataset_solver[(str(row["dataset"]), str(row["solver_key"]))].append(dict(row))
    summary: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "artifact": "adaptive_lob_solver_matched_nfe_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "match_metric": "avg_relative_cw1_tstr_f1",
        "target_count": len(target_rows),
        "adaptive_stat_rows": len(adaptive_stats_rows),
        "unmatched": [],
        "brackets": [],
    }
    for target in sorted(target_rows, key=lambda row: (str(row["dataset"]), int(row["target_nfe"]))):
        base: Dict[str, Any] = {
            "dataset": str(target["dataset"]),
            "target_nfe": int(target["target_nfe"]),
            "best_fixed_solver": str(target["fixed_solver_key"]),
            "best_fixed_schedule": str(target["fixed_schedule_key"]),
            "fixed_avg_relative_score": float(target["fixed_avg_relative_score"]),
            "fixed_realized_nfe": int(target["fixed_realized_nfe"]),
            "fixed_relative_cw1": float(target["fixed_relative_cw1"]),
            "fixed_relative_tstr_f1": float(target["fixed_relative_tstr_f1"]),
            "uniform_conditional_w1_mean": float(target["uniform_conditional_w1_mean"]),
            "uniform_tstr_macro_f1_mean": float(target["uniform_tstr_macro_f1_mean"]),
        }
        target_score = float(target["fixed_avg_relative_score"])
        for solver_key in ADAPTIVE_SOLVERS:
            candidates: List[Dict[str, Any]] = []
            for row in adaptive_by_dataset_solver.get((str(target["dataset"]), solver_key), []):
                adaptive_cw1, adaptive_tstr, adaptive_score = lob_average_relative_score(
                    conditional_w1=row["conditional_w1_mean"],
                    tstr_macro_f1=row["tstr_macro_f1_mean"],
                    uniform_conditional_w1=target["uniform_conditional_w1_mean"],
                    uniform_tstr_macro_f1=target["uniform_tstr_macro_f1_mean"],
                )
                scored = dict(row)
                scored["adaptive_relative_cw1"] = adaptive_cw1
                scored["adaptive_relative_tstr_f1"] = adaptive_tstr
                scored["adaptive_avg_relative_score"] = adaptive_score
                candidates.append(scored)
            candidates.sort(key=lambda row: float(row["used_nfe_mean_mean"]))
            matches = [row for row in candidates if float(row["adaptive_avg_relative_score"]) <= target_score]
            prefix = "rk45" if solver_key == "rk45_adaptive" else "dopri5"
            if matches:
                best = min(matches, key=lambda row: (float(row["used_nfe_mean_mean"]), float(row["adaptive_avg_relative_score"])))
                base[f"{prefix}_matched"] = True
                base[f"{prefix}_matched_used_nfe"] = float(best["used_nfe_mean_mean"])
                base[f"{prefix}_matched_used_nfe_std"] = float(best["used_nfe_mean_std"])
                base[f"{prefix}_matched_rtol"] = float(best["rtol"])
                base[f"{prefix}_matched_atol"] = float(best["adaptive_atol"])
                base[f"{prefix}_matched_avg_relative_score"] = float(best["adaptive_avg_relative_score"])
                base[f"{prefix}_matched_relative_cw1"] = float(best["adaptive_relative_cw1"])
                base[f"{prefix}_matched_relative_tstr_f1"] = float(best["adaptive_relative_tstr_f1"])
            else:
                base[f"{prefix}_matched"] = False
                base[f"{prefix}_matched_used_nfe"] = ""
                base[f"{prefix}_matched_used_nfe_std"] = ""
                base[f"{prefix}_matched_rtol"] = ""
                base[f"{prefix}_matched_atol"] = ""
                base[f"{prefix}_matched_avg_relative_score"] = ""
                base[f"{prefix}_matched_relative_cw1"] = ""
                base[f"{prefix}_matched_relative_tstr_f1"] = ""
                diagnostics["unmatched"].append(
                    {"dataset": str(target["dataset"]), "target_nfe": int(target["target_nfe"]), "solver_key": solver_key}
                )
            bracket = bracket_for_target(candidates, target_score)
            diagnostics["brackets"].append(
                {
                    "dataset": str(target["dataset"]),
                    "target_nfe": int(target["target_nfe"]),
                    "solver_key": solver_key,
                    "target_score": target_score,
                    **bracket,
                }
            )
        summary.append(base)
    diagnostics["matched_cells"] = int(
        sum(int(bool(row.get("rk45_matched"))) + int(bool(row.get("dopri5_matched"))) for row in summary)
    )
    diagnostics["total_cells"] = int(len(summary) * len(ADAPTIVE_SOLVERS))
    return summary, diagnostics


def bracket_for_target(candidates: Sequence[Mapping[str, Any]], target_score: float) -> Dict[str, Any]:
    if not candidates:
        return {"status": "missing_adaptive_candidates"}
    ordered = sorted(candidates, key=lambda row: float(row["used_nfe_mean_mean"]))
    previous: Optional[Mapping[str, Any]] = None
    for row in ordered:
        score = float(row["adaptive_avg_relative_score"])
        if score <= float(target_score):
            if previous is None:
                return {
                    "status": "already_matched_at_lowest_nfe",
                    "upper_used_nfe": float(row["used_nfe_mean_mean"]),
                    "upper_score": score,
                    "upper_rtol": float(row["rtol"]),
                }
            prev_score = float(previous["adaptive_avg_relative_score"])
            prev_nfe = float(previous["used_nfe_mean_mean"])
            row_nfe = float(row["used_nfe_mean_mean"])
            if abs(score - prev_score) < 1e-12:
                interpolated = row_nfe
            else:
                frac = (float(target_score) - prev_score) / (score - prev_score)
                interpolated = prev_nfe + frac * (row_nfe - prev_nfe)
            return {
                "status": "straddled",
                "lower_used_nfe": prev_nfe,
                "lower_score": prev_score,
                "lower_rtol": float(previous["rtol"]),
                "upper_used_nfe": row_nfe,
                "upper_score": score,
                "upper_rtol": float(row["rtol"]),
                "interpolated_used_nfe": float(interpolated),
            }
        previous = row
    nearest = min(ordered, key=lambda row: abs(float(row["adaptive_avg_relative_score"]) - float(target_score)))
    return {
        "status": "unmatched",
        "nearest_used_nfe": float(nearest["used_nfe_mean_mean"]),
        "nearest_score": float(nearest["adaptive_avg_relative_score"]),
        "nearest_rtol": float(nearest["rtol"]),
    }


def write_latex_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Fixed target & Fixed NFE & Fixed score & RK45 used NFE & Dopri5 used NFE \\",
        r"\midrule",
    ]
    for row in rows:
        dataset = str(row["dataset"]).replace("_", r"\_")
        fixed = f"{row['best_fixed_solver']}+{row['best_fixed_schedule']}".replace("_", r"\_")
        rk = "--" if row["rk45_matched_used_nfe"] == "" else f"{float(row['rk45_matched_used_nfe']):.1f}"
        dp = "--" if row["dopri5_matched_used_nfe"] == "" else f"{float(row['dopri5_matched_used_nfe']):.1f}"
        lines.append(
            f"{dataset} / {int(row['target_nfe'])} & {fixed} & {int(row['fixed_realized_nfe'])} & "
            f"{float(row['fixed_avg_relative_score']):.3f} & {rk} & {dp} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def read_json_payload(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return dict(json.load(fh))


def parse_bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _bracket_lookup(diagnostics: Mapping[str, Any]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for item in diagnostics.get("brackets", []) or []:
        if not isinstance(item, Mapping):
            continue
        lookup[(str(item["dataset"]), int(item["target_nfe"]), str(item["solver_key"]))] = dict(item)
    return lookup


def build_adaptive_matched_nfe_plot_points(
    summary_rows: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    brackets = _bracket_lookup(diagnostics)
    points: List[Dict[str, Any]] = []
    solver_specs = (
        ("rk45", "rk45_adaptive"),
        ("dopri5", "dopri5_adaptive"),
    )
    for row in sorted(summary_rows, key=lambda item: (str(item["dataset"]), int(item["target_nfe"]))):
        dataset = str(row["dataset"])
        target_nfe = int(row["target_nfe"])
        fixed_score = float(row["fixed_avg_relative_score"])
        for prefix, solver_key in solver_specs:
            matched = parse_bool_cell(row.get(f"{prefix}_matched"))
            bracket = brackets.get((dataset, target_nfe, solver_key), {})
            if matched:
                used_nfe = _finite_float_or_none(row.get(f"{prefix}_matched_used_nfe"))
                adaptive_score = _finite_float_or_none(row.get(f"{prefix}_matched_avg_relative_score"))
                rtol = _finite_float_or_none(row.get(f"{prefix}_matched_rtol"))
                bracket_status = str(bracket.get("status", "matched"))
                point_status = "matched"
            else:
                used_nfe = _finite_float_or_none(bracket.get("nearest_used_nfe"))
                adaptive_score = _finite_float_or_none(bracket.get("nearest_score"))
                rtol = _finite_float_or_none(bracket.get("nearest_rtol"))
                bracket_status = str(bracket.get("status", "unmatched"))
                point_status = "unmatched_censored"
                if used_nfe is None:
                    candidates = [
                        _finite_float_or_none(bracket.get(key))
                        for key in ("upper_used_nfe", "lower_used_nfe", "interpolated_used_nfe")
                    ]
                    candidates = [value for value in candidates if value is not None]
                    used_nfe = max(candidates) if candidates else None
            performance_match_ratio = None
            match_gap_percent = None
            match_class = "missing"
            if adaptive_score is not None and fixed_score > 0.0:
                performance_match_ratio = float(adaptive_score) / fixed_score
                match_gap_percent = 100.0 * (performance_match_ratio - 1.0)
                if performance_match_ratio < 1.0 - 1e-12:
                    match_class = "over_matched"
                elif performance_match_ratio > 1.0 + 1e-12:
                    match_class = "under_matched"
                else:
                    match_class = "matched_exact"
            points.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "target_nfe": target_nfe,
                    "adaptive_solver_key": solver_key,
                    "adaptive_solver_label": SOLVER_LABELS.get(solver_key, solver_key),
                    "matched": bool(matched),
                    "point_status": point_status,
                    "bracket_status": bracket_status,
                    "realized_nfe": "" if used_nfe is None else float(used_nfe),
                    "fixed_avg_relative_score": fixed_score,
                    "adaptive_avg_relative_score": "" if adaptive_score is None else float(adaptive_score),
                    "performance_match_ratio": "" if performance_match_ratio is None else float(performance_match_ratio),
                    "match_gap_percent": "" if match_gap_percent is None else float(match_gap_percent),
                    "match_class": match_class,
                    "adaptive_rtol": "" if rtol is None else float(rtol),
                    "best_fixed_solver": str(row.get("best_fixed_solver", "")),
                    "best_fixed_schedule": str(row.get("best_fixed_schedule", "")),
                    "fixed_realized_nfe": int(row.get("fixed_realized_nfe", target_nfe)),
                }
            )
    return points


def _axis_limits(values: Sequence[float], *, include: Sequence[float] = (), pad_fraction: float = 0.08) -> Tuple[float, float]:
    arr = np.asarray([float(value) for value in list(values) + list(include) if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if abs(hi - lo) < 1e-12:
        pad = max(abs(lo) * 0.05, 1.0)
    else:
        pad = (hi - lo) * float(pad_fraction)
    return lo - pad, hi + pad


def build_adaptive_matched_nfe_plot_diagnostics(points: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    finite_points = [point for point in points if _finite_float_or_none(point.get("realized_nfe")) is not None]
    matched_points = [point for point in points if parse_bool_cell(point.get("matched"))]
    ratio_points = [
        point for point in points if _finite_float_or_none(point.get("performance_match_ratio")) is not None
    ]
    over_matched_points = [
        point for point in ratio_points if float(point["performance_match_ratio"]) < 1.0 - 1e-12
    ]
    exact_target_points = [
        point for point in ratio_points if abs(float(point["performance_match_ratio"]) - 1.0) <= 1e-12
    ]
    under_matched_points = [
        point for point in ratio_points if float(point["performance_match_ratio"]) > 1.0 + 1e-12
    ]
    return {
        "artifact": "adaptive_matched_nfe_plot_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "point_count": int(len(points)),
        "finite_realized_nfe_points": int(len(finite_points)),
        "matched_points": int(len(matched_points)),
        "unmatched_censored_points": int(len(points) - len(matched_points)),
        "ratio_points": int(len(ratio_points)),
        "over_matched_points": int(len(over_matched_points)),
        "exact_target_points": int(len(exact_target_points)),
        "under_matched_points": int(len(under_matched_points)),
        "datasets": sorted({str(point["dataset"]) for point in points}),
        "target_nfes": sorted({int(point["target_nfe"]) for point in points}),
        "adaptive_solvers": sorted({str(point["adaptive_solver_key"]) for point in points}),
        "x_key": "realized_nfe",
        "y_key": "performance_match_ratio",
        "y_reference": 1.0,
        "y_interpretation": "lower_is_better; 1.0 equals fixed target performance",
    }


def configure_matplotlib_for_paper() -> None:
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.0,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def plot_adaptive_matched_nfe_figure(
    points: Sequence[Mapping[str, Any]],
    *,
    png_path: Path = DEFAULT_PLOT_PNG,
    pdf_path: Path = DEFAULT_PLOT_PDF,
    dpi: int = 600,
) -> Dict[str, str]:
    if not points:
        raise ValueError("Cannot plot adaptive matched-NFE figure with no points.")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    configure_matplotlib_for_paper()
    plot_points = [
        point
        for point in points
        if _finite_float_or_none(point.get("realized_nfe")) is not None
        and _finite_float_or_none(point.get("performance_match_ratio")) is not None
    ]
    if not plot_points:
        raise ValueError("No finite realized NFE and performance-ratio values are available for plotting.")
    x = np.asarray([float(point["realized_nfe"]) for point in plot_points], dtype=np.float64)
    y = np.asarray([float(point["performance_match_ratio"]) for point in plot_points], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.2, 2.5))
    for solver_key in ADAPTIVE_SOLVERS:
        for target_nfe in TARGET_NFES:
            selected = [
                point
                for point in plot_points
                if str(point["adaptive_solver_key"]) == solver_key and int(point["target_nfe"]) == int(target_nfe)
            ]
            if not selected:
                continue
            ax.scatter(
                [float(point["realized_nfe"]) for point in selected],
                [float(point["performance_match_ratio"]) for point in selected],
                marker=TARGET_NFE_MARKERS[int(target_nfe)],
                s=TARGET_NFE_SIZES[int(target_nfe)],
                facecolor=SOLVER_COLORS[solver_key],
                edgecolor="white",
                linewidth=0.75,
                alpha=0.84,
                zorder=3,
            )

    ax.axhline(1.0, color="#222222", linewidth=1.2, linestyle="--", zorder=0)

    ax.text(
        0.97,
        0.04,
        f"n = {len(points)}\nmatched = {sum(parse_bool_cell(point.get('matched')) for point in points)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cfcfcf", "linewidth": 0.7},
    )
    ax.set_xlabel("Adaptive realized NFE", fontsize=12.0, labelpad=3.0)
    ax.set_ylabel("Adaptive / target\nperformance ratio", fontsize=11.6, labelpad=6.0)
    ax.set_xlim(*_axis_limits(x, pad_fraction=0.08))
    ax.set_ylim(*_axis_limits(y, include=(1.0,), pad_fraction=0.12))
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=10.4, pad=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.125, right=0.78, bottom=0.27, top=0.95)

    solver_handles = [
        mlines.Line2D(
            [],
            [],
            color=SOLVER_COLORS[solver_key],
            marker="o",
            linestyle="None",
            markersize=6.8,
            label=SOLVER_LABELS[solver_key],
        )
        for solver_key in ADAPTIVE_SOLVERS
    ]
    size_handles = [
        mlines.Line2D(
            [],
            [],
            color="#444444",
            marker=TARGET_NFE_MARKERS[int(target_nfe)],
            linestyle="None",
            markersize=math.sqrt(TARGET_NFE_SIZES[int(target_nfe)]),
            label=str(target_nfe),
        )
        for target_nfe in TARGET_NFES
    ]
    fig.legend(
        handles=solver_handles,
        title="Solver",
        loc="upper left",
        bbox_to_anchor=(0.80, 0.91),
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        labelspacing=0.28,
        handletextpad=0.42,
        fontsize=8.8,
        title_fontsize=9.4,
    )
    fig.legend(
        handles=size_handles,
        title="Target NFE",
        loc="upper left",
        bbox_to_anchor=(0.80, 0.50),
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        labelspacing=0.28,
        handletextpad=0.42,
        fontsize=8.8,
        title_fontsize=9.4,
    )
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=int(dpi))
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _finite_matched_plot_points(points: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        point
        for point in points
        if _finite_float_or_none(point.get("realized_nfe")) is not None
        and _finite_float_or_none(point.get("performance_match_ratio")) is not None
    ]


def plot_combined_adaptive_matched_nfe_figure(
    forecast_points: Sequence[Mapping[str, Any]],
    lob_points: Sequence[Mapping[str, Any]],
    *,
    png_path: Path = DEFAULT_COMBINED_PLOT_PNG,
    pdf_path: Path = DEFAULT_COMBINED_PLOT_PDF,
    dpi: int = 600,
) -> Dict[str, str]:
    if not forecast_points or not lob_points:
        raise ValueError("Combined adaptive matched-NFE figure requires non-empty forecast and LOB points.")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    configure_matplotlib_for_paper()
    panel_specs = (
        ("Extrapolation", _finite_matched_plot_points(forecast_points)),
        ("Conditional generation", _finite_matched_plot_points(lob_points)),
    )
    if any(not points for _, points in panel_specs):
        raise ValueError("Each combined figure panel must have finite points.")
    all_y = np.asarray(
        [float(point["performance_match_ratio"]) for _, points in panel_specs for point in points],
        dtype=np.float64,
    )
    y_limits = _axis_limits(all_y, include=(1.0,), pad_fraction=0.12)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.75), sharey=True)
    for ax, (title, points) in zip(axes, panel_specs):
        x = np.asarray([float(point["realized_nfe"]) for point in points], dtype=np.float64)
        for solver_key in ADAPTIVE_SOLVERS:
            for target_nfe in TARGET_NFES:
                selected = [
                    point
                    for point in points
                    if str(point["adaptive_solver_key"]) == solver_key and int(point["target_nfe"]) == int(target_nfe)
                ]
                if not selected:
                    continue
                ax.scatter(
                    [float(point["realized_nfe"]) for point in selected],
                    [float(point["performance_match_ratio"]) for point in selected],
                    marker=TARGET_NFE_MARKERS[int(target_nfe)],
                    s=TARGET_NFE_SIZES[int(target_nfe)],
                    facecolor=SOLVER_COLORS[solver_key],
                    edgecolor="white",
                    linewidth=0.75,
                    alpha=0.84,
                    zorder=3,
                )
        ax.axhline(1.0, color="#222222", linewidth=1.2, linestyle="--", zorder=0)
        ax.text(
            0.97,
            0.04,
            f"n = {len(points)}\nmatched = {sum(parse_bool_cell(point.get('matched')) for point in points)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cfcfcf", "linewidth": 0.7},
        )
        ax.set_title(title, fontsize=13.0, pad=7.0)
        ax.set_xlabel("Adaptive realized NFE", fontsize=12.0)
        ax.set_xlim(*_axis_limits(x, pad_fraction=0.08))
        ax.set_ylim(*y_limits)
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
        ax.tick_params(axis="both", labelsize=10.4, pad=2.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Adaptive / target performance ratio", fontsize=12.0)
    solver_handles = [
        mlines.Line2D(
            [],
            [],
            color=SOLVER_COLORS[solver_key],
            marker="o",
            linestyle="None",
            markersize=6.8,
            label=SOLVER_LABELS[solver_key],
        )
        for solver_key in ADAPTIVE_SOLVERS
    ]
    size_handles = [
        mlines.Line2D(
            [],
            [],
            color="#444444",
            marker=TARGET_NFE_MARKERS[int(target_nfe)],
            linestyle="None",
            markersize=math.sqrt(TARGET_NFE_SIZES[int(target_nfe)]),
            label=str(target_nfe),
        )
        for target_nfe in TARGET_NFES
    ]
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.14, top=0.90, wspace=0.22)
    fig.legend(
        handles=solver_handles,
        title="Solver",
        loc="upper left",
        bbox_to_anchor=(0.86, 0.87),
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        labelspacing=0.28,
        handletextpad=0.42,
        fontsize=9.4,
        title_fontsize=10.0,
    )
    fig.legend(
        handles=size_handles,
        title="Target NFE",
        loc="upper left",
        bbox_to_anchor=(0.86, 0.55),
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        labelspacing=0.28,
        handletextpad=0.42,
        fontsize=9.4,
        title_fontsize=10.0,
    )
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=int(dpi))
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the adaptive RK45/Dopri5 matched-NFE forecast study.")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-baseline")
    extract.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    extract.add_argument("--out-csv", type=Path, default=DEFAULT_RESULTS_DIR / "fixed_20k_targets.csv")

    extract_lob = sub.add_parser("extract-lob-baseline")
    extract_lob.add_argument("--rows-jsonl", type=Path, default=DEFAULT_LOB_BASELINE_ROWS_PATH)
    extract_lob.add_argument("--out-csv", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "fixed_20k_lob_targets.csv")

    dry = sub.add_parser("dry-run")
    dry.add_argument("--datasets", type=str, default=",".join(DATASETS))
    dry.add_argument("--solvers", type=str, default=",".join(ADAPTIVE_SOLVERS))
    dry.add_argument("--rtols", type=str, default=",".join(str(x) for x in RTOLS))
    dry.add_argument("--seeds", type=str, default=",".join(str(x) for x in SEEDS))

    dry_lob = sub.add_parser("dry-run-lob")
    dry_lob.add_argument("--datasets", type=str, default=",".join(LOB_DATASETS))
    dry_lob.add_argument("--solvers", type=str, default=",".join(ADAPTIVE_SOLVERS))
    dry_lob.add_argument("--rtols", type=str, default=",".join(str(x) for x in RTOLS))
    dry_lob.add_argument("--seeds", type=str, default=",".join(str(x) for x in LOB_SEEDS))

    collect = sub.add_parser("collect-adaptive")
    collect.add_argument("--rows-jsonl", type=Path, default=DEFAULT_RESULTS_DIR / "adaptive_rows.jsonl")
    collect.add_argument("--seed-stats-csv", type=Path, default=DEFAULT_RESULTS_DIR / "adaptive_seed_stats.csv")
    collect.add_argument("--datasets", type=str, default=",".join(DATASETS))
    collect.add_argument("--solvers", type=str, default=",".join(ADAPTIVE_SOLVERS))
    collect.add_argument("--rtols", type=str, default=",".join(str(x) for x in RTOLS))
    collect.add_argument("--seeds", type=str, default=",".join(str(x) for x in SEEDS))
    collect.add_argument("--backbone-manifest", type=Path, default=DEFAULT_BACKBONE_MANIFEST)
    collect.add_argument("--num-eval-samples", type=int, default=1)
    collect.add_argument("--test-windows", type=int, default=0)
    collect.add_argument("--adaptive-initial-steps", type=int, default=16)
    collect.add_argument("--adaptive-max-nfe", type=int, default=512)
    collect.add_argument("--adaptive-safety", type=float, default=0.9)
    collect.add_argument("--adaptive-min-step", type=float, default=1e-5)
    collect.add_argument("--max-rows", type=int, default=0)

    collect_lob = sub.add_parser("collect-adaptive-lob")
    collect_lob.add_argument("--rows-jsonl", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "adaptive_lob_rows.jsonl")
    collect_lob.add_argument("--seed-stats-csv", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "adaptive_lob_seed_stats.csv")
    collect_lob.add_argument("--datasets", type=str, default=",".join(LOB_DATASETS))
    collect_lob.add_argument("--solvers", type=str, default=",".join(ADAPTIVE_SOLVERS))
    collect_lob.add_argument("--rtols", type=str, default=",".join(str(x) for x in RTOLS))
    collect_lob.add_argument("--seeds", type=str, default=",".join(str(x) for x in LOB_SEEDS))
    collect_lob.add_argument("--backbone-manifest", type=Path, default=DEFAULT_BACKBONE_MANIFEST)
    collect_lob.add_argument("--adaptive-initial-steps", type=int, default=16)
    collect_lob.add_argument("--adaptive-max-nfe", type=int, default=512)
    collect_lob.add_argument("--adaptive-safety", type=float, default=0.9)
    collect_lob.add_argument("--adaptive-min-step", type=float, default=1e-5)
    collect_lob.add_argument("--max-rows", type=int, default=0)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--targets-csv", type=Path, default=DEFAULT_RESULTS_DIR / "fixed_20k_targets.csv")
    summarize.add_argument("--adaptive-seed-stats-csv", type=Path, default=DEFAULT_RESULTS_DIR / "adaptive_seed_stats.csv")
    summarize.add_argument("--summary-csv", type=Path, default=DEFAULT_RESULTS_DIR / "matched_used_nfe_summary.csv")
    summarize.add_argument("--summary-tex", type=Path, default=DEFAULT_RESULTS_DIR / "matched_used_nfe_summary.tex")
    summarize.add_argument("--diagnostics-json", type=Path, default=DEFAULT_RESULTS_DIR / "adaptive_matched_nfe_diagnostics.json")

    summarize_lob = sub.add_parser("summarize-lob")
    summarize_lob.add_argument("--targets-csv", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "fixed_20k_lob_targets.csv")
    summarize_lob.add_argument("--adaptive-seed-stats-csv", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "adaptive_lob_seed_stats.csv")
    summarize_lob.add_argument("--summary-csv", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "matched_used_nfe_lob_summary.csv")
    summarize_lob.add_argument("--summary-tex", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "matched_used_nfe_lob_summary.tex")
    summarize_lob.add_argument("--diagnostics-json", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "adaptive_lob_matched_nfe_diagnostics.json")

    plot = sub.add_parser("plot")
    plot.add_argument("--summary-csv", type=Path, default=DEFAULT_RESULTS_DIR / "matched_used_nfe_summary.csv")
    plot.add_argument("--match-diagnostics-json", type=Path, default=DEFAULT_RESULTS_DIR / "adaptive_matched_nfe_diagnostics.json")
    plot.add_argument("--points-csv", type=Path, default=DEFAULT_PLOT_POINTS_CSV)
    plot.add_argument("--plot-diagnostics-json", type=Path, default=DEFAULT_PLOT_DIAGNOSTICS_JSON)
    plot.add_argument("--png", type=Path, default=DEFAULT_PLOT_PNG)
    plot.add_argument("--pdf", type=Path, default=DEFAULT_PLOT_PDF)
    plot.add_argument("--dpi", type=int, default=600)

    combined = sub.add_parser("plot-combined")
    combined.add_argument("--forecast-points-csv", type=Path, default=DEFAULT_PLOT_POINTS_CSV)
    combined.add_argument("--lob-summary-csv", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "matched_used_nfe_lob_summary.csv")
    combined.add_argument("--lob-match-diagnostics-json", type=Path, default=DEFAULT_LOB_RESULTS_DIR / "adaptive_lob_matched_nfe_diagnostics.json")
    combined.add_argument("--lob-points-csv", type=Path, default=DEFAULT_LOB_PLOT_POINTS_CSV)
    combined.add_argument("--lob-plot-diagnostics-json", type=Path, default=DEFAULT_LOB_PLOT_DIAGNOSTICS_JSON)
    combined.add_argument("--png", type=Path, default=DEFAULT_COMBINED_PLOT_PNG)
    combined.add_argument("--pdf", type=Path, default=DEFAULT_COMBINED_PLOT_PDF)
    combined.add_argument("--dpi", type=int, default=600)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.command == "extract-baseline":
        targets = extract_fixed_targets(Path(args.zip_path))
        write_csv_rows(targets, Path(args.out_csv))
        print(json.dumps({"out_csv": str(Path(args.out_csv)), "rows": len(targets)}, indent=2, sort_keys=True))
        return
    if args.command == "extract-lob-baseline":
        targets = extract_lob_fixed_targets(Path(args.rows_jsonl))
        write_csv_rows(targets, Path(args.out_csv))
        print(json.dumps({"out_csv": str(Path(args.out_csv)), "rows": len(targets)}, indent=2, sort_keys=True))
        return
    if args.command == "dry-run":
        payload = {
            "datasets": parse_csv(str(args.datasets)),
            "solvers": parse_csv(str(args.solvers)),
            "rtols": parse_float_csv(str(args.rtols)),
            "seeds": parse_int_csv(str(args.seeds)),
        }
        payload["expected_adaptive_rows"] = expected_adaptive_row_count(
            payload["datasets"], payload["solvers"], payload["rtols"], payload["seeds"]
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "dry-run-lob":
        payload = {
            "datasets": parse_csv(str(args.datasets)),
            "solvers": parse_csv(str(args.solvers)),
            "rtols": parse_float_csv(str(args.rtols)),
            "seeds": parse_int_csv(str(args.seeds)),
        }
        payload["expected_adaptive_rows"] = expected_lob_adaptive_row_count(
            payload["datasets"], payload["solvers"], payload["rtols"], payload["seeds"]
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "collect-adaptive":
        rows = collect_adaptive_rows(args)
        stats = aggregate_adaptive_seed_stats(rows)
        if stats:
            write_csv_rows(stats, Path(args.seed_stats_csv))
        print(
            json.dumps(
                {
                    "rows_jsonl": str(Path(args.rows_jsonl)),
                    "seed_stats_csv": str(Path(args.seed_stats_csv)),
                    "rows": len(rows),
                    "complete_rows": sum(1 for row in rows if str(row.get("row_status")) == "complete"),
                    "seed_stats_rows": len(stats),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "collect-adaptive-lob":
        rows = collect_adaptive_lob_rows(args)
        stats = aggregate_adaptive_lob_seed_stats(rows)
        if stats:
            write_csv_rows(stats, Path(args.seed_stats_csv))
        print(
            json.dumps(
                {
                    "rows_jsonl": str(Path(args.rows_jsonl)),
                    "seed_stats_csv": str(Path(args.seed_stats_csv)),
                    "rows": len(rows),
                    "complete_rows": sum(1 for row in rows if str(row.get("row_status")) == "complete"),
                    "seed_stats_rows": len(stats),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "summarize":
        target_rows = load_csv_rows(Path(args.targets_csv))
        adaptive_rows = load_csv_rows(Path(args.adaptive_seed_stats_csv))
        summary, diagnostics = summarize_matches(target_rows, adaptive_rows)
        write_csv_rows(summary, Path(args.summary_csv))
        write_latex_summary(summary, Path(args.summary_tex))
        write_json(diagnostics, Path(args.diagnostics_json))
        print(json.dumps({"summary_csv": str(Path(args.summary_csv)), "rows": len(summary), "matched_cells": diagnostics["matched_cells"]}, indent=2, sort_keys=True))
        return
    if args.command == "summarize-lob":
        target_rows = load_csv_rows(Path(args.targets_csv))
        adaptive_rows = load_csv_rows(Path(args.adaptive_seed_stats_csv))
        summary, diagnostics = summarize_lob_matches(target_rows, adaptive_rows)
        write_csv_rows(summary, Path(args.summary_csv))
        write_latex_summary(summary, Path(args.summary_tex))
        write_json(diagnostics, Path(args.diagnostics_json))
        print(json.dumps({"summary_csv": str(Path(args.summary_csv)), "rows": len(summary), "matched_cells": diagnostics["matched_cells"]}, indent=2, sort_keys=True))
        return
    if args.command == "plot":
        summary_rows = load_csv_rows(Path(args.summary_csv))
        match_diagnostics = read_json_payload(Path(args.match_diagnostics_json))
        points = build_adaptive_matched_nfe_plot_points(summary_rows, match_diagnostics)
        plot_diagnostics = build_adaptive_matched_nfe_plot_diagnostics(points)
        write_csv_rows(points, Path(args.points_csv))
        write_json(plot_diagnostics, Path(args.plot_diagnostics_json))
        outputs = plot_adaptive_matched_nfe_figure(points, png_path=Path(args.png), pdf_path=Path(args.pdf), dpi=int(args.dpi))
        print(
            json.dumps(
                {
                    **outputs,
                    "points_csv": str(Path(args.points_csv)),
                    "plot_diagnostics_json": str(Path(args.plot_diagnostics_json)),
                    "points": len(points),
                    "matched_points": plot_diagnostics["matched_points"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "plot-combined":
        forecast_points = load_csv_rows(Path(args.forecast_points_csv))
        lob_summary_rows = load_csv_rows(Path(args.lob_summary_csv))
        lob_match_diagnostics = read_json_payload(Path(args.lob_match_diagnostics_json))
        lob_points = build_adaptive_matched_nfe_plot_points(lob_summary_rows, lob_match_diagnostics)
        lob_plot_diagnostics = build_adaptive_matched_nfe_plot_diagnostics(lob_points)
        write_csv_rows(lob_points, Path(args.lob_points_csv))
        write_json(lob_plot_diagnostics, Path(args.lob_plot_diagnostics_json))
        outputs = plot_combined_adaptive_matched_nfe_figure(
            forecast_points,
            lob_points,
            png_path=Path(args.png),
            pdf_path=Path(args.pdf),
            dpi=int(args.dpi),
        )
        print(
            json.dumps(
                {
                    **outputs,
                    "forecast_points": len(forecast_points),
                    "lob_points": len(lob_points),
                    "lob_points_csv": str(Path(args.lob_points_csv)),
                    "lob_plot_diagnostics_json": str(Path(args.lob_plot_diagnostics_json)),
                    "lob_matched_points": lob_plot_diagnostics["matched_points"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    main()
