from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PHASE_LABELS = {
    "ser_ptg_reference_center": "SER/PTG reference",
    "initial_sobol_kl_perturbation": "Initial KL perturbation",
    "qLogNoisyExpectedImprovement": "qLogNEI",
}
BASELINE_STYLE_ORDER = ("uniform", "ays", "gits", "ots", "ser_ptg_reference")
TOP_COLOR_CYCLE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_float_list(values: Any, *, name: str) -> List[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of numbers.")
    out: List[float] = []
    for value in values:
        number = _finite_float(value)
        if number is None:
            raise ValueError(f"{name} contains a non-finite value.")
        out.append(float(number))
    if len(out) < 2:
        raise ValueError(f"{name} must contain at least two nodes.")
    return out


def _schedule_intervals(grid: Sequence[float]) -> List[float]:
    arr = np.asarray(_as_float_list(grid, name="schedule_grid"), dtype=np.float64)
    widths = np.diff(arr)
    if np.any(widths <= 0.0):
        raise ValueError("schedule_grid must be strictly increasing.")
    return [float(x) for x in widths.tolist()]


def _safe_text(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path, *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _safe_text(row.get(field)) for field in fieldnames})


def _load_optional_json(path: Path) -> Dict[str, Any]:
    return load_json(path) if path.exists() else {}


def discover_solver_dirs(run_root: str | Path) -> List[Path]:
    root = Path(run_root)
    if not root.exists():
        raise ValueError(f"run_root does not exist: {root}")
    solvers = sorted(path for path in root.iterdir() if path.is_dir() and (path / "observations.json").exists())
    if not solvers:
        raise ValueError(f"No solver observation artifacts found under {root}.")
    return solvers


def load_solver_artifacts(solver_dir: str | Path) -> Dict[str, Any]:
    path = Path(solver_dir)
    observations_path = path / "observations.json"
    reference_path = path / "reference_schedule.json"
    if not observations_path.exists():
        raise ValueError(f"Missing required observations artifact: {observations_path}")
    if not reference_path.exists():
        raise ValueError(f"Missing required reference schedule artifact: {reference_path}")
    observations = load_json(observations_path)
    rows = observations.get("observations", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"observations.json has no observation rows: {observations_path}")
    return {
        "solver_key": path.name,
        "solver_dir": path,
        "observations": observations,
        "reference": load_json(reference_path),
        "best": _load_optional_json(path / "best_schedule.json"),
        "confirmation": _load_optional_json(path / "confirmation_rows.json"),
    }


def load_run_artifacts(run_root: str | Path) -> Dict[str, Any]:
    root = Path(run_root)
    solvers = {path.name: load_solver_artifacts(path) for path in discover_solver_dirs(root)}
    return {
        "run_root": root,
        "run_config": _load_optional_json(root / "run_config.json"),
        "final_summary": _load_optional_json(root / "final_summary.json"),
        "final_rows": _load_optional_json(root / "final_comparison_rows.json"),
        "solvers": solvers,
    }


def _objective_sort_value(row: Mapping[str, Any]) -> Tuple[int, float, float, str]:
    objective = _finite_float(row.get("objective_value"))
    metric = _finite_float(row.get("metric_val"))
    if objective is not None:
        return (0, -objective, metric if metric is not None else float("inf"), str(row.get("candidate_id", "")))
    if metric is not None:
        return (1, metric, 0.0, str(row.get("candidate_id", "")))
    return (2, float("inf"), 0.0, str(row.get("candidate_id", "")))


def _candidate_id(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate_id", row.get("observation_id", "")))


def _merge_unique_candidates(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = _candidate_id(row)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def select_top_candidates(solver_artifacts: Mapping[str, Any], *, top_k: int = 5) -> List[Dict[str, Any]]:
    confirmation_rows = solver_artifacts.get("confirmation", {}).get("rows", [])
    source_rows = confirmation_rows if confirmation_rows else solver_artifacts.get("observations", {}).get("observations", [])
    sorted_rows = sorted((dict(row) for row in source_rows), key=_objective_sort_value)
    best = solver_artifacts.get("best", {}).get("best_observation")
    candidates: List[Mapping[str, Any]] = []
    if isinstance(best, Mapping):
        candidates.append(best)
    candidates.extend(sorted_rows)
    merged = _merge_unique_candidates(candidates)
    return merged[: max(1, int(top_k))]


def _summary_rows(run_artifacts: Mapping[str, Any], solver_key: str) -> List[Dict[str, Any]]:
    rows = run_artifacts.get("final_summary", {}).get("summaries", [])
    if rows:
        return [dict(row) for row in rows if str(row.get("solver_key")) == str(solver_key)]
    final_rows = run_artifacts.get("final_rows", {}).get("rows", [])
    by_schedule: Dict[str, Dict[str, Any]] = {}
    for row in final_rows:
        if str(row.get("solver_key")) != str(solver_key):
            continue
        schedule_key = str(row.get("schedule_key"))
        by_schedule.setdefault(schedule_key, dict(row))
    return list(by_schedule.values())


def _schedule_from_final_rows(
    run_artifacts: Mapping[str, Any],
    *,
    solver_key: str,
    schedule_key: str,
) -> Optional[List[float]]:
    for row in _summary_rows(run_artifacts, solver_key):
        if str(row.get("schedule_key")) == str(schedule_key) and row.get("schedule_grid") is not None:
            return _as_float_list(row["schedule_grid"], name=f"{solver_key}:{schedule_key}:schedule_grid")
    return None


def schedule_series_for_solver(
    run_artifacts: Mapping[str, Any],
    solver_key: str,
    top_candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    solver_artifacts = run_artifacts["solvers"][solver_key]
    reference_grid = _as_float_list(solver_artifacts["reference"]["schedule_grid"], name="reference_schedule_grid")
    series: List[Dict[str, Any]] = []
    for schedule_key in BASELINE_STYLE_ORDER:
        grid = reference_grid if schedule_key == "ser_ptg_reference" else _schedule_from_final_rows(
            run_artifacts,
            solver_key=solver_key,
            schedule_key=schedule_key,
        )
        if grid is None:
            if schedule_key == "uniform":
                grid = np.linspace(0.0, 1.0, len(reference_grid), dtype=np.float64).tolist()
            else:
                continue
        series.append(
            {
                "label": schedule_key,
                "schedule_key": schedule_key,
                "series_type": "baseline",
                "candidate_id": "",
                "schedule_grid": [float(x) for x in grid],
            }
        )
    for rank, row in enumerate(top_candidates, start=1):
        grid = row.get("schedule_grid")
        if grid is None:
            continue
        candidate_id = _candidate_id(row)
        series.append(
            {
                "label": f"top {rank}: {candidate_id}",
                "schedule_key": "bo_candidate",
                "series_type": "top_candidate",
                "candidate_id": candidate_id,
                "metric_val": row.get("metric_val"),
                "objective_value": row.get("objective_value"),
                "schedule_grid": _as_float_list(grid, name=f"{solver_key}:{candidate_id}:schedule_grid"),
            }
        )
    return series


def top_candidates_table(solver_key: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        out.append(
            {
                "solver_key": solver_key,
                "rank": rank,
                "candidate_id": _candidate_id(row),
                "source": row.get("source"),
                "metric_val": row.get("metric_val"),
                "relative_crps_ratio": row.get("relative_crps_ratio"),
                "relative_mase_ratio": row.get("relative_mase_ratio"),
                "objective_value": row.get("objective_value"),
                "kl_to_reference": row.get("kl_to_reference"),
                "theta": row.get("theta"),
                "schedule_grid": row.get("schedule_grid"),
                "latency_ms_per_sample": row.get("latency_ms_per_sample"),
            }
        )
    return out


def schedule_nodes_table(solver_key: str, series: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in series:
        grid = _as_float_list(item["schedule_grid"], name="schedule_grid")
        for idx, value in enumerate(grid):
            rows.append(
                {
                    "solver_key": solver_key,
                    "series_label": item["label"],
                    "series_type": item["series_type"],
                    "schedule_key": item["schedule_key"],
                    "candidate_id": item.get("candidate_id", ""),
                    "node_index": idx,
                    "time": float(value),
                }
            )
    return rows


def interval_widths_table(solver_key: str, series: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in series:
        widths = _schedule_intervals(item["schedule_grid"])
        for idx, value in enumerate(widths):
            rows.append(
                {
                    "solver_key": solver_key,
                    "series_label": item["label"],
                    "series_type": item["series_type"],
                    "schedule_key": item["schedule_key"],
                    "candidate_id": item.get("candidate_id", ""),
                    "interval_index": idx,
                    "width": float(value),
                }
            )
    return rows


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is absent.
        raise RuntimeError(
            "visualize-run requires matplotlib. Install future-work optional dependencies with "
            "`python -m pip install -r future_work/bayesian_rl_schedule_optimization/requirements.txt`."
        ) from exc
    return plt


def plot_bo_trajectory(solver_artifacts: Mapping[str, Any], out_path: str | Path) -> None:
    plt = _require_matplotlib()
    rows = list(solver_artifacts["observations"]["observations"])
    x = np.arange(len(rows), dtype=np.int64)
    metrics = np.asarray([np.nan if _finite_float(row.get("metric_val")) is None else float(row["metric_val"]) for row in rows])
    objectives = np.asarray(
        [np.nan if _finite_float(row.get("objective_value")) is None else float(row["objective_value"]) for row in rows]
    )
    kls = np.asarray([np.nan if _finite_float(row.get("kl_to_reference")) is None else float(row["kl_to_reference"]) for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True, constrained_layout=True)
    phase_colors = {
        "ser_ptg_reference_center": "#111111",
        "initial_sobol_kl_perturbation": "#4c78a8",
        "qLogNoisyExpectedImprovement": "#f58518",
    }
    for source in sorted({str(row.get("source", "unknown")) for row in rows}):
        mask = np.asarray([str(row.get("source", "unknown")) == source for row in rows], dtype=bool)
        label = PHASE_LABELS.get(source, source)
        color = phase_colors.get(source, "#777777")
        axes[0].scatter(x[mask], metrics[mask], s=24, color=color, alpha=0.8, label=label)
        axes[1].scatter(x[mask], objectives[mask], s=24, color=color, alpha=0.8, label=label)
        axes[2].scatter(x[mask], kls[mask], s=24, color=color, alpha=0.8, label=label)

    if np.isfinite(metrics).any():
        valid_idx = np.where(np.isfinite(metrics))[0]
        axes[0].plot(valid_idx, np.minimum.accumulate(metrics[valid_idx]), color="#222222", linewidth=1.8, label="running best")
    if np.isfinite(objectives).any():
        valid_idx = np.where(np.isfinite(objectives))[0]
        axes[1].plot(valid_idx, np.maximum.accumulate(objectives[valid_idx]), color="#222222", linewidth=1.8, label="running best")

    best = solver_artifacts.get("best", {}).get("best_observation", {})
    best_id = str(best.get("candidate_id", ""))
    if best_id:
        for idx, row in enumerate(rows):
            if str(row.get("candidate_id")) == best_id:
                for axis in axes:
                    axis.axvline(idx, color="#2ca02c", linestyle="--", linewidth=1.4, alpha=0.8)
                break

    axes[0].set_ylabel("metric_val lower better")
    axes[1].set_ylabel("objective higher better")
    axes[2].set_ylabel("KL to reference")
    axes[2].set_xlabel("candidate index")
    axes[0].set_title(f"{solver_artifacts['solver_key']} BO trajectory")
    axes[0].legend(loc="best", fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_schedule_nodes(solver_key: str, series: Sequence[Mapping[str, Any]], out_path: str | Path) -> None:
    plt = _require_matplotlib()
    fig, axis = plt.subplots(figsize=(10.0, 5.6), constrained_layout=True)
    for idx, item in enumerate(series):
        grid = np.asarray(item["schedule_grid"], dtype=np.float64)
        x = np.arange(len(grid), dtype=np.int64)
        if item["series_type"] == "top_candidate":
            color = TOP_COLOR_CYCLE[idx % len(TOP_COLOR_CYCLE)]
            axis.plot(x, grid, marker="o", linewidth=1.9, color=color, label=item["label"])
        else:
            schedule_key = str(item["schedule_key"])
            color = "#111111" if schedule_key == "ser_ptg_reference" else "#777777"
            linestyle = "--" if schedule_key in {"uniform", "ser_ptg_reference"} else ":"
            linewidth = 2.3 if schedule_key == "ser_ptg_reference" else 1.5
            axis.plot(x, grid, marker=".", linewidth=linewidth, linestyle=linestyle, color=color, label=schedule_key)
    axis.set_title(f"{solver_key} inference schedule nodes")
    axis.set_xlabel("runtime node index")
    axis.set_ylabel("inference time t_k")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_interval_widths(solver_key: str, series: Sequence[Mapping[str, Any]], out_path: str | Path) -> None:
    plt = _require_matplotlib()
    labels = [str(item["label"]) for item in series]
    widths = [_schedule_intervals(item["schedule_grid"]) for item in series]
    max_len = max(len(row) for row in widths)
    data = np.full((len(widths), max_len), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(widths):
        data[row_idx, : len(row)] = row
    fig, axis = plt.subplots(figsize=(10.0, max(4.5, 0.36 * len(labels))), constrained_layout=True)
    image = axis.imshow(data, aspect="auto", cmap="viridis")
    axis.set_title(f"{solver_key} interval widths")
    axis.set_xlabel("interval index")
    axis.set_ylabel("schedule")
    axis.set_xticks(np.arange(max_len))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    fig.colorbar(image, ax=axis, label="interval width")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _final_summary_records(run_artifacts: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = run_artifacts.get("final_summary", {}).get("summaries", [])
    if rows:
        return [dict(row) for row in rows]
    final_rows = run_artifacts.get("final_rows", {}).get("rows", [])
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in final_rows:
        grouped.setdefault((str(row.get("solver_key")), str(row.get("schedule_key"))), []).append(row)
    out: List[Dict[str, Any]] = []
    for (solver_key, schedule_key), items in sorted(grouped.items()):
        values = [_finite_float(row.get("avg_relative_ratio")) for row in items]
        values = [value for value in values if value is not None]
        if values:
            out.append(
                {
                    "solver_key": solver_key,
                    "schedule_key": schedule_key,
                    "avg_relative_ratio_mean": float(np.mean(values)),
                    "avg_relative_ratio_std": float(np.std(values)),
                }
            )
    return out


def plot_final_comparison_summary(run_artifacts: Mapping[str, Any], out_path: str | Path) -> None:
    records = _final_summary_records(run_artifacts)
    if not records:
        return
    plt = _require_matplotlib()
    solvers = sorted({str(row["solver_key"]) for row in records})
    configured = run_artifacts.get("run_config", {}).get("comparison_schedules", [])
    schedule_order = [str(key) for key in configured if any(str(row.get("schedule_key")) == str(key) for row in records)]
    schedule_order.extend(
        key
        for key in sorted({str(row.get("schedule_key")) for row in records})
        if key not in schedule_order
    )
    lookup = {(str(row["solver_key"]), str(row["schedule_key"])): row for row in records}
    x = np.arange(len(solvers), dtype=np.float64)
    width = min(0.12, 0.78 / max(1, len(schedule_order)))
    fig, axis = plt.subplots(figsize=(max(8.0, 1.4 * len(solvers) + 1.0 * len(schedule_order)), 5.4), constrained_layout=True)
    for idx, schedule_key in enumerate(schedule_order):
        offset = (idx - (len(schedule_order) - 1) / 2.0) * width
        values = [
            _finite_float(lookup.get((solver, schedule_key), {}).get("avg_relative_ratio_mean")) or np.nan
            for solver in solvers
        ]
        axis.bar(x + offset, values, width=width, label=schedule_key)
    axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    axis.set_xticks(x)
    axis.set_xticklabels(solvers)
    axis.set_ylabel("locked-test avg relative ratio lower better")
    axis.set_title("Final locked-test schedule comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def visualize_run(run_root: str | Path, *, top_k: int = 5) -> Dict[str, Any]:
    run_artifacts = load_run_artifacts(run_root)
    root = Path(run_root)
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    outputs: Dict[str, Any] = {"figures": [], "tables": []}

    for solver_key, solver_artifacts in sorted(run_artifacts["solvers"].items()):
        top_rows = select_top_candidates(solver_artifacts, top_k=top_k)
        series = schedule_series_for_solver(run_artifacts, solver_key, top_rows)

        top_csv = tables_dir / f"top_candidates_{solver_key}.csv"
        nodes_csv = tables_dir / f"schedule_nodes_{solver_key}.csv"
        widths_csv = tables_dir / f"interval_widths_{solver_key}.csv"
        _write_csv(
            top_candidates_table(solver_key, top_rows),
            top_csv,
            fieldnames=[
                "solver_key",
                "rank",
                "candidate_id",
                "source",
                "metric_val",
                "relative_crps_ratio",
                "relative_mase_ratio",
                "objective_value",
                "kl_to_reference",
                "theta",
                "schedule_grid",
                "latency_ms_per_sample",
            ],
        )
        _write_csv(
            schedule_nodes_table(solver_key, series),
            nodes_csv,
            fieldnames=[
                "solver_key",
                "series_label",
                "series_type",
                "schedule_key",
                "candidate_id",
                "node_index",
                "time",
            ],
        )
        _write_csv(
            interval_widths_table(solver_key, series),
            widths_csv,
            fieldnames=[
                "solver_key",
                "series_label",
                "series_type",
                "schedule_key",
                "candidate_id",
                "interval_index",
                "width",
            ],
        )
        outputs["tables"].extend([str(top_csv), str(nodes_csv), str(widths_csv)])

        trajectory_png = figures_dir / f"bo_trajectory_{solver_key}.png"
        nodes_png = figures_dir / f"schedule_nodes_{solver_key}.png"
        widths_png = figures_dir / f"interval_widths_{solver_key}.png"
        plot_bo_trajectory(solver_artifacts, trajectory_png)
        plot_schedule_nodes(solver_key, series, nodes_png)
        plot_interval_widths(solver_key, series, widths_png)
        outputs["figures"].extend([str(trajectory_png), str(nodes_png), str(widths_png)])

    summary_png = figures_dir / "final_comparison_summary.png"
    plot_final_comparison_summary(run_artifacts, summary_png)
    if summary_png.exists():
        outputs["figures"].append(str(summary_png))
    return outputs


def add_visualize_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("visualize-run", help="Generate read-only BO run trajectory and schedule figures.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)


def run_visualize_command(args: argparse.Namespace) -> Dict[str, Any]:
    return visualize_run(args.run_root, top_k=int(args.top_k))
