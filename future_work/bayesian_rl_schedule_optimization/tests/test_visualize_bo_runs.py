from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from visualize_bo_runs import (  # noqa: E402
    discover_solver_dirs,
    load_run_artifacts,
    schedule_nodes_table,
    schedule_series_for_solver,
    select_top_candidates,
    visualize_run,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def schedule_grid(nodes: int) -> list[float]:
    return [float(idx) / float(nodes - 1) for idx in range(nodes)]


def observation(candidate_id: str, *, source: str, metric: float, objective: float, nodes: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "source": source,
        "metric_val": metric,
        "objective_value": objective,
        "relative_crps_ratio": metric,
        "relative_mase_ratio": metric,
        "kl_to_reference": max(0.0, -objective - metric),
        "theta": [0.0, 0.1, -0.1, 0.0, 0.2],
        "schedule_grid": schedule_grid(nodes),
        "latency_ms_per_sample": 12.5,
    }


def build_synthetic_run(root: Path) -> None:
    write_json(
        root / "run_config.json",
        {
            "comparison_schedules": ["uniform", "gits", "ser_ptg_reference", "bo_best"],
            "dataset": "synthetic",
            "target_nfe": 10,
        },
    )
    summaries = []
    for solver, nodes in (("euler", 11), ("heun", 6)):
        solver_dir = root / solver
        reference = schedule_grid(nodes)
        write_json(
            solver_dir / "reference_schedule.json",
            {
                "artifact": "bo_reference_schedule_v1",
                "solver_key": solver,
                "schedule_grid": reference,
                "q_ref": [1.0 / float(nodes - 1)] * (nodes - 1),
            },
        )
        observations = [
            observation("reference_center", source="ser_ptg_reference_center", metric=1.1, objective=-1.1, nodes=nodes),
            observation("init_000", source="initial_sobol_kl_perturbation", metric=1.0, objective=-1.01, nodes=nodes),
            observation("bo_002", source="qLogNoisyExpectedImprovement", metric=0.9, objective=-0.95, nodes=nodes),
            observation("bo_003", source="qLogNoisyExpectedImprovement", metric=0.85, objective=-0.97, nodes=nodes),
        ]
        write_json(solver_dir / "observations.json", {"observations": observations})
        write_json(solver_dir / "best_schedule.json", {"best_observation": observations[2]})
        if solver == "euler":
            write_json(solver_dir / "confirmation_rows.json", {"rows": [observations[3], observations[2]]})
        for schedule_key, value in (
            ("uniform", 1.0),
            ("gits", 0.98),
            ("ser_ptg_reference", 1.05),
            ("bo_best", 0.9),
        ):
            summaries.append(
                {
                    "solver_key": solver,
                    "schedule_key": schedule_key,
                    "avg_relative_ratio_mean": value,
                    "schedule_grid": reference,
                }
            )
    write_json(root / "final_summary.json", {"summaries": summaries})


class VisualizeBoRunsTests(unittest.TestCase):
    def test_artifact_loading_and_top_candidate_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_synthetic_run(root)

            self.assertEqual([path.name for path in discover_solver_dirs(root)], ["euler", "heun"])
            artifacts = load_run_artifacts(root)

            euler_top = select_top_candidates(artifacts["solvers"]["euler"], top_k=2)
            self.assertEqual([row["candidate_id"] for row in euler_top], ["bo_002", "bo_003"])

            heun_top = select_top_candidates(artifacts["solvers"]["heun"], top_k=2)
            self.assertEqual([row["candidate_id"] for row in heun_top], ["bo_002", "bo_003"])

    def test_schedule_series_handles_ten_and_five_interval_solvers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_synthetic_run(root)
            artifacts = load_run_artifacts(root)

            euler_series = schedule_series_for_solver(
                artifacts,
                "euler",
                select_top_candidates(artifacts["solvers"]["euler"], top_k=2),
            )
            heun_series = schedule_series_for_solver(
                artifacts,
                "heun",
                select_top_candidates(artifacts["solvers"]["heun"], top_k=2),
            )

            euler_nodes = schedule_nodes_table("euler", euler_series)
            heun_nodes = schedule_nodes_table("heun", heun_series)

            self.assertTrue(any(row["node_index"] == 10 for row in euler_nodes))
            self.assertFalse(any(row["node_index"] == 10 for row in heun_nodes))
            self.assertTrue(any(row["node_index"] == 5 for row in heun_nodes))

    def test_missing_required_observations_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "euler").mkdir()

            with self.assertRaisesRegex(ValueError, "No solver observation artifacts"):
                discover_solver_dirs(root)

    def test_visualize_run_writes_expected_png_and_csv_outputs(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_synthetic_run(root)

            outputs = visualize_run(root, top_k=2)

            expected = [
                root / "figures" / "bo_trajectory_euler.png",
                root / "figures" / "schedule_nodes_heun.png",
                root / "figures" / "interval_widths_euler.png",
                root / "figures" / "final_comparison_summary.png",
                root / "tables" / "top_candidates_euler.csv",
                root / "tables" / "schedule_nodes_heun.csv",
            ]
            for path in expected:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)
            self.assertTrue(outputs["figures"])
            self.assertTrue(outputs["tables"])


if __name__ == "__main__":
    unittest.main()
