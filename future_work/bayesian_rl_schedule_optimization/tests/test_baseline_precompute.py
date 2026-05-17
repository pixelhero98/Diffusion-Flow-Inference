from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from baseline_precompute import (  # noqa: E402
    DEFAULT_BASELINE_SCHEDULES,
    build_baseline_manifest,
    discover_source_cache_roots,
    summarize_global_final_rows,
)
from cli import build_argparser  # noqa: E402
from forecast_bo_runner import _matching_cached_final_row, indices_hash, schedule_hash  # noqa: E402


class BaselinePrecomputeTests(unittest.TestCase):
    def test_cli_parser_accepts_precompute_command(self) -> None:
        args = build_argparser().parse_args(
            [
                "precompute-forecast-baselines",
                "--datasets",
                "san_francisco_traffic,solar_energy_10m",
                "--target-nfes",
                "10,12,16",
                "--out-root",
                "outputs/future_work/forecast_schedule_baseline_cache/demo",
            ]
        )

        self.assertEqual(args.command, "precompute-forecast-baselines")
        self.assertEqual(args.schedules, ",".join(DEFAULT_BASELINE_SCHEDULES))
        self.assertTrue(args.resume)

    def test_manifest_counts_expected_432_rows(self) -> None:
        datasets = ["san_francisco_traffic", "solar_energy_10m"]
        target_nfes = [10, 12, 16]
        solvers = ["euler", "heun", "midpoint_rk2", "dpmpp2m"]
        schedules = list(DEFAULT_BASELINE_SCHEDULES)
        seeds = [0, 1, 2]
        rows = [
            {
                "dataset": dataset,
                "target_nfe": target_nfe,
                "solver_key": solver,
                "schedule_key": schedule,
                "seed": seed,
            }
            for dataset in datasets
            for target_nfe in target_nfes
            for solver in solvers
            for schedule in schedules
            for seed in seeds
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for dataset in datasets:
                for target_nfe in target_nfes:
                    for solver in solvers:
                        ref = Path(tmpdir) / dataset / f"nfe_{target_nfe}" / solver / "reference_schedule.json"
                        ref.parent.mkdir(parents=True, exist_ok=True)
                        ref.write_text(
                            '{"schedule_grid":[0.0,1.0],"schedule_hash":"'
                            + schedule_hash([0.0, 1.0])
                            + '","reference_macro_factor":16.0,"calibration_indices":[0,1]}',
                            encoding="utf-8",
                        )
            manifest = build_baseline_manifest(
                rows,
                datasets=datasets,
                target_nfes=target_nfes,
                solvers=solvers,
                schedules=schedules,
                seeds=seeds,
                out_root=tmpdir,
            )

        self.assertEqual(manifest["expected_rows"], 432)
        self.assertEqual(manifest["present_rows"], 432)
        self.assertEqual(manifest["missing_rows"], 0)
        self.assertTrue(manifest["complete"])
        self.assertEqual(len(manifest["references"]), 24)

    def test_manifest_reports_missing_rows(self) -> None:
        manifest = build_baseline_manifest(
            [],
            datasets=["solar_energy_10m"],
            target_nfes=[10],
            solvers=["euler"],
            schedules=["uniform", "ser_ptg_reference"],
            seeds=[0, 1],
            out_root=Path("missing"),
        )

        self.assertEqual(manifest["expected_rows"], 4)
        self.assertEqual(manifest["present_rows"], 0)
        self.assertEqual(manifest["missing_rows"], 4)
        self.assertFalse(manifest["complete"])

    def test_summary_keeps_raw_and_relative_metrics_by_dataset_nfe(self) -> None:
        rows = [
            {
                "dataset": "solar_energy_10m",
                "target_nfe": 10,
                "solver_key": "euler",
                "schedule_key": "gits",
                "seed": 0,
                "schedule_grid": [0.0, 1.0],
                "crps": 2.0,
                "mase": 4.0,
                "relative_crps_ratio": 0.8,
                "relative_mase_ratio": 0.9,
                "avg_relative_ratio": 0.85,
                "kl_to_reference": None,
            },
            {
                "dataset": "solar_energy_10m",
                "target_nfe": 10,
                "solver_key": "euler",
                "schedule_key": "gits",
                "seed": 1,
                "schedule_grid": [0.0, 1.0],
                "crps": 4.0,
                "mase": 8.0,
                "relative_crps_ratio": 1.0,
                "relative_mase_ratio": 1.1,
                "avg_relative_ratio": 1.05,
                "kl_to_reference": None,
            },
        ]

        summary = summarize_global_final_rows(rows)[0]

        self.assertEqual(summary["dataset"], "solar_energy_10m")
        self.assertEqual(summary["target_nfe"], 10)
        self.assertAlmostEqual(summary["crps_mean"], 3.0)
        self.assertAlmostEqual(summary["mase_mean"], 6.0)
        self.assertAlmostEqual(summary["relative_crps_ratio_mean"], 0.9)
        self.assertAlmostEqual(summary["relative_mase_ratio_mean"], 1.0)
        self.assertAlmostEqual(summary["avg_relative_ratio_mean"], 0.95)

    def test_discover_source_cache_roots_uses_only_explicit_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bo = root / "outputs" / "future_work" / "bo_schedule_search" / "bo_run"
            ppo = root / "outputs" / "future_work" / "joint_progression_ppo" / "ppo_run"
            removed_ppo = root / "outputs" / "future_work" / "removed_policy_run"
            baseline = root / "outputs" / "future_work" / "forecast_schedule_baseline_cache" / "baseline_run"
            for path in (bo, ppo, removed_ppo, baseline):
                path.mkdir(parents=True)

            found = discover_source_cache_roots(root, ["outputs/future_work/forecast_schedule_baseline_cache/baseline_run"])

        self.assertEqual([path.name for path in found], ["baseline_run"])

    def test_precomputed_late_power_row_is_reusable_by_bo_matching(self) -> None:
        import argparse

        schedule = [0.0, 0.25, 1.0]
        test_hash = indices_hash([0, 1, 2])
        row = {
            "dataset": "san_francisco_traffic",
            "solver_key": "euler",
            "schedule_key": "late_power_3",
            "target_nfe": 10,
            "runtime_nfe": 2,
            "seed": 0,
            "num_eval_samples": 5,
            "eval_examples": 3,
            "checkpoint_id": "ckpt",
            "schedule_grid": schedule,
            "schedule_hash": schedule_hash(schedule),
            "test_indices_hash": test_hash,
        }

        match = _matching_cached_final_row(
            [row],
            args=argparse.Namespace(dataset="san_francisco_traffic", target_nfe=10, num_eval_samples=5),
            checkpoint={"checkpoint_id": "ckpt"},
            solver_key="euler",
            schedule_key="late_power_3",
            seed=0,
            runtime_nfe=2,
            schedule_grid=schedule,
            expected_eval_examples=3,
            expected_test_indices_hash=test_hash,
        )

        self.assertIsNotNone(match)
        self.assertTrue(match["reused_from_cache"])


if __name__ == "__main__":
    unittest.main()
