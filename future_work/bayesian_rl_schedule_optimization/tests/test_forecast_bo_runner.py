from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from forecast_bo_runner import (  # noqa: E402
    COMPARISON_SCHEDULE_KEYS,
    _normalize_candidate_observation,
    candidate_budget_breakdown,
    deterministic_validation_partition,
    make_observation_payload,
    indices_hash,
    parse_comparison_schedules,
    schedule_hash,
    selected_indices_from_pool,
    _matching_cached_final_row,
    pending_candidate_records,
    select_best_observation,
    select_top_observations,
    summarize_final_rows,
    validate_final_comparison_coverage,
)


class ForecastBoRunnerTests(unittest.TestCase):
    def test_budget_accounting_reaches_requested_total(self) -> None:
        breakdown = candidate_budget_breakdown(100, n_initial=16)

        self.assertEqual(breakdown["reference"], 1)
        self.assertEqual(breakdown["initial"], 16)
        self.assertEqual(breakdown["bo"], 83)
        self.assertEqual(breakdown["total"], 100)

    def test_small_budget_uses_reference_then_initial_candidates(self) -> None:
        breakdown = candidate_budget_breakdown(4, n_initial=12)

        self.assertEqual(breakdown, {"reference": 1, "initial": 3, "bo": 0, "total": 4})

    def test_validation_partition_is_deterministic_and_disjoint(self) -> None:
        calibration, bo_validation = deterministic_validation_partition(100, calibration_fraction=0.7, seed=11)
        calibration_again, bo_validation_again = deterministic_validation_partition(100, calibration_fraction=0.7, seed=11)

        self.assertEqual(calibration, calibration_again)
        self.assertEqual(bo_validation, bo_validation_again)
        self.assertEqual(len(calibration), 70)
        self.assertEqual(len(bo_validation), 30)
        self.assertFalse(set(calibration).intersection(bo_validation))

    def test_selected_indices_from_pool_samples_without_leaving_pool(self) -> None:
        pool = [10, 20, 30, 40, 50]

        selected = selected_indices_from_pool(pool, 3, seed=5)

        self.assertEqual(len(selected), 3)
        self.assertTrue(set(selected).issubset(pool))

    def test_observation_payload_stores_session_uniform_baseline(self) -> None:
        payload = make_observation_payload(
            dataset="san_francisco_traffic",
            solver_key="euler",
            target_nfe=10,
            runtime_nfe=10,
            uniform_metrics={"crps": 4.0, "mase": 2.0, "mse": 9.0},
        )

        self.assertEqual(payload["uniform_baseline"], {"crps": 4.0, "mase": 2.0})
        self.assertEqual(payload["observations"], [])

    def test_pending_candidate_records_skip_resumed_rows(self) -> None:
        observations = {"observations": [{"candidate_id": "reference_center"}, {"candidate_id": "init_000"}]}
        candidates = [
            {"candidate_id": "reference_center"},
            {"candidate_id": "init_000"},
            {"candidate_id": "init_001"},
        ]

        pending = pending_candidate_records(candidates, observations)

        self.assertEqual([row["candidate_id"] for row in pending], ["init_001"])

    def test_select_best_observation_uses_validation_objective(self) -> None:
        payload = {
            "observations": [
                {"candidate_id": "a", "objective_value": -1.0, "metric_val": 1.0},
                {"candidate_id": "b", "objective_value": -0.8, "metric_val": 0.9},
            ]
        }

        self.assertEqual(select_best_observation(payload)["candidate_id"], "b")

    def test_select_top_observations_uses_validation_objective_order(self) -> None:
        payload = {
            "observations": [
                {"candidate_id": "a", "objective_value": -1.0, "metric_val": 1.0},
                {"candidate_id": "b", "objective_value": -0.8, "metric_val": 0.9},
                {"candidate_id": "c", "objective_value": -0.9, "metric_val": 0.8},
            ]
        }

        self.assertEqual([row["candidate_id"] for row in select_top_observations(payload, top_k=2)], ["b", "c"])

    def test_cached_baseline_row_requires_matching_metadata_and_schedule_hash(self) -> None:
        import argparse

        schedule = [0.0, 0.5, 1.0]
        args = argparse.Namespace(
            dataset="san_francisco_traffic",
            target_nfe=10,
            num_eval_samples=5,
        )
        checkpoint = {"checkpoint_id": "ckpt"}
        test_hash = indices_hash([0, 1, 2])
        row = {
            "dataset": "san_francisco_traffic",
            "solver_key": "euler",
            "schedule_key": "uniform",
            "target_nfe": 10,
            "runtime_nfe": 2,
            "seed": 0,
            "num_eval_samples": 5,
            "eval_examples": 862,
            "checkpoint_id": "ckpt",
            "schedule_grid": schedule,
            "test_indices_hash": test_hash,
        }

        match = _matching_cached_final_row(
            [row],
            args=args,
            checkpoint=checkpoint,
            solver_key="euler",
            schedule_key="uniform",
            seed=0,
            runtime_nfe=2,
            schedule_grid=schedule,
            expected_eval_examples=862,
            expected_test_indices_hash=test_hash,
        )

        self.assertIsNotNone(match)
        self.assertTrue(match["reused_from_cache"])
        self.assertEqual(match["schedule_hash"], schedule_hash(schedule))

        mismatch = _matching_cached_final_row(
            [row],
            args=args,
            checkpoint=checkpoint,
            solver_key="euler",
            schedule_key="uniform",
            seed=0,
            runtime_nfe=2,
            schedule_grid=[0.0, 0.25, 1.0],
            expected_eval_examples=862,
            expected_test_indices_hash=test_hash,
        )
        self.assertIsNone(mismatch)

    def test_transferred_schedule_cache_reuse_uses_metadata_and_hash(self) -> None:
        import argparse

        schedule = [0.0, 0.6, 1.0]
        args = argparse.Namespace(dataset="solar_energy_10m", target_nfe=10, num_eval_samples=5)
        checkpoint = {"checkpoint_id": "solar_energy_10m_otflow_forecast_20k_seed0"}
        test_hash = indices_hash([4, 5, 6])
        rows = []
        for schedule_key in ("uniform", "ays", "gits", "ots"):
            rows.append(
                {
                    "dataset": "solar_energy_10m",
                    "solver_key": "heun",
                    "schedule_key": schedule_key,
                    "target_nfe": 10,
                    "runtime_nfe": 5,
                    "seed": 2,
                    "num_eval_samples": 5,
                    "eval_examples": 137,
                    "checkpoint_id": "solar_energy_10m_otflow_forecast_20k_seed0",
                    "schedule_grid": schedule,
                    "test_indices_hash": test_hash,
                }
            )

        for schedule_key in ("uniform", "ays", "gits", "ots"):
            match = _matching_cached_final_row(
                rows,
                args=args,
                checkpoint=checkpoint,
                solver_key="heun",
                schedule_key=schedule_key,
                seed=2,
                runtime_nfe=5,
                schedule_grid=schedule,
                expected_eval_examples=137,
                expected_test_indices_hash=test_hash,
            )
            self.assertIsNotNone(match)
            self.assertTrue(match["reused_from_cache"])

        self.assertIsNone(
            _matching_cached_final_row(
                rows,
                args=args,
                checkpoint=checkpoint,
                solver_key="heun",
                schedule_key="bo_best",
                seed=2,
                runtime_nfe=5,
                schedule_grid=schedule,
                expected_eval_examples=137,
                expected_test_indices_hash=test_hash,
            )
        )

    def test_cached_baseline_row_rejects_missing_checkpoint_or_test_hash(self) -> None:
        import argparse

        schedule = [0.0, 0.5, 1.0]
        args = argparse.Namespace(dataset="demo", target_nfe=10, num_eval_samples=5)
        checkpoint = {"checkpoint_id": "ckpt"}
        row = {
            "dataset": "demo",
            "solver_key": "euler",
            "schedule_key": "uniform",
            "target_nfe": 10,
            "runtime_nfe": 2,
            "seed": 0,
            "num_eval_samples": 5,
            "eval_examples": 3,
            "schedule_grid": schedule,
        }

        self.assertIsNone(
            _matching_cached_final_row(
                [row],
                args=args,
                checkpoint=checkpoint,
                solver_key="euler",
                schedule_key="uniform",
                seed=0,
                runtime_nfe=2,
                schedule_grid=schedule,
                expected_eval_examples=3,
                expected_test_indices_hash=indices_hash([0, 1, 2]),
            )
        )

    def test_candidate_reevaluation_recomputes_split_specific_baseline(self) -> None:
        reference = {"q_ref": [0.5, 0.5]}
        payload = {"uniform_baseline": {"crps": 5.0, "mase": 2.0}}
        candidate = {
            "candidate_id": "bo_001",
            "source": "qLogNoisyExpectedImprovement",
            "theta": [0.0, 0.0, 0.0, 0.0, 0.0],
            "uniform_crps": 4.0,
            "uniform_mase": 1.0,
            "metric_val": 0.9,
            "objective_value": -0.9,
        }

        row = _normalize_candidate_observation(
            reference,
            payload,
            candidate,
            {"crps": 4.0, "mase": 1.6, "eval_examples": 3, "num_eval_samples": 5},
            lambda_kl=0.05,
        )

        self.assertEqual(row["uniform_crps"], 5.0)
        self.assertEqual(row["uniform_mase"], 2.0)
        self.assertAlmostEqual(row["metric_val"], 0.8)
        self.assertNotEqual(row["objective_value"], -0.9)

    def test_final_comparison_requires_four_schedules_for_each_solver(self) -> None:
        rows = []
        for solver in ("euler", "dpmpp2m"):
            for schedule in COMPARISON_SCHEDULE_KEYS:
                rows.append({"solver_key": solver, "schedule_key": schedule, "seed": 0})

        validate_final_comparison_coverage(rows, ["euler", "dpmpp2m"])

        with self.assertRaisesRegex(ValueError, "Missing final comparison rows"):
            validate_final_comparison_coverage(rows[:-1], ["euler", "dpmpp2m"])

    def test_solar_transfer_comparison_requires_six_schedules_for_four_solvers(self) -> None:
        schedules = parse_comparison_schedules("uniform,ays,gits,ots,ser_ptg_reference,bo_best")
        rows = []
        for solver in ("euler", "heun", "midpoint_rk2", "dpmpp2m"):
            for schedule in schedules:
                rows.append({"solver_key": solver, "schedule_key": schedule, "seed": 0})

        validate_final_comparison_coverage(rows, ["euler", "heun", "midpoint_rk2", "dpmpp2m"], schedules)

        with self.assertRaisesRegex(ValueError, "Missing final comparison rows"):
            validate_final_comparison_coverage(rows[:-1], ["euler", "heun", "midpoint_rk2", "dpmpp2m"], schedules)

    def test_comparison_schedules_require_uniform_and_known_keys(self) -> None:
        self.assertEqual(
            parse_comparison_schedules("uniform,ays,gits,ots,ser_ptg_reference,bo_best"),
            ["uniform", "ays", "gits", "ots", "ser_ptg_reference", "bo_best"],
        )
        with self.assertRaisesRegex(ValueError, "must include uniform"):
            parse_comparison_schedules("ays,gits")
        with self.assertRaisesRegex(ValueError, "Unknown comparison schedule"):
            parse_comparison_schedules("uniform,missing")

    def test_summarize_final_rows_averages_metrics(self) -> None:
        rows = [
            {
                "solver_key": "euler",
                "schedule_key": "bo_best",
                "seed": 0,
                "schedule_grid": [0.0, 0.5, 1.0],
                "crps": 2.0,
                "mase": 1.0,
                "relative_crps_ratio": 0.5,
                "relative_mase_ratio": 0.5,
                "avg_relative_ratio": 0.5,
                "kl_to_reference": 0.1,
            },
            {
                "solver_key": "euler",
                "schedule_key": "bo_best",
                "seed": 1,
                "schedule_grid": [0.0, 0.5, 1.0],
                "crps": 4.0,
                "mase": 3.0,
                "relative_crps_ratio": 1.0,
                "relative_mase_ratio": 1.5,
                "avg_relative_ratio": 1.25,
                "kl_to_reference": 0.1,
            },
        ]

        summary = summarize_final_rows(rows)[0]

        self.assertEqual(summary["n_seeds"], 2)
        self.assertAlmostEqual(summary["crps_mean"], 3.0)
        self.assertAlmostEqual(summary["avg_relative_ratio_mean"], 0.875)


if __name__ == "__main__":
    unittest.main()
