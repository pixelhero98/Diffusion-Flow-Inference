from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bo_trajectory_dataset import calibrate_from_bo_elites, find_matching_bo_solver_dir  # noqa: E402
from forecast_bo_runner import deterministic_validation_partition  # noqa: E402
from kl_ppo_bandit_schedule import (  # noqa: E402
    PPO_COMPARISON_SCHEDULES,
    _DiagonalGaussianPolicy,
    _complete_train_rows_for_resume,
    _final_row_matches_current,
    _global_final_key,
    _indices_hash,
    _matching_cached_row_for_ppo,
    _summarize_global_final_rows,
    diagonal_policy_kl,
    ppo_update,
    ppo_budget_total,
)
from reward import forecast_log_ratio_reward  # noqa: E402
from schedule_param import build_default_basis_for_reference, schedule_diagnostics, theta_to_checked_schedule  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def synthetic_reference() -> dict:
    return {
        "artifact": "bo_reference_schedule_v1",
        "dataset": "synthetic",
        "solver_key": "euler",
        "target_nfe": 10,
        "runtime_nfe": 10,
        "q_ref": [0.1] * 10,
        "schedule_grid": [idx / 10.0 for idx in range(11)],
    }


def synthetic_observations() -> dict:
    rows = []
    for idx in range(20):
        theta = [0.02 * idx, -0.01 * idx, 0.005 * idx, 0.0, -0.002 * idx]
        crps = 4.0 * (1.05 - 0.01 * idx)
        mase = 2.0 * (1.04 - 0.008 * idx)
        rows.append(
            {
                "candidate_id": f"bo_{idx:03d}",
                "theta": theta,
                "crps": crps,
                "mase": mase,
                "uniform_crps": 4.0,
                "uniform_mase": 2.0,
                "kl_to_reference": 0.01 + 0.003 * idx,
            }
        )
    return {"observations": rows}


class BanditKlPpoTests(unittest.TestCase):
    def test_theta_to_checked_schedule_records_diagnostics_and_rejects_large_kl(self) -> None:
        reference = synthetic_reference()
        basis = build_default_basis_for_reference(reference["q_ref"])

        record = theta_to_checked_schedule(reference["q_ref"], [0.0, 0.0, 0.0, 0.0, 0.0], basis=basis)

        self.assertEqual(record["schedule_grid"][0], 0.0)
        self.assertEqual(record["schedule_grid"][-1], 1.0)
        self.assertAlmostEqual(record["kl_to_reference"], 0.0)
        self.assertGreater(record["min_dt"], 0.0)
        self.assertIn("smoothness", record)

        with self.assertRaisesRegex(ValueError, "hard_kl_cap"):
            theta_to_checked_schedule(reference["q_ref"], [3.0, 0.0, 0.0, 0.0, 0.0], basis=basis, hard_kl_cap=0.001)

    def test_reward_uses_log_ratios_bad_penalty_and_kl_without_smoothness_penalty(self) -> None:
        out = forecast_log_ratio_reward(
            crps=3.6,
            mase=2.2,
            uniform_crps=4.0,
            uniform_mase=2.0,
            kl_to_reference=0.2,
            beta_ref=0.05,
            lambda_bad=3.0,
        )

        self.assertAlmostEqual(out["relative_crps_ratio"], 0.9)
        self.assertAlmostEqual(out["relative_mase_ratio"], 1.1)
        self.assertGreater(out["bad_penalty"], 0.0)
        self.assertAlmostEqual(out["kl_penalty"], 0.01)
        self.assertEqual(out["smoothness_penalty"], 0.0)
        self.assertEqual(out["guard_penalty"], 0.0)

    def test_reward_rejects_negative_penalty_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "beta_ref"):
            forecast_log_ratio_reward(
                crps=1.0,
                mase=1.0,
                uniform_crps=1.0,
                uniform_mase=1.0,
                kl_to_reference=0.0,
                beta_ref=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "lambda_bad"):
            forecast_log_ratio_reward(
                crps=1.0,
                mase=1.0,
                uniform_crps=1.0,
                uniform_mase=1.0,
                kl_to_reference=0.0,
                beta_ref=0.0,
                lambda_bad=-1.0,
            )

    def test_bo_elite_calibration_sets_beta_hard_cap_and_clipped_policy_std(self) -> None:
        calibration = calibrate_from_bo_elites(synthetic_reference(), synthetic_observations(), elite_fraction=0.2)

        self.assertEqual(calibration["n_observations"], 20)
        self.assertEqual(calibration["n_elites"], 4)
        self.assertGreaterEqual(calibration["beta_ref"], 0.01)
        self.assertLessEqual(calibration["beta_ref"], 0.20)
        self.assertGreaterEqual(calibration["hard_kl_cap"], 0.10)
        self.assertEqual(len(calibration["policy_mu"]), 5)
        self.assertEqual(len(calibration["policy_std"]), 5)
        self.assertTrue(all(0.03 <= value <= 0.20 for value in calibration["policy_std"]))

    def test_ppo_budget_and_policy_kl(self) -> None:
        self.assertEqual(ppo_budget_total(8, 20), 160)
        self.assertAlmostEqual(diagonal_policy_kl([0.0], [0.1], [0.0], [0.1]), 0.0)
        self.assertGreater(diagonal_policy_kl([0.0], [0.1], [0.1], [0.1]), 0.0)

    def test_ppo_update_restores_policy_when_max_kl_is_exceeded(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this local test environment.")

        policy = _DiagonalGaussianPolicy([0.0], [0.1], device="cpu")
        theta = torch.as_tensor([[0.1], [-0.1]], dtype=torch.float64)
        old_logprobs = [float(x) for x in policy.log_prob(theta).detach().cpu().tolist()]
        old_mu = policy.mu.detach().clone()

        out = ppo_update(
            policy,
            theta_batch=[[0.1], [-0.1]],
            old_logprobs=old_logprobs,
            rewards=[1.0, -1.0],
            ppo_epochs=3,
            clip_eps=0.1,
            lr_actor=1.0,
            lr_value=0.0,
            entropy_coef=0.0,
            target_policy_kl=10.0,
            max_policy_kl=0.0,
        )

        self.assertTrue(out["restored_by_max_policy_kl"])
        self.assertTrue(torch.allclose(policy.mu.detach(), old_mu))

    def test_validation_partition_remains_70_30_and_disjoint(self) -> None:
        calibration, selector = deterministic_validation_partition(100, calibration_fraction=0.7, seed=0)

        self.assertEqual(len(calibration), 70)
        self.assertEqual(len(selector), 30)
        self.assertFalse(set(calibration).intersection(selector))

    def test_matching_bo_solver_dir_checks_dataset_nfe_and_solver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "outputs" / "future_work" / "bo_schedule_search" / "run_a"
            solver_dir = run / "euler"
            write_json(run / "run_config.json", {"dataset": "synthetic", "target_nfe": 10})
            write_json(solver_dir / "reference_schedule.json", synthetic_reference())
            write_json(solver_dir / "observations.json", synthetic_observations())
            write_json(solver_dir / "best_schedule.json", {"best_observation": synthetic_observations()["observations"][0]})
            write_json(solver_dir / "validation_split.json", {"calibration_indices": [0], "bo_validation_indices": [1]})
            write_json(solver_dir / "confirmation_rows.json", {"rows": [synthetic_observations()["observations"][0]]})

            found = find_matching_bo_solver_dir(
                workspace_root=root,
                dataset="synthetic",
                target_nfe=10,
                solver_key="euler",
            )

            self.assertEqual(found, solver_dir)
            self.assertIsNone(
                find_matching_bo_solver_dir(
                    workspace_root=root,
                    dataset="synthetic",
                    target_nfe=12,
                    solver_key="euler",
                )
            )

    def test_matching_bo_solver_dir_rejects_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "outputs" / "future_work" / "bo_schedule_search" / "run_a"
            solver_dir = run / "euler"
            write_json(
                run / "run_config.json",
                {
                    "dataset": "synthetic",
                    "target_nfe": 10,
                    "checkpoint_id": "ckpt_a",
                    "otflow_train_steps": 20000,
                    "bo_budget": 100,
                    "calibration_fraction": 0.7,
                    "reference_macro_factor": 16.0,
                    "num_eval_samples": 5,
                    "basis_kind": "tilt_curvature_three_local_bumps",
                },
            )
            reference = synthetic_reference()
            reference["checkpoint_id"] = "ckpt_a"
            reference["reference_macro_factor"] = 16.0
            observations = synthetic_observations()
            observations["observations"] = observations["observations"] * 5
            observations["basis_kind"] = "tilt_curvature_three_local_bumps"
            write_json(solver_dir / "reference_schedule.json", reference)
            write_json(solver_dir / "observations.json", observations)
            write_json(solver_dir / "best_schedule.json", {"best_observation": observations["observations"][0]})
            write_json(solver_dir / "validation_split.json", {"calibration_indices": [0], "bo_validation_indices": [1]})
            write_json(solver_dir / "confirmation_rows.json", {"rows": [observations["observations"][0]]})

            self.assertIsNotNone(
                find_matching_bo_solver_dir(
                    workspace_root=root,
                    dataset="synthetic",
                    target_nfe=10,
                    solver_key="euler",
                    checkpoint_id="ckpt_a",
                    train_steps=20000,
                    bo_budget=100,
                    calibration_fraction=0.7,
                    reference_macro_factor=16.0,
                    num_eval_samples=5,
                    basis_kind="tilt_curvature_three_local_bumps",
                )
            )
            self.assertIsNone(
                find_matching_bo_solver_dir(
                    workspace_root=root,
                    dataset="synthetic",
                    target_nfe=10,
                    solver_key="euler",
                    checkpoint_id="ckpt_b",
                    train_steps=20000,
                    bo_budget=100,
                    calibration_fraction=0.7,
                    reference_macro_factor=16.0,
                    num_eval_samples=5,
                    basis_kind="tilt_curvature_three_local_bumps",
                )
            )

    def test_resume_training_rows_drop_partial_updates_and_selector_rows(self) -> None:
        rows = [
            {"split": "ppo_train_70pct", "update": 0, "sample_id": 0},
            {"split": "ppo_train_70pct", "update": 0, "sample_id": 1},
            {"split": "ppo_train_70pct", "update": 1, "sample_id": 0},
            {"split": "ppo_selector_30pct", "update": 0, "sample_id": 0},
        ]

        kept, completed = _complete_train_rows_for_resume(rows, batch_size=2)

        self.assertEqual(completed, 1)
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(row["update"] == 0 for row in kept))

    def test_ppo_final_cache_requires_checkpoint_and_test_hash(self) -> None:
        import argparse

        args = argparse.Namespace(num_eval_samples=5)
        schedule = [0.0, 0.5, 1.0]
        test_hash = _indices_hash([0, 1, 2])
        checkpoint = {
            "checkpoint_id": "ckpt",
            "checkpoint_path": "/tmp/ckpt.pt",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        row = {
            "dataset": "synthetic",
            "target_nfe": 10,
            "solver_key": "euler",
            "schedule_key": "uniform",
            "runtime_nfe": 2,
            "seed": 0,
            "num_eval_samples": 5,
            "eval_examples": 3,
            "checkpoint_id": "ckpt",
            "schedule_grid": schedule,
            "test_indices_hash": test_hash,
        }

        match = _matching_cached_row_for_ppo(
            [row],
            args=args,
            checkpoint=checkpoint,
            dataset="synthetic",
            target_nfe=10,
            solver_key="euler",
            schedule_key="uniform",
            seed=0,
            runtime_nfe=2,
            schedule_grid=schedule,
            expected_eval_examples=3,
            expected_test_indices_hash=test_hash,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["checkpoint_id"], "ckpt")
        self.assertIsNone(
            _matching_cached_row_for_ppo(
                [{key: value for key, value in row.items() if key != "test_indices_hash"}],
                args=args,
                checkpoint=checkpoint,
                dataset="synthetic",
                target_nfe=10,
                solver_key="euler",
                schedule_key="uniform",
                seed=0,
                runtime_nfe=2,
                schedule_grid=schedule,
                expected_eval_examples=3,
                expected_test_indices_hash=test_hash,
            )
        )

    def test_legacy_full_test_final_row_without_hash_is_accepted(self) -> None:
        checkpoint = {"checkpoint_id": "ckpt", "splits": {"test": [object(), object(), object()]}}
        row = {
            "checkpoint_id": "ckpt",
            "runtime_nfe": 2,
            "num_eval_samples": 5,
            "eval_examples": 3,
            "schedule_grid": [0.0, 0.5, 1.0],
        }

        self.assertTrue(
            _final_row_matches_current(
                row,
                checkpoint=checkpoint,
                runtime_nfe=2,
                schedule_grid=[0.0, 0.5, 1.0],
                num_eval_samples=5,
                expected_eval_examples=3,
                expected_test_indices_hash=_indices_hash([0, 1, 2]),
            )
        )
        self.assertFalse(
            _final_row_matches_current(
                row,
                checkpoint={"checkpoint_id": "ckpt", "splits": {"test": [object(), object(), object(), object()]}},
                runtime_nfe=2,
                schedule_grid=[0.0, 0.5, 1.0],
                num_eval_samples=5,
                expected_eval_examples=3,
                expected_test_indices_hash=_indices_hash([0, 1, 2]),
            )
        )

    def test_global_final_summary_keeps_dataset_and_nfe_distinct(self) -> None:
        rows = []
        for dataset in ("a", "b"):
            for target_nfe in (10, 12):
                rows.append(
                    {
                        "dataset": dataset,
                        "target_nfe": target_nfe,
                        "solver_key": "euler",
                        "schedule_key": "uniform",
                        "seed": 0,
                        "crps": 1.0,
                        "mase": 1.0,
                        "relative_crps_ratio": 1.0,
                        "relative_mase_ratio": 1.0,
                        "avg_relative_ratio": 1.0,
                        "kl_to_reference": None,
                        "schedule_grid": [0.0, 1.0],
                    }
                )

        summary = _summarize_global_final_rows(rows)

        self.assertEqual(len(summary), 4)
        self.assertEqual(len({_global_final_key(row) for row in rows}), 4)
        self.assertIn("ppo_bandit_best", PPO_COMPARISON_SCHEDULES)


if __name__ == "__main__":
    unittest.main()
