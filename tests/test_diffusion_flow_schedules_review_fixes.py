from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from diffusion_flow_inference.schedule_transfer.diffusion_flow_schedules import build_schedule_grid
import diffusion_flow_inference.schedule_transfer.otflow_schedule_diagnostics as schedule_diagnostics
from diffusion_flow_inference.schedule_transfer.otflow_schedule_diagnostics import _collect_rollout_diagnostics
from diffusion_flow_inference.schedule_transfer.otflow_paper_tables import augment_rows_with_relative_metrics


class DiffusionFlowScheduleReviewFixTests(unittest.TestCase):
    def test_transfer_schedule_grids_are_valid_for_runtime_steps(self) -> None:
        for schedule_key in ("ays", "gits", "ots"):
            for runtime_steps in (5, 6, 8, 10, 12, 16):
                with self.subTest(schedule_key=schedule_key, runtime_steps=runtime_steps):
                    grid = build_schedule_grid(schedule_key, runtime_steps)
                    self.assertIsNotNone(grid)
                    assert grid is not None
                    self.assertEqual(len(grid), runtime_steps + 1)
                    self.assertAlmostEqual(grid[0], 0.0)
                    self.assertAlmostEqual(grid[-1], 1.0)
                    self.assertTrue(all(b > a for a, b in zip(grid, grid[1:])))

    def test_schedule_grid_rejects_nonpositive_steps(self) -> None:
        for schedule_key in ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"):
            for runtime_steps in (0, -1):
                with self.subTest(schedule_key=schedule_key, runtime_steps=runtime_steps):
                    with self.assertRaisesRegex(ValueError, "n_steps must be positive"):
                        build_schedule_grid(schedule_key, runtime_steps)

    def test_flowts_power_sampling_grid_is_supported(self) -> None:
        grid = build_schedule_grid("flowts_power_sampling", 4)

        self.assertIsNotNone(grid)
        assert grid is not None
        self.assertEqual(len(grid), 5)
        self.assertAlmostEqual(grid[0], 0.0)
        self.assertAlmostEqual(grid[-1], 1.0)
        self.assertTrue(all(b > a for a, b in zip(grid, grid[1:])))

    def test_summary_relative_metrics_preserve_seed_paired_gains(self) -> None:
        rows = [
            {
                "benchmark_family": "forecast_extrapolation",
                "dataset": "demo",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "train_budget_label": "20k",
                "crps_mean": 10.0,
                "mase_mean": 5.0,
            },
            {
                "benchmark_family": "forecast_extrapolation",
                "dataset": "demo",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "gits",
                "train_budget_label": "20k",
                "crps_mean": 9.0,
                "mase_mean": 4.0,
                "relative_crps_gain_vs_uniform_mean": 0.11,
                "relative_mase_gain_vs_uniform_mean": 0.25,
            },
        ]
        augmented = augment_rows_with_relative_metrics(rows)
        gits = next(row for row in augmented if row["scheduler_key"] == "gits")
        self.assertAlmostEqual(gits["relative_crps_gain_vs_uniform"], 0.11)
        self.assertAlmostEqual(gits["relative_mase_gain_vs_uniform"], 0.25)

    def test_rollout_diagnostics_rejects_empty_chosen_t0s(self) -> None:
        cfg = SimpleNamespace(device=torch.device("cpu"))
        ds = SimpleNamespace(cond=None)

        with self.assertRaisesRegex(ValueError, "chosen_t0s must be a non-empty"):
            _collect_rollout_diagnostics(
                object(),
                ds,
                cfg,
                horizon=2,
                macro_steps=3,
                n_windows=1,
                seed=0,
                solver="euler",
                chosen_t0s=[],
            )

    def test_rollout_diagnostics_use_fixed_trace_fields_only(self) -> None:
        original_get_item = schedule_diagnostics._get_dataset_item_by_t
        original_parse_batch = schedule_diagnostics._parse_batch
        original_sample_eval_trace = schedule_diagnostics._sample_eval_trace
        original_crop_history = schedule_diagnostics.crop_history_window
        original_future_context = schedule_diagnostics._future_time_context_seq
        try:
            schedule_diagnostics._get_dataset_item_by_t = lambda ds, t0: object()
            schedule_diagnostics._parse_batch = lambda batch: (
                torch.zeros(2, 1),
                None,
                None,
                None,
                None,
            )
            schedule_diagnostics.crop_history_window = lambda hist, context_len: hist
            schedule_diagnostics._future_time_context_seq = lambda ds, t0, horizon: None

            def fake_sample_eval_trace(model, hist_t, *, cond_t, steps, solver):
                del model, hist_t, cond_t, solver
                trace = {
                    "field_evals_by_step": torch.ones(1, int(steps)),
                    "disagreement": torch.arange(int(steps), dtype=torch.float32)[None, :],
                    "mean_total_field_evals_per_rollout": float(steps),
                }
                return torch.zeros(1, 1, 1), trace, 1

            schedule_diagnostics._sample_eval_trace = fake_sample_eval_trace

            payload = _collect_rollout_diagnostics(
                object(),
                SimpleNamespace(cond=None),
                SimpleNamespace(device=torch.device("cpu")),
                horizon=1,
                macro_steps=3,
                n_windows=1,
                seed=0,
                solver="euler",
                chosen_t0s=np.asarray([0], dtype=np.int64),
            )
        finally:
            schedule_diagnostics._get_dataset_item_by_t = original_get_item
            schedule_diagnostics._parse_batch = original_parse_batch
            schedule_diagnostics._sample_eval_trace = original_sample_eval_trace
            schedule_diagnostics.crop_history_window = original_crop_history
            schedule_diagnostics._future_time_context_seq = original_future_context

        self.assertEqual(payload["n_rollout_calls"], 1)
        self.assertEqual(payload["field_evals_by_step"], [1.0, 1.0, 1.0])
        self.assertEqual(payload["disagreement_by_step"], [0.0, 1.0, 2.0])
        self.assertNotIn("trigger_rate", payload)
        self.assertNotIn("normalized_disagreement_by_step", payload)


if __name__ == "__main__":
    unittest.main()
