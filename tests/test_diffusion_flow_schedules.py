from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

import diffusion_flow_inference.schedule_transfer.otflow_schedule_diagnostics as schedule_diagnostics
from diffusion_flow_inference.schedule_transfer.diffusion_flow_schedules import (
    SCHEDULE_KEYS,
    _validate_schedule_grid,
    build_schedule_grid,
)
from diffusion_flow_inference.schedule_transfer.otflow_schedule_diagnostics import (
    _append_rollout_context_features,
    _collect_rollout_diagnostics,
)
from diffusion_flow_inference.schedule_transfer.result_tables import (
    augment_rows_with_relative_metrics,
)


class DiffusionFlowScheduleTests(unittest.TestCase):
    def test_transfer_schedule_grids_are_valid_for_runtime_steps(self) -> None:
        for schedule_key in ("ays", "gits", "ots"):
            for runtime_steps in (5, 6, 8, 10, 12, 16):
                with self.subTest(schedule_key=schedule_key, runtime_steps=runtime_steps):
                    grid = build_schedule_grid(schedule_key, runtime_steps)
                    self.assertEqual(len(grid), runtime_steps + 1)
                    self.assertAlmostEqual(grid[0], 0.0)
                    self.assertAlmostEqual(grid[-1], 1.0)
                    self.assertTrue(all(b > a for a, b in zip(grid, grid[1:])))

    def test_schedule_grid_rejects_nonpositive_steps(self) -> None:
        for schedule_key in SCHEDULE_KEYS:
            for runtime_steps in (0, -1):
                with self.subTest(schedule_key=schedule_key, runtime_steps=runtime_steps):
                    with self.assertRaisesRegex(ValueError, "n_steps must be positive"):
                        build_schedule_grid(schedule_key, runtime_steps)

    def test_flowts_power_sampling_grid_is_supported(self) -> None:
        grid = build_schedule_grid("flowts_power_sampling", 4)

        self.assertEqual(len(grid), 5)
        self.assertAlmostEqual(grid[0], 0.0)
        self.assertAlmostEqual(grid[-1], 1.0)
        self.assertTrue(all(b > a for a, b in zip(grid, grid[1:])))

    def test_schedule_grid_rejects_unknown_schedule(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported schedule"):
            build_schedule_grid("unknown", 4)

    def test_schedule_grid_validation_is_strict(self) -> None:
        invalid_grids = (
            [0.0, float("nan"), 1.0],
            [0.0, 0.6, 0.5, 1.0],
            [0.0, 0.5, 0.5, 1.0],
            [0.0, 1.1, 1.0],
            [0.1, 0.5, 1.0],
        )
        for grid in invalid_grids:
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    _validate_schedule_grid(grid, name="test")

    def test_summary_relative_metrics_preserve_seed_paired_gains(self) -> None:
        rows = [
            {
                "benchmark_family": "forecast_extrapolation",
                "dataset": "demo",
                "target_nfe": 10,
                "solver_key": "euler",
                "schedule_key": "uniform",
                "train_budget_label": "20k",
                "crps_mean": 10.0,
                "mase_mean": 5.0,
            },
            {
                "benchmark_family": "forecast_extrapolation",
                "dataset": "demo",
                "target_nfe": 10,
                "solver_key": "euler",
                "schedule_key": "gits",
                "train_budget_label": "20k",
                "crps_mean": 9.0,
                "mase_mean": 4.0,
                "relative_crps_gain_vs_uniform_mean": 0.11,
                "relative_mase_gain_vs_uniform_mean": 0.25,
            },
        ]
        augmented = augment_rows_with_relative_metrics(rows)
        gits = next(row for row in augmented if row["schedule_key"] == "gits")
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

    def test_rollout_context_rejects_incompatible_shapes(self) -> None:
        history = torch.zeros(1, 3, 2)
        block = torch.zeros(1, 2, 1)
        with self.assertRaisesRegex(ValueError, "none were provided"):
            _append_rollout_context_features(
                block,
                x_hist=history,
                future_context_seq=None,
                cursor=0,
                take=2,
            )
        with self.assertRaisesRegex(ValueError, "feature width"):
            _append_rollout_context_features(
                block,
                x_hist=history,
                future_context_seq=torch.zeros(1, 2, 2),
                cursor=0,
                take=2,
            )
        with self.assertRaisesRegex(ValueError, "cover the requested time slice"):
            _append_rollout_context_features(
                block,
                x_hist=history,
                future_context_seq=torch.zeros(1, 1, 1),
                cursor=0,
                take=2,
            )
        with self.assertRaisesRegex(ValueError, "exceeds history feature width"):
            _append_rollout_context_features(
                torch.zeros(1, 2, 3),
                x_hist=history,
                future_context_seq=None,
                cursor=0,
                take=2,
            )


if __name__ == "__main__":
    unittest.main()
