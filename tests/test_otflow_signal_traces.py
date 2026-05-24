from __future__ import annotations

import unittest

import numpy as np
import torch

from diffusion_flow_inference.schedule_transfer.otflow_signal_traces import (
    VELOCITY_VARIATION_DIFFICULTY_ROW_KEY,
    VELOCITY_VARIATION_DIFFICULTY_TRACE_KEY,
    compute_velocity_variation_difficulty,
    compute_velocity_variation_difficulty_numpy,
    resolved_velocity_variation_scale,
)


class OTFlowSignalTraceTests(unittest.TestCase):
    def test_velocity_variation_names_are_strict(self) -> None:
        self.assertEqual(VELOCITY_VARIATION_DIFFICULTY_ROW_KEY, "velocity_variation_difficulty")
        self.assertEqual(VELOCITY_VARIATION_DIFFICULTY_TRACE_KEY, "velocity_variation_difficulty_by_step")

    def test_velocity_variation_formula_matches_numpy_and_torch(self) -> None:
        residual = np.asarray([0.0, 2.0, -3.0, 6.0], dtype=np.float64)
        disagreement = np.asarray([1.0, 0.5, 2.0, 0.25], dtype=np.float64)
        scale = 2.0

        expected = disagreement * np.log1p(np.clip(residual, 0.0, None) / scale)
        actual_np = compute_velocity_variation_difficulty_numpy(residual, disagreement, scale=scale)
        actual_torch = compute_velocity_variation_difficulty(
            torch.as_tensor(residual, dtype=torch.float64),
            torch.as_tensor(disagreement, dtype=torch.float64),
            scale=scale,
        ).numpy()

        self.assertTrue(np.allclose(actual_np, expected, atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(actual_torch, expected, atol=1e-12, rtol=1e-12))

    def test_velocity_variation_scale_uses_clipped_finite_residual_mean(self) -> None:
        scale = resolved_velocity_variation_scale([float("nan"), -3.0, 1.0, 3.0])
        self.assertAlmostEqual(scale, 4.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
