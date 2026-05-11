from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from botorch_bo import botorch_available, suggest_bo_batch  # noqa: E402
from residual_parameterization import FORECAST_AVG_RELATIVE_OBJECTIVE, observation_objective  # noqa: E402


class BotorchBoTests(unittest.TestCase):
    def test_objective_excludes_latency_and_smoothness(self) -> None:
        self.assertAlmostEqual(observation_objective(1.5, 0.2, lambda_kl=0.05), -1.51)

    @unittest.skipUnless(botorch_available(), "BoTorch is optional for future-work BO suggestions.")
    def test_qlognei_candidate_batch_shape(self) -> None:
        reference = {"q_ref": [0.15, 0.2, 0.25, 0.25, 0.15], "runtime_nfe": 5}
        observations = {
            "uniform_baseline": {"crps": 4.0, "mase": 2.0},
            "observations": [
                {"theta": [0.0, 0.0, 0.0, 0.0, 0.0], "crps": 4.0, "mase": 2.0},
                {"theta": [0.2, -0.1, 0.1, -0.2, 0.0], "crps": 3.8, "mase": 1.9},
                {"theta": [-0.2, 0.1, -0.1, 0.2, 0.1], "crps": 4.2, "mase": 2.1},
                {"theta": [0.1, 0.2, -0.2, 0.0, -0.1], "crps": 3.9, "mase": 2.0},
            ]
        }

        batch = suggest_bo_batch(reference, observations, batch_size=2, raw_samples=16, num_restarts=2, n_mc_samples=16)

        self.assertEqual(batch["acquisition"], "qLogNoisyExpectedImprovement")
        self.assertEqual(batch["objective_type"], FORECAST_AVG_RELATIVE_OBJECTIVE)
        self.assertEqual(batch["uniform_baseline"], {"crps": 4.0, "mase": 2.0})
        self.assertEqual(batch["basis_kind"], "tilt_curvature_three_local_bumps")
        self.assertEqual(batch["basis_dim"], 5)
        self.assertEqual(len(batch["candidates"]), 2)
        for row in batch["candidates"]:
            self.assertEqual(len(row["theta"]), 5)
            self.assertEqual(len(row["schedule_grid"]), 6)


if __name__ == "__main__":
    unittest.main()
