from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from residual_parameterization import (  # noqa: E402
    DEFAULT_BASIS_KIND,
    DEFAULT_MEDIUM_KL_BAND,
    DEFAULT_SMALL_KL_BAND,
    FORECAST_AVG_RELATIVE_OBJECTIVE,
    build_residual_basis,
    candidate_from_theta,
    forecast_average_relative_metric,
    generate_initial_perturbations,
    kl_divergence,
    normalize_observations,
    observation_objective,
)


class ResidualParameterizationTests(unittest.TestCase):
    def test_softmax_candidate_is_monotone_with_endpoints(self) -> None:
        q_ref = np.asarray([0.1, 0.2, 0.3, 0.25, 0.15], dtype=np.float64)
        basis = build_residual_basis(len(q_ref), q_ref=q_ref)
        theta = np.asarray([0.2, -0.1, 0.05, -0.15, 0.1], dtype=np.float64)
        q, grid = candidate_from_theta(q_ref, theta, basis=basis)

        self.assertEqual(basis.shape, (5, 5))
        self.assertAlmostEqual(float(np.sum(q)), 1.0)
        self.assertAlmostEqual(float(grid[0]), 0.0)
        self.assertAlmostEqual(float(grid[-1]), 1.0)
        self.assertTrue(bool(np.all(np.diff(grid) > 0.0)))

    def test_generate_initial_perturbations_hits_kl_bands(self) -> None:
        reference = {"q_ref": [0.1, 0.2, 0.3, 0.25, 0.15], "runtime_nfe": 5}
        payload = generate_initial_perturbations(reference, n_initial=16, seed=7, use_sobol=False)

        self.assertEqual(payload["basis_kind"], DEFAULT_BASIS_KIND)
        self.assertEqual(payload["basis_dim"], 5)
        self.assertEqual(payload["n_initial"], 16)
        self.assertEqual(len(payload["candidates"]), 16)
        for row in payload["candidates"]:
            kl = float(row["kl_to_reference"])
            if row["kl_band"] == "small":
                self.assertGreaterEqual(kl, DEFAULT_SMALL_KL_BAND[0] - 1e-7)
                self.assertLessEqual(kl, DEFAULT_SMALL_KL_BAND[1] + 1e-7)
            else:
                self.assertGreaterEqual(kl, DEFAULT_MEDIUM_KL_BAND[0] - 1e-7)
                self.assertLessEqual(kl, DEFAULT_MEDIUM_KL_BAND[1] + 1e-7)
            self.assertTrue(np.all(np.diff(row["schedule_grid"]) > 0.0))

    def test_generate_initial_perturbations_requires_8_to_16(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_initial must be in"):
            generate_initial_perturbations({"q_ref": [0.5, 0.5]}, n_initial=7, use_sobol=False)

    def test_four_dimensional_theta_is_rejected(self) -> None:
        q_ref = [0.2, 0.3, 0.25, 0.25]

        with self.assertRaisesRegex(ValueError, "expected 5"):
            candidate_from_theta(q_ref, [0.1, -0.1, 0.05, 0.0])

    def test_normalize_observations_computes_objective_and_schedule(self) -> None:
        q_ref = [0.2, 0.3, 0.25, 0.25]
        observations = [{"theta": [0.1, 0.0, 0.0, 0.0, 0.0], "metric_val": 2.0}]
        rows = normalize_observations(observations, q_ref=q_ref, lambda_kl=0.05)

        self.assertEqual(len(rows), 1)
        self.assertIn("schedule_grid", rows[0])
        self.assertAlmostEqual(
            rows[0]["objective_value"],
            observation_objective(2.0, rows[0]["kl_to_reference"], lambda_kl=0.05),
        )
        self.assertEqual(rows[0]["objective_type"], "provided_metric_val")

    def test_forecast_observation_computes_average_relative_metric(self) -> None:
        q_ref = [0.2, 0.3, 0.25, 0.25]
        observations = {
            "uniform_baseline": {"crps": 4.0, "mase": 2.0},
            "observations": [{"theta": [0.1, 0.0, 0.0, 0.0, 0.0], "crps": 3.0, "mase": 1.5}],
        }
        rows = normalize_observations(observations, q_ref=q_ref, lambda_kl=0.05)

        self.assertAlmostEqual(rows[0]["uniform_crps"], 4.0)
        self.assertAlmostEqual(rows[0]["uniform_mase"], 2.0)
        self.assertAlmostEqual(rows[0]["relative_crps_ratio"], 0.75)
        self.assertAlmostEqual(rows[0]["relative_mase_ratio"], 0.75)
        self.assertAlmostEqual(rows[0]["metric_val"], 0.75)
        self.assertEqual(rows[0]["objective_type"], FORECAST_AVG_RELATIVE_OBJECTIVE)
        self.assertAlmostEqual(
            rows[0]["objective_value"],
            observation_objective(0.75, rows[0]["kl_to_reference"], lambda_kl=0.05),
        )

    def test_row_level_forecast_baselines_without_session_are_rejected(self) -> None:
        q_ref = [0.2, 0.3, 0.25, 0.25]
        observations = [
            {
                "theta": [0.1, 0.0, 0.0, 0.0, 0.0],
                "crps": 3.0,
                "mase": 1.5,
                "uniform_crps": 4.0,
                "uniform_mase": 2.0,
            }
        ]

        with self.assertRaisesRegex(ValueError, "session uniform_baseline"):
            normalize_observations(observations, q_ref=q_ref, lambda_kl=0.05)

    def test_resumed_forecast_rows_keep_session_baseline_metadata(self) -> None:
        q_ref = [0.2, 0.3, 0.25, 0.25]
        observations = {
            "uniform_baseline": {"crps": 4.0, "mase": 2.0},
            "observations": [
                {
                    "theta": [0.1, 0.0, 0.0, 0.0, 0.0],
                    "crps": 3.0,
                    "mase": 1.5,
                    "metric_val": 0.75,
                    "objective_value": -0.75,
                    "objective_type": FORECAST_AVG_RELATIVE_OBJECTIVE,
                }
            ],
        }
        rows = normalize_observations(observations, q_ref=q_ref, lambda_kl=0.05)

        self.assertAlmostEqual(rows[0]["uniform_crps"], 4.0)
        self.assertAlmostEqual(rows[0]["uniform_mase"], 2.0)
        self.assertEqual(rows[0]["objective_type"], FORECAST_AVG_RELATIVE_OBJECTIVE)

    def test_forecast_metric_helper_uses_crps_and_mase_ratios(self) -> None:
        payload = forecast_average_relative_metric(
            {"crps": 2.0, "mase": 3.0},
            uniform_baseline={"uniform_crps": 4.0, "uniform_mase": 4.0},
        )

        self.assertAlmostEqual(payload["relative_crps_ratio"], 0.5)
        self.assertAlmostEqual(payload["relative_mase_ratio"], 0.75)
        self.assertAlmostEqual(payload["metric_val"], 0.625)

    def test_forecast_observation_requires_positive_uniform_baselines(self) -> None:
        with self.assertRaisesRegex(ValueError, "uniform_crps must be positive"):
            forecast_average_relative_metric(
                {"crps": 1.0, "mase": 1.0},
                uniform_baseline={"uniform_crps": 0.0, "uniform_mase": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "uniform_mase must be positive"):
            forecast_average_relative_metric(
                {"crps": 1.0, "mase": 1.0},
                uniform_baseline={"uniform_crps": 1.0, "uniform_mase": -1.0},
            )

    def test_session_uniform_baseline_requires_positive_values(self) -> None:
        q_ref = [0.2, 0.3, 0.25, 0.25]
        observations = {
            "uniform_baseline": {"crps": 0.0, "mase": 2.0},
            "observations": [{"theta": [0.0, 0.0, 0.0, 0.0, 0.0], "crps": 3.0, "mase": 1.5}],
        }

        with self.assertRaisesRegex(ValueError, "uniform_baseline.crps must be positive"):
            normalize_observations(observations, q_ref=q_ref)

    def test_row_level_baseline_must_match_session_baseline(self) -> None:
        q_ref = [0.2, 0.3, 0.25, 0.25]
        observations = {
            "uniform_baseline": {"crps": 4.0, "mase": 2.0},
            "observations": [
                {
                    "theta": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "crps": 3.0,
                    "mase": 1.5,
                    "uniform_crps": 4.1,
                    "uniform_mase": 2.0,
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "row-level uniform_crps must match session uniform_crps"):
            normalize_observations(observations, q_ref=q_ref)

    def test_kl_divergence_is_zero_at_reference(self) -> None:
        q_ref = [0.2, 0.3, 0.5]
        self.assertAlmostEqual(kl_divergence(q_ref, q_ref), 0.0)
        self.assertGreaterEqual(kl_divergence(q_ref, q_ref), 0.0)


if __name__ == "__main__":
    unittest.main()
