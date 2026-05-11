from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import reference_schedule as reference_schedule_module  # noqa: E402
from reference_schedule import (  # noqa: E402
    build_reference_from_cell,
    build_reference_from_payload,
    local_defect_from_oracle,
    runtime_grid_from_density,
    validate_time_grid,
)


def _cell(**overrides):
    payload = {
        "dataset": "electricity",
        "benchmark_family": "forecast_extrapolation",
        "solver_key": "euler",
        "target_nfe": 4,
        "runtime_nfe": 4,
        "reference_time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "validation_local_defect_trace": [1.0, 8.0, 1.0, 1.0],
        "validation_info_growth_trace": [1.0, 1.0, 8.0, 1.0],
        "checkpoint_id": "synthetic",
    }
    payload.update(overrides)
    return payload


class ReferenceScheduleTests(unittest.TestCase):
    def test_validate_time_grid_rejects_invalid_dense_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "start at 0.0"):
            validate_time_grid([0.1, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_time_grid([0.0, 0.5, 0.5, 1.0])

    def test_local_defect_is_primary_trace_source(self) -> None:
        reference = build_reference_from_cell(_cell())

        self.assertEqual(reference["trace_source"], "local_defect")
        self.assertEqual(reference["runtime_nfe"], 4)
        self.assertEqual(len(reference["schedule_grid"]), 5)
        self.assertTrue(np.all(np.diff(reference["schedule_grid"]) > 0.0))
        self.assertAlmostEqual(sum(reference["q_ref"]), 1.0)
        self.assertLess(reference["q_ref"][1], 0.25)

    def test_oracle_local_error_fallback_converts_to_defect(self) -> None:
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        local_defect = [1.0, 8.0, 1.0, 1.0]
        widths = np.diff(grid)
        oracle = (np.asarray(local_defect) * widths ** 2).tolist()
        reference = build_reference_from_cell(
            _cell(validation_local_defect_trace=[0.0, 0.0, 0.0, 0.0], validation_oracle_local_error_trace=oracle)
        )

        self.assertEqual(reference["trace_source"], "oracle_local_error")
        converted = local_defect_from_oracle(oracle, grid, solver_order=1.0)
        self.assertTrue(np.allclose(converted, local_defect, atol=1e-10, rtol=1e-10))

    def test_info_growth_is_last_fallback(self) -> None:
        reference = build_reference_from_cell(
            _cell(
                validation_local_defect_trace=[0.0, 0.0, 0.0, 0.0],
                validation_oracle_local_error_trace=None,
            )
        )

        self.assertEqual(reference["trace_source"], "info_growth_fallback")
        self.assertGreater(reference["schedule_grid"][2], 0.5)

    def test_inverse_cdf_places_runtime_nodes(self) -> None:
        grid = runtime_grid_from_density([0.0, 0.5, 1.0], [3.0, 1.0], runtime_nfe=4)

        self.assertEqual(len(grid), 5)
        self.assertTrue(np.all(np.diff(grid) > 0.0))
        self.assertLess(grid[1], 0.25)

    def test_build_reference_from_payload_selects_cell(self) -> None:
        payload = {"cells": [_cell(), _cell(dataset="solar_energy_10m")]}
        reference = build_reference_from_payload(payload, dataset="solar_energy_10m", solver="euler", target_nfe=4)

        self.assertEqual(reference["dataset"], "solar_energy_10m")

    def test_runtime_nfe_uses_solver_macro_steps_when_available(self) -> None:
        old_solver_macro_steps = reference_schedule_module.solver_macro_steps
        reference_schedule_module.solver_macro_steps = lambda solver, target_nfe: 5
        try:
            reference = build_reference_from_cell(_cell(solver_key="heun", target_nfe=10, runtime_nfe=None))
        finally:
            reference_schedule_module.solver_macro_steps = old_solver_macro_steps

        self.assertEqual(reference["runtime_nfe"], 5)


if __name__ == "__main__":
    unittest.main()
