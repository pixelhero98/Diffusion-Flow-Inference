from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization as runner
from diffusion_flow_inference.schedules.diffusion_flow import build_schedule_grid, load_external_schedule_catalog, schedule_time_alignment
from diffusion_flow_inference.schedules.paper_registry import (
    BASELINE_SCHEDULE_KEYS,
    FLOW_TIME_SCHEDULE_KEYS,
    METHOD_KEY,
    TRANSFER_SCHEDULE_KEYS,
    paper_registry_snapshot,
    paper_schedule_specs,
)
from diffusion_flow_inference.diagnostics.signal_traces import NATIVE_INFO_GROWTH_TRACE_KEY, NATIVE_SIGNAL_TRACE_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AYS_SD15_SIGMAS = np.asarray([14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029], dtype=np.float64)
AYS_SD15_TIMESTEPS = np.asarray([999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0], dtype=np.float64)
GITS_OLD_EDM_CIFAR_SIGMAS = np.asarray([80.0, 10.9836, 3.8811, 1.5840, 0.5666, 0.1698, 0.0020], dtype=np.float64)
GITS_COMFY_COEFF_120_SIGMAS = {
    10: np.asarray([14.61464119, 5.85520077, 2.84484982, 1.67050016, 1.08895338, 0.74807048, 0.50118381, 0.32104823, 0.19894916, 0.09824532, 0.02916753], dtype=np.float64),
    12: np.asarray([14.61464119, 5.85520077, 3.07277966, 1.98035145, 1.36964464, 0.95350921, 0.69515091, 0.50118381, 0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753], dtype=np.float64),
    16: np.asarray([14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.83188516, 0.64427125, 0.50118381, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753], dtype=np.float64),
    20: np.asarray([14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.19988537, 1.61558151, 1.24153244, 0.95350921, 0.74807048, 0.59516323, 0.50118381, 0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753], dtype=np.float64),
}


def _normalize_descending(values: np.ndarray) -> np.ndarray:
    return (float(values[0]) - values) / float(values[0] - values[-1])


def _old_ays_flow_time_resample(n_steps: int) -> np.ndarray:
    progression = _normalize_descending(AYS_SD15_TIMESTEPS)
    src = np.linspace(0.0, 1.0, int(progression.size), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n_steps) + 1, dtype=np.float64)
    return np.interp(dst, src, progression)


def _old_gits_flow_time_resample(n_steps: int) -> np.ndarray:
    progression = _normalize_descending(GITS_OLD_EDM_CIFAR_SIGMAS)
    src = np.linspace(0.0, 1.0, int(progression.size), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n_steps) + 1, dtype=np.float64)
    return np.interp(dst, src, progression)


def _loglinear_descending(values: np.ndarray, n_steps: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, int(values.size), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n_steps) + 1, dtype=np.float64)
    interpolated = np.exp(np.interp(dst, src, np.log(values)))
    interpolated[0] = values[0]
    interpolated[-1] = values[-1]
    return interpolated


def _ays_logsigma_expected(n_steps: int) -> np.ndarray:
    sigmas = _loglinear_descending(AYS_SD15_SIGMAS, n_steps)
    return _normalize_descending(sigmas)


class DiffusionFlowPaperPrepTests(unittest.TestCase):
    def test_registry_exposes_diffusion_flow_method_not_tvd(self) -> None:
        snapshot = paper_registry_snapshot()
        self.assertEqual(METHOD_KEY, "diffusion_flow_time_reparameterization")
        self.assertEqual(snapshot["paper_method"], "diffusion_flow_time_reparameterization")
        self.assertFalse(any(spec.comparison_role == "paper_method" and spec.key == "tvd" for spec in paper_schedule_specs()))

    def test_schedule_sets_are_exact(self) -> None:
        self.assertEqual(BASELINE_SCHEDULE_KEYS, ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"))
        self.assertEqual(FLOW_TIME_SCHEDULE_KEYS, ("late_power_3", "flowts_power_sampling"))
        self.assertEqual(TRANSFER_SCHEDULE_KEYS, ("ays", "gits", "ots"))

    def test_active_schedule_grids_have_endpoints(self) -> None:
        for key in BASELINE_SCHEDULE_KEYS:
            grid = build_schedule_grid(key, 4)
            self.assertIsNotNone(grid, key)
            self.assertEqual(len(grid), 5)
            self.assertAlmostEqual(grid[0], 0.0)
            self.assertAlmostEqual(grid[-1], 1.0)
            self.assertTrue(all(right > left for left, right in zip(grid, grid[1:])), key)

    def test_active_schedule_grids_reject_non_positive_steps(self) -> None:
        for key in BASELINE_SCHEDULE_KEYS:
            for n_steps in (0, -1):
                with self.assertRaisesRegex(ValueError, "n_steps must be positive"):
                    build_schedule_grid(key, n_steps)

    def test_scheduler_cases_evaluate_uniform_first(self) -> None:
        args = runner.build_argparser().parse_args(["--baseline_scheduler_names", "ays,uniform"])
        cases = runner._scheduler_cases_for_datasets(args, ["electricity"])
        self.assertEqual([case["scheduler_key"] for case in cases["electricity"]], ["uniform", "ays"])

    def test_aggregate_relative_gain_uses_fraction_units(self) -> None:
        rows = [
            {
                "benchmark_family": "forecast_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "dataset": "electricity",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "ays",
                "experiment_scope": "main",
                "row_status": "complete",
                "crps": 3.0,
            },
            {
                "benchmark_family": "forecast_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "dataset": "electricity",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "experiment_scope": "main",
                "row_status": "complete",
                "crps": 4.0,
            },
        ]

        summary = runner._aggregate_main_table(rows)["seed_summaries"]
        by_schedule = {row["scheduler_key"]: row for row in summary}

        self.assertAlmostEqual(runner._safe_relative_gain(3.0, 4.0), 0.25)
        self.assertAlmostEqual(by_schedule["ays"]["relative_crps_gain_vs_uniform"], 0.25)
        self.assertAlmostEqual(by_schedule["uniform"]["relative_crps_gain_vs_uniform"], 0.0)

    def test_flowts_power_sampling_grid_is_active_flow_time_schedule(self) -> None:
        n_steps = 10
        grid = np.asarray(build_schedule_grid("flowts_power_sampling", n_steps), dtype=np.float64)
        expected = (np.arange(n_steps + 1, dtype=np.float64) / float(n_steps)) ** 0.03
        expected[0] = 0.0
        expected[-1] = 1.0
        self.assertTrue(np.allclose(grid, expected, atol=1e-12, rtol=1e-12))
        self.assertTrue(bool(np.all(np.diff(grid) > 0.0)))

    def test_ays_uses_published_sd15_logsigma_grid_for_exact_10_steps(self) -> None:
        grid = np.asarray(build_schedule_grid("ays", 10), dtype=np.float64)
        expected = _ays_logsigma_expected(10)
        self.assertTrue(np.allclose(grid, expected, atol=1e-12, rtol=1e-12))

    def test_ays_interpolates_in_diffusion_domain_before_flow_mapping(self) -> None:
        for n_steps in (12, 16):
            grid = np.asarray(build_schedule_grid("ays", n_steps), dtype=np.float64)
            expected = _ays_logsigma_expected(n_steps)
            old_flow_resample = _old_ays_flow_time_resample(n_steps)
            self.assertTrue(np.allclose(grid, expected, atol=1e-12, rtol=1e-12))
            self.assertFalse(np.allclose(grid, old_flow_resample, atol=1e-6, rtol=1e-6))
            self.assertEqual(len(grid), n_steps + 1)
            self.assertAlmostEqual(float(grid[0]), 0.0)
            self.assertAlmostEqual(float(grid[-1]), 1.0)
            self.assertTrue(bool(np.all(np.diff(grid) > 0.0)))

    def test_gits_uses_comfyui_sd_sigma_tables_for_main_nfes(self) -> None:
        self.assertEqual(schedule_time_alignment("gits"), "runtime_gits_sd15_logsigma_coeff_1_20")
        for n_steps in (10, 12, 16):
            grid = np.asarray(build_schedule_grid("gits", n_steps), dtype=np.float64)
            expected = _normalize_descending(GITS_COMFY_COEFF_120_SIGMAS[n_steps])
            old_flow_resample = _old_gits_flow_time_resample(n_steps)
            self.assertTrue(np.allclose(grid, expected, atol=1e-12, rtol=1e-12))
            self.assertFalse(np.allclose(grid, old_flow_resample, atol=1e-6, rtol=1e-6))
            self.assertEqual(len(grid), n_steps + 1)
            self.assertAlmostEqual(float(grid[0]), 0.0)
            self.assertAlmostEqual(float(grid[-1]), 1.0)
            self.assertTrue(bool(np.all(np.diff(grid) > 0.0)))

    def test_gits_above_20_steps_uses_comfyui_loglinear_interpolation(self) -> None:
        n_steps = 24
        grid = np.asarray(build_schedule_grid("gits", n_steps), dtype=np.float64)
        expected_sigmas = _loglinear_descending(GITS_COMFY_COEFF_120_SIGMAS[20], n_steps)
        expected = _normalize_descending(expected_sigmas)
        self.assertTrue(np.allclose(grid, expected, atol=1e-12, rtol=1e-12))
        self.assertEqual(len(grid), n_steps + 1)
        self.assertAlmostEqual(float(grid[0]), 0.0)
        self.assertAlmostEqual(float(grid[-1]), 1.0)
        self.assertTrue(bool(np.all(np.diff(grid) > 0.0)))

    def test_external_schedule_catalog_documents_mapping_limits(self) -> None:
        catalog = load_external_schedule_catalog()
        self.assertIn("diffusion noise space", catalog["ays"]["notes"])
        self.assertIn("ComfyUI Stable Diffusion GITSScheduler", catalog["gits"]["notes"])
        self.assertIn("not a full GITS dynamic-programming warmup optimizer", catalog["gits"]["notes"])
        self.assertIn("not a vendored full replication", catalog["ots"]["notes"])
        self.assertIn("ATSS-inspired", catalog["atss"]["notes"])
        self.assertIn("flow-time late-biased", catalog["flowts_power_sampling"]["notes"])

    def test_registry_classifies_flow_time_and_transfer_schedules_separately(self) -> None:
        specs = {spec.key: spec for spec in paper_schedule_specs()}
        snapshot = paper_registry_snapshot()
        self.assertEqual(snapshot["flow_time_schedule_keys"], ["late_power_3", "flowts_power_sampling"])
        self.assertEqual(specs["late_power_3"].family, "flow_time_late_biased")
        self.assertEqual(specs["flowts_power_sampling"].family, "flow_time_late_biased")
        self.assertNotEqual(specs["flowts_power_sampling"].family, "diffusion_schedule_transfer")
        self.assertEqual(specs["ays"].family, "diffusion_schedule_transfer")

    def test_native_hardness_trace_is_info_growth(self) -> None:
        self.assertEqual(NATIVE_INFO_GROWTH_TRACE_KEY, "info_growth_hardness_by_step")
        self.assertIn("info_growth_hardness_by_step", NATIVE_SIGNAL_TRACE_KEYS)

    def test_runner_dry_run_writes_combined_summary(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--lob_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                ]
            )
            payload = runner.run_diffusion_flow_time_reparameterization(args)
            summary = json.loads((Path(tmpdir) / "combined_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["runner_mode"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["method_key"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["flow_time_schedule_keys"], ["late_power_3", "flowts_power_sampling"])
        self.assertEqual(summary["transfer_schedule_keys"], ["ays", "gits", "ots"])

    def test_public_source_surface_excludes_local_and_cluster_artifacts(self) -> None:
        self.assertFalse((PROJECT_ROOT / "code").exists())
        self.assertFalse((PROJECT_ROOT / "legacy").exists())
        self.assertFalse((PROJECT_ROOT / "ops").exists())
        self.assertFalse((PROJECT_ROOT / "lesson.md").exists())


if __name__ == "__main__":
    unittest.main()
