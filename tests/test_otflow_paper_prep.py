from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization as runner
from diffusion_flow_inference.schedules.diffusion_flow import build_schedule_grid, load_external_schedule_catalog
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


def _normalize_descending(values: np.ndarray) -> np.ndarray:
    return (float(values[0]) - values) / float(values[0] - values[-1])


def _old_ays_flow_time_resample(n_steps: int) -> np.ndarray:
    progression = _normalize_descending(AYS_SD15_TIMESTEPS)
    src = np.linspace(0.0, 1.0, int(progression.size), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n_steps) + 1, dtype=np.float64)
    return np.interp(dst, src, progression)


def _ays_logsigma_expected(n_steps: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, int(AYS_SD15_SIGMAS.size), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n_steps) + 1, dtype=np.float64)
    sigmas = np.exp(np.interp(dst, src, np.log(AYS_SD15_SIGMAS)))
    sigmas[0] = AYS_SD15_SIGMAS[0]
    sigmas[-1] = AYS_SD15_SIGMAS[-1]
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

    def test_external_schedule_catalog_documents_mapping_limits(self) -> None:
        catalog = load_external_schedule_catalog()
        self.assertIn("diffusion noise space", catalog["ays"]["notes"])
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
