from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization as runner
from diffusion_flow_inference.schedule_transfer.diffusion_flow_schedules import build_schedule_grid
from diffusion_flow_inference.evaluation.fm_backbone_registry import materialize_backbone_manifest
from diffusion_flow_inference.schedule_transfer.otflow_paper_registry import (
    BASELINE_SCHEDULE_KEYS,
    METHOD_KEY,
    TRANSFER_SCHEDULE_KEYS,
    paper_registry_snapshot,
    paper_schedule_specs,
)
from diffusion_flow_inference.schedule_transfer.otflow_signal_traces import NATIVE_INFO_GROWTH_TRACE_KEY, NATIVE_SIGNAL_TRACE_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiffusionFlowPaperPrepTests(unittest.TestCase):
    def test_registry_exposes_diffusion_flow_method_not_tvd(self) -> None:
        snapshot = paper_registry_snapshot()
        self.assertEqual(METHOD_KEY, "diffusion_flow_time_reparameterization")
        self.assertEqual(snapshot["paper_method"], "diffusion_flow_time_reparameterization")
        self.assertFalse(any(spec.comparison_role == "paper_method" and spec.key == "tvd" for spec in paper_schedule_specs()))
        self.assertIn("flowts_power_sampling", {spec.key for spec in paper_schedule_specs()})
        self.assertNotIn("atss", {spec.key for spec in paper_schedule_specs()})

    def test_schedule_sets_are_exact(self) -> None:
        self.assertEqual(BASELINE_SCHEDULE_KEYS, ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"))
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
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                ]
            )
            payload = runner.run_diffusion_flow_time_reparameterization(args)
            summary = json.loads((Path(tmpdir) / "combined_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["runner_mode"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["method_key"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["conditional_generation_datasets"], [])
        retired_dataset_key = "lo" + "b_datasets"
        self.assertNotIn(retired_dataset_key, summary)
        self.assertIn("flowts_power_sampling", summary["baseline_schedule_keys"])
        self.assertEqual(summary["transfer_schedule_keys"], ["ays", "gits", "ots"])

    def test_conditional_generation_build_row_preserves_full_metrics(self) -> None:
        row = runner._build_row(
            benchmark_family="conditional_generation",
            split_phase="locked_test",
            seed=0,
            dataset="sleep_edf",
            checkpoint={
                "checkpoint_id": "ck",
                "checkpoint_path": "outputs/example/model.pt",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
            },
            target_nfe=10,
            runtime_nfe=10,
            solver_key="euler",
            scheduler_key="uniform",
            details={"reference_macro_steps": 10, "schedule_grid_hash": "grid"},
            metrics={
                "score_main": 0.4,
                "tstr_macro_f1": 0.5,
                "disc_auc": 0.6,
                "disc_auc_gap": 0.1,
                "unconditional_w1": 0.2,
                "conditional_w1": 0.3,
                "u_l1": 0.7,
                "c_l1": 0.8,
                "spread_specific_error": 0.9,
                "imbalance_specific_error": 1.1,
                "ret_vol_acf_error": 1.2,
                "impact_response_error": 0.25,
                "stage_mismatch_rate": 0.2,
                "stage_classifier_real_macro_f1": 0.75,
                "sleep_signal_mae": 0.9,
                "sleep_spectral_mae": 1.1,
                "sleep_stage_mismatch_rate": 0.2,
                "sleep_stage_classifier_real_macro_f1": 0.75,
                "eval_horizon": 3000,
                "evaluation_protocol_hash": "protocol",
                "chosen_t0s_hash": "windows",
                "chosen_examples_hash": "examples",
                "stage_counts_json": '{"N2":2}',
            },
            row_signature="sig",
            protocol_hash="hash",
        )

        for key in (
            "disc_auc",
            "disc_auc_gap",
            "unconditional_w1",
            "u_l1",
            "c_l1",
            "spread_specific_error",
            "imbalance_specific_error",
            "ret_vol_acf_error",
            "impact_response_error",
            "stage_mismatch_rate",
            "stage_classifier_real_macro_f1",
            "sleep_signal_mae",
            "sleep_spectral_mae",
            "sleep_stage_mismatch_rate",
        ):
            self.assertIn(key, row)
        self.assertEqual(row["eval_horizon"], 3000)
        self.assertEqual(row["schedule_grid_hash"], "grid")
        self.assertEqual(row["chosen_examples_hash"], "examples")

    def test_row_recorder_drops_stale_protocol_rows(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "10",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            recorder["fh"].close()
            row_path = Path(tmpdir) / "rows.jsonl"
            row_path.write_text('{"protocol_hash":"old","row_status":"complete"}\n', encoding="utf-8")

            args_changed = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "12",
                ]
            )
            recorder_changed = runner._init_row_recorder(Path(tmpdir), args_changed)
            recorder_changed["fh"].close()
            self.assertEqual(row_path.read_text(encoding="utf-8"), "")

    def test_protocol_hash_tracks_data_path_identity(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            data_a = Path(tmpdir) / "cryptos_a.npz"
            data_b = Path(tmpdir) / "cryptos_b.npz"
            data_a.write_bytes(b"a")
            data_b.write_bytes(b"bb")
            args_a = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--backbone_manifest",
                    str(manifest),
                    "--cryptos_path",
                    str(data_a),
                ]
            )
            args_b = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--backbone_manifest",
                    str(manifest),
                    "--cryptos_path",
                    str(data_b),
                ]
            )
            self.assertNotEqual(runner._protocol_config_fingerprint(args_a), runner._protocol_config_fingerprint(args_b))

    def test_protocol_hash_tracks_selected_seeds(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
        args_a = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                str(manifest),
                "--seeds",
                "0",
            ]
        )
        args_b = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                str(manifest),
                "--seeds",
                "1",
            ]
        )
        self.assertNotEqual(runner._protocol_config_fingerprint(args_a), runner._protocol_config_fingerprint(args_b))

    def test_site_specific_ops_scripts_are_not_in_source_release(self) -> None:
        self.assertFalse((PROJECT_ROOT / "code" / "ops").exists())
        self.assertFalse(any(PROJECT_ROOT.glob("opsi*")))

    def test_retired_source_trees_are_absent(self) -> None:
        self.assertFalse((PROJECT_ROOT / "code").exists())
        self.assertFalse((PROJECT_ROOT / "src" / "old_code").exists())
        self.assertFalse((PROJECT_ROOT / "old_code").exists())

    def test_legacy_cleanup_targets_are_removed(self) -> None:
        removed = {
            "adaptive_noise_sampler_followup.py",
            "adaptive_deterministic_refinement_followup.py",
            "build_adaptive_solver_matched_nfe_study.py",
            "benchmark_otflow_suite.py",
            "baselines.py",
            "deepmarket_baselines.py",
            "temporal_baselines.py",
            "otflow_baselines.py",
            "fm_backbone_readiness_audit.py",
            "merge_otflow_baseline_main_table.py",
            "otflow_dataset_audit.py",
            "otflow_rollout_length_review.py",
        }
        src_root = PROJECT_ROOT / "src"
        self.assertFalse(any((src_root / name).exists() for name in removed))
        source_text = "\n".join(
            path.read_text(encoding="utf-8") for path in src_root.rglob("*.py") if path.name != Path(__file__).name
        )
        for name in removed:
            self.assertNotIn(name.removesuffix(".py"), source_text)

    def test_retired_generic_naming_tokens_are_absent(self) -> None:
        retired_patterns = (
            r"RectifiedFlowL[O]B",
            r"L[O]BConfig",
            r"L[O]BDataConfig",
            r"WindowedL[O]BParamsDataset",
            r"L[O]B_FAMILY",
            r"--l[o]b_datasets",
            r"l[o]b_conditional_generation",
            r"['\"]l[o]b['\"]",
            r"[/\\]l[o]b[/\\]",
            r"models\.otflow_backbone",
        )
        source_paths = [
            *Path(PROJECT_ROOT / "src").rglob("*.py"),
            *Path(PROJECT_ROOT / "tests").rglob("*.py"),
            *Path(PROJECT_ROOT / "scripts").rglob("*.py"),
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "pyproject.toml",
        ]
        source_text = "\n".join(
            path.read_text(encoding="utf-8") for path in source_paths if path.exists() and path != Path(__file__)
        )
        for pattern in retired_patterns:
            self.assertIsNone(re.search(pattern, source_text), pattern)

    def test_backbone_manifest_tracks_40_active_artifacts_without_private_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = materialize_backbone_manifest(
                matrix_root=root / "matrix",
                otflow_reuse_root=root / "reuse",
                imported_backbone_root=root / "imported",
                write_path=root / "manifest.json",
            )
        self.assertEqual(int(payload.get("artifact_count", 0)), 40)
        self.assertEqual(int(payload.get("ready_count", -1)), 0)
        self.assertEqual(int(payload.get("missing_count", 0)), 40)


if __name__ == "__main__":
    unittest.main()
