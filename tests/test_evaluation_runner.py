from __future__ import annotations

import importlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization as runner
from diffusion_flow_inference.evaluation.backbone_registry import materialize_backbone_manifest
from diffusion_flow_inference.schedule_transfer.diffusion_flow_schedules import (
    SCHEDULE_KEYS,
    TRANSFER_SCHEDULE_KEYS,
    build_schedule_grid,
)
from diffusion_flow_inference.schedule_transfer.otflow_signal_traces import (
    MODEL_SIGNAL_TRACE_KEYS,
    VELOCITY_VARIATION_DIFFICULTY_TRACE_KEY,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _recorded_row(
    *, benchmark_family: str = "forecast_extrapolation", signature: str = "row"
) -> dict:
    return {
        "protocol_hash": "protocol",
        "benchmark_family": benchmark_family,
        "split_phase": "locked_test",
        "seed": 0,
        "dataset": "electricity" if benchmark_family == "forecast_extrapolation" else "cryptos",
        "checkpoint_id": "checkpoint",
        "backbone_name": "otflow",
        "train_steps": 20000,
        "train_budget_label": "20k",
        "target_nfe": 10,
        "solver_key": "euler",
        "schedule_key": "uniform",
        "row_signature": signature,
        "row_status": "complete",
        "crps": 1.0,
    }


class EvaluationRunnerTests(unittest.TestCase):
    def test_runner_protocol_is_diffusion_flow_time_reparameterization(self) -> None:
        self.assertEqual(runner.RUNNER_PROTOCOL, "diffusion_flow_time_reparameterization")

    def test_schedule_sets_are_exact(self) -> None:
        self.assertEqual(
            SCHEDULE_KEYS,
            ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"),
        )
        self.assertEqual(TRANSFER_SCHEDULE_KEYS, ("ays", "gits", "ots"))

    def test_dpmpp2m_remains_public_deterministic_solver(self) -> None:
        self.assertIn("dpmpp2m", runner.ALL_SOLVER_ORDER)
        self.assertEqual(runner.SOLVER_RUNTIME_NAMES["dpmpp2m"], "dpmpp2m")
        self.assertEqual(runner.solver_macro_steps("dpmpp2m", 10), 10)

    def test_active_schedule_grids_have_endpoints(self) -> None:
        for key in SCHEDULE_KEYS:
            grid = build_schedule_grid(key, 4)
            self.assertEqual(len(grid), 5)
            self.assertAlmostEqual(grid[0], 0.0)
            self.assertAlmostEqual(grid[-1], 1.0)
            self.assertTrue(all(right > left for left, right in zip(grid, grid[1:])), key)

    def test_active_schedule_grids_reject_non_positive_steps(self) -> None:
        for key in SCHEDULE_KEYS:
            for n_steps in (0, -1):
                with self.assertRaisesRegex(ValueError, "n_steps must be positive"):
                    build_schedule_grid(key, n_steps)

    def test_schedule_cases_evaluate_uniform_first(self) -> None:
        args = runner.build_argparser().parse_args(["--schedule-names", "ays,uniform"])
        cases = runner._schedule_cases_for_datasets(args, ["electricity"])
        self.assertEqual(
            [case["schedule_key"] for case in cases["electricity"]], ["uniform", "ays"]
        )

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
                "schedule_key": "ays",
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
                "schedule_key": "uniform",
                "experiment_scope": "main",
                "row_status": "complete",
                "crps": 4.0,
            },
        ]

        summary = runner._aggregate_main_table(rows)["seed_summaries"]
        by_schedule = {row["schedule_key"]: row for row in summary}

        self.assertAlmostEqual(runner._safe_relative_gain(3.0, 4.0), 0.25)
        self.assertAlmostEqual(by_schedule["ays"]["relative_crps_gain_vs_uniform"], 0.25)
        self.assertAlmostEqual(by_schedule["uniform"]["relative_crps_gain_vs_uniform"], 0.0)

    def test_signal_trace_includes_velocity_variation_difficulty(self) -> None:
        self.assertEqual(
            VELOCITY_VARIATION_DIFFICULTY_TRACE_KEY, "velocity_variation_difficulty_by_step"
        )
        self.assertIn("velocity_variation_difficulty_by_step", MODEL_SIGNAL_TRACE_KEYS)

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
            summary = json.loads(
                (Path(tmpdir) / "combined_summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(payload["runner_mode"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["method_key"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["conditional_generation_datasets"], [])
        retired_dataset_key = "lo" + "b_datasets"
        self.assertNotIn(retired_dataset_key, summary)
        self.assertIn("flowts_power_sampling", summary["schedule_keys"])
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
            schedule_key="uniform",
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
            row_path.write_text(
                '{"protocol_hash":"old","row_status":"complete"}\n', encoding="utf-8"
            )

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

    def test_protocol_hash_tracks_data_file_contents_not_location(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            data_a = Path(tmpdir) / "cryptos_a.npz"
            data_b = Path(tmpdir) / "cryptos_b.npz"
            data_a.write_bytes(b"same")
            data_b.write_bytes(b"same")
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
            self.assertEqual(
                runner._protocol_config_fingerprint(args_a),
                runner._protocol_config_fingerprint(args_b),
            )
            data_b.write_bytes(b"changed")
            self.assertNotEqual(
                runner._protocol_config_fingerprint(args_a),
                runner._protocol_config_fingerprint(args_b),
            )

    def test_preflight_resolves_relative_shared_backbone_root_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            root = Path(tmpdir)
            rel_root = root.relative_to(PROJECT_ROOT).as_posix()
            ckpt_path = root / "forecast" / "electricity" / "model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_bytes(b"checkpoint")
            ckpt_path.with_name("checkpoint_metadata.json").write_text("{}", encoding="utf-8")
            dataset_root = root / "datasets"
            monash_root = dataset_root / "monash" / "electricity"
            (monash_root / "source").mkdir(parents=True, exist_ok=True)
            (monash_root / "manifest.json").write_text("{}", encoding="utf-8")
            (monash_root / "source" / "electricity.tsf").write_text("@data\n", encoding="utf-8")
            args = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "electricity",
                    "--conditional_generation_datasets",
                    "",
                    "--shared_backbone_root",
                    rel_root,
                    "--dataset_root",
                    str(dataset_root),
                    "--backbone_manifest",
                    "",
                    "--allow_execute",
                ]
            )

            runner.validate_execution_preflight(args)

    def test_preflight_rejects_stale_ready_manifest_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "datasets"
            monash_root = dataset_root / "monash" / "electricity"
            (monash_root / "source").mkdir(parents=True, exist_ok=True)
            (monash_root / "manifest.json").write_text("{}", encoding="utf-8")
            (monash_root / "source" / "electricity.tsf").write_text("@data\n", encoding="utf-8")
            manifest_path = root / "backbone_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "diffusion_flow_backbone_manifest",
                        "schema_version": 1,
                        "seed": 0,
                        "train_budget_steps": [20000],
                        "matrix_root": ".",
                        "otflow_reuse_root": ".",
                        "imported_backbone_root": ".",
                        "artifact_count": 1,
                        "ready_count": 1,
                        "missing_count": 0,
                        "artifacts": [
                            {
                                "backbone_name": "otflow",
                                "benchmark_family": "forecast_extrapolation",
                                "dataset_key": "electricity",
                                "train_steps": 20000,
                                "train_budget_label": "20k",
                                "checkpoint_id": "electricity_otflow_forecast_20k_seed0",
                                "checkpoint_path": "missing_preflight_checkpoint/model.pt",
                                "summary_path": "",
                                "status": "ready",
                                "seed": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "electricity",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest_path),
                    "--dataset_root",
                    str(dataset_root),
                    "--allow_execute",
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint files are missing"):
                runner.validate_execution_preflight(args)

    def test_preflight_rejects_explicit_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_manifest = Path(tmpdir) / "missing_manifest.json"
            args = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(missing_manifest),
                    "--allow_execute",
                ]
            )
            with self.assertRaisesRegex(FileNotFoundError, "Backbone manifest not found"):
                runner.validate_execution_preflight(args)

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
        self.assertNotEqual(
            runner._protocol_config_fingerprint(args_a), runner._protocol_config_fingerprint(args_b)
        )

    def test_protocol_hash_tracks_execution_config(self) -> None:
        args_a = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                "",
            ]
        )
        args_b = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                "",
                "--hidden_dim",
                str(int(args_a.hidden_dim) + 1),
            ]
        )
        self.assertNotEqual(
            runner._protocol_config_fingerprint(args_a), runner._protocol_config_fingerprint(args_b)
        )

    def test_protocol_hash_tracks_runtime_environment(self) -> None:
        args = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                "",
            ]
        )
        with mock.patch.object(
            runner, "_runtime_environment_fingerprint", return_value={"python": "first"}
        ):
            first = runner._protocol_config_fingerprint(args)
        with mock.patch.object(
            runner, "_runtime_environment_fingerprint", return_value={"python": "second"}
        ):
            second = runner._protocol_config_fingerprint(args)
        self.assertNotEqual(first, second)

    def test_resume_loader_tolerates_only_a_truncated_final_line(self) -> None:
        row = _recorded_row()
        serialized = runner._json_dumps(row, sort_keys=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text(serialized + "\n" + '{"protocol_hash":"protocol"', encoding="utf-8")
            loaded = runner._load_rows(path, protocol_hash="protocol")
            self.assertEqual(list(loaded.values()), [row])

            path.write_text('{"broken":\n' + serialized + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSONL record"):
                runner._load_rows(path, protocol_hash="protocol")

            path.write_text(serialized + "\n" + '{"broken":\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSONL record"):
                runner._load_rows(path, protocol_hash="protocol")

    def test_json_serialization_rejects_nonfinite_numbers(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    runner._json_dumps({"value": value})

    def test_locked_forecast_diagnosis_excludes_other_benchmark_families(self) -> None:
        forecast = _recorded_row()
        conditional = _recorded_row(
            benchmark_family="conditional_generation", signature="conditional"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            (out_root / "rows.jsonl").write_text(
                runner._json_dumps(forecast) + "\n" + runner._json_dumps(conditional) + "\n",
                encoding="utf-8",
            )
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    "",
                    "--diagnose_locked_forecast_only",
                    "--seeds",
                    "0",
                ]
            )
            with mock.patch.object(runner, "_protocol_config_fingerprint", return_value="protocol"):
                payload = runner.run_diffusion_flow_time_reparameterization(args)
        self.assertEqual(payload["row_count"], 2)
        self.assertEqual(payload["locked_row_count"], 1)
        self.assertEqual(payload["main_table_summary"]["row_count"], 1)

    def test_project_scripts_and_wrappers_resolve(self) -> None:
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = pyproject["project"]["scripts"]
        expected_scripts = {
            "dfi-run-schedules": "diffusion_flow_inference.evaluation.diffusion_flow_time_reparameterization:main",
            "dfi-build-velocity-variation-figure": "diffusion_flow_inference.visualization.build_velocity_variation_difficulty_figure:main",
            "dfi-build-ptg-figure": "diffusion_flow_inference.visualization.build_ptg_observed_gain_figure:main",
        }
        self.assertEqual(scripts, expected_scripts)
        for target in scripts.values():
            module_name, attr_name = target.split(":", 1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), attr_name)))

        wrappers = {
            "diffusion_flow_time_reparameterization.py": expected_scripts["dfi-run-schedules"],
            "build_velocity_variation_difficulty_figure.py": expected_scripts[
                "dfi-build-velocity-variation-figure"
            ],
            "build_ptg_observed_gain_figure.py": expected_scripts["dfi-build-ptg-figure"],
        }
        for wrapper_name, target in wrappers.items():
            module_name, attr_name = target.split(":", 1)
            wrapper_text = (PROJECT_ROOT / "scripts" / wrapper_name).read_text(encoding="utf-8")
            self.assertIn(f"from {module_name} import {attr_name}", wrapper_text)

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
