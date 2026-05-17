from __future__ import annotations

import csv
import io
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import build_adaptive_solver_matched_nfe_study as study

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _write_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _synthetic_matched_plot_inputs():
    rows = []
    brackets = []
    for dataset_idx, dataset in enumerate(study.DATASETS):
        for target_nfe in study.TARGET_NFES:
            score = 0.8 + 0.01 * dataset_idx + 0.001 * target_nfe
            row = {
                "dataset": dataset,
                "target_nfe": target_nfe,
                "best_fixed_solver": "euler",
                "best_fixed_schedule": "gits",
                "fixed_avg_relative_score": score,
                "fixed_realized_nfe": target_nfe,
                "rk45_matched": True,
                "rk45_matched_used_nfe": 20.0 + target_nfe,
                "rk45_matched_used_nfe_std": 1.0,
                "rk45_matched_rtol": 0.1,
                "rk45_matched_atol": 1e-4,
                "rk45_matched_avg_relative_score": score - 0.01,
                "rk45_matched_relative_crps": score,
                "rk45_matched_relative_mase": score - 0.02,
                "dopri5_matched": True,
                "dopri5_matched_used_nfe": 24.0 + target_nfe,
                "dopri5_matched_used_nfe_std": 1.0,
                "dopri5_matched_rtol": 0.1,
                "dopri5_matched_atol": 1e-4,
                "dopri5_matched_avg_relative_score": score - 0.02,
                "dopri5_matched_relative_crps": score,
                "dopri5_matched_relative_mase": score - 0.04,
            }
            if dataset == "electricity" and target_nfe == 10:
                row["rk45_matched"] = False
                row["rk45_matched_used_nfe"] = ""
                row["rk45_matched_avg_relative_score"] = ""
                row["rk45_matched_rtol"] = ""
                brackets.append(
                    {
                        "dataset": dataset,
                        "target_nfe": target_nfe,
                        "solver_key": "rk45_adaptive",
                        "status": "unmatched",
                        "nearest_used_nfe": 99.0,
                        "nearest_score": score + 0.2,
                        "nearest_rtol": 0.001,
                    }
                )
            rows.append(row)
    diagnostics = {"brackets": brackets}
    return rows, diagnostics


def _synthetic_lob_summary_inputs():
    rows = []
    brackets = []
    for dataset_idx, dataset in enumerate(study.LOB_DATASETS):
        for target_nfe in study.TARGET_NFES:
            score = 0.78 + 0.02 * dataset_idx + 0.001 * target_nfe
            row = {
                "dataset": dataset,
                "target_nfe": target_nfe,
                "best_fixed_solver": "dpmpp2m",
                "best_fixed_schedule": "ots",
                "fixed_avg_relative_score": score,
                "fixed_realized_nfe": target_nfe,
                "rk45_matched": True,
                "rk45_matched_used_nfe": 18.0 + target_nfe,
                "rk45_matched_used_nfe_std": 1.0,
                "rk45_matched_rtol": 0.1,
                "rk45_matched_atol": 1e-4,
                "rk45_matched_avg_relative_score": score - 0.01,
                "rk45_matched_relative_cw1": score,
                "rk45_matched_relative_tstr_f1": score - 0.02,
                "dopri5_matched": True,
                "dopri5_matched_used_nfe": 22.0 + target_nfe,
                "dopri5_matched_used_nfe_std": 1.0,
                "dopri5_matched_rtol": 0.1,
                "dopri5_matched_atol": 1e-4,
                "dopri5_matched_avg_relative_score": score - 0.02,
                "dopri5_matched_relative_cw1": score,
                "dopri5_matched_relative_tstr_f1": score - 0.04,
            }
            if dataset == "cryptos" and target_nfe == 10:
                row["dopri5_matched"] = False
                row["dopri5_matched_used_nfe"] = ""
                row["dopri5_matched_avg_relative_score"] = ""
                row["dopri5_matched_rtol"] = ""
                brackets.append(
                    {
                        "dataset": dataset,
                        "target_nfe": target_nfe,
                        "solver_key": "dopri5_adaptive",
                        "status": "unmatched",
                        "nearest_used_nfe": 88.0,
                        "nearest_score": score + 0.2,
                        "nearest_rtol": 0.001,
                    }
                )
            rows.append(row)
    diagnostics = {"brackets": brackets}
    return rows, diagnostics


class AdaptiveSolverMatchedNfeStudyTests(unittest.TestCase):
    def test_expected_dry_run_count(self) -> None:
        self.assertEqual(study.expected_adaptive_row_count(), 400)
        self.assertEqual(
            study.expected_adaptive_row_count(datasets=("electricity",), solvers=("rk45_adaptive",), rtols=(0.1, 0.01), seeds=(0,)),
            2,
        )

    def test_atol_mapping(self) -> None:
        self.assertAlmostEqual(study.adaptive_atol_for_rtol(0.3), 0.0003)
        self.assertAlmostEqual(study.adaptive_atol_for_rtol(0.0001), 1e-6)

    def test_extract_fixed_targets_excludes_late_power_3(self) -> None:
        raw_rows = []
        rel_rows = []
        for dataset in study.DATASETS:
            for nfe in study.TARGET_NFES:
                for solver in ("euler", "heun"):
                    raw_rows.append(
                        {
                            "dataset": dataset,
                            "split_phase": "locked_test",
                            "backbone_name": "otflow",
                            "train_budget_label": "20k",
                            "train_steps": "20000",
                            "checkpoint_id": "ckpt",
                            "target_nfe": str(nfe),
                            "solver_key": solver,
                            "solver_name": solver,
                            "schedule_key": "uniform",
                            "schedule_name": "uniform",
                            "n_seeds": "5",
                            "seed_values": "[0,1,2,3,4]",
                            "crps_mean": "10.0",
                            "crps_std": "0.0",
                            "mse_mean": "1.0",
                            "mse_std": "0.0",
                            "mase_mean": "20.0",
                            "mase_std": "0.0",
                        }
                    )
                for schedule, avg in (("ays", 0.9), ("gits", 0.8), ("ots", 0.85), ("late_power_3", 0.1)):
                    raw_rows.append(
                        {
                            **raw_rows[-1],
                            "solver_key": "euler",
                            "solver_name": "euler",
                            "schedule_key": schedule,
                            "schedule_name": schedule,
                            "crps_mean": str(10.0 * avg),
                            "mase_mean": str(20.0 * avg),
                        }
                    )
                    rel_rows.append(
                        {
                            "dataset": dataset,
                            "split_phase": "locked_test",
                            "backbone_name": "otflow",
                            "train_budget_label": "20k",
                            "train_steps": "20000",
                            "checkpoint_id": "ckpt",
                            "target_nfe": str(nfe),
                            "solver_key": "euler",
                            "solver_name": "euler",
                            "schedule_key": schedule,
                            "schedule_name": schedule,
                            "n_seeds": "5",
                            "seed_values": "[0,1,2,3,4]",
                            "relative_crps_vs_uniform_mean": str(avg),
                            "relative_crps_vs_uniform_std": "0.0",
                            "relative_mase_vs_uniform_mean": str(avg),
                            "relative_mase_vs_uniform_std": "0.0",
                        }
                    )
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "twenty.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(study.RAW_STATS_MEMBER, _write_csv(raw_rows))
                zf.writestr(study.RELATIVE_STATS_MEMBER, _write_csv(rel_rows))
            targets = study.extract_fixed_targets(zip_path)
        self.assertEqual(len(targets), len(study.DATASETS) * len(study.TARGET_NFES))
        self.assertEqual({row["fixed_schedule_key"] for row in targets}, {"gits"})
        self.assertTrue(all(row["excluded_schedule_keys"] == "late_power_3" for row in targets))

    def test_lob_average_relative_score_uses_cw1_and_tstr_direction(self) -> None:
        relative_cw1, relative_tstr, average = study.lob_average_relative_score(
            conditional_w1=8.0,
            tstr_macro_f1=0.625,
            uniform_conditional_w1=10.0,
            uniform_tstr_macro_f1=0.5,
        )
        self.assertAlmostEqual(relative_cw1, 0.8)
        self.assertAlmostEqual(relative_tstr, 0.8)
        self.assertAlmostEqual(average, 0.8)
        _, zero_tstr_ratio, zero_average = study.lob_average_relative_score(
            conditional_w1=8.0,
            tstr_macro_f1=0.0,
            uniform_conditional_w1=10.0,
            uniform_tstr_macro_f1=0.0,
        )
        self.assertAlmostEqual(zero_tstr_ratio, 1.0)
        self.assertAlmostEqual(zero_average, 0.9)

    def test_extract_lob_fixed_targets_excludes_non_transferred_schedules(self) -> None:
        rows = []
        for dataset in study.LOB_DATASETS:
            for seed in study.LOB_SEEDS:
                for nfe in study.TARGET_NFES:
                    for solver in ("euler", "dpmpp2m"):
                        uniform = {
                            "row_status": "complete",
                            "benchmark_family": "lob_conditional_generation",
                            "split_phase": "locked_test",
                            "dataset": dataset,
                            "seed": seed,
                            "solver_key": solver,
                            "solver_name": solver,
                            "target_nfe": nfe,
                            "scheduler_key": "uniform",
                            "schedule_name": "uniform",
                            "realized_nfe": nfe,
                            "conditional_w1": 10.0,
                            "tstr_macro_f1": 0.5,
                        }
                        rows.append(uniform)
                        for schedule, ratio in (("ays", 0.9), ("gits", 0.8), ("ots", 0.85), ("late_power_3", 0.1)):
                            rows.append(
                                {
                                    **uniform,
                                    "scheduler_key": schedule,
                                    "schedule_name": schedule,
                                    "conditional_w1": 10.0 * ratio,
                                    "tstr_macro_f1": 0.5 / ratio,
                                }
                            )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lob_rows.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            targets = study.extract_lob_fixed_targets(path)
        self.assertEqual(len(targets), len(study.LOB_DATASETS) * len(study.TARGET_NFES))
        self.assertEqual({row["fixed_schedule_key"] for row in targets}, {"gits"})
        self.assertTrue(all(row["excluded_schedule_keys"] == "late_power_3,uniform" for row in targets))

    def test_matching_chooses_lowest_used_nfe_that_meets_target(self) -> None:
        targets = [
            {
                "dataset": "electricity",
                "target_nfe": 10,
                "fixed_solver_key": "euler",
                "fixed_schedule_key": "gits",
                "fixed_avg_relative_score": 0.8,
                "fixed_realized_nfe": 10,
                "uniform_crps_mean": 10.0,
                "uniform_mase_mean": 20.0,
            }
        ]
        adaptive = [
            {
                "dataset": "electricity",
                "solver_key": "rk45_adaptive",
                "rtol": 0.1,
                "adaptive_atol": 1e-4,
                "crps_mean": 9.0,
                "mase_mean": 18.0,
                "used_nfe_mean_mean": 20.0,
                "used_nfe_mean_std": 1.0,
            },
            {
                "dataset": "electricity",
                "solver_key": "rk45_adaptive",
                "rtol": 0.01,
                "adaptive_atol": 1e-5,
                "crps_mean": 7.8,
                "mase_mean": 15.8,
                "used_nfe_mean_mean": 40.0,
                "used_nfe_mean_std": 2.0,
            },
            {
                "dataset": "electricity",
                "solver_key": "dopri5_adaptive",
                "rtol": 0.1,
                "adaptive_atol": 1e-4,
                "crps_mean": 9.5,
                "mase_mean": 19.0,
                "used_nfe_mean_mean": 18.0,
                "used_nfe_mean_std": 1.0,
            },
        ]
        summary, diagnostics = study.summarize_matches(targets, adaptive)
        self.assertTrue(summary[0]["rk45_matched"])
        self.assertAlmostEqual(summary[0]["rk45_matched_used_nfe"], 40.0)
        self.assertFalse(summary[0]["dopri5_matched"])
        self.assertEqual(diagnostics["matched_cells"], 1)

    def test_lob_matching_chooses_lowest_used_nfe_that_meets_cw1_tstr_target(self) -> None:
        targets = [
            {
                "dataset": "cryptos",
                "target_nfe": 10,
                "fixed_solver_key": "dpmpp2m",
                "fixed_schedule_key": "ots",
                "fixed_avg_relative_score": 0.8,
                "fixed_realized_nfe": 10,
                "fixed_relative_cw1": 0.8,
                "fixed_relative_tstr_f1": 0.8,
                "uniform_conditional_w1_mean": 10.0,
                "uniform_tstr_macro_f1_mean": 0.5,
            }
        ]
        adaptive = [
            {
                "dataset": "cryptos",
                "solver_key": "rk45_adaptive",
                "rtol": 0.1,
                "adaptive_atol": 1e-4,
                "conditional_w1_mean": 9.0,
                "tstr_macro_f1_mean": 0.55,
                "used_nfe_mean_mean": 20.0,
                "used_nfe_mean_std": 1.0,
            },
            {
                "dataset": "cryptos",
                "solver_key": "rk45_adaptive",
                "rtol": 0.01,
                "adaptive_atol": 1e-5,
                "conditional_w1_mean": 7.8,
                "tstr_macro_f1_mean": 0.64,
                "used_nfe_mean_mean": 40.0,
                "used_nfe_mean_std": 2.0,
            },
            {
                "dataset": "cryptos",
                "solver_key": "dopri5_adaptive",
                "rtol": 0.1,
                "adaptive_atol": 1e-4,
                "conditional_w1_mean": 9.5,
                "tstr_macro_f1_mean": 0.52,
                "used_nfe_mean_mean": 18.0,
                "used_nfe_mean_std": 1.0,
            },
        ]
        summary, diagnostics = study.summarize_lob_matches(targets, adaptive)
        self.assertTrue(summary[0]["rk45_matched"])
        self.assertAlmostEqual(summary[0]["rk45_matched_used_nfe"], 40.0)
        self.assertFalse(summary[0]["dopri5_matched"])
        self.assertEqual(diagnostics["matched_cells"], 1)

    def test_plot_points_include_matched_and_censored_unmatched_cells(self) -> None:
        summary_rows, diagnostics = _synthetic_matched_plot_inputs()
        points = study.build_adaptive_matched_nfe_plot_points(summary_rows, diagnostics)
        self.assertEqual(len(points), len(study.DATASETS) * len(study.TARGET_NFES) * len(study.ADAPTIVE_SOLVERS))
        unmatched = [point for point in points if point["point_status"] == "unmatched_censored"]
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched[0]["dataset"], "electricity")
        self.assertEqual(unmatched[0]["target_nfe"], 10)
        self.assertEqual(unmatched[0]["adaptive_solver_key"], "rk45_adaptive")
        self.assertAlmostEqual(float(unmatched[0]["realized_nfe"]), 99.0)
        self.assertGreater(float(unmatched[0]["performance_match_ratio"]), 1.0)
        self.assertGreater(float(unmatched[0]["match_gap_percent"]), 0.0)
        matched = [point for point in points if point["point_status"] == "matched"]
        self.assertTrue(all(float(point["performance_match_ratio"]) < 1.0 for point in matched))
        plot_diagnostics = study.build_adaptive_matched_nfe_plot_diagnostics(points)
        self.assertEqual(plot_diagnostics["point_count"], 30)
        self.assertEqual(plot_diagnostics["matched_points"], 29)
        self.assertEqual(plot_diagnostics["unmatched_censored_points"], 1)
        self.assertEqual(plot_diagnostics["ratio_points"], 30)
        self.assertEqual(plot_diagnostics["over_matched_points"], 29)
        self.assertEqual(plot_diagnostics["under_matched_points"], 1)
        self.assertEqual(plot_diagnostics["y_key"], "performance_match_ratio")

    def test_lob_plot_points_have_expected_eighteen_cells(self) -> None:
        summary_rows, diagnostics = _synthetic_lob_summary_inputs()
        points = study.build_adaptive_matched_nfe_plot_points(summary_rows, diagnostics)
        self.assertEqual(len(points), len(study.LOB_DATASETS) * len(study.TARGET_NFES) * len(study.ADAPTIVE_SOLVERS))
        plot_diagnostics = study.build_adaptive_matched_nfe_plot_diagnostics(points)
        self.assertEqual(plot_diagnostics["point_count"], 18)
        self.assertEqual(plot_diagnostics["ratio_points"], 18)
        self.assertEqual(plot_diagnostics["unmatched_censored_points"], 1)

    def test_plot_adaptive_matched_nfe_figure_writes_png_and_pdf(self) -> None:
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib is not installed in this local Python")
        summary_rows, diagnostics = _synthetic_matched_plot_inputs()
        points = study.build_adaptive_matched_nfe_plot_points(summary_rows, diagnostics)
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "adaptive.png"
            pdf_path = Path(tmpdir) / "adaptive.pdf"
            outputs = study.plot_adaptive_matched_nfe_figure(points, png_path=png_path, pdf_path=pdf_path, dpi=120)
            self.assertEqual(outputs["png"], str(png_path))
            self.assertEqual(outputs["pdf"], str(pdf_path))
            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)
            import matplotlib.image as mpimg

            image = mpimg.imread(png_path)
            self.assertEqual(image.shape[1], 1344)
            self.assertEqual(image.shape[0], 300)

    def test_combined_plot_writes_two_panel_png_and_pdf(self) -> None:
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib is not installed in this local Python")
        forecast_summary, forecast_diagnostics = _synthetic_matched_plot_inputs()
        lob_summary, lob_diagnostics = _synthetic_lob_summary_inputs()
        forecast_points = study.build_adaptive_matched_nfe_plot_points(forecast_summary, forecast_diagnostics)
        lob_points = study.build_adaptive_matched_nfe_plot_points(lob_summary, lob_diagnostics)
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "combined.png"
            pdf_path = Path(tmpdir) / "combined.pdf"
            outputs = study.plot_combined_adaptive_matched_nfe_figure(
                forecast_points,
                lob_points,
                png_path=png_path,
                pdf_path=pdf_path,
                dpi=120,
            )
            self.assertEqual(outputs["png"], str(png_path))
            self.assertEqual(outputs["pdf"], str(pdf_path))
            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            import matplotlib.image as mpimg

            image = mpimg.imread(png_path)
            self.assertEqual(image.shape[1], 1344)
            self.assertEqual(image.shape[0], 570)

    def test_visual_cleanup_constants_and_labels(self) -> None:
        self.assertEqual(study.TARGET_NFE_SIZES, {10: 24.0, 12: 38.0, 16: 56.0})
        self.assertEqual(study.TARGET_NFE_MARKERS, {10: "o", 12: "s", 16: "^"})


if __name__ == "__main__":
    unittest.main()
