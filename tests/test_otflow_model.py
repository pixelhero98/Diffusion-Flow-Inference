from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from diffusion_flow_inference.evaluation.otflow_evaluation_support import load_checkpoint_model
from diffusion_flow_inference.evaluation.otflow_sampling_support import temporary_sample_config
from diffusion_flow_inference.models.config import OTFlowConfig
from diffusion_flow_inference.models.modules import ResMLP, RoPEAttention
from diffusion_flow_inference.models.otflow_model import OTFlow, _solve_linear_assignment
from diffusion_flow_inference.models.otflow_train_val import (
    _binary_auc,
    _validity_metrics,
    generate_continuation,
)


class OTFlowModelTest(unittest.TestCase):
    def _cfg(
        self,
        *,
        use_minibatch_ot: bool = True,
        rollout_mode: str = "autoregressive",
        cond_dim: int = 0,
        use_time_gaps: bool = False,
    ) -> OTFlowConfig:
        future_block_len = 2 if rollout_mode == "non_ar" else 1
        return OTFlowConfig(
            device=torch.device("cpu"),
            levels=2,
            history_len=4,
            hidden_dim=16,
            dropout=0.0,
            ctx_heads=4,
            ctx_layers=1,
            fu_net_layers=1,
            fu_net_heads=4,
            rollout_mode=rollout_mode,
            future_block_len=future_block_len,
            cond_dim=cond_dim,
            use_time_gaps=use_time_gaps,
            use_minibatch_ot=use_minibatch_ot,
            use_amp=False,
        )

    def test_core_loss_logs_only_velocity_regression_terms(self) -> None:
        torch.manual_seed(0)
        cfg = self._cfg(use_minibatch_ot=True)
        model = OTFlow(cfg)
        hist = torch.randn(3, cfg.history_len, cfg.context_dim)
        target = torch.randn(3, cfg.snapshot_dim)

        loss, logs = model.loss(target, hist)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(logs), {"mean", "ot_cost", "ot_used", "loss"})
        self.assertEqual(logs["ot_used"], 1.0)

    def test_minibatch_ot_can_be_disabled(self) -> None:
        torch.manual_seed(1)
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.randn(3, cfg.history_len, cfg.context_dim)
        target = torch.randn(3, cfg.snapshot_dim)

        _, logs = model.loss(target, hist)

        self.assertEqual(logs["ot_used"], 0.0)
        self.assertEqual(logs["ot_cost"], 0.0)

    def test_assignment_rejects_nonfinite_costs(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                cost = torch.tensor([[0.0, value], [1.0, 0.0]])
                with self.assertRaisesRegex(ValueError, "must all be finite"):
                    _solve_linear_assignment(cost)

    def test_non_ar_future_block_loss_runs_and_validates_shape(self) -> None:
        torch.manual_seed(2)
        cfg = self._cfg(use_minibatch_ot=True, rollout_mode="non_ar")
        model = OTFlow(cfg)
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)
        target = torch.randn(2, cfg.snapshot_dim)
        future = torch.randn(2, 1, cfg.snapshot_dim)

        loss, logs = model.loss(target, hist, fut=future)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(logs), {"mean", "ot_cost", "ot_used", "loss"})
        with self.assertRaisesRegex(ValueError, "Future trajectories must have shape"):
            model.loss(target, hist, fut=torch.randn(2, 1, cfg.snapshot_dim + 1))

    def test_context_and_condition_are_encoded_once_per_solve(self) -> None:
        torch.manual_seed(3)
        cfg = self._cfg(use_minibatch_ot=False, cond_dim=3)
        model = OTFlow(cfg).eval()
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)
        cond = torch.randn(2, cfg.model.cond_dim)
        cond_mlp = model.backbone.conditioner.cond_mlp
        assert cond_mlp is not None

        with (
            patch.object(
                model.backbone.context_encoder,
                "forward",
                wraps=model.backbone.context_encoder.forward,
            ) as encode_context,
            patch.object(cond_mlp, "forward", wraps=cond_mlp.forward) as encode_condition,
        ):
            sample = model.sample(hist, cond=cond, steps=3, solver="heun")

        self.assertEqual(sample.shape, (2, cfg.snapshot_dim))
        self.assertEqual(encode_context.call_count, 1)
        self.assertEqual(encode_condition.call_count, 1)

    def test_conditional_model_requires_conditioning(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False, cond_dim=3)
        model = OTFlow(cfg)
        hist = torch.randn(1, cfg.history_len, cfg.context_dim)

        with self.assertRaisesRegex(ValueError, "requires a conditioning tensor"):
            model.sample(hist, steps=1)

    def test_sampling_preserves_dtype_and_skips_trace_diagnostics(self) -> None:
        torch.manual_seed(4)
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg).double().eval()
        hist = torch.randn(2, cfg.history_len, cfg.context_dim, dtype=torch.float64)

        with patch.object(
            model,
            "_top_of_book_feature_weights",
            side_effect=AssertionError("trace diagnostics should not run"),
        ):
            sample = model.sample(hist, steps=2, solver="euler")

        self.assertEqual(sample.dtype, torch.float64)

    def test_temporary_sample_config_restores_both_owners_after_error(self) -> None:
        model_cfg = self._cfg(use_minibatch_ot=False)
        external_cfg = self._cfg(use_minibatch_ot=False)
        model_cfg.sample.steps = 2
        external_cfg.sample.steps = 5
        model = OTFlow(model_cfg)
        original_model_sample = model_cfg.sample
        original_external_sample = external_cfg.sample
        original_train_steps = external_cfg.train.steps

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with temporary_sample_config(model, external_cfg, steps=7, solver="heun"):
                self.assertEqual(model.cfg.sample.steps, 7)
                self.assertEqual(external_cfg.sample.steps, 7)
                self.assertEqual(model.cfg.sample.solver, "heun")
                self.assertEqual(external_cfg.train.steps, original_train_steps)
                self.assertIsNot(model.cfg.sample, original_model_sample)
                self.assertIsNot(external_cfg.sample, original_external_sample)
                raise RuntimeError("sentinel")

        self.assertEqual(model.cfg.sample.steps, 2)
        self.assertEqual(external_cfg.sample.steps, 5)
        self.assertEqual(external_cfg.train.steps, original_train_steps)

    def test_cfg_scale_is_not_part_of_the_sampling_api(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.randn(1, cfg.history_len, cfg.context_dim)

        self.assertNotIn("cfg_scale", inspect.signature(model.sample).parameters)
        self.assertFalse(hasattr(cfg.sample, "cfg_scale"))
        with self.assertRaises(TypeError):
            model.sample(hist, cfg_scale=1.0)  # type: ignore[call-arg]
        with self.assertRaisesRegex(TypeError, "Unknown sample config fields"):
            with temporary_sample_config(model, cfg, cfg_scale=1.0):
                pass

    def test_scalar_resmlp_output_depends_on_input(self) -> None:
        torch.manual_seed(5)
        model = ResMLP(in_dim=2, hidden_dim=8, out_dim=1, n_blocks=1)
        inputs = torch.tensor([[-1.0, 0.0], [0.5, 1.0], [2.0, -1.0]])

        output = model(inputs)

        self.assertEqual(output.shape, (3, 1))
        self.assertGreater(float(output.detach().std()), 0.0)

    def test_binary_auc_uses_average_ranks_for_ties(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        tied_scores = np.ones(4, dtype=np.float64)

        self.assertAlmostEqual(_binary_auc(labels, tied_scores), 0.5)

    def test_nonfinite_rows_are_invalid(self) -> None:
        ask_price = np.asarray([[101.0, 102.0], [101.0, np.nan]])
        bid_price = np.asarray([[100.0, 99.0], [100.0, 99.0]])
        ask_volume = np.ones((2, 2), dtype=np.float64)
        bid_volume = np.ones((2, 2), dtype=np.float64)

        metrics = _validity_metrics(ask_price, ask_volume, bid_price, bid_volume)

        self.assertEqual(metrics["valid_rate"], 0.5)
        self.assertEqual(metrics["nonfinite_rate"], 0.5)

    def test_config_rejects_invalid_rollout_and_averaging_settings(self) -> None:
        invalid_cases = []

        invalid_rollout = self._cfg()
        invalid_rollout.model.rollout_mode = "unsupported"
        invalid_cases.append((invalid_rollout, "model.rollout_mode"))

        invalid_ar_block = self._cfg()
        invalid_ar_block.model.future_block_len = 2
        invalid_cases.append((invalid_ar_block, "future_block_len=1"))

        invalid_non_ar_block = self._cfg(rollout_mode="non_ar")
        invalid_non_ar_block.model.future_block_len = 1
        invalid_cases.append((invalid_non_ar_block, "future_block_len>1"))

        invalid_schedule = self._cfg()
        invalid_schedule.train.lr_schedule = "mystery"
        invalid_cases.append((invalid_schedule, "train.lr_schedule"))

        conflicting_averages = self._cfg()
        conflicting_averages.train.use_swa = True
        invalid_cases.append((conflicting_averages, "EMA and SWA"))

        for cfg, error in invalid_cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    OTFlow(cfg)

    def test_default_config_uses_one_averaging_strategy(self) -> None:
        cfg = OTFlowConfig()
        self.assertGreater(cfg.train.ema_decay, 0.0)
        self.assertFalse(cfg.train.use_swa)

    def test_rotary_attention_requires_even_head_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "even per-head dimension"):
            RoPEAttention(hidden_dim=12, n_heads=4)

    def test_solver_names_are_limited_to_active_implementations(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)

        for solver_name in ("euler", "heun", "midpoint_rk2", "dpmpp2m"):
            with self.subTest(solver_name=solver_name):
                self.assertEqual(model._resolve_solver_name(solver_name), solver_name)
        for unsupported_solver in (
            "dopri5",
            "dopri5_adaptive",
            "rk45",
            "rk45_adaptive",
            "dpm++2m",
            "euler_adaptive",
        ):
            with self.subTest(unsupported_solver=unsupported_solver):
                with self.assertRaisesRegex(ValueError, "Unknown sample solver"):
                    model._resolve_solver_name(unsupported_solver)

    def test_fixed_solver_trace_omits_removed_adaptive_fields(self) -> None:
        torch.manual_seed(6)
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)

        _, trace = model.sample_trace(hist, steps=3, solver="dpmpp2m")

        removed_fields = {
            "accepted_steps",
            "rejected_steps",
            "trial_accepted",
            "gamma",
            "trigger_strength",
            "noise_norm",
            "normalized_disagreement",
        }
        self.assertFalse(removed_fields & set(trace))
        self.assertEqual(trace["solver"], "dpmpp2m")
        self.assertIn("disagreement", trace)
        self.assertIn("field_evals_by_step", trace)

    def test_sampling_rejects_nonpositive_step_counts(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.randn(1, cfg.history_len, cfg.context_dim)

        for requested_steps in (0, -1):
            with self.subTest(requested_steps=requested_steps):
                with self.assertRaisesRegex(ValueError, "steps must be positive"):
                    model.sample_trace(hist, steps=requested_steps)

        cfg.sample.steps = 0
        with self.assertRaisesRegex(ValueError, "sample.steps must be positive"):
            model.sample_trace(hist)

    def test_time_grid_rejects_nonfinite_nodes(self) -> None:
        for invalid_node in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid_node=invalid_node):
                cfg = self._cfg(use_minibatch_ot=False)
                cfg.sample.time_grid = (0.0, invalid_node, 1.0)
                model = OTFlow(cfg)
                with self.assertRaisesRegex(ValueError, "contain only finite values"):
                    model._resolved_time_grid(2)

    def test_rollout_requires_future_context_features(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False, use_time_gaps=True)
        model = OTFlow(cfg)
        hist = torch.randn(1, cfg.history_len, cfg.context_dim)

        with self.assertRaisesRegex(ValueError, "future_context_seq is required"):
            generate_continuation(model, hist, None, steps=1, nfe=1)

    def test_checkpoint_loader_rejects_removed_fields_and_unexpected_state(self) -> None:
        torch.manual_seed(7)
        cfg = self._cfg(use_minibatch_ot=True)
        model = OTFlow(cfg)
        cfg_dict = cfg.to_dict()
        cfg_dict["sample"].update(
            {
                "cfg_scale": 1.0,
                "adaptive_rtol": 1e-3,
                "adaptive_max_nfe": 64,
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"cfg": cfg_dict, "model_state": model.state_dict()}, ckpt_path)
            with self.assertRaisesRegex(TypeError, "config section 'sample' has invalid keys"):
                load_checkpoint_model(ckpt_path, torch.device("cpu"))

            incomplete_cfg = cfg.to_dict()
            incomplete_cfg["sample"].pop("solver")
            torch.save({"cfg": incomplete_cfg, "model_state": model.state_dict()}, ckpt_path)
            with self.assertRaisesRegex(TypeError, r"missing=\['solver'\]"):
                load_checkpoint_model(ckpt_path, torch.device("cpu"))

            state = dict(model.state_dict())
            state["backbone.conditioner.removed.weight"] = torch.zeros(1)
            torch.save({"cfg": cfg.to_dict(), "model_state": state}, ckpt_path)
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                load_checkpoint_model(ckpt_path, torch.device("cpu"))

    def test_checkpoint_loader_rejects_invalid_checkpoint_structure(self) -> None:
        cfg = self._cfg(use_minibatch_ot=True)
        invalid_payloads = (
            ({"model_state": {}}, "invalid top-level keys"),
            ({"cfg": [], "model_state": {}}, "'cfg' must be a mapping"),
            ({"cfg": cfg.to_dict(), "model_state": []}, "'model_state' must be a mapping"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.pt"
            for payload, message in invalid_payloads:
                with self.subTest(message=message):
                    torch.save(payload, ckpt_path)
                    with self.assertRaisesRegex(TypeError, message):
                        load_checkpoint_model(ckpt_path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
