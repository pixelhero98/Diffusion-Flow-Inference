from __future__ import annotations

import unittest

import numpy as np
import torch

from diffusion_flow_inference.backbones.settings.config import Config
from diffusion_flow_inference.backbones.training.train_val import (
    _future_time_context_seq,
    generate_continuation,
)
from diffusion_flow_inference.datasets.lob_datasets import _build_windowed_dataset, fit_standardizer
from diffusion_flow_inference.diagnostics.adaptive_deterministic_refinement_followup import (
    _append_rollout_context_features,
)


class _DummyCfg:
    snapshot_dim = 2
    prediction_horizon = 1
    adaptive_context = False


class _DummyModel(torch.nn.Module):
    cfg = _DummyCfg()


class _FutureFeatureDataset:
    def future_time_features(self, t0: int, horizon: int):
        del t0
        return torch.arange(int(horizon) * 2, dtype=torch.float32).reshape(int(horizon), 2)

    def future_time_gap_features(self, t0: int, horizon: int):
        del t0, horizon
        raise AssertionError("gap-only features should not be used when full future time features exist")


class ContextFeedingTests(unittest.TestCase):
    def test_generate_continuation_requires_future_context_for_extra_channels(self):
        hist = torch.zeros(1, 4, 3)

        with self.assertRaisesRegex(ValueError, "future_context_seq is required"):
            generate_continuation(_DummyModel(), hist, None, steps=2, nfe=1)

    def test_future_time_context_prefers_full_time_features(self):
        features = _future_time_context_seq(_FutureFeatureDataset(), 10, 3, expected_dim=2)

        self.assertEqual(tuple(features.shape), (3, 2))
        self.assertTrue(torch.equal(features[:, 1], torch.tensor([1.0, 3.0, 5.0])))

    def test_append_rollout_context_features_rejects_missing_context(self):
        block = torch.zeros(1, 2, 2)
        x_hist = torch.zeros(1, 4, 4)

        with self.assertRaisesRegex(ValueError, "future_context_seq is required"):
            _append_rollout_context_features(
                block,
                x_hist=x_hist,
                future_context_seq=None,
                cursor=0,
                take=2,
            )

    def test_append_rollout_context_features_rejects_mismatched_context(self):
        block = torch.zeros(1, 2, 2)
        x_hist = torch.zeros(1, 4, 4)

        with self.assertRaisesRegex(ValueError, "expected"):
            _append_rollout_context_features(
                block,
                x_hist=x_hist,
                future_context_seq=torch.zeros(1, 2, 1),
                cursor=0,
                take=2,
            )

    def test_single_windowed_dataset_uses_prefix_normalization_by_default(self):
        cfg = Config(levels=1, token_dim=2, history_len=4, standardize=True)
        params = np.stack(
            [np.arange(20, dtype=np.float32), np.arange(20, dtype=np.float32) ** 2],
            axis=1,
        )
        mids = np.arange(20, dtype=np.float32)

        ds = _build_windowed_dataset(params, mids, cfg, stride=1)
        expected_mu, expected_sig = fit_standardizer(params[:14])
        full_mu, _ = fit_standardizer(params)

        self.assertTrue(np.allclose(ds.params_mean, expected_mu))
        self.assertTrue(np.allclose(ds.params_std, expected_sig))
        self.assertFalse(np.allclose(ds.params_mean, full_mu))


if __name__ == "__main__":
    unittest.main()
