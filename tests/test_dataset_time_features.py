from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from diffusion_flow_inference.data.otflow_datasets import (
    L2FeatureMap,
    WindowedParamSequenceDataset,
    build_dataset_splits_from_arrays,
    load_l2_npz,
)
from diffusion_flow_inference.data.otflow_forecast_data import (
    ForecastExampleRef,
    ForecastSeriesRecord,
    MonashForecastWindowDataset,
    _fill_missing_values,
    _regular_time_features,
)
from diffusion_flow_inference.evaluation.otflow_evaluation_support import (
    choose_forecast_example_indices,
    evaluate_forecast_schedule,
    parse_forecast_datasets,
)
from diffusion_flow_inference.models.config import OTFlowConfig


class DatasetTimeFeatureTests(unittest.TestCase):
    def _forecast_record(self, *, time_feature_mode: str) -> ForecastSeriesRecord:
        raw = np.arange(12, dtype=np.float32)
        return ForecastSeriesRecord(
            dataset_key="dummy",
            series_id="series_0",
            raw_values=raw,
            norm_values=raw[:, None],
            time_features=_regular_time_features(12, 1, time_feature_mode=time_feature_mode),
            mean=0.0,
            std=1.0,
            total_length=12,
            train_prefix_end=8,
            val_start=8,
            test_start=10,
        )

    def test_forecast_time_feature_modes_match_context_width(self) -> None:
        refs = [ForecastExampleRef(series_idx=0, target_t=8)]
        expected_extra = {"none": 0, "gap_only": 1, "gap_elapsed": 2}
        for mode, extra_dim in expected_extra.items():
            ds = MonashForecastWindowDataset(
                dataset_key="dummy",
                split_name="val",
                history_len=4,
                horizon=2,
                series_records=[self._forecast_record(time_feature_mode=mode)],
                example_refs=refs,
                time_feature_mode=mode,
            )
            hist, _, _, _ = ds[0]
            self.assertEqual(hist.shape[-1], 1 + extra_dim)

    def test_time_feature_modes_reject_ambiguous_configuration(self) -> None:
        cfg = OTFlowConfig(use_time_features=True, use_time_gaps=True)
        with self.assertRaisesRegex(ValueError, "exactly one mode"):
            _ = cfg.context_dim

    def test_windowed_dataset_includes_last_valid_target(self) -> None:
        params = np.arange(10, dtype=np.float32)[:, None]
        mids = np.arange(10, dtype=np.float32)
        ds_one_step = WindowedParamSequenceDataset(params, mids, history_len=3, future_horizon=0)
        self.assertEqual(ds_one_step.start_indices[-1], 9)
        ds_two_step = WindowedParamSequenceDataset(params, mids, history_len=3, future_horizon=1)
        self.assertEqual(ds_two_step.start_indices[-1], 8)

    def test_l2_feature_map_rejects_invalid_shapes_without_asserts(self) -> None:
        feature_map = L2FeatureMap(levels=2)
        wrong_levels = np.zeros((4, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "ask_p has 3 levels, expected 2"):
            feature_map.encode_sequence(wrong_levels, wrong_levels, wrong_levels, wrong_levels)
        with self.assertRaisesRegex(ValueError, "params must have 8 columns, got 7"):
            feature_map.decode_sequence(np.zeros((4, 7), dtype=np.float32), init_mid=100.0)

    def test_load_l2_npz_rejects_pickled_object_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            numeric_path = root / "numeric.npz"
            np.savez(
                numeric_path,
                params_raw=np.zeros((8, 4), dtype=np.float32),
                mids=np.zeros(8, dtype=np.float32),
            )
            loaded = load_l2_npz(str(numeric_path))
            self.assertEqual(loaded["params_raw"].dtype, np.float32)

            object_path = root / "object.npz"
            np.savez(object_path, payload=np.asarray([{"unsafe": True}], dtype=object))
            with self.assertRaisesRegex(ValueError, "Object arrays cannot be loaded"):
                load_l2_npz(str(object_path))

    def test_partial_explicit_split_bounds_are_honored(self) -> None:
        params = np.arange(120 * 4, dtype=np.float32).reshape(120, 4)
        mids = np.arange(120, dtype=np.float32)

        def build_stats(**kwargs):
            return build_dataset_splits_from_arrays(
                params,
                mids,
                OTFlowConfig(levels=1, history_len=4, standardize=False),
                **kwargs,
            )["stats"]

        train_explicit = build_stats(train_end=60)
        self.assertEqual((train_explicit["train_end"], train_explicit["val_end"]), (60, 96))
        val_explicit = build_stats(val_end=100)
        self.assertEqual((val_explicit["train_end"], val_explicit["val_end"]), (84, 100))

        segment_ends = np.asarray([30, 60, 90, 120], dtype=np.int64)
        segment_train_explicit = build_stats(
            segment_ends=segment_ends,
            train_frac=0.5,
            val_frac=0.25,
            train_end=60,
        )
        self.assertEqual(
            (segment_train_explicit["train_end"], segment_train_explicit["val_end"]),
            (60, 90),
        )
        segment_val_explicit = build_stats(
            segment_ends=segment_ends,
            train_frac=0.5,
            val_frac=0.25,
            val_end=90,
        )
        self.assertEqual(
            (segment_val_explicit["train_end"], segment_val_explicit["val_end"]),
            (60, 90),
        )

    def test_missing_values_are_filled_without_future_leakage(self) -> None:
        observed = _fill_missing_values(np.asarray([1.0, np.nan, 3.0], dtype=np.float32))
        changed_holdout = _fill_missing_values(np.asarray([1.0, np.nan, 30.0], dtype=np.float32))
        np.testing.assert_array_equal(observed[:2], np.asarray([1.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(observed[:2], changed_holdout[:2])
        with self.assertRaisesRegex(ValueError, "begins with a missing value"):
            _fill_missing_values(np.asarray([np.nan, 1.0], dtype=np.float32))

    def test_parse_forecast_datasets_rejects_unknown_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown forecast datasets"):
            parse_forecast_datasets("electricity,not_a_dataset")

    def test_horizon_one_forecast_schedule_uses_target_without_future_tuple(self) -> None:
        cfg = OTFlowConfig(
            device=torch.device("cpu"),
            levels=1,
            token_dim=1,
            history_len=2,
            hidden_dim=8,
            dropout=0.0,
            ctx_heads=1,
            ctx_layers=1,
            rollout_mode="autoregressive",
            future_block_len=1,
            use_amp=False,
        )

        class DummyDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                del idx
                return torch.zeros(2, 1), torch.ones(1), {"target_t": 2}

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        class DummyModel:
            def __init__(self, cfg):
                self.cfg = cfg

            def sample_future(self, hist, steps=None, solver=None):
                del steps, solver
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            runtime_nfe=1,
            time_grid=(0.0, 1.0),
            num_eval_samples=1,
            seed=0,
        )
        self.assertEqual(metrics["eval_examples"], 1)
        self.assertTrue(np.isfinite(metrics["mse"]))

    def test_forecast_schedule_uses_deterministic_example_subset(self) -> None:
        cfg = OTFlowConfig(
            device=torch.device("cpu"),
            levels=1,
            token_dim=1,
            history_len=2,
            hidden_dim=8,
            dropout=0.0,
            ctx_heads=1,
            ctx_layers=1,
            use_amp=False,
        )

        class DummyDataset:
            horizon = 1

            def __len__(self) -> int:
                return 5

            def __getitem__(self, idx: int):
                return torch.zeros(2, 1), torch.tensor([float(idx)]), {"target_t": int(idx)}

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        class DummyModel:
            def __init__(self, cfg):
                self.cfg = cfg

            def sample_future(self, hist, steps=None, solver=None):
                del steps, solver
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        chosen = choose_forecast_example_indices(DummyDataset(), n_examples=2, seed=7)
        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            runtime_nfe=1,
            time_grid=(0.0, 1.0),
            num_eval_samples=1,
            seed=3,
            example_indices=chosen,
        )
        self.assertEqual(metrics["eval_examples"], 2)
        self.assertIn("chosen_examples_hash", metrics)


if __name__ == "__main__":
    unittest.main()
