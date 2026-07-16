from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import diffusion_flow_inference.data.otflow_medical_datasets as medical_datasets
from diffusion_flow_inference.data.otflow_datasets import build_dataset_splits_from_arrays
from diffusion_flow_inference.data.otflow_medical_datasets import prepare_sleep_edf_dataset
from diffusion_flow_inference.evaluation.backbone_registry import (
    BACKBONE_NAME_OTFLOW,
    CONDITIONAL_GENERATION_FAMILY,
    materialize_backbone_manifest,
)
from diffusion_flow_inference.evaluation.otflow_evaluation_support import (
    load_conditional_generation_checkpoint_splits,
)
from diffusion_flow_inference.models.config import OTFlowConfig
from diffusion_flow_inference.models.otflow_model import OTFlow
from diffusion_flow_inference.models.otflow_train_val import (
    _parse_batch,
    select_eval_window_starts,
    train_loop,
)


def _tiny_cfg(*, cond_dim: int = 0) -> OTFlowConfig:
    return OTFlowConfig(
        device=torch.device("cpu"),
        levels=1,
        token_dim=4,
        history_len=4,
        hidden_dim=16,
        dropout=0.0,
        ctx_heads=4,
        ctx_layers=1,
        fu_net_layers=1,
        fu_net_heads=4,
        rollout_mode="non_ar",
        future_block_len=2,
        use_cond_features=True,
        cond_standardize=False,
        cond_dim=int(cond_dim),
        use_amp=False,
    )


class ConditionalGenerationTests(unittest.TestCase):
    def test_dataset_builder_updates_model_cond_dim_without_shadow_field(self) -> None:
        rng = np.random.default_rng(0)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=0)

        splits = build_dataset_splits_from_arrays(
            params,
            mids,
            cfg,
            cond_raw_full=cond,
            train_frac=0.6,
            val_frac=0.2,
        )

        self.assertGreater(len(splits["train"]), 0)
        self.assertEqual(cfg.model.cond_dim, 5)
        self.assertNotIn("cond_dim", vars(cfg))
        model = OTFlow(cfg)
        self.assertIsNotNone(model.backbone.conditioner.cond_mlp)

    def test_dataset_builder_rejects_condition_width_mismatch(self) -> None:
        rng = np.random.default_rng(1)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=4)

        with self.assertRaisesRegex(ValueError, "model.cond_dim=4"):
            build_dataset_splits_from_arrays(params, mids, cfg, cond_raw_full=cond)

    def test_parse_batch_distinguishes_batched_and_unbatched_future_from_condition(self) -> None:
        hist_b = torch.zeros(2, 4, 3)
        tgt_b = torch.zeros(2, 3)
        fut_b = torch.zeros(2, 5, 3)
        cond_b = torch.zeros(2, 5)
        meta = {"t": 4}
        self.assertIs(_parse_batch((hist_b, tgt_b, fut_b, meta))[2], fut_b)
        self.assertIs(_parse_batch((hist_b, tgt_b, cond_b, meta))[3], cond_b)

        hist = torch.zeros(4, 3)
        tgt = torch.zeros(3)
        fut = torch.zeros(5, 3)
        cond = torch.zeros(5)
        self.assertIs(_parse_batch((hist, tgt, fut, meta))[2], fut)
        self.assertIs(_parse_batch((hist, tgt, cond, meta))[3], cond)

    def test_loader_rejects_conditional_metadata_with_unconditional_checkpoint(self) -> None:
        cfg = _tiny_cfg(cond_dim=0)
        model = OTFlow(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "conditional_generation" / "sleep_edf" / "transformer"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"cfg": cfg.to_dict(), "model_state": model.state_dict()}, artifact_dir / "model.pt"
            )
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_key": "sleep_edf",
                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                        "train_steps": 20000,
                        "history_len": 12000,
                        "future_block_len": 3000,
                        "field_network_type": "transformer",
                        "split_stats": {"cond_dim": 5, "history_len": 12000},
                    }
                ),
                encoding="utf-8",
            )
            args = type("Args", (), {"backbone_manifest": "", "otflow_train_steps": 20000})()
            with self.assertRaisesRegex(RuntimeError, "model.cond_dim=0"):
                load_conditional_generation_checkpoint_splits(
                    cli_args=args,
                    shared_backbone_root=root,
                    dataset="sleep_edf",
                    device=torch.device("cpu"),
                )

    def test_sleep_metadata_remains_valid_after_directory_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "original"
            original_dir.mkdir()
            requested = original_dir / "custom_sleep_edf.npz"
            npz_bytes = b"portable-placeholder"
            requested.write_bytes(npz_bytes)
            requested.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "schema": "sleep_edf_prepared_dataset",
                        "schema_version": 1,
                        "dataset_key": "sleep_edf",
                        "history_len": 12000,
                        "official_horizon": 3000,
                        "prepared_npz_file": requested.name,
                        "prepared_npz_sha256": hashlib.sha256(npz_bytes).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            relocated_dir = root / "relocated"
            original_dir.rename(relocated_dir)
            metadata = prepare_sleep_edf_dataset(relocated_dir / requested.name)
            self.assertEqual(metadata["prepared_npz_file"], requested.name)

    def test_sleep_loader_disables_pickle(self) -> None:
        cfg = OTFlowConfig(
            device=torch.device("cpu"),
            levels=1,
            token_dim=3,
            history_len=medical_datasets.SLEEP_EDF_HISTORY_LEN,
            rollout_mode="non_ar",
            future_block_len=medical_datasets.SLEEP_EDF_HORIZON_LEN,
            use_cond_features=True,
            cond_standardize=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "sleep_edf.npz"
            npz_path.write_bytes(b"placeholder")
            metadata = {
                "sampling_rate_hz": 100.0,
                "channels": [],
                "stage_names": [],
                "epoch_samples": medical_datasets.SLEEP_EDF_EPOCH_SAMPLES,
            }
            with (
                patch.object(
                    medical_datasets, "_load_validated_sleep_edf_metadata", return_value=metadata
                ),
                patch.object(
                    medical_datasets.np, "load", side_effect=RuntimeError("sentinel")
                ) as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "sentinel"):
                    medical_datasets.build_dataset_splits_from_sleep_edf(str(npz_path), cfg)
                load.assert_called_once_with(str(npz_path.resolve()), allow_pickle=False)

    def test_readiness_manifest_marks_conditional_checkpoint_without_conditional_state_invalid(
        self,
    ) -> None:
        cfg = _tiny_cfg(cond_dim=0)
        model = OTFlow(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            artifact_dir = (
                matrix_root
                / "otflow"
                / "conditional_generation"
                / "20k"
                / "sleep_edf"
                / "transformer"
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"cfg": cfg.to_dict(), "model_state": model.state_dict()}, artifact_dir / "model.pt"
            )
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint_id": "sleep_bad",
                        "dataset_key": "sleep_edf",
                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                        "train_steps": 20000,
                        "history_len": 12000,
                        "future_block_len": 3000,
                        "field_network_type": "transformer",
                        "split_stats": {"cond_dim": 5, "history_len": 12000},
                    }
                ),
                encoding="utf-8",
            )

            payload = materialize_backbone_manifest(
                matrix_root=matrix_root,
                otflow_reuse_root=Path(tmpdir) / "reuse",
                imported_backbone_root=Path(tmpdir) / "imported",
                budget_steps=(20000,),
                write_path=Path(tmpdir) / "manifest.json",
            )

        sleep_rows = [
            row
            for row in payload["artifacts"]
            if row["backbone_name"] == BACKBONE_NAME_OTFLOW
            and row["benchmark_family"] == CONDITIONAL_GENERATION_FAMILY
            and row["dataset_key"] == "sleep_edf"
        ]
        self.assertEqual(sleep_rows[0]["status"], "invalid")
        self.assertIn("metadata cond_dim=5", sleep_rows[0]["compatibility_error"])

    def test_sleep_window_selection_is_stage_stratified(self) -> None:
        rng = np.random.default_rng(2)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=5)
        splits = build_dataset_splits_from_arrays(
            params,
            mids,
            cfg,
            cond_raw_full=cond,
            train_frac=0.6,
            val_frac=0.2,
            dataset_kind="sleep_edf",
            dataset_metadata={"stage_names": ["W", "N1", "N2", "N3", "REM"]},
        )

        chosen = select_eval_window_starts(splits["test"], horizon=2, n_windows=3, seed=5)
        stages = {int(np.argmax(splits["test"].cond[int(t0)])) for t0 in chosen.tolist()}
        self.assertEqual(stages, {0, 1, 2, 3, 4})

    def test_unsupported_model_names_are_rejected(self) -> None:
        rng = np.random.default_rng(3)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=0)
        splits = build_dataset_splits_from_arrays(
            params,
            mids,
            cfg,
            cond_raw_full=cond,
            train_frac=0.6,
            val_frac=0.2,
        )

        with self.assertRaisesRegex(ValueError, "Only model_name='otflow' is supported"):
            train_loop(splits["train"], cfg, model_name="cgan", steps=1)


if __name__ == "__main__":
    unittest.main()
