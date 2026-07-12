from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from diffusion_flow_inference.data.otflow_datasets import (
    build_dataset_splits_from_cryptos,
    build_dataset_splits_from_es_mbp_10,
    build_dataset_splits_from_npz_l2,
)
from diffusion_flow_inference.data.otflow_medical_datasets import (
    SLEEP_EDF_DATASET_KEY,
    build_dataset_splits_from_sleep_edf,
)
from diffusion_flow_inference.models.config import OTFlowConfig


PAPER_BACKBONE_PRESETS: Mapping[str, Mapping[str, object]] = {
    "cryptos": {
        "levels": 10,
        "token_dim": 4,
        "history_len": 256,
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_time_features": True,
        "use_time_gaps": False,
    },
    "es_mbp_10": {
        "levels": 10,
        "token_dim": 4,
        "history_len": 256,
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_time_features": True,
        "use_time_gaps": False,
    },
    SLEEP_EDF_DATASET_KEY: {
        "levels": 1,
        "token_dim": 3,
        "history_len": 12_000,
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_time_features": False,
        "use_time_gaps": False,
        "use_cond_features": True,
        "cond_standardize": False,
        "rollout_mode": "non_ar",
        "future_block_len": 3_000,
    },
}


@dataclass(frozen=True)
class DatasetPlan:
    evaluation_windows: int
    train_fraction: float = 0.7
    validation_fraction: float = 0.1
    test_fraction: float = 0.2
    training_stride: int = 1
    evaluation_stride: int = 1


DATASET_PLANS: Mapping[str, DatasetPlan] = {
    "cryptos": DatasetPlan(evaluation_windows=20),
    "es_mbp_10": DatasetPlan(evaluation_windows=20),
    SLEEP_EDF_DATASET_KEY: DatasetPlan(
        evaluation_windows=12,
        training_stride=3_000,
        evaluation_stride=3_000,
    ),
}


def paper_backbone_preset(dataset_key: str) -> Dict[str, object]:
    key = str(dataset_key).strip()
    if key not in PAPER_BACKBONE_PRESETS:
        raise ValueError(f"No paper backbone preset is defined for dataset={dataset_key!r}.")
    return dict(PAPER_BACKBONE_PRESETS[key])


def build_dataset_splits(arguments, cfg: OTFlowConfig):
    dataset_key = str(arguments.dataset)
    data_path = str(getattr(arguments, "data_path", "") or "").strip() or None
    split_kwargs = {
        "cfg": cfg,
        "stride_train": int(arguments.stride_train),
        "stride_eval": int(arguments.stride_eval),
        "train_frac": float(arguments.train_frac),
        "val_frac": float(arguments.val_frac),
        "test_frac": float(arguments.test_frac),
    }
    if dataset_key == "npz_l2":
        if data_path is None:
            raise ValueError("--data_path is required when --dataset npz_l2.")
        return build_dataset_splits_from_npz_l2(path=data_path, **split_kwargs)
    if dataset_key == "cryptos":
        return build_dataset_splits_from_cryptos(path=data_path, **split_kwargs)
    if dataset_key == "es_mbp_10":
        return build_dataset_splits_from_es_mbp_10(path=data_path, **split_kwargs)
    if dataset_key == SLEEP_EDF_DATASET_KEY:
        return build_dataset_splits_from_sleep_edf(path=data_path, **split_kwargs)
    raise ValueError(f"Unsupported dataset={dataset_key!r}.")


__all__ = [
    "DATASET_PLANS",
    "DatasetPlan",
    "PAPER_BACKBONE_PRESETS",
    "build_dataset_splits",
    "paper_backbone_preset",
]
