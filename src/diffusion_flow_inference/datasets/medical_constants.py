from __future__ import annotations

from diffusion_flow_inference.common.paths import project_data_root

SLEEP_EDF_DATASET_KEY = "sleep_edf"
DEFAULT_SLEEP_EDF_NPZ_NAME = "sleep_edf_3ch_100hz_stage_conditioned.npz"
DEFAULT_SLEEP_EDF_METADATA_NAME = "sleep_edf_3ch_100hz_stage_conditioned.json"


def default_sleep_edf_data_path() -> str:
    return str(project_data_root() / DEFAULT_SLEEP_EDF_NPZ_NAME)


def default_sleep_edf_metadata_path() -> str:
    return str(project_data_root() / DEFAULT_SLEEP_EDF_METADATA_NAME)


__all__ = [
    "DEFAULT_SLEEP_EDF_METADATA_NAME",
    "DEFAULT_SLEEP_EDF_NPZ_NAME",
    "SLEEP_EDF_DATASET_KEY",
    "default_sleep_edf_data_path",
    "default_sleep_edf_metadata_path",
]
