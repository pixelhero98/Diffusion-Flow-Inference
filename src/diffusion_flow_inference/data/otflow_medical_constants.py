from __future__ import annotations

from pathlib import Path

SLEEP_EDF_DATASET_KEY = "sleep_edf"


def sleep_edf_metadata_path_for_npz(npz_path: str | Path) -> Path:
    return Path(npz_path).expanduser().resolve().with_suffix(".json")


__all__ = [
    "SLEEP_EDF_DATASET_KEY",
    "sleep_edf_metadata_path_for_npz",
]
