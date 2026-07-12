from __future__ import annotations

from pathlib import Path

LONG_TERM_HEADERED_ECG_DATASET_KEY = "long_term_headered_ECG_records"
SLEEP_EDF_DATASET_KEY = "sleep_edf"
LONG_TERM_ECG_MANIFEST_FILENAME = "manifest.json"


def long_term_headered_ecg_dataset_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root).resolve() / LONG_TERM_HEADERED_ECG_DATASET_KEY


def long_term_headered_ecg_manifest_path(dataset_root: str | Path) -> Path:
    return long_term_headered_ecg_dataset_dir(dataset_root) / LONG_TERM_ECG_MANIFEST_FILENAME


def sleep_edf_metadata_path_for_npz(npz_path: str | Path) -> Path:
    return Path(npz_path).expanduser().resolve().with_suffix(".json")


__all__ = [
    "LONG_TERM_HEADERED_ECG_DATASET_KEY",
    "LONG_TERM_ECG_MANIFEST_FILENAME",
    "SLEEP_EDF_DATASET_KEY",
    "long_term_headered_ecg_dataset_dir",
    "long_term_headered_ecg_manifest_path",
    "sleep_edf_metadata_path_for_npz",
]
