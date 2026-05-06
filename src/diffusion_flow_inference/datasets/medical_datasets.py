from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from diffusion_flow_inference.backbones.settings.config import LOBConfig
from diffusion_flow_inference.datasets.lob_datasets import build_dataset_splits_from_arrays
from diffusion_flow_inference.datasets.medical_constants import (
    DEFAULT_SLEEP_EDF_METADATA_NAME,
    DEFAULT_SLEEP_EDF_NPZ_NAME,
    SLEEP_EDF_DATASET_KEY,
    default_sleep_edf_data_path,
    default_sleep_edf_metadata_path,
)

DEFAULT_MEDICAL_STAGING_ROOT: Path | None = None

SLEEP_EDF_SAMPLING_RATE_HZ = 100.0
SLEEP_EDF_EPOCH_SECONDS = 30
SLEEP_EDF_EPOCH_SAMPLES = int(SLEEP_EDF_SAMPLING_RATE_HZ * SLEEP_EDF_EPOCH_SECONDS)
SLEEP_EDF_HISTORY_EPOCHS = 4
SLEEP_EDF_HISTORY_LEN = SLEEP_EDF_HISTORY_EPOCHS * SLEEP_EDF_EPOCH_SAMPLES
SLEEP_EDF_HORIZON_LEN = SLEEP_EDF_EPOCH_SAMPLES
SLEEP_EDF_COMMON_CHANNELS: Tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")
SLEEP_EDF_STAGE_NAMES: Tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
SLEEP_EDF_STAGE_MAP: Mapping[str, Optional[str]] = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": None,
    "Sleep stage ?": None,
}


def medical_staging_root() -> Path:
    raw = str(os.environ.get("OTFLOW_MEDICAL_STAGING_ROOT", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    raise RuntimeError("Set OTFLOW_MEDICAL_STAGING_ROOT to prepare raw medical datasets.")


def sleep_edf_source_dir() -> Path:
    return medical_staging_root() / "extracted" / "sleep_edf"

def _sleep_pairing_key(path: Path) -> str:
    stem = str(path.stem).split("-")[0]
    return stem[:7]


def _canonical_sleep_label(raw_label: str) -> Optional[str]:
    return SLEEP_EDF_STAGE_MAP.get(str(raw_label).strip(), None)


def _read_sleep_annotations(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from pyedflib import EdfReader

    reader = EdfReader(str(path))
    try:
        onset, duration, labels = reader.readAnnotations()
    finally:
        reader.close()
    label_list = [str(label) for label in labels]
    return np.asarray(onset, dtype=np.float64), np.asarray(duration, dtype=np.float64), label_list


def _build_sleep_epoch_labels(total_epochs: int, hyp_path: Path) -> np.ndarray:
    epoch_labels = np.full(int(total_epochs), -1, dtype=np.int64)
    onset, duration, labels = _read_sleep_annotations(hyp_path)
    for start_s, duration_s, raw_label in zip(onset.tolist(), duration.tolist(), labels):
        canonical = _canonical_sleep_label(raw_label)
        start_epoch = int(round(float(start_s) / float(SLEEP_EDF_EPOCH_SECONDS)))
        epoch_count = int(round(float(duration_s) / float(SLEEP_EDF_EPOCH_SECONDS)))
        if epoch_count <= 0:
            continue
        stop_epoch = min(int(total_epochs), int(start_epoch) + int(epoch_count))
        if canonical is None:
            continue
        label_idx = int(SLEEP_EDF_STAGE_NAMES.index(str(canonical)))
        epoch_labels[int(start_epoch) : int(stop_epoch)] = int(label_idx)
    return epoch_labels


def prepare_sleep_edf_dataset(
    out_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    npz_path = Path(out_path or default_sleep_edf_data_path()).resolve()
    metadata_path = Path(default_sleep_edf_metadata_path()).resolve()
    if npz_path.exists() and metadata_path.exists() and not bool(force):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    source_dir = sleep_edf_source_dir()
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Missing sleep_edf source directory: {source_dir}. "
            "Set OTFLOW_MEDICAL_STAGING_ROOT or restore the audited staging area."
        )
    try:
        from pyedflib import EdfReader
    except ImportError as exc:
        raise ImportError("pyedflib is required for sleep_edf support.") from exc

    psg_paths = sorted(source_dir.glob("*-PSG.edf"))
    hyp_paths = sorted(source_dir.glob("*-Hypnogram.edf"))
    hyp_by_key = {
        _sleep_pairing_key(path): path
        for path in hyp_paths
        if path.exists() and path.stat().st_size > 0
    }

    params_parts: List[np.ndarray] = []
    cond_parts: List[np.ndarray] = []
    mids_parts: List[np.ndarray] = []
    valid_start_parts: List[np.ndarray] = []
    segment_ends: List[int] = []
    matched_pairs: List[Dict[str, Any]] = []
    stage_counts = {name: 0 for name in SLEEP_EDF_STAGE_NAMES}
    running_total = 0

    for psg_path in psg_paths:
        key = _sleep_pairing_key(psg_path)
        hyp_path = hyp_by_key.get(key)
        if hyp_path is None:
            continue

        reader = EdfReader(str(psg_path))
        try:
            labels = [str(label) for label in reader.getSignalLabels()]
            freqs = np.asarray(reader.getSampleFrequencies(), dtype=np.float64)
            channel_indices = []
            for channel_name in SLEEP_EDF_COMMON_CHANNELS:
                if channel_name not in labels:
                    channel_indices = []
                    break
                idx = labels.index(channel_name)
                if abs(float(freqs[idx]) - float(SLEEP_EDF_SAMPLING_RATE_HZ)) > 1e-6:
                    channel_indices = []
                    break
                channel_indices.append(int(idx))
            if len(channel_indices) != len(SLEEP_EDF_COMMON_CHANNELS):
                continue

            channel_arrays = [
                np.asarray(reader.readSignal(int(idx)), dtype=np.float32)
                for idx in channel_indices
            ]
            min_samples = min(int(arr.shape[0]) for arr in channel_arrays)
        finally:
            reader.close()

        usable_samples = int(min_samples // int(SLEEP_EDF_EPOCH_SAMPLES)) * int(SLEEP_EDF_EPOCH_SAMPLES)
        if usable_samples < int(SLEEP_EDF_HISTORY_LEN + SLEEP_EDF_HORIZON_LEN):
            continue

        signal = np.stack([arr[:usable_samples] for arr in channel_arrays], axis=1).astype(np.float32, copy=False)
        total_epochs = int(usable_samples // int(SLEEP_EDF_EPOCH_SAMPLES))
        epoch_labels = _build_sleep_epoch_labels(total_epochs, hyp_path)
        cond = np.zeros((usable_samples, len(SLEEP_EDF_STAGE_NAMES)), dtype=np.float32)
        for epoch_idx, label_idx in enumerate(epoch_labels.tolist()):
            if int(label_idx) < 0:
                continue
            start = int(epoch_idx) * int(SLEEP_EDF_EPOCH_SAMPLES)
            stop = int(start + int(SLEEP_EDF_EPOCH_SAMPLES))
            cond[start:stop, int(label_idx)] = 1.0
            stage_counts[SLEEP_EDF_STAGE_NAMES[int(label_idx)]] += 1

        valid_start_mask = np.zeros(usable_samples, dtype=bool)
        valid_epochs = epoch_labels >= 0
        for epoch_idx in range(int(SLEEP_EDF_HISTORY_EPOCHS), int(total_epochs)):
            left = int(epoch_idx) - int(SLEEP_EDF_HISTORY_EPOCHS)
            if not bool(np.all(valid_epochs[left : int(epoch_idx) + 1])):
                continue
            start = int(epoch_idx) * int(SLEEP_EDF_EPOCH_SAMPLES)
            valid_start_mask[int(start)] = True

        params_parts.append(signal)
        cond_parts.append(cond)
        mids_parts.append(np.zeros(usable_samples, dtype=np.float32))
        valid_start_parts.append(valid_start_mask)
        running_total += int(usable_samples)
        segment_ends.append(int(running_total))
        matched_pairs.append(
            {
                "recording_key": key,
                "psg_path": str(psg_path),
                "hypnogram_path": str(hyp_path),
                "total_epochs": int(total_epochs),
                "usable_samples": int(usable_samples),
                "valid_target_epochs": int(np.count_nonzero(valid_start_mask)),
            }
        )

    if not params_parts:
        raise ValueError("No usable matched sleep_edf PSG+hypnogram pairs were prepared.")

    params_raw = np.concatenate(params_parts, axis=0).astype(np.float32, copy=False)
    cond_raw = np.concatenate(cond_parts, axis=0).astype(np.float32, copy=False)
    mids = np.concatenate(mids_parts, axis=0).astype(np.float32, copy=False)
    valid_start_mask = np.concatenate(valid_start_parts, axis=0).astype(bool, copy=False)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_path),
        params_raw=params_raw,
        cond_raw=cond_raw,
        mids=mids,
        segment_ends=np.asarray(segment_ends, dtype=np.int64),
        valid_start_mask=valid_start_mask.astype(np.uint8),
    )

    metadata = {
        "dataset_key": SLEEP_EDF_DATASET_KEY,
        "display_name": "Sleep-EDF (3ch, 100Hz)",
        "sampling_rate_hz": float(SLEEP_EDF_SAMPLING_RATE_HZ),
        "epoch_seconds": int(SLEEP_EDF_EPOCH_SECONDS),
        "epoch_samples": int(SLEEP_EDF_EPOCH_SAMPLES),
        "history_len": int(SLEEP_EDF_HISTORY_LEN),
        "official_horizon": int(SLEEP_EDF_HORIZON_LEN),
        "channels": [str(name) for name in SLEEP_EDF_COMMON_CHANNELS],
        "stage_names": [str(name) for name in SLEEP_EDF_STAGE_NAMES],
        "source_dir": str(source_dir),
        "prepared_npz_path": str(npz_path),
        "n_psg_total": int(len(psg_paths)),
        "n_hypnogram_nonzero": int(len(hyp_by_key)),
        "n_recordings_matched": int(len(matched_pairs)),
        "n_segments": int(len(segment_ends)),
        "n_samples_total": int(params_raw.shape[0]),
        "n_valid_target_starts": int(np.count_nonzero(valid_start_mask)),
        "stage_epoch_counts": {key: int(value) for key, value in stage_counts.items()},
        "matched_pairs": matched_pairs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_dataset_splits_from_sleep_edf(
    path: str,
    cfg: LOBConfig,
    *,
    stride_train: int = SLEEP_EDF_EPOCH_SAMPLES,
    stride_eval: int = SLEEP_EDF_EPOCH_SAMPLES,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    resolved_path = Path(path or default_sleep_edf_data_path()).resolve()
    if not resolved_path.exists():
        prepare_sleep_edf_dataset(resolved_path)
    metadata = prepare_sleep_edf_dataset(resolved_path)
    data = np.load(str(resolved_path), allow_pickle=True)
    params_raw = np.asarray(data["params_raw"], dtype=np.float32)
    cond_raw = np.asarray(data["cond_raw"], dtype=np.float32)
    mids = np.asarray(data["mids"], dtype=np.float32)
    segment_ends = np.asarray(data["segment_ends"], dtype=np.int64)
    valid_start_mask = np.asarray(data["valid_start_mask"], dtype=np.uint8).astype(bool)
    return build_dataset_splits_from_arrays(
        params_raw=params_raw,
        mids=mids,
        cfg=cfg,
        stride_train=int(stride_train),
        stride_eval=int(stride_eval),
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
        segment_ends=segment_ends,
        cond_raw_full=cond_raw,
        valid_start_mask=valid_start_mask,
        dataset_kind=SLEEP_EDF_DATASET_KEY,
        dataset_metadata={
            "sampling_rate_hz": float(metadata["sampling_rate_hz"]),
            "channel_names": [str(name) for name in metadata["channels"]],
            "stage_names": [str(name) for name in metadata["stage_names"]],
            "epoch_samples": int(metadata["epoch_samples"]),
        },
    )


__all__ = [
    DEFAULT_MEDICAL_STAGING_ROOT,
    SLEEP_EDF_COMMON_CHANNELS,
    SLEEP_EDF_DATASET_KEY,
    SLEEP_EDF_EPOCH_SAMPLES,
    SLEEP_EDF_HISTORY_LEN,
    SLEEP_EDF_HORIZON_LEN,
    SLEEP_EDF_STAGE_NAMES,
    build_dataset_splits_from_sleep_edf,
    default_sleep_edf_data_path,
    default_sleep_edf_metadata_path,
    medical_staging_root,
    prepare_sleep_edf_dataset,
    sleep_edf_source_dir,
]
