from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from diffusion_flow_inference.data.otflow_datasets import build_dataset_splits_from_arrays
from diffusion_flow_inference.data.otflow_medical_constants import (
    SLEEP_EDF_DATASET_KEY,
    sleep_edf_metadata_path_for_npz,
)
from diffusion_flow_inference.data.otflow_paths import sleep_edf_data_path
from diffusion_flow_inference.models.config import OTFlowConfig

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

_SLEEP_EDF_METADATA_SCHEMA = "sleep_edf_prepared_dataset"
_SLEEP_EDF_METADATA_SCHEMA_VERSION = 1


def medical_staging_root() -> Path:
    raw = str(os.environ.get("DFI_MEDICAL_STAGING_ROOT", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    raise RuntimeError("Set DFI_MEDICAL_STAGING_ROOT to prepare raw medical datasets.")


def sleep_edf_source_dir() -> Path:
    return medical_staging_root() / "extracted" / "sleep_edf"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sleep_pairing_key(path: Path) -> str:
    stem = str(path.stem).split("-")[0]
    return stem[:7]


def _normalize_sleep_label(raw_label: str) -> Optional[str]:
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
        normalized_label = _normalize_sleep_label(raw_label)
        start_epoch = int(round(float(start_s) / float(SLEEP_EDF_EPOCH_SECONDS)))
        epoch_count = int(round(float(duration_s) / float(SLEEP_EDF_EPOCH_SECONDS)))
        if epoch_count <= 0:
            continue
        stop_epoch = min(int(total_epochs), int(start_epoch) + int(epoch_count))
        if normalized_label is None:
            continue
        label_idx = int(SLEEP_EDF_STAGE_NAMES.index(str(normalized_label)))
        epoch_labels[int(start_epoch) : int(stop_epoch)] = int(label_idx)
    return epoch_labels


def _load_validated_sleep_edf_metadata(npz_path: str | Path) -> Dict[str, Any]:
    resolved_npz = Path(npz_path).expanduser().resolve()
    metadata_path = sleep_edf_metadata_path_for_npz(resolved_npz)
    missing = [path for path in (resolved_npz, metadata_path) if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Prepared Sleep-EDF artifacts are missing: {missing_text}. "
            "Run prepare_sleep_edf_dataset(...) explicitly before evaluation."
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Sleep-EDF metadata file: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Sleep-EDF metadata must be a JSON object: {metadata_path}")
    if str(metadata.get("schema", "")) != _SLEEP_EDF_METADATA_SCHEMA:
        raise ValueError(f"Unsupported Sleep-EDF metadata schema in {metadata_path}.")
    if int(metadata.get("schema_version", -1)) != _SLEEP_EDF_METADATA_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Sleep-EDF metadata schema version in {metadata_path}.")
    if str(metadata.get("dataset_key", "")) != SLEEP_EDF_DATASET_KEY:
        raise ValueError(f"Sleep-EDF metadata has the wrong dataset_key: {metadata_path}")

    file_identity = str(metadata.get("prepared_npz_file", "")).strip()
    if (
        not file_identity
        or Path(file_identity).is_absolute()
        or Path(file_identity).name != file_identity
    ):
        raise ValueError(f"Sleep-EDF metadata must identify the NPZ by filename: {metadata_path}")
    if file_identity != resolved_npz.name:
        raise ValueError(
            f"Sleep-EDF metadata identifies {file_identity!r}, but the requested file is {resolved_npz.name!r}."
        )
    if int(metadata.get("history_len", -1)) != int(SLEEP_EDF_HISTORY_LEN):
        raise ValueError(
            "Sleep-EDF metadata history_len does not match the locked 12000-sample task."
        )
    if int(metadata.get("official_horizon", -1)) != int(SLEEP_EDF_HORIZON_LEN):
        raise ValueError(
            "Sleep-EDF metadata official_horizon does not match the locked 3000-sample task."
        )

    expected_sha256 = str(metadata.get("prepared_npz_sha256", "")).strip().lower()
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError(f"Sleep-EDF metadata has an invalid prepared_npz_sha256: {metadata_path}")
    observed_sha256 = _sha256_file(resolved_npz)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Sleep-EDF NPZ checksum does not match its metadata: expected={expected_sha256}, "
            f"observed={observed_sha256}."
        )
    return metadata


def prepare_sleep_edf_dataset(
    out_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    npz_path = Path(out_path or sleep_edf_data_path()).expanduser().resolve()
    metadata_path = sleep_edf_metadata_path_for_npz(npz_path)
    if npz_path.is_file() and metadata_path.is_file() and not bool(force):
        return _load_validated_sleep_edf_metadata(npz_path)

    source_dir = sleep_edf_source_dir()
    if not source_dir.exists():
        raise FileNotFoundError(
            "Missing Sleep-EDF source directory. Set DFI_MEDICAL_STAGING_ROOT to the audited staging area."
        )
    try:
        from pyedflib import EdfReader
    except ImportError as exc:
        raise ImportError("pyedflib is required for Sleep-EDF preparation.") from exc

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
            channel_indices: List[int] = []
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
                np.asarray(reader.readSignal(int(idx)), dtype=np.float32) for idx in channel_indices
            ]
            min_samples = min(int(arr.shape[0]) for arr in channel_arrays)
        finally:
            reader.close()

        usable_samples = int(min_samples // SLEEP_EDF_EPOCH_SAMPLES) * SLEEP_EDF_EPOCH_SAMPLES
        if usable_samples < int(SLEEP_EDF_HISTORY_LEN + SLEEP_EDF_HORIZON_LEN):
            continue

        signal = np.stack([arr[:usable_samples] for arr in channel_arrays], axis=1).astype(
            np.float32, copy=False
        )
        total_epochs = int(usable_samples // SLEEP_EDF_EPOCH_SAMPLES)
        epoch_labels = _build_sleep_epoch_labels(total_epochs, hyp_path)
        cond = np.zeros((usable_samples, len(SLEEP_EDF_STAGE_NAMES)), dtype=np.float32)
        for epoch_idx, label_idx in enumerate(epoch_labels.tolist()):
            if int(label_idx) < 0:
                continue
            start = int(epoch_idx) * SLEEP_EDF_EPOCH_SAMPLES
            stop = int(start + SLEEP_EDF_EPOCH_SAMPLES)
            cond[start:stop, int(label_idx)] = 1.0
            stage_counts[SLEEP_EDF_STAGE_NAMES[int(label_idx)]] += 1

        valid_start_mask = np.zeros(usable_samples, dtype=bool)
        valid_epochs = epoch_labels >= 0
        for epoch_idx in range(SLEEP_EDF_HISTORY_EPOCHS, total_epochs):
            left = int(epoch_idx) - SLEEP_EDF_HISTORY_EPOCHS
            if not bool(np.all(valid_epochs[left : int(epoch_idx) + 1])):
                continue
            valid_start_mask[int(epoch_idx) * SLEEP_EDF_EPOCH_SAMPLES] = True

        params_parts.append(signal)
        cond_parts.append(cond)
        mids_parts.append(np.zeros(usable_samples, dtype=np.float32))
        valid_start_parts.append(valid_start_mask)
        running_total += usable_samples
        segment_ends.append(running_total)
        matched_pairs.append(
            {
                "recording_key": key,
                "psg_file": psg_path.name,
                "hypnogram_file": hyp_path.name,
                "total_epochs": total_epochs,
                "usable_samples": usable_samples,
                "valid_target_epochs": int(np.count_nonzero(valid_start_mask)),
            }
        )

    if not params_parts:
        raise ValueError("No usable matched Sleep-EDF PSG and hypnogram pairs were prepared.")

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
        "schema": _SLEEP_EDF_METADATA_SCHEMA,
        "schema_version": _SLEEP_EDF_METADATA_SCHEMA_VERSION,
        "dataset_key": SLEEP_EDF_DATASET_KEY,
        "display_name": "Sleep-EDF (3ch, 100Hz)",
        "sampling_rate_hz": float(SLEEP_EDF_SAMPLING_RATE_HZ),
        "epoch_seconds": SLEEP_EDF_EPOCH_SECONDS,
        "epoch_samples": SLEEP_EDF_EPOCH_SAMPLES,
        "history_len": SLEEP_EDF_HISTORY_LEN,
        "official_horizon": SLEEP_EDF_HORIZON_LEN,
        "channels": list(SLEEP_EDF_COMMON_CHANNELS),
        "stage_names": list(SLEEP_EDF_STAGE_NAMES),
        "prepared_npz_file": npz_path.name,
        "prepared_npz_sha256": _sha256_file(npz_path),
        "n_psg_total": len(psg_paths),
        "n_hypnogram_nonzero": len(hyp_by_key),
        "n_recordings_matched": len(matched_pairs),
        "n_segments": len(segment_ends),
        "n_samples_total": int(params_raw.shape[0]),
        "n_valid_target_starts": int(np.count_nonzero(valid_start_mask)),
        "stage_epoch_counts": {key: int(value) for key, value in stage_counts.items()},
        "matched_pairs": matched_pairs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _validate_sleep_edf_arrays(
    *,
    params_raw: np.ndarray,
    cond_raw: np.ndarray,
    mids: np.ndarray,
    segment_ends: np.ndarray,
    valid_start_mask: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    if params_raw.ndim != 2 or params_raw.shape[1] != len(SLEEP_EDF_COMMON_CHANNELS):
        raise ValueError(f"Sleep-EDF params_raw has invalid shape={params_raw.shape}.")
    if cond_raw.ndim != 2 or cond_raw.shape[1] != len(SLEEP_EDF_STAGE_NAMES):
        raise ValueError(f"Sleep-EDF cond_raw has invalid shape={cond_raw.shape}.")
    sample_count = int(params_raw.shape[0])
    if mids.ndim != 1 or cond_raw.shape[0] != sample_count or mids.shape[0] != sample_count:
        raise ValueError("Sleep-EDF arrays must share the same sample axis.")
    if valid_start_mask.ndim != 1 or valid_start_mask.shape[0] != sample_count:
        raise ValueError("Sleep-EDF valid_start_mask must have one entry per sample.")
    if segment_ends.ndim != 1 or segment_ends.size == 0:
        raise ValueError("Sleep-EDF segment_ends must be a non-empty one-dimensional array.")
    if (
        np.any(segment_ends <= 0)
        or np.any(np.diff(segment_ends) <= 0)
        or int(segment_ends[-1]) != sample_count
    ):
        raise ValueError(
            "Sleep-EDF segment_ends must be strictly increasing and end at the sample count."
        )
    if (
        not np.isfinite(params_raw).all()
        or not np.isfinite(cond_raw).all()
        or not np.isfinite(mids).all()
    ):
        raise ValueError("Sleep-EDF arrays contain non-finite values.")
    if int(metadata.get("n_samples_total", -1)) != sample_count:
        raise ValueError("Sleep-EDF metadata sample count does not match the NPZ arrays.")


def build_dataset_splits_from_sleep_edf(
    path: str | Path,
    cfg: OTFlowConfig,
    *,
    stride_train: int = SLEEP_EDF_EPOCH_SAMPLES,
    stride_eval: int = SLEEP_EDF_EPOCH_SAMPLES,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    if int(cfg.history_len) != SLEEP_EDF_HISTORY_LEN:
        raise ValueError(
            f"Sleep-EDF uses a locked 120-second context: history_len must be "
            f"{SLEEP_EDF_HISTORY_LEN}, got {int(cfg.history_len)}."
        )
    if int(cfg.prediction_horizon) != SLEEP_EDF_HORIZON_LEN:
        raise ValueError(
            f"Sleep-EDF uses a locked 30-second continuation: prediction_horizon must be "
            f"{SLEEP_EDF_HORIZON_LEN}, got {int(cfg.prediction_horizon)}."
        )
    if not bool(cfg.use_cond_features):
        raise ValueError("Sleep-EDF requires use_cond_features=True.")
    if bool(cfg.cond_standardize):
        raise ValueError("Sleep-EDF stage indicators require cond_standardize=False.")

    resolved_path = Path(path or sleep_edf_data_path()).expanduser().resolve()
    metadata = _load_validated_sleep_edf_metadata(resolved_path)
    required_arrays = {"params_raw", "cond_raw", "mids", "segment_ends", "valid_start_mask"}
    with np.load(str(resolved_path), allow_pickle=False) as data:
        missing_arrays = sorted(required_arrays.difference(data.files))
        if missing_arrays:
            raise ValueError(f"Sleep-EDF NPZ is missing arrays: {', '.join(missing_arrays)}")
        params_raw = np.asarray(data["params_raw"], dtype=np.float32).copy()
        cond_raw = np.asarray(data["cond_raw"], dtype=np.float32).copy()
        mids = np.asarray(data["mids"], dtype=np.float32).copy()
        segment_ends = np.asarray(data["segment_ends"], dtype=np.int64).copy()
        valid_start_mask = np.asarray(data["valid_start_mask"], dtype=np.uint8).astype(bool)

    _validate_sleep_edf_arrays(
        params_raw=params_raw,
        cond_raw=cond_raw,
        mids=mids,
        segment_ends=segment_ends,
        valid_start_mask=valid_start_mask,
        metadata=metadata,
    )
    if int(cfg.snapshot_dim) != int(params_raw.shape[1]):
        raise ValueError(
            f"Sleep-EDF requires snapshot_dim={params_raw.shape[1]}, got {int(cfg.snapshot_dim)}."
        )
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
    "SLEEP_EDF_COMMON_CHANNELS",
    "SLEEP_EDF_DATASET_KEY",
    "SLEEP_EDF_EPOCH_SAMPLES",
    "SLEEP_EDF_HISTORY_LEN",
    "SLEEP_EDF_HORIZON_LEN",
    "SLEEP_EDF_STAGE_NAMES",
    "build_dataset_splits_from_sleep_edf",
    "medical_staging_root",
    "prepare_sleep_edf_dataset",
    "sleep_edf_source_dir",
]
