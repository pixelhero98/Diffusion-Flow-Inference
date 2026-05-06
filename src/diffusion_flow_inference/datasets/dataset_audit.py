#!/usr/bin/env python3
"""Dataset audit helpers for the OTFlow paper benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from diffusion_flow_inference.datasets.experiment_plan import CANONICAL_FORECAST_PAPER_DATASETS, CANONICAL_LOB_PAPER_DATASETS, experiment_plan_by_key
from diffusion_flow_inference.datasets.medical_datasets import (
    SLEEP_EDF_DATASET_KEY,
    default_sleep_edf_data_path,
    default_sleep_edf_metadata_path,
    prepare_sleep_edf_dataset,
)
from diffusion_flow_inference.datasets.monash_datasets import (
    default_audit_path,
    default_manifest_path,
    find_tsf_file,
    get_monash_dataset_spec,
    iter_tsf_series,
    load_monash_manifest,
    parse_tsf_header,
)


LOB_PAPER_DATASETS = tuple(CANONICAL_LOB_PAPER_DATASETS)
STATIONARITY_SAMPLE_SERIES = 16
STATIONARITY_MAX_POINTS = 4096
LOB_STATIONARITY_WINDOW = 4096
LOB_STATIONARITY_WINDOWS = 16
LOB_AUDIT_METADATA: Mapping[str, Mapping[str, Any]] = {
    "cryptos": {
        "display_name": "cryptos",
        "official_horizon": 300,
        "context_length": 256,
    },
    "es_mbp_10": {
        "display_name": "es_mbp_10",
        "official_horizon": 300,
        "context_length": 256,
    },
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Audit Monash + conditional-generation datasets for the OTFlow paper.")
    ap.add_argument("--dataset_root", type=str, default="paper_datasets")
    ap.add_argument("--out_root", type=str, default="results_otflow_paper_prep")
    ap.add_argument(
        "--forecast_datasets",
        type=str,
        default=",".join(CANONICAL_FORECAST_PAPER_DATASETS),
    )
    ap.add_argument("--lob_datasets", type=str, default=",".join(CANONICAL_LOB_PAPER_DATASETS))
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--sleep_edf_path", type=str, default=default_sleep_edf_data_path())
    return ap


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _frequency_to_seconds(freq: str) -> Optional[float]:
    normalized = str(freq).strip().lower().replace(" ", "_")
    mapping = {
        "minutely": 60.0,
        "10_minutes": 600.0,
        "half_hourly": 1800.0,
        "hourly": 3600.0,
        "daily": 86400.0,
        "weekly": 604800.0,
        "monthly": 2629800.0,
        "quarterly": 7889400.0,
        "yearly": 31557600.0,
    }
    return mapping.get(normalized)


def _granularity_bucket(frequency_seconds: Optional[float]) -> str:
    if frequency_seconds is None:
        return "unknown"
    if float(frequency_seconds) <= 900.0:
        return "very_fine"
    if float(frequency_seconds) <= 3600.0:
        return "hourly_medium"
    return "coarse"


def _downsample_series(values: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= int(max_points):
        return arr
    idx = np.linspace(0, arr.size - 1, int(max_points), dtype=np.int64)
    return arr[idx]


def _series_stationarity_score(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 16:
        return {
            "score": float("nan"),
            "label": "insufficient",
            "mean_shift": float("nan"),
            "scale_shift": float("nan"),
            "trend_strength": float("nan"),
            "n_points": int(arr.size),
        }
    arr = _downsample_series(arr, STATIONARITY_MAX_POINTS)
    std = float(np.std(arr))
    if std < 1e-12:
        return {
            "score": 0.0,
            "label": "high",
            "mean_shift": 0.0,
            "scale_shift": 0.0,
            "trend_strength": 0.0,
            "n_points": int(arr.size),
        }
    chunks = [chunk for chunk in np.array_split(arr, 4) if chunk.size > 0]
    chunk_means = np.asarray([float(np.mean(chunk)) for chunk in chunks], dtype=np.float64)
    chunk_stds = np.asarray([float(np.std(chunk)) for chunk in chunks], dtype=np.float64)
    mean_shift = float(np.std(chunk_means) / max(std, 1e-12))
    scale_shift = float(np.std(chunk_stds) / max(float(np.mean(chunk_stds)), 1e-12))
    x = np.arange(arr.size, dtype=np.float64)
    slope = float(np.polyfit(x, arr, 1)[0])
    trend_strength = float(abs(slope) * float(arr.size) / max(std, 1e-12))
    score = float(0.45 * mean_shift + 0.35 * scale_shift + 0.20 * trend_strength)
    if score <= 0.35:
        label = "high"
    elif score <= 0.75:
        label = "medium"
    else:
        label = "low"
    return {
        "score": score,
        "label": label,
        "mean_shift": mean_shift,
        "scale_shift": scale_shift,
        "trend_strength": trend_strength,
        "n_points": int(arr.size),
    }


def _aggregate_stationarity(samples: Sequence[np.ndarray]) -> Dict[str, Any]:
    raw_rows: List[Mapping[str, Any]] = []
    diff_rows: List[Mapping[str, Any]] = []
    for sample in samples:
        arr = np.asarray(sample, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 16:
            continue
        raw_rows.append(_series_stationarity_score(arr))
        diff_rows.append(_series_stationarity_score(np.diff(arr)))
    raw_scores = np.asarray([float(row["score"]) for row in raw_rows if np.isfinite(float(row["score"]))], dtype=np.float64)
    diff_scores = np.asarray([float(row["score"]) for row in diff_rows if np.isfinite(float(row["score"]))], dtype=np.float64)
    raw_mean = float(np.mean(raw_scores)) if raw_scores.size else float("nan")
    diff_mean = float(np.mean(diff_scores)) if diff_scores.size else float("nan")
    overall_score = float(0.35 * raw_mean + 0.65 * diff_mean) if np.isfinite(raw_mean) and np.isfinite(diff_mean) else float("nan")

    def _label(score: float) -> str:
        if not np.isfinite(score):
            return "insufficient"
        if score <= 0.35:
            return "high"
        if score <= 0.75:
            return "medium"
        return "low"

    return {
        "raw_level": {
            "mean_score": raw_mean,
            "label": _label(raw_mean),
            "n_series": int(raw_scores.size),
        },
        "first_difference": {
            "mean_score": diff_mean,
            "label": _label(diff_mean),
            "n_series": int(diff_scores.size),
        },
        "overall": {
            "mean_score": overall_score,
            "label": _label(overall_score),
        },
    }


def _label_regularity_from_gap_stats(gap_cv: float, irregular_share: float) -> str:
    if gap_cv <= 0.05 and irregular_share <= 0.05:
        return "high"
    if gap_cv <= 0.25 and irregular_share <= 0.25:
        return "medium"
    return "low"


def _deterministic_series_indices(total_series: int, sample_count: int) -> List[int]:
    if total_series <= 0:
        return []
    n = min(int(sample_count), int(total_series))
    if n == 1:
        return [0]
    return sorted(set(int(round(idx)) for idx in np.linspace(0, total_series - 1, n)))


def audit_monash_dataset(dataset_root: str | Path, dataset_key: str) -> Dict[str, Any]:
    manifest = load_monash_manifest(default_manifest_path(dataset_root, dataset_key))
    spec = get_monash_dataset_spec(dataset_key)
    source_dir = Path(dataset_root).resolve() / spec.data_subdir / "source"
    try:
        tsf_path = find_tsf_file(source_dir)
        header = parse_tsf_header(tsf_path)
    except FileNotFoundError:
        freq_seconds = _frequency_to_seconds(manifest.frequency)
        payload = {
            "benchmark_family": "forecast_extrapolation",
            "dataset_key": str(dataset_key),
            "display_name": spec.display_name,
            "audit_version": 1,
            "status": "manifest_only",
            "frequency": str(manifest.frequency),
            "frequency_seconds": None if freq_seconds is None else float(freq_seconds),
            "granularity_bucket": _granularity_bucket(freq_seconds),
            "official_horizon": int(manifest.official_horizon),
            "context_length": int(manifest.context_length),
            "n_series": int(manifest.n_series),
            "series_length": {
                "min": int(manifest.min_series_length),
                "median": int(manifest.min_series_length),
                "max": int(manifest.max_series_length),
                "equal_length_header": bool(manifest.min_series_length == manifest.max_series_length),
                "equal_length_empirical": bool(manifest.min_series_length == manifest.max_series_length),
            },
            "missingness": {
                "header_missing": False,
                "missing_rate": 0.0,
                "missing_values": 0,
                "total_values": 0,
            },
            "regularity": {
                "regular_grid": True,
                "label": "high",
                "reason": "Manifest-only fallback inferred from Monash metadata.",
            },
            "stationarity": {
                "raw_level": {"mean_score": float("nan"), "label": "insufficient", "n_series": 0},
                "first_difference": {"mean_score": float("nan"), "label": "insufficient", "n_series": 0},
                "overall": {"mean_score": float("nan"), "label": "insufficient"},
            },
            "sampled_series_count": 0,
            "source_tsf_path": None,
        }
        default_audit_path(dataset_root, dataset_key).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    total_series = 0
    lengths: List[int] = []
    total_values = 0
    total_missing = 0
    for _, _, series_values in iter_tsf_series(tsf_path):
        total_series += 1
        lengths.append(int(len(series_values)))
        total_values += int(len(series_values))
        total_missing += int(sum(1 for item in series_values if item is None))

    sample_indices = set(_deterministic_series_indices(total_series, STATIONARITY_SAMPLE_SERIES))
    sample_series: List[np.ndarray] = []
    for series_index, (_, _, series_values) in enumerate(iter_tsf_series(tsf_path)):
        if series_index not in sample_indices:
            continue
        arr = np.asarray([float(item) for item in series_values if item is not None], dtype=np.float64)
        if arr.size >= 16:
            sample_series.append(arr)

    freq_seconds = _frequency_to_seconds(header.frequency or manifest.frequency)
    missing_rate = float(total_missing) / max(float(total_values), 1.0)
    regularity_label = "high" if bool(header.equal_length) else "medium"
    stationarity = _aggregate_stationarity(sample_series)
    payload = {
        "benchmark_family": "forecast_extrapolation",
        "dataset_key": str(dataset_key),
        "display_name": spec.display_name,
        "audit_version": 1,
        "frequency": str(header.frequency or manifest.frequency),
        "frequency_seconds": None if freq_seconds is None else float(freq_seconds),
        "granularity_bucket": _granularity_bucket(freq_seconds),
        "official_horizon": int(manifest.official_horizon),
        "context_length": int(manifest.context_length),
        "n_series": int(total_series),
        "series_length": {
            "min": int(min(lengths)),
            "median": int(np.median(np.asarray(lengths, dtype=np.int64))),
            "max": int(max(lengths)),
            "equal_length_header": bool(header.equal_length),
            "equal_length_empirical": bool(len(set(lengths)) == 1),
        },
        "missingness": {
            "header_missing": bool(header.missing),
            "missing_rate": float(missing_rate),
            "missing_values": int(total_missing),
            "total_values": int(total_values),
        },
        "regularity": {
            "regular_grid": True,
            "label": str(regularity_label),
            "reason": "Monash TSF datasets are regular-grid benchmark series with explicit frequency metadata.",
        },
        "stationarity": stationarity,
        "sampled_series_count": int(len(sample_series)),
        "source_tsf_path": str(tsf_path),
    }
    default_audit_path(dataset_root, dataset_key).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _lob_data_path(dataset_key: str, cli_args: argparse.Namespace) -> str:
    if str(dataset_key) == "cryptos":
        return str(getattr(cli_args, "cryptos_path", ""))
    if str(dataset_key) == "es_mbp_10":
        return str(getattr(cli_args, "es_path", ""))
    raise ValueError(f"Unknown LOB dataset: {dataset_key}")


def _load_l2_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    for key in ("ask_p", "ask_v", "bid_p", "bid_v", "mids", "params_raw"):
        if key in out:
            out[key] = np.asarray(out[key], dtype=np.float32)
    return out


def _npz_timestamps(data: Mapping[str, Any]) -> Optional[np.ndarray]:
    for key in ("timestamps", "local_timestamps", "ts_event", "ts_recv", "ts"):
        if key in data:
            return np.asarray(data[key], dtype=np.int64)
    return None


def _infer_timestamp_unit(timestamps: np.ndarray) -> Tuple[float, str]:
    arr = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return 1.0, "seconds"
    magnitude = float(np.median(np.abs(arr.astype(np.float64))))
    if magnitude >= 1e17:
        return 1e9, "nanoseconds"
    if magnitude >= 1e14:
        return 1e6, "microseconds"
    if magnitude >= 1e11:
        return 1e3, "milliseconds"
    return 1.0, "seconds"


def _segment_break_threshold(positive_gaps: np.ndarray) -> Optional[float]:
    if positive_gaps.size == 0:
        return None
    median_gap = float(np.median(positive_gaps))
    p99_gap = float(np.percentile(positive_gaps, 99))
    return float(max(10.0 * median_gap, 2.0 * p99_gap))


def _segment_ends(data: Mapping[str, Any], total_length: int, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
    if "segment_ends" in data:
        base_segment_ends = np.asarray(data["segment_ends"], dtype=np.int64).reshape(-1)
    else:
        base_segment_ends = np.asarray([int(total_length)], dtype=np.int64)
    base_segment_ends = np.unique(base_segment_ends[(base_segment_ends > 0) & (base_segment_ends <= int(total_length))])
    if timestamps is None or timestamps.size < 2:
        return base_segment_ends

    raw_gaps = np.diff(np.asarray(timestamps, dtype=np.int64)).astype(np.float64)
    positive_gaps = raw_gaps[np.isfinite(raw_gaps) & (raw_gaps > 0)]
    threshold = _segment_break_threshold(positive_gaps)
    if threshold is None:
        return base_segment_ends

    split_points: List[int] = []
    seg_starts = np.concatenate(([0], base_segment_ends[:-1]))
    for seg_start, seg_end in zip(seg_starts, base_segment_ends):
        start = int(seg_start)
        stop = int(seg_end)
        if stop - start <= 1:
            split_points.append(stop)
            continue
        local_gaps = raw_gaps[start : stop - 1]
        local_split_points = np.nonzero(local_gaps > threshold)[0] + start + 1
        if local_split_points.size > 0:
            split_points.extend(int(point) for point in local_split_points)
        split_points.append(stop)
    return np.unique(np.asarray(split_points, dtype=np.int64))


def _default_lob_audit_path(out_root: Path, dataset_key: str) -> Path:
    return Path(out_root).resolve() / "lob_conditional_generation" / str(dataset_key) / "audit.json"


def _representative_lob_windows(mids: np.ndarray, segment_ends: np.ndarray, window_len: int, n_windows: int) -> List[np.ndarray]:
    windows: List[np.ndarray] = []
    seg_starts = np.concatenate(([0], np.asarray(segment_ends[:-1], dtype=np.int64)))
    for seg_start, seg_end in zip(seg_starts, segment_ends):
        start = int(seg_start)
        stop = int(seg_end)
        seg_len = stop - start
        if seg_len < max(32, int(window_len)):
            continue
        if len(windows) >= int(n_windows):
            break
        window_starts = _deterministic_series_indices(seg_len - int(window_len) + 1, max(1, int(math.ceil(n_windows / max(len(segment_ends), 1)))))
        for offset in window_starts:
            if len(windows) >= int(n_windows):
                break
            left = start + int(offset)
            right = left + int(window_len)
            if right <= stop:
                windows.append(np.asarray(mids[left:right], dtype=np.float64))
    return windows[: int(n_windows)]


def audit_lob_dataset(dataset_key: str, data_path: str) -> Dict[str, Any]:
    plan = LOB_AUDIT_METADATA[str(dataset_key)]
    if not str(data_path).strip() or not Path(str(data_path)).exists():
        return {
            "benchmark_family": "lob_conditional_generation",
            "dataset_key": str(dataset_key),
            "display_name": str(plan["display_name"]),
            "audit_version": 1,
            "status": "missing_source",
            "frequency": "timestamped_event_buckets",
            "frequency_seconds": None,
            "granularity_bucket": "very_fine",
            "official_horizon": int(plan["official_horizon"]),
            "context_length": int(plan["context_length"]),
            "n_series": 1,
            "timestamps_present": False,
            "regularity": {
                "label": "insufficient",
                "regular_grid": False,
            },
            "stationarity": {
                "raw_level": {"mean_score": float("nan"), "label": "insufficient", "n_series": 0},
                "first_difference": {"mean_score": float("nan"), "label": "insufficient", "n_series": 0},
                "overall": {"mean_score": float("nan"), "label": "insufficient"},
            },
            "sampled_windows": 0,
            "source_npz_path": str(data_path),
        }
    data = _load_l2_npz(data_path)
    mids = np.asarray(data["mids"], dtype=np.float64)
    timestamps = _npz_timestamps(data)
    segment_ends = _segment_ends(data, len(mids), timestamps=timestamps)
    time_divisor, time_unit = _infer_timestamp_unit(timestamps) if timestamps is not None else (1.0, "seconds")

    global_gaps = None if timestamps is None else np.diff(timestamps).astype(np.float64)
    if timestamps is None:
        within_gaps = np.asarray([], dtype=np.float64)
    else:
        within_chunks: List[np.ndarray] = []
        seg_starts = np.concatenate(([0], segment_ends[:-1]))
        for seg_start, seg_end in zip(seg_starts, segment_ends):
            start = int(seg_start)
            stop = int(seg_end)
            if stop - start > 1:
                within_chunks.append(np.diff(timestamps[start:stop]).astype(np.float64))
        within_gaps = np.concatenate(within_chunks) if within_chunks else np.asarray([], dtype=np.float64)
    positive_within_gaps = within_gaps[np.isfinite(within_gaps) & (within_gaps > 0)]

    if positive_within_gaps.size > 0:
        median_gap = float(np.median(positive_within_gaps))
        gap_cv = float(np.std(positive_within_gaps) / max(np.mean(positive_within_gaps), 1e-12))
        irregular_share = float(np.mean(np.abs(positive_within_gaps - median_gap) > max(1.0, 0.01 * median_gap)))
        p95 = float(np.percentile(positive_within_gaps, 95))
        p99 = float(np.percentile(positive_within_gaps, 99))
        max_gap = float(np.max(positive_within_gaps))
    else:
        median_gap = float("nan")
        gap_cv = float("nan")
        irregular_share = float("nan")
        p95 = float("nan")
        p99 = float("nan")
        max_gap = float("nan")

    segment_lengths = np.diff(np.concatenate(([0], segment_ends.astype(np.int64))))
    min_segment_length = int(np.min(segment_lengths)) if segment_lengths.size > 0 else int(len(mids))
    window_len = int(min(int(LOB_STATIONARITY_WINDOW), max(32, min_segment_length)))
    stationarity_windows = _representative_lob_windows(mids=mids, segment_ends=segment_ends, window_len=window_len, n_windows=int(LOB_STATIONARITY_WINDOWS))
    stationarity = _aggregate_stationarity(stationarity_windows)
    regularity_label = _label_regularity_from_gap_stats(float(gap_cv), float(irregular_share)) if positive_within_gaps.size > 0 else "unknown"
    gap_seconds = None if not np.isfinite(median_gap) else float(median_gap) / float(time_divisor)
    payload = {
        "benchmark_family": "lob_conditional_generation",
        "dataset_key": str(dataset_key),
        "display_name": str(plan["display_name"]),
        "audit_version": 1,
        "frequency": "timestamped_event_buckets",
        "frequency_seconds": gap_seconds,
        "granularity_bucket": _granularity_bucket(gap_seconds),
        "official_horizon": int(plan["official_horizon"]),
        "context_length": int(plan["context_length"]),
        "n_series": 1,
        "timestamps_present": bool(timestamps is not None),
        "regularity": {
            "label": str(regularity_label),
            "regular_grid": bool(regularity_label == "high"),
            "time_unit": str(time_unit),
            "time_divisor_to_seconds": float(time_divisor),
            "median_gap_raw": None if not np.isfinite(median_gap) else float(median_gap),
            "median_gap_seconds": gap_seconds,
            "gap_cv": None if not np.isfinite(gap_cv) else float(gap_cv),
            "irregular_share": None if not np.isfinite(irregular_share) else float(irregular_share),
            "p95_gap_raw": None if not np.isfinite(p95) else float(p95),
            "p99_gap_raw": None if not np.isfinite(p99) else float(p99),
            "max_gap_raw": None if not np.isfinite(max_gap) else float(max_gap),
            "segment_count": int(segment_ends.size),
            "session_breaks": int(max(0, segment_ends.size - 1)),
            "global_max_gap_raw": None if global_gaps is None or global_gaps.size == 0 else float(np.max(global_gaps)),
        },
        "stationarity": stationarity,
        "sampled_windows": int(len(stationarity_windows)),
        "source_npz_path": str(data_path),
    }
    return payload


def audit_sleep_edf_dataset(data_path: str) -> Dict[str, Any]:
    spec = experiment_plan_by_key()[SLEEP_EDF_DATASET_KEY]
    resolved_path = Path(str(data_path or default_sleep_edf_data_path())).resolve()
    metadata_path = Path(default_sleep_edf_metadata_path()).resolve()
    if not resolved_path.exists():
        try:
            prepare_sleep_edf_dataset(resolved_path)
        except Exception:
            return {
                "benchmark_family": "lob_conditional_generation",
                "dataset_key": SLEEP_EDF_DATASET_KEY,
                "display_name": "sleep_edf",
                "audit_version": 1,
                "status": "missing_source",
                "frequency": "100_hz",
                "frequency_seconds": 1.0 / 100.0,
                "granularity_bucket": "very_fine",
                "official_horizon": int(spec.experiment_horizon),
                "context_length": int(spec.history_len),
                "n_series": 0,
                "timestamps_present": False,
                "regularity": {"label": "high", "regular_grid": True},
                "stationarity": {
                    "raw_level": {"mean_score": float("nan"), "label": "insufficient", "n_series": 0},
                    "first_difference": {"mean_score": float("nan"), "label": "insufficient", "n_series": 0},
                    "overall": {"mean_score": float("nan"), "label": "insufficient"},
                },
                "sampled_windows": 0,
                "source_npz_path": str(resolved_path),
            }

    if not metadata_path.exists():
        prepare_sleep_edf_dataset(resolved_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = np.load(str(resolved_path), allow_pickle=True)
    params_raw = np.asarray(data["params_raw"], dtype=np.float32)
    segment_ends = np.asarray(data["segment_ends"], dtype=np.int64)
    sample_windows = _representative_lob_windows(
        mids=np.asarray(params_raw[:, 0], dtype=np.float64),
        segment_ends=segment_ends,
        window_len=min(int(spec.experiment_horizon), int(params_raw.shape[0])),
        n_windows=12,
    )
    stationarity = _aggregate_stationarity(sample_windows)
    return {
        "benchmark_family": "lob_conditional_generation",
        "dataset_key": SLEEP_EDF_DATASET_KEY,
        "display_name": "sleep_edf",
        "audit_version": 1,
        "frequency": "100_hz",
        "frequency_seconds": 1.0 / 100.0,
        "granularity_bucket": "very_fine",
        "official_horizon": int(spec.experiment_horizon),
        "context_length": int(spec.history_len),
        "n_series": int(metadata.get("n_segments", 0)),
        "timestamps_present": False,
        "regularity": {
            "label": "high",
            "regular_grid": True,
            "reason": "Prepared Sleep-EDF windows use uniformly sampled 100 Hz physiological channels.",
        },
        "stationarity": stationarity,
        "sampled_windows": int(len(sample_windows)),
        "source_npz_path": str(resolved_path),
    }


def run_dataset_audit(cli_args: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = Path(str(cli_args.dataset_root)).resolve()
    out_root = Path(str(cli_args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    forecast_rows = [audit_monash_dataset(dataset_root, key) for key in _parse_csv(cli_args.forecast_datasets)]
    lob_rows = []
    for key in _parse_csv(cli_args.lob_datasets):
        if str(key) == SLEEP_EDF_DATASET_KEY:
            row = audit_sleep_edf_dataset(str(getattr(cli_args, "sleep_edf_path", "")))
        else:
            row = audit_lob_dataset(key, _lob_data_path(key, cli_args))
        lob_path = _default_lob_audit_path(out_root, key)
        lob_path.parent.mkdir(parents=True, exist_ok=True)
        lob_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        lob_rows.append(row)
    payload = {
        "audit_version": 1,
        "forecast_extrapolation": forecast_rows,
        "lob_conditional_generation": lob_rows,
    }
    (out_root / "dataset_audit_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    run_dataset_audit(build_argparser().parse_args())


if __name__ == "__main__":
    main()
