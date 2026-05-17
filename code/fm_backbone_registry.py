from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from otflow_experiment_plan import CANONICAL_FORECAST_PAPER_DATASETS, CANONICAL_LOB_PAPER_DATASETS
from otflow_medical_constants import (
    LONG_TERM_HEADERED_ECG_DATASET_KEY,
    SLEEP_EDF_DATASET_KEY,
    default_long_term_headered_ecg_manifest_path,
    default_sleep_edf_data_path,
)
from otflow_paths import project_backbone_matrix_root as default_project_backbone_matrix_root, project_data_root, project_outputs_root, project_paper_dataset_root

FORECAST_FAMILY = "forecast_extrapolation"
LOB_FAMILY = "lob_conditional_generation"
BACKBONE_NAME_OTFLOW = "otflow"
LOB_FIELD_NETWORK_TYPE = "transformer"
DEFAULT_SEED = 0
TRAIN_BUDGET_STEPS: Tuple[int, ...] = (4000, 8000, 12000, 16000, 20000)
STANDARD_ARTIFACT_SUMMARY_NAME = "artifact_summary.json"
MANIFEST_VERSION = "fm_backbone_manifest_v1"
AUDIT_VERSION = "fm_backbone_readiness_audit_v1"
IMPORTED_EXTERNAL_SOURCE_KIND = "imported_external"

ACTIVE_FORECAST_BACKBONE_BUDGETS: Mapping[str, Tuple[int, ...]] = {
    "san_francisco_traffic": (4000, 8000, 12000, 16000, 20000),
    "london_smart_meters_wo_missing": (4000, 8000, 12000, 16000, 20000),
    "electricity": (4000, 8000, 12000, 16000, 20000),
    "solar_energy_10m": (4000, 8000, 12000, 16000, 20000),
    "wind_farms_wo_missing": (4000, 8000, 12000, 16000, 20000),
}
ACTIVE_LOB_BACKBONE_BUDGETS: Mapping[str, Tuple[int, ...]] = {
    "cryptos": (4000, 8000, 12000, 16000, 20000),
    "es_mbp_10": (4000, 8000, 12000, 16000, 20000),
    SLEEP_EDF_DATASET_KEY: (4000, 8000, 12000, 16000, 20000),
}


@dataclass(frozen=True)
class BackboneArtifactSpec:
    backbone_name: str
    benchmark_family: str
    dataset_key: str
    train_steps: int
    train_budget_label: str
    checkpoint_id: str
    checkpoint_path: str
    summary_path: str
    status: str
    seed: int = DEFAULT_SEED
    source_kind: str = "planned"
    metadata_path: Optional[str] = None
    field_network_type: Optional[str] = None
    notes: Optional[str] = None
    model_cond_dim: Optional[int] = None
    compatibility_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def train_budget_label(train_steps: int) -> str:
    steps = int(train_steps)
    for value in TRAIN_BUDGET_STEPS:
        if int(value) == steps:
            return f"{int(value) // 1000}k"
    if steps % 1000 == 0:
        return f"{steps // 1000}k"
    return f"{steps}_steps"


def project_backbone_matrix_root() -> Path:
    return default_project_backbone_matrix_root()


def default_backbone_manifest_path() -> Path:
    return project_backbone_matrix_root() / "backbone_manifest.json"


def default_otflow_reuse_root() -> Path:
    return project_outputs_root() / "shared_backbones" / "otflow_fullhorizon_seed0"


def default_imported_otflow_backbone_root() -> Path:
    return project_outputs_root() / "imported_backbones" / "otflow"


def build_backbone_checkpoint_id(
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    seed: int = DEFAULT_SEED,
    field_network_type: Optional[str] = None,
) -> str:
    family_token = "forecast" if str(benchmark_family) == FORECAST_FAMILY else "lob"
    parts = [str(dataset_key), str(backbone_name), family_token]
    if str(benchmark_family) == LOB_FAMILY and field_network_type:
        parts.append(str(field_network_type))
    parts.extend([train_budget_label(int(train_steps)), f"seed{int(seed)}"])
    return "_".join(parts)


def _active_budget_map(benchmark_family: str) -> Mapping[str, Tuple[int, ...]]:
    if str(benchmark_family) == FORECAST_FAMILY:
        return ACTIVE_FORECAST_BACKBONE_BUDGETS
    if str(benchmark_family) == LOB_FAMILY:
        return ACTIVE_LOB_BACKBONE_BUDGETS
    raise ValueError(f"Unsupported benchmark_family={benchmark_family}")


def _forecast_artifact_root(matrix_root: Path, backbone_name: str, dataset_key: str, train_steps: int) -> Path:
    return matrix_root / str(backbone_name) / "forecast" / train_budget_label(int(train_steps)) / str(dataset_key)


def _lob_artifact_root(matrix_root: Path, backbone_name: str, dataset_key: str, train_steps: int) -> Path:
    return matrix_root / str(backbone_name) / "lob" / train_budget_label(int(train_steps)) / str(dataset_key) / LOB_FIELD_NETWORK_TYPE


def expected_artifact_root(
    matrix_root: str | Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
) -> Path:
    root = Path(matrix_root).resolve()
    if str(benchmark_family) == FORECAST_FAMILY:
        return _forecast_artifact_root(root, str(backbone_name), str(dataset_key), int(train_steps))
    if str(benchmark_family) == LOB_FAMILY:
        return _lob_artifact_root(root, str(backbone_name), str(dataset_key), int(train_steps))
    raise ValueError(f"Unsupported benchmark_family={benchmark_family}")


def _expected_materialized_paths(
    matrix_root: str | Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
) -> Dict[str, Path]:
    artifact_root = expected_artifact_root(
        matrix_root,
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    checkpoint_name = "model.pt"
    metadata_name = "checkpoint_metadata.json"
    return {
        "artifact_root": artifact_root,
        "checkpoint_path": artifact_root / checkpoint_name,
        "summary_path": artifact_root / STANDARD_ARTIFACT_SUMMARY_NAME,
        "metadata_path": artifact_root / metadata_name,
    }


def _existing_summary_path(
    artifact_root: Path,
    *,
    preferred: Path,
    benchmark_family: str,
    backbone_name: str,
) -> Path:
    candidates = [preferred]
    if str(backbone_name) == BACKBONE_NAME_OTFLOW and str(benchmark_family) == FORECAST_FAMILY:
        candidates.append(artifact_root / "forecast_summary.json")
    candidates.append(artifact_root / "checkpoint_metadata.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return preferred


def _safe_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_cond_dim(metadata: Mapping[str, Any]) -> int:
    split_stats = dict(metadata.get("split_stats", {}) or {})
    if "cond_dim" in split_stats:
        return int(split_stats.get("cond_dim") or 0)
    return int(metadata.get("cond_dim") or 0)


def _checkpoint_cond_signature(checkpoint_path: Path) -> Tuple[Optional[int], Optional[str]]:
    try:
        import torch

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return None, f"Unable to inspect checkpoint compatibility: {type(exc).__name__}: {exc}"
    cfg = dict(checkpoint.get("cfg", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    model_cond_dim = int(model_cfg.get("cond_dim", 0) or 0)
    state = dict(checkpoint.get("model_state", {}) or {})
    has_cond_mlp = any("cond_mlp" in str(key) for key in state)
    if model_cond_dim > 0 and not has_cond_mlp:
        return model_cond_dim, "Checkpoint declares model.cond_dim>0 but has no cond_mlp weights."
    return model_cond_dim, None


def _conditional_artifact_compatibility(metadata: Optional[Mapping[str, Any]], checkpoint_path: Path) -> Tuple[Optional[int], str, Optional[str]]:
    if not metadata:
        return None, "ready", None
    cond_dim = _metadata_cond_dim(metadata)
    if cond_dim <= 0:
        return None, "ready", None
    model_cond_dim, error = _checkpoint_cond_signature(checkpoint_path)
    if error is not None:
        return model_cond_dim, "invalid", error
    if int(model_cond_dim or 0) != int(cond_dim):
        return (
            model_cond_dim,
            "invalid",
            f"split_stats.cond_dim={int(cond_dim)} but checkpoint model.cond_dim={int(model_cond_dim or 0)}.",
        )
    return model_cond_dim, "ready", None


def _existing_matrix_artifact(
    matrix_root: Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    seed: int,
) -> Optional[BackboneArtifactSpec]:
    paths = _expected_materialized_paths(
        matrix_root,
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    if not paths["checkpoint_path"].exists():
        return None
    metadata = _safe_json(paths["metadata_path"])
    model_cond_dim, status, compatibility_error = _conditional_artifact_compatibility(metadata, paths["checkpoint_path"])
    checkpoint_id = None if metadata is None else metadata.get("checkpoint_id")
    summary_path = _existing_summary_path(
        paths["artifact_root"],
        preferred=paths["summary_path"],
        benchmark_family=str(benchmark_family),
        backbone_name=str(backbone_name),
    )
    field_network_type = None if metadata is None else metadata.get("field_network_type")
    return BackboneArtifactSpec(
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
        train_budget_label=train_budget_label(int(train_steps)),
        checkpoint_id=str(
            checkpoint_id
            or build_backbone_checkpoint_id(
                backbone_name=str(backbone_name),
                benchmark_family=str(benchmark_family),
                dataset_key=str(dataset_key),
                train_steps=int(train_steps),
                seed=int(seed),
                field_network_type=field_network_type if field_network_type else None,
            )
        ),
        checkpoint_path=str(paths["checkpoint_path"]),
        summary_path=str(summary_path),
        status=str(status),
        seed=int(seed),
        source_kind="matrix_output",
        metadata_path=str(paths["metadata_path"]),
        field_network_type=None if field_network_type is None else str(field_network_type),
        notes=compatibility_error,
        model_cond_dim=model_cond_dim,
        compatibility_error=compatibility_error,
    )


def _existing_otflow_reuse_artifact(
    reuse_root: Path,
    *,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    seed: int,
) -> Optional[BackboneArtifactSpec]:
    if int(train_steps) != 20000:
        return None
    if str(benchmark_family) == FORECAST_FAMILY:
        artifact_root = reuse_root / "forecast" / str(dataset_key)
        checkpoint_path = artifact_root / "model.pt"
        metadata_path = artifact_root / "checkpoint_metadata.json"
        summary_path = artifact_root / "forecast_summary.json"
        field_network_type = None
    elif str(benchmark_family) == LOB_FAMILY:
        artifact_root = reuse_root / "lob" / str(dataset_key) / LOB_FIELD_NETWORK_TYPE
        checkpoint_path = artifact_root / "model.pt"
        metadata_path = artifact_root / "checkpoint_metadata.json"
        summary_path = artifact_root / "checkpoint_metadata.json"
        field_network_type = LOB_FIELD_NETWORK_TYPE
    else:
        raise ValueError(f"Unsupported benchmark_family={benchmark_family}")
    if not checkpoint_path.exists():
        return None
    metadata = _safe_json(metadata_path) or {}
    model_cond_dim, status, compatibility_error = _conditional_artifact_compatibility(metadata, checkpoint_path)
    resolved_steps = int(metadata.get("train_steps", 20000))
    if resolved_steps != 20000:
        return None
    return BackboneArtifactSpec(
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=resolved_steps,
        train_budget_label=train_budget_label(resolved_steps),
        checkpoint_id=build_backbone_checkpoint_id(
            backbone_name=BACKBONE_NAME_OTFLOW,
            benchmark_family=str(benchmark_family),
            dataset_key=str(dataset_key),
            train_steps=resolved_steps,
            seed=int(seed),
            field_network_type=field_network_type,
        ),
        checkpoint_path=str(checkpoint_path),
        summary_path=str(summary_path if summary_path.exists() else metadata_path),
        status=str(status),
        seed=int(seed),
        source_kind="reused_shared_20k",
        metadata_path=str(metadata_path),
        field_network_type=field_network_type,
        notes=compatibility_error,
        model_cond_dim=model_cond_dim,
        compatibility_error=compatibility_error,
    )


def _imported_source_dir(
    imported_root: Path,
    *,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
) -> Path:
    return (
        imported_root
        / str(benchmark_family)
        / str(dataset_key)
        / train_budget_label(int(train_steps))
    )


def _rewrite_normalized_json_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    metadata_path: Path,
    summary_path: Path,
    normalized_from: Path,
) -> Dict[str, Any]:
    data = dict(payload)
    if "checkpoint_path" in data:
        data["checkpoint_path"] = str(checkpoint_path)
    if "checkpoint_metadata_path" in data:
        data["checkpoint_metadata_path"] = str(metadata_path)
    if "metadata_path" in data:
        data["metadata_path"] = str(metadata_path)
    if "summary_path" in data:
        data["summary_path"] = str(summary_path)
    data["normalized_from"] = str(normalized_from)
    return data


def _copy_json_with_rewritten_paths(
    src_path: Path,
    dst_path: Path,
    *,
    checkpoint_path: Path,
    metadata_path: Path,
    summary_path: Path,
) -> None:
    payload = _safe_json(src_path)
    if payload is None:
        shutil.copy2(src_path, dst_path)
        return
    rewritten = _rewrite_normalized_json_payload(
        payload,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        summary_path=summary_path,
        normalized_from=src_path.parent,
    )
    dst_path.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")


def _normalize_imported_artifact(
    imported_root: Path,
    matrix_root: Path,
    *,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
) -> bool:
    source_dir = _imported_source_dir(
        imported_root,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    if not source_dir.exists():
        return False
    source_checkpoint = source_dir / "model.pt"
    source_metadata = source_dir / "checkpoint_metadata.json"
    if not source_checkpoint.exists() or not source_metadata.exists():
        return False
    paths = _expected_materialized_paths(
        matrix_root,
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    artifact_root = paths["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, paths["checkpoint_path"])
    for name in ("artifact_summary.json", "forecast_summary.json", "transfer_record.json"):
        source_file = source_dir / name
        if not source_file.exists():
            continue
        if source_file.suffix.lower() == ".json":
            summary_target = (
                paths["summary_path"]
                if name == STANDARD_ARTIFACT_SUMMARY_NAME
                else artifact_root / name
            )
            _copy_json_with_rewritten_paths(
                source_file,
                summary_target,
                checkpoint_path=paths["checkpoint_path"],
                metadata_path=paths["metadata_path"],
                summary_path=paths["summary_path"],
            )
        else:
            shutil.copy2(source_file, artifact_root / name)
    _copy_json_with_rewritten_paths(
        source_metadata,
        paths["metadata_path"],
        checkpoint_path=paths["checkpoint_path"],
        metadata_path=paths["metadata_path"],
        summary_path=paths["summary_path"],
    )
    if not paths["summary_path"].exists():
        payload = _safe_json(paths["metadata_path"]) or {}
        summary_payload = _rewrite_normalized_json_payload(
            payload,
            checkpoint_path=paths["checkpoint_path"],
            metadata_path=paths["metadata_path"],
            summary_path=paths["summary_path"],
            normalized_from=source_dir,
        )
        paths["summary_path"].write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return True


def normalize_imported_backbone_artifacts(
    *,
    matrix_root: str | Path | None = None,
    imported_root: str | Path | None = None,
    budget_steps: Sequence[int] = TRAIN_BUDGET_STEPS,
) -> Dict[str, Any]:
    resolved_matrix_root = Path(matrix_root or project_backbone_matrix_root()).resolve()
    resolved_import_root = Path(imported_root or default_imported_otflow_backbone_root()).resolve()
    normalized: List[Dict[str, Any]] = []
    if not resolved_import_root.exists():
        return {
            "matrix_root": str(resolved_matrix_root),
            "imported_root": str(resolved_import_root),
            "normalized_count": 0,
            "normalized": normalized,
        }
    requested_steps = {int(value) for value in budget_steps}
    for benchmark_family, budget_map in (
        (FORECAST_FAMILY, ACTIVE_FORECAST_BACKBONE_BUDGETS),
        (LOB_FAMILY, ACTIVE_LOB_BACKBONE_BUDGETS),
    ):
        for dataset_key, active_steps in budget_map.items():
            for train_steps in active_steps:
                if int(train_steps) not in requested_steps:
                    continue
                if not _normalize_imported_artifact(
                    resolved_import_root,
                    resolved_matrix_root,
                    benchmark_family=str(benchmark_family),
                    dataset_key=str(dataset_key),
                    train_steps=int(train_steps),
                ):
                    continue
                normalized.append(
                    {
                        "benchmark_family": str(benchmark_family),
                        "dataset_key": str(dataset_key),
                        "train_steps": int(train_steps),
                    }
                )
    return {
        "matrix_root": str(resolved_matrix_root),
        "imported_root": str(resolved_import_root),
        "normalized_count": int(len(normalized)),
        "normalized": normalized,
    }


def _planned_artifact(
    matrix_root: Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    seed: int,
) -> BackboneArtifactSpec:
    paths = _expected_materialized_paths(
        matrix_root,
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
    )
    field_network_type = LOB_FIELD_NETWORK_TYPE if str(benchmark_family) == LOB_FAMILY else None
    return BackboneArtifactSpec(
        backbone_name=str(backbone_name),
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(train_steps),
        train_budget_label=train_budget_label(int(train_steps)),
        checkpoint_id=build_backbone_checkpoint_id(
            backbone_name=str(backbone_name),
            benchmark_family=str(benchmark_family),
            dataset_key=str(dataset_key),
            train_steps=int(train_steps),
            seed=int(seed),
            field_network_type=field_network_type,
        ),
        checkpoint_path=str(paths["checkpoint_path"]),
        summary_path=str(paths["summary_path"]),
        status="missing",
        seed=int(seed),
        source_kind="planned",
        metadata_path=str(paths["metadata_path"]),
        field_network_type=field_network_type,
    )


def _iter_target_specs(
    *,
    matrix_root: Path,
    seed: int,
    budget_steps: Sequence[int],
) -> Iterable[BackboneArtifactSpec]:
    requested_steps = {int(value) for value in budget_steps}
    for dataset_key in tuple(CANONICAL_FORECAST_PAPER_DATASETS):
        for train_steps in ACTIVE_FORECAST_BACKBONE_BUDGETS.get(str(dataset_key), ()):
            if int(train_steps) not in requested_steps:
                continue
            yield _planned_artifact(
                matrix_root,
                backbone_name=BACKBONE_NAME_OTFLOW,
                benchmark_family=FORECAST_FAMILY,
                dataset_key=str(dataset_key),
                train_steps=int(train_steps),
                seed=int(seed),
            )
    for dataset_key in tuple(CANONICAL_LOB_PAPER_DATASETS):
        for train_steps in ACTIVE_LOB_BACKBONE_BUDGETS.get(str(dataset_key), ()):
            if int(train_steps) not in requested_steps:
                continue
            yield _planned_artifact(
                matrix_root,
                backbone_name=BACKBONE_NAME_OTFLOW,
                benchmark_family=LOB_FAMILY,
                dataset_key=str(dataset_key),
                train_steps=int(train_steps),
                seed=int(seed),
            )
def materialize_backbone_manifest(
    *,
    matrix_root: str | Path | None = None,
    otflow_reuse_root: str | Path | None = None,
    imported_backbone_root: str | Path | None = None,
    budget_steps: Sequence[int] = TRAIN_BUDGET_STEPS,
    seed: int = DEFAULT_SEED,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_matrix_root = Path(matrix_root or project_backbone_matrix_root()).resolve()
    resolved_reuse_root = Path(otflow_reuse_root or default_otflow_reuse_root()).resolve()
    resolved_import_root = Path(imported_backbone_root or default_imported_otflow_backbone_root()).resolve()
    artifacts: List[Dict[str, Any]] = []
    ready_count = 0
    for planned in _iter_target_specs(matrix_root=resolved_matrix_root, seed=int(seed), budget_steps=budget_steps):
        resolved = _existing_matrix_artifact(
            resolved_matrix_root,
            backbone_name=str(planned.backbone_name),
            benchmark_family=str(planned.benchmark_family),
            dataset_key=str(planned.dataset_key),
            train_steps=int(planned.train_steps),
            seed=int(seed),
        )
        if resolved is None and str(planned.backbone_name) == BACKBONE_NAME_OTFLOW:
            resolved = _existing_otflow_reuse_artifact(
                resolved_reuse_root,
                benchmark_family=str(planned.benchmark_family),
                dataset_key=str(planned.dataset_key),
                train_steps=int(planned.train_steps),
                seed=int(seed),
            )
        artifact = planned if resolved is None else resolved
        if str(artifact.status) == "ready":
            ready_count += 1
        artifacts.append(artifact.to_dict())
    payload = {
        "version": MANIFEST_VERSION,
        "seed": int(seed),
        "train_budget_steps": [int(value) for value in budget_steps],
        "matrix_root": str(resolved_matrix_root),
        "otflow_reuse_root": str(resolved_reuse_root),
        "imported_backbone_root": str(resolved_import_root),
        "artifact_count": int(len(artifacts)),
        "ready_count": int(ready_count),
        "missing_count": int(len(artifacts) - ready_count),
        "artifacts": artifacts,
    }
    target_path = Path(write_path or default_backbone_manifest_path()).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_backbone_manifest(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def find_backbone_artifact(
    manifest_payload: Mapping[str, Any],
    *,
    backbone_name: str,
    benchmark_family: str,
    dataset_key: str,
    train_steps: int,
    status: str = "ready",
) -> Dict[str, Any]:
    for artifact in manifest_payload.get("artifacts", []):
        if (
            str(artifact.get("backbone_name")) == str(backbone_name)
            and str(artifact.get("benchmark_family")) == str(benchmark_family)
            and str(artifact.get("dataset_key")) == str(dataset_key)
            and int(artifact.get("train_steps", -1)) == int(train_steps)
            and str(artifact.get("status")) == str(status)
        ):
            return dict(artifact)
    raise KeyError(
        "No matching backbone artifact found for "
        f"{backbone_name}/{benchmark_family}/{dataset_key}/{int(train_steps)} with status={status}"
    )


def build_runtime_probe(
    *,
    dataset_root: str | Path | None = None,
    sleep_edf_path: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_dataset_root = Path(dataset_root or project_paper_dataset_root()).resolve()
    resolved_sleep_path = Path(sleep_edf_path or default_sleep_edf_data_path()).resolve()
    monash_root = resolved_dataset_root / "monash"
    import_names = ("numpy", "torch", "wfdb", "pyedflib")
    imports = {
        name: bool(importlib.util.find_spec(name) is not None)
        for name in import_names
    }
    forecast_dataset_presence = {
        str(dataset_key): bool((monash_root / str(dataset_key) / "manifest.json").exists())
        for dataset_key in CANONICAL_FORECAST_PAPER_DATASETS
        if str(dataset_key) != LONG_TERM_HEADERED_ECG_DATASET_KEY
    }
    dataset_presence = {
        "monash_manifests": forecast_dataset_presence,
        LONG_TERM_HEADERED_ECG_DATASET_KEY: bool(default_long_term_headered_ecg_manifest_path(resolved_dataset_root).exists()),
        SLEEP_EDF_DATASET_KEY: bool(resolved_sleep_path.exists()),
        "cryptos_npz": bool((project_data_root() / "cryptos_binance_spot_monthly_1s_l10.npz").exists()),
        "es_mbp_10_npz": bool((project_data_root() / "es_mbp_10.npz").exists()),
    }
    return {
        "dataset_root": str(resolved_dataset_root),
        "sleep_edf_path": str(resolved_sleep_path),
        "imports": imports,
        "dataset_presence": dataset_presence,
    }


def build_backbone_readiness_audit(
    *,
    matrix_root: str | Path | None = None,
    otflow_reuse_root: str | Path | None = None,
    imported_backbone_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    sleep_edf_path: str | Path | None = None,
    budget_steps: Sequence[int] = TRAIN_BUDGET_STEPS,
    seed: int = DEFAULT_SEED,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    normalization = normalize_imported_backbone_artifacts(
        matrix_root=matrix_root,
        imported_root=imported_backbone_root,
        budget_steps=budget_steps,
    )
    manifest = materialize_backbone_manifest(
        matrix_root=matrix_root,
        otflow_reuse_root=otflow_reuse_root,
        imported_backbone_root=imported_backbone_root,
        budget_steps=budget_steps,
        seed=int(seed),
        write_path=write_path,
    )
    readiness = {
        "version": AUDIT_VERSION,
        "manifest_path": str(Path(write_path or default_backbone_manifest_path()).resolve()),
        "manifest": manifest,
        "normalization": normalization,
        "runtime_probe": build_runtime_probe(dataset_root=dataset_root, sleep_edf_path=sleep_edf_path),
    }
    return readiness


__all__ = [
    "ACTIVE_FORECAST_BACKBONE_BUDGETS",
    "ACTIVE_LOB_BACKBONE_BUDGETS",
    "AUDIT_VERSION",
    "BACKBONE_NAME_OTFLOW",
    "DEFAULT_SEED",
    "FORECAST_FAMILY",
    "IMPORTED_EXTERNAL_SOURCE_KIND",
    "LOB_FAMILY",
    "LOB_FIELD_NETWORK_TYPE",
    "MANIFEST_VERSION",
    "STANDARD_ARTIFACT_SUMMARY_NAME",
    "TRAIN_BUDGET_STEPS",
    "BackboneArtifactSpec",
    "build_backbone_checkpoint_id",
    "build_backbone_readiness_audit",
    "default_backbone_manifest_path",
    "default_imported_otflow_backbone_root",
    "default_otflow_reuse_root",
    "expected_artifact_root",
    "find_backbone_artifact",
    "load_backbone_manifest",
    "materialize_backbone_manifest",
    "normalize_imported_backbone_artifacts",
    "project_backbone_matrix_root",
    "train_budget_label",
]
