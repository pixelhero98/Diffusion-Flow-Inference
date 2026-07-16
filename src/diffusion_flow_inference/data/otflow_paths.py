from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV_VAR = "DFI_PROJECT_ROOT"


def project_root() -> Path:
    """Return the explicit project root, or the current working directory.

    Package installation paths are intentionally never used as writable project
    roots. Set ``DFI_PROJECT_ROOT`` when commands are launched outside the project
    directory.
    """

    configured_root = os.environ.get(PROJECT_ROOT_ENV_VAR, "").strip()
    root = Path(configured_root).expanduser() if configured_root else Path.cwd()
    return root.resolve()


def resolve_project_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (project_root() / raw).resolve()


def project_data_root() -> Path:
    return project_root() / "data"


def project_paper_dataset_root() -> Path:
    return project_root() / "paper_datasets"


def project_outputs_root() -> Path:
    return project_root() / "outputs"


def project_backbone_matrix_root() -> Path:
    return project_outputs_root() / "backbone_matrix"


def backbone_manifest_path() -> Path:
    return project_backbone_matrix_root() / "backbone_manifest.json"


def cryptos_data_path() -> Path:
    return project_data_root() / "cryptos_binance_spot_monthly_1s_l10.npz"


def es_mbp_10_data_path() -> Path:
    return project_data_root() / "es_mbp_10.npz"


def sleep_edf_data_path() -> Path:
    return project_data_root() / "sleep_edf_3ch_100hz_stage_conditioned.npz"
