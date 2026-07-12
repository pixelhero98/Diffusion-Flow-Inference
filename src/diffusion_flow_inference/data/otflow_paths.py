
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def cryptos_data_path() -> str:
    return str(project_data_root() / "cryptos_binance_spot_monthly_1s_l10.npz")


def es_mbp_10_data_path() -> str:
    return str(project_data_root() / "es_mbp_10.npz")


def sleep_edf_data_path() -> str:
    return str(project_data_root() / "sleep_edf_3ch_100hz_stage_conditioned.npz")
