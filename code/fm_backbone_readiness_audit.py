#!/usr/bin/env python3
"""Write a readiness audit for the FM backbone matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fm_backbone_registry import (
    TRAIN_BUDGET_STEPS,
    build_backbone_readiness_audit,
    default_backbone_manifest_path,
    default_imported_otflow_backbone_root,
    default_otflow_reuse_root,
    project_backbone_matrix_root,
)
from otflow_medical_constants import default_sleep_edf_data_path


def _save_json(payload, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Audit the manifest-driven OTFlow backbone matrix.")
    ap.add_argument("--dataset_root", type=str, default="paper_datasets")
    ap.add_argument("--matrix_root", type=str, default=str(project_backbone_matrix_root()))
    ap.add_argument("--manifest_path", type=str, default=str(default_backbone_manifest_path()))
    ap.add_argument("--otflow_reuse_root", type=str, default=str(default_otflow_reuse_root()))
    ap.add_argument("--imported_backbone_root", type=str, default=str(default_imported_otflow_backbone_root()))
    ap.add_argument("--sleep_edf_path", type=str, default=default_sleep_edf_data_path())
    ap.add_argument("--budget_steps", type=str, default=",".join(str(value) for value in TRAIN_BUDGET_STEPS))
    ap.add_argument("--seed", type=int, default=0)
    return ap


def _parse_int_csv(text: str):
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def run_backbone_readiness_audit(cli_args: argparse.Namespace):
    matrix_root = Path(str(cli_args.matrix_root)).resolve()
    matrix_root.mkdir(parents=True, exist_ok=True)
    payload = build_backbone_readiness_audit(
        matrix_root=matrix_root,
        otflow_reuse_root=str(cli_args.otflow_reuse_root),
        imported_backbone_root=str(cli_args.imported_backbone_root),
        dataset_root=str(cli_args.dataset_root),
        sleep_edf_path=str(cli_args.sleep_edf_path),
        budget_steps=_parse_int_csv(str(cli_args.budget_steps)),
        seed=int(cli_args.seed),
        write_path=str(cli_args.manifest_path),
    )
    _save_json(payload, str(matrix_root / "backbone_readiness_audit.json"))
    return payload


def main() -> None:
    run_backbone_readiness_audit(build_argparser().parse_args())


if __name__ == "__main__":
    main()
