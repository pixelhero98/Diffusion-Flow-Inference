#!/usr/bin/env python3
"""Generate audit-driven non-AR rollout-length review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from otflow_dataset_audit import run_dataset_audit
from otflow_medical_datasets import LONG_TERM_HEADERED_ECG_DATASET_KEY, SLEEP_EDF_DATASET_KEY, default_sleep_edf_data_path
from otflow_monash_datasets import monash_paper_dataset_keys


LOB_PAPER_DATASETS = ("cryptos", "es_mbp_10", SLEEP_EDF_DATASET_KEY)
FORECAST_BASE_CANDIDATES = (4, 8, 16, 32)
LOB_BASE_CANDIDATES = (4, 8, 16, 32)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate audit-driven rollout-length decisions without changing defaults.")
    ap.add_argument("--out_root", type=str, default="results_otflow_paper_prep")
    ap.add_argument("--dataset_root", type=str, default="paper_datasets")
    ap.add_argument(
        "--forecast_datasets",
        type=str,
        default=",".join(list(monash_paper_dataset_keys()) + [LONG_TERM_HEADERED_ECG_DATASET_KEY]),
    )
    ap.add_argument("--lob_datasets", type=str, default=f"cryptos,es_mbp_10,{SLEEP_EDF_DATASET_KEY}")
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--es_path", type=str, default="")
    ap.add_argument("--sleep_edf_path", type=str, default=default_sleep_edf_data_path())
    ap.add_argument("--allow_execute", action="store_true", default=False)
    return ap


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _clean_candidates(values: Iterable[int], *, horizon: Optional[int]) -> List[int]:
    cleaned: List[int] = []
    for raw in values:
        value = int(raw)
        if horizon is not None:
            value = min(int(horizon), value)
        value = max(1, int(value))
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def _is_clean_candidate(value: int, horizon: Optional[int]) -> bool:
    if int(value) > 0 and int(value) & (int(value) - 1) == 0:
        return True
    if horizon is not None and int(value) > 0 and int(horizon) % int(value) == 0:
        return True
    return False


def _granularity_base_length(bucket: str) -> int:
    key = str(bucket)
    if key == "very_fine":
        return 8
    if key == "hourly_medium":
        return 16
    if key == "coarse":
        return 32
    return 16


def _adjust_target_length(base_length: int, regularity_label: str, stationarity_label: str) -> int:
    levels = [4, 8, 16, 32]
    idx = levels.index(int(base_length)) if int(base_length) in levels else 2
    if str(regularity_label) == "low" or str(stationarity_label) == "low":
        idx = max(0, idx - 1)
    elif str(regularity_label) == "high" and str(stationarity_label) == "high":
        idx = min(len(levels) - 1, idx + 1)
    return int(levels[idx])


def _candidate_review_rows(candidates: List[int], *, target_length: int, horizon: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in candidates:
        distance = abs(int(value) - int(target_length))
        rows.append(
            {
                "rollout_length": int(value),
                "target_distance": int(distance),
                "is_power_of_two": bool(int(value) & (int(value) - 1) == 0),
                "clean_horizon_divisor": bool(horizon is not None and int(value) > 0 and int(horizon) % int(value) == 0),
                "is_clean": bool(_is_clean_candidate(int(value), horizon)),
                "validation_score": None,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["target_distance"]),
            0 if bool(row["is_clean"]) else 1,
            -int(row["rollout_length"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["provisional_rank"] = int(rank)
    return rows


def _decision_row(row: Mapping[str, Any], *, benchmark_family: str) -> Dict[str, Any]:
    horizon = int(row["official_horizon"]) if row.get("official_horizon") is not None else None
    candidates = _clean_candidates(
        list(FORECAST_BASE_CANDIDATES) + ([] if horizon is None or benchmark_family != "forecast_extrapolation" else [int(horizon)])
        if benchmark_family == "forecast_extrapolation"
        else list(LOB_BASE_CANDIDATES),
        horizon=horizon,
    )
    regularity_label = str(row["regularity"]["label"])
    stationarity_label = str(row["stationarity"]["overall"]["label"])
    base_length = _granularity_base_length(str(row["granularity_bucket"]))
    adjusted_target = _adjust_target_length(int(base_length), regularity_label, stationarity_label)
    ranked_candidates = _candidate_review_rows(candidates, target_length=int(adjusted_target), horizon=horizon)
    final_choice = int(ranked_candidates[0]["rollout_length"])
    return {
        "dataset_key": str(row["dataset_key"]),
        "benchmark_family": str(benchmark_family),
        "display_name": str(row["display_name"]),
        "official_horizon": None if horizon is None else int(horizon),
        "granularity_bucket": str(row["granularity_bucket"]),
        "regularity_label": regularity_label,
        "stationarity_label": stationarity_label,
        "base_rollout_length": int(base_length),
        "adjusted_target_length": int(adjusted_target),
        "final_rollout_length": int(final_choice),
        "candidate_review": ranked_candidates,
        "decision_rationale": {
            "base_rule": "very_fine->8, hourly_medium->16, coarse->32",
            "regularity_adjustment": "down one level for low regularity, up one level only when regularity and stationarity are both high",
            "stationarity_adjustment": "down one level for low stationarity",
            "tie_break": "longest clean meaningful length among the nearest acceptable candidates",
        },
    }


def run_rollout_length_review(cli_args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(str(cli_args.out_root)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    audit_summary = run_dataset_audit(cli_args)
    forecast_rows = [_decision_row(row, benchmark_family="forecast_extrapolation") for row in audit_summary["forecast_extrapolation"]]
    lob_rows = [_decision_row(row, benchmark_family="lob_conditional_generation") for row in audit_summary["lob_conditional_generation"]]

    review_payload = {
        "prep_only": True,
        "rollout_length_defaults_changed": False,
        "review_policy": {
            "primary": "granularity_regularity_stationarity_audit",
            "tie_break": "longest clean meaningful length among nearest acceptable candidates",
            "discussion_required_before_default_change": True,
        },
        "forecast_extrapolation": forecast_rows,
        "lob_conditional_generation": lob_rows,
    }
    decisions_payload = {
        "forecast_extrapolation": [
            {
                "dataset_key": row["dataset_key"],
                "final_rollout_length": row["final_rollout_length"],
                "granularity_bucket": row["granularity_bucket"],
                "regularity_label": row["regularity_label"],
                "stationarity_label": row["stationarity_label"],
            }
            for row in forecast_rows
        ],
        "lob_conditional_generation": [
            {
                "dataset_key": row["dataset_key"],
                "final_rollout_length": row["final_rollout_length"],
                "granularity_bucket": row["granularity_bucket"],
                "regularity_label": row["regularity_label"],
                "stationarity_label": row["stationarity_label"],
            }
            for row in lob_rows
        ],
    }
    (out_root / "rollout_length_review.json").write_text(json.dumps(review_payload, indent=2), encoding="utf-8")
    (out_root / "rollout_length_decisions.json").write_text(json.dumps(decisions_payload, indent=2), encoding="utf-8")
    return review_payload


def main() -> None:
    run_rollout_length_review(build_argparser().parse_args())


if __name__ == "__main__":
    main()
