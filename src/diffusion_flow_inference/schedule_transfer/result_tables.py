from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _summary_match_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row["benchmark_family"]),
        str(row["dataset"]),
        int(row["target_nfe"]),
        str(row["solver_key"]),
        str(row["train_budget_label"]),
    )


def _safe_relative_gain(metric_value: Any, baseline_value: Any) -> Optional[float]:
    try:
        metric = float(metric_value)
        baseline = float(baseline_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metric) or not math.isfinite(baseline) or baseline <= 0.0:
        return None
    return float(1.0 - (metric / baseline))


def _summary_relative_gain(row: Mapping[str, Any], relative_key: str) -> Optional[float]:
    value = row.get(f"{relative_key}_mean")
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cast):
        return None
    return cast


def _summary_metric(row: Mapping[str, Any], metric_key: str) -> Any:
    return row.get(f"{metric_key}_mean")


def augment_rows_with_relative_metrics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    baseline_rows: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if str(row["schedule_key"]) == "uniform":
            baseline_rows[_summary_match_key(row)] = row

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        baseline = baseline_rows.get(_summary_match_key(row))
        family = str(row["benchmark_family"])
        payload["relative_crps_gain_vs_uniform"] = _summary_relative_gain(
            row, "relative_crps_gain_vs_uniform"
        )
        payload["relative_mase_gain_vs_uniform"] = _summary_relative_gain(
            row, "relative_mase_gain_vs_uniform"
        )
        payload["relative_score_gain_vs_uniform"] = _summary_relative_gain(
            row, "relative_score_gain_vs_uniform"
        )
        if baseline is not None and family == "forecast_extrapolation":
            if payload["relative_crps_gain_vs_uniform"] is None:
                payload["relative_crps_gain_vs_uniform"] = _safe_relative_gain(
                    _summary_metric(row, "crps"),
                    _summary_metric(baseline, "crps"),
                )
            if payload["relative_mase_gain_vs_uniform"] is None:
                payload["relative_mase_gain_vs_uniform"] = _safe_relative_gain(
                    _summary_metric(row, "mase"),
                    _summary_metric(baseline, "mase"),
                )
        if baseline is not None and family == "conditional_generation":
            if payload["relative_score_gain_vs_uniform"] is None:
                payload["relative_score_gain_vs_uniform"] = _safe_relative_gain(
                    _summary_metric(row, "score_main"),
                    _summary_metric(baseline, "score_main"),
                )
        enriched.append(payload)
    return enriched


__all__ = ["augment_rows_with_relative_metrics"]
