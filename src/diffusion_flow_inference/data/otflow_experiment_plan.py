#!/usr/bin/env python3
"""Locked paper experiment horizons and non-AR rollout chunk sizes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from diffusion_flow_inference.data.otflow_medical_constants import SLEEP_EDF_DATASET_KEY

FORECAST_FAMILY = "forecast_extrapolation"
CONDITIONAL_GENERATION_FAMILY = "conditional_generation"


@dataclass(frozen=True)
class DatasetExperimentSpec:
    dataset_key: str
    benchmark_family: str
    display_name: str
    experiment_horizon: int
    future_block_len: int
    history_len: int


PAPER_EXPERIMENT_SPECS: tuple[DatasetExperimentSpec, ...] = (
    DatasetExperimentSpec(
        dataset_key="wind_farms_wo_missing",
        benchmark_family=FORECAST_FAMILY,
        display_name="Wind Farms (Monash, W/O Missing)",
        experiment_horizon=1440,
        future_block_len=1440,
        history_len=1440,
    ),
    DatasetExperimentSpec(
        dataset_key="san_francisco_traffic",
        benchmark_family=FORECAST_FAMILY,
        display_name="San Francisco Traffic (Monash)",
        experiment_horizon=168,
        future_block_len=168,
        history_len=336,
    ),
    DatasetExperimentSpec(
        dataset_key="london_smart_meters_wo_missing",
        benchmark_family=FORECAST_FAMILY,
        display_name="London Smart Meters (Monash, W/O Missing)",
        experiment_horizon=336,
        future_block_len=336,
        history_len=672,
    ),
    DatasetExperimentSpec(
        dataset_key="electricity",
        benchmark_family=FORECAST_FAMILY,
        display_name="Electricity (Monash)",
        experiment_horizon=168,
        future_block_len=168,
        history_len=336,
    ),
    DatasetExperimentSpec(
        dataset_key="solar_energy_10m",
        benchmark_family=FORECAST_FAMILY,
        display_name="Solar Energy (Monash, 10m)",
        experiment_horizon=1008,
        future_block_len=1008,
        history_len=1008,
    ),
    DatasetExperimentSpec(
        dataset_key="cryptos",
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="cryptos",
        experiment_horizon=200,
        future_block_len=200,
        history_len=256,
    ),
    DatasetExperimentSpec(
        dataset_key="es_mbp_10",
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="es_mbp_10",
        experiment_horizon=200,
        future_block_len=200,
        history_len=256,
    ),
    DatasetExperimentSpec(
        dataset_key=SLEEP_EDF_DATASET_KEY,
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        display_name="sleep_edf",
        experiment_horizon=3000,
        future_block_len=3000,
        history_len=12000,
    ),
)

PAPER_FORECAST_DATASETS: tuple[str, ...] = tuple(
    spec.dataset_key for spec in PAPER_EXPERIMENT_SPECS if spec.benchmark_family == FORECAST_FAMILY
)
PAPER_CONDITIONAL_GENERATION_DATASETS: tuple[str, ...] = tuple(
    spec.dataset_key
    for spec in PAPER_EXPERIMENT_SPECS
    if spec.benchmark_family == CONDITIONAL_GENERATION_FAMILY
)


def experiment_plan_by_key() -> Dict[str, DatasetExperimentSpec]:
    return {spec.dataset_key: spec for spec in PAPER_EXPERIMENT_SPECS}


__all__ = [
    "CONDITIONAL_GENERATION_FAMILY",
    "DatasetExperimentSpec",
    "FORECAST_FAMILY",
    "PAPER_CONDITIONAL_GENERATION_DATASETS",
    "PAPER_EXPERIMENT_SPECS",
    "PAPER_FORECAST_DATASETS",
    "experiment_plan_by_key",
]
