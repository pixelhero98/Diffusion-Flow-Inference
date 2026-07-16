from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

VELOCITY_VARIATION_DIFFICULTY_ROW_KEY = "velocity_variation_difficulty"
VELOCITY_VARIATION_DIFFICULTY_TRACE_KEY = "velocity_variation_difficulty_by_step"

MODEL_SIGNAL_SPECS: Tuple[Tuple[str, str], ...] = (
    ("disagreement", "disagreement_by_step"),
    ("residual_norm", "residual_norm_by_step"),
    ("hybrid_signal", "hybrid_signal_by_step"),
    ("u_disagreement", "u_disagreement_by_step"),
    ("u_residual_norm", "u_residual_norm_by_step"),
    ("u_hybrid_signal", "u_hybrid_signal_by_step"),
    ("variance_scaled_signal", "variance_scaled_signal_by_step"),
    ("top_book_disagreement", "top_book_disagreement_by_step"),
    ("top_book_residual_norm", "top_book_residual_norm_by_step"),
    ("top_book_hybrid_signal", "top_book_hybrid_signal_by_step"),
)
MODEL_SIGNAL_TRACE_KEYS: Tuple[str, ...] = tuple(out_key for _, out_key in MODEL_SIGNAL_SPECS) + (
    VELOCITY_VARIATION_DIFFICULTY_TRACE_KEY,
)


def resolved_velocity_variation_scale(residual_norm_values: Sequence[float]) -> float:
    residual = np.asarray(residual_norm_values, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    return 1.0 if residual.size == 0 else max(float(np.mean(np.clip(residual, 0.0, None))), 1e-8)


def compute_velocity_variation_difficulty_numpy(
    residual_norm: Sequence[float] | np.ndarray,
    disagreement: Sequence[float] | np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale}")
    residual = np.asarray(residual_norm, dtype=np.float64)
    disagreement_values = np.asarray(disagreement, dtype=np.float64)
    if residual.shape != disagreement_values.shape:
        raise ValueError(
            "residual_norm and disagreement must have identical shapes; "
            f"got {residual.shape} and {disagreement_values.shape}."
        )
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(disagreement_values)):
        raise ValueError("residual_norm and disagreement must contain only finite values.")
    return disagreement_values * np.log1p(np.clip(residual, 0.0, None) / scale_value)


__all__ = [
    "MODEL_SIGNAL_SPECS",
    "MODEL_SIGNAL_TRACE_KEYS",
    "VELOCITY_VARIATION_DIFFICULTY_ROW_KEY",
    "VELOCITY_VARIATION_DIFFICULTY_TRACE_KEY",
    "compute_velocity_variation_difficulty_numpy",
    "resolved_velocity_variation_scale",
]
