from __future__ import annotations

import math
from typing import Dict

from .features import LABELS, probs_to_label


def report_probability(value: float, digits: int = 4) -> float:
    rounded = round(float(value), digits)
    if rounded >= 1.0:
        return round(1.0 - (10 ** -digits), digits)
    if rounded <= 0.0:
        return 0.0
    return rounded


def entropy(probs: Dict[str, float]) -> float:
    values = [max(1e-9, float(probs.get(label, 0))) for label in LABELS]
    return -sum(p * math.log(p) for p in values)


def fuse(xgb_probs: Dict[str, float], lstm_probs: Dict[str, float], anomaly_score: float, sequence_available: bool, alpha: float | None = None) -> dict:
    if alpha is None:
        raise RuntimeError("Adaptive fusion gate is not available. Train Stage 3 before prediction.")
    alpha = max(0.0, min(1.0, float(alpha)))
    final_probs = {
        label: alpha * xgb_probs.get(label, 0.0) + (1.0 - alpha) * lstm_probs.get(label, 0.0)
        for label in LABELS
    }
    total = sum(final_probs.values()) or 1.0
    final_probs = {label: value / total for label, value in final_probs.items()}
    label, confidence = probs_to_label(final_probs)
    return {
        "alpha": round(alpha, 4),
        "final_label": label,
        "final_probability": report_probability(confidence),
        "final_probabilities": {k: report_probability(v) for k, v in final_probs.items()},
        "warning": None,
    }
