from __future__ import annotations

from typing import Mapping, Optional


def mean_aggregate(values: Mapping[str, float]) -> float:
    if not values:
        return 0.0
    return sum(float(v) for v in values.values()) / len(values)


def min_aggregate(values: Mapping[str, float]) -> float:
    if not values:
        return 0.0
    return min(float(v) for v in values.values())


def weighted_aggregate(values: Mapping[str, float], weights: Optional[Mapping[str, float]] = None) -> float:
    if not values:
        return 0.0
    if not weights:
        return mean_aggregate(values)
    acc = 0.0
    total = 0.0
    for k, v in values.items():
        w = float(weights.get(k, 1.0))
        acc += w * float(v)
        total += w
    if total == 0.0:
        return 0.0
    return acc / total


def aggregate(values: Mapping[str, float], method: str = "mean", weights: Optional[Mapping[str, float]] = None) -> float:
    m = (method or "").strip().lower()
    if m == "min":
        return min_aggregate(values)
    if m in ("weighted", "weighted_mean"):
        return weighted_aggregate(values, weights)
    return mean_aggregate(values)


__all__ = ["mean_aggregate", "min_aggregate", "weighted_aggregate", "aggregate"]
