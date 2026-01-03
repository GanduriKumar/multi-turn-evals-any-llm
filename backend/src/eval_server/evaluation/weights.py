from __future__ import annotations

from typing import Dict, Mapping, Optional


MetricScores = Mapping[str, float]
MetricWeights = Mapping[str, float]


def weighted_average(scores: MetricScores, weights: Optional[MetricWeights] = None) -> float:
    if not scores:
        return 0.0
    if not weights:
        # Simple mean
        return sum(scores.values()) / len(scores)
    total_w = 0.0
    acc = 0.0
    for k, v in scores.items():
        w = float(weights.get(k, 0.0))
        total_w += w
        acc += w * float(v)
    if total_w == 0.0:
        return 0.0
    return acc / total_w


def aggregate_conversation(turn_scores: Mapping[str, float], turn_weights: Optional[Mapping[str, float]] = None) -> float:
    """Aggregate per-turn scores into a conversation score using optional per-turn weights.

    - turn_scores: mapping turn_id -> score
    - turn_weights: mapping turn_id -> weight (defaults to 1.0 when not provided)
    """
    if not turn_scores:
        return 0.0
    if not turn_weights:
        # Unweighted mean
        return sum(turn_scores.values()) / len(turn_scores)
    acc = 0.0
    total = 0.0
    for t_id, score in turn_scores.items():
        w = float(turn_weights.get(t_id, 1.0))
        acc += w * float(score)
        total += w
    if total == 0.0:
        return 0.0
    return acc / total


__all__ = ["weighted_average", "aggregate_conversation"]
