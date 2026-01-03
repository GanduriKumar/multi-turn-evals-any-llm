from __future__ import annotations

from typing import Mapping, Optional, Tuple

from .aggregation import aggregate as aggregate_by


def aggregate_conversation_scores(
    turn_scores: Mapping[str, float],
    *,
    method: str = "mean",
    turn_weights: Optional[Mapping[str, float]] = None,
    threshold: Optional[float] = None,
) -> Tuple[float, bool]:
    """Aggregate per-turn scores into a conversation score and pass/fail.

    - method: "mean" | "min" | "weighted"
    - turn_weights: used when method is "weighted" (defaults to 1.0 for missing turns)
    - threshold: pass if aggregated score >= threshold (defaults to 0.5 if None)
    """
    if not turn_scores:
        return 0.0, False

    score = float(aggregate_by(turn_scores, method=method, weights=turn_weights))

    thr = 0.5 if threshold is None else float(threshold)
    passed = score >= thr
    return score, passed


__all__ = ["aggregate_conversation_scores"]
