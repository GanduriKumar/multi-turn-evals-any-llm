from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ThresholdPolicy:
    """Configuration for per-turn threshold evaluation.

    require: 
      - 'all': all metrics must meet their thresholds
      - 'any': at least one metric must meet its threshold
      - 'weighted': weighted fraction of passing metrics must meet or exceed pass_ratio
    pass_ratio: used only when require='weighted'; value in [0,1]
    default_threshold: used when a metric has no explicit threshold
    """

    require: str = "all"
    pass_ratio: float = 1.0
    default_threshold: float = 0.5


def evaluate_turn_thresholds(
    scores: Mapping[str, float],
    thresholds: Optional[Mapping[str, float]] = None,
    *,
    weights: Optional[Mapping[str, float]] = None,
    policy: Optional[ThresholdPolicy] = None,
) -> Tuple[bool, Dict[str, object]]:
    """Evaluate pass/fail for a turn using per-metric thresholds and optional weights.

    Returns (passed, details) where details include per-metric pass flags and weighted pass fraction.
    """
    pol = policy or ThresholdPolicy()
    thr = thresholds or {}

    per_metric: Dict[str, Dict[str, object]] = {}
    passed_metrics = 0
    total_metrics = 0
    w_pass = 0.0
    w_total = 0.0

    for m, val in scores.items():
        t = float(thr.get(m, pol.default_threshold))
        v = float(val)
        ok = v >= t
        per_metric[m] = {"score": v, "threshold": t, "passed": ok, "weight": float(weights.get(m, 1.0)) if weights else 1.0}
        total_metrics += 1
        passed_metrics += 1 if ok else 0
        w = float(weights.get(m, 1.0)) if weights else 1.0
        w_total += w
        if ok:
            w_pass += w

    # No metrics: treat as fail-safe False
    if total_metrics == 0:
        return False, {
            "policy": pol.__dict__,
            "metrics": per_metric,
            "passed_fraction": 0.0,
            "weighted_pass_fraction": 0.0,
        }

    require = (pol.require or "all").strip().lower()
    if require == "any":
        passed = passed_metrics >= 1
    elif require == "weighted":
        frac = (w_pass / w_total) if w_total > 0 else 0.0
        passed = frac >= float(pol.pass_ratio)
    else:  # all
        passed = passed_metrics == total_metrics

    details: Dict[str, object] = {
        "policy": pol.__dict__,
        "metrics": per_metric,
        "passed_fraction": passed_metrics / total_metrics,
        "weighted_pass_fraction": (w_pass / w_total) if w_total > 0 else 0.0,
    }
    return passed, details


__all__ = ["ThresholdPolicy", "evaluate_turn_thresholds"]
 
# ---- Conversation-level thresholds ----

def evaluate_conversation_thresholds(
    aggregated_scores: Mapping[str, float],
    thresholds: Optional[Mapping[str, float]] = None,
    *,
    weights: Optional[Mapping[str, float]] = None,
    policy: Optional[ThresholdPolicy] = None,
) -> Tuple[bool, Dict[str, object]]:
    """Evaluate pass/fail at the conversation level using aggregated metric scores.

    This reuses the same mechanics as per-turn thresholds, applied to a mapping of
    conversation-level metrics (e.g., mean correctness, adherence, safety).

    Returns (passed, details) where details include per-metric outcomes and fractions.
    """
    return evaluate_turn_thresholds(aggregated_scores, thresholds, weights=weights, policy=policy)


__all__.extend(["evaluate_conversation_thresholds"])
