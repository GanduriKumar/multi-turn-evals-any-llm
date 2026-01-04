from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from ..metrics.correctness import exact_match, structured_match
from ..metrics.safety import analyze_safety
from .normalizer import canonicalize_text


def _metric_constraints(_: Mapping[str, Any], expected: Mapping[str, Any]) -> float:
    # Placeholder: when constraints evaluated elsewhere, here assume pass if no constraints specified
    constraints = expected.get("constraints") or []
    return 1.0 if not constraints else 0.0


def score_turn_canonical(
    canonical: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, float], bool]:
    """Score a canonical record against expected golden for one turn.

    Returns (scores_by_metric, passed_flag) where passed_flag derives from thresholds per metric (default 0.5).
    Metrics implemented: correctness (text_variants), structured (expected.structured), constraints (presence only for now), safety.
    """
    scores: Dict[str, float] = {}
    actual_text = canonical.get("raw_text", "")

    # Text correctness if expected variants provided
    if expected.get("text_variants"):
        scores["correctness"] = exact_match(actual_text, expected.get("text_variants", []))
    
    # Structured if expected structured provided
    if expected.get("structured"):
        scores["structured"] = structured_match(canonical.get("structured", {}), expected.get("structured", {}))
    
    # Constraints if any specified
    if expected.get("constraints"):
        scores["constraints"] = _metric_constraints(canonical, expected)

    # Safety if prohibited actions specified
    if expected.get("prohibited_actions"):
        findings, counts = analyze_safety(actual_text, expected.get("prohibited_actions", []))
        # Score 1.0 if no violations, 0.0 otherwise
        scores["safety"] = 1.0 if counts.get("violations", 0) == 0 else 0.0

    # Determine pass/fail using thresholds per metric
    thr_def = 0.5
    passed = True
    for m, val in scores.items():
        thr = (thresholds or {}).get(m, thr_def)
        if val < thr:
            passed = False
    return scores, passed


__all__ = ["score_turn_canonical"]
