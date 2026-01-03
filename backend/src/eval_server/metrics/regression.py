from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple, Literal, Any


Direction = Literal["higher_is_better", "lower_is_better"]
MissingPolicy = Literal["ignore", "current_zero", "baseline_zero"]


@dataclass(frozen=True)
class RegressionSummary:
    total_compared: int
    improved: int
    regressed: int
    unchanged: int
    mean_delta: float
    worst_delta: float | None
    threshold: float
    direction: Direction


def _iter_pairs(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    *,
    missing: MissingPolicy,
):
    keys: set[str] = set(baseline.keys()) | set(current.keys())
    for k in keys:
        b = baseline.get(k)
        c = current.get(k)
        if b is None and c is None:
            continue
        if b is None:
            if missing == "ignore":
                continue
            if missing == "baseline_zero":
                b = 0.0
            elif missing == "current_zero":
                # Nothing to do, baseline missing does not apply to this case
                continue
        if c is None:
            if missing == "ignore":
                continue
            if missing == "current_zero":
                c = 0.0
            elif missing == "baseline_zero":
                # Nothing to do, current missing does not apply to this case
                continue
        if b is None or c is None:
            # after policies, if still missing, skip
            continue
        yield k, float(b), float(c)


def detect_regressions(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    *,
    threshold: float = 0.0,
    direction: Direction = "higher_is_better",
    missing: MissingPolicy = "ignore",
) -> Tuple[Dict[str, Dict[str, Any]], RegressionSummary]:
    """Compare baseline vs current scores, compute deltas, and flag regressions.

    - threshold: minimum absolute change to consider (small changes within threshold are unchanged)
    - direction: if higher_is_better, delta = current - baseline; regression if delta < -threshold
                 if lower_is_better, delta = current - baseline; regression if delta > threshold
    - missing: how to handle missing keys (ignore, current_zero, baseline_zero)
    Returns (per_key_details, summary)
    """
    details: Dict[str, Dict[str, Any]] = {}
    improved = regressed = unchanged = 0
    total = 0
    sum_delta = 0.0
    worst_delta: float | None = None

    eps = 1e-9
    for k, b, c in _iter_pairs(baseline, current, missing=missing):
        d = c - b
        total += 1
        sum_delta += d
        if worst_delta is None:
            worst_delta = d
        else:
            # For worst we record most negative for higher-is-better; most positive for lower-is-better
            if direction == "higher_is_better":
                worst_delta = min(worst_delta, d)
            else:
                worst_delta = max(worst_delta, d)

        # Threshold with numeric tolerance: treat values within threshold +/- eps as unchanged
        if direction == "higher_is_better":
            is_imp = d > (threshold + eps)
            is_reg = d < (-(threshold + eps))
        else:  # lower_is_better
            is_imp = d < (-(threshold + eps))
            is_reg = d > (threshold + eps)
        is_unch = not (is_reg or is_imp)
        regressed += 1 if is_reg else 0
        improved += 1 if is_imp else 0
        unchanged += 1 if is_unch else 0

        details[k] = {
            "baseline": b,
            "current": c,
            "delta": d,
            "regressed": is_reg,
            "improved": is_imp,
            "unchanged": is_unch,
        }

    mean_delta = (sum_delta / total) if total > 0 else 0.0
    summary = RegressionSummary(
        total_compared=total,
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
        mean_delta=mean_delta,
        worst_delta=worst_delta,
        threshold=float(threshold),
        direction=direction,
    )
    return details, summary


def score_regressions(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    *,
    threshold: float = 0.0,
    direction: Direction = "higher_is_better",
    missing: MissingPolicy = "ignore",
) -> Tuple[float, Dict[str, Any]]:
    """Return a regression score in [0,1] and accompanying details.

    Score = 1 - (regressed / max(1, total_compared)). 1.0 indicates no regressions.
    """
    details, summary = detect_regressions(
        baseline,
        current,
        threshold=threshold,
        direction=direction,
        missing=missing,
    )
    denom = max(1, summary.total_compared)
    score = max(0.0, 1.0 - (summary.regressed / denom))
    return score, {
        "details": details,
        "summary": {
            "total_compared": summary.total_compared,
            "improved": summary.improved,
            "regressed": summary.regressed,
            "unchanged": summary.unchanged,
            "mean_delta": summary.mean_delta,
            "worst_delta": summary.worst_delta,
            "threshold": summary.threshold,
            "direction": summary.direction,
        },
    }


__all__ = ["detect_regressions", "score_regressions", "RegressionSummary"]
