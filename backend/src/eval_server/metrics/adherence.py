from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..scoring.constraints import ConstraintResult, evaluate_constraints
from ..scoring.normalizer import canonicalize_text


@dataclass(frozen=True)
class TurnAdherence:
    turn_index: int
    passed: bool
    passed_count: int
    total_constraints: int
    results: List[ConstraintResult]


def _build_context(turn: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a context dict for constraint evaluation from a turn.

    - Includes raw text as 'text'
    - Flattens structured fields at top-level for easy reference in expressions
    """
    ctx: Dict[str, Any] = {}
    ctx["text"] = canonicalize_text(str(turn.get("text", "")))
    structured = turn.get("structured", {}) or {}
    if isinstance(structured, Mapping):
        for k, v in structured.items():
            ctx[str(k)] = v
    return ctx


def evaluate_turn_adherence(
    turn: Mapping[str, Any],
    constraints: Optional[List[Mapping[str, Any]]],
) -> TurnAdherence:
    """Evaluate a single turn against a list of constraints.

    Returns a TurnAdherence with per-constraint results.
    """
    cons = constraints or []
    if not cons:
        # No constraints -> fully adherent by definition
        return TurnAdherence(turn_index=-1, passed=True, passed_count=0, total_constraints=0, results=[])
    ctx = _build_context(turn)
    all_ok, results = evaluate_constraints(cons, ctx)
    passed_count = sum(1 for r in results if r.passed)
    return TurnAdherence(
        turn_index=-1,
        passed=all_ok,
        passed_count=passed_count,
        total_constraints=len(results),
        results=results,
    )


def score_adherence(
    turns: List[Mapping[str, Any]],
    *,
    constraints: Optional[List[Mapping[str, Any]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute adherence score across turns and return details.

    - Per-turn score is (#passed / total) when constraints exist, else 1.0.
    - Overall score is the mean of per-turn scores.
    Details include per-turn breakdown and total violation count.
    """
    if not turns:
        return 1.0, {"turns": [], "violations": 0}

    cons = constraints or []
    per_turn: List[Dict[str, Any]] = []
    scores: List[float] = []
    total_violations = 0

    for idx, t in enumerate(turns):
        ta = evaluate_turn_adherence(t, cons)
        # Attach index information
        object.__setattr__(ta, "turn_index", idx)  # keep dataclass frozen semantics otherwise
        if ta.total_constraints == 0:
            s = 1.0
        else:
            s = ta.passed_count / ta.total_constraints
        scores.append(s)
        violations = ta.total_constraints - ta.passed_count
        total_violations += max(0, violations)
        per_turn.append(
            {
                "turn_index": idx,
                "score": s,
                "passed": ta.passed,
                "passed_count": ta.passed_count,
                "total_constraints": ta.total_constraints,
                "results": [{"passed": r.passed, "message": r.message} for r in ta.results],
            }
        )

    overall = sum(scores) / len(scores)
    details = {"turns": per_turn, "violations": total_violations}
    return overall, details


__all__ = [
    "TurnAdherence",
    "evaluate_turn_adherence",
    "score_adherence",
]
