from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import re

from ..scoring.normalizer import canonicalize_text


@dataclass(frozen=True)
class Inconsistency:
    kind: str  # "contradiction" | "drift" | "constraint_violation"
    key: str
    turn_index_prev: Optional[int]
    turn_index_curr: int
    prev_value: Optional[Any]
    curr_value: Any


def _extract_boolean_signals(text: str) -> Dict[str, Optional[bool]]:
    """Extract simple boolean intent signals from text.

    Signals:
      - will_refund: True/False if confidently stated, else None
      - can_help: True/False if explicitly stated, else None
    """
    t = canonicalize_text(text)
    signals: Dict[str, Optional[bool]] = {"will_refund": None, "can_help": None}

    # Refund intent patterns
    neg_refund = re.search(r"\b(will\s+not|won't|cannot|can't|no)\s+(issue\s+)?refund\b", t)
    pos_refund = re.search(r"\b(will|shall|can)\s+(issue\s+)?refund\b", t)
    if neg_refund:
        signals["will_refund"] = False
    elif pos_refund:
        signals["will_refund"] = True

    # Can help patterns
    if re.search(r"\b(can\s+help|able\s+to\s+help)\b", t):
        signals["can_help"] = True
    if re.search(r"\b(cannot\s+help|can't\s+help|unable\s+to\s+help|won't\s+help)\b", t):
        signals["can_help"] = False

    return signals


def _norm_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return canonicalize_text(v)
    return canonicalize_text(str(v))


def _equals_ci(a: Any, b: Any) -> bool:
    a1, b1 = _norm_str(a), _norm_str(b)
    return a1 == b1


def extract_consistency_state(turn: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a small state dict with fields used for consistency checks.

    Expected keys in turn:
      - text: str (optional)
      - structured: Mapping with optional keys: decision, reason_code, next_action, days
    """
    text = canonicalize_text(str(turn.get("text", "")))
    structured = dict(turn.get("structured", {}) or {})

    state: Dict[str, Any] = {
        "decision": _norm_str(structured.get("decision")),
        "reason_code": _norm_str(structured.get("reason_code")),
        "next_action": _norm_str(structured.get("next_action")),
        "days": structured.get("days"),
    }

    # merge boolean signals from text
    state.update(_extract_boolean_signals(text))
    return state


def detect_inconsistencies(
    turns: List[Mapping[str, Any]],
    *,
    constraints: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Inconsistency], Dict[str, Any]]:
    """Detect contradictions/drift across turns and optional constraint violations.

    Returns (issues, details) where details includes per-type counts.
    """
    states = [extract_consistency_state(t) for t in turns]

    issues: List[Inconsistency] = []
    # Pairwise contradiction/drift checks
    keys_categorical = ["decision", "reason_code", "next_action", "will_refund", "can_help"]
    keys_numeric = ["days"]

    for i in range(1, len(states)):
        prev, curr = states[i - 1], states[i]
        # categorical contradictions
        for k in keys_categorical:
            pv, cv = prev.get(k), curr.get(k)
            if pv is None or cv is None:
                continue
            if not _equals_ci(pv, cv):
                issues.append(
                    Inconsistency(
                        kind="contradiction",
                        key=k,
                        turn_index_prev=i - 1,
                        turn_index_curr=i,
                        prev_value=pv,
                        curr_value=cv,
                    )
                )
        # numeric drift
        for k in keys_numeric:
            pv, cv = prev.get(k), curr.get(k)
            if pv is None or cv is None:
                continue
            try:
                if float(pv) != float(cv):
                    issues.append(
                        Inconsistency(
                            kind="drift",
                            key=k,
                            turn_index_prev=i - 1,
                            turn_index_curr=i,
                            prev_value=pv,
                            curr_value=cv,
                        )
                    )
            except Exception:
                # non-numeric; ignore
                pass

    # Constraint violations: simple invariants, e.g., {"decision": "approve"}
    constraints = constraints or {}
    for i, st in enumerate(states):
        for k, expected in constraints.items():
            actual = st.get(k)
            if actual is None:
                continue
            if not _equals_ci(actual, expected):
                issues.append(
                    Inconsistency(
                        kind="constraint_violation",
                        key=k,
                        turn_index_prev=None,
                        turn_index_curr=i,
                        prev_value=None,
                        curr_value=actual,
                    )
                )

    # Summarize details
    details: Dict[str, Any] = {
        "counts": {
            "contradiction": sum(1 for x in issues if x.kind == "contradiction"),
            "drift": sum(1 for x in issues if x.kind == "drift"),
            "constraint_violation": sum(1 for x in issues if x.kind == "constraint_violation"),
        }
    }
    return issues, details


def score_consistency(
    turns: List[Mapping[str, Any]],
    *,
    constraints: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Return a consistency score in [0,1] and details.

    Scoring: 1.0 when no issues. Otherwise, penalize per detected issue relative to the number
    of transitions and constraints checked.
    """
    issues, details = detect_inconsistencies(turns, constraints=constraints)
    if not turns:
        return 1.0, {"issues": [], **details}
    transitions = max(1, len(turns) - 1)
    constraints_checked = len(constraints or {}) * len(turns)
    denom = transitions + constraints_checked if (transitions + constraints_checked) > 0 else 1
    score = max(0.0, 1.0 - (len(issues) / denom))
    return score, {"issues": [i.__dict__ for i in issues], **details}


__all__ = ["Inconsistency", "extract_consistency_state", "detect_inconsistencies", "score_consistency"]
