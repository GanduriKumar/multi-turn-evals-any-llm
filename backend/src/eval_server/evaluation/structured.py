from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


def _equals_ci(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    return a == b


def compare_structured(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Compare structured outputs.

    Checks for equality of explicit keys present in expected. Extra keys in actual are ignored.
    Returns (passed, details) where details includes mismatches dict.
    """
    mismatches: Dict[str, Any] = {}
    for key, exp_val in expected.items():
        act_val = actual.get(key)
        if not _equals_ci(act_val, exp_val):
            mismatches[key] = {"expected": exp_val, "actual": act_val}
    return (len(mismatches) == 0, {"mismatches": mismatches})


def compare_decision_bundle(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Specialized comparison for decision, reason_code, next_action fields.

    This is a thin wrapper around compare_structured, but kept for clarity and future extensibility.
    """
    keys = [k for k in ("decision", "reason_code", "next_action") if k in expected]
    exp_subset = {k: expected[k] for k in keys}
    act_subset = {k: actual.get(k) for k in keys}
    return compare_structured(act_subset, exp_subset)


__all__ = ["compare_structured", "compare_decision_bundle"]
