from __future__ import annotations

from eval_server.evaluation.structured import compare_decision_bundle, compare_structured


def test_compare_structured_exact_match_and_mismatch():
    expected = {"decision": "ALLOW", "reason_code": "OK", "next_action": "ship"}
    actual_ok = {"decision": "ALLOW", "reason_code": "OK", "next_action": "ship", "extra": 1}
    actual_bad = {"decision": "DENY", "reason_code": "OK"}

    ok, details = compare_structured(actual_ok, expected)
    assert ok is True
    assert details["mismatches"] == {}

    bad, details2 = compare_structured(actual_bad, expected)
    assert bad is False
    assert set(details2["mismatches"].keys()) == {"decision", "next_action"}


def test_compare_decision_bundle_wrapper():
    expected = {"decision": "DENY", "reason_code": "RISK", "next_action": "manual_review"}
    actual = {"decision": "deny", "reason_code": "risk", "next_action": "MANUAL_REVIEW", "note": "..."}
    ok, details = compare_decision_bundle(actual, expected)
    assert ok is True
    assert details["mismatches"] == {}
