from __future__ import annotations

from eval_server.metrics.consistency import (
    detect_inconsistencies,
    extract_consistency_state,
    score_consistency,
)


def test_extract_consistency_state_text_signals():
    st = extract_consistency_state({"text": "We will issue refund today."})
    assert st["will_refund"] is True
    st = extract_consistency_state({"text": "We will not issue refund."})
    assert st["will_refund"] is False


def test_detect_contradiction_categorical():
    turns = [
        {"structured": {"decision": "approve", "reason_code": "A1"}},
        {"structured": {"decision": "deny", "reason_code": "A1"}},
    ]
    issues, details = detect_inconsistencies(turns)
    assert any(i.kind == "contradiction" and i.key == "decision" for i in issues)
    # Only decision changed
    assert details["counts"]["contradiction"] >= 1


def test_detect_drift_numeric():
    turns = [
        {"structured": {"days": 3}},
        {"structured": {"days": 5}},
    ]
    issues, details = detect_inconsistencies(turns)
    assert any(i.kind == "drift" and i.key == "days" for i in issues)


def test_constraint_violation():
    turns = [
        {"structured": {"decision": "approve"}},
        {"structured": {"decision": "deny"}},
    ]
    issues, details = detect_inconsistencies(turns, constraints={"decision": "approve"})
    # Second turn violates constraint
    assert any(i.kind == "constraint_violation" and i.turn_index_curr == 1 for i in issues)


def test_score_consistency_penalizes_issues():
    turns = [
        {"structured": {"decision": "approve"}},
        {"structured": {"decision": "deny"}},
        {"structured": {"decision": "deny"}},
    ]
    score, info = score_consistency(turns)
    assert 0.0 <= score < 1.0
    assert len(info["issues"]) >= 1
