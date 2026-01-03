from __future__ import annotations

from typing import Any, List, Mapping

from eval_server.metrics.adherence import score_adherence, evaluate_turn_adherence


def test_adherence_no_constraints_full_score():
    turns: List[Mapping[str, Any]] = [
        {"text": "Hello"},
        {"text": "World"},
    ]
    score, details = score_adherence(turns, constraints=None)
    assert score == 1.0
    assert details["violations"] == 0


def test_adherence_with_expr_constraints():
    constraints: List[Mapping[str, Any]] = [
        {"expr": "amount <= budget", "message": "Budget exceeded"},
        {"expr": "eligibility == True", "message": "Not eligible"},
    ]
    turns: List[Mapping[str, Any]] = [
        {"structured": {"amount": 80, "budget": 100, "eligibility": True}},  # pass both
        {"structured": {"amount": 120, "budget": 100, "eligibility": True}},  # budget fail
        {"structured": {"amount": 50, "budget": 100, "eligibility": False}},  # eligibility fail
    ]
    score, details = score_adherence(turns, constraints=constraints)
    # First: 2/2 -> 1.0 ; second: 1/2 -> 0.5 ; third: 1/2 -> 0.5 ; mean = 2.0/3 = 0.666...
    assert 0.65 < score < 0.68
    assert details["violations"] == 2


def test_adherence_with_rule_constraints_and_text():
    constraints: List[Mapping[str, Any]] = [
        {"rule": "text_must_not_include", "params": {"terms": ["password", "ssn"]}, "message": "Sensitive content"},
        {"rule": "text_must_include", "params": {"terms": ["thank you"]}, "message": "Missing courtesy"},
    ]
    turns: List[Mapping[str, Any]] = [
        {"text": "Thank you for your request."},  # pass both
        {"text": "Here is my password: 123", "structured": {"note": "oops"}},  # includes sensitive, and lacks 'thank you'
    ]
    score, details = score_adherence(turns, constraints=constraints)
    # First: 2/2 -> 1.0 ; Second: 0/2 -> 0.0 ; mean -> 0.5
    assert 0.49 < score < 0.51
    assert details["violations"] == 2


def test_evaluate_turn_adherence_breakdown():
    constraints: List[Mapping[str, Any]] = [
        {"expr": "refund == True"},
        {"expr": "days <= 5"},
    ]
    turn: Mapping[str, Any] = {"structured": {"refund": True, "days": 7}}
    ta = evaluate_turn_adherence(turn, constraints)
    assert ta.total_constraints == 2
    assert ta.passed_count == 1
    assert ta.passed is False
