from __future__ import annotations

from eval_server.scoring.constraints import evaluate_constraints


def test_constraint_builtin_rule_refund_after_ship():
    constraints = [
        {"rule": "commerce.refund_after_ship", "message": "No refunds after shipment"}
    ]
    # Should pass when not shipped or no refund
    ok1, res1 = evaluate_constraints(constraints, {"shipped": False, "requested_refund": True})
    assert ok1 is True

    # Should fail when shipped and refund requested
    ok2, res2 = evaluate_constraints(constraints, {"shipped": True, "requested_refund": True})
    assert ok2 is False
    assert any(r.message for r in res2)


def test_constraint_expression_eval():
    constraints = [
        {"expr": "total >= 100 and country == 'US'"},
        {"expr": "status in ['paid', 'fulfilled']"},
        {"expr": "not vip"},
    ]
    ok, res = evaluate_constraints(constraints, {"total": 120, "country": "US", "status": "paid", "vip": False})
    assert ok is True

    ok2, res2 = evaluate_constraints(constraints, {"total": 80, "country": "US", "status": "paid", "vip": False})
    assert ok2 is False


def test_additional_builtin_rules():
    # max_discount_percent
    c1 = [{"rule": "commerce.max_discount_percent", "params": {"max": 20}}]
    ok, _ = evaluate_constraints(c1, {"discount_percent": 15})
    assert ok is True
    ok2, _ = evaluate_constraints(c1, {"discount_percent": 25})
    assert ok2 is False

    # require_reason_code_in
    c2 = [{"rule": "commerce.require_reason_code_in", "params": {"allowed": ["OK", "RISK"]}}]
    assert evaluate_constraints(c2, {"reason_code": "OK"})[0] is True
    assert evaluate_constraints(c2, {"reason_code": "BAD"})[0] is False

    # order_total_min
    c3 = [{"rule": "commerce.order_total_min", "params": {"min": 100}}]
    assert evaluate_constraints(c3, {"total": 150})[0] is True
    assert evaluate_constraints(c3, {"total": 80})[0] is False

    # allowed_countries (case-insensitive)
    c4 = [{"rule": "commerce.allowed_countries", "params": {"allowed": ["US", "CA"]}}]
    assert evaluate_constraints(c4, {"country": "us"})[0] is True
    assert evaluate_constraints(c4, {"country": "fr"})[0] is False

    # text includes / not includes
    c5 = [{"rule": "text_must_include", "params": {"terms": ["hello", "world"]}}]
    assert evaluate_constraints(c5, {"text": "Hello there, world!"})[0] is True
    c6 = [{"rule": "text_must_not_include", "params": {"terms": ["error", "fail"]}}]
    assert evaluate_constraints(c6, {"text": "All good."})[0] is True
    assert evaluate_constraints(c6, {"text": "Encountered ERROR."})[0] is False
