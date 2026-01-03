from __future__ import annotations

from eval_server.scoring.thresholds import ThresholdPolicy, evaluate_turn_thresholds


def test_thresholds_all_require_all_metrics():
    scores = {"correctness": 0.9, "structured": 0.6}
    thresholds = {"correctness": 0.8, "structured": 0.6}
    passed, details = evaluate_turn_thresholds(scores, thresholds, policy=ThresholdPolicy(require="all"))
    assert passed is True
    thresholds = {"correctness": 0.95}
    passed, details = evaluate_turn_thresholds(scores, thresholds, policy=ThresholdPolicy(require="all", default_threshold=0.7))
    assert passed is False


def test_thresholds_any_policy():
    scores = {"correctness": 0.6, "structured": 0.4}
    thresholds = {"correctness": 0.7, "structured": 0.5}
    passed, details = evaluate_turn_thresholds(scores, thresholds, policy=ThresholdPolicy(require="any"))
    assert passed is False
    # increase correctness so at least one passes
    scores["correctness"] = 0.8
    passed, details = evaluate_turn_thresholds(scores, thresholds, policy=ThresholdPolicy(require="any"))
    assert passed is True


def test_thresholds_weighted_policy_with_weights():
    scores = {"correctness": 0.7, "structured": 0.49, "safety": 0.8}
    thresholds = {"correctness": 0.7, "structured": 0.5, "safety": 0.75}
    weights = {"correctness": 2.0, "structured": 1.0, "safety": 1.0}
    # correctness and safety pass (weights 2 + 1); structured fails (1) -> pass fraction 3/4 = 0.75
    passed, details = evaluate_turn_thresholds(scores, thresholds, weights=weights, policy=ThresholdPolicy(require="weighted", pass_ratio=0.75))
    assert passed is True
    # increase pass_ratio to 0.8 -> should fail
    passed, details = evaluate_turn_thresholds(scores, thresholds, weights=weights, policy=ThresholdPolicy(require="weighted", pass_ratio=0.8))
    assert passed is False


def test_thresholds_empty_metrics_fail_safe():
    passed, details = evaluate_turn_thresholds({}, {"x": 0.5})
    assert passed is False
