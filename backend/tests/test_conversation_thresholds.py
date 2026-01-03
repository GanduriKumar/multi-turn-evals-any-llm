from __future__ import annotations

from eval_server.scoring.thresholds import ThresholdPolicy, evaluate_conversation_thresholds


def test_conversation_thresholds_all():
    agg = {"correctness": 0.82, "safety": 0.95, "adherence": 0.76}
    thr = {"correctness": 0.8, "safety": 0.9, "adherence": 0.75}
    passed, details = evaluate_conversation_thresholds(agg, thr, policy=ThresholdPolicy(require="all"))
    assert passed is True


def test_conversation_thresholds_any():
    agg = {"correctness": 0.45, "safety": 0.4, "adherence": 0.9}
    thr = {"correctness": 0.6, "safety": 0.6, "adherence": 0.85}
    passed, details = evaluate_conversation_thresholds(agg, thr, policy=ThresholdPolicy(require="any"))
    assert passed is True


def test_conversation_thresholds_weighted():
    agg = {"correctness": 0.8, "safety": 0.7, "adherence": 0.9}
    thr = {"correctness": 0.8, "safety": 0.75, "adherence": 0.85}
    weights = {"correctness": 2.0, "safety": 1.0, "adherence": 1.0}
    # correctness/adherence pass, safety fails: weights 2 + 1 over total 4 => 0.75; threshold 0.7 -> pass
    passed, details = evaluate_conversation_thresholds(agg, thr, weights=weights, policy=ThresholdPolicy(require="weighted", pass_ratio=0.7))
    assert passed is True


def test_conversation_thresholds_default_threshold():
    agg = {"correctness": 0.6}
    # No explicit threshold, default 0.5 -> pass
    passed, _ = evaluate_conversation_thresholds(agg, None, policy=ThresholdPolicy(default_threshold=0.5))
    assert passed is True
