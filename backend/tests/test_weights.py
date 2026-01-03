from __future__ import annotations

from eval_server.evaluation.weights import aggregate_conversation, weighted_average


def test_weighted_average_metrics():
    scores = {"correctness": 0.9, "consistency": 0.6, "adherence": 0.8, "hallucination": 1.0}
    weights = {"correctness": 0.6, "consistency": 0.2, "adherence": 0.15, "hallucination": 0.05}
    out = weighted_average(scores, weights)
    # Manual: 0.9*0.6 + 0.6*0.2 + 0.8*0.15 + 1.0*0.05 = 0.54 + 0.12 + 0.12 + 0.05 = 0.83
    assert abs(out - 0.83) < 1e-9

    # No weights -> mean
    assert abs(weighted_average(scores) - sum(scores.values())/len(scores)) < 1e-9


def test_aggregate_conversation_turn_weights():
    turn_scores = {"t1": 0.9, "t2": 0.6, "t3": 1.0}
    # Higher weight for t1, lower for t2, default 1.0 for t3
    turn_weights = {"t1": 2.0, "t2": 0.5}
    out = aggregate_conversation(turn_scores, turn_weights)
    # (0.9*2 + 0.6*0.5 + 1.0*1) / (2 + 0.5 + 1) = (1.8 + 0.3 + 1) / 3.5 = 3.1/3.5 ≈ 0.8857
    assert abs(out - (3.1/3.5)) < 1e-9
