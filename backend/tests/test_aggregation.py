from __future__ import annotations

from eval_server.scoring.aggregation import mean_aggregate, min_aggregate, weighted_aggregate, aggregate


def test_mean_aggregate():
    vals = {"t1": 0.5, "t2": 1.0, "t3": 0.5}
    assert abs(mean_aggregate(vals) - (2.0/3.0)) < 1e-9


def test_min_aggregate():
    vals = {"t1": 0.5, "t2": 1.0, "t3": 0.2}
    assert min_aggregate(vals) == 0.2


def test_weighted_aggregate():
    vals = {"t1": 1.0, "t2": 0.0, "t3": 0.5}
    weights = {"t1": 2.0, "t2": 1.0}
    # (1.0*2 + 0.0*1 + 0.5*1) / (2+1+1) = 2.5/4 = 0.625
    assert abs(weighted_aggregate(vals, weights) - 0.625) < 1e-9


def test_dispatcher():
    vals = {"a": 0.8, "b": 0.6}
    w = {"a": 2}
    assert aggregate(vals, method="mean") == sum(vals.values())/2
    assert aggregate(vals, method="min") == 0.6
    assert abs(aggregate(vals, method="weighted", weights=w) - ((0.8*2+0.6*1)/3)) < 1e-9
