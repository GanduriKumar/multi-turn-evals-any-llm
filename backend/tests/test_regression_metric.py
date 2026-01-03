from __future__ import annotations

from eval_server.metrics.regression import detect_regressions, score_regressions


def test_regressions_higher_is_better_with_threshold():
    baseline = {"a": 0.8, "b": 0.7, "c": 0.5}
    current = {"a": 0.78, "b": 0.75, "c": 0.3}
    details, summary = detect_regressions(baseline, current, threshold=0.02, direction="higher_is_better")
    # a: delta -0.02 -> unchanged (meets threshold); b: +0.05 -> improved; c: -0.2 -> regressed
    assert details["a"]["unchanged"] is True
    assert details["b"]["improved"] is True
    assert details["c"]["regressed"] is True
    assert summary.regressed == 1


def test_regressions_lower_is_better():
    baseline = {"latency": 200.0, "errors": 5.0}
    current = {"latency": 180.0, "errors": 8.0}
    details, summary = detect_regressions(baseline, current, threshold=10.0, direction="lower_is_better")
    # latency: -20 -> improved (lower is better); errors: +3 -> unchanged (below threshold 10)
    assert details["latency"]["improved"] is True
    assert details["errors"]["unchanged"] is True
    assert summary.improved == 1


def test_regression_score_normalization():
    baseline = {"a": 1.0, "b": 1.0, "c": 1.0}
    # Choose values so only 'a' is a regression beyond threshold 0.05
    # b: -0.03 (unchanged), c: -0.05 (boundary -> unchanged)
    current = {"a": 0.0, "b": 0.97, "c": 0.95}
    score, info = score_regressions(baseline, current, threshold=0.05)
    # Only 'a' is a regression beyond threshold -> 1/3 => score = 1 - 1/3
    assert 0.65 < score < 0.68


def test_missing_policy_current_zero():
    baseline = {"a": 1.0, "b": 1.0}
    current = {"a": 1.0}
    details, summary = detect_regressions(baseline, current, missing="current_zero")
    # 'b' missing in current -> treat as current=0.0 -> regression
    assert details["b"]["regressed"] is True


def test_missing_policy_baseline_zero():
    baseline = {"a": 1.0}
    current = {"a": 1.0, "b": 0.1}
    details, summary = detect_regressions(baseline, current, missing="baseline_zero")
    # 'b' missing in baseline -> treat as baseline=0.0 -> improved
    assert details["b"]["improved"] is True
