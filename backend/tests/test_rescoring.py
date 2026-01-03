from __future__ import annotations

from typing import Any, Dict

from eval_server.scoring.rescoring import rescore_payload


def _make_payload() -> Dict[str, Any]:
    # Minimal consolidated results payload (mirrors results_writer.assemble_results output shape)
    return {
        "run": {"run_id": "test-run"},
        "summary": {},
        "results": [
            {
                "dataset_id": "ds1",
                "conversation_id": "convA",
                "model_name": "modelX",
                "aggregate": {"score": 0.6, "passed": False},
                "turns": [
                    {
                        "turn_id": 1,
                        "prompt": "p1",
                        "response": "r1",
                        "context": [],
                        "canonical": {"raw_text": "foo"},
                        "metrics": {"correctness": 0.0},
                        "weights": {"correctness": 1.0},
                        "weighted_score": 0.0,
                        "passed": False,
                    },
                    {
                        "turn_id": 2,
                        "prompt": "p2",
                        "response": "r2",
                        "context": [],
                        "canonical": {"raw_text": "bar"},
                        "metrics": {"correctness": 1.0},
                        "weights": {"correctness": 1.0},
                        "weighted_score": 1.0,
                        "passed": True,
                    },
                ],
            }
        ],
    }


def test_rescoring_with_list_annotations_updates_final_fields_only():
    payload = _make_payload()
    # Override first turn to pass with score 0.9
    anns = {
        "annotations": [
            {
                "dataset_id": "ds1",
                "conversation_id": "convA",
                "model_name": "modelX",
                "turn_id": 1,
                "rating": 4.5,
                "notes": "manual pass",
                "override_pass": True,
                "override_score": 0.9,
            }
        ]
    }

    updated = rescore_payload(payload, annotations=anns)

    group = updated["results"][0]
    t1 = group["turns"][0]
    t2 = group["turns"][1]

    # Original auto-scores remain
    assert t1["passed"] is False
    assert t1["weighted_score"] == 0.0
    # Final fields reflect overrides
    assert t1["final_pass"] is True
    assert abs(t1["final_weighted_score"] - 0.9) < 1e-9
    assert (t1["evaluator"] or {}).get("override_pass") is True

    # Unannotated turn should mirror originals in final fields
    assert t2["final_pass"] is True
    assert abs(t2["final_weighted_score"] - 1.0) < 1e-9

    # Conversation aggregate_final should exist and use final fields
    agg_final = group.get("aggregate_final") or {}
    assert "avg_turn_score" in agg_final
    expected_avg = (0.9 + 1.0) / 2.0
    assert abs(agg_final["avg_turn_score"] - expected_avg) < 1e-9
    # passed_all considers final_pass
    assert agg_final["passed_all"] is True


def test_rescoring_with_keyed_annotations():
    payload = _make_payload()
    # Use keyed form, flip second turn to fail and lower its score
    key = "ds1|convA|modelX|2"
    anns = {
        key: {
            "rating": 2,
            "notes": "manual fail",
            "override_pass": False,
            "override_score": 0.2,
        }
    }

    updated = rescore_payload(payload, annotations=anns)

    group = updated["results"][0]
    t1 = group["turns"][0]
    t2 = group["turns"][1]

    # Turn1 unaffected
    assert t1["final_pass"] is False
    assert abs(t1["final_weighted_score"] - 0.0) < 1e-9

    # Turn2 overridden
    assert t2["passed"] is True
    assert abs(t2["weighted_score"] - 1.0) < 1e-9
    assert t2["final_pass"] is False
    assert abs(t2["final_weighted_score"] - 0.2) < 1e-9

    agg_final = group.get("aggregate_final") or {}
    expected_avg = (0.0 + 0.2) / 2.0
    assert abs(agg_final["avg_turn_score"] - expected_avg) < 1e-9
    assert agg_final["passed_all"] is False
