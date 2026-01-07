import json
from pathlib import Path

from schemas import SchemaValidator


def test_dataset_schema_valid():
    sv = SchemaValidator()
    sample = {
        "dataset_id": "commerce_sample",
        "version": "1.0.0",
        "metadata": {"domain": "commerce", "difficulty": "easy", "tags": ["sample"]},
        "conversations": [
            {
                "conversation_id": "conv1",
                "turns": [
                    {"role": "user", "text": "Where is my order?"},
                    {"role": "assistant", "text": "Please share order ID."}
                ]
            }
        ]
    }
    errs = sv.validate("dataset", sample)
    assert errs == []


def test_golden_schema_valid():
    sv = SchemaValidator()
    sample = {
        "dataset_id": "commerce_sample",
        "version": "1.0.0",
        "entries": [
            {
                "conversation_id": "conv1",
                "turns": [
                    {"turn_index": 1, "expected": {"variants": ["Please share the order ID."]}}
                ],
                "final_outcome": {"decision": "ALLOW", "next_action": "confirm_order"},
                "constraints": {"refund_after_ship": False}
            }
        ]
    }
    errs = sv.validate("golden", sample)
    assert errs == []


def test_run_config_schema_valid():
    sv = SchemaValidator()
    sample = {
        "run_id": "run-001",
        "datasets": ["commerce_sample"],
        "models": ["ollama:llama3.2:2b"],
        "metrics": ["exact", "semantic"],
        "thresholds": {"semantic": 0.8},
        "concurrency": 1
    }
    errs = sv.validate("run_config", sample)
    assert errs == []
