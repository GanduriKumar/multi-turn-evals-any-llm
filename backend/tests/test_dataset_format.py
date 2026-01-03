from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_examples_validate_against_schemas():
    repo = Path(__file__).resolve().parents[2]
    schemas_dir = repo / "configs" / "schemas"
    examples_dir = repo / "configs" / "datasets" / "examples"

    conv_schema = load_json(schemas_dir / "conversation.schema.json")
    golden_schema = load_json(schemas_dir / "golden.schema.json")

    conv = load_json(examples_dir / "conversation_001.json")
    golden_yaml = load_yaml(examples_dir / "conversation_001.golden.yaml")
    golden_json = load_json(examples_dir / "conversation_001.golden.json")

    conv_validator = Draft202012Validator(conv_schema)
    golden_validator = Draft202012Validator(golden_schema)

    conv_validator.validate(conv)
    golden_validator.validate(golden_yaml)
    golden_validator.validate(golden_json)


def test_invalid_examples_fail_validation(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    schemas_dir = repo / "configs" / "schemas"

    conv_schema = load_json(schemas_dir / "conversation.schema.json")
    golden_schema = load_json(schemas_dir / "golden.schema.json")
    conv_validator = Draft202012Validator(conv_schema)
    golden_validator = Draft202012Validator(golden_schema)

    # Conversation missing required field
    bad_conv = {
        # "conversation_id": "missing",
        "metadata": {},
        "turns": [
            {"turn_id": "t1", "role": "user", "content": "Hi"}
        ]
    }

    with pytest.raises(Exception):
        conv_validator.validate(bad_conv)

    # Golden with extra field and missing required in expected
    bad_golden = {
        "conversation_id": "conv-001",
        "extra": 1,
        "expectations": [
            {
                "turn_id": "t1",
                "expected": {
                    # missing text_variants
                    "weights": {"correctness": 1}
                }
            }
        ]
    }

    with pytest.raises(Exception):
        golden_validator.validate(bad_golden)
