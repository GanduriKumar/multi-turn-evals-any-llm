from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from eval_server.data.versioning import (
    SemVer,
    compare_versions,
    is_version_greater,
    parse_version,
    versioned_name,
)


def test_parse_and_compare_versions():
    v = parse_version("1.2.3")
    assert isinstance(v, SemVer)
    assert str(v) == "1.2.3"
    assert compare_versions("1.0.0", "0.9.5") == 1
    assert compare_versions("1.0.0", "1.0.0") == 0
    assert compare_versions("1.0.0", "1.0.1") == -1
    assert is_version_greater("2.0.0", "1.9.9")
    assert versioned_name("results", "1.0.0") == "results-1.0.0"


def test_invalid_versions():
    with pytest.raises(ValueError):
        parse_version("")
    with pytest.raises(ValueError):
        parse_version("1.2")
    with pytest.raises(ValueError):
        parse_version("1.2.x")
    with pytest.raises(ValueError):
        versioned_name("results", "1.2.x")


def test_schemas_require_version():
    repo = Path(__file__).resolve().parents[2]
    schemas_dir = repo / "configs" / "schemas"
    conv_schema = json.loads((schemas_dir / "conversation.schema.json").read_text(encoding="utf-8"))
    golden_schema = json.loads((schemas_dir / "golden.schema.json").read_text(encoding="utf-8"))

    conv_validator = Draft202012Validator(conv_schema)
    golden_validator = Draft202012Validator(golden_schema)

    # Missing version should fail
    bad_conv = {"conversation_id": "id", "metadata": {}, "turns": [{"turn_id": "t", "role": "user", "content": "hi"}]}
    with pytest.raises(Exception):
        conv_validator.validate(bad_conv)

    # Malformed version should fail
    bad_golden = {"version": "1.2", "conversation_id": "id", "expectations": [{"turn_id": "t", "expected": {"text_variants": ["x"]}}]}
    with pytest.raises(Exception):
        golden_validator.validate(bad_golden)
