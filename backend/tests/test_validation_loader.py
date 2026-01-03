from __future__ import annotations

from pathlib import Path

import pytest

from eval_server.data.validation_loader import (
    assert_valid,
    validate_dir,
    validate_file,
)


def test_validate_file_success_and_failure(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    examples = repo / "configs" / "datasets" / "examples"

    # Success cases
    ok_conv = examples / "conversation_001.json"
    ok_golden_yaml = examples / "conversation_001.golden.yaml"
    ok_golden_json = examples / "conversation_001.golden.json"

    res1 = validate_file(ok_conv, "conversation")
    assert res1["ok"] and not res1["issues"]

    res2 = validate_file(ok_golden_yaml, "golden")
    assert res2["ok"]
    res3 = validate_file(ok_golden_json, "golden")
    assert res3["ok"]

    # Failure JSON syntax
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ not json }", encoding="utf-8")
    res_bad = validate_file(bad_json, "conversation")
    assert not res_bad["ok"]
    assert "Invalid JSON" in res_bad["issues"][0]["message"]

    # Failure schema
    bad_conv = tmp_path / "conv.json"
    bad_conv.write_text('{"metadata": {}, "turns": []}', encoding="utf-8")
    res_bad_schema = validate_file(bad_conv, "conversation")
    assert not res_bad_schema["ok"]
    # Expect at least one issue mentioning missing required "conversation_id"
    messages = "\n".join(i["message"] for i in res_bad_schema["issues"])
    assert "conversation_id" in messages


def test_assert_valid_raises_with_details(tmp_path):
    bad = tmp_path / "g.yaml"
    bad.write_text("conversation_id: id\nexpectations: [{turn_id: t, expected: {}}]", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        assert_valid(bad, "golden")
    s = str(ei.value)
    assert "Validation failed" in s and "text_variants" in s


def test_validate_dir_collects_results(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    examples = repo / "configs" / "datasets" / "examples"
    results = validate_dir(examples, "conversation")
    # Should validate only conversation files (json)
    assert any(r["ok"] for r in results)
