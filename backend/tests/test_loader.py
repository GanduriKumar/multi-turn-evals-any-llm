from __future__ import annotations

from pathlib import Path

import pytest

from eval_server.data.loader import load_conversation, load_golden


def test_load_json_and_yaml_examples():
    repo = Path(__file__).resolve().parents[2]
    examples = repo / "configs" / "datasets" / "examples"

    conv = load_conversation(examples / "conversation_001.json")
    assert conv["conversation_id"] == "conv-001"
    assert conv["turns"][1]["tool_calls"][0]["tool_name"] == "doc_search"

    golden_yaml = load_golden(examples / "conversation_001.golden.yaml")
    golden_json = load_golden(examples / "conversation_001.golden.json")
    for golden in (golden_yaml, golden_json):
        assert golden["conversation_id"] == "conv-001"
        assert golden["expectations"][0]["expected"]["text_variants"]


def test_invalid_files_raise_clear_errors(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ not json }", encoding="utf-8")

    with pytest.raises(ValueError) as ei:
        load_conversation(bad_json)
    assert "Invalid JSON" in str(ei.value)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(": invalid yaml", encoding="utf-8")
    with pytest.raises(ValueError) as ei2:
        load_golden(bad_yaml)
    assert "Invalid YAML" in str(ei2.value)

    # Unsupported extension
    bad_txt = tmp_path / "file.txt"
    bad_txt.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError) as ei3:
        load_conversation(bad_txt)
    assert "Unsupported file extension" in str(ei3.value)
