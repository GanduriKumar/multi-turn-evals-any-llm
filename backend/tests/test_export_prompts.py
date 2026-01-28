import json
import tempfile
from pathlib import Path

import pytest

from backend.export_prompts import export_prompts_to_csv


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture
def temp_datasets_dir():
    """Create a temporary datasets directory with sample data."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        vertical_dir = root / "datasets" / "commerce"
        vertical_dir.mkdir(parents=True, exist_ok=True)
        yield vertical_dir


def test_export_single_dataset_with_golden(temp_datasets_dir):
    """Test exporting a single dataset with golden data."""
    # Create dataset
    dataset = {
        "dataset_id": "test-dataset-1",
        "version": "1.0.0",
        "metadata": {"domain": "Orders & Returns", "difficulty": "easy"},
        "conversations": [
            {
                "conversation_id": "ord-001",
                "title": "Refund request for damaged item",
                "metadata": {"behavior": "Refund/Exchange/Cancellation"},
                "turns": [
                    {"role": "user", "text": "My order arrived damaged, I want a refund."},
                    {"role": "assistant", "text": "I'm sorry to hear that. Can you share your order ID?"},
                    {"role": "user", "text": "Order #A123, item: headphones, price $79."},
                    {"role": "assistant", "text": "I'll process a refund."}
                ]
            }
        ]
    }
    write_json(temp_datasets_dir / "test-dataset-1.dataset.json", dataset)
    
    # Create golden
    golden = {
        "dataset_id": "test-dataset-1",
        "version": "1.0.0",
        "entries": [
            {
                "conversation_id": "ord-001",
                "turns": [
                    {
                        "turn_index": 1,
                        "expected": {
                            "variants": [
                                "I'm sorry to hear that. Can you share your order ID?",
                                "Please provide your order ID so I can help."
                            ]
                        }
                    },
                    {
                        "turn_index": 3,
                        "expected": {"variants": ["I'll process a refund."]}
                    }
                ],
                "final_outcome": {"decision": "ALLOW", "next_action": "issue_refund"}
            }
        ]
    }
    write_json(temp_datasets_dir / "test-dataset-1.golden.json", golden)
    
    # Export
    csv_content = export_prompts_to_csv("commerce", "test-dataset-1")
    
    # Verify
    lines = csv_content.strip().split("\n")
    assert len(lines) == 5  # header + 4 turns
    
    # Check header
    header = lines[0]
    assert "dataset_id" in header
    assert "conversation_id" in header
    assert "conversation_title" in header
    assert "domain" in header
    assert "behavior" in header
    assert "turn_index" in header
    assert "role" in header
    assert "prompt_text" in header
    assert "expected_variants" in header
    assert "final_decision" in header
    assert "final_next_action" in header
    
    # Check first turn (user)
    assert "test-dataset-1" in lines[1]
    assert "ord-001" in lines[1]
    assert "0,user" in lines[1]
    assert "My order arrived damaged" in lines[1]
    assert "ALLOW" in lines[1]
    assert "issue_refund" in lines[1]
    
    # Check second turn (assistant with golden)
    assert "1,assistant" in lines[2]
    assert "I'm sorry to hear that. Can you share your order ID?|Please provide your order ID so I can help." in lines[2]
    
    # Check fourth turn (assistant with golden)
    assert "3,assistant" in lines[4]
    assert "I'll process a refund." in lines[4]


def test_export_all_datasets_in_vertical(temp_datasets_dir):
    """Test exporting all datasets in a vertical."""
    # Create two datasets
    dataset1 = {
        "dataset_id": "test-ds-1",
        "version": "1.0.0",
        "metadata": {"domain": "Domain A", "difficulty": "easy"},
        "conversations": [
            {
                "conversation_id": "conv-1",
                "title": "Test 1",
                "turns": [
                    {"role": "user", "text": "Hello"},
                    {"role": "assistant", "text": "Hi there"}
                ]
            }
        ]
    }
    write_json(temp_datasets_dir / "test-ds-1.dataset.json", dataset1)
    
    dataset2 = {
        "dataset_id": "test-ds-2",
        "version": "1.0.0",
        "metadata": {"domain": "Domain B", "difficulty": "medium"},
        "conversations": [
            {
                "conversation_id": "conv-2",
                "title": "Test 2",
                "turns": [
                    {"role": "user", "text": "Goodbye"},
                    {"role": "assistant", "text": "See you"}
                ]
            }
        ]
    }
    write_json(temp_datasets_dir / "test-ds-2.dataset.json", dataset2)
    
    # Export all
    csv_content = export_prompts_to_csv("commerce", None)
    
    # Verify
    lines = csv_content.strip().split("\n")
    assert len(lines) == 5  # header + 2 conversations * 2 turns each
    
    # Check that both datasets are included
    csv_text = "\n".join(lines)
    assert "test-ds-1" in csv_text
    assert "test-ds-2" in csv_text
    assert "conv-1" in csv_text
    assert "conv-2" in csv_text


def test_export_missing_golden_data(temp_datasets_dir):
    """Test exporting dataset without golden data."""
    # Create dataset without golden
    dataset = {
        "dataset_id": "no-golden",
        "version": "1.0.0",
        "metadata": {"domain": "Test Domain", "difficulty": "easy"},
        "conversations": [
            {
                "conversation_id": "conv-x",
                "title": "No Golden",
                "turns": [
                    {"role": "user", "text": "Question"},
                    {"role": "assistant", "text": "Answer"}
                ]
            }
        ]
    }
    write_json(temp_datasets_dir / "no-golden.dataset.json", dataset)
    
    # Export should work without errors
    csv_content = export_prompts_to_csv("commerce", "no-golden")
    
    # Verify
    lines = csv_content.strip().split("\n")
    assert len(lines) == 3  # header + 2 turns
    
    # Check that expected_variants is empty
    assert lines[1].count('""') >= 2  # Should have empty expected_variants
    assert lines[2].count('""') >= 2


def test_csv_format_correctness(temp_datasets_dir):
    """Test that CSV is properly formatted."""
    # Create dataset with special characters
    dataset = {
        "dataset_id": "special-chars",
        "version": "1.0.0",
        "metadata": {"domain": "Test", "difficulty": "easy"},
        "conversations": [
            {
                "conversation_id": "conv-special",
                "title": 'Title with "quotes" and, commas',
                "turns": [
                    {"role": "user", "text": 'Text with "quotes", commas, and\nnewlines'},
                    {"role": "assistant", "text": "Simple response"}
                ]
            }
        ]
    }
    write_json(temp_datasets_dir / "special-chars.dataset.json", dataset)
    
    # Export
    csv_content = export_prompts_to_csv("commerce", "special-chars")
    
    # Verify it can be parsed
    import csv
    from io import StringIO
    reader = csv.reader(StringIO(csv_content))
    rows = list(reader)
    
    assert len(rows) == 3  # header + 2 turns
    
    # Check that special characters are properly escaped
    assert rows[1][2] == 'Title with "quotes" and, commas'
    assert 'quotes' in rows[1][7]  # prompt_text column


def test_deterministic_output(temp_datasets_dir):
    """Test that output is sorted deterministically."""
    # Create multiple datasets with different IDs
    for i in range(3):
        dataset = {
            "dataset_id": f"dataset-{3-i}",  # Reverse order
            "version": "1.0.0",
            "metadata": {"domain": "Test", "difficulty": "easy"},
            "conversations": [
                {
                    "conversation_id": f"conv-{3-i}",
                    "title": f"Conv {3-i}",
                    "turns": [
                        {"role": "user", "text": f"Message {3-i}"}
                    ]
                }
            ]
        }
        write_json(temp_datasets_dir / f"dataset-{3-i}.dataset.json", dataset)
    
    # Export multiple times
    csv1 = export_prompts_to_csv("commerce", None)
    csv2 = export_prompts_to_csv("commerce", None)
    
    # Should be identical
    assert csv1 == csv2
    
    # Check sorting (dataset-1 should come before dataset-2, etc.)
    lines = csv1.strip().split("\n")
    dataset_ids = [line.split(",")[0] for line in lines[1:]]
    assert dataset_ids == sorted(dataset_ids)


def test_golden_data_joining(temp_datasets_dir):
    """Test that golden data is correctly joined by turn_index."""
    # Create dataset
    dataset = {
        "dataset_id": "join-test",
        "version": "1.0.0",
        "metadata": {"domain": "Test", "difficulty": "easy"},
        "conversations": [
            {
                "conversation_id": "conv-join",
                "title": "Join Test",
                "turns": [
                    {"role": "user", "text": "Turn 0"},
                    {"role": "assistant", "text": "Turn 1"},
                    {"role": "user", "text": "Turn 2"},
                    {"role": "assistant", "text": "Turn 3"},
                ]
            }
        ]
    }
    write_json(temp_datasets_dir / "join-test.dataset.json", dataset)
    
    # Create golden with non-sequential turn indices
    golden = {
        "dataset_id": "join-test",
        "version": "1.0.0",
        "entries": [
            {
                "conversation_id": "conv-join",
                "turns": [
                    {"turn_index": 1, "expected": {"variants": ["Variant for turn 1"]}},
                    {"turn_index": 3, "expected": {"variants": ["Variant for turn 3"]}},
                ],
                "final_outcome": {"decision": "ALLOW"}
            }
        ]
    }
    write_json(temp_datasets_dir / "join-test.golden.json", golden)
    
    # Export
    csv_content = export_prompts_to_csv("commerce", "join-test")
    
    # Verify
    lines = csv_content.strip().split("\n")
    
    # Turn 0 should have no variants
    assert '""' in lines[1] or ',,' in lines[1]
    
    # Turn 1 should have variant
    assert "Variant for turn 1" in lines[2]
    
    # Turn 2 should have no variants
    assert '""' in lines[3] or ',,' in lines[3]
    
    # Turn 3 should have variant
    assert "Variant for turn 3" in lines[4]


def test_multiple_conversations_same_dataset(temp_datasets_dir):
    """Test exporting dataset with multiple conversations."""
    # Create dataset with multiple conversations
    dataset = {
        "dataset_id": "multi-conv",
        "version": "1.0.0",
        "metadata": {"domain": "Test Domain", "difficulty": "easy"},
        "conversations": [
            {
                "conversation_id": "conv-a",
                "title": "Conversation A",
                "metadata": {"behavior": "Behavior A"},
                "turns": [
                    {"role": "user", "text": "Question A"}
                ]
            },
            {
                "conversation_id": "conv-b",
                "title": "Conversation B",
                "metadata": {"behavior": "Behavior B"},
                "turns": [
                    {"role": "user", "text": "Question B"}
                ]
            }
        ]
    }
    write_json(temp_datasets_dir / "multi-conv.dataset.json", dataset)
    
    # Export
    csv_content = export_prompts_to_csv("commerce", "multi-conv")
    
    # Verify
    lines = csv_content.strip().split("\n")
    assert len(lines) == 3  # header + 2 conversations
    
    # Check both conversations are present
    csv_text = "\n".join(lines)
    assert "conv-a" in csv_text
    assert "conv-b" in csv_text
    assert "Behavior A" in csv_text
    assert "Behavior B" in csv_text


def test_empty_vertical(temp_datasets_dir):
    """Test exporting from empty vertical."""
    # Don't create any files
    csv_content = export_prompts_to_csv("commerce", None)
    
    # Should return CSV with just header
    lines = csv_content.strip().split("\n")
    assert len(lines) == 1  # Only header
    assert "dataset_id" in lines[0]


def test_nonexistent_dataset(temp_datasets_dir):
    """Test exporting nonexistent dataset."""
    # Export non-existent dataset should return empty CSV with header
    csv_content = export_prompts_to_csv("commerce", "nonexistent")
    
    lines = csv_content.strip().split("\n")
    assert len(lines) == 1  # Only header
