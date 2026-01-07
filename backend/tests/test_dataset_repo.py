from pathlib import Path
import json
import tempfile

from dataset_repo import DatasetRepository


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_repo_listing_and_gets():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # valid dataset
        ds = {
            "dataset_id": "commerce_sample",
            "version": "1.0.0",
            "metadata": {"domain": "commerce", "difficulty": "easy"},
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
        write_json(root / "commerce_sample.dataset.json", ds)

        # valid golden
        golden = {
            "dataset_id": "commerce_sample",
            "version": "1.0.0",
            "entries": [
                {
                    "conversation_id": "conv1",
                    "turns": [
                        {"turn_index": 1, "expected": {"variants": ["Please share the order ID."]}}
                    ],
                    "final_outcome": {"decision": "ALLOW"}
                }
            ]
        }
        write_json(root / "commerce_sample.golden.json", golden)

        repo = DatasetRepository(root)
        items = repo.list_datasets()
        assert len(items) == 1
        assert items[0]["dataset_id"] == "commerce_sample"
        assert items[0]["has_golden"] is True

        ds_loaded = repo.get_dataset("commerce_sample")
        assert ds_loaded["version"] == "1.0.0"

        conv = repo.get_conversation("conv1")
        assert conv["dataset_id"] == "commerce_sample"

        gold = repo.get_golden("conv1")
        assert gold["dataset_id"] == "commerce_sample"


def test_invalid_json_and_schema_rejection():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # invalid dataset missing required fields
        write_json(root / "bad.dataset.json", {"foo": 1})
        repo = DatasetRepository(root)
        items = repo.list_datasets()
        assert len(items) == 1
        assert items[0]["valid"] is False
        assert "errors" in items[0]
