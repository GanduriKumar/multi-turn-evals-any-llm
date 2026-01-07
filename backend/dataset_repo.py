from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

from schemas import SchemaValidator

DEFAULT_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"


class DatasetRepository:
    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir: Path = Path(root_dir) if root_dir else DEFAULT_DATASETS_DIR
        self.sv = SchemaValidator()

    # File conventions: <dataset_id>.dataset.json and <dataset_id>.golden.json
    def _dataset_files(self) -> List[Path]:
        return sorted(self.root_dir.glob("*.dataset.json"))

    def _golden_files(self) -> List[Path]:
        return sorted(self.root_dir.glob("*.golden.json"))

    def _load_json(self, p: Path) -> Dict[str, Any]:
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {p.name}: {e}") from e

    def list_datasets(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        golden_index = {self._load_json(p).get("dataset_id"): p for p in self._golden_files()}
        for p in self._dataset_files():
            data = self._load_json(p)
            errors = self.sv.validate("dataset", data)
            if errors:
                # Skip invalid datasets in listing but annotate error info
                items.append({
                    "dataset_id": data.get("dataset_id") or p.stem.replace(".dataset", ""),
                    "version": data.get("version"),
                    "domain": data.get("metadata", {}).get("domain"),
                    "difficulty": data.get("metadata", {}).get("difficulty"),
                    "conversations": len(data.get("conversations", []) or []),
                    "has_golden": data.get("dataset_id") in golden_index,
                    "valid": False,
                    "errors": errors,
                })
                continue
            items.append({
                "dataset_id": data["dataset_id"],
                "version": data["version"],
                "domain": data["metadata"]["domain"],
                "difficulty": data["metadata"]["difficulty"],
                "conversations": len(data["conversations"]),
                "has_golden": data["dataset_id"] in golden_index,
                "valid": True,
            })
        return items

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        p = self.root_dir / f"{dataset_id}.dataset.json"
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p.name}")
        data = self._load_json(p)
        errors = self.sv.validate("dataset", data)
        if errors:
            raise ValueError("Dataset schema validation failed: " + "; ".join(errors))
        return data

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        found: Optional[Dict[str, Any]] = None
        for p in self._dataset_files():
            data = self._load_json(p)
            if self.sv.validate("dataset", data):
                continue
            for conv in data.get("conversations", []):
                if conv.get("conversation_id") == conversation_id:
                    if found is not None:
                        raise ValueError(
                            f"Conversation ID '{conversation_id}' found in multiple datasets"
                        )
                    found = {
                        "dataset_id": data["dataset_id"],
                        "version": data["version"],
                        "metadata": data.get("metadata", {}),
                        "conversation": conv,
                    }
        if not found:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return found

    def get_golden(self, conversation_id: str) -> Dict[str, Any]:
        found_entry: Optional[Dict[str, Any]] = None
        found_header: Optional[Dict[str, Any]] = None
        for p in self._golden_files():
            golden = self._load_json(p)
            if self.sv.validate("golden", golden):
                continue
            for entry in golden.get("entries", []):
                if entry.get("conversation_id") == conversation_id:
                    if found_entry is not None:
                        raise ValueError(
                            f"Golden for conversation '{conversation_id}' found in multiple golden files"
                        )
                    found_entry = entry
                    found_header = {"dataset_id": golden["dataset_id"], "version": golden["version"]}
        if not found_entry or not found_header:
            raise KeyError(f"Golden not found for conversation: {conversation_id}")
        return {
            **found_header,
            "entry": found_entry,
        }
