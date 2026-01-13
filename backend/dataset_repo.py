from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

try:
    from .schemas import SchemaValidator
except ImportError:
    from schemas import SchemaValidator

DEFAULT_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"


class DatasetRepository:
    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir: Path = Path(root_dir) if root_dir else DEFAULT_DATASETS_DIR
        self.sv = SchemaValidator()

    # File conventions: <dataset_id>.dataset.json and <dataset_id>.golden.json
    def _dataset_files(self) -> List[Path]:
        # Support both flat and hierarchical layouts under the vertical root
        # e.g., datasets/<vertical>/*.dataset.json or datasets/<vertical>/<behavior>/<version>/*.dataset.json
        return sorted(self.root_dir.rglob("*.dataset.json"))

    def _golden_files(self) -> List[Path]:
        return sorted(self.root_dir.rglob("*.golden.json"))

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
            # derive turns per conversation (typical): use first conversation length if available
            try:
                convs = data.get("conversations") or []
                tpc = len((convs[0] or {}).get("turns", [])) if convs else None
            except Exception:
                tpc = None
            if errors:
                # Skip invalid datasets in listing but annotate error info
                items.append({
                    "dataset_id": data.get("dataset_id") or p.stem.replace(".dataset", ""),
                    "version": data.get("version"),
                    "domain": data.get("metadata", {}).get("domain"),
                    "difficulty": data.get("metadata", {}).get("difficulty"),
                    "conversations": len(data.get("conversations", []) or []),
                    "turns_per_conversation": tpc,
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
                "turns_per_conversation": tpc,
                "has_golden": data["dataset_id"] in golden_index,
                "valid": True,
            })
        return items

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        # Search in root and nested folders for the exact dataset filename
        candidates = list(self.root_dir.rglob(f"{dataset_id}.dataset.json"))
        if not candidates:
            raise FileNotFoundError(f"Dataset file not found: {dataset_id}.dataset.json")
        # prefer the shallowest path
        p = sorted(candidates, key=lambda x: len(x.parts))[0]
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
        """
        Locate the golden record for a conversation.

        Important: The same conversation_id may appear across multiple golden files
        if there are both per-scenario and combined coverage sets present. To avoid
        collisions, we first determine the dataset_id that contains this conversation
        and then restrict our search to golden files that match that dataset_id.
        """
        # Determine which dataset this conversation belongs to
        try:
            conv_info = self.get_conversation(conversation_id)
            target_dataset_id: Optional[str] = conv_info.get("dataset_id")
        except Exception:
            target_dataset_id = None

        found_entry: Optional[Dict[str, Any]] = None
        found_header: Optional[Dict[str, Any]] = None

        for p in self._golden_files():
            golden = self._load_json(p)
            if self.sv.validate("golden", golden):
                continue
            # If we know the dataset that contains this conversation, only consider matching golden files
            if target_dataset_id and golden.get("dataset_id") != target_dataset_id:
                continue
            for entry in golden.get("entries", []):
                if entry.get("conversation_id") == conversation_id:
                    if found_entry is not None:
                        # If duplicates occur even after filtering by dataset_id,
                        # keep the first and prefer the file that exactly matches target_dataset_id.
                        # This avoids failing runs when both combined and per-scenario goldens exist.
                        continue
                    found_entry = entry
                    found_header = {"dataset_id": golden.get("dataset_id"), "version": golden.get("version")}

        if not found_entry or not found_header:
            # As a fallback (e.g., if get_conversation failed), search across all golden files
            # and pick the first match deterministically.
            if not target_dataset_id:
                for p in self._golden_files():
                    golden = self._load_json(p)
                    if self.sv.validate("golden", golden):
                        continue
                    for entry in golden.get("entries", []):
                        if entry.get("conversation_id") == conversation_id:
                            found_entry = entry
                            found_header = {"dataset_id": golden.get("dataset_id"), "version": golden.get("version")}
                            break
                    if found_entry:
                        break

        if not found_entry or not found_header:
            raise KeyError(f"Golden not found for conversation: {conversation_id}")

        return {**found_header, "entry": found_entry}
