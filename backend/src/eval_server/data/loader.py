from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import yaml
from jsonschema import Draft202012Validator


# Typed structures for conversations and goldens
class ToolCall(TypedDict, total=False):
    tool_name: str
    arguments: Dict[str, Any]
    result: Any


class ConversationTurn(TypedDict, total=False):
    turn_id: str
    role: str
    content: str
    tool_calls: List[ToolCall]


class Conversation(TypedDict):
    conversation_id: str
    metadata: Dict[str, Any]
    turns: List[ConversationTurn]


class Expected(TypedDict, total=False):
    text_variants: List[str]
    structured: Dict[str, Any]
    weights: Dict[str, float]
    constraints: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]


class Expectation(TypedDict):
    turn_id: str
    expected: Expected


class Golden(TypedDict):
    conversation_id: str
    expectations: List[Expectation]


def _repo_root() -> Path:
    """Resolve repository root by locating 'configs/schemas' up the tree.

    Falls back to the parent of a folder named 'backend' if schemas not found.
    """
    p = Path(__file__).resolve()
    for ancestor in p.parents:
        if (ancestor / "configs" / "schemas").exists():
            return ancestor
    for ancestor in p.parents:
        if ancestor.name == "backend":
            return ancestor.parent
    raise FileNotFoundError("Could not locate repository root containing configs/schemas")


def _load_text_file(path: Path) -> Any:
    try:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        if path.suffix.lower() in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e

    raise ValueError(f"Unsupported file extension for {path}. Use .json, .yaml, or .yml")


def _load_schema(name: str) -> Dict[str, Any]:
    schema_path = _repo_root() / "configs" / "schemas" / name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate(instance: Any, schema_name: str) -> None:
    schema = _load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        first = errors[0]
        path = "/".join([str(p) for p in first.path])
        msg = f"Schema validation failed at '{path}': {first.message}"
        raise ValueError(msg)


def load_conversation(path: str | Path) -> Conversation:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Conversation file not found: {p}")
    data = _load_text_file(p)
    _validate(data, "conversation.schema.json")
    return data  # type: ignore[return-value]


def load_golden(path: str | Path) -> Golden:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden file not found: {p}")
    data = _load_text_file(p)
    _validate(data, "golden.schema.json")
    return data  # type: ignore[return-value]


__all__ = [
    "Conversation",
    "ConversationTurn",
    "ToolCall",
    "Expected",
    "Expectation",
    "Golden",
    "load_conversation",
    "load_golden",
]
