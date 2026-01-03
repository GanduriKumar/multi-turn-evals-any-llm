from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import yaml
from jsonschema import Draft202012Validator


Kind = Literal["conversation", "golden"]


class ValidationIssue(TypedDict):
    file: str
    kind: Kind
    path: str
    message: str


class ValidationResult(TypedDict):
    file: str
    kind: Kind
    ok: bool
    issues: List[ValidationIssue]


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for ancestor in p.parents:
        if (ancestor / "configs" / "schemas").exists():
            return ancestor
    for ancestor in p.parents:
        if ancestor.name == "backend":
            return ancestor.parent
    raise FileNotFoundError("Could not locate repository root containing configs/schemas")


def _load_schema(name: str) -> Dict[str, Any]:
    schema_path = _repo_root() / "configs" / "schemas" / name
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _schema_name(kind: Kind) -> str:
    return "conversation.schema.json" if kind == "conversation" else "golden.schema.json"


def _read_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        if suffix in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e.msg} at line {e.lineno} column {e.colno}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    raise ValueError(f"Unsupported file extension for {path}. Use .json, .yaml, or .yml")


def _format_path(error_path: list[Any]) -> str:
    # Convert deque/list of JSON pointers to a readable path like /turns/0/role
    if not error_path:
        return "/"
    segments: List[str] = []
    for p in error_path:
        segments.append(str(p))
    return "/" + "/".join(segments)


def validate_file(path: str | Path, kind: Kind) -> ValidationResult:
    p = Path(path)
    issues: List[ValidationIssue] = []
    if not p.exists():
        issues.append({
            "file": str(p),
            "kind": kind,
            "path": "/",
            "message": "File not found",
        })
        return {"file": str(p), "kind": kind, "ok": False, "issues": issues}

    try:
        data = _read_file(p)
    except ValueError as e:
        issues.append({
            "file": str(p),
            "kind": kind,
            "path": "/",
            "message": str(e),
        })
        return {"file": str(p), "kind": kind, "ok": False, "issues": issues}

    schema = _load_schema(_schema_name(kind))
    validator = Draft202012Validator(schema)
    errs = sorted(validator.iter_errors(data), key=lambda e: e.path)
    for err in errs:
        issues.append({
            "file": str(p),
            "kind": kind,
            "path": _format_path(list(err.path)),
            "message": err.message,
        })

    return {"file": str(p), "kind": kind, "ok": not issues, "issues": issues}


def validate_dir(dir_path: str | Path, kind: Kind) -> List[ValidationResult]:
    d = Path(dir_path)
    if not d.exists():
        return [{"file": str(d), "kind": kind, "ok": False, "issues": [{"file": str(d), "kind": kind, "path": "/", "message": "Directory not found"}]}]

    results: List[ValidationResult] = []
    for pattern in ("**/*.json", "**/*.yaml", "**/*.yml"):
        for f in sorted(d.glob(pattern)):
            results.append(validate_file(f, kind))
    return results


def assert_valid(path: str | Path, kind: Kind) -> None:
    res = validate_file(path, kind)
    if not res["ok"]:
        details = "\n".join(f"- {i['file']}{i['path']}: {i['message']}" for i in res["issues"])
        raise ValueError(f"Validation failed for {res['file']} ({res['kind']}):\n{details}")


__all__ = [
    "Kind",
    "ValidationIssue",
    "ValidationResult",
    "validate_file",
    "validate_dir",
    "assert_valid",
]
