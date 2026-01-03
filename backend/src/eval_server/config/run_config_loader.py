from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml
from jsonschema import Draft202012Validator


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
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate(instance: Any, schema_name: str) -> None:
    schema = _load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        # Build a concise, helpful error message including the JSON path
        first = errors[0]
        path = "/".join([str(p) for p in first.path])
        msg = f"Run config validation failed at '{path}': {first.message}"
        raise ValueError(msg)


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


@dataclass(frozen=True)
class ModelConcurrency:
    max_requests_per_second: Optional[float] = None
    max_concurrent_requests: Optional[int] = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model: str
    params: Mapping[str, Any] | None = None
    concurrency: Optional[ModelConcurrency] = None


@dataclass(frozen=True)
class DatasetConfig:
    conversation: str
    golden: str
    id: Optional[str] = None
    tags: Optional[List[str]] = None
    difficulty: Optional[str] = None


@dataclass(frozen=True)
class TruncationPolicy:
    strategy: str = "none"
    max_input_tokens: Optional[int] = None


@dataclass(frozen=True)
class GlobalConcurrency:
    max_workers: Optional[int] = None
    per_model: Optional[int] = None


@dataclass(frozen=True)
class Thresholds:
    turn_pass: Optional[float] = None
    conversation_pass: Optional[float] = None
    metric: Optional[Mapping[str, float]] = None


@dataclass(frozen=True)
class RunConfig:
    version: str
    datasets: List[DatasetConfig]
    models: List[ModelConfig]
    run_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    output_dir: Optional[str] = None
    random_seed: Optional[int] = None
    metric_bundles: Optional[List[str]] = None
    truncation: Optional[TruncationPolicy] = None
    concurrency: Optional[GlobalConcurrency] = None
    thresholds: Optional[Thresholds] = None


def load_run_config(path: str | Path) -> RunConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Run config file not found: {p}")
    data = _load_text_file(p)
    _validate(data, "run_config.schema.json")

    # Convert validated dict into dataclasses
    datasets = [
        DatasetConfig(
            id=d.get("id"),
            conversation=d["conversation"],
            golden=d["golden"],
            tags=d.get("tags"),
            difficulty=d.get("difficulty"),
        )
        for d in data["datasets"]
    ]

    def _to_model(m: Dict[str, Any]) -> ModelConfig:
        conc = None
        if "concurrency" in m and m["concurrency"] is not None:
            c = m["concurrency"]
            conc = ModelConcurrency(
                max_requests_per_second=c.get("max_requests_per_second"),
                max_concurrent_requests=c.get("max_concurrent_requests"),
            )
        return ModelConfig(
            name=m["name"],
            provider=m["provider"],
            model=m["model"],
            params=m.get("params"),
            concurrency=conc,
        )

    models = [_to_model(m) for m in data["models"]]

    trunc = None
    if data.get("truncation") is not None:
        t = data["truncation"]
        trunc = TruncationPolicy(
            strategy=t.get("strategy", "none"),
            max_input_tokens=t.get("max_input_tokens"),
        )

    conc = None
    if data.get("concurrency") is not None:
        c = data["concurrency"]
        conc = GlobalConcurrency(
            max_workers=c.get("max_workers"),
            per_model=c.get("per_model"),
        )

    thr = None
    if data.get("thresholds") is not None:
        t = data["thresholds"]
        thr = Thresholds(
            turn_pass=t.get("turn_pass"),
            conversation_pass=t.get("conversation_pass"),
            metric=t.get("metric"),
        )

    return RunConfig(
        version=data["version"],
        run_id=data.get("run_id"),
        name=data.get("name"),
        description=data.get("description"),
        output_dir=data.get("output_dir"),
        random_seed=data.get("random_seed"),
        datasets=datasets,
        models=models,
        metric_bundles=data.get("metric_bundles"),
        truncation=trunc,
        concurrency=conc,
        thresholds=thr,
    )


__all__ = [
    "RunConfig",
    "DatasetConfig",
    "ModelConfig",
    "ModelConcurrency",
    "TruncationPolicy",
    "GlobalConcurrency",
    "Thresholds",
    "load_run_config",
]
