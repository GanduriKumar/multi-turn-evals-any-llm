from __future__ import annotations
from pathlib import Path
import json
from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "configs" / "schemas"

class SchemaValidator:
    def __init__(self):
        self._schemas = {}
        for name in ["dataset", "golden", "run_config"]:
            path = SCHEMAS_DIR / f"{name}.schema.json"
            with open(path, "r", encoding="utf-8") as f:
                schema = json.load(f)
                self._schemas[name] = Draft202012Validator(schema)

    def validate(self, name: str, data: dict) -> list[str]:
        if name not in self._schemas:
            raise KeyError(f"Unknown schema {name}")
        validator = self._schemas[name]
        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors]
