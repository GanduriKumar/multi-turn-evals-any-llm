from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, Tuple, List
import json as _json
from jsonschema import Draft202012Validator

from .schemas import SchemaValidator

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "configs" / "schemas"


class CoverageConfig:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).resolve().parents[1] / "configs"
        self.sv = SchemaValidator()

    def load_taxonomy(self, path: Path | None = None) -> Dict[str, Any]:
        p = path or (self.root / "taxonomy.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        # JSON Schema validation (optional)
        try:
            schema_path = SCHEMAS_DIR / "coverage_taxonomy.schema.json"
            schema = _json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator(schema).validate(data)
        except Exception:
            # continue with basic checks; schema file may be missing in some installs
            pass
        # Validate shape with simple checks, then lint
        if not isinstance(data.get("domains"), list) or not data["domains"]:
            raise ValueError("taxonomy.domains must be a non-empty list")
        if not isinstance(data.get("behaviors"), list) or not data["behaviors"]:
            raise ValueError("taxonomy.behaviors must be a non-empty list")
        axes = data.get("axes") or {}
        for key in ("price_sensitivity", "brand_bias", "availability", "policy_boundary"):
            vals = axes.get(key)
            if not isinstance(vals, list) or not vals:
                raise ValueError(f"taxonomy.axes.{key} must be a non-empty list")
        self._lint_taxonomy(data)
        return data

    def load_exclusions(self, path: Path | None = None) -> Dict[str, Any]:
        p = path or (self.root / "exclusions.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        # JSON Schema validation (optional)
        try:
            schema_path = SCHEMAS_DIR / "exclusions.schema.json"
            schema = _json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator(schema).validate(data)
        except Exception:
            pass
        if not isinstance(data.get("rules"), list):
            raise ValueError("exclusions.rules must be a list")
        self._lint_exclusions(data)
        return data

    # ---- Governance & Linting ----
    def _lint_taxonomy(self, tax: Dict[str, Any]) -> None:
        domains = tax.get("domains", [])
        behaviors = tax.get("behaviors", [])
        # duplicates
        self._ensure_unique("taxonomy.domains", domains)
        self._ensure_unique("taxonomy.behaviors", behaviors)
        for axis_name, vals in (tax.get("axes") or {}).items():
            self._ensure_unique(f"taxonomy.axes.{axis_name}", vals)

    def _lint_exclusions(self, exc: Dict[str, Any]) -> None:
        rules = exc.get("rules", [])
        names = [r.get("name") for r in rules if isinstance(r, dict)]
        self._ensure_unique("exclusions.rules.name", [n for n in names if n])
        # Detect multiple caps for identical applies/when scopes
        seen_caps: Dict[str, List[str]] = {}
        for r in rules:
            if not isinstance(r, dict):
                continue
            applies = r.get("applies") or {}
            when = r.get("when") or {}
            key = json.dumps({"applies": applies, "when": when}, sort_keys=True)
            if "cap" in r:
                seen_caps.setdefault(key, []).append(r.get("name", ""))
        for key, names in seen_caps.items():
            if len(names) > 1:
                raise ValueError(f"Multiple caps defined for same applies/when scope: {names}")
        # Empty rule guard
        for r in rules:
            if not isinstance(r, dict):
                continue
            if ("exclude" not in r) and ("cap" not in r):
                raise ValueError(f"Rule '{r.get('name','')}' has neither 'exclude' nor 'cap'")

    def _ensure_unique(self, label: str, items: List[str]) -> None:
        s = set()
        dups = sorted({x for x in items if (x in s or s.add(x))})  # clever set-add trick
        if dups:
            raise ValueError(f"Duplicate values in {label}: {', '.join(dups)}")

    # ---- Coverage generation settings ----
    def load_coverage(self, path: Path | None = None) -> Dict[str, Any]:
        """
        Load coverage generation settings. This controls optimization strategy.
        Example (configs/coverage.json):
        {
          "mode": "pairwise",        # one of: exhaustive, pairwise
          "t": 2,                     # pairwise only (ignored for exhaustive)
          "per_behavior_budget": 120, # optional soft cap per Domain×Behavior
          "anchors": [                # always-include patterns
            {"applies": {"domains": ["Trust, Safety & Fraud"]},
             "when": {"policy_boundary": ["out-of-policy"]}}
          ]
        }
        If file is missing or invalid, defaults to exhaustive mode.
        """
        p = path or (self.root / "coverage.json")
        if not p.exists():
            return {"mode": "exhaustive"}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            mode = data.get("mode") or "exhaustive"
            # basic validation
            if mode not in ("exhaustive", "pairwise"):
                mode = "exhaustive"
            t = int(data.get("t", 2))
            if t < 2:
                t = 2
            per_behavior_budget = data.get("per_behavior_budget")
            if per_behavior_budget is not None:
                try:
                    per_behavior_budget = int(per_behavior_budget)
                except Exception:
                    per_behavior_budget = None
            anchors = data.get("anchors") or []
            return {
                "mode": mode,
                "t": t,
                "per_behavior_budget": per_behavior_budget,
                "anchors": anchors,
            }
        except Exception:
            return {"mode": "exhaustive"}
