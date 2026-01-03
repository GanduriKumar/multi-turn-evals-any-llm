from __future__ import annotations

"""Rescoring utilities to apply evaluator overrides without mutating auto-scores.

This module augments a consolidated results payload (results.json structure) with
"final" fields derived from evaluator overrides while preserving original
automatic scoring fields. Specifically:

- Keep original per-turn fields immutable: `passed`, `weighted_score`.
- Add per-turn fields:
    * `evaluator`: {rating, notes, override_pass, override_score}
    * `final_pass`: bool
    * `final_weighted_score`: float
- Add per-conversation fields:
    * `aggregate_final`: {avg_turn_score: float, passed_all: bool}

Overrides can be provided as:
- List[ {dataset_id, conversation_id, model_name, turn_id, rating?, notes?, override_pass?, override_score?} ]
- Dict[key -> {...}] where key == "dataset|conversation|model|turn"
- Dict{"annotations": [...]} wrapper.
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import json


def _key(ds: str | None, cid: str | None, model: str | None, turn_id: str | int | None) -> str:
    return f"{ds}|{cid}|{model}|{turn_id}"


def _normalize_annotations(annotations: Mapping[str, Any] | list[Mapping[str, Any]] | Mapping[str, Any] | None) -> dict[str, dict]:
    """Normalize various annotation payload shapes into a keyed dict.

    Accepts either:
    - {"annotations": [...]} wrapper
    - A list of dicts
    - A dict keyed by composite key
    Returns: dict[key(str) -> {rating, notes, override_pass, override_score}]
    """
    if annotations is None:
        return {}

    source: Any = annotations
    if isinstance(annotations, dict) and "annotations" in annotations:
        source = annotations.get("annotations")

    ann_map: dict[str, dict] = {}
    if isinstance(source, list):
        for item in source:
            if not isinstance(item, Mapping):
                continue
            k = _key(
                str(item.get("dataset_id")) if item.get("dataset_id") is not None else None,
                str(item.get("conversation_id")) if item.get("conversation_id") is not None else None,
                item.get("model_name"),
                item.get("turn_id"),
            )
            ann_map[k] = {
                "rating": item.get("rating"),
                "notes": item.get("notes"),
                "override_pass": item.get("override_pass"),
                "override_score": item.get("override_score"),
            }
    elif isinstance(source, Mapping):
        for k, v in source.items():
            if isinstance(v, Mapping):
                ann_map[str(k)] = {
                    "rating": v.get("rating"),
                    "notes": v.get("notes"),
                    "override_pass": v.get("override_pass"),
                    "override_score": v.get("override_score"),
                }
    return ann_map


def rescore_payload(
    payload: Mapping[str, Any],
    *,
    annotations: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Apply evaluator overrides to a results payload, returning a new payload.

    The input payload must follow `reporting.results_writer.assemble_results` structure.
    The returned payload contains additional `final_*` fields while preserving the
    original auto-scored fields.
    """
    ann_map = _normalize_annotations(annotations)

    # Deep copy to avoid mutating caller payload
    updated: Dict[str, Any] = json.loads(json.dumps(payload))

    for group in updated.get("results", []) or []:
        ds = group.get("dataset_id")
        cid = group.get("conversation_id")
        model = group.get("model_name")

        final_scores: list[float] = []
        final_passes: list[bool] = []

        for turn in group.get("turns", []) or []:
            k = _key(ds, cid, model, turn.get("turn_id"))
            ann = ann_map.get(k)

            auto_pass = bool(turn.get("passed", False))
            auto_score = float(turn.get("weighted_score", 0.0) or 0.0)
            final_pass = auto_pass
            final_score = auto_score

            evaluator: dict[str, Any] = {}
            if ann:
                evaluator = {
                    "rating": ann.get("rating"),
                    "notes": ann.get("notes"),
                    "override_pass": ann.get("override_pass"),
                    "override_score": ann.get("override_score"),
                }
                if ann.get("override_pass") is not None:
                    final_pass = bool(ann.get("override_pass"))
                if ann.get("override_score") is not None:
                    try:
                        final_score = float(ann.get("override_score"))
                    except Exception:
                        # Ignore invalid override score values
                        pass

            # Preserve auto results as-is; attach evaluator and final fields
            turn["evaluator"] = evaluator
            turn["final_pass"] = final_pass
            turn["final_weighted_score"] = final_score

            final_scores.append(final_score)
            final_passes.append(final_pass)

        # Conversation-level final aggregation (simple average of final turn scores)
        avg_final = (sum(final_scores) / len(final_scores)) if final_scores else 0.0
        group["aggregate_final"] = {
            "avg_turn_score": avg_final,
            "passed_all": all(final_passes) if final_passes else bool(group.get("aggregate", {}).get("passed", False)),
        }

    return updated


def rescore_results_file(
    results_json_path: Path,
    *,
    annotations_path: Optional[Path] = None,
    annotations: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Rescore a results.json file using provided annotations and write an updated file.

    If `output_path` is not provided, writes next to the input as `results_rescored.json`.
    Returns the path written.
    """
    with results_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Load annotations from file if provided and explicit annotations not passed
    ann = annotations
    if ann is None and annotations_path is not None and Path(annotations_path).exists():
        try:
            ann = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
        except Exception:
            ann = None

    updated = rescore_payload(payload, annotations=ann)

    if output_path is None:
        output_path = results_json_path.with_name("results_rescored.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    return output_path


__all__ = ["rescore_payload", "rescore_results_file"]
