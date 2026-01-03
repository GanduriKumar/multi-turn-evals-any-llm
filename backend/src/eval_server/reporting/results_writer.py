from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping
import json


def _key(row: Mapping[str, Any]) -> str:
    return f"{row.get('dataset_id')}|{row.get('conversation_id')}|{row.get('model_name')}|{row.get('turn_id')}"


def assemble_results(
    *,
    run_info: Mapping[str, Any],
    summary: Mapping[str, Any],
    raw_rows: List[Mapping[str, Any]],
    normalized_rows: List[Mapping[str, Any]],
    turn_scores_rows: List[Mapping[str, Any]],
    conversation_scores: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Assemble a single machine-readable results JSON structure.

    Structure:
    {
      "run": {...},               # run metadata (id, datasets, models, etc.)
      "summary": {...},           # orchestrator summary (per-conversation aggregated scores)
      "results": [
         { "dataset_id":..., "conversation_id":..., "model_name":...,
           "aggregate": {"score": float, "passed": bool},
           "turns": [
              {"turn_id": str|int,
               "prompt": str, "response": str, "context": list[str],
               "canonical": Mapping,
               "metrics": Mapping[str, float],
               "weights": Mapping[str, float],
               "weighted_score": float,
               "passed": bool}
           ]
         }
      ]
    }
    """
    # Index helpers by key
    raw_ix: Dict[str, Mapping[str, Any]] = {_key(r): r for r in raw_rows}
    norm_ix: Dict[str, Mapping[str, Any]] = {_key(r): r for r in normalized_rows}
    score_ix: Dict[str, Mapping[str, Any]] = {_key(r): r for r in turn_scores_rows}

    # Group keys per (dataset, conversation, model)
    groups: Dict[str, Dict[str, Any]] = {}
    for agg in conversation_scores:
        ds = agg.get("dataset_id")
        cid = agg.get("conversation_id")
        model = agg.get("model_name")
        group_key = f"{ds}|{cid}|{model}"
        groups[group_key] = {
            "dataset_id": ds,
            "conversation_id": cid,
            "model_name": model,
            "aggregate": {"score": float(agg.get("score", 0.0)), "passed": bool(agg.get("passed", False))},
            "turns": [],
        }

    # Attach turn-level rows
    for r in raw_rows:
        group_key = f"{r.get('dataset_id')}|{r.get('conversation_id')}|{r.get('model_name')}"
        if group_key not in groups:
            # ensure group exists even if not in conversation_scores list
            groups[group_key] = {
                "dataset_id": r.get("dataset_id"),
                "conversation_id": r.get("conversation_id"),
                "model_name": r.get("model_name"),
                "aggregate": {"score": 0.0, "passed": False},
                "turns": [],
            }
        k = _key(r)
        norm = norm_ix.get(k, {})
        sc = score_ix.get(k, {})
        groups[group_key]["turns"].append(
            {
                "turn_id": r.get("turn_id"),
                "prompt": r.get("prompt"),
                "response": r.get("response"),
                "context": r.get("context"),
                "canonical": norm.get("canonical"),
                "metrics": sc.get("scores", {}),
                "weights": sc.get("weights", {}),
                "weighted_score": sc.get("weighted_score", 0.0),
                "passed": sc.get("passed", False),
            }
        )

    return {
        "run": dict(run_info),
        "summary": dict(summary),
        "results": list(groups.values()),
    }


def write_results_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


__all__ = ["assemble_results", "write_results_json"]
