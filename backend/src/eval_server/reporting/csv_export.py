from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import csv
import json


def _collect_metric_keys(results: Mapping[str, Any]) -> List[str]:
    keys: set[str] = set()
    for group in results.get("results", []) or []:
        for t in group.get("turns", []) or []:
            metrics = t.get("metrics") or {}
            for k in metrics.keys():
                keys.add(str(k))
    # Always include an overall metric column for weighted score
    keys.add("weighted")
    return sorted(keys)


def _flatten_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    run = payload.get("run", {})
    run_id = run.get("run_id")
    metric_cols = _collect_metric_keys(payload)
    rows: List[Dict[str, Any]] = []
    for group in payload.get("results", []) or []:
        ds = group.get("dataset_id")
        cid = group.get("conversation_id")
        model = group.get("model_name")
        agg = group.get("aggregate") or {}
        conv_score = agg.get("score", 0.0)
        conv_pass = agg.get("passed", False)
        for t in group.get("turns", []) or []:
            base = {
                "run_id": run_id,
                "dataset_id": ds,
                "conversation_id": cid,
                "model_name": model,
                "conv_score": conv_score,
                "conv_pass": conv_pass,
                "turn_id": t.get("turn_id"),
                "weighted_score": t.get("weighted_score", 0.0),
                "passed": t.get("passed", False),
                "prompt": t.get("prompt"),
                "response": t.get("response"),
            }
            metrics = t.get("metrics") or {}
            # Per-metric columns
            for k in metric_cols:
                if k == "weighted":
                    base[f"metric_{k}"] = base.get("weighted_score")
                else:
                    base[f"metric_{k}"] = metrics.get(k)
            rows.append(base)
    return rows


def export_results_csv(results_json_path: Path, csv_path: Path | None = None) -> Path:
    """Export consolidated results.json into a CSV file for analysts.

    Columns include run_id, dataset_id, conversation_id, model_name, conv_score, conv_pass,
    turn_id, weighted_score, passed, prompt, response, and per-metric columns metric_<name>.
    """
    with results_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = _flatten_rows(payload)
    if not rows:
        # still write a header from minimal default
        rows = [{
            "run_id": payload.get("run", {}).get("run_id"),
            "dataset_id": None,
            "conversation_id": None,
            "model_name": None,
            "conv_score": None,
            "conv_pass": None,
            "turn_id": None,
            "weighted_score": None,
            "passed": None,
            "prompt": None,
            "response": None,
        }]

    # Use union of keys as header
    header_keys: List[str] = list(rows[0].keys())
    # ensure determinism and inclusion of metric columns
    all_keys: set[str] = set(header_keys)
    for r in rows:
        all_keys.update(r.keys())
    header = [
        "run_id",
        "dataset_id",
        "conversation_id",
        "model_name",
        "conv_score",
        "conv_pass",
        "turn_id",
        "weighted_score",
        "passed",
        "prompt",
        "response",
    ] + sorted([k for k in all_keys if k.startswith("metric_")])

    if csv_path is None:
        csv_path = results_json_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in header})
    return csv_path


__all__ = ["export_results_csv"]
