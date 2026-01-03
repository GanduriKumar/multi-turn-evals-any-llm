from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from ..config.run_config_loader import RunConfig
from ..data.loader import load_conversation, load_golden


def _to_plain(obj: Any) -> Any:
    """Convert dataclasses and Paths to plain JSON-serializable structures."""
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _stable_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_run_id(run: RunConfig | Mapping[str, Any]) -> str:
    """Compute a deterministic run id from dataset versions, model ids, metric bundles, and config checksum.

    - Loads each dataset's conversation and golden files to extract their `version` fields.
    - Includes (provider, model, name) tuples for models.
    - Includes metric_bundles from the run config (order preserved).
    - Computes a checksum over a normalized representation of the run config (excluding run_id/output_dir).
    """
    # Normalize run config to plain dict
    rc: Dict[str, Any]
    if isinstance(run, Mapping):
        rc = _to_plain(dict(run))  # type: ignore[arg-type]
    else:
        rc = _to_plain(run)

    # Build dataset version entries
    datasets = []
    for ds in rc.get("datasets", []) or []:
        conv_path = str(ds.get("conversation"))
        golden_path = str(ds.get("golden"))
        conv = load_conversation(conv_path)
        gold = load_golden(golden_path)
        datasets.append({
            "id": ds.get("id"),
            "conversation": conv_path,
            "golden": golden_path,
            "conversation_version": conv.get("version"),
            "golden_version": gold.get("version"),
        })

    # Models identity (treat model string as version identifier)
    models = []
    for m in rc.get("models", []) or []:
        models.append({
            "name": m.get("name"),
            "provider": m.get("provider"),
            "model": m.get("model"),
        })

    # Metric bundles
    metric_bundles = list(rc.get("metric_bundles", []) or [])

    # Compute configuration fingerprint (exclude volatile keys)
    config_for_hash = dict(rc)
    config_for_hash.pop("run_id", None)
    config_for_hash.pop("output_dir", None)
    cfg_str = _stable_json_dumps(config_for_hash)
    cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()

    core = {
        "datasets": datasets,
        "models": models,
        "metric_bundles": metric_bundles,
        "config_hash": cfg_hash,
    }
    core_str = _stable_json_dumps(core)
    digest = hashlib.sha256(core_str.encode("utf-8")).hexdigest()
    # Shorten for readability
    return f"run_{digest[:16]}"


__all__ = ["compute_run_id"]
