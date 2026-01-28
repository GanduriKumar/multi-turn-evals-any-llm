from __future__ import annotations

import csv
from io import StringIO
from typing import Any, Dict, List, Optional

try:
    from .dataset_repo import DatasetRepository
except ImportError:
    from backend.dataset_repo import DatasetRepository


def export_prompts_to_csv(vertical: str, dataset_id: Optional[str] = None) -> str:
    """
    Export prompts and golden data from dataset files to CSV format.
    
    Args:
        vertical: The vertical subfolder (e.g., "commerce")
        dataset_id: Optional specific dataset to export. If None, exports all datasets.
    
    Returns:
        CSV string content with columns:
        - dataset_id, conversation_id, conversation_title, domain, behavior
        - turn_index, role, prompt_text, expected_variants
        - final_decision, final_next_action
    """
    from pathlib import Path
    
    # Get datasets directory for the vertical
    datasets_root = Path(__file__).resolve().parents[1] / "datasets" / vertical
    repo = DatasetRepository(datasets_root)
    
    # Get datasets to export
    if dataset_id:
        # Export specific dataset
        try:
            dataset = repo.get_dataset(dataset_id)
            datasets = [dataset]
        except FileNotFoundError:
            # Return empty CSV with headers if dataset not found
            datasets = []
    else:
        # Export all datasets
        dataset_list = repo.list_datasets()
        datasets = []
        for ds_info in dataset_list:
            if ds_info.get("valid"):
                try:
                    ds = repo.get_dataset(ds_info["dataset_id"])
                    datasets.append(ds)
                except Exception:
                    # Skip datasets that can't be loaded
                    continue
    
    # Build golden lookup map
    golden_map: Dict[str, Dict[str, Any]] = {}
    for p in repo._golden_files():
        try:
            golden = repo._load_json(p)
            if repo.sv.validate("golden", golden):
                continue
            ds_id = golden.get("dataset_id")
            if ds_id:
                golden_map[ds_id] = golden
        except Exception:
            continue
    
    # Generate CSV
    out = StringIO()
    w = csv.writer(out)
    w.writerow([
        "dataset_id",
        "conversation_id",
        "conversation_title",
        "domain",
        "behavior",
        "turn_index",
        "role",
        "prompt_text",
        "expected_variants",
        "final_decision",
        "final_next_action",
    ])
    
    # Collect all rows for sorting
    rows: List[List[Any]] = []
    
    for dataset in datasets:
        ds_id = dataset.get("dataset_id")
        golden = golden_map.get(ds_id)
        
        # Build golden lookup for this dataset
        exp_map: Dict[str, Dict[int, List[str]]] = {}
        outcomes: Dict[str, Dict[str, Any]] = {}
        
        if golden:
            for entry in golden.get("entries", []):
                conv_id = entry.get("conversation_id")
                # Build expected variants map by turn index
                exp_map[conv_id] = {}
                for turn_data in entry.get("turns", []):
                    turn_idx = turn_data.get("turn_index")
                    variants = (turn_data.get("expected") or {}).get("variants") or []
                    exp_map[conv_id][turn_idx] = list(variants)
                
                # Get final outcome
                final_outcome = entry.get("final_outcome", {})
                outcomes[conv_id] = final_outcome
        
        # Get metadata
        ds_metadata = dataset.get("metadata", {}) or {}
        ds_domain = ds_metadata.get("domain", "")
        
        # Process conversations
        for conv in dataset.get("conversations", []):
            conv_id = conv.get("conversation_id", "")
            conv_title = conv.get("title", "")
            
            # Get conversation metadata
            conv_metadata = conv.get("metadata", {}) or {}
            domain = conv_metadata.get("domain") or ds_domain
            behavior = conv_metadata.get("behavior", "")
            
            # Get final outcome for this conversation
            outcome = outcomes.get(conv_id, {})
            final_decision = outcome.get("decision", "")
            final_next_action = outcome.get("next_action", "")
            
            # Process turns
            turns = conv.get("turns", [])
            for idx, turn in enumerate(turns):
                role = turn.get("role", "")
                text = turn.get("text", "")
                
                # Get expected variants if this turn has golden data
                variants = exp_map.get(conv_id, {}).get(idx, [])
                expected_str = "|".join(variants) if variants else ""
                
                rows.append([
                    ds_id,
                    conv_id,
                    conv_title,
                    domain,
                    behavior,
                    idx,
                    role,
                    text,
                    expected_str,
                    final_decision,
                    final_next_action,
                ])
    
    # Sort by dataset_id, conversation_id, turn_index for deterministic output
    rows.sort(key=lambda r: (r[0] or "", r[1] or "", r[5]))
    
    # Write sorted rows
    for row in rows:
        w.writerow(row)
    
    return out.getvalue()
