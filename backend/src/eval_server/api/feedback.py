from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import json

router = APIRouter()


class TurnFeedback(BaseModel):
    dataset_id: str = Field(...)
    conversation_id: str = Field(...)
    model_name: str = Field(...)
    turn_id: int | str = Field(...)
    rating: Optional[float] = Field(None, ge=0, le=5)
    notes: Optional[str] = None
    override_pass: Optional[bool] = None
    override_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class SubmitFeedbackRequest(BaseModel):
    run_id: str
    feedback: List[TurnFeedback]


class SubmitFeedbackResponse(BaseModel):
    run_id: str
    stored_path: str
    total_records: int


def _runs_root() -> Path:
    return Path("configs") / "runs"


def _find_run_output_dir(run_id: str) -> Path:
    # Search typical runs directory for a folder that ends with the run_id
    runs_dir = Path("configs") / "runs"  # fallback search root
    # Prefer standard runs output dir hierarchy
    candidates: List[Path] = []
    # Common output parent
    common_parent = Path("runs")
    if common_parent.exists():
        for child in common_parent.iterdir():
            if child.is_dir() and run_id in child.name:
                candidates.append(child)
    # Fallback: temp or other dirs containing run_id
    if not candidates:
        for root in [Path("."), Path("configs")]:
            for p in root.rglob("*"):
                if p.is_dir() and run_id in p.name and (p / "results.json").exists():
                    candidates.append(p)
    if not candidates:
        # If no dir found, default to runs/<run_id>
        return Path("runs") / run_id
    # Prefer exact match
    for c in candidates:
        if c.name == run_id:
            return c
    return candidates[0]


def _feedback_store_path(run_output_dir: Path) -> Path:
    return run_output_dir / "annotations.json"


def _load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"annotations": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"annotations": []}


@router.post("/feedback", response_model=SubmitFeedbackResponse)
def submit_feedback(req: SubmitFeedbackRequest) -> SubmitFeedbackResponse:
    """Submit evaluator feedback for a run. Appends to annotations.json in the run directory.

    The stored file structure:
    {
      "run_id": "...",
      "annotations": [ {dataset_id, conversation_id, model_name, turn_id, rating, notes, override_pass, override_score}, ... ]
    }
    """
    run_id = req.run_id.strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")

    out_dir = _find_run_output_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = _feedback_store_path(out_dir)

    existing = _load_existing(store_path)
    annotations = existing.get("annotations") or []

    # Convert pydantic models to dicts
    new_items = [fb.model_dump() for fb in req.feedback]
    annotations.extend(new_items)

    payload = {"run_id": run_id, "annotations": annotations}
    store_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return SubmitFeedbackResponse(run_id=run_id, stored_path=str(store_path.resolve()), total_records=len(new_items))
