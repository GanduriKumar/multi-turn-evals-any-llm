from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient
from eval_server.__main__ import create_app

client = TestClient(create_app())

def test_submit_feedback_creates_file(tmp_path: Path, monkeypatch):
    # Mock the runs root to use tmp_path
    # The API uses _find_run_output_dir which searches in "runs" or "configs/runs"
    # We'll mock _find_run_output_dir to return a dir in tmp_path
    
    run_id = "test-run-123"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    
    # Mock the finder
    from eval_server.api import feedback
    monkeypatch.setattr(feedback, "_find_run_output_dir", lambda rid: run_dir)
    
    payload = {
        "run_id": run_id,
        "feedback": [
            {
                "dataset_id": "ds1",
                "conversation_id": "c1",
                "model_name": "m1",
                "turn_id": 1,
                "rating": 4.5,
                "notes": "Good job",
                "override_pass": True,
                "override_score": 0.95
            }
        ]
    }
    
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == run_id
    assert data["total_records"] == 1
    
    # Verify file content
    store_path = run_dir / "annotations.json"
    assert store_path.exists()
    content = json.loads(store_path.read_text(encoding="utf-8"))
    assert content["run_id"] == run_id
    assert len(content["annotations"]) == 1
    ann = content["annotations"][0]
    assert ann["rating"] == 4.5
    assert ann["notes"] == "Good job"

def test_append_feedback(tmp_path: Path, monkeypatch):
    run_id = "test-run-append"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    
    from eval_server.api import feedback
    monkeypatch.setattr(feedback, "_find_run_output_dir", lambda rid: run_dir)
    
    # Initial feedback
    initial = {
        "run_id": run_id,
        "annotations": [
            {
                "dataset_id": "ds1",
                "conversation_id": "c1",
                "model_name": "m1",
                "turn_id": 1,
                "rating": 3.0
            }
        ]
    }
    (run_dir / "annotations.json").write_text(json.dumps(initial), encoding="utf-8")
    
    # Append new feedback
    payload = {
        "run_id": run_id,
        "feedback": [
            {
                "dataset_id": "ds1",
                "conversation_id": "c1",
                "model_name": "m1",
                "turn_id": 2,
                "rating": 5.0
            }
        ]
    }
    
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 200
    
    content = json.loads((run_dir / "annotations.json").read_text(encoding="utf-8"))
    assert len(content["annotations"]) == 2
    assert content["annotations"][0]["rating"] == 3.0
    assert content["annotations"][1]["rating"] == 5.0

def test_invalid_feedback_validation():
    payload = {
        "run_id": "any",
        "feedback": [
            {
                "dataset_id": "ds1",
                "conversation_id": "c1",
                "model_name": "m1",
                "turn_id": 1,
                "rating": 6.0  # Invalid > 5
            }
        ]
    }
    response = client.post("/api/feedback", json=payload)
    assert response.status_code == 422  # Validation error
