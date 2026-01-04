from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from eval_server.__main__ import create_app


def make_valid_payload(repo_root: Path) -> dict:
    conv = repo_root / 'datasets' / 'examples' / 'conversation_001.json'
    gold = repo_root / 'datasets' / 'examples' / 'conversation_001.golden.yaml'
    return {
        'version': '1.0.0',
        'datasets': [
            {'id': 'conv001', 'conversation': conv.as_posix(), 'golden': gold.as_posix()},
        ],
        'models': [
            {'name': 'dummy', 'provider': 'dummy', 'model': 'dummy'},
        ],
    }


def test_start_run_valid(tmp_path: Path):
    app = create_app()
    client = TestClient(app)
    repo_root = Path(__file__).resolve().parents[2]

    payload = make_valid_payload(repo_root)
    r = client.post('/api/v1/runs/', json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['run_id'].startswith('run_')
    assert data['status'] == 'queued'
    assert data['job_id']


def test_start_run_invalid_missing_fields():
    app = create_app()
    client = TestClient(app)

    payload = {'version': '1.0.0', 'datasets': [], 'models': []}
    r = client.post('/api/v1/runs/', json=payload)
    assert r.status_code == 400
