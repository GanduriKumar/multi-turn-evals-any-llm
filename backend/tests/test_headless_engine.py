from __future__ import annotations

import json
from pathlib import Path

from eval_server.headless_engine import run_headless


def test_headless_engine_creates_artifacts(tmp_path: Path):
    # Use sample run config and write to a temp output dir
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"

    out_dir = tmp_path / "out"
    path = run_headless(sample, output_dir=out_dir)

    assert path == out_dir
    summary_file = out_dir / "summary.json"
    assert summary_file.exists()

    data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "results" in data
    assert isinstance(data["results"], list)
    # Expect at least one result entry
    assert len(data["results"]) >= 1
