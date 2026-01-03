from __future__ import annotations

import json
from pathlib import Path

from eval_server.config.run_config_loader import load_run_config
from eval_server.headless_engine import run_headless


def test_results_writer_creates_file(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"

    rc = load_run_config(sample)
    out_dir = run_headless(rc, output_dir=tmp_path)

    results_path = out_dir / "results.json"
    assert results_path.exists()

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "run" in data and "summary" in data and "results" in data
    # Expect at least one result group
    assert isinstance(data["results"], list) and len(data["results"]) >= 1
    grp = data["results"][0]
    assert "aggregate" in grp and "turns" in grp
    if grp["turns"]:
        t = grp["turns"][0]
        assert "prompt" in t and "response" in t and "metrics" in t
