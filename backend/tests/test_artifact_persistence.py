from __future__ import annotations

from pathlib import Path
import json

from eval_server.config.run_config_loader import load_run_config
from eval_server.headless_engine import run_headless


def test_artifacts_created(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"

    rc = load_run_config(sample)
    out_dir = run_headless(rc, output_dir=tmp_path)

    # Basic files
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "raw_outputs.jsonl").exists()

    # Subdirectories
    inputs = out_dir / "inputs"
    logs = out_dir / "logs"
    scores = out_dir / "scores"
    assert inputs.exists() and inputs.is_dir()
    assert logs.exists() and logs.is_dir()
    assert scores.exists() and scores.is_dir()

    # Inputs content
    assert (inputs / "run_config.json").exists()

    # Logs
    assert (logs / "progress.jsonl").exists()
    assert (logs / "run.log").exists()

    # Scores
    assert (scores / "turn_scores.jsonl").exists()
    assert (scores / "conversation_scores.json").exists()

    # Normalized
    assert (out_dir / "normalized.jsonl").exists()

    # Sanity check JSON content
    with (out_dir / "summary.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
        assert "results" in data
