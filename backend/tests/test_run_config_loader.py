from __future__ import annotations

from pathlib import Path
import json
import textwrap

import pytest

from eval_server.config.run_config_loader import load_run_config, RunConfig


def test_load_sample_run_config():
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    rc: RunConfig = load_run_config(sample)

    assert rc.version == "1.0.0"
    assert len(rc.datasets) >= 1
    assert len(rc.models) >= 1

    # Verify a few fields mapped as expected
    ds0 = rc.datasets[0]
    assert ds0.conversation.endswith("conversation_001.json")
    assert ds0.golden.endswith("conversation_001.golden.yaml")

    m0 = rc.models[0]
    assert m0.name == "dummy-1"
    assert m0.provider == "dummy"
    assert m0.model.startswith("dummy-")

    # Thesholds map
    assert rc.thresholds is not None
    assert rc.thresholds.metric is not None
    assert rc.thresholds.metric.get("correctness") == 0.8


def test_invalid_missing_required_fields(tmp_path: Path):
    bad_yaml = textwrap.dedent(
        """
        version: "1.0.0"
        # Missing required datasets/models
        name: bad
        """
    )
    p = tmp_path / "bad.yaml"
    p.write_text(bad_yaml, encoding="utf-8")

    with pytest.raises(ValueError) as e:
        load_run_config(p)
    msg = str(e.value)
    assert "datasets" in msg or "models" in msg


def test_invalid_field_types(tmp_path: Path):
    # max_concurrent_requests must be integer
    content = {
        "version": "1.0.0",
        "datasets": [
            {"conversation": "a.json", "golden": "b.yaml"}
        ],
        "models": [
            {
                "name": "m",
                "provider": "dummy",
                "model": "dummy-v1",
                "concurrency": {"max_concurrent_requests": 0.5},
            }
        ]
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError) as e:
        load_run_config(p)
    assert "max_concurrent_requests" in str(e.value)
