from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Mapping

from eval_server.config.run_config_loader import load_run_config
from eval_server.orchestrator import evaluate_run, OrchestratorSummary


def _mock_get_response(turn: Mapping[str, Any], model: Mapping[str, Any], conversation: Mapping[str, Any]) -> str:
    # Simulate model-specific behavior: include model name and echo content
    return f"{model['name']}::{turn.get('content','')}"


def test_orchestrator_sequential_execution(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"

    # Force sequential execution
    rc = load_run_config(sample)
    rc = type(rc)(
        version=rc.version,
        datasets=rc.datasets,
        models=rc.models,
        run_id=rc.run_id,
        name=rc.name,
        description=rc.description,
        output_dir=rc.output_dir,
        random_seed=rc.random_seed,
        metric_bundles=rc.metric_bundles,
        truncation=rc.truncation,
        concurrency=type(rc.concurrency)(max_workers=1, per_model=1) if rc.concurrency else None,
        thresholds=rc.thresholds,
    )

    progress_events = []
    def on_progress(evt: Dict[str, Any]) -> None:
        progress_events.append(evt)

    result: OrchestratorSummary = evaluate_run(rc, on_progress=on_progress, get_response=_mock_get_response)

    # We should have one result per dataset x model
    assert len(result.results) == len(rc.datasets) * len(rc.models)
    assert not result.cancelled

    # Progress should include start, turn events, and end
    has_start = any(e.get("event") == "start" for e in progress_events)
    has_turn = any(e.get("event") == "turn" for e in progress_events)
    has_end = any(e.get("event") == "end" for e in progress_events)
    assert has_start and has_turn and has_end


def test_orchestrator_parallel_and_cancellation(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"

    rc = load_run_config(sample)
    # Increase workers to parallelize
    rc = type(rc)(
        version=rc.version,
        datasets=rc.datasets,
        models=rc.models,
        run_id=rc.run_id,
        name=rc.name,
        description=rc.description,
        output_dir=rc.output_dir,
        random_seed=rc.random_seed,
        metric_bundles=rc.metric_bundles,
        truncation=rc.truncation,
        concurrency=type(rc.concurrency)(max_workers=4, per_model=2) if rc.concurrency else None,
        thresholds=rc.thresholds,
    )

    cancel = threading.Event()

    # Cancel immediately after first progress callback
    first = True
    def on_progress(evt: Dict[str, Any]) -> None:
        nonlocal first
        if first:
            first = False
            cancel.set()

    result: OrchestratorSummary = evaluate_run(rc, cancel_event=cancel, on_progress=on_progress, get_response=_mock_get_response)

    # Some tasks should be cancelled or not started
    assert result.cancelled is True or any(r.status == "cancelled" for r in result.results)
