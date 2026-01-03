from __future__ import annotations

import threading
import time
from pathlib import Path

from eval_server.config.run_config_loader import load_run_config
from eval_server.orchestrator import evaluate_run_with_queue
from eval_server.queue import ExecutionQueue, JobState


def test_cancellation_stops_evaluation(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"

    # Force sequential to make ordering/cancel deterministic
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

    q = ExecutionQueue()
    jid = q.add_job()

    # Start evaluation in a thread
    def runner():
        evaluate_run_with_queue(q, jid, rc)

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    # Allow it to start and then cancel
    time.sleep(0.05)
    q.cancel_job(jid, reason="test cancel")

    t.join(timeout=5)

    job = q.get_job(jid)
    assert job["state"] == JobState.CANCELLED.value
    # Progress should be <= 100 and updated
    assert 0.0 <= job["progress"] <= 100.0
