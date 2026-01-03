from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from eval_server.queue import ExecutionQueue, JobState


def test_add_and_query_job():
    q = ExecutionQueue()
    jid = q.add_job(metadata={"name": "job1"})
    job = q.get_job(jid)
    assert job["job_id"] == jid
    assert job["state"] == JobState.QUEUED.value
    assert job["progress"] == 0.0
    assert job["metadata"]["name"] == "job1"


def test_state_transitions_and_progress():
    q = ExecutionQueue()
    jid = q.add_job()

    q.update_state(jid, JobState.RUNNING)
    q.update_progress(jid, 10)
    q.increment_progress(jid, 15)
    q.increment_progress(jid, 100)  # clamps at 100

    job = q.get_job(jid)
    assert job["state"] == JobState.RUNNING.value
    assert job["progress"] == 100.0

    q.update_state(jid, JobState.SUCCEEDED, message="done")
    job = q.get_job(jid)
    assert job["state"] == JobState.SUCCEEDED.value
    assert job["message"] == "done"


def test_cancel_and_list_delete():
    q = ExecutionQueue()
    jid1 = q.add_job()
    jid2 = q.add_job()

    q.cancel_job(jid1, reason="user request")

    cancelled = q.list_jobs(state=JobState.CANCELLED)
    assert len(cancelled) == 1
    assert cancelled[0]["job_id"] == jid1

    q.delete_job(jid2)

    listed = q.list_jobs()
    ids = {j["job_id"] for j in listed}
    assert jid2 not in ids


def test_thread_safety_concurrent_updates():
    q = ExecutionQueue()
    jid = q.add_job()

    def worker():
        for _ in range(1000):
            q.increment_progress(jid, 0.1)
            q.update_state(jid, JobState.RUNNING)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker) for _ in range(8)]
        for f in futures:
            f.result()

    job = q.get_job(jid)
    # Progress must be clamped to 100 and state should be RUNNING
    assert job["progress"] == 100.0
    assert job["state"] == JobState.RUNNING.value
