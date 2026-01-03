from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import RLock, Event
from typing import Any, Dict, List, Optional


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    state: JobState = JobState.QUEUED
    progress: float = 0.0  # percentage 0..100
    message: Optional[str] = None
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Enum to value
        d["state"] = self.state.value
        return d


class ExecutionQueue:
    """Thread-safe, in-memory execution queue.

    Supports adding jobs, updating state/progress, querying, and cancellation.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = RLock()
        self._cancel_events: Dict[str, Event] = {}

    # ---- helpers ----
    def _now(self) -> float:
        return time.time()

    def _get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as e:
            raise KeyError(f"Job not found: {job_id}") from e

    # ---- API ----
    def add_job(self, job_id: Optional[str] = None, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new job in queued state and return its id.

        If job_id is not provided, a UUID4 string is generated.
        """
        with self._lock:
            jid = job_id or str(uuid.uuid4())
            if jid in self._jobs:
                raise ValueError(f"Job already exists: {jid}")
            job = Job(job_id=jid, state=JobState.QUEUED, progress=0.0, metadata=dict(metadata or {}))
            self._jobs[jid] = job
            return jid

    def start_job(self, job_id: str) -> None:
        """Mark a job as RUNNING and initialize its cancel event."""
        with self._lock:
            job = self._get(job_id)
            job.state = JobState.RUNNING
            job.updated_at = self._now()
            if job_id not in self._cancel_events:
                self._cancel_events[job_id] = Event()

    def update_state(self, job_id: str, state: JobState, *, message: Optional[str] = None) -> None:
        with self._lock:
            job = self._get(job_id)
            job.state = state
            if message is not None:
                job.message = message
            job.updated_at = self._now()

    def update_progress(self, job_id: str, progress: float) -> None:
        """Set absolute progress as a percentage [0, 100]."""
        with self._lock:
            job = self._get(job_id)
            # clamp to 0..100
            p = max(0.0, min(100.0, float(progress)))
            job.progress = p
            job.updated_at = self._now()

    def increment_progress(self, job_id: str, delta: float) -> None:
        """Increment progress by delta; clamps to [0, 100]."""
        with self._lock:
            job = self._get(job_id)
            p = max(0.0, min(100.0, float(job.progress) + float(delta)))
            job.progress = p
            job.updated_at = self._now()

    def set_message(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._get(job_id)
            job.message = message
            job.updated_at = self._now()

    def cancel_job(self, job_id: str, *, reason: Optional[str] = None) -> None:
        with self._lock:
            job = self._get(job_id)
            job.state = JobState.CANCELLED
            if reason:
                job.message = reason
            job.updated_at = self._now()
            # Signal cancellation to any workers observing this job
            ev = self._cancel_events.get(job_id)
            if ev is None:
                ev = Event()
                self._cancel_events[job_id] = ev
            ev.set()

    def get_cancel_event(self, job_id: str) -> Event:
        """Return the cancellation Event associated with the job (created if missing)."""
        with self._lock:
            ev = self._cancel_events.get(job_id)
            if ev is None:
                ev = Event()
                self._cancel_events[job_id] = ev
            return ev

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._get(job_id)
            return job.to_dict()

    def list_jobs(self, state: Optional[JobState] = None) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
            if state is not None:
                jobs = [j for j in jobs if j.state == state]
            return [j.to_dict() for j in jobs]

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                self._cancel_events.pop(job_id, None)
            else:
                raise KeyError(f"Job not found: {job_id}")


__all__ = [
    "JobState",
    "Job",
    "ExecutionQueue",
]
