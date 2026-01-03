from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .config.run_config_loader import RunConfig, load_run_config
from .data.loader import load_conversation, load_golden
from .data.golden_access import index_golden
from .evaluation.scoring import score_turn
from .evaluation.weights import aggregate_conversation
from .queue import ExecutionQueue, JobState


# ---- Data types for reporting ----


@dataclass(frozen=True)
class TurnEvaluation:
    turn_id: str
    passed: bool
    matched_variant: Optional[str]


@dataclass(frozen=True)
class ConversationEvaluation:
    dataset_id: Optional[str]
    conversation_id: str
    model_name: str
    status: str  # "completed" | "cancelled" | "error"
    score: float
    turn_results: List[TurnEvaluation]


@dataclass(frozen=True)
class OrchestratorSummary:
    results: List[ConversationEvaluation]
    cancelled: bool


# Progress callback payload (kept simple for tests)
ProgressCallback = Callable[[Dict[str, Any]], None]


def default_get_response(turn: Mapping[str, Any], model: Mapping[str, Any], conversation: Mapping[str, Any]) -> str:
    """Default response provider: use existing assistant content.

    This enables running evaluations without contacting an LLM.
    """
    content = str(turn.get("content", "") or "")
    return content


def _evaluate_dataset_model(
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    cancel_event: threading.Event,
    on_progress: Optional[ProgressCallback],
    get_response: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str],
) -> ConversationEvaluation:
    ds_id = dataset.get("id")
    conv_path = dataset["conversation"]
    golden_path = dataset["golden"]

    if on_progress:
        on_progress({
            "event": "start",
            "dataset_id": ds_id,
            "model": model.get("name"),
            "conversation": conv_path,
        })

    # Load files
    conv = load_conversation(conv_path)
    gold = load_golden(golden_path)
    conv_id = str(conv.get("conversation_id"))

    # Build turn lookup and expected index
    turns = {str(t.get("turn_id")): t for t in (conv.get("turns") or [])}
    expected_index = index_golden(gold)

    # Evaluate only turns that have expectations
    turn_results: List[TurnEvaluation] = []
    turn_scores: Dict[str, float] = {}
    turn_weights: Dict[str, float] = {}

    # Collect expected turns for this conversation_id
    expected_turn_ids: List[str] = [
        tid for (cid, tid) in expected_index.keys() if cid == conv_id
    ]

    status = "completed"

    for tid in expected_turn_ids:
        if cancel_event.is_set():
            status = "cancelled"
            break
        t = turns.get(tid)
        if not t:  # turn missing; count as fail
            turn_results.append(TurnEvaluation(turn_id=tid, passed=False, matched_variant=None))
            turn_scores[tid] = 0.0
            turn_weights[tid] = 1.0
            if on_progress:
                on_progress({
                    "event": "turn",
                    "dataset_id": ds_id,
                    "model": model.get("name"),
                    "turn_id": tid,
                    "passed": False,
                })
            continue

        expected = expected_index.get((conv_id, tid), {})
        # Fetch actual text using callback
        actual_text = get_response(t, model, conv)
        res = score_turn(actual_text, expected)
        passed = bool(res.get("passed", False))
        turn_results.append(TurnEvaluation(turn_id=tid, passed=passed, matched_variant=res.get("matched_variant")))
        turn_scores[tid] = 1.0 if passed else 0.0
        # Per-turn weight if provided
        turn_weights[tid] = float(expected.get("turn_weight", 1.0) or 1.0)

        if on_progress:
            on_progress({
                "event": "turn",
                "dataset_id": ds_id,
                "model": model.get("name"),
                "turn_id": tid,
                "passed": passed,
            })

    # Aggregate conversation score (weighted over turns)
    score = aggregate_conversation(turn_scores, turn_weights if turn_weights else None)

    if on_progress:
        on_progress({
            "event": "end",
            "dataset_id": ds_id,
            "model": model.get("name"),
            "conversation_id": conv_id,
            "status": status,
            "score": score,
        })

    return ConversationEvaluation(
        dataset_id=ds_id,
        conversation_id=conv_id,
        model_name=str(model.get("name")),
        status=status,
        score=score,
        turn_results=turn_results,
    )


def evaluate_run(
    run: RunConfig | str | Path,
    *,
    cancel_event: Optional[threading.Event] = None,
    on_progress: Optional[ProgressCallback] = None,
    get_response: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str] = default_get_response,
) -> OrchestratorSummary:
    """Run an evaluation over all datasets x models.

    - Respects global concurrency from run.concurrency.max_workers. If not set or <=1, runs sequentially.
    - Supports cancellation via threading.Event.
    - Uses get_response callback to obtain the text to score for a turn.
    """
    if isinstance(run, (str, Path)):
        rc = load_run_config(run)
    else:
        rc = run

    cancel = cancel_event or threading.Event()
    max_workers = 1
    if rc.concurrency and rc.concurrency.max_workers and rc.concurrency.max_workers > 1:
        max_workers = int(rc.concurrency.max_workers)

    # Prepare tasks (dataset x model)
    tasks: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for ds in rc.datasets:
        ds_dict = {
            "id": ds.id,
            "conversation": ds.conversation,
            "golden": ds.golden,
        }
        for m in rc.models:
            m_dict = {
                "name": m.name,
                "provider": m.provider,
                "model": m.model,
                "params": m.params or {},
            }
            tasks.append((ds_dict, m_dict))

    results: List[ConversationEvaluation] = []
    cancelled_flag = False

    if max_workers <= 1:
        # Sequential execution
        for ds_dict, m_dict in tasks:
            if cancel.is_set():
                cancelled_flag = True
                break
            result = _evaluate_dataset_model(ds_dict, m_dict, cancel, on_progress, get_response)
            results.append(result)
            if result.status == "cancelled":
                cancelled_flag = True
                # Continue to drain or stop immediately; choose to stop
                break
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for ds_dict, m_dict in tasks:
                if cancel.is_set():
                    cancelled_flag = True
                    break
                futures.append(pool.submit(_evaluate_dataset_model, ds_dict, m_dict, cancel, on_progress, get_response))

            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if res.status == "cancelled":
                    cancelled_flag = True

    return OrchestratorSummary(results=results, cancelled=cancelled_flag)


def evaluate_run_with_queue(
    queue: ExecutionQueue,
    job_id: str,
    run: RunConfig | str | Path,
    *,
    on_progress: Optional[ProgressCallback] = None,
    get_response: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str] = default_get_response,
) -> OrchestratorSummary:
    """Wrapper that integrates the orchestrator with ExecutionQueue cancellation and state updates."""
    queue.start_job(job_id)
    summary = evaluate_run(run, cancel_event=queue.get_cancel_event(job_id), on_progress=on_progress, get_response=get_response)
    # Update final state based on summary.cancelled
    if summary.cancelled:
        queue.update_state(job_id, JobState.CANCELLED)
    else:
        queue.update_state(job_id, JobState.SUCCEEDED)
    queue.update_progress(job_id, 100.0)
    return summary


__all__ = [
    "TurnEvaluation",
    "ConversationEvaluation",
    "OrchestratorSummary",
    "evaluate_run",
]
