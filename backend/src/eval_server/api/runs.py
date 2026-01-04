from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Mapping
import threading

from fastapi import APIRouter, HTTPException, Query, Header
from ..utils.errors import NotFoundError, BadRequestError
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from ..config.run_config_loader import (
    RunConfig,
    DatasetConfig,
    ModelConfig,
    TruncationPolicy,
    GlobalConcurrency,
    Thresholds,
    load_run_config,
)
from ..orchestrator import evaluate_run
from ..headless_engine import run_headless
from ..utils.run_id import compute_run_id
from .queue_context import queue as _queue
from .progress_registry import registry as _progress


router = APIRouter(prefix="/runs", tags=["runs"])


class DatasetConfigModel(BaseModel):
    id: str
    conversation: str
    golden: str
    tags: Optional[list[str]] = None
    difficulty: Optional[str] = None

class ModelConfigModel(BaseModel):
    name: str
    provider: str
    model: str
    params: Optional[Dict[str, Any]] = None

class RunStartRequest(BaseModel):
    version: str
    datasets: list[DatasetConfigModel]
    models: list[ModelConfigModel]
    run_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    output_dir: Optional[str] = None
    random_seed: Optional[int] = None
    metric_bundles: Optional[list[str]] = None
    truncation: Optional[Dict[str, Any]] = None
    concurrency: Optional[Dict[str, Any]] = None
    thresholds: Optional[Dict[str, Any]] = None


class RunStartResponse(BaseModel):
    run_id: str
    job_id: str
    status: str


@router.post("/", response_model=RunStartResponse, summary="Start a run", description="Queue and start an evaluation run based on the provided configuration.")
def start_run(req: RunStartRequest) -> RunStartResponse:
    # Validate by round-tripping through RunConfig dataclass via loader-like behavior
    # Save request to temp file-like structure not needed; we can construct RunConfig by reusing schema via load_run_config
    # But load_run_config expects a path; instead, we mimic minimal validation by ensuring required keys exist
    try:
        # Basic required fields check
        if not req.version or len(req.datasets) == 0 or len(req.models) == 0:
            raise ValueError("invalid run configuration")
    except Exception as e:
        raise BadRequestError(str(e))

    # Compute run_id (deterministic) using compute_run_id requires RunConfig, so construct it with loader by writing to a temp JSON in memory is not supported.
    # We fallback to using compute_run_id on the dict directly (it accepts Mapping), which will read versions from files.
    # Build RunConfig dataclass
    try:
        datasets = [
            DatasetConfig(
                id=d.id,
                conversation=d.conversation,
                golden=d.golden,
                tags=d.tags,
                difficulty=d.difficulty,
            )
            for d in req.datasets
        ]
        models = [
            ModelConfig(
                name=m.name,
                provider=m.provider,
                model=m.model,
                params=m.params,
                concurrency=None,
            )
            for m in req.models
        ]
        trunc = None
        if req.truncation is not None:
            t = req.truncation
            trunc = TruncationPolicy(strategy=t.get("strategy", "none"), max_input_tokens=t.get("max_input_tokens"))
        conc = None
        if req.concurrency is not None:
            c = req.concurrency
            conc = GlobalConcurrency(max_workers=c.get("max_workers"), per_model=c.get("per_model"))
        thr = None
        if req.thresholds is not None:
            t = req.thresholds
            thr = Thresholds(turn_pass=t.get("turn_pass"), conversation_pass=t.get("conversation_pass"), metric=t.get("metric"))

        rc = RunConfig(
            version=req.version,
            datasets=datasets,
            models=models,
            run_id=req.run_id,
            name=req.name,
            description=req.description,
            output_dir=req.output_dir,
            random_seed=req.random_seed,
            metric_bundles=req.metric_bundles,
            truncation=trunc,
            concurrency=conc,
            thresholds=thr,
        )
    except KeyError as e:
        raise BadRequestError(f"missing field: {e}")
    except Exception as e:
        raise BadRequestError(f"invalid run configuration: {e}")

    run_id = compute_run_id(rc)

    # Queue job and execute in a background thread using orchestrator with cancel event support
    job_id = _queue.add_job()
    _progress.init_run(run_id)

    def _worker() -> None:
        try:
            _queue.start_job(job_id)
            # Use headless engine to produce full artifacts (raw, normalized, scores, results.json)
            def _on_progress(evt: Dict[str, Any]) -> None:  # type: ignore[name-defined]
                _progress.record_event(run_id, evt)
            # Single-pass execution: generate artifacts and forward progress; support cancellation
            run_headless(
                rc,
                output_dir=None,
                on_progress=_on_progress,
                cancel_event=_queue.get_cancel_event(job_id),
            )
            _queue.update_state(job_id, JobState.SUCCEEDED)  # type: ignore[name-defined]
        except Exception as e:  # pragma: no cover - unexpected runtime errors
            _queue.update_state(job_id, JobState.FAILED, message=str(e))  # type: ignore[name-defined]

    # Late import to avoid cycle
    from ..queue import JobState
    t = threading.Thread(target=_worker, name=f"run-{run_id}", daemon=True)
    t.start()

    return RunStartResponse(run_id=run_id, job_id=job_id, status="queued")


# --- Conversation metrics endpoint ---

from fastapi import Path as _PathParam
import json as _json
from .results import _find_run_dir as _find_run_dir_results


@router.get("/{run_id}/conversations/{conversation_id}", summary="Conversation metrics", description="Return per-turn outputs, normalized canonical, scores, and thresholds for a conversation in a run.")
def get_conversation_metrics(
    run_id: str,
    conversation_id: str = _PathParam(..., description="Conversation ID to retrieve"),
    model_name: str | None = Query(None, description="Optional model filter. When set, only this model's results are returned."),
):
    """Return per-turn outputs, normalized canonical, scores, and thresholds for a conversation in a run.

    Response structure:
    {
      "run_id": str,
      "conversation_id": str,
      "results": [
         {
           "dataset_id": str,
           "model_name": str,
           "aggregate": {"score": float, "passed": bool},
           "thresholds": {"turn_pass": float, "conversation_pass": float, "metric": {...}},
           "turns": [ { "turn_id": ..., "prompt": ..., "response": ..., "context": ..., "canonical": ..., "metrics": {...}, "weights": {...}, "weighted_score": float, "passed": bool } ]
         }
      ]
    }
    """
    try:
        run_dir = _find_run_dir_results(run_id)
    except NotFoundError as e:
        raise e

    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise NotFoundError("results not found for run")

    data = _json.loads(results_path.read_text(encoding="utf-8"))
    all_results = data.get("results") or []
    # Filter entries for this conversation_id
    filtered = [r for r in all_results if str(r.get("conversation_id")) == str(conversation_id)]

    # Load thresholds from stored run config if available
    thresholds = None
    cfg_path = run_dir / "inputs" / "run_config.json"
    if cfg_path.exists():
        try:
            cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
            thr = cfg.get("thresholds") or {}
            thresholds = {
                "turn_pass": thr.get("turn_pass"),
                "conversation_pass": thr.get("conversation_pass", thr.get("turn_pass")),
                "metric": thr.get("metric", {}),
            }
        except Exception:
            thresholds = None

    # Optionally filter by model
    if model_name:
        filtered = [r for r in filtered if str(r.get("model_name")) == str(model_name)]

    # Attach thresholds to each per-model entry
    for r in filtered:
        r["thresholds"] = thresholds

    return {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "results": filtered,
    }


@router.get("/{run_id}/artifacts", summary="Download run artifacts", description="Download one or more run artifacts (results, summary, csv, html, markdown, raw, normalized, turn_scores, progress). Returns a file or a ZIP when multiple.")
def download_artifacts(
    run_id: str,
    artifact: list[str] = Query(["results"], description="Artifact(s) to download", alias="artifact"),
    theme: str = Query("default", enum=["default", "dark", "compact"], description="Theme for HTML report if generated"),
):
    """Download run artifacts.

    Supported artifacts:
    - results: consolidated results.json
    - summary: summary.json
    - csv: results.csv (generated on demand)
    - html: report.html (generated on demand)
    - markdown: report.md (generated on demand)

    If multiple artifacts are requested, a ZIP archive is returned.
    """
    allowed = {"results", "summary", "csv", "html", "markdown", "raw", "normalized", "turn_scores", "progress"}
    req = [a.lower() for a in (artifact or ["results"]) if a]
    if not req:
        req = ["results"]
    for a in req:
        if a not in allowed:
            raise HTTPException(status_code=400, detail=f"invalid artifact: {a}")

    try:
        run_dir = _find_run_dir_results(run_id)
    except NotFoundError:
        # Normalize error shape for clients expecting HTTPException body
        raise HTTPException(status_code=404, detail="run not found")

    # Resolve base paths
    results_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="results not found for run")

    # Generate derivatives on demand
    from ..reporting.csv_export import export_results_csv
    from ..reporting.report_generator import generate_html_report
    from ..reporting.markdown_export import generate_markdown_report

    resolved_files: list[tuple[str, str]] = []  # (name, path)

    for item in req:
        if item == "results":
            resolved_files.append(("results.json", str(results_path)))
        elif item == "summary":
            if not summary_path.exists():
                raise NotFoundError("summary not found for run")
            resolved_files.append(("summary.json", str(summary_path)))
        elif item == "csv":
            csv_path = results_path.with_suffix(".csv")
            if not csv_path.exists():
                try:
                    export_results_csv(results_path, csv_path)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"failed to generate csv: {e}")
            resolved_files.append((csv_path.name, str(csv_path)))
        elif item == "html":
            html_path = results_path.with_name("report.html")
            if not html_path.exists():
                try:
                    generate_html_report(results_path, theme=theme)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"failed to generate html report: {e}")
            resolved_files.append((html_path.name, str(html_path)))
        elif item == "markdown":
            md_path = results_path.with_name("report.md")
            if not md_path.exists():
                try:
                    generate_markdown_report(results_path, md_path)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"failed to generate markdown report: {e}")
            resolved_files.append((md_path.name, str(md_path)))
        elif item == "raw":
            raw_path = run_dir / "raw_outputs.jsonl"
            if not raw_path.exists():
                raise NotFoundError("raw outputs not found for run")
            resolved_files.append((raw_path.name, str(raw_path)))
        elif item == "normalized":
            norm_path = run_dir / "normalized.jsonl"
            if not norm_path.exists():
                raise NotFoundError("normalized outputs not found for run")
            resolved_files.append((norm_path.name, str(norm_path)))
        elif item == "turn_scores":
            ts_path = run_dir / "scores" / "turn_scores.jsonl"
            if not ts_path.exists():
                raise NotFoundError("turn scores not found for run")
            resolved_files.append((ts_path.name, str(ts_path)))
        elif item == "progress":
            pr_path = run_dir / "logs" / "progress.jsonl"
            if not pr_path.exists():
                raise NotFoundError("progress log not found for run")
            resolved_files.append((pr_path.name, str(pr_path)))

    # Single file: return directly
    if len(resolved_files) == 1:
        name, path = resolved_files[0]
        media = "application/octet-stream"
        if name.endswith(".json"):
            media = "application/json"
        elif name.endswith(".csv"):
            media = "text/csv"
        elif name.endswith(".html"):
            media = "text/html"
        elif name.endswith(".md"):
            media = "text/markdown"
        return FileResponse(path, media_type=media, filename=name)

    # Multiple: zip up in-memory
    import io, zipfile, os
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, path in resolved_files:
            # Ensure unique names in archive by prefixing run id folder name
            arcname = name
            zf.write(path, arcname=arcname)
    buf.seek(0)
    headers = {"Content-Disposition": f"attachment; filename={run_id}_artifacts.zip"}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


# ---- Run-scoped feedback endpoint ----
from pydantic import BaseModel, Field
from typing import List
from eval_server.utils.audit import log_audit_event as _log_audit_event


class RunTurnFeedback(BaseModel):
    dataset_id: str
    conversation_id: str
    model_name: str
    turn_id: int | str
    rating: float | None = Field(default=None, ge=0.0, le=5.0)
    notes: str | None = None
    override_pass: bool | None = None
    override_score: float | None = Field(default=None, ge=0.0, le=1.0)
    actor: str | None = None
    ts: float | None = None


class RunFeedbackRequest(BaseModel):
    feedback: List[RunTurnFeedback]


class RunFeedbackResponse(BaseModel):
    run_id: str
    stored_path: str
    total_records: int


def _load_json(path: Path) -> dict:
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _feedback_path_for_run(run_dir: Path) -> Path:
    return run_dir / "annotations.json"


@router.post("/{run_id}/feedback", response_model=RunFeedbackResponse, summary="Submit feedback for run", description="Submit evaluator feedback for specific conversation turns in a run. Validates references against results.json.")
def submit_run_feedback(
    run_id: str,
    req: RunFeedbackRequest,
    x_user: str | None = Header(default=None, alias="X-User"),
) -> RunFeedbackResponse:
    # Ensure run exists and results.json is present
    try:
        run_dir = _find_run_dir_results(run_id)
    except NotFoundError as e:
        raise e

    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise NotFoundError("results not found for run")

    # Build an index of valid (dataset_id, conversation_id, model_name, turn_id)
    results_data = _load_json(results_path)
    valid_keys: set[tuple[str, str, str, str]] = set()
    for grp in results_data.get("results", []) or []:
        ds = str(grp.get("dataset_id"))
        cid = str(grp.get("conversation_id"))
        model = str(grp.get("model_name"))
        for t in grp.get("turns", []) or []:
            tid = str(t.get("turn_id"))
            valid_keys.add((ds, cid, model, tid))

    # Validate all feedback entries before writing (atomic behavior)
    errors: list[str] = []
    for i, fb in enumerate(req.feedback or []):
        key = (str(fb.dataset_id), str(fb.conversation_id), str(fb.model_name), str(fb.turn_id))
        if key not in valid_keys:
            errors.append(f"feedback[{i}] references unknown item: {key}")

    if errors:
        # For this endpoint, return a simple validation shape expected by tests
        return JSONResponse(status_code=400, content={"detail": {"errors": errors}})

    # Load existing annotations
    store_path = _feedback_path_for_run(run_dir)
    existing = _load_json(store_path)
    annotations = list(existing.get("annotations") or [])
    # Append all new feedback (as dicts)
    import time as _time
    new_items = []
    for fb in (req.feedback or []):
        rec = fb.model_dump()
        if not rec.get("actor") and x_user:
            rec["actor"] = x_user
        if not rec.get("ts"):
            rec["ts"] = _time.time()
        new_items.append(rec)
    annotations.extend(new_items)
    payload = {"run_id": run_id, "annotations": annotations}
    store_path.write_text(_json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Audit log
    _log_audit_event(
        action="feedback_submitted",
        actor=x_user,
        run_id=run_id,
        source="api",
        config_path=None,
        config_fingerprint=None,
        details={"records": len(new_items), "scoped": "run"},
    )

    return RunFeedbackResponse(run_id=run_id, stored_path=str(store_path.resolve()), total_records=len(new_items))


@router.get("/{run_id}/feedback", summary="Get feedback for run", description="Retrieve stored annotations (evaluator feedback) for a run.")
def get_run_feedback(run_id: str):
    try:
        run_dir = _find_run_dir_results(run_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    path = _feedback_path_for_run(run_dir)
    if not path.exists():
        return {"run_id": run_id, "annotations": []}
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read feedback: {e}")


# ---- Runs comparison endpoint ----

def _load_results_payload(run_id: str) -> Dict[str, Any]:
    try:
        rdir = _find_run_dir_results(run_id)
    except NotFoundError:
        raise NotFoundError(f"run not found: {run_id}")
    rpath = rdir / "results.json"
    if not rpath.exists():
        raise HTTPException(status_code=404, detail=f"results not found for run: {run_id}")
    try:
        return _json.loads(rpath.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read results for run {run_id}: {e}")


@router.get("/compare", summary="Compare two runs", description="Compare baseline vs current runs and return deltas overall, per dataset, per metric (averaged), and per conversation.")
def compare_runs(
    baseline: str = Query(..., description="Baseline run_id"),
    current: str = Query(..., description="Current run_id"),
):
    """Compare two runs and return regression summaries and per-dataset metric deltas.

    Response fields:
    - summary.overall: average conversation aggregate score delta
    - per_dataset: aggregated conversation score delta per dataset
    - metrics_by_dataset: per-metric averages over turns per dataset with deltas
    - per_conversation: per (dataset,conversation,model) aggregate deltas (overlap only)
    """
    base = _load_results_payload(baseline)
    curr = _load_results_payload(current)

    b_results = base.get("results") or []
    c_results = curr.get("results") or []

    # Index aggregate scores by key
    def _key(g: Mapping[str, Any]) -> str:
        return f"{g.get('dataset_id')}|{g.get('conversation_id')}|{g.get('model_name')}"

    b_map: Dict[str, Mapping[str, Any]] = {_key(g): g for g in b_results}
    c_map: Dict[str, Mapping[str, Any]] = {_key(g): g for g in c_results}

    common_keys = sorted(set(b_map.keys()) & set(c_map.keys()))
    baseline_only = sorted(set(b_map.keys()) - set(c_map.keys()))
    current_only = sorted(set(c_map.keys()) - set(b_map.keys()))

    per_conv: list[Dict[str, Any]] = []
    b_scores_all: list[float] = []
    c_scores_all: list[float] = []
    for k in common_keys:
        b = b_map[k]
        c = c_map[k]
        bs = float((b.get("aggregate") or {}).get("score", 0.0))
        cs = float((c.get("aggregate") or {}).get("score", 0.0))
        b_scores_all.append(bs)
        c_scores_all.append(cs)
        ds, cid, model = str(b.get("dataset_id")), str(b.get("conversation_id")), str(b.get("model_name"))
        per_conv.append({
            "dataset_id": ds,
            "conversation_id": cid,
            "model_name": model,
            "baseline": bs,
            "current": cs,
            "delta": cs - bs,
        })

    # Overall
    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0
    overall_baseline = _avg(b_scores_all)
    overall_current = _avg(c_scores_all)

    # Per-dataset aggregation (average of conversation aggregate scores)
    from collections import defaultdict
    b_ds_scores: Dict[str, list[float]] = defaultdict(list)
    for g in b_results:
        b_ds_scores[str(g.get("dataset_id"))].append(float((g.get("aggregate") or {}).get("score", 0.0)))
    c_ds_scores: Dict[str, list[float]] = defaultdict(list)
    for g in c_results:
        c_ds_scores[str(g.get("dataset_id"))].append(float((g.get("aggregate") or {}).get("score", 0.0)))

    all_ds = sorted(set(b_ds_scores.keys()) | set(c_ds_scores.keys()))
    per_dataset: list[Dict[str, Any]] = []
    for ds in all_ds:
        bavg = _avg(b_ds_scores.get(ds, []))
        cavg = _avg(c_ds_scores.get(ds, []))
        per_dataset.append({
            "dataset_id": ds,
            "baseline": bavg,
            "current": cavg,
            "delta": cavg - bavg,
        })

    # Per-dataset per-metric averages over turns
    def _collect_metric_avgs(results: list[Mapping[str, Any]]) -> Dict[tuple[str, str], tuple[float, int]]:
        sums: Dict[tuple[str, str], float] = defaultdict(float)
        counts: Dict[tuple[str, str], int] = defaultdict(int)
        for g in results:
            ds = str(g.get("dataset_id"))
            for t in g.get("turns", []) or []:
                metrics = t.get("metrics") or {}
                for name, val in metrics.items():
                    key = (ds, str(name))
                    try:
                        v = float(val)
                    except Exception:
                        v = 0.0
                    sums[key] += v
                    counts[key] += 1
        return {k: (sums[k], counts[k]) for k in sums.keys()}

    b_metric = _collect_metric_avgs(b_results)
    c_metric = _collect_metric_avgs(c_results)
    keys_metric = sorted(set(b_metric.keys()) | set(c_metric.keys()))
    metrics_by_dataset: list[Dict[str, Any]] = []
    for (ds, metric) in keys_metric:
        bsum, bcnt = b_metric.get((ds, metric), (0.0, 0))
        csum, ccnt = c_metric.get((ds, metric), (0.0, 0))
        bavg = (bsum / bcnt) if bcnt else 0.0
        cavg = (csum / ccnt) if ccnt else 0.0
        metrics_by_dataset.append({
            "dataset_id": ds,
            "metric": metric,
            "baseline": bavg,
            "current": cavg,
            "delta": cavg - bavg,
        })

    return {
        "baseline_run_id": baseline,
        "current_run_id": current,
        "summary": {
            "overall": {
                "baseline": overall_baseline,
                "current": overall_current,
                "delta": overall_current - overall_baseline,
            },
            "counts": {
                "conversations_common": len(common_keys),
                "baseline_only": len(baseline_only),
                "current_only": len(current_only),
            },
        },
        "per_dataset": per_dataset,
        "metrics_by_dataset": metrics_by_dataset,
        "per_conversation": per_conv,
    }
