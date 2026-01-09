from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import json

try:
    from .dataset_repo import DatasetRepository
    from .orchestrator import Orchestrator
    from .artifacts import RunArtifactWriter, RunArtifactReader
    from .reporter import Reporter
except ImportError:  # fallback for test runs importing as top-level modules
    from backend.dataset_repo import DatasetRepository
    from backend.orchestrator import Orchestrator
    from backend.artifacts import RunArtifactWriter, RunArtifactReader
    from backend.reporter import Reporter
    from backend.coverage_builder import (
        build_per_behavior_datasets,
        build_domain_combined_datasets,
        build_global_combined_dataset,
    )
    from backend.coverage_manifest import CoverageManifestor
else:
    from .coverage_builder import (
        build_per_behavior_datasets,
        build_domain_combined_datasets,
        build_global_combined_dataset,
    )
    from .coverage_manifest import CoverageManifestor
    from .coverage_reports import coverage_summary_csv, coverage_heatmap_csv, per_turn_csv

APP_VERSION = "0.1.0-mvp"

# Load .env from repo root (dev convenience)
def _load_env_from_file() -> None:
    try:
        root = Path(__file__).resolve().parents[1]
        env_path = root / '.env'
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip(); v = v.strip()
            if k and v and k not in os.environ:
                os.environ[k] = v
    except Exception:
        # non-fatal
        pass

_load_env_from_file()

class Health(BaseModel):
    status: str

class VersionInfo(BaseModel):
    version: str
    gemini_enabled: bool
    ollama_host: str | None
    semantic_threshold: float
    openai_enabled: bool | None = None
    models: dict[str, str] | None = None
    hallucination_threshold: float | None = None


class SettingsBody(BaseModel):
    ollama_host: Optional[str] = None
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    semantic_threshold: Optional[float] = None
    hallucination_threshold: Optional[float] = None
    ollama_model: Optional[str] = None
    gemini_model: Optional[str] = None
    openai_model: Optional[str] = None
    embed_model: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None  # persisted UI toggles


def get_settings():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    semantic_threshold = float(os.getenv("SEMANTIC_THRESHOLD", "0.80"))
    hallucination_threshold = float(os.getenv("HALLUCINATION_THRESHOLD", "0.80"))
    # models per provider (defaults)
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5.1")
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    # Load persisted metrics config if present
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / 'configs' / 'metrics.json'
    metrics_cfg = None
    try:
        if cfg_path.exists():
            metrics_cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    except Exception:
        metrics_cfg = None
    return {
        "GOOGLE_API_KEY": google_api_key,
        "OPENAI_API_KEY": openai_api_key,
        "OLLAMA_HOST": ollama_host,
        "SEMANTIC_THRESHOLD": semantic_threshold,
        "HALLUCINATION_THRESHOLD": hallucination_threshold,
        "METRICS_CFG": metrics_cfg,
        "DEFAULT_MODELS": {"ollama": ollama_model, "gemini": gemini_model, "openai": openai_model},
        "EMBED_MODEL": embed_model,
    }

app = FastAPI(title="LLM Eval Backend", version=APP_VERSION)

# CORS for local dev (frontend on Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App state singletons
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
import uuid
BOOT_ID = str(uuid.uuid4())
app.state.orch = Orchestrator(runs_root=RUNS_ROOT, boot_id=BOOT_ID)
app.state.artifacts = RunArtifactWriter(RUNS_ROOT)
app.state.reader = RunArtifactReader(RUNS_ROOT)
app.state.reporter = Reporter(Path(__file__).resolve().parent / "templates")


class StartRunRequest(BaseModel):
    dataset_id: str
    model_spec: str  # e.g., "ollama:llama3.2:latest" or "gemini:gemini-2.5"
    metrics: Optional[list[str]] = None
    thresholds: Optional[dict[str, Any]] = None
    context: Optional[dict[str, Any]] = None

class StartRunResponse(BaseModel):
    job_id: str
    run_id: str
    state: str

class ControlBody(BaseModel):
    action: str  # 'pause' | 'resume' | 'cancel'

@app.get("/health", response_model=Health)
async def health():
    return Health(status="ok")

@app.get("/version", response_model=VersionInfo)
async def version():
    s = get_settings()
    return VersionInfo(
        version=APP_VERSION,
        gemini_enabled=bool(s["GOOGLE_API_KEY"]),
        ollama_host=s["OLLAMA_HOST"],
        semantic_threshold=s["SEMANTIC_THRESHOLD"],
        openai_enabled=bool(s.get("OPENAI_API_KEY")),
        models=s.get("DEFAULT_MODELS"),
        hallucination_threshold=s.get("HALLUCINATION_THRESHOLD"),
    )


@app.get("/settings")
async def get_settings_api():
    s = get_settings()
    return {
        "ollama_host": s["OLLAMA_HOST"],
        "gemini_enabled": bool(s["GOOGLE_API_KEY"]),
        "openai_enabled": bool(s["OPENAI_API_KEY"]),
        "semantic_threshold": s["SEMANTIC_THRESHOLD"],
        "hallucination_threshold": s["HALLUCINATION_THRESHOLD"],
        "metrics": s.get("METRICS_CFG"),
        "models": s.get("DEFAULT_MODELS"),
        "embed_model": s.get("EMBED_MODEL"),
    }


@app.post("/settings")
async def update_settings_api(body: SettingsBody):
    """Dev-only: update .env in repo root. Do NOT store secrets elsewhere.
    """
    # .env at repo root
    root = Path(__file__).resolve().parents[1]
    env_path = root / '.env'
    # Load existing
    env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()
    # Update
    if body.ollama_host is not None:
        env['OLLAMA_HOST'] = body.ollama_host
    if body.google_api_key is not None:
        env['GOOGLE_API_KEY'] = body.google_api_key
    if body.openai_api_key is not None:
        env['OPENAI_API_KEY'] = body.openai_api_key
    if body.semantic_threshold is not None:
        env['SEMANTIC_THRESHOLD'] = str(body.semantic_threshold)
    if body.hallucination_threshold is not None:
        env['HALLUCINATION_THRESHOLD'] = str(body.hallucination_threshold)
    if body.ollama_model is not None:
        env['OLLAMA_MODEL'] = body.ollama_model
    if body.gemini_model is not None:
        env['GEMINI_MODEL'] = body.gemini_model
    if body.openai_model is not None:
        env['OPENAI_MODEL'] = body.openai_model
    if body.embed_model is not None:
        env['EMBED_MODEL'] = body.embed_model
    # Write
    lines = [f"{k}={v}" for k, v in env.items()]
    env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Also update process env for current process
    if 'OLLAMA_HOST' in env: os.environ['OLLAMA_HOST'] = env['OLLAMA_HOST']
    if 'GOOGLE_API_KEY' in env: os.environ['GOOGLE_API_KEY'] = env['GOOGLE_API_KEY']
    if 'OPENAI_API_KEY' in env: os.environ['OPENAI_API_KEY'] = env['OPENAI_API_KEY']
    if 'SEMANTIC_THRESHOLD' in env: os.environ['SEMANTIC_THRESHOLD'] = env['SEMANTIC_THRESHOLD']
    if 'HALLUCINATION_THRESHOLD' in env: os.environ['HALLUCINATION_THRESHOLD'] = env['HALLUCINATION_THRESHOLD']
    if 'OLLAMA_MODEL' in env: os.environ['OLLAMA_MODEL'] = env['OLLAMA_MODEL']
    if 'GEMINI_MODEL' in env: os.environ['GEMINI_MODEL'] = env['GEMINI_MODEL']
    if 'OPENAI_MODEL' in env: os.environ['OPENAI_MODEL'] = env['OPENAI_MODEL']
    if 'EMBED_MODEL' in env: os.environ['EMBED_MODEL'] = env['EMBED_MODEL']
    return {"ok": True}


@app.get("/embeddings/test")
async def embeddings_test():
    """Quick check to validate embeddings endpoint and model are working."""
    try:
        # defer import to avoid import-time failures
        try:
            from .embeddings.ollama_embed import OllamaEmbeddings
        except ImportError:
            from backend.embeddings.ollama_embed import OllamaEmbeddings
        emb = OllamaEmbeddings()
        vecs = await emb.embed(["hello", "world"])
        if not isinstance(vecs, list) or not vecs or not isinstance(vecs[0], list):
            raise RuntimeError("unexpected embeddings shape")
        dim = len(vecs[0])
        return {"ok": True, "count": len(vecs), "dim": dim, "model": os.getenv("EMBED_MODEL", "nomic-embed-text"), "host": os.getenv("OLLAMA_HOST", "http://localhost:11434")}
    except Exception as e:
        from fastapi import Response
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
async def list_datasets():
    repo: DatasetRepository = app.state.orch.repo
    return repo.list_datasets()


@app.post("/datasets/upload")
async def upload_dataset(dataset: UploadFile = File(...), golden: Optional[UploadFile] = File(None), overwrite: bool = False):
    """Upload a dataset (.dataset.json) and optional golden (.golden.json).
    Validates against schemas and writes to the datasets folder.
    """
    repo: DatasetRepository = app.state.orch.repo
    datasets_dir: Path = repo.root_dir
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Read and validate dataset JSON
    try:
        dataset_text = (await dataset.read()).decode("utf-8")
        dataset_obj = json.loads(dataset_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset JSON: {e}")

    ds_errors = repo.sv.validate("dataset", dataset_obj)
    if ds_errors:
        raise HTTPException(status_code=400, detail={"type": "dataset", "errors": ds_errors})

    dataset_id = dataset_obj.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id missing in dataset JSON")

    # Paths
    ds_path = datasets_dir / f"{dataset_id}.dataset.json"
    gt_path = datasets_dir / f"{dataset_id}.golden.json"

    # Golden optional
    golden_saved = False
    if golden is not None:
        try:
            golden_text = (await golden.read()).decode("utf-8")
            golden_obj = json.loads(golden_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid golden JSON: {e}")
        gt_errors = repo.sv.validate("golden", golden_obj)
        if gt_errors:
            raise HTTPException(status_code=400, detail={"type": "golden", "errors": gt_errors})
        # Ensure same dataset_id
        if golden_obj.get("dataset_id") != dataset_id:
            raise HTTPException(status_code=400, detail="Golden dataset_id must match dataset dataset_id")

        if gt_path.exists() and not overwrite:
            raise HTTPException(status_code=409, detail=f"Golden already exists: {gt_path.name}. Set overwrite=true to replace.")
        gt_path.write_text(json.dumps(golden_obj, indent=2), encoding="utf-8")
        golden_saved = True

    # Write dataset (after golden validation)
    if ds_path.exists() and not overwrite:
        raise HTTPException(status_code=409, detail=f"Dataset already exists: {ds_path.name}. Set overwrite=true to replace.")
    ds_path.write_text(json.dumps(dataset_obj, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "dataset_saved": True,
        "golden_saved": golden_saved,
        "files": {"dataset": ds_path.name, "golden": gt_path.name if golden_saved else None},
    }


@app.get("/datasets/{dataset_id}")
async def get_dataset_by_id(dataset_id: str):
    repo: DatasetRepository = app.state.orch.repo
    try:
        data = repo.get_dataset(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return data


@app.get("/goldens/{dataset_id}")
async def get_golden_by_dataset(dataset_id: str):
    # golden file is <dataset_id>.golden.json
    repo: DatasetRepository = app.state.orch.repo
    p = repo.root_dir / f"{dataset_id}.golden.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="golden not found")
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class SaveDatasetBody(BaseModel):
    dataset: Dict[str, Any]
    golden: Optional[Dict[str, Any]] = None
    overwrite: bool = False
    bump_version: bool = False


@app.post("/datasets/save")
async def save_dataset(body: SaveDatasetBody):
    repo: DatasetRepository = app.state.orch.repo
    root: Path = repo.root_dir
    ds = body.dataset
    gt = body.golden
    # validate
    ds_errors = repo.sv.validate("dataset", ds)
    if ds_errors:
        raise HTTPException(status_code=400, detail={"type": "dataset", "errors": ds_errors})
    if gt is not None:
        gt_errors = repo.sv.validate("golden", gt)
        if gt_errors:
            raise HTTPException(status_code=400, detail={"type": "golden", "errors": gt_errors})
    dataset_id = ds.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id required")
    if gt is not None and gt.get("dataset_id") != dataset_id:
        raise HTTPException(status_code=400, detail="golden.dataset_id must match dataset.dataset_id")
    # bump version (patch)
    if body.bump_version:
        ver = str(ds.get("version", "0.0.0"))
        parts = [int(p) if str(p).isdigit() else 0 for p in ver.split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        parts[2] += 1
        ds["version"] = f"{parts[0]}.{parts[1]}.{parts[2]}"
        if gt is not None:
            gt["version"] = ds["version"]
    # paths
    ds_path = root / f"{dataset_id}.dataset.json"
    gt_path = root / f"{dataset_id}.golden.json"
    if ds_path.exists() and not body.overwrite:
        raise HTTPException(status_code=409, detail="dataset exists; set overwrite=true")
    if gt is not None and gt_path.exists() and not body.overwrite:
        raise HTTPException(status_code=409, detail="golden exists; set overwrite=true")
    # write
    ds_path.write_text(json.dumps(ds, indent=2), encoding='utf-8')
    golden_saved = False
    if gt is not None:
        gt_path.write_text(json.dumps(gt, indent=2), encoding='utf-8')
        golden_saved = True
    return {"ok": True, "dataset_id": dataset_id, "version": ds.get("version"), "dataset_saved": True, "golden_saved": golden_saved}


# --- Coverage Generation API (Prompt 7) ---

class CoverageGenerateRequest(BaseModel):
    domains: Optional[list[str]] = None
    behaviors: Optional[list[str]] = None
    combined: bool = True
    dry_run: bool = True
    save: bool = False
    overwrite: bool = False
    version: str = "1.0.0"


@app.post("/coverage/generate")
async def coverage_generate(req: CoverageGenerateRequest):
    # Build datasets/goldens per request
    try:
        if req.combined:
            # Build per-domain combined and a global combined
            domain_outputs = build_domain_combined_datasets(domains=req.domains, behaviors=req.behaviors, version=req.version)
            global_ds, global_gd = build_global_combined_dataset(domains=req.domains, behaviors=req.behaviors, version=req.version)
            outputs = domain_outputs + [(global_ds, global_gd)]
        else:
            outputs = build_per_behavior_datasets(domains=req.domains, behaviors=req.behaviors, version=req.version)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"generation failed: {e}")

    # If dry_run, just return manifest-like summary
    if req.dry_run and not req.save:
        summary = []
        for ds, gd in outputs:
            summary.append({
                "dataset_id": ds["dataset_id"],
                "version": ds["version"],
                "conversations": len(ds["conversations"]),
                "golden_entries": len(gd["entries"]),
            })
        return {"ok": True, "saved": False, "outputs": summary}

    # Save if requested
    if req.save:
        repo: DatasetRepository = app.state.orch.repo
        root: Path = repo.root_dir
        root.mkdir(parents=True, exist_ok=True)
        written = []
        for ds, gd in outputs:
            # validate
            ds_errors = repo.sv.validate("dataset", ds)
            if ds_errors:
                raise HTTPException(status_code=400, detail={"type": "dataset", "dataset_id": ds.get("dataset_id"), "errors": ds_errors})
            gt_errors = repo.sv.validate("golden", gd)
            if gt_errors:
                raise HTTPException(status_code=400, detail={"type": "golden", "dataset_id": gd.get("dataset_id"), "errors": gt_errors})
            dataset_id = ds["dataset_id"]
            ds_path = root / f"{dataset_id}.dataset.json"
            gt_path = root / f"{dataset_id}.golden.json"
            if not req.overwrite and (ds_path.exists() or gt_path.exists()):
                raise HTTPException(status_code=409, detail=f"{dataset_id} already exists; set overwrite=true")
            ds_path.write_text(json.dumps(ds, indent=2), encoding="utf-8")
            gt_path.write_text(json.dumps(gd, indent=2), encoding="utf-8")
            written.append({"dataset": ds_path.name, "golden": gt_path.name})
        return {"ok": True, "saved": True, "files": written}

    # Default: not saved, not dry-run (shouldn't happen); return detailed outputs
    return {"ok": True, "saved": False, "outputs": outputs}


@app.get("/coverage/taxonomy")
async def coverage_taxonomy():
    cm = CoverageManifestor()
    return {"domains": cm.taxonomy.get("domains", []), "behaviors": cm.taxonomy.get("behaviors", [])}


@app.get("/coverage/manifest")
async def coverage_manifest(domains: Optional[str] = None, behaviors: Optional[str] = None, seed: int = 42):
    cm = CoverageManifestor()
    manifest = cm.build(seed=seed)
    pairs = manifest.get("pairs", [])
    doms = set(domains.split(",")) if isinstance(domains, str) and domains else None
    behs = set(behaviors.split(",")) if isinstance(behaviors, str) and behaviors else None
    if doms:
        pairs = [p for p in pairs if p.get("domain") in doms]
    if behs:
        pairs = [p for p in pairs if p.get("behavior") in behs]
    return {"seed": seed, "axes_order": manifest.get("axes_order"), "pairs": pairs}


@app.get("/coverage/report.csv")
async def coverage_report_csv(type: str = "summary", domains: Optional[str] = None, behaviors: Optional[str] = None):
    doms = domains.split(",") if domains else None
    behs = behaviors.split(",") if behaviors else None
    if type == "summary":
        content = coverage_summary_csv(doms, behs)
    elif type == "heatmap":
        content = coverage_heatmap_csv(doms, behs)
    else:
        raise HTTPException(status_code=400, detail="unknown report type")
    # Return as CSV response
    from fastapi.responses import Response
    return Response(content, media_type="text/csv")


class PerTurnReportBody(BaseModel):
    dataset: Dict[str, Any]
    golden: Dict[str, Any]


@app.post("/coverage/per-turn.csv")
async def coverage_per_turn_csv(body: PerTurnReportBody):
    try:
        content = per_turn_csv(body.dataset, body.golden)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    from fastapi.responses import Response
    return Response(content, media_type="text/csv")


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    repo: DatasetRepository = app.state.orch.repo
    conv = repo.get_conversation(conversation_id)
    try:
        golden = repo.get_golden(conversation_id)
    except Exception:
        golden = None
    return {"conversation": conv, "golden": golden}


@app.post("/runs", response_model=StartRunResponse)
async def start_run(req: StartRunRequest):
    orch: Orchestrator = app.state.orch
    cfg: Dict[str, Any] = {
        "metrics": req.metrics or [],
        "thresholds": req.thresholds or {},
        "context": req.context or {},
    }
    jr = orch.submit(dataset_id=req.dataset_id, model_spec=req.model_spec, config=cfg)
    # initialize run folder with config
    app.state.artifacts.init_run(jr.run_id, {"dataset_id": req.dataset_id, "model_spec": req.model_spec, **cfg})
    orch.start(jr.job_id)
    return StartRunResponse(job_id=jr.job_id, run_id=jr.run_id, state=jr.state)


@app.post("/runs/{job_id}/control")
async def control_run(job_id: str, body: ControlBody):
    orch: Orchestrator = app.state.orch
    jr = orch.jobs.get(job_id)
    if not jr:
        # Allow 'cancel' to mark a stale job as cancelled if persisted job.json exists
        reader: RunArtifactReader = app.state.reader
        for p in sorted(reader.layout.runs_root.iterdir()):
            if not p.is_dir():
                continue
            jpath = p / "job.json"
            if not jpath.exists():
                continue
            try:
                obj = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            if obj.get("job_id") == job_id:
                act = (body.action or '').lower()
                if act in ('cancel','abort'):
                    obj["state"] = "cancelled"
                    obj["error"] = "cancelled by user after restart"
                    jpath.write_text(json.dumps(obj, indent=2), encoding="utf-8")
                    return obj
                raise HTTPException(status_code=404, detail="job not running")
        raise HTTPException(status_code=404, detail="job not found")
    act = (body.action or '').lower()
    try:
        if act == 'pause':
            orch.pause(job_id)
        elif act == 'resume':
            orch.resume(job_id)
        elif act == 'cancel' or act == 'abort':
            orch.cancel(job_id)
        else:
            raise HTTPException(status_code=400, detail="unknown action")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Return current job status snapshot
    return {
        "job_id": jr.job_id,
        "run_id": jr.run_id,
        "state": jr.state,
        "progress_pct": jr.progress_pct,
        "total_conversations": jr.total_conversations,
        "completed_conversations": jr.completed_conversations,
        "error": jr.error,
    }


@app.get("/runs/{job_id}/status")
async def run_status(job_id: str):
    orch: Orchestrator = app.state.orch
    jr = orch.jobs.get(job_id)
    if not jr:
        # Try to recover from persisted job status if the process lost in-memory job
        # Search runs folder for a job.json containing this job_id
        reader: RunArtifactReader = app.state.reader
        for p in sorted(reader.layout.runs_root.iterdir()):
            if not p.is_dir():
                continue
            try:
                obj = json.loads((p / "job.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if obj.get("job_id") == job_id:
                # If the recorded boot_id differs from current, mark stale running states as failed
                if obj.get("boot_id") != BOOT_ID and obj.get("state") in ("running", "paused", "cancelling"):
                    obj = {**obj, "state": "failed", "error": "stale status from previous server session"}
                return obj
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": jr.job_id,
        "run_id": jr.run_id,
        "state": jr.state,
        "progress_pct": jr.progress_pct,
        "total_conversations": jr.total_conversations,
        "completed_conversations": jr.completed_conversations,
        "error": jr.error,
    }


@app.get("/runs/{run_id}/results")
async def run_results(run_id: str):
    reader: RunArtifactReader = app.state.reader
    path = reader.layout.results_json_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="results not found")
    return get_json_file(path)


def get_json_file(path: Path):
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}/artifacts")
async def run_artifacts(run_id: str, type: str = "json"):
    reader: RunArtifactReader = app.state.reader
    reporter: Reporter = app.state.reporter
    if type == "json":
        path = reader.layout.results_json_path(run_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="results.json not found")
        return FileResponse(str(path), media_type="application/json", filename="results.json")
    elif type == "csv":
        path = reader.layout.results_csv_path(run_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="results.csv not found")
        return FileResponse(str(path), media_type="text/csv", filename="results.csv")
    elif type == "html":
        # generate on the fly from results.json
        json_path = reader.layout.results_json_path(run_id)
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="results.json not found")
        results = get_json_file(json_path)
        try:
            html = reporter.render_html(results)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"cannot render html: {e}")
        out_path = reader.layout.run_dir(run_id) / "report.html"
        out_path.write_text(html, encoding="utf-8")
        return FileResponse(str(out_path), media_type="text/html", filename="report.html")
    else:
        raise HTTPException(status_code=400, detail="unknown type")


    @app.post("/runs/{run_id}/rebuild")
    async def rebuild_run_artifacts(run_id: str):
        """Rebuild and enrich results.json and results.csv for an existing run.
        Adds human-friendly identity, per-turn snippets, rollups, and writes CSV.
        """
        reader: RunArtifactReader = app.state.reader
        writer: RunArtifactWriter = app.state.artifacts
        repo: DatasetRepository = app.state.orch.repo
        # Load existing
        res_path = reader.layout.results_json_path(run_id)
        if not res_path.exists():
            raise HTTPException(status_code=404, detail="results.json not found")
        try:
            results = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid results.json: {e}")
        ds_id = results.get("dataset_id")
        if not ds_id:
            raise HTTPException(status_code=400, detail="results missing dataset_id")
        try:
            ds = repo.get_dataset(ds_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"dataset not found: {e}")

        # Build conversation map from dataset
        ds_meta = ds.get("metadata", {}) or {}
        domain_description = ds_meta.get("short_description")
        conv_map: dict[str, dict] = {c.get("conversation_id"): c for c in (ds.get("conversations") or [])}

        # Helpers
        import re
        def slugify(text: str) -> str:
            t = (text or "").lower()
            t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
            return t[:80]

        def conv_identity(cid: str) -> dict:
            c = conv_map.get(cid) or {}
            meta = c.get("metadata") or {}
            d = meta.get("domain") or ds_meta.get("domain")
            b = meta.get("behavior") or ds_meta.get("behavior")
            s = meta.get("scenario") or meta.get("case")
            persona = meta.get("persona")
            locale = meta.get("locale")
            channel = meta.get("channel")
            complexity = meta.get("complexity") or ds_meta.get("difficulty")
            case_type = meta.get("case_type") or meta.get("type")
            title = c.get("title") or ((f"{b}: {s}" if b and s else (b or s)) if (b or s) else None) or cid
            parts = [p for p in [d, b, s, persona, locale] if p]
            slug = slugify("-".join(parts)) if parts else slugify(cid)
            return {
                "conversation_slug": slug,
                "conversation_title": title,
                "domain": d,
                "behavior": b,
                "scenario": s,
                "persona": persona,
                "locale": locale,
                "channel": channel,
                "complexity": complexity,
                "case_type": case_type,
            }

        # Enrich per conversation
        from .artifacts import RunFolderLayout
        layout = RunFolderLayout(reader.layout.runs_root)
        updated = 0
        for conv in results.get("conversations", []) or []:
            cid = conv.get("conversation_id")
            if not cid:
                continue
            ident = conv_identity(cid)
            conv.update({k: v for k, v in ident.items() if k not in conv or conv.get(k) in (None, "")})
            # trace dir
            conv["trace_dir"] = str(layout.conversation_subdir(run_id, cid))
            # set conversation_description if present in dataset
            if "conversation_description" not in conv:
                try:
                    conv_desc = (conv_map.get(cid, {}).get("metadata") or {}).get("short_description")
                    if conv_desc:
                        conv["conversation_description"] = conv_desc
                except Exception:
                    pass
            # per-turn enrich
            # open turn files to get assistant output snippet
            from glob import glob
            try:
                conv_dir = layout.conversation_subdir(run_id, cid)
                turn_files = sorted(conv_dir.glob("turn_*.json"))
            except Exception:
                turn_files = []
            # map turn_index -> response content
            resp_by_idx: dict[int, str] = {}
            for tf in turn_files:
                try:
                    rec = json.loads(tf.read_text(encoding="utf-8"))
                    uidx = int(rec.get("turn_index", 0))
                    resp_by_idx[uidx] = ((rec.get("response", {}) or {}).get("content")) or ""
                except Exception:
                    continue
            # dataset turns for user prompt snippet
            ds_turns = (conv_map.get(cid, {}).get("turns") or []) if cid in conv_map else []
            def snippet(t: str, n: int = 160) -> str:
                t = (t or "").strip().replace("\n", " ")
                return t if len(t) <= n else (t[: n - 1] + "…")
            for t in conv.get("turns", []) or []:
                idx = int(t.get("turn_index", 0))
                if "turn_pass" not in t:
                    mets = t.get("metrics", {}) or {}
                    pass_vals = [bool(v.get("pass")) for v in mets.values() if isinstance(v, dict) and "pass" in v]
                    t["turn_pass"] = (all(pass_vals) if pass_vals else True)
                if "user_prompt_snippet" not in t:
                    try:
                        user_text = str(ds_turns[idx].get("text") or "") if 0 <= idx < len(ds_turns) else ""
                    except Exception:
                        user_text = ""
                    t["user_prompt_snippet"] = snippet(user_text)
                if "assistant_output_snippet" not in t:
                    t["assistant_output_snippet"] = snippet(resp_by_idx.get(idx, ""), 200)
            # summary rollups
            summ = conv.get("summary") or {}
            if "total_user_turns" not in summ:
                summ["total_user_turns"] = len(conv.get("turns") or [])
            if "failed_turns_count" not in summ:
                summ["failed_turns_count"] = sum(1 for tt in (conv.get("turns") or []) if tt.get("turn_pass") is False)
            if "failed_metrics" not in summ:
                failed_metrics = sorted({
                    name for tt in (conv.get("turns") or []) for name, m in (tt.get("metrics") or {}).items()
                    if isinstance(m, dict) and m.get("pass") is False
                })
                summ["failed_metrics"] = failed_metrics
            conv["summary"] = summ
            updated += 1

        # Write back results.json and results.csv
        # add domain description at top level
        if domain_description:
            results["domain_description"] = domain_description
        writer.write_results_json(run_id, results)
        try:
            writer.write_results_csv(run_id, results)
        except Exception as e:
            # still return ok if JSON was updated
            return {"ok": True, "updated_json": True, "updated_csv": False, "error": str(e), "conversations": updated}
        return {"ok": True, "updated_json": True, "updated_csv": True, "conversations": updated}


@app.post("/runs/{run_id}/feedback")
async def submit_feedback(run_id: str, body: Dict[str, Any]):
    # Append feedback objects to runs/<run_id>/feedback.json
    run_dir = app.state.reader.layout.run_dir(run_id)
    fb_path = run_dir / "feedback.json"
    arr: list = []
    if fb_path.exists():
        try:
            import json
            arr = json.loads(fb_path.read_text(encoding="utf-8"))
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
    arr.append(body)
    fb_path.write_text(json.dumps(arr, indent=2), encoding="utf-8")
    return {"ok": True, "count": len(arr)}


@app.get("/compare")
async def compare_runs(runA: str, runB: str):
    reader: RunArtifactReader = app.state.reader
    a = reader.layout.results_json_path(runA)
    b = reader.layout.results_json_path(runB)
    if not a.exists() or not b.exists():
        raise HTTPException(status_code=404, detail="one or both results.json missing")
    A = get_json_file(a)
    B = get_json_file(b)

    def summarize(res: Dict[str, Any]):
        convs = res.get("conversations", []) or []
        total = len(convs)
        passed = sum(1 for c in convs if c.get("summary", {}).get("conversation_pass") is True)
        rate = (passed / total) if total else 0.0
        return {"total": total, "passed": passed, "pass_rate": rate}

    SA = summarize(A)
    SB = summarize(B)
    delta = SB["pass_rate"] - SA["pass_rate"]
    return {"runA": SA, "runB": SB, "delta_pass_rate": delta}


class RunListItem(BaseModel):
    run_id: str
    dataset_id: Optional[str] = None
    model_spec: Optional[str] = None
    has_results: bool
    created_ts: Optional[float] = None


@app.get("/runs")
async def list_runs():
    """List runs by inspecting the runs/ folder."""
    reader: RunArtifactReader = app.state.reader
    layout = reader.layout
    items: list[dict[str, Any]] = []
    if not layout.runs_root.exists():
        return items
    for p in sorted(layout.runs_root.iterdir()):
        if not p.is_dir():
            continue
        run_id = p.name
        cfg_path = p / 'run_config.json'
        res_path = p / 'results.json'
        cfg = {}
        try:
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
        # Try to read persisted job status to enrich item
        job_state = app.state.reader.read_job_status(run_id)
        # Determine staleness
        boot_id = (job_state or {}).get('boot_id')
        is_stale = (boot_id is None) or (boot_id != BOOT_ID)
        state_val = (job_state or {}).get('state')
        # If stale and previously active, show 'stale'
        if is_stale and state_val in ("running","paused","cancelling"):
            state_val = "stale"
        items.append({
            'run_id': run_id,
            'dataset_id': cfg.get('dataset_id'),
            'model_spec': cfg.get('model_spec'),
            'has_results': res_path.exists(),
            'created_ts': cfg_path.stat().st_mtime if cfg_path.exists() else None,
            'state': state_val,
            'progress_pct': (job_state or {}).get('progress_pct'),
            'completed_conversations': (job_state or {}).get('completed_conversations'),
            'job_id': (job_state or {}).get('job_id'),
            'stale': is_stale,
        })
    return items


@app.post("/validate")
async def validate_json(body: Dict[str, Any]):
    """Validate payload against a named schema without saving.
    Body shape: {"type": "dataset"|"golden"|"run_config", "payload": {...}}
    """
    sv = app.state.orch.repo.sv
    t = body.get("type")
    payload = body.get("payload")
    if t not in ("dataset", "golden", "run_config"):
        raise HTTPException(status_code=400, detail="invalid type")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be object")
    errors = sv.validate(t, payload)
    return {"ok": len(errors) == 0, "errors": errors}


@app.get("/metrics-config")
async def get_metrics_config():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / 'configs' / 'metrics.json'
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    # default
    return {
        "metrics": [
            {"name": "exact_match", "enabled": True, "weight": 1.0},
            {"name": "semantic_similarity", "enabled": True, "weight": 1.0, "threshold": float(os.getenv("SEMANTIC_THRESHOLD", "0.80"))},
            {"name": "consistency", "enabled": True, "weight": 1.0},
            {"name": "adherence", "enabled": True, "weight": 1.0},
            {"name": "hallucination", "enabled": True, "weight": 1.0},
        ]
    }


@app.post("/metrics-config")
async def set_metrics_config(body: Dict[str, Any]):
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / 'configs' / 'metrics.json'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg_path.write_text(json.dumps(body, indent=2), encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}

