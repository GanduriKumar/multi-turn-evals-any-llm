from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Optional
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import json
import csv
from datetime import datetime
import asyncio
from typing import List

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
    from backend.commerce_taxonomy import load_commerce_config
    from backend.coverage_builder_v2 import (
        build_per_behavior_datasets_v2,
        build_domain_combined_datasets_v2,
        build_global_combined_dataset_v2,
    )
    from backend.coverage_manifest import CoverageManifestor, build_manifest
    from backend.report_diff import diff_results
else:
    from .coverage_builder_v2 import (
        build_per_behavior_datasets_v2,
        build_domain_combined_datasets_v2,
        build_global_combined_dataset_v2,
    )
    from .coverage_manifest import CoverageManifestor, build_manifest
    from .report_diff import diff_results
    from .coverage_reports import coverage_summary_csv, coverage_heatmap_csv, per_turn_csv
    from .coverage_config import CoverageConfig
    from .array_builder_v2 import build_combined_array
    from .coverage_config import CoverageConfig
    from .commerce_taxonomy import load_commerce_config

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
    industry_vertical: str | None = None


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
    industry_vertical: Optional[str] = None


SUPPORTED_VERTICALS = ["commerce", "banking", "finance", "healthcare"]


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
    industry_vertical = os.getenv("INDUSTRY_VERTICAL", "commerce")
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
        "INDUSTRY_VERTICAL": industry_vertical,
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

# App state singletons (vertical-aware)
RUNS_BASE = Path(__file__).resolve().parents[1] / "runs"
RUNS_BASE.mkdir(parents=True, exist_ok=True)
import uuid
BOOT_ID = str(uuid.uuid4())

# Per-vertical contexts { vertical: { 'orch', 'artifacts', 'reader' } }
app.state.vctx: dict[str, dict[str, Any]] = {}
app.state.reporter = Reporter(Path(__file__).resolve().parent / "templates")
try:
    # Provider registry for chat endpoints
    from .providers.registry import ProviderRegistry  # type: ignore
except Exception:
    from backend.providers.registry import ProviderRegistry  # type: ignore
app.state.providers = ProviderRegistry()
app.state.chat_jobs: dict[str, dict[str, Any]] = {}

def _ensure_vertical_name(name: Optional[str]) -> str:
    v = (name or os.getenv("INDUSTRY_VERTICAL") or "commerce").lower()
    if v not in SUPPORTED_VERTICALS:
        v = "commerce"
    return v

def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _get_or_create_vertical_context(vertical: Optional[str] = None) -> dict[str, Any]:
    v = _ensure_vertical_name(vertical)
    if v in app.state.vctx:
        return app.state.vctx[v]
    # Back-compat: if tests or callers set app.state.orch directly, adopt it for this vertical
    try:
        legacy_orch = getattr(app.state, 'orch', None)
    except Exception:
        legacy_orch = None
    if legacy_orch is not None:
        try:
            runs_root = legacy_orch.runs_root
        except Exception:
            runs_root = RUNS_BASE / v
            runs_root.mkdir(parents=True, exist_ok=True)
        ctx = {
            "orch": legacy_orch,
            "artifacts": RunArtifactWriter(runs_root),
            "reader": RunArtifactReader(runs_root),
            "vertical": v,
        }
        app.state.vctx[v] = ctx
        return ctx
    root = Path(__file__).resolve().parents[1]
    datasets_root = root / "datasets" / v
    datasets_root.mkdir(parents=True, exist_ok=True)
    runs_root = RUNS_BASE / v
    runs_root.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator(datasets_dir=datasets_root, runs_root=runs_root, boot_id=BOOT_ID)
    ctx = {
        "orch": orch,
        "artifacts": RunArtifactWriter(runs_root),
        "reader": RunArtifactReader(runs_root),
        "vertical": v,
    }
    app.state.vctx[v] = ctx
    return ctx

def _iter_all_contexts() -> list[dict[str, Any]]:
    # Ensure at least default exists
    _get_or_create_vertical_context(os.getenv("INDUSTRY_VERTICAL", "commerce"))
    return list(app.state.vctx.values())

def _migrate_legacy_assets():
    """Move legacy datasets/runs at root into 'commerce' vertical folders once."""
    root = Path(__file__).resolve().parents[1]
    # datasets
    ds_root = root / "datasets"
    commerce_ds = ds_root / "commerce"
    try:
        commerce_ds.mkdir(parents=True, exist_ok=True)
        moved_any = False
        # move arrays folder if at root
        arrays_src = ds_root / "arrays"
        arrays_dst = commerce_ds / "arrays"
        if arrays_src.exists() and arrays_src.is_dir() and not arrays_dst.exists():
            try:
                arrays_dst.parent.mkdir(parents=True, exist_ok=True)
                arrays_src.rename(arrays_dst)
                moved_any = True
            except Exception:
                pass
        for p in ds_root.glob("*.dataset.json"):
            try:
                p.rename(commerce_ds / p.name)
                moved_any = True
            except Exception:
                pass
        for p in ds_root.glob("*.golden.json"):
            try:
                p.rename(commerce_ds / p.name)
                moved_any = True
            except Exception:
                pass
    except Exception:
        pass
    # runs
    runs_root = root / "runs"
    commerce_runs = runs_root / "commerce"
    try:
        commerce_runs.mkdir(parents=True, exist_ok=True)
        for p in sorted(runs_root.iterdir()):
            if not p.is_dir():
                continue
            if p.name in SUPPORTED_VERTICALS:
                continue
            # Heuristic: run folder contains job.json or run_config.json
            if (p / "job.json").exists() or (p / "run_config.json").exists():
                try:
                    p.rename(commerce_runs / p.name)
                except Exception:
                    pass
    except Exception:
        pass

# Initialize default vertical context and migrate once
_migrate_legacy_assets()
_get_or_create_vertical_context(os.getenv("INDUSTRY_VERTICAL", "commerce"))


class StartRunRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
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

class ChatDatasetBody(BaseModel):
    dataset_id: str
    conversation_id: Optional[str] = None  # optional: when omitted, chat about the whole dataset
    message: str
    model: Optional[str] = None  # provider:model
    history: Optional[list[dict[str, str]]] = None  # [{role, content}]
    async_mode: Optional[bool] = None

class ChatReportBody(BaseModel):
    run_id: str
    message: str
    model: Optional[str] = None  # provider:model
    history: Optional[list[dict[str, str]]] = None
    conversation_id: Optional[str] = None  # optional focus on a single conversation
    async_mode: Optional[bool] = None  # when true, use background job

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
        industry_vertical=s.get("INDUSTRY_VERTICAL"),
    )


@app.post("/chat/dataset")
async def chat_with_dataset(body: ChatDatasetBody, vertical: Optional[str] = None):
    # Resolve model: allow provider:model or fallback to defaults
    model_spec = (body.model or os.getenv("OPENAI_MODEL") or os.getenv("GEMINI_MODEL") or os.getenv("OLLAMA_MODEL") or "")
    if ":" not in model_spec:
        # default to openai: if OPENAI enabled else ollama
        if os.getenv("OPENAI_API_KEY"):
            model_spec = f"openai:{model_spec or os.getenv('OPENAI_MODEL','gpt-5.1')}"
        elif os.getenv("GOOGLE_API_KEY"):
            model_spec = f"gemini:{model_spec or os.getenv('GEMINI_MODEL','gemini-2.5')}"
        else:
            model_spec = f"ollama:{model_spec or os.getenv('OLLAMA_MODEL','llama3.2:latest')}"
    provider_name, model_name = model_spec.split(":", 1)
    # Load dataset/conversation for context
    ctx = _get_or_create_vertical_context(vertical)
    repo: DatasetRepository = ctx['orch'].repo
    try:
        ds = repo.get_dataset(body.dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"dataset not found: {e}")
    conv = None
    if body.conversation_id:
        for c in (ds.get("conversations") or []):
            if str(c.get("conversation_id")) == str(body.conversation_id):
                conv = c; break
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found in dataset")
    # Build system prompt and few-shot context from conversation or dataset
    sys_lines = []
    if conv is not None:
        sys_lines = [
            "You are assisting with analyzing a specific dataset conversation.",
            "Answer succinctly and cite turns by index when referring to them.",
        ]
        meta = conv.get("metadata") or {}
        identity = [
            f"Domain: {meta.get('domain') or conv.get('domain') or ''}",
            f"Behavior: {meta.get('behavior') or conv.get('behavior') or ''}",
            f"Scenario: {meta.get('scenario') or meta.get('case') or ''}",
        ]
        sys_lines += [" | ".join([p for p in identity if p and p.strip()])]
        # Include brief transcript snippets (user only) for context
        def snip(t: str, n: int = 200) -> str:
            t = (t or "").replace("\n", " ").strip()
            return t if len(t) <= n else (t[: n - 1] + "…")
        uturns = []
        for t in (conv.get("turns") or []):
            try:
                if t.get("role") == "user" or t.get("user") is not None:
                    idx = int(t.get("turn_index", len(uturns)))
                    uturns.append(f"[{idx+1}] {snip(t.get('text') or t.get('user') or '')}")
            except Exception:
                continue
        if uturns:
            sys_lines.append("User turns:")
            sys_lines += uturns[:8]
    else:
        # Dataset-level chat (no specific conversation selected)
        sys_lines = [
            "You are assisting with analyzing a dataset.",
            "Answer succinctly and, when relevant, refer to conversations by title or ID.",
            f"Dataset: {body.dataset_id}",
        ]
        try:
            convs = (ds.get("conversations") or [])
            total = len(convs)
            titles = [
                (c.get("conversation_title") or c.get("conversation_slug") or str(c.get("conversation_id")))
                for c in convs
            ]
            if titles:
                # Show all when reasonably small; otherwise show a capped list with counts
                cap = 30
                if total <= cap:
                    sys_lines.append(f"Conversations ({total}): " + "; ".join(titles))
                else:
                    sys_lines.append(f"Conversations (showing {cap} of {total}): " + "; ".join(titles[:cap]))
        except Exception:
            pass
    # Lightweight RAG: retrieve top passages from the selected conversation or whole dataset
    try:
        from .rag import build_dataset_index  # type: ignore
    except Exception:
        try:
            from backend.rag import build_dataset_index  # type: ignore
        except Exception:
            build_dataset_index = None  # type: ignore
    if build_dataset_index is not None:
        try:
            # Build index and search using last user messages + current message
            idx = build_dataset_index(ds, restrict_conversation_id=(conv.get("conversation_id") if conv else None))
            # Compose a retrieval query from recent user turns
            user_hist = [m.get("content") for m in (body.history or []) if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)]
            query_text = "\n".join(([t for t in user_hist[-2:]] + [body.message or ""]) or [])
            try:
                from .embeddings.ollama_embed import OllamaEmbeddings  # type: ignore
            except Exception:
                from backend.embeddings.ollama_embed import OllamaEmbeddings  # type: ignore
            embedder = OllamaEmbeddings()
            hits = []
            try:
                # Adaptive top_k for dataset queries too
                ql = (body.message or "").strip()
                short = len(ql.split()) <= 4
                diag = any(w in ql.lower() for w in ("turn","user","assistant","domain","behavior","scenario","title","id","slug"))
                topk = 16 if (short or diag) else 8
                hits = await idx.search(query_text, embedder, top_k=topk)  # type: ignore[attr-defined]
            except Exception:
                hits = []
            if hits:
                sys_lines.append("Retrieved context (top matches):")
                for e, s in hits:
                    try:
                        sys_lines.append(f"- {e.text}")
                    except Exception:
                        continue
        except Exception:
            # Non-fatal: if RAG fails, continue without retrieved context
            pass
    system_prompt = "\n".join([l for l in sys_lines if l])
    # Build messages
    history = body.history or []
    msgs = ([{"role": "system", "content": system_prompt}] +
            [m for m in history if isinstance(m, dict) and m.get("role") in ("user","assistant") and isinstance(m.get("content"), str)] +
            [{"role": "user", "content": body.message or ''}])
    # Call provider
    try:
        provider = app.state.providers.get(provider_name)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown provider: {provider_name}")
    try:
        from .providers.types import ProviderRequest  # type: ignore
    except Exception:
        from backend.providers.types import ProviderRequest  # type: ignore
    req = ProviderRequest(model=model_name, messages=msgs, metadata={"dataset_id": body.dataset_id, "conversation_id": body.conversation_id})
    try:
        # Support either .complete(req) or .chat(req) depending on provider implementation
        if hasattr(provider, 'complete'):
            resp = await provider.complete(req)  # type: ignore[attr-defined]
        elif hasattr(provider, 'chat'):
            resp = await provider.chat(req)  # type: ignore[attr-defined]
        else:
            raise RuntimeError("provider missing chat interface")
    except Exception as e:
        # Fallback: synthesize a facts-only response from dataset when provider fails
        facts = []
        try:
            facts.append("Provider failed; returning dataset facts-only summary.")
            if conv is not None:
                meta = conv.get("metadata") or {}
                title = conv.get("conversation_title") or conv.get("conversation_slug") or conv.get("conversation_id")
                dom = meta.get("domain") or conv.get("domain")
                beh = meta.get("behavior") or conv.get("behavior")
                tcount = len(conv.get("turns") or [])
                facts.append(f"Conversation: {title} | Domain: {dom} | Behavior: {beh} | Turns: {tcount}")
                # Include a few user snippets as evidence
                shown = 0
                for t in (conv.get("turns") or []):
                    if shown >= 6:
                        break
                    try:
                        idx = int(t.get("turn_index", shown))
                        u = t.get("text") or t.get("user") or t.get("user_text") or t.get("user_prompt_snippet") or ""
                        if u:
                            facts.append(f"[{idx+1}] U: {u[:180]}")
                            shown += 1
                    except Exception:
                        continue
            else:
                convs = (ds.get("conversations") or [])
                facts.append(f"Dataset: {body.dataset_id} | Conversations: {len(convs)}")
                titles = [c.get("conversation_title") or c.get("conversation_slug") or str(c.get("conversation_id")) for c in convs[:30]]
                if titles:
                    facts.append("Sample titles: " + "; ".join(titles))
            return {
                "ok": False,
                "content": "\n".join(facts),
                "error": f"provider call failed: {e}",
                "provider_meta": {},
                "model": model_spec,
            }
        except Exception:
            raise HTTPException(status_code=500, detail=f"provider call failed: {e}")
    # Validate provider response
    _content = getattr(resp, 'content', None)
    _ok = bool(getattr(resp, 'ok', False))
    _err = getattr(resp, 'error', None)
    if (not _ok) or (not isinstance(_content, str)) or (isinstance(_content, str) and not _content.strip()):
        # Fallback: facts-only when provider returns empty
        try:
            facts = ["Provider returned empty content; dataset facts-only summary:"]
            if conv is not None:
                meta = conv.get("metadata") or {}
                title = conv.get("conversation_title") or conv.get("conversation_slug") or conv.get("conversation_id")
                dom = meta.get("domain") or conv.get("domain")
                beh = meta.get("behavior") or conv.get("behavior")
                tcount = len(conv.get("turns") or [])
                facts.append(f"Conversation: {title} | Domain: {dom} | Behavior: {beh} | Turns: {tcount}")
            else:
                convs = (ds.get("conversations") or [])
                facts.append(f"Dataset: {body.dataset_id} | Conversations: {len(convs)}")
            return {
                "ok": False,
                "content": "\n".join(facts),
                "error": _err or "empty provider response",
                "provider_meta": getattr(resp, 'provider_meta', {}),
                "model": model_spec,
            }
        except Exception:
            raise HTTPException(status_code=502, detail=_err or "empty provider response")
    # Persist chat transcript under datasets/<vertical>/chats/<dataset_id>/<conversation_id>.jsonl
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "datasets" / ctx['vertical'] / "chats" / body.dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{body.conversation_id}.jsonl"
    try:
        with out_path.open("a", encoding="utf-8") as f:
            import json as _json
            evt = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "model": model_spec,
                "user": body.message,
                "assistant": getattr(resp, 'content', None),
                "ok": getattr(resp, 'ok', False),
                "provider_meta": getattr(resp, 'provider_meta', {})
            }
            f.write(_json.dumps(evt) + "\n")
    except Exception:
        pass
    return {
        "ok": bool(getattr(resp, 'ok', False)),
        "content": getattr(resp, 'content', None),
        "error": getattr(resp, 'error', None),
        "provider_meta": getattr(resp, 'provider_meta', {}),
        "model": model_spec,
    }


@app.get("/chat/dataset/log")
async def get_dataset_chat_log(dataset_id: str, conversation_id: Optional[str] = None, vertical: Optional[str] = None, limit: int = 50):
    """Return recent dataset chat events persisted on disk for this dataset/conversation.
    The dataset chat persists to datasets/<vertical>/chats/<dataset_id>/<conversation_id>.jsonl
    where conversation_id is the literal 'None' when not provided.
    """
    try:
        v = _ensure_vertical_name(vertical)
        root = Path(__file__).resolve().parents[1]
        conv_key = str(conversation_id) if conversation_id is not None else 'None'
        path = root / "datasets" / v / "chats" / dataset_id / f"{conv_key}.jsonl"
        events: List[dict] = []
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()[-max(1, min(limit, 500)):]
                import json as _json
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(_json.loads(line))
                    except Exception:
                        continue
            except Exception:
                events = []
        return {"events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/report")
async def chat_with_report(body: ChatReportBody, vertical: Optional[str] = None):
    # Resolve run results path by vertical or search
    readers: list[RunArtifactReader] = []
    if vertical:
        readers = [_get_or_create_vertical_context(vertical)['reader']]
    else:
        readers = [c['reader'] for c in _iter_all_contexts()]
    rd = None
    json_path = None
    for r in readers:
        p = r.layout.results_json_path(body.run_id)
        if p.exists():
            rd = r
            json_path = p
            break
    # Fallback: if a vertical was provided but not found there, search all contexts
    if json_path is None and vertical is not None:
        for c in _iter_all_contexts():
            r = c['reader']
            p = r.layout.results_json_path(body.run_id)
            if p.exists():
                rd = r
                json_path = p
                break
    if json_path is None:
        raise HTTPException(status_code=404, detail="results.json not found")
    results = get_json_file(json_path)
    # Determine model
    model_spec = (body.model or os.getenv("OPENAI_MODEL") or os.getenv("GEMINI_MODEL") or os.getenv("OLLAMA_MODEL") or "")
    if ":" not in model_spec:
        if os.getenv("OPENAI_API_KEY"):
            model_spec = f"openai:{model_spec or os.getenv('OPENAI_MODEL','gpt-5.1')}"
        elif os.getenv("GOOGLE_API_KEY"):
            model_spec = f"gemini:{model_spec or os.getenv('GEMINI_MODEL','gemini-2.5')}"
        else:
            model_spec = f"ollama:{model_spec or os.getenv('OLLAMA_MODEL','llama3.2:latest')}"
    provider_name, model_name = model_spec.split(":", 1)
    # Build system prompt summarizing report
    convs = (results.get("conversations") or []) if isinstance(results, dict) else []
    total = len(convs)
    passed = sum(1 for c in convs if (c.get("summary") or {}).get("conversation_pass") is True)
    turns_total = sum((c.get("summary") or {}).get("total_user_turns", len(c.get("turns") or [])) for c in convs)
    turns_failed = sum((c.get("summary") or {}).get("failed_turns_count", 0) for c in convs)
    meta = results if isinstance(results, dict) else {}
    header = [
        "You are an evaluation report assistant.",
        "Explain failures, drill down into outliers, and cite conversation titles and turn indices when relevant.",
        f"Run: {meta.get('run_id')} | Dataset: {meta.get('dataset_id')} | Model: {meta.get('model_spec')}",
        f"Conversations: {total} (passed {passed}) | Turns failed: {turns_failed} of ~{turns_total}",
    ]
    # Optionally include a specific conversation context
    conv_block = []
    focus_id = body.conversation_id
    focus_conv = None
    if focus_id:
        for c in convs:
            if str(c.get("conversation_id")) == str(focus_id) or str(c.get("conversation_slug")) == str(focus_id):
                focus_conv = c
                break
    if focus_conv:
        title = focus_conv.get("conversation_title") or focus_conv.get("conversation_slug") or focus_conv.get("conversation_id")
        summ = focus_conv.get("summary") or {}
        conv_block.append(f"Focus: {title} — pass={bool(summ.get('conversation_pass'))}, failed_turns={summ.get('failed_turns_count')}")
        failed_metrics = ", ".join((summ.get("failed_metrics") or [])[:8])
        if failed_metrics:
            conv_block.append(f"Failed metrics: {failed_metrics}")
        # Include a few user/assistant snippets
        def snip(t: str, n: int = 200) -> str:
            t = (t or "").replace("\n", " ").strip()
            return t if len(t) <= n else (t[: n - 1] + "…")
        for t in (focus_conv.get("turns") or [])[:6]:
            idx = int(t.get("turn_index", 0))
            u = t.get("user_prompt_snippet") or ""
            a = t.get("assistant_output_snippet") or ""
            conv_block.append(f"[{idx+1}] U: {snip(u)}")
            if a:
                conv_block.append(f"    A: {snip(a, 160)}")
    # Lightweight RAG over report results to enrich context
    sys_lines = [*header, *conv_block]
    # Facts-first: if user asks about pass/fail, pre-append a deterministic overview list
    try:
        qtext = (body.message or "").lower()
        if any(w in qtext for w in ("pass", "passed", "fail", "failed", "conversation_pass")):
            passed_list = []
            failed_list = []
            for c in convs:
                title = c.get("conversation_title") or c.get("conversation_slug") or c.get("conversation_id")
                ok = bool((c.get("summary") or {}).get("conversation_pass"))
                (passed_list if ok else failed_list).append(str(title))
            sys_lines.append("Pass/Fail overview:")
            sys_lines.append(f"Passed ({len(passed_list)}): " + (", ".join(passed_list) if passed_list else "none"))
            sys_lines.append(f"Failed ({len(failed_list)}): " + (", ".join(failed_list) if failed_list else "none"))
    except Exception:
        pass
    try:
        from .rag import build_report_index  # type: ignore
    except Exception:
        try:
            from backend.rag import build_report_index  # type: ignore
        except Exception:
            build_report_index = None  # type: ignore
    if build_report_index is not None:
        try:
            idx = build_report_index(results)
            user_hist = [m.get("content") for m in (body.history or []) if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)]
            query_text = "\n".join(([t for t in user_hist[-2:]] + [body.message or ""]) or [])
            try:
                from .embeddings.ollama_embed import OllamaEmbeddings  # type: ignore
            except Exception:
                from backend.embeddings.ollama_embed import OllamaEmbeddings  # type: ignore
            embedder = OllamaEmbeddings()
            hits = []
            try:
                # Adaptive top_k: more for short or diagnostic queries
                ql = (body.message or "").strip()
                short = len(ql.split()) <= 4
                diag = any(w in ql.lower() for w in ("pass","passed","fail","failed","conversation_pass","metric","domain","behavior","id","slug"))
                topk = 16 if (short or diag) else 8
                hits = await idx.search(query_text, embedder, top_k=topk)  # type: ignore[attr-defined]
            except Exception:
                hits = []
            if hits:
                sys_lines.append("Retrieved context (top matches):")
                for e, s in hits:
                    try:
                        sys_lines.append(f"- {e.text}")
                    except Exception:
                        continue
        except Exception:
            pass
    system_prompt = "\n".join(sys_lines)
    # Messages
    history = body.history or []
    msgs = ([{"role": "system", "content": system_prompt}] +
            [m for m in history if isinstance(m, dict) and m.get("role") in ("user","assistant") and isinstance(m.get("content"), str)] +
            [{"role": "user", "content": body.message or ''}])
    # Call provider
    try:
        provider = app.state.providers.get(provider_name)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown provider: {provider_name}")
    try:
        from .providers.types import ProviderRequest  # type: ignore
    except Exception:
        from backend.providers.types import ProviderRequest  # type: ignore
    req = ProviderRequest(model=model_name, messages=msgs, metadata={"run_id": body.run_id, "conversation_id": focus_id})
    try:
        if hasattr(provider, 'complete'):
            resp = await provider.complete(req)  # type: ignore[attr-defined]
        elif hasattr(provider, 'chat'):
            resp = await provider.chat(req)  # type: ignore[attr-defined]
        else:
            raise RuntimeError("provider missing chat interface")
    except Exception as e:
        # Fallback: synthesize a facts-only response from results.json when provider fails
        facts = []
        try:
            facts.append("Provider failed; returning facts-only summary.")
            passed_list = []
            failed_list = []
            for c in convs:
                title = c.get("conversation_title") or c.get("conversation_slug") or c.get("conversation_id")
                ok = bool((c.get("summary") or {}).get("conversation_pass"))
                (passed_list if ok else failed_list).append(title)
            facts.append(f"Passed ({len(passed_list)}): " + (", ".join(map(str, passed_list)) or "none"))
            facts.append(f"Failed ({len(failed_list)}): " + (", ".join(map(str, failed_list)) or "none"))
            # list failed metrics per failed conversation
            fm_lines = []
            for c in convs:
                summ = c.get("summary") or {}
                if not bool(summ.get("conversation_pass")):
                    title = c.get("conversation_title") or c.get("conversation_slug") or c.get("conversation_id")
                    fms = ", ".join((summ.get("failed_metrics") or [])[:8]) or "(none listed)"
                    fm_lines.append(f"- {title}: failed_metrics={fms}; failed_turns={summ.get('failed_turns_count')}")
            if fm_lines:
                facts.append("Failures detail:")
                facts.extend(fm_lines[:12])
            return {
                "ok": False,
                "content": "\n".join(facts),
                "error": f"provider call failed: {e}",
                "provider_meta": {},
                "model": model_spec,
            }
        except Exception:
            raise HTTPException(status_code=500, detail=f"provider call failed: {e}")
    # Validate provider response to avoid returning HTTP 200 with empty content
    _content = getattr(resp, 'content', None)
    _ok = bool(getattr(resp, 'ok', False))
    _err = getattr(resp, 'error', None)
    if (not _ok) or (not isinstance(_content, str)) or (isinstance(_content, str) and not _content.strip()):
        # Fallback: facts-only when provider returns empty
        try:
            facts = ["Provider returned empty content; facts-only summary:"]
            passed_list = []
            failed_list = []
            for c in convs:
                title = c.get("conversation_title") or c.get("conversation_slug") or c.get("conversation_id")
                ok = bool((c.get("summary") or {}).get("conversation_pass"))
                (passed_list if ok else failed_list).append(title)
            facts.append(f"Passed ({len(passed_list)}): " + (", ".join(map(str, passed_list)) or "none"))
            facts.append(f"Failed ({len(failed_list)}): " + (", ".join(map(str, failed_list)) or "none"))
            return {
                "ok": False,
                "content": "\n".join(facts),
                "error": _err or "empty provider response",
                "provider_meta": getattr(resp, 'provider_meta', {}),
                "model": model_spec,
            }
        except Exception:
            raise HTTPException(status_code=502, detail=_err or "empty provider response")
    # Persist under run folder
    out_dir = rd.layout.run_dir(body.run_id) / "chats"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"report-{(focus_id or 'all')}.jsonl"
    out_path = out_dir / name
    try:
        with out_path.open("a", encoding="utf-8") as f:
            import json as _json
            evt = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "model": model_spec,
                "user": body.message,
                "assistant": getattr(resp, 'content', None),
                "ok": getattr(resp, 'ok', False),
                "provider_meta": getattr(resp, 'provider_meta', {})
            }
            f.write(_json.dumps(evt) + "\n")
    except Exception:
        pass
    return {
        "ok": bool(getattr(resp, 'ok', False)),
        "content": getattr(resp, 'content', None),
        "error": getattr(resp, 'error', None),
        "provider_meta": getattr(resp, 'provider_meta', {}),
        "model": model_spec,
    }


@app.get("/chat/report/log")
async def get_report_chat_log(run_id: str, conversation_id: Optional[str] = None, vertical: Optional[str] = None, limit: int = 50):
    """Return recent report chat events persisted on disk for this run/conversation.
    Stored at runs/<vertical>/<run_id>/chats/report-<conversation_id or 'all'>.jsonl
    """
    # Resolve reader/run dir
    readers: list[RunArtifactReader] = []
    if vertical:
        readers = [_get_or_create_vertical_context(vertical)['reader']]
    else:
        readers = [c['reader'] for c in _iter_all_contexts()]
    out_dir = None
    for r in readers:
        d = r.layout.run_dir(run_id)
        if d.exists():
            out_dir = d
            break
    if out_dir is None:
        raise HTTPException(status_code=404, detail="run not found")
    name = f"report-{(conversation_id or 'all')}.jsonl"
    path = out_dir / "chats" / name
    events: List[dict] = []
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-max(1, min(limit, 500)):]
            import json as _json
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(_json.loads(line))
                except Exception:
                    continue
        except Exception:
            events = []
    return {"events": events}


# ---------------- Background jobs for dataset chat -----------------

def _dataset_job_path(vertical: str, dataset_id: str, job_id: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    d = root / "datasets" / vertical / "chats" / dataset_id / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job_id}.json"

async def _run_dataset_chat_job(job_path: Path, body: ChatDatasetBody, vertical: Optional[str]):
    def write_job(data: dict):
        try:
            job_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception:
            pass
    try:
        import httpx
        async with httpx.AsyncClient(timeout=None) as client:
            payload = body.model_dump()
            url = f"http://127.0.0.1:8000/chat/dataset?vertical={vertical or ''}"
            write_job({"status": "running", "updated": _utcnow()})
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                js = resp.json()
                write_job({"status": "completed", "content": js.get("content"), "ok": js.get("ok"), "model": js.get("model"), "updated": _utcnow()})
            else:
                try:
                    js = resp.json()
                    err = js.get("detail") or js.get("error")
                except Exception:
                    err = resp.text
                write_job({"status": "failed", "error": err, "updated": _utcnow()})
    except Exception as e:
        write_job({"status": "failed", "error": str(e), "updated": _utcnow()})


@app.post("/chat/dataset/submit")
async def submit_dataset_chat(body: ChatDatasetBody, vertical: Optional[str] = None):
    v = _ensure_vertical_name(vertical)
    if not body.dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id required")
    job_id = str(uuid.uuid4())
    job_path = _dataset_job_path(v, body.dataset_id, job_id)
    try:
        job_path.write_text(json.dumps({"status": "queued", "created": _utcnow()}, indent=2), encoding='utf-8')
    except Exception:
        pass
    # Persist pending user message to dataset chat log
    try:
        root = Path(__file__).resolve().parents[1]
        conv_key = str(body.conversation_id) if body.conversation_id is not None else 'None'
        out_dir = root / "datasets" / v / "chats" / body.dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{conv_key}.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            evt = {
                "ts": _utcnow(),
                "model": body.model,
                "user": body.message,
                "assistant": None,
                "ok": None,
                "provider_meta": {"job_id": job_id, "status": "queued"}
            }
            f.write(json.dumps(evt) + "\n")
    except Exception:
        pass
    asyncio.create_task(_run_dataset_chat_job(job_path, body, v))
    return {"job_id": job_id}


@app.get("/chat/dataset/job/{job_id}")
async def get_dataset_chat_job(job_id: str, dataset_id: str, vertical: Optional[str] = None):
    v = _ensure_vertical_name(vertical)
    p = _dataset_job_path(v, dataset_id, job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="job not found")
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- Background jobs for report chat -----------------

def _report_job_path(run_reader: RunArtifactReader, run_id: str, job_id: str) -> Path:
    d = run_reader.layout.run_dir(run_id) / "chats" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job_id}.json"

async def _run_report_chat_job(job_path: Path, body: ChatReportBody, vertical: Optional[str]):
    # Write running status
    def write_job(data: dict):
        try:
            job_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception:
            pass
    # Resolve reader for paths
    readers: list[RunArtifactReader] = []
    if vertical:
        readers = [_get_or_create_vertical_context(vertical)['reader']]
    else:
        readers = [c['reader'] for c in _iter_all_contexts()]
    rd = None
    for r in readers:
        if r.layout.run_dir(body.run_id).exists():
            rd = r; break
    if rd is None:
        write_job({"status": "failed", "error": "run not found", "updated": _utcnow()})
        return
    # Build a synthetic request to reuse the same logic: call provider locally without streaming
    try:
        # Reuse the same logic by invoking the internal function (duplicate minimal parts)
        # We'll call the chat_with_report endpoint logic by constructing messages inline for the job
        # For simplicity and consistency, we perform an HTTP call to ourselves
        import httpx
        async with httpx.AsyncClient(timeout=None) as client:
            payload = body.model_dump()
            url = f"http://127.0.0.1:8000/chat/report?vertical={vertical or ''}"
            write_job({"status": "running", "updated": _utcnow()})
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                js = resp.json()
                write_job({"status": "completed", "content": js.get("content"), "ok": js.get("ok"), "model": js.get("model"), "updated": _utcnow()})
            else:
                try:
                    js = resp.json()
                    err = js.get("detail") or js.get("error")
                except Exception:
                    err = resp.text
                write_job({"status": "failed", "error": err, "updated": _utcnow()})
    except Exception as e:
        write_job({"status": "failed", "error": str(e), "updated": _utcnow()})


@app.post("/chat/report/submit")
async def submit_report_chat(body: ChatReportBody, vertical: Optional[str] = None):
    # Create job id and persist queued
    job_id = str(uuid.uuid4())
    readers: list[RunArtifactReader] = []
    if vertical:
        readers = [_get_or_create_vertical_context(vertical)['reader']]
    else:
        readers = [c['reader'] for c in _iter_all_contexts()]
    rd = None
    for r in readers:
        if r.layout.run_dir(body.run_id).exists():
            rd = r; break
    if rd is None:
        raise HTTPException(status_code=404, detail="run not found")
    job_path = _report_job_path(rd, body.run_id, job_id)
    try:
        job_path.write_text(json.dumps({"status": "queued", "created": _utcnow()}, indent=2), encoding='utf-8')
    except Exception:
        pass
    # Persist pending user message to report chat log
    try:
        out_dir = rd.layout.run_dir(body.run_id) / "chats"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"report-{(body.conversation_id or 'all')}.jsonl"
        out_path = out_dir / name
        with out_path.open("a", encoding="utf-8") as f:
            evt = {
                "ts": _utcnow(),
                "model": body.model,
                "user": body.message,
                "assistant": None,
                "ok": None,
                "provider_meta": {"job_id": job_id, "status": "queued"}
            }
            f.write(json.dumps(evt) + "\n")
    except Exception:
        pass
    # Schedule background task
    asyncio.create_task(_run_report_chat_job(job_path, body, vertical))
    return {"job_id": job_id}


@app.get("/chat/report/job/{job_id}")
async def get_report_chat_job(job_id: str, run_id: str, vertical: Optional[str] = None):
    readers: list[RunArtifactReader] = []
    if vertical:
        readers = [_get_or_create_vertical_context(vertical)['reader']]
    else:
        readers = [c['reader'] for c in _iter_all_contexts()]
    rd = None
    for r in readers:
        p = r.layout.run_dir(run_id)
        if p.exists():
            rd = r; break
    if rd is None:
        raise HTTPException(status_code=404, detail="run not found")
    p = _report_job_path(rd, run_id, job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="job not found")
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        "industry_vertical": s.get("INDUSTRY_VERTICAL"),
        "supported_verticals": SUPPORTED_VERTICALS,
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
    if body.industry_vertical is not None and isinstance(body.industry_vertical, str):
        v = body.industry_vertical.lower()
        if v in SUPPORTED_VERTICALS:
            env['INDUSTRY_VERTICAL'] = v
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
    if 'INDUSTRY_VERTICAL' in env:
        os.environ['INDUSTRY_VERTICAL'] = env['INDUSTRY_VERTICAL']
        # Pre-create vertical context so subsequent calls use the updated selection
        _get_or_create_vertical_context(env['INDUSTRY_VERTICAL'])
    # Hot‑reload provider registry so new keys/hosts/models take effect without restart
    try:
        reg = ProviderRegistry()
        app.state.providers = reg
        # Also update TurnRunner providers for all existing orchestrators
        for ctx in app.state.vctx.values():
            try:
                ctx_orch = ctx.get('orch')
                if ctx_orch is not None and hasattr(ctx_orch, "_runner"):
                    ctx_orch._runner.providers = reg
            except Exception:
                pass
    except Exception:
        # non-fatal; settings file/env still updated
        pass
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
        # Return plain text so UI doesn't try to parse JSON
        from fastapi import Response
        return Response(content=f"Internal Server Error: {e}", media_type="text/plain", status_code=500)


@app.get("/datasets")
async def list_datasets(vertical: Optional[str] = None):
    ctx = _get_or_create_vertical_context(vertical)
    repo: DatasetRepository = ctx['orch'].repo
    return repo.list_datasets()


@app.post("/datasets/upload")
async def upload_dataset(dataset: UploadFile = File(...), golden: Optional[UploadFile] = File(None), overwrite: bool = False, vertical: Optional[str] = None):
    """Upload a dataset (.dataset.json) and optional golden (.golden.json).
    Validates against schemas and writes to the datasets folder.
    """
    ctx = _get_or_create_vertical_context(vertical)
    repo: DatasetRepository = ctx['orch'].repo
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
async def get_dataset_by_id(dataset_id: str, vertical: Optional[str] = None):
    ctx = _get_or_create_vertical_context(vertical)
    repo: DatasetRepository = ctx['orch'].repo
    try:
        data = repo.get_dataset(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return data


@app.get("/goldens/{dataset_id}")
async def get_golden_by_dataset(dataset_id: str, vertical: Optional[str] = None):
    # golden file is <dataset_id>.golden.json
    ctx = _get_or_create_vertical_context(vertical)
    repo: DatasetRepository = ctx['orch'].repo
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
async def save_dataset(body: SaveDatasetBody, vertical: Optional[str] = None):
    ctx = _get_or_create_vertical_context(vertical)
    repo: DatasetRepository = ctx['orch'].repo
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
    as_array: bool = False
    vertical: Optional[str] = None
    user_turns: Optional[int] = 2
    scenario_styles: Optional[list[str]] = None


@app.post("/coverage/generate")
async def coverage_generate(req: CoverageGenerateRequest):
    # Build datasets/goldens per request
    try:
        # Normalize empty lists to None (treat as 'all')
        domains = req.domains if (isinstance(req.domains, list) and len(req.domains) > 0) else None
        behaviors = req.behaviors if (isinstance(req.behaviors, list) and len(req.behaviors) > 0) else None
        orig_domains = list(domains) if isinstance(domains, list) else None
        orig_behaviors = list(behaviors) if isinstance(behaviors, list) else None
        # Align to v2 taxonomy labels when possible (ensures filtering hits)
        try:
            cfg = load_commerce_config()
            tax_domains = set(cfg["taxonomy"].get("domains", []))
            tax_behaviors = set(cfg["taxonomy"].get("behaviors", []))
            orig_dom_len = len(domains) if isinstance(domains, list) else 0
            orig_beh_len = len(behaviors) if isinstance(behaviors, list) else 0
            if domains is not None:
                domains = [d for d in domains if d in tax_domains]
                if orig_dom_len and len(domains) == 0:
                    # If UI provided non-matching labels, surface a 400 to avoid generating ALL.
                    unknown = [d for d in (orig_domains or []) if d not in tax_domains]
                    raise HTTPException(status_code=400, detail=f"Unknown domain(s) for v2 taxonomy: {', '.join(unknown)}")
            if behaviors is not None:
                behaviors = [b for b in behaviors if b in tax_behaviors]
                if orig_beh_len and len(behaviors) == 0:
                    unknown = [b for b in (orig_behaviors or []) if b not in tax_behaviors]
                    raise HTTPException(status_code=400, detail=f"Unknown behavior(s) for v2 taxonomy: {', '.join(unknown)}")
        except HTTPException:
            # Bubble up to client so UI can show a helpful message
            raise
        except Exception:
            # If taxonomy is unavailable for any reason, skip alignment
            pass
        if req.as_array:
            items, counts = build_combined_array(domains=domains, behaviors=behaviors, version=req.version)
            if req.dry_run or not req.save:
                return {"ok": True, "saved": False, "count": len(items), "counts_by_risk": counts}
            # Save array to datasets folder as a single file
            ctx = _get_or_create_vertical_context(req.vertical)
            repo: DatasetRepository = ctx['orch'].repo
            root: Path = repo.root_dir
            root.mkdir(parents=True, exist_ok=True)
            out = {
                "schema": "combined_array.v1",
                "version": req.version,
                "items": items,
            }
            # Path honors dataset_paths if hierarchical: write under datasets/arrays/
            arrays_dir = root / "arrays"
            arrays_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"combined_array-{req.version}.json"
            p = arrays_dir / file_name
            if p.exists() and not req.overwrite:
                raise HTTPException(status_code=409, detail=f"{file_name} exists; set overwrite=true")
            p.write_text(json.dumps(out, indent=2), encoding="utf-8")
            return {"ok": True, "saved": True, "file": str(p)}
        if req.combined:
            # Build per-domain combined and a global combined using v2 schema
            domain_outputs = build_domain_combined_datasets_v2(domains=domains, behaviors=behaviors, version=req.version, user_turns=(req.user_turns or 2), scenario_styles=req.scenario_styles)
            global_ds, global_gd = build_global_combined_dataset_v2(domains=domains, behaviors=behaviors, version=req.version, user_turns=(req.user_turns or 2), scenario_styles=req.scenario_styles)
            # Filter out any empty datasets (no conversations) to avoid schema errors
            outputs = [(ds, gd) for (ds, gd) in domain_outputs if (ds.get('conversations') or [])]
            # Only include global combined when generating for ALL domains (no domain filter)
            if domains is None and (global_ds.get('conversations') or []):
                outputs.append((global_ds, global_gd))
            # Fallback: if combined yielded nothing (e.g., over-filtered), build per-behavior instead
            if not outputs:
                outputs = build_per_behavior_datasets_v2(domains=domains, behaviors=behaviors, version=req.version, user_turns=(req.user_turns or 2), scenario_styles=req.scenario_styles)
        else:
            # Per-behavior datasets (v2 schema)
            outputs = build_per_behavior_datasets_v2(domains=domains, behaviors=behaviors, version=req.version, user_turns=(req.user_turns or 2), scenario_styles=req.scenario_styles)

        # If styles over-filtered to zero, retry once without styles so user isn't left with 0
        if (not outputs) and req.scenario_styles:
            if req.combined:
                domain_outputs = build_domain_combined_datasets_v2(domains=domains, behaviors=behaviors, version=req.version, user_turns=(req.user_turns or 2), scenario_styles=None)
                global_ds, global_gd = build_global_combined_dataset_v2(domains=domains, behaviors=behaviors, version=req.version, user_turns=(req.user_turns or 2), scenario_styles=None)
                outputs = [(ds, gd) for (ds, gd) in domain_outputs if (ds.get('conversations') or [])]
                if domains is None and (global_ds.get('conversations') or []):
                    outputs.append((global_ds, global_gd))
                if not outputs:
                    outputs = build_per_behavior_datasets_v2(domains=domains, behaviors=behaviors, version=req.version, user_turns=(req.user_turns or 2), scenario_styles=None)
            else:
                outputs = build_per_behavior_datasets_v2(domains=domains, behaviors=behaviors, version=req.version, user_turns=(req.user_turns or 2), scenario_styles=None)
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
        ctx = _get_or_create_vertical_context(req.vertical)
        repo: DatasetRepository = ctx['orch'].repo
        root: Path = repo.root_dir  # e.g., <repo>/datasets/<vertical>
        root.mkdir(parents=True, exist_ok=True)
        written = []
        # Read path settings
        cov = CoverageConfig().load_coverage()
        dp = (cov or {}).get('dataset_paths') or {}
        path_mode = (dp.get('mode') or 'flat').lower()
        base_sub = dp.get('base') or None
        # Avoid duplicating the vertical subfolder (e.g., datasets/commerce/commerce)
        try:
            current_vertical = str(ctx.get('vertical', '')).strip().lower() or None
        except Exception:
            current_vertical = None
        def build_paths(ds_id: str, ds_obj: dict) -> tuple[Path, Path]:
            if path_mode == 'hierarchical':
                # datasets/commerce/<behavior>/<version>/<slug>.json
                behavior = None
                try:
                    # prefer from first conversation metadata
                    c0 = (ds_obj.get('conversations') or [{}])[0]
                    behavior = ((c0.get('metadata') or {}).get('behavior'))
                except Exception:
                    behavior = None
                if not behavior:
                    # fallback: parse from dataset_id like "<domain>-<behavior>-v<version>"
                    try:
                        parts = ds_id.split('-')
                        if len(parts) >= 3:
                            behavior = parts[1]
                    except Exception:
                        behavior = 'unknown'
                version = str(ds_obj.get('version') or '1.0.0')
                slug = ds_id
                # Only honor base_sub if it is set and not equal to the current vertical
                use_sub = (isinstance(base_sub, str) and base_sub.strip()) or None
                if use_sub and current_vertical and use_sub.strip().lower() == current_vertical:
                    use_sub = None
                folder = (root / Path(use_sub)) if use_sub else root
                folder = folder / behavior / version
                folder.mkdir(parents=True, exist_ok=True)
                return folder / f"{slug}.dataset.json", folder / f"{slug}.golden.json"
            # default flat: use base subfolder if specified
            use_sub = (isinstance(base_sub, str) and base_sub.strip()) or None
            if use_sub and current_vertical and use_sub.strip().lower() == current_vertical:
                use_sub = None
            folder = (root / use_sub) if use_sub else root
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f"{ds_id}.dataset.json", folder / f"{ds_id}.golden.json"
        for ds, gd in outputs:
            # validate
            ds_errors = repo.sv.validate("dataset", ds)
            if ds_errors:
                raise HTTPException(
                    status_code=400,
                    detail=json.dumps({"type": "dataset", "dataset_id": ds.get("dataset_id"), "errors": ds_errors}),
                )
            gt_errors = repo.sv.validate("golden", gd)
            if gt_errors:
                raise HTTPException(
                    status_code=400,
                    detail=json.dumps({"type": "golden", "dataset_id": gd.get("dataset_id"), "errors": gt_errors}),
                )
            dataset_id = ds["dataset_id"]
            ds_path, gt_path = build_paths(dataset_id, ds)
            if not req.overwrite and (ds_path.exists() or gt_path.exists()):
                raise HTTPException(status_code=409, detail=f"{dataset_id} already exists; set overwrite=true")
            # Ensure parent directories exist (defensive against misconfigured dataset_paths)
            try:
                ds_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                gt_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
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


@app.get("/coverage/taxonomy_v2")
async def coverage_taxonomy_v2():
    try:
        cfg = load_commerce_config()
        tax = cfg.get("taxonomy", {})
        return {
            "domains": tax.get("domains", []),
            "behaviors": tax.get("behaviors", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load v2 taxonomy: {e}")


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


@app.get("/coverage/manifest_v2")
async def coverage_manifest_v2(domains: Optional[str] = None, behaviors: Optional[str] = None, seed: int = 42):
    """Preview manifest built using v2 commerce taxonomy.
    Falls back to legacy exclusions rules (configs/exclusions.json) but enumerates scenarios
    against configs/commerce_taxonomy.json so domain/behavior filters match the UI.
    """
    try:
        cfg2 = load_commerce_config()
        taxonomy_v2 = cfg2.get("taxonomy", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load v2 taxonomy: {e}")
    # Reuse exclusions.json from current config root
    try:
        cov_cfg = CoverageConfig()
        exclusions = cov_cfg.load_exclusions()
    except Exception:
        exclusions = {"rules": []}
    manifest = build_manifest(taxonomy_v2, exclusions, seed)
    pairs = manifest.get("pairs", [])
    doms = set(domains.split(",")) if isinstance(domains, str) and domains else None
    behs = set(behaviors.split(",")) if isinstance(behaviors, str) and behaviors else None
    if doms:
        pairs = [p for p in pairs if p.get("domain") in doms]
    if behs:
        pairs = [p for p in pairs if p.get("behavior") in behs]
    return {"seed": seed, "axes_order": manifest.get("axes_order"), "pairs": pairs}


class CoverageSettingsBody(BaseModel):
    mode: Optional[str] = None
    t: Optional[int] = None
    per_behavior_budget: Optional[int] = None
    anchors: Optional[list[dict]] = None
    sampler: Optional[dict] = None  # { rng_seed?: int, per_behavior_total?: int, min_per_domain?: int }


@app.get("/coverage/settings")
async def coverage_settings_get():
    cfg = CoverageConfig()
    data = cfg.load_coverage()
    return data


@app.post("/coverage/settings")
async def coverage_settings_set(body: CoverageSettingsBody):
    """Update configs/coverage.json (strategy controls). Only merges allowed keys."""
    cfg = CoverageConfig()
    p = cfg.root / 'coverage.json'
    # read existing or default
    current: dict = {}
    if p.exists():
        try:
            current = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            current = {}
    # merge
    def set_if(v, key):
        nonlocal current
        if v is not None:
            current[key] = v
    set_if(body.mode, 'mode')
    set_if(body.t, 't')
    set_if(body.per_behavior_budget, 'per_behavior_budget')
    if body.anchors is not None:
        current['anchors'] = body.anchors
    if body.sampler is not None:
        samp = current.get('sampler') or {}
        for k in ('rng_seed','per_behavior_total','min_per_domain','max_per_domain'):
            if k in body.sampler and body.sampler[k] is not None:
                try:
                    val = int(body.sampler[k])
                except Exception:
                    continue
                if k == 'max_per_domain':
                    if val > 100:
                        val = 100
                    if val < 1:
                        val = 1
                samp[k] = val
        current['sampler'] = samp
    # basic validation
    try:
        # load_coverage will normalize and validate basic fields
        tmp = (cfg.root / 'coverage.json.tmp')
        tmp.write_text(json.dumps(current, indent=2), encoding='utf-8')
        _ = cfg.load_coverage(tmp)
        # ok → write final
        p.write_text(json.dumps(current, indent=2), encoding='utf-8')
        try:
            tmp.unlink()
        except Exception:
            pass
    except Exception as e:
        # rollback temp, surface error
        try:
            tmp.unlink()
        except Exception:
            pass
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"invalid coverage settings: {e}")
    return {"ok": True, "settings": current}


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
async def get_conversation(conversation_id: str, vertical: Optional[str] = None):
    ctx = _get_or_create_vertical_context(vertical)
    repo: DatasetRepository = ctx['orch'].repo
    conv = repo.get_conversation(conversation_id)
    try:
        golden = repo.get_golden(conversation_id)
    except Exception:
        golden = None
    return {"conversation": conv, "golden": golden}


@app.post("/runs", response_model=StartRunResponse)
async def start_run(req: StartRunRequest):
    # choose vertical context from settings if not provided in config.context
    vertical = None
    try:
        vertical = (req.context or {}).get("vertical")  # allow frontend to pass context.vertical
    except Exception:
        vertical = None
    ctx = _get_or_create_vertical_context(vertical)
    orch: Orchestrator = ctx['orch']
    cfg: Dict[str, Any] = {
        "metrics": req.metrics or [],
        "thresholds": req.thresholds or {},
        "context": req.context or {},
    }
    jr = orch.submit(dataset_id=req.dataset_id, model_spec=req.model_spec, config=cfg)
    # initialize run folder with config
    ctx['artifacts'].init_run(jr.run_id, {"dataset_id": req.dataset_id, "model_spec": req.model_spec, **cfg})
    orch.start(jr.job_id)
    return StartRunResponse(job_id=jr.job_id, run_id=jr.run_id, state=jr.state)


@app.post("/runs/{job_id}/control")
async def control_run(job_id: str, body: ControlBody):
    # find job across all vertical orchestrators
    jr = None
    orch: Orchestrator | None = None
    for c in _iter_all_contexts():
        o: Orchestrator = c['orch']
        if job_id in o.jobs:
            orch = o
            jr = o.jobs.get(job_id)
            break
    if orch is None:
        # Allow 'cancel' to mark a stale job as cancelled if persisted job.json exists
        # search across readers
        for c in _iter_all_contexts():
            reader: RunArtifactReader = c['reader']
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
    # search in-memory jobs across verticals
    for c in _iter_all_contexts():
        orch: Orchestrator = c['orch']
        jr = orch.jobs.get(job_id)
        if jr:
            return {
                "job_id": jr.job_id,
                "run_id": jr.run_id,
                "state": jr.state,
                "progress_pct": jr.progress_pct,
                "total_conversations": jr.total_conversations,
                "completed_conversations": jr.completed_conversations,
                "total_turns": getattr(jr, 'total_turns', 0),
                "completed_turns": getattr(jr, 'completed_turns', 0),
                "current_conv_id": getattr(jr, 'current_conv_id', None),
                "current_conv_idx": getattr(jr, 'current_conv_idx', 0),
                "current_conv_total_turns": getattr(jr, 'current_conv_total_turns', 0),
                "current_conv_completed_turns": getattr(jr, 'current_conv_completed_turns', 0),
                "input_tokens_total": getattr(jr, 'input_tokens_total', 0),
                "output_tokens_total": getattr(jr, 'output_tokens_total', 0),
                "error": jr.error,
            }
    # Try to recover from persisted job status if the process lost in-memory job across verticals
    for c in _iter_all_contexts():
        reader: RunArtifactReader = c['reader']
        for p in sorted(reader.layout.runs_root.iterdir()):
            if not p.is_dir():
                continue
            try:
                obj = json.loads((p / "job.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if obj.get("job_id") == job_id:
                if obj.get("boot_id") != BOOT_ID and obj.get("state") in ("running", "paused", "cancelling"):
                    obj = {**obj, "state": "failed", "error": "stale status from previous server session"}
                # Ensure keys exist for frontend
                obj.setdefault("total_turns", 0)
                obj.setdefault("completed_turns", 0)
                obj.setdefault("current_conv_id", None)
                obj.setdefault("current_conv_idx", 0)
                obj.setdefault("current_conv_total_turns", 0)
                obj.setdefault("current_conv_completed_turns", 0)
                obj.setdefault("input_tokens_total", 0)
                obj.setdefault("output_tokens_total", 0)
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
async def run_results(run_id: str, vertical: Optional[str] = None):
    paths = []
    if vertical:
        c = _get_or_create_vertical_context(vertical)
        paths.append(c['reader'].layout.results_json_path(run_id))
    else:
        for c in _iter_all_contexts():
            paths.append(c['reader'].layout.results_json_path(run_id))
    for path in paths:
        if path.exists():
            return get_json_file(path)
    raise HTTPException(status_code=404, detail="results not found")


def get_json_file(path: Path):
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}/artifacts")
async def run_artifacts(run_id: str, type: str = "json", vertical: Optional[str] = None):
    reporter: Reporter = app.state.reporter
    # pick reader by vertical or search
    readers: list[RunArtifactReader] = []
    if vertical:
        readers = [_get_or_create_vertical_context(vertical)['reader']]
    else:
        readers = [c['reader'] for c in _iter_all_contexts()]
    if type == "json":
        for reader in readers:
            path = reader.layout.results_json_path(run_id)
            if path.exists():
                return FileResponse(str(path), media_type="application/json", filename="results.json")
        raise HTTPException(status_code=404, detail="results.json not found")
    elif type == "csv":
        for reader in readers:
            path = reader.layout.results_csv_path(run_id)
            if path.exists():
                return FileResponse(str(path), media_type="text/csv", filename="results.csv")
        raise HTTPException(status_code=404, detail="results.csv not found")
    elif type == "html":
        # generate on the fly from results.json
        json_path = None
        rd_for_html = None
        for reader in readers:
            cand = reader.layout.results_json_path(run_id)
            if cand.exists():
                json_path = cand
                rd_for_html = reader
                break
        if json_path is None:
            raise HTTPException(status_code=404, detail="results.json not found")
        results = get_json_file(json_path)
        try:
            html = reporter.render_html(results)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"cannot render html: {e}")
        # name report using domain, behavior and model if available
        ds_meta = (results.get("metadata") or {}) if isinstance(results, dict) else {}
        convs = results.get("conversations") if isinstance(results, dict) else None
        domain = None
        behavior = None
        if isinstance(convs, list) and len(convs) > 0:
            # infer from first conversation
            c0 = convs[0] or {}
            domain = c0.get("domain") or ds_meta.get("domain")
            behavior = c0.get("behavior") or ds_meta.get("behavior")
        elif ds_meta:
            domain = ds_meta.get("domain")
            behavior = ds_meta.get("behavior")
        model_spec = results.get("model_spec") if isinstance(results, dict) else None
        import re
        def slug(s: str|None) -> str:
            t = (s or "").strip().lower()
            t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
            return t
        fname = "report.html"
        base = "-".join([x for x in [slug(domain), slug(behavior), slug(model_spec)] if x])
        if base:
            fname = f"report-{base}.html"
        out_path = rd_for_html.layout.run_dir(run_id) / fname
        out_path.write_text(html, encoding="utf-8")
        return FileResponse(str(out_path), media_type="text/html", filename=out_path.name)
    elif type == "pdf":
        # Render HTML then convert to PDF (requires WeasyPrint)
        json_path = None
        rd_for_html = None
        for reader in readers:
            cand = reader.layout.results_json_path(run_id)
            if cand.exists():
                json_path = cand
                rd_for_html = reader
                break
        if json_path is None:
            raise HTTPException(status_code=404, detail="results.json not found")
        results = get_json_file(json_path)
        try:
            html = reporter.render_html(results)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"cannot render html: {e}")
        out_dir = rd_for_html.layout.run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        # build pdf filename similarly
        domain = None
        behavior = None
        ds_meta = (results.get("metadata") or {}) if isinstance(results, dict) else {}
        convs = results.get("conversations") if isinstance(results, dict) else None
        if isinstance(convs, list) and len(convs) > 0:
            c0 = convs[0] or {}
            domain = c0.get("domain") or ds_meta.get("domain")
            behavior = c0.get("behavior") or ds_meta.get("behavior")
        else:
            domain = ds_meta.get("domain")
            behavior = ds_meta.get("behavior")
        model_spec = results.get("model_spec") if isinstance(results, dict) else None
        import re, os, shutil
        def slug(s: str|None) -> str:
            t = (s or "").strip().lower()
            t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
            return t
        base = "-".join([x for x in [slug(domain), slug(behavior), slug(model_spec)] if x])
        pdf_name = f"report-{base}.pdf" if base else "report.pdf"
        pdf_path = out_dir / pdf_name

        # Try WeasyPrint first
        weasy_error = None
        try:
            from weasyprint import HTML  # type: ignore
            HTML(string=html, base_url=str(out_dir)).write_pdf(str(pdf_path))
            return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
        except Exception as e:
            weasy_error = e

        # Fallback 1: Playwright (Chromium) rendering
        playwright_error = None
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
            with sync_playwright() as p:
                browser = None
                # 1) Try known channels that use system-installed browsers (no download)
                for ch in ("msedge", "chrome"):
                    try:
                        browser = p.chromium.launch(channel=ch)
                        break
                    except Exception:
                        browser = None
                # 2) Try explicit executable path from env or common Windows installs
                if browser is None:
                    import os
                    cand_paths = [
                        os.environ.get("PLAYWRIGHT_CHROME_PATH"),
                        os.environ.get("CHROME_PATH"),
                        os.environ.get("EDGE_PATH"),
                        r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                        r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                        r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
                        r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
                    ]
                    for ep in [c for c in cand_paths if c]:
                        try:
                            if os.path.exists(ep):
                                browser = p.chromium.launch(executable_path=ep)
                                break
                        except Exception:
                            browser = None
                # 3) Last resort: default launch (may require downloaded browser)
                if browser is None:
                    browser = p.chromium.launch()

                context = browser.new_context()
                page = context.new_page()
                page.set_content(html, wait_until="load")
                page.pdf(path=str(pdf_path), format="A4", print_background=True)
                context.close()
                browser.close()
                return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
        except Exception as e:
            playwright_error = e

        # Fallback 2: wkhtmltopdf via pdfkit
        try:
            import pdfkit  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=501, detail=f"PDF generation not available (WeasyPrint failed: {weasy_error}; Playwright failed: {playwright_error}); pdfkit not installed: {e}")

        # Try default pdfkit auto-detection first (PATH)
        try:
            pdfkit.from_string(html, str(pdf_path), options={"quiet": ""})
            return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
        except Exception:
            pass

        # Locate wkhtmltopdf executable manually
        exe = os.environ.get("WKHTMLTOPDF_PATH") or os.environ.get("WKHTMLTOPDF_BIN") or os.environ.get("WKHTMLTOPDF_BINARY")
        if not exe:
            # common Windows install paths
            candidates = [
                r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
                r"C:\\Program Files (x86)\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
                r"C:\\ProgramData\\chocolatey\\bin\\wkhtmltopdf.exe",
            ]
            for cpath in candidates:
                if os.path.exists(cpath):
                    exe = cpath
                    break
        if not exe:
            exe = shutil.which("wkhtmltopdf")
        # If WKHTMLTOPDF_PATH points to a folder, append executable
        if exe and os.path.isdir(exe):
            candidate = os.path.join(exe, "wkhtmltopdf.exe")
            if os.path.exists(candidate):
                exe = candidate
        if not exe:
            env_path = os.environ.get("PATH", "")
            short_path = (env_path[:240] + '…') if len(env_path) > 240 else env_path
            raise HTTPException(status_code=501, detail=f"wkhtmltopdf not found. Install wkhtmltopdf and/or set WKHTMLTOPDF_PATH to the executable. Alternatively install Playwright: pip install playwright and python -m playwright install chromium. PATH={short_path}")

        try:
            config = pdfkit.configuration(wkhtmltopdf=exe)
            # quiet option to reduce console noise
            pdfkit.from_string(html, str(pdf_path), configuration=config, options={"quiet": ""})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"cannot render pdf with wkhtmltopdf: {e}")
        return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
    else:
        raise HTTPException(status_code=400, detail="unknown type")


    @app.post("/runs/{run_id}/rebuild")
    async def rebuild_run_artifacts(run_id: str, vertical: Optional[str] = None):
        """Rebuild and enrich results.json and results.csv for an existing run.
        Adds human-friendly identity, per-turn snippets, rollups, and writes CSV.
        """
        # locate by vertical or search across
        if vertical:
            ctx = _get_or_create_vertical_context(vertical)
            reader: RunArtifactReader = ctx['reader']
            writer: RunArtifactWriter = ctx['artifacts']
            repo: DatasetRepository = ctx['orch'].repo
        else:
            # search for run_id
            reader = None
            writer = None
            repo = None
            for c in _iter_all_contexts():
                if (c['reader'].layout.run_dir(run_id)).exists():
                    reader = c['reader']
                    writer = c['artifacts']
                    repo = c['orch'].repo
                    break
            if reader is None or writer is None or repo is None:
                raise HTTPException(status_code=404, detail="run not found")
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
async def submit_feedback_api(run_id: str, body: Dict[str, Any], vertical: Optional[str] = None):
    return await _submit_feedback(run_id, body, vertical)

@app.get("/reports/compare")
async def compare_reports(runA: str, runB: str, vertical: Optional[str] = None, type: str = "json"):
    # Locate results for each run; if vertical is provided but doesn't contain the run, fallback to search all
    def _find_reader(run_id: str) -> Optional[RunArtifactReader]:
        if vertical:
            rd = _get_or_create_vertical_context(vertical)['reader']
            p = rd.layout.results_json_path(run_id)
            if p.exists():
                return rd
            # fallback cross-vertical search
            for c in _iter_all_contexts():
                rdx: RunArtifactReader = c['reader']
                if (rdx.layout.results_json_path(run_id)).exists():
                    return rdx
            return None
        # No vertical specified: search all
        for c in _iter_all_contexts():
            rd: RunArtifactReader = c['reader']
            if (rd.layout.results_json_path(run_id)).exists():
                return rd
        return None
    rdA = _find_reader(runA)
    rdB = _find_reader(runB)
    if rdA is None:
        raise HTTPException(status_code=404, detail=f"results.json not found for runA={runA}")
    if rdB is None:
        raise HTTPException(status_code=404, detail=f"results.json not found for runB={runB}")
    pA = rdA.layout.results_json_path(runA)
    pB = rdB.layout.results_json_path(runB)
    if not pA.exists():
        raise HTTPException(status_code=404, detail=f"results.json not found for runA={runA}")
    if not pB.exists():
        raise HTTPException(status_code=404, detail=f"results.json not found for runB={runB}")
    a = json.loads(pA.read_text(encoding='utf-8'))
    b = json.loads(pB.read_text(encoding='utf-8'))
    diff = diff_results(a, b)
    if type == "json":
        return diff
    if type == "csv":
        # Write a compact CSV with sections for per_conversation and per_turn
        # Store under runA folder for convenience
        out_dir = rdA.layout.run_dir(runA)
        safeA = (runA or "").replace("/", "_")[:12]
        safeB = (runB or "").replace("/", "_")[:12]
        out_path = out_dir / f"compare-{safeA}-vs-{safeB}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # per-conversation
            w.writerow(["section", "key", "a_conversation_pass", "b_conversation_pass", "failed_turns_delta"]) 
            for r in (diff.get("per_conversation") or []):
                w.writerow(["per_conversation", r.get("key"), (r.get("a") or {}).get("conversation_pass"), (r.get("b") or {}).get("conversation_pass"), (r.get("delta") or {}).get("failed_turns_delta")])
            # blank line
            w.writerow([])
            # per-turn
            w.writerow(["section", "key", "turn_index", "turn_pass_a", "turn_pass_b", "metrics_changed"]) 
            for r in (diff.get("per_turn") or []):
                m = r.get("metrics") or {}
                changed = sorted([k for k, v in m.items() if (v or {}).get("changed")])
                w.writerow(["per_turn", r.get("key"), r.get("turn_index"), (r.get("turn_pass") or {}).get("a"), (r.get("turn_pass") or {}).get("b"), ";".join(changed)])
        return FileResponse(str(out_path), media_type="text/csv", filename=out_path.name)
    if type == "pdf":
        # Render a simple HTML summary of the diff and convert to PDF
        out_dir = rdA.layout.run_dir(runA)
        out_dir.mkdir(parents=True, exist_ok=True)
        safeA = (runA or "").replace("/", "_")[:32]
        safeB = (runB or "").replace("/", "_")[:32]
        pdf_name = f"compare-{safeA}-vs-{safeB}.pdf"
        pdf_path = out_dir / pdf_name

        def pct(v):
            try:
                return f"{float(v):.2f}%"
            except Exception:
                return "0.00%"

        # Build minimal HTML (self-contained)
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<style>body{font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#111;padding:20px;}",
            "h1{font-size:20px;margin:0 0 8px;} h2{font-size:16px;margin:16px 0 8px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:6px 8px;} th{background:#f6f6f6;text-align:left;} .note{color:#a15c00;font-weight:600;margin:8px 0;} .muted{color:#666;} .good{color:#167516;} .bad{color:#b30000;} .mono{font-family:Consolas,monospace;}</style>",
            "</head><body>",
            f"<h1>Report Comparison</h1>",
            f"<div class='muted'>Generated {ts}</div>",
            f"<div class='mono'>A: {runA} &nbsp;&nbsp; B: {runB}</div>",
        ]
        note = diff.get("note") or (diff.get("alignment") or {}).get("note")
        if note:
            html.append(f"<div class='note'>Note: {note}</div>")
        # Summary cards
        try:
            ca = diff.get("runA", {}).get("summary", {})
            cb = diff.get("runB", {}).get("summary", {})
            html.append("<h2>Summary</h2>")
            html.append("<table><thead><tr><th></th><th>A</th><th>B</th></tr></thead><tbody>")
            html.append(f"<tr><td>Conversations pass rate</td><td>{pct(ca.get('conv',{}).get('pass_rate',0))}</td><td>{pct(cb.get('conv',{}).get('pass_rate',0))}</td></tr>")
            html.append(f"<tr><td>Turns pass rate</td><td>{pct(ca.get('turn',{}).get('pass_rate',0))}</td><td>{pct(cb.get('turn',{}).get('pass_rate',0))}</td></tr>")
            html.append("</tbody></table>")
        except Exception:
            pass
        # Metrics delta
        md = diff.get("metrics_delta") or {}
        if md:
            html.append("<h2>Per-metric Pass Rate Delta</h2>")
            html.append("<table><thead><tr><th>Metric</th><th>A</th><th>B</th><th>Δ (B−A)</th></tr></thead><tbody>")
            for k, row in md.items():
                a_pr = row.get("a_pass_rate", 0)
                b_pr = row.get("b_pass_rate", 0)
                d = row.get("delta", 0)
                cls = "good" if d >= 0 else "bad"
                html.append(f"<tr><td>{k}</td><td>{pct(a_pr)}</td><td>{pct(b_pr)}</td><td class='{cls}'>{pct(d)}</td></tr>")
            html.append("</tbody></table>")
        # Domain/behavior
        db = diff.get("domain_behavior_delta") or []
        if db:
            html.append("<h2>Domain/Behavior Delta</h2>")
            html.append("<table><thead><tr><th>Domain</th><th>Behavior</th><th>A pass%</th><th>B pass%</th><th>Δ</th></tr></thead><tbody>")
            for r in db[:200]:
                a_pr = (r.get("a") or {}).get("pass_rate", 0)
                b_pr = (r.get("b") or {}).get("pass_rate", 0)
                d = (r.get("delta") or {}).get("pass_rate", 0)
                cls = "good" if d >= 0 else "bad"
                html.append(f"<tr><td>{r.get('domain')}</td><td>{r.get('behavior')}</td><td>{pct(a_pr)}</td><td>{pct(b_pr)}</td><td class='{cls}'>{pct(d)}</td></tr>")
            html.append("</tbody></table>")
        # Per conversation
        pc = diff.get("per_conversation") or []
        if pc:
            html.append("<h2>Per-conversation Changes (Top 50)</h2>")
            html.append("<table><thead><tr><th>Key</th><th>Pass A→B</th><th>Failed turns Δ</th></tr></thead><tbody>")
            for r in pc[:50]:
                atxt = str(((r.get("a") or {}).get("conversation_pass")))
                btxt = str(((r.get("b") or {}).get("conversation_pass")))
                delta = (r.get("delta") or {}).get("failed_turns_delta", 0)
                cls = "good" if delta < 0 else ("bad" if delta > 0 else "")
                html.append(f"<tr><td>{r.get('key')}</td><td>{atxt} → {btxt}</td><td class='{cls}'>{delta}</td></tr>")
            html.append("</tbody></table>")
        html.append("</body></html>")
        html = "".join(html)

        # Try WeasyPrint first
        weasy_error = None
        try:
            from weasyprint import HTML  # type: ignore
            HTML(string=html, base_url=str(out_dir)).write_pdf(str(pdf_path))
            return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
        except Exception as e:
            weasy_error = e

        # Fallback 1: Playwright (Chromium) rendering
        playwright_error = None
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
            with sync_playwright() as p:
                browser = None
                for ch in ("msedge", "chrome"):
                    try:
                        browser = p.chromium.launch(channel=ch)
                        break
                    except Exception:
                        browser = None
                if browser is None:
                    import os
                    cand_paths = [
                        os.environ.get("PLAYWRIGHT_CHROME_PATH"),
                        os.environ.get("CHROME_PATH"),
                        os.environ.get("EDGE_PATH"),
                        r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                        r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                        r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
                        r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
                    ]
                    for ep in [c for c in cand_paths if c]:
                        try:
                            if os.path.exists(ep):
                                browser = p.chromium.launch(executable_path=ep)
                                break
                        except Exception:
                            browser = None
                if browser is None:
                    browser = p.chromium.launch()
                context = browser.new_context()
                page = context.new_page()
                page.set_content(html, wait_until="load")
                page.pdf(path=str(pdf_path), format="A4", print_background=True)
                context.close()
                browser.close()
                return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
        except Exception as e:
            playwright_error = e

        # Fallback 2: wkhtmltopdf via pdfkit
        try:
            import pdfkit  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=501, detail=f"PDF generation not available (WeasyPrint failed: {weasy_error}; Playwright failed: {playwright_error}); pdfkit not installed: {e}")

        try:
            pdfkit.from_string(html, str(pdf_path), options={"quiet": ""})
            return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
        except Exception:
            pass
        import os, shutil
        exe = os.environ.get("WKHTMLTOPDF_PATH") or os.environ.get("WKHTMLTOPDF_BIN") or os.environ.get("WKHTMLTOPDF_BINARY")
        if not exe:
            candidates = [
                r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
                r"C:\\Program Files (x86)\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
                r"C:\\ProgramData\\chocolatey\\bin\\wkhtmltopdf.exe",
            ]
            for cpath in candidates:
                if os.path.exists(cpath):
                    exe = cpath
                    break
        if not exe:
            exe = shutil.which("wkhtmltopdf")
        if exe and os.path.isdir(exe):
            candidate = os.path.join(exe, "wkhtmltopdf.exe")
            if os.path.exists(candidate):
                exe = candidate
        if not exe:
            env_path = os.environ.get("PATH", "")
            short_path = (env_path[:240] + '…') if len(env_path) > 240 else env_path
            raise HTTPException(status_code=501, detail=f"wkhtmltopdf not found. Install wkhtmltopdf and/or set WKHTMLTOPDF_PATH to the executable. Alternatively install Playwright: pip install playwright and python -m playwright install chromium. PATH={short_path}")
        try:
            config = pdfkit.configuration(wkhtmltopdf=exe)
            pdfkit.from_string(html, str(pdf_path), configuration=config, options={"quiet": ""})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"cannot render pdf with wkhtmltopdf: {e}")
        return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)
    # PDF export is not implemented yet
    raise HTTPException(status_code=501, detail="unknown export type")

async def _submit_feedback(run_id: str, body: Dict[str, Any], vertical: Optional[str] = None):
    # Append feedback objects to runs/<run_id>/feedback.json
    # locate run dir
    if vertical:
        run_dir = _get_or_create_vertical_context(vertical)['reader'].layout.run_dir(run_id)
    else:
        run_dir = None
        for c in _iter_all_contexts():
            cand = c['reader'].layout.run_dir(run_id)
            if cand.exists():
                run_dir = cand
                break
        if run_dir is None:
            raise HTTPException(status_code=404, detail="run not found")
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
async def _submit_feedback(run_id: str, body: Dict[str, Any], vertical: Optional[str] = None):
    # Append feedback objects to runs/<run_id>/feedback.json
    # locate run dir
    if vertical:
        run_dir = _get_or_create_vertical_context(vertical)['reader'].layout.run_dir(run_id)
    else:
        run_dir = None
        for c in _iter_all_contexts():
            cand = c['reader'].layout.run_dir(run_id)
            if cand.exists():
                run_dir = cand
                break
        if run_dir is None:
            raise HTTPException(status_code=404, detail="run not found")
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
async def compare_runs(runA: str, runB: str, verticalA: Optional[str] = None, verticalB: Optional[str] = None):
    paths = []
    # resolve for A
    if verticalA:
        ca = _get_or_create_vertical_context(verticalA)['reader']
        a = ca.layout.results_json_path(runA)
    else:
        a = None
        for c in _iter_all_contexts():
            cand = c['reader'].layout.results_json_path(runA)
            if cand.exists():
                a = cand
                break
    # resolve for B
    if verticalB:
        cb = _get_or_create_vertical_context(verticalB)['reader']
        b = cb.layout.results_json_path(runB)
    else:
        b = None
        for c in _iter_all_contexts():
            cand = c['reader'].layout.results_json_path(runB)
            if cand.exists():
                b = cand
                break
    if a is None or b is None or (not a.exists()) or (not b.exists()):
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
    model_config = ConfigDict(protected_namespaces=())
    run_id: str
    dataset_id: Optional[str] = None
    model_spec: Optional[str] = None
    has_results: bool
    created_ts: Optional[float] = None
    vertical: Optional[str] = None


@app.get("/runs")
async def list_runs(vertical: Optional[str] = None):
    """List runs by inspecting runs/<vertical>/ folder. Defaults to selected vertical."""
    contexts = [_get_or_create_vertical_context(vertical)] if vertical else _iter_all_contexts()
    items: list[dict[str, Any]] = []
    for c in contexts:
        vname = c['vertical']
        reader: RunArtifactReader = c['reader']
        layout = reader.layout
        if not layout.runs_root.exists():
            continue
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
            job_state = reader.read_job_status(run_id)
            # Determine staleness
            boot_id = (job_state or {}).get('boot_id')
            is_stale = (boot_id is None) or (boot_id != BOOT_ID)
            state_val = (job_state or {}).get('state')
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
                'vertical': vname,
            })
    return items


@app.post("/validate")
async def validate_json(body: Dict[str, Any]):
    """Validate payload against a named schema without saving.
    Body shape: {"type": "dataset"|"golden"|"run_config", "payload": {...}}
    """
    # Use SchemaValidator directly (not tied to a vertical)
    try:
        from .schemas import SchemaValidator  # type: ignore
    except Exception:
        from backend.schemas import SchemaValidator  # type: ignore
    sv = SchemaValidator()
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

