from __future__ import annotations
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .dataset_repo import DatasetRepository
    from .turn_runner import TurnRunner
    from .artifacts import RunArtifactWriter
    from .metrics import exact_match, semantic_similarity
    from .metrics_extra import consistency, adherence, hallucination
    from .conversation_scoring import aggregate_conversation
except ImportError:  # test fallback
    from backend.dataset_repo import DatasetRepository
    from backend.turn_runner import TurnRunner
    from backend.artifacts import RunArtifactWriter
    from backend.metrics import exact_match, semantic_similarity
    from backend.metrics_extra import consistency, adherence, hallucination
    from backend.conversation_scoring import aggregate_conversation


JobState = str  # 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_run_id(dataset_id: str, dataset_version: str, model_spec: str, config: Dict[str, Any]) -> str:
    # Deterministic checksum of relevant config fields
    relevant = {
        "metrics": config.get("metrics"),
        "thresholds": config.get("thresholds"),
        "context": config.get("context"),
    }
    blob = json.dumps(relevant, sort_keys=True).encode("utf-8")
    cksum = hashlib.sha256(blob).hexdigest()[:8]
    safe_model = model_spec.replace(":", "-")
    return f"{dataset_id}-{dataset_version}-{safe_model}-{cksum}"


@dataclass
class JobRecord:
    job_id: str
    run_id: str
    config: Dict[str, Any]
    state: JobState = "queued"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    progress_pct: int = 0
    total_conversations: int = 0
    completed_conversations: int = 0
    total_turns: int = 0
    completed_turns: int = 0
    current_conv_id: Optional[str] = None
    current_conv_idx: int = 0
    current_conv_total_turns: int = 0
    current_conv_completed_turns: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    error: Optional[str] = None
    _task: Optional[asyncio.Task] = None
    _cancel: bool = False
    _pause: bool = False


class Orchestrator:
    def __init__(self, datasets_dir: Optional[Path] = None, runs_root: Optional[Path] = None, boot_id: Optional[str] = None) -> None:
        self.repo = DatasetRepository(datasets_dir)
        self.runs_root = Path(runs_root) if runs_root else Path(__file__).resolve().parents[1] / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, JobRecord] = {}
        self._id_seq = 0
        self._runner = TurnRunner(self.runs_root)
        self._writer = RunArtifactWriter(self.runs_root)
        self.boot_id = boot_id or "unknown"

    @staticmethod
    def parse_model_spec(model_spec: str) -> tuple[str, str]:
        # format provider:model, e.g., 'ollama:llama3.2:latest', 'gemini:gemini-2.5', 'openai:gpt-5.1'
        parts = model_spec.split(":", 1)
        if len(parts) != 2:
            raise ValueError("model spec must be 'provider:model'")
        return parts[0], parts[1]

    def submit(self, *, dataset_id: str, model_spec: str, config: Dict[str, Any]) -> JobRecord:
        ds = self.repo.get_dataset(dataset_id)
        run_id = compute_run_id(ds["dataset_id"], ds["version"], model_spec, config)
        self._id_seq += 1
        job_id = f"job-{self._id_seq:04d}"
        jr = JobRecord(job_id=job_id, run_id=run_id, config={"dataset_id": dataset_id, "model_spec": model_spec, **config})
        jr.total_conversations = len(ds.get("conversations", []))
        # compute total user turns across all conversations for turn-level progress
        try:
            jr.total_turns = sum(
                sum(1 for t in (conv.get("turns", []) or []) if t.get("role") == "user")
                for conv in (ds.get("conversations", []) or [])
            )
        except Exception:
            jr.total_turns = 0
        self.jobs[job_id] = jr
        # persist initial job status
        try:
            self._writer.write_job_status(run_id, {
                "job_id": job_id,
                "run_id": run_id,
                "state": jr.state,
                "progress_pct": jr.progress_pct,
                "total_conversations": jr.total_conversations,
                "completed_conversations": jr.completed_conversations,
                "total_turns": jr.total_turns,
                "completed_turns": jr.completed_turns,
                "input_tokens_total": jr.input_tokens_total,
                "output_tokens_total": jr.output_tokens_total,
                "error": None,
                "boot_id": self.boot_id,
            })
        except Exception:
            pass
        return jr

    def cancel(self, job_id: str) -> None:
        jr = self.jobs[job_id]
        jr._cancel = True
        jr.updated_at = _now_iso()
        # If there's an active task, cancel it immediately and mark cancelled
        if jr._task and not jr._task.done():
            try:
                jr._task.cancel()
            except Exception:
                pass
            jr.state = "cancelled"
        else:
            # If not yet started or already paused, mark cancelled
            jr.state = "cancelled" if jr.state in ("queued", "paused") else "cancelling"
        try:
            self._writer.write_job_status(jr.run_id, {
                "job_id": jr.job_id,
                "run_id": jr.run_id,
                "state": jr.state,
                "progress_pct": jr.progress_pct,
                "total_conversations": jr.total_conversations,
                "completed_conversations": jr.completed_conversations,
                "error": "cancelled by user" if jr.state == "cancelled" else None,
                "boot_id": self.boot_id,
            })
        except Exception:
            pass

    def pause(self, job_id: str) -> None:
        jr = self.jobs[job_id]
        if jr.state in ("succeeded", "failed", "cancelled") or (jr._task and jr._task.done()):
            raise RuntimeError("cannot pause a completed job")
        if jr.state == "paused":
            return
        jr._pause = True
        jr.state = "paused"
        jr.updated_at = _now_iso()
        try:
            self._writer.write_job_status(jr.run_id, {
                "job_id": jr.job_id,
                "run_id": jr.run_id,
                "state": jr.state,
                "progress_pct": jr.progress_pct,
                "total_conversations": jr.total_conversations,
                "completed_conversations": jr.completed_conversations,
                "error": None,
                "boot_id": self.boot_id,
            })
        except Exception:
            pass

    def resume(self, job_id: str) -> None:
        jr = self.jobs[job_id]
        if jr.state in ("succeeded", "failed", "cancelled") or (jr._task and jr._task.done()):
            raise RuntimeError("cannot resume a completed job")
        jr._pause = False
        jr.state = "running"
        jr.updated_at = _now_iso()
        try:
            self._writer.write_job_status(jr.run_id, {
                "job_id": jr.job_id,
                "run_id": jr.run_id,
                "state": jr.state,
                "progress_pct": jr.progress_pct,
                "total_conversations": jr.total_conversations,
                "completed_conversations": jr.completed_conversations,
                "error": None,
                "boot_id": self.boot_id,
            })
        except Exception:
            pass

    async def run_job(self, job_id: str) -> JobRecord:
        jr = self.jobs[job_id]
        try:
            if jr.state not in ("queued",):
                return jr
            jr.state = "running"
            jr.updated_at = _now_iso()
            # write running status
            try:
                self._writer.write_job_status(jr.run_id, {
                    "job_id": jr.job_id,
                    "run_id": jr.run_id,
                    "state": jr.state,
                    "progress_pct": jr.progress_pct,
                    "total_conversations": jr.total_conversations,
                    "completed_conversations": jr.completed_conversations,
                    "total_turns": jr.total_turns,
                    "completed_turns": jr.completed_turns,
                    "input_tokens_total": jr.input_tokens_total,
                    "output_tokens_total": jr.output_tokens_total,
                    "error": None,
                    "boot_id": self.boot_id,
                })
            except Exception:
                pass

            ds = self.repo.get_dataset(jr.config["dataset_id"])
            provider, model = self.parse_model_spec(jr.config["model_spec"])  # e.g., 'ollama', 'llama3.2:2b'
            domain = ds.get("metadata", {}).get("domain", "commerce")
            # Normalize metric selection from run config
            wanted_raw: List[str] = list(jr.config.get("metrics") or ["exact"])  # default to exact only
            map_names = {
                "exact_match": "exact",
                "exact": "exact",
                "semantic_similarity": "semantic",
                "semantic": "semantic",
                "consistency": "consistency",
                "adherence": "adherence",
                "hallucination": "hallucination",
            }
            metrics_wanted: List[str] = []
            for name in wanted_raw:
                norm = map_names.get(str(name))
                if norm and norm not in metrics_wanted:
                    metrics_wanted.append(norm)

            # Ensure run folder exists even if downstream is mocked
            run_folder = self.runs_root / jr.run_id
            run_folder.mkdir(parents=True, exist_ok=True)

            # Helpers for identity enrichment
            def _slugify(text: str) -> str:
                import re
                t = (text or "").lower()
                t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
                return t[:80]

            def _conv_identity(conv_obj: Dict[str, Any]) -> Dict[str, Any]:
                meta_ds = ds.get("metadata", {}) or {}
                meta = (conv_obj.get("metadata") or {}) if isinstance(conv_obj.get("metadata"), dict) else {}
                # fields
                d = meta.get("domain") or meta_ds.get("domain")
                b = meta.get("behavior") or meta_ds.get("behavior")
                s = meta.get("scenario") or meta.get("case")
                persona = meta.get("persona")
                locale = meta.get("locale")
                channel = meta.get("channel")
                complexity = meta.get("complexity") or meta_ds.get("difficulty")
                case_type = meta.get("case_type") or meta.get("type")
                title = conv_obj.get("title") or (
                    (f"{b}: {s}" if b and s else (b or s)) if (b or s) else None
                ) or conv_obj.get("conversation_id")
                parts = [p for p in [d, b, s, persona, locale] if p]
                slug = _slugify("-".join(parts)) if parts else _slugify(conv_obj.get("conversation_id", "conv"))
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

            # Simple per-run embedding cache for semantic metric
            embed_cache: Dict[str, List[float]] = {}

            for cidx, conv in enumerate(ds.get("conversations", [])):
                if jr._cancel:
                    jr.state = "cancelled"
                    jr.updated_at = _now_iso()
                    try:
                        self._writer.write_job_status(jr.run_id, {
                            "job_id": jr.job_id,
                            "run_id": jr.run_id,
                            "state": jr.state,
                            "progress_pct": jr.progress_pct,
                            "total_conversations": jr.total_conversations,
                            "completed_conversations": jr.completed_conversations,
                            "error": None,
                            "boot_id": self.boot_id,
                        })
                    except Exception:
                        pass
                    return jr
                # Pause gate before each conversation and between turns
                if jr._pause:
                    jr.state = "paused"
                    jr.updated_at = _now_iso()
                    try:
                        self._writer.write_job_status(jr.run_id, {
                            "job_id": jr.job_id,
                            "run_id": jr.run_id,
                            "state": jr.state,
                            "progress_pct": jr.progress_pct,
                            "total_conversations": jr.total_conversations,
                            "completed_conversations": jr.completed_conversations,
                            "error": None,
                            "boot_id": self.boot_id,
                        })
                    except Exception:
                        pass
                    # wait until unpaused or cancelled
                    while jr._pause and not jr._cancel:
                        await asyncio.sleep(0.3)
                    if jr._cancel:
                        jr.state = "cancelled"
                        jr.updated_at = _now_iso()
                        return jr
                    jr.state = "running"
                    jr.updated_at = _now_iso()
                    try:
                        self._writer.write_job_status(jr.run_id, {
                            "job_id": jr.job_id,
                            "run_id": jr.run_id,
                            "state": jr.state,
                            "progress_pct": jr.progress_pct,
                            "total_conversations": jr.total_conversations,
                            "completed_conversations": jr.completed_conversations,
                            "error": None,
                            "boot_id": self.boot_id,
                        })
                    except Exception:
                        pass
                conv_id = conv.get("conversation_id")
                conv_meta = (conv.get("metadata") or {}) if isinstance(conv.get("metadata"), dict) else {}
                turns = conv.get("turns", [])
                # initialize per-conversation turn progress
                try:
                    jr.current_conv_id = conv_id
                    jr.current_conv_idx = int(cidx) + 1
                    jr.current_conv_total_turns = sum(1 for t in (turns or []) if t.get("role") == "user")
                    jr.current_conv_completed_turns = 0
                except Exception:
                    jr.current_conv_total_turns = 0
                    jr.current_conv_completed_turns = 0
                try:
                    self._writer.write_job_status(jr.run_id, {
                        "job_id": jr.job_id,
                        "run_id": jr.run_id,
                        "state": jr.state,
                        "progress_pct": jr.progress_pct,
                        "total_conversations": jr.total_conversations,
                        "completed_conversations": jr.completed_conversations,
                        "total_turns": jr.total_turns,
                        "completed_turns": jr.completed_turns,
                        "current_conv_id": jr.current_conv_id,
                        "current_conv_idx": jr.current_conv_idx,
                        "current_conv_total_turns": jr.current_conv_total_turns,
                        "current_conv_completed_turns": jr.current_conv_completed_turns,
                        "input_tokens_total": jr.input_tokens_total,
                        "output_tokens_total": jr.output_tokens_total,
                        "error": None,
                        "boot_id": self.boot_id,
                    })
                except Exception:
                    pass
                # iterate user turns only
                for idx, t in enumerate(turns):
                    if t.get("role") == "user":
                        # inner pause gate before each user turn
                        if jr._cancel:
                            jr.state = "cancelled"
                            jr.updated_at = _now_iso()
                            try:
                                self._writer.write_job_status(jr.run_id, {
                                    "job_id": jr.job_id,
                                    "run_id": jr.run_id,
                                    "state": jr.state,
                                    "progress_pct": jr.progress_pct,
                                    "total_conversations": jr.total_conversations,
                                    "completed_conversations": jr.completed_conversations,
                                    "total_turns": jr.total_turns,
                                    "completed_turns": jr.completed_turns,
                                    "error": None,
                                    "boot_id": self.boot_id,
                                })
                            except Exception:
                                pass
                            return jr
                        if jr._pause:
                            jr.state = "paused"
                            jr.updated_at = _now_iso()
                            try:
                                self._writer.write_job_status(jr.run_id, {
                                    "job_id": jr.job_id,
                                    "run_id": jr.run_id,
                                    "state": jr.state,
                                    "progress_pct": jr.progress_pct,
                                    "total_conversations": jr.total_conversations,
                                    "completed_conversations": jr.completed_conversations,
                                    "total_turns": jr.total_turns,
                                    "completed_turns": jr.completed_turns,
                                    "error": None,
                                    "boot_id": self.boot_id,
                                })
                            except Exception:
                                pass
                            while jr._pause and not jr._cancel:
                                await asyncio.sleep(0.3)
                            if jr._cancel:
                                jr.state = "cancelled"
                                jr.updated_at = _now_iso()
                                try:
                                    self._writer.write_job_status(jr.run_id, {
                                        "job_id": jr.job_id,
                                        "run_id": jr.run_id,
                                        "state": jr.state,
                                        "progress_pct": jr.progress_pct,
                                        "total_conversations": jr.total_conversations,
                                        "completed_conversations": jr.completed_conversations,
                                        "total_turns": jr.total_turns,
                                        "completed_turns": jr.completed_turns,
                                        "error": None,
                                        "boot_id": self.boot_id,
                                    })
                                except Exception:
                                    pass
                                return jr
                            jr.state = "running"
                            jr.updated_at = _now_iso()
                            try:
                                self._writer.write_job_status(jr.run_id, {
                                    "job_id": jr.job_id,
                                    "run_id": jr.run_id,
                                    "state": jr.state,
                                    "progress_pct": jr.progress_pct,
                                    "total_conversations": jr.total_conversations,
                                    "completed_conversations": jr.completed_conversations,
                                    "total_turns": jr.total_turns,
                                    "completed_turns": jr.completed_turns,
                                    "error": None,
                                    "boot_id": self.boot_id,
                                })
                            except Exception:
                                pass
                        # Allow run-level decoding overrides via config.context.params
                        params_override = None
                        try:
                            params_override = (jr.config.get("context") or {}).get("params")
                        except Exception:
                            params_override = None
                        rec = await self._runner.run_turn(
                            run_id=jr.run_id,
                            provider=provider,
                            model=model,
                            domain=domain,
                            conversation_id=conv_id,
                            turn_index=idx,
                            turns=turns[: idx + 1],
                            conv_meta=conv_meta,
                            params_override=params_override,
                        )
                        # Extract and accumulate token usage from this turn
                        try:
                            provider_meta = (rec.get("response", {}) or {}).get("provider_meta", {})
                            usage = provider_meta.get("usage") if isinstance(provider_meta, dict) else None
                            if usage:
                                in_tok = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                                out_tok = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                                jr.input_tokens_total += int(in_tok)
                                jr.output_tokens_total += int(out_tok)
                        except Exception:
                            pass
                        # Surface provider errors as a job-level warning for frontend display
                        try:
                            resp_ok = bool(((rec.get("response", {}) or {}).get("ok")))
                            if not resp_ok:
                                err_txt = (rec.get("response", {}) or {}).get("error")
                                if err_txt:
                                    # Truncate to a reasonable summary length
                                    jr.error = f"{provider}:{model} — {str(err_txt)[:280]}"
                            else:
                                # Clear transient provider error on success
                                if getattr(jr, "error", None):
                                    jr.error = None
                        except Exception:
                            pass
                        # update turn-level progress
                        try:
                            jr.completed_turns += 1
                            jr.current_conv_completed_turns += 1
                        except Exception:
                            pass
                        try:
                            self._writer.write_job_status(jr.run_id, {
                                "job_id": jr.job_id,
                                "run_id": jr.run_id,
                                "state": jr.state,
                                "progress_pct": jr.progress_pct,
                                "total_conversations": jr.total_conversations,
                                "completed_conversations": jr.completed_conversations,
                                "total_turns": jr.total_turns,
                                "completed_turns": jr.completed_turns,
                                "current_conv_id": jr.current_conv_id,
                                "current_conv_idx": jr.current_conv_idx,
                                "current_conv_total_turns": jr.current_conv_total_turns,
                                "current_conv_completed_turns": jr.current_conv_completed_turns,
                                "error": jr.error,
                                "boot_id": self.boot_id,
                            })
                        except Exception:
                            pass
                jr.completed_conversations += 1
                jr.progress_pct = int(jr.completed_conversations * 100 / max(1, jr.total_conversations))
                jr.updated_at = _now_iso()
                try:
                    self._writer.write_job_status(jr.run_id, {
                        "job_id": jr.job_id,
                        "run_id": jr.run_id,
                        "state": jr.state,
                        "progress_pct": jr.progress_pct,
                        "total_conversations": jr.total_conversations,
                        "completed_conversations": jr.completed_conversations,
                        "error": None,
                        "boot_id": self.boot_id,
                    })
                except Exception:
                    pass

            # Aggregate results across conversations and write artifacts
            results: Dict[str, Any] = {
                "run_id": jr.run_id,
                "dataset_id": ds.get("dataset_id"),
                "model_spec": jr.config.get("model_spec"),
                "conversations": [],
            }
            # Accumulate token usage across all turns
            total_input_tokens = 0
            total_output_tokens = 0
            # include dataset/domain short description if present
            try:
                results["domain_description"] = (ds.get("metadata", {}) or {}).get("short_description")
            except Exception:
                pass

            for conv in ds.get("conversations", []):
                cid = conv.get("conversation_id")
                # Locate conversation trace directory (support both hashed and plain layouts)
                try:
                    from .artifacts import RunFolderLayout  # type: ignore
                except Exception:
                    from artifacts import RunFolderLayout  # type: ignore
                # Prefer plain layout used by TurnRunner in tests; fallback to hashed
                conv_dir_plain = self.runs_root / jr.run_id / "conversations" / cid
                conv_dir_hashed = RunFolderLayout(self.runs_root).conversation_subdir(jr.run_id, cid)
                if conv_dir_plain.exists():
                    conv_dir = conv_dir_plain
                elif conv_dir_hashed.exists():
                    conv_dir = conv_dir_hashed
                else:
                    # default to hashed to avoid path length issues
                    conv_dir = conv_dir_hashed
                turn_files = sorted(conv_dir.glob("turn_*.json"))
                per_turn: List[Dict[str, Any]] = []
                identity = _conv_identity(conv)
                # Preserve axes for downstream risk rollups
                try:
                    axes = (conv.get("metadata") or {}).get("axes") or {}
                    if isinstance(axes, dict):
                        identity["axes"] = axes
                except Exception:
                    pass
                # attach axes for downstream rollups/reporting
                try:
                    axes = (conv.get("metadata") or {}).get("axes") or {}
                    if isinstance(axes, dict):
                        identity["axes"] = axes
                except Exception:
                    pass
                # build golden maps
                golden_entry = None
                golden_outcome: Dict[str, Any] = {}
                golden_constraints: Dict[str, Any] | None = None
                try:
                    g = self.repo.get_golden(cid)
                    golden_entry = {t.get("turn_index"): (t.get("expected", {}) or {}).get("variants", []) for t in (g.get("entry", {}).get("turns", []) or [])}
                    # Properly handle final_outcome: prefer entry.final_outcome, fallback to top-level final_outcome
                    entry_outcome = g.get("entry", {}).get("final_outcome")
                    if entry_outcome is not None:
                        golden_outcome = entry_outcome
                    else:
                        golden_outcome = g.get("final_outcome") or {}
                    golden_constraints = g.get("entry", {}).get("constraints") or g.get("constraints")
                except Exception as e:
                    import sys
                    print(f"[DEBUG] Failed to load golden for {cid}: {e}", file=sys.stderr)
                    pass

                # Build a map of available turn records by index
                rec_map: Dict[int, Dict[str, Any]] = {}
                for tf in turn_files:
                    try:
                        rec = json.loads(tf.read_text(encoding="utf-8"))
                        try:
                            ridx = int(rec.get("turn_index", -1))
                        except Exception:
                            ridx = -1
                        if ridx >= 0:
                            rec_map[ridx] = rec
                    except Exception:
                        continue

                # Determine expected user turn indices from dataset conversation
                tlist = conv.get("turns", []) or []
                expected_user_idxs: List[int] = []
                try:
                    # Prefer explicit roles; fallback to all indices when roles absent
                    user_positions = [i for i, tt in enumerate(tlist) if (tt or {}).get("role", "user") == "user"]
                    expected_user_idxs = user_positions if user_positions else list(range(len(tlist)))
                except Exception:
                    expected_user_idxs = list(range(len(tlist)))

                last_state: Dict[str, Any] = {}
                for uidx in expected_user_idxs:
                    rec = rec_map.get(uidx)
                    # Read fields from record if present; otherwise synthesize placeholders
                    if rec is None:
                        rec = {
                            "turn_index": uidx,
                            "request": {"messages": [], "params": {}},
                            "response": {"ok": False, "content": "", "latency_ms": 0, "provider_meta": {}, "error": "turn artifact missing"},
                            "state": {},
                        }
                    out_text = ((rec.get("response", {}) or {}).get("content")) or ""
                    resp_ok = bool(((rec.get("response", {}) or {}).get("ok")))
                    resp_err = ((rec.get("response", {}) or {}).get("error"))
                    resp_meta = ((rec.get("response", {}) or {}).get("provider_meta"))
                    uidx = int(rec.get("turn_index", 0))
                    # Token accounting from provider metadata when available; otherwise approximate
                    try:
                        pm = ((rec.get("response", {}) or {}).get("provider_meta") or {})
                        usage = pm.get("usage") if isinstance(pm, dict) else None
                        in_tok = None
                        out_tok = None
                        if isinstance(usage, dict):
                            # OpenAI-style usage
                            if "prompt_tokens" in usage:
                                in_tok = int(usage.get("prompt_tokens") or 0)
                            if "completion_tokens" in usage:
                                out_tok = int(usage.get("completion_tokens") or 0)
                            if in_tok is None and "input_tokens" in usage:
                                in_tok = int(usage.get("input_tokens") or 0)
                            if out_tok is None and "output_tokens" in usage:
                                out_tok = int(usage.get("output_tokens") or 0)
                        # Ollama-style counters
                        if in_tok is None and isinstance(pm, dict) and "prompt_eval_count" in pm:
                            try:
                                in_tok = int(pm.get("prompt_eval_count") or 0)
                            except Exception:
                                in_tok = 0
                        if out_tok is None and isinstance(pm, dict) and "eval_count" in pm:
                            try:
                                out_tok = int(pm.get("eval_count") or 0)
                            except Exception:
                                out_tok = 0
                        # Fallback to rough estimates if still missing
                        if in_tok is None:
                            try:
                                ctx_est = int((rec.get("context_audit", {}) or {}).get("token_estimate") or 0)
                            except Exception:
                                ctx_est = 0
                            in_tok = ctx_est
                        if out_tok is None:
                            try:
                                out_tok = max(0, int(len(out_text) / 4.0))
                            except Exception:
                                out_tok = 0
                        total_input_tokens += int(in_tok or 0)
                        total_output_tokens += int(out_tok or 0)
                    except Exception:
                        pass
                    # Robust mapping of user turn index -> assistant turn index in golden
                    # Preferred (convgen_v2): A1=1, A2=3 => assistant_idx = 2*uidx + 1
                    cand_idxs = [2 * uidx + 1, uidx + 1, uidx]
                    # derive user prompt snippet from dataset conversation
                    user_text = ""
                    try:
                        if 0 <= uidx < len(tlist):
                            user_text = str((tlist[uidx] or {}).get("text") or "")
                    except Exception:
                        user_text = ""
                    def _snippet(t: str, n: int = 160) -> str:
                        t = (t or "").strip().replace("\n", " ")
                        return t if len(t) <= n else (t[: n - 1] + "...")
                    mets: Dict[str, Any] = {}
                    # If provider failed, record a provider metric that forces failure
                    if not resp_ok:
                        mets["provider"] = {"metric": "provider", "pass": False, "error": resp_err or "provider error"}
                    # exact (if selected and golden exists)
                    exp_variants = []
                    if golden_entry:
                        # pick first matching candidate index
                        for ax in cand_idxs:
                            if ax in golden_entry:
                                exp_variants = golden_entry[ax]
                                break
                        if "exact" in metrics_wanted:
                            try:
                                mets["exact"] = exact_match(out_text, exp_variants)
                            except Exception as e:
                                mets["exact"] = {"metric": "exact", "pass": False, "error": str(e)}
                        if "semantic" in metrics_wanted:
                            try:
                                # semantic may fail if embeddings not available
                                thr = (jr.config.get("thresholds", {}) or {}).get("semantic")
                                if thr is None:
                                    thr = (jr.config.get("thresholds", {}) or {}).get("semantic_threshold")
                                mets["semantic"] = await semantic_similarity(out_text, exp_variants, threshold=thr, cache=embed_cache)
                            except Exception as e:
                                mets["semantic"] = {"metric": "semantic", "pass": False, "error": str(e)}
                    # policy/consistency metrics don't require gold variants
                    try:
                        mets["consistency"] = consistency(out_text, rec.get("state") or {})
                    except Exception as e:
                        mets["consistency"] = {"metric": "consistency", "pass": False, "error": str(e)}
                    try:
                        exp_decision = (golden_outcome or {}).get("decision")
                        mets["adherence"] = adherence(out_text, golden_constraints, expected_decision=exp_decision)
                    except Exception as e:
                        mets["adherence"] = {"metric": "adherence", "pass": False, "error": str(e)}
                    try:
                        history_msgs = [m.get("content", "") for m in (rec.get("request", {}) or {}).get("messages", [])]
                        # Threshold from run config or settings
                        thr = (jr.config.get("thresholds", {}) or {}).get("hallucination_threshold")
                        mets["hallucination"] = hallucination(out_text, rec.get("state") or {}, history_msgs, threshold=thr)
                    except Exception as e:
                        mets["hallucination"] = {"metric": "hallucination", "pass": False, "error": str(e)}

                    # compute turn_pass ignoring metrics that were explicitly skipped
                    try:
                        considered = [v for v in mets.values() if isinstance(v, dict) and ("pass" in v) and not v.get("skipped")]
                        pass_vals = [bool(v.get("pass")) for v in considered]
                        turn_pass = all(pass_vals) if pass_vals else True
                    except Exception:
                        turn_pass = False
                    # Provider error always forces turn failure
                    if not resp_ok:
                        turn_pass = False
                    per_turn.append({
                        "turn_index": uidx,
                        "metrics": mets,
                        "turn_pass": turn_pass,
                        "user_prompt_snippet": _snippet(user_text),
                        "user_prompt_full": user_text,
                        "assistant_output_snippet": _snippet(out_text, 200),
                        "assistant_output_full": out_text,
                        "assistant_ok": resp_ok,
                        "assistant_error": resp_err,
                        "provider_meta": resp_meta,
                    })
                    last_state = rec.get("state") or last_state

                # conversation summary
                summary = aggregate_conversation(per_turn, last_state or {}, golden_outcome or {})
                # augment summary with counts and failed metrics
                try:
                    total_user_turns = len(per_turn)
                    failed_turns_count = sum(1 for t in per_turn if not t.get("turn_pass", True))
                    failed_metrics = sorted({
                        name for t in per_turn for name, m in (t.get("metrics") or {}).items()
                        if isinstance(m, dict) and m.get("pass") is False and not m.get("skipped")
                    })
                    summary = {
                        **(summary or {}),
                        "total_user_turns": total_user_turns,
                        "failed_turns_count": failed_turns_count,
                        "failed_metrics": failed_metrics,
                    }
                except Exception:
                    pass
                # add conversation description from metadata if present
                conv_description = None
                try:
                    conv_description = (conv.get("metadata") or {}).get("short_description")
                except Exception:
                    conv_description = None
                results["conversations"].append({
                    "conversation_id": cid,
                    **identity,
                    "conversation_description": conv_description,
                    "turns": per_turn,
                    "summary": summary,
                    "trace_dir": str(conv_dir),
                })

            # persist results
            try:
                results["input_tokens_total"] = int(total_input_tokens)
                results["output_tokens_total"] = int(total_output_tokens)
            except Exception:
                pass
            self._writer.write_results_json(jr.run_id, results)
            try:
                self._writer.write_results_csv(jr.run_id, results)
            except Exception:
                pass

            jr.state = "succeeded"
            jr.updated_at = _now_iso()
            jr.progress_pct = 100
            try:
                self._writer.write_job_status(jr.run_id, {
                    "job_id": jr.job_id,
                    "run_id": jr.run_id,
                    "state": jr.state,
                    "progress_pct": jr.progress_pct,
                    "total_conversations": jr.total_conversations,
                    "completed_conversations": jr.completed_conversations,
                    "total_turns": jr.total_turns,
                    "completed_turns": jr.completed_turns,
                    "input_tokens_total": jr.input_tokens_total,
                    "output_tokens_total": jr.output_tokens_total,
                    "error": None,
                    "boot_id": self.boot_id,
                })
            except Exception:
                pass
            return jr
        except asyncio.CancelledError:
            # Task cancelled externally (via control cancel). Reflect immediately.
            jr.state = "cancelled"
            jr.updated_at = _now_iso()
            try:
                self._writer.write_job_status(jr.run_id, {
                    "job_id": jr.job_id,
                    "run_id": jr.run_id,
                    "state": jr.state,
                    "progress_pct": jr.progress_pct,
                    "total_conversations": jr.total_conversations,
                    "completed_conversations": jr.completed_conversations,
                    "total_turns": jr.total_turns,
                    "completed_turns": jr.completed_turns,
                    "input_tokens_total": jr.input_tokens_total,
                    "output_tokens_total": jr.output_tokens_total,
                    "error": "cancelled by user",
                    "boot_id": self.boot_id,
                })
            except Exception:
                pass
            return jr
        except Exception as e:
            jr.state = "failed"
            jr.error = str(e)
            jr.updated_at = _now_iso()
            try:
                self._writer.write_job_status(jr.run_id, {
                    "job_id": jr.job_id,
                    "run_id": jr.run_id,
                    "state": jr.state,
                    "progress_pct": jr.progress_pct,
                    "total_conversations": jr.total_conversations,
                    "completed_conversations": jr.completed_conversations,
                    "total_turns": jr.total_turns,
                    "completed_turns": jr.completed_turns,
                    "input_tokens_total": jr.input_tokens_total,
                    "output_tokens_total": jr.output_tokens_total,
                    "error": jr.error,
                    "boot_id": self.boot_id,
                })
            except Exception:
                pass
            return jr

    def start(self, job_id: str) -> None:
        jr = self.jobs[job_id]
        if jr._task and not jr._task.done():
            return
        jr._task = asyncio.create_task(self.run_job(job_id))

    async def wait(self, job_id: str) -> JobRecord:
        jr = self.jobs[job_id]
        if jr._task:
            await jr._task
        return jr
