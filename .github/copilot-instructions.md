# Copilot instructions for this repo

Purpose
- Multi‑turn LLM evaluation platform. FastAPI backend evaluates datasets with goldens using exact/semantic/consistency/adherence/hallucination metrics; writes artifacts and reports; React/Vite frontend manages runs and settings.

Architecture
- Backend (`backend/`): FastAPI app (`backend/app.py`) routes for datasets, runs, reports, settings. `Orchestrator` (`backend/orchestrator.py`) executes user turns with providers and writes artifacts via `RunArtifactWriter`.
- Providers (`backend/providers/`): registry loads `.env` (`registry.py`) and resolves `provider:model` (e.g., `openai:gpt-5.1`). Implementations return `ProviderResponse`.
- Artifacts (`backend/artifacts.py`): stable short paths to avoid Windows MAX_PATH. Results in `results.json`/`results.csv`, per‑turn `conversations/c-<hash>/turn_*.json`, job status `job.json`.
- Data (`datasets/<vertical>/*.dataset.json|*.golden.json`): `DatasetRepository` loads dataset and golden. Goldens may specify `entry.final_outcome` or top‑level `final_outcome`.
- Metrics (`backend/metrics*.py`, `conversation_scoring.py`): exact, semantic (Ollama embeddings), plus extra metrics; aggregated per conversation.
- Coverage (`backend/coverage_builder_v2.py`, `convgen_v2.py`, `array_builder_v2.py`, `coverage_config.py`, `risk_sampler.py`, `commerce_taxonomy.py`): v2 pipeline only.
- Frontend (`frontend/`): React + Vite + Tailwind UI; Vite proxy targets backend.

Core workflows
- Tests: use VS Code tasks (e.g., “pytest - prompt13”). `backend/tests/conftest.py` adds `backend/` to `sys.path`.
- Backend dev: `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000` (task: “Start backend API (uvicorn)”). `.env` at repo root is auto‑loaded.
- Frontend dev: in `frontend/` run `npm install` then `npm run dev` (backend at http://localhost:8000). See `frontend/vite.config.ts`.
- CLI: `python -m backend.cli init` to scaffold; `python -m backend.cli run --file configs/sample.run.json [--no-semantic]`; coverage: `python -m backend.cli coverage --combined|--split [--save --overwrite]` (v2 only).

Conventions and patterns
- Model spec: `provider:model` (parsed by `Orchestrator.parse_model_spec`).
- Determinism: temperature 0.0, seed 42; prompt context uses last 5 turns; budgets ~1800 input, 400 completion (see `README.md`).
- Vertical storage: datasets/runs under `datasets/<vertical>/` and `runs/<vertical>/`; default `INDUSTRY_VERTICAL` in `.env`. Startup migrates legacy root folders.
- Artifact layout: always use `RunFolderLayout`; conversation dirs are short hashed (`c-<sha>`). Avoid long IDs in paths.
- Results shape: `results.json` includes run identity, conversation identity (slug/title/domain/behavior/etc), per‑turn metrics/snippets, conversation summary, and `input_tokens_total`/`output_tokens_total`. CSV mirrors this and adds rollups (e.g., risk tier).
- Metrics selection: run config `metrics` may include `exact`, `semantic`, `consistency`, `adherence`, `hallucination`. Thresholds from `thresholds` (e.g., `semantic`, `hallucination_threshold`) or settings.
- Embeddings: semantic uses Ollama with `EMBED_MODEL`; quick check `GET /embeddings/test`.
- Settings: `POST /settings` persists to `.env`; UI persists metric toggles to `configs/metrics.json` read by backend.

Integration points
- Keys/hosts via `.env`: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OLLAMA_HOST`, `EMBED_MODEL`, default models `OLLAMA_MODEL`/`GEMINI_MODEL`/`OPENAI_MODEL`, thresholds `SEMANTIC_THRESHOLD`/`HALLUCINATION_THRESHOLD`.
- Reports: `backend/reporter.py` renders `backend/templates/report.html.j2`.

Pitfalls and gotchas
- Windows paths: rely on `RunFolderLayout` and hashed conversation dirs; do not embed long conversation IDs in disk paths.
- Golden mapping: assistant index may map as `2*user_idx+1`, `user_idx+1`, or `user_idx`. Code tries candidates.
- Semantic scoring can fail if embeddings not available; handle and proceed (metrics entry contains `error`).

Key references
- Orchestrator flow: `backend/orchestrator.py` → `TurnRunner` → per‑turn files → aggregate to `results.json`/`results.csv` and update `job.json`.
- Providers/config: `backend/providers/registry.py`.
- Frontend dev: `frontend/README.md`, `frontend/vite.config.ts`.
- Additional docs: root `README.md`, `backend/README.md`, `configs/README.md`, `UserGuide.md`.
