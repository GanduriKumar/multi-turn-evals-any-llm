# Backend

FastAPI REST API providing dataset management, run orchestration, metrics, and reports.

- Stack: Python, FastAPI, Uvicorn
- Providers: Ollama, Gemini, OpenAI
- Artifacts: filesystem under `runs/`

Run locally
- Create a virtual env and install requirements: `pip install -r backend/requirements.txt`
- Start server (from repo root): `python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000`

Environment / Settings
- .env at repo root is auto‑loaded on startup
- OLLAMA_HOST (default http://localhost:11434)
- GOOGLE_API_KEY, OPENAI_API_KEY
- SEMANTIC_THRESHOLD (default 0.80)
- OLLAMA_MODEL, GEMINI_MODEL, OPENAI_MODEL (defaults for Runs dropdown)
- EMBED_MODEL (default `nomic-embed-text`) for semantic scoring via Ollama embeddings

Key endpoints
- Health/version: `GET /health`, `GET /version`
- Settings: `GET/POST /settings` (.env persistence), `GET /embeddings/test`
- Datasets: `GET /datasets`, `POST /datasets/upload`, `POST /datasets/save`, `GET /datasets/{id}`, `GET /goldens/{id}`
- Runs: `POST /runs`, `GET /runs`, `GET /runs/{job_id}/status`, `POST /runs/{job_id}/control`
- Artifacts: `GET /runs/{run_id}/results`, `GET /runs/{run_id}/artifacts?type=json|csv|html`, `POST /runs/{run_id}/rebuild`
- Compare: `GET /compare?runA=&runB=`

Job orchestration
- Pause/Resume/Abort controls with persisted `job.json`
- Stale detection via `boot_id`; UI can “Mark as cancelled” stale runs

Metrics
- exact, semantic, consistency, adherence, hallucination
- Semantic uses Ollama embeddings; ensure Ollama is running and `EMBED_MODEL` is available

See `UserGuide.md` for usage.
