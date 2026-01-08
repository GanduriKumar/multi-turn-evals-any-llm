# Multi-Turn LLM Evaluation System (MVP)

Monorepo for a multi‑turn LLM evaluation platform.

- Backend: FastAPI REST API (under `backend/`)
- Frontend: React + Vite + TypeScript + Tailwind (under `frontend/`)
- Shared: `configs/`, `datasets/`, `scripts/`, `docs/`

Key capabilities
- Upload JSON datasets and goldens (multiple acceptable variants per turn)
- Select LLM provider/model: Ollama, Gemini, OpenAI
	- Default models configurable in Settings: `OLLAMA_MODEL`, `GEMINI_MODEL`, `OPENAI_MODEL`
- Configure/run evaluations with conversation state + last‑N turns context
- Job controls: Pause, Resume, Abort; stale run detection and “Mark as cancelled” for stale rows
- Metrics: exact, semantic (via local embeddings), consistency, adherence, hallucination
	- Semantic uses Ollama embeddings; set `EMBED_MODEL` (e.g., `nomic-embed-text`) and `OLLAMA_HOST`
	- Quick check endpoint: `GET /embeddings/test`
- Reports: HTML/CSV/JSON with conversation identity (slug/title/metadata), per‑turn snippets, rollups
- Compare two runs: `GET /compare?runA=&runB=`
- Persistence: filesystem (`runs/`)

What’s new recently
- OpenAI provider support (set `OPENAI_API_KEY`; pick model in Runs)
- Settings page now manages API keys, hosts, semantic threshold, default models, and `EMBED_MODEL`
- Metrics config persisted to `configs/metrics.json` and respected across UI and backend
- Rebuild endpoint to enrich old runs: `POST /runs/{run_id}/rebuild`
- Dataset schema additions: conversation `title` and `metadata.short_description`

Getting started
- See `UserGuide.md` for end‑to‑end setup and usage.

License
- Licensed under GNU GPL v3. See `LICENSE`.
