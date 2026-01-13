# Frontend

React + Vite + TypeScript + Tailwind UI for datasets, runs, reports, settings, metrics, and golden tools.

Main pages
- Dataset Generator: create datasets from server strategy (first nav item)
- Datasets Viewer: upload/validate datasets and goldens
- Runs: start runs, live status, Pause/Resume/Abort, recent runs, stale handling
- Reports: view/download HTML/CSV/JSON with identity and per‑turn snippets
 - Reports: view/download HTML/CSV/JSON with identity, per‑turn snippets, and token usage totals
- Settings: configure OLLAMA_HOST, GOOGLE_API_KEY, OPENAI_API_KEY, semantic threshold, default models, EMBED_MODEL
- Metrics: toggle metrics and thresholds (persisted to `configs/metrics.json`)
- Golden Editor / Generator
 - Coverage generation uses the v2 pipeline exclusively (backend `coverage_builder_v2.py`).

Dev
- `npm install` then `npm run dev`
- Backend must run at http://localhost:8000
- Vite proxy forwards API calls; ensure backend host/port matches `frontend/vite.config.ts`.
 - PDF button is hidden by default; backend supports PDF if system dependencies are installed.
