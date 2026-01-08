# Frontend

React + Vite + TypeScript + Tailwind UI for datasets, runs, reports, settings, metrics, and golden tools.

Main pages
- Datasets: upload/validate datasets and goldens
- Runs: start runs, live status, Pause/Resume/Abort, recent runs, stale handling
- Reports: view/download HTML/CSV/JSON with identity and per‑turn snippets
- Settings: configure OLLAMA_HOST, GOOGLE_API_KEY, OPENAI_API_KEY, semantic threshold, default models, EMBED_MODEL
- Metrics: toggle metrics and thresholds (persisted to `configs/metrics.json`)
- Golden Editor / Generator

Dev
- `npm install` then `npm run dev`
- Backend must run at http://localhost:8000
