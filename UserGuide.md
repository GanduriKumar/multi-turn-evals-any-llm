# User Guide — Multi‑Turn LLM Evaluation System (MVP)

Welcome! This guide will help you install, run, and use the Multi‑Turn LLM Evaluation System. It uses simple language and walks you through the key features step‑by‑step.

If you’re new to the project, start here.

---

## What this app does

This app lets you evaluate Large Language Models (LLMs) on multi‑turn conversations. You can:
- Upload datasets (conversations) and matching golden files (expected answers).
- Choose a model (local Ollama or Google Gemini).
- Run evaluations and watch progress live.
- View a report with scores per turn and per conversation.
- Export results (JSON/CSV/HTML) and add human feedback.

Everything is stored on your local filesystem. No database needed. The app is a monorepo with a Python backend (FastAPI) and a React frontend (Vite + TypeScript).

---

## Requirements

- Windows 10/11 (PowerShell)
- Python 3.12 (recommended) with virtual environment
- Node.js 18+ and npm
- Optional for LLMs (choose any):
  - Ollama running locally (default host http://localhost:11434) with model `OLLLAMA_MODEL` (default `llama3.2:latest`)
  - Google Gemini API Key (for model `GEMINI_MODEL`, default `gemini-2.5`)
  - OpenAI API Key (for model `OPENAI_MODEL`, default `gpt-5.1`)

Example: Install Ollama and pull a model
- Download Ollama from https://ollama.com and start it.
- In a terminal: `ollama pull llama3.2:latest`

---

## First‑time setup

1) Clone or open the repo folder in VS Code.

2) Python environment (from repo root):
- Create and activate a virtual environment, then install backend dependencies
  - `python -m venv .venv`
  - `.\\.venv\\Scripts\\Activate.ps1`
  - `pip install -r backend/requirements.txt`

3) Node dependencies (frontend):
- `cd frontend`
- `npm install`

4) Optional providers
- Ollama: ensure it’s running and pull the model:
  - `ollama pull llama3.2:latest`
- Gemini: obtain an API key from Google AI Studio.

---

## Start the app (dev)

Use the provided script to run both backend and frontend together:
- Run: `.\\scripts\\dev.ps1`

This will:
- Start backend API at http://localhost:8000
- Start frontend at http://localhost:5173

If you prefer manually:
- Backend (from repo root):
  - Ensure you run from the repo root so imports work
  - `python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000`
- Frontend:
  - `cd frontend`
  - `npm run dev`

Tip (Windows): avoid --reload when starting uvicorn for this project.

---

## Quick smoke test

- Run: `.\\scripts\\smoke.ps1`
- It will check /health and /datasets on the backend.

---

## Using the UI

Open http://localhost:5173 in your browser.

### 1. Datasets page
- Upload a dataset (.dataset.json) and optional golden (.golden.json).
- Files are validated on the server and saved under the repo `datasets/` folder.
- The table shows: dataset id, version, domain, difficulty, number of conversations, and validation status.

Tips and examples:
- File naming is handled by the server using the dataset_id. If your dataset JSON contains `"dataset_id": "banking_easy"`, the saved files will be:
  - `datasets/banking_easy.dataset.json`
  - `datasets/banking_easy.golden.json`
- You can also place files manually in `datasets/`. Use the same naming convention.

Example dataset entry (short):
```json
{
  "dataset_id": "commerce_easy",
  "version": "1.0.0",
  "metadata": { "domain": "commerce", "difficulty": "easy" },
  "conversations": [
    {
      "conversation_id": "conv42",
      "turns": [
        { "role": "user", "text": "I need a refund for order 123" },
        { "role": "assistant", "text": "Please share your order ID." }
      ]
    }
  ]
}
```

Example golden entry (matching assistant turns by index):
```json
{
  "dataset_id": "commerce_easy",
  "version": "1.0.0",
  "entries": [
    {
      "conversation_id": "conv42",
      "turns": [
        { "turn_index": 1, "expected": { "variants": ["Please share your order ID.", "Provide order id, please."] } }
      ],
      "final_outcome": { "decision": "ALLOW", "next_action": "issue_refund" }
    }
  ]
}
```

### 2. Runs page
- Choose a dataset and a model.
- Choose metrics (exact, semantic, consistency, adherence, hallucination) and set the semantic threshold.
- Click Start Run. You’ll see live progress and completion status.

Notes and examples:
- Semantic scoring uses local embeddings (via Ollama). Configure in Settings:
  - OLLAMA_HOST (e.g., http://localhost:11434)
  - EMBED_MODEL (e.g., `nomic-embed-text`)
  - Test endpoint: `GET /embeddings/test`

Example run configuration sent by the UI:
```json
{
  "dataset_id": "demo",
  "model_spec": "ollama:llama3.2:latest",
  "metrics": ["exact_match", "semantic_similarity", "consistency", "adherence", "hallucination"],
  "thresholds": { "semantic_threshold": 0.8 }
}
```

What to expect:
- The job appears with a job id like `job-0001`.
- Progress shows completed conversations out of total.
- When finished, a run id is created deterministically (dataset + version + model + config hash).

### 3. Reports page
- Select a run to view a summary.
- Download results as JSON/CSV, or open the HTML report.
– Submit human feedback (rating, notes). Optionally override conversation‑level pass/fail or a specific turn’s pass/fail. Feedback is stored at `runs/<run_id>/feedback.json` and does not change auto‑scores.

Examples:
- Download JSON: opens `runs/<run_id>/results.json`.
- Download CSV: opens `runs/<run_id>/results.csv`.
- Open Report: generates and opens `runs/<run_id>/report.html` (rendered from results).

### 4. Settings page
- Configure providers and thresholds:
  - OLLAMA_HOST
  - GOOGLE_API_KEY, OPENAI_API_KEY
  - Semantic threshold
  - Default models: OLLAMA_MODEL, GEMINI_MODEL, OPENAI_MODEL
  - EMBED_MODEL for semantic similarity
– These settings are saved to a local `.env` at the repo root and loaded at startup. Restart backend after changes.

Example `.env` file:
```
OLLAMA_HOST=http://localhost:11434
GOOGLE_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key
SEMANTIC_THRESHOLD=0.80
OLLAMA_MODEL=llama3.2:latest
GEMINI_MODEL=gemini-2.5
OPENAI_MODEL=gpt-5.1
EMBED_MODEL=nomic-embed-text
```

### 5. Metrics page
- Enable/disable metrics and adjust weights/thresholds.
– Saved to `configs/metrics.json`.

Example `configs/metrics.json` structure:
```json
{
  "metrics": [
    { "name": "exact_match", "enabled": true, "weight": 1.0 },
    { "name": "semantic_similarity", "enabled": true, "weight": 1.0, "threshold": 0.8 },
    { "name": "consistency", "enabled": true, "weight": 1.0 },
    { "name": "adherence", "enabled": true, "weight": 1.0 },
    { "name": "hallucination", "enabled": true, "weight": 1.0 }
  ]
}
```

### 6. Golden Generator page
- Generate a sample dataset and golden for Commerce or Banking.
- Choose difficulty and outcome (ALLOW/DENY/PARTIAL).
– Download the generated JSON files. You can upload them on the Datasets page or edit them in the Golden Editor.

Example generation choices:
- Domain: commerce
- Difficulty: medium
- Outcome: PARTIAL
This generates a 6‑turn conversation with a partial refund and matching golden entries for key assistant turns.

### 7. Golden Editor page
- Load an existing dataset and its golden.
- Edit JSON directly and save. You can choose to overwrite and bump the patch version.
– Server validates JSON against schemas before saving. If validation fails, the Save call returns an error list with precise locations.

Example save payload (what the UI sends):
```json
{
  "dataset": { /* full dataset object */ },
  "golden": { /* full golden object (optional) */ },
  "overwrite": true,
  "bump_version": true
}
```

---

## File layout (important folders)

- backend/ — FastAPI app and evaluation engine
- frontend/ — React + Vite UI
- configs/schemas/ — JSON Schemas (dataset, golden, run_config)
- datasets/ — Your datasets and golden files
- runs/ — Generated run artifacts (results.json, results.csv, report.html, per‑turn JSON)
- scripts/ — Helper scripts (dev.ps1, smoke.ps1)
- docs/ — Additional documentation
  - docs/GOVERNANCE.md — Coverage governance & versioning guide (Prompt 13)

---

## REST API overview (for advanced users)

Common endpoints:
- GET /health — basic status
- GET /version — version, provider flags, and default models
- GET /datasets — list datasets
- POST /datasets/upload — upload dataset and optional golden
- GET /datasets/{dataset_id} — get dataset JSON
- GET /goldens/{dataset_id} — get golden JSON
- POST /datasets/save — validate and write dataset/golden
- POST /validate — validate JSON against schemas (dataset/golden/run_config)
- POST /runs — start a run
- GET /runs — list runs
- GET /runs/{job_id}/status — live job status
- GET /runs/{run_id}/results — get results.json
- GET /runs/{run_id}/artifacts?type=json|csv|html — download/export
- POST /runs/{run_id}/feedback — append human feedback
- GET /compare?runA=&runB= — compare run summaries
- GET /settings — read settings (providers, models, embedding)
- POST /settings — update .env (dev only)
- GET /embeddings/test — verify embedding endpoint/model
- GET/POST /metrics-config — read/write metrics configuration
- GET /coverage/taxonomy — list domains and behaviors
- GET /coverage/manifest — preview counts per domain×behavior pair (query params: domains, behaviors)
- POST /coverage/generate — generate datasets/goldens (combined or split) with options to save
- GET /coverage/report.csv?type=summary|heatmap — download CSV reports
- POST /coverage/per-turn.csv — generate a per-turn CSV for a dataset/golden payload

---

## Troubleshooting

- ModuleNotFoundError: No module named 'backend'
  - Start backend from repo root, or run .\scripts\dev.ps1 which sets PYTHONPATH.
  - Avoid using --reload on Windows; it can break imports.

- Frontend cannot reach backend
  - Ensure backend is running on http://localhost:8000.
  - Vite dev server proxies API calls automatically, including /coverage endpoints.
  - If proxy fails, check `frontend/vite.config.ts` for the route and restart the dev server.

- Ollama errors or model not found
  - Install and start Ollama.
  - ollama pull llama3.2:latest
  - Check OLLAMA_HOST in Settings page.
  - Try a quick curl: `curl http://localhost:11434/api/tags` should list models.

- Gemini not enabled
  - Set GOOGLE_API_KEY via Settings, then restart backend.

- Semantic metric failures
  - Ensure Ollama is running, `EMBED_MODEL` is pulled, and `OLLAMA_HOST` is correct.
  - Use `GET /embeddings/test` to validate before running.

Logs and artifacts:
- Backend logs in the terminal where uvicorn runs.
- Per‑turn artifacts under `runs/<run_id>/conversations/<conversation_id>/turn_XXX.json`.
- Aggregated results in `runs/<run_id>/results.json` and `.csv`.
 - Coverage CSVs via the Coverage Generator page: summary, heatmap, and per-turn exports.

---

## Data formats

- Dataset (.dataset.json)
  - Fields: dataset_id, version, metadata { domain, difficulty, tags? }, conversations [ { conversation_id, turns: [{ role, text }] } ]
- Golden (.golden.json)
  - Fields: dataset_id, version, entries [ { conversation_id, turns: [{ turn_index, expected { variants: string[] } }], final_outcome, constraints? } ]
- Run results (results.json)
  - Summary per conversation with per‑turn metrics and overall pass/fail.

Example files are in `datasets/demo.dataset.json` and `datasets/demo.golden.json`.

Run results (snippet):
```json
{
  "run_id": "demo-1.0.0-ollama-llama3.2-latest-abcdef12",
  "dataset_id": "demo",
  "model_spec": "ollama:llama3.2:latest",
  "conversations": [
    {
      "conversation_id": "conv1",
      "turns": [
        {
          "turn_index": 2,
          "metrics": {
            "exact": { "metric": "exact", "pass": false },
            "semantic": { "metric": "semantic", "pass": true, "score_max": 0.86 },
            "adherence": { "metric": "adherence", "pass": true },
            "hallucination": { "metric": "hallucination", "pass": true },
            "consistency": { "metric": "consistency", "pass": true }
          }
        }
      ],
      "summary": { "conversation_pass": true, "weighted_pass_rate": 0.92 }
    }
  ]
}
```

---

## Contributing

- See CONTRIBUTING.md for coding standards and tests. For governance/versioning of coverage rules, see `docs/GOVERNANCE.md`.
- License: GPLv3 (see LICENSE).

---

## Where to get help

- Open an issue in the repository.
- Or ask in your team chat with a link to this guide.
