# User Guide — Plain and simple

Welcome! This guide explains what this tool does and how to use it, step by step, with simple language and examples.

---

## What and Why (in simple words)

What this tool does
- It tests Large Language Models (LLMs) using multi‑turn conversations (a back‑and‑forth chat).
- It checks if the model replies match “golden answers” (what we expect) and measures quality.
- It produces clear reports you can save (JSON/CSV/HTML).

Why multi‑turn evals are needed
- Real users don’t ask only one question. They ask, the bot answers, then they ask again, and so on.
- Many models look good at the first reply but make mistakes in later turns (forget facts, break rules, or change decisions).
- Multi‑turn tests catch these problems before you ship a new model or prompt.

Simple example story
- User: “I want a refund for order A1.”
- Bot: “Please share the order ID.”
- User: “Order ID is A1.”

The golden file says the bot must ask for the order ID. If the bot asks something else, this turn fails. Multi‑turn evals check each turn, not just the first.

---

## How it works (high‑level picture)

- Dataset: conversations you want to test (who said what at each step)
- Golden: what the correct or acceptable reply should be at each turn
- Model: the LLM you test (Ollama, Gemini, or OpenAI)
- Metrics: rules used to score replies (e.g., exact match, semantic similarity)

You run a test. The tool creates a run folder with results and a report.

---

## Requirements

- Windows 10/11 with PowerShell (local dev)
- Python 3.12 (recommended)
- Node.js 18+ and npm (only needed for the web UI)
- One LLM provider (pick any)
  - Ollama (local, default host http://localhost:11434, default model llama3.2:latest)
  - Google Gemini (needs GOOGLE_API_KEY)
  - OpenAI (needs OPENAI_API_KEY, default model gpt-5.1)

Tip: If new to LLMs, start with Ollama on your machine. It’s simple for demos.

---

## First‑time setup (step by step)

1) Create a Python virtual environment (from the repo root)
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1
   - pip install -r backend/requirements.txt
   Why: keeps dependencies clean and separate from your system Python.

2) Optional — Web UI
   - cd frontend
   - npm install
   Why: installs the UI so you can click around instead of using only the CLI.

3) Provider setup
   - Ollama: install from https://ollama.com, start it, then ollama pull llama3.2:latest
   - Gemini/OpenAI: get an API key; store it as an environment variable or in a local .env (for dev only)
   Why: the tool needs a model to talk to.

Common mistake to avoid
- If you see “No module named backend”, run commands from the repo root or use .\\scripts\\dev.ps1

---

## Quick Start A — Use the UI

1) Start both servers
   - .\\scripts\\dev.ps1
   What it does: starts the backend (http://localhost:8000) and the frontend (http://localhost:5173).

2) Open the app
   - Go to http://localhost:5173

3) Run a test
   - Upload or choose a dataset
   - Pick a model (e.g., gemini:gemini-2.5 or ollama:llama3.2:latest)
   - Click “Start Run”

4) See results
   - You’ll see progress, then a summary
   - Download JSON/CSV or open the HTML report

Screenshots (examples)
![App Home](docs/images/ui-home.png)
![Datasets Page](docs/images/datasets-page.png)
![Start Run (GIF)](docs/images/start-run.gif)
![Report View](docs/images/report-view.png)

Tip: If calls fail, check your API key or that Ollama is running.

---

## Frontend — start and use (step by step)

You can run the frontend (web app) with or without the helper script.

Option 1 — One command for both backend + frontend (easiest)
- .\\scripts\\dev.ps1
- Opens: backend at http://localhost:8000 and frontend at http://localhost:5173

Option 2 — Start backend and frontend in separate terminals
1) Terminal A (backend)
  - .\\.venv\\Scripts\\Activate.ps1
  - python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
2) Terminal B (frontend)
  - cd frontend
  - npm install            # first time only
  - npm run dev            # starts Vite dev server
  - Open the URL shown (usually http://localhost:5173)

What you will see in the UI
- Header: choose your Vertical (e.g., commerce)
- Pages:
  - Datasets: upload or select a dataset and golden
  - Runs: start a run, watch progress, open reports
  - Metrics: turn metrics on/off, change thresholds/weights
  - Settings: set OLLAMA_HOST, API keys, default models (saved to .env in dev)
  - Dataset Generator: auto‑create test datasets based on coverage rules

Screenshots
![Settings Page](docs/images/settings.png)
![Metrics Page](docs/images/metrics.png)

Simple example using the UI
1) Go to Settings and set your provider (e.g., GOOGLE_API_KEY for Gemini)
2) Go to Datasets and choose the demo dataset
3) Click Start Run (pick model gemini:gemini-2.5)
4) When done, click “Open Report” to see results

Production build (optional)
- cd frontend
- npm run build      # creates a production build in dist/
- npm run preview    # quick local preview of the built app
Note: For real deployments, serve dist/ with any static web server and point it to a running backend.

Frontend troubleshooting
- Node version: use Node 18+ (node -v)
- Port in use: if 5173 is busy, Vite will suggest another port; use that URL
- Backend not reachable: ensure backend is on http://localhost:8000
- API keys: set in Settings (dev) or as environment variables
- CORS/proxy: Vite dev server proxies to 8000; restart if you change ports

---

## Quick Start B — Use the CLI (no UI)

1) Create sample files
   - python -m backend.cli init
   Creates:
   - datasets/demo.dataset.json and datasets/demo.golden.json
   - configs/sample.run.json (a ready‑to‑run config)
   - runs/ (where outputs go)

2) Run an evaluation
   - python -m backend.cli run --file configs/sample.run.json
   Output:
   - A new runs/<run_id>/ folder containing results.json, results.csv, report.html

3) Generate bigger test sets (optional)
   - python -m backend.cli coverage --combined --v2 --save --overwrite --version 1.0.0
   What it does: automatically creates more datasets + goldens based on coverage rules.

Small example: sample.run.json (annotated)
{
  "datasets": ["demo"],            // which datasets to run
  "models": ["gemini:gemini-2.5"], // which model(s) to test
  "metrics": ["exact", "semantic"],// how to score replies
  "thresholds": { "semantic": 0.80 } // pass bar for semantic similarity
}

---

## Reading your results (what to look for)

Where files are written
- runs/<run_id>/results.json — detailed results
- runs/<run_id>/results.csv — spreadsheet‑friendly
- runs/<run_id>/report.html — pretty report for sharing

What “conversation_pass” means
- If a conversation’s turns meet the metric thresholds, it passes.
- You can count pass vs total to get a pass rate (used later in CI).

---

## REST API (when you want to script it)

Start the server
- python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

Start a run (curl)
curl -X POST http://127.0.0.1:8000/runs -H "Content-Type: application/json" -d "{\n  \"dataset_id\": \"demo\",\n  \"model_spec\": \"gemini:gemini-2.5\",\n  \"metrics\": [\"exact\", \"semantic\"],\n  \"thresholds\": { \"semantic\": 0.8 },\n  \"context\": { \"vertical\": \"commerce\" }\n}"

Download artifacts
- GET /runs/{run_id}/artifacts?type=json|csv|html

Tip: API is useful inside pipelines, but the CLI is simpler.

---

## Automate in CI/CD (make it part of release)

Goal
- Run tests automatically on every pull request or before release.
- Fail the pipeline if quality is below your bar (e.g., pass rate < 90%).

Two ways
- A) CLI (recommended): run the CLI, then check results.json
- B) API: start the server, call /runs, then download artifacts

Credentials in CI
- Gemini: set GOOGLE_API_KEY as a secret
- OpenAI: set OPENAI_API_KEY as a secret
- Ollama: best on a self‑hosted runner (needs the Ollama service and a local model)

Example run config (repeat)
{
  "datasets": ["demo"],
  "models": ["gemini:gemini-2.5"],
  "metrics": ["exact", "semantic"],
  "thresholds": { "semantic": 0.80 }
}

GitHub Actions (CLI) — with comments
name: LLM Eval
on: [push, pull_request]
jobs:
  run-evals:
    runs-on: ubuntu-latest              # runner OS
    env:
      GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps:
      - uses: actions/checkout@v4       # get code
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install deps               # set up CLI
        run: |
          python -m venv .venv
          . .venv/bin/activate
          pip install -r backend/requirements.txt
      - name: Run evaluation
        run: |
          . .venv/bin/activate
          python -m backend.cli init     # demo files
          python -m backend.cli run --file configs/sample.run.json
      - name: Enforce quality bar (>= 90% conversations pass)
        run: |
          python - << 'PY'
          import json, pathlib, sys
          runs = sorted(pathlib.Path('runs').iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
          if not runs:
              print('No runs found'); sys.exit(2)
          res = json.load(open(runs[0]/'results.json', 'r', encoding='utf-8'))
          convs = res.get('conversations') or []
          total = len(convs)
          passed = sum(1 for c in convs if (c.get('summary') or {}).get('conversation_pass'))
          rate = (passed/total) if total else 0.0
          print(f'Pass rate: {rate:.2%} ({passed}/{total})')
          sys.exit(0 if rate >= 0.90 else 1)
          PY
      - name: Upload artifacts (optional)
        uses: actions/upload-artifact@v4
        with:
          name: llm-eval-results
          path: runs/**

Notes
- For large suites, you can parallelize with coverage sharding: --shards N --shard-index i
- Keep keys in CI secrets (never hardcode API keys in the repo)

GitLab CI (CLI)
llm_eval:
  image: python:3.12
  script:
    - python -m venv .venv
    - . .venv/bin/activate
    - pip install -r backend/requirements.txt
    - python -m backend.cli init
    - python -m backend.cli run --file configs/sample.run.json
    - python - << 'PY'
      import json, pathlib, sys
      runs = sorted(pathlib.Path('runs').iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
      res = json.load(open(runs[0]/'results.json', 'r', encoding='utf-8'))
      convs = res.get('conversations') or []
      total = len(convs)
      passed = sum(1 for c in convs if (c.get('summary') or {}).get('conversation_pass'))
      rate = (passed/total) if total else 0.0
      print(f'Pass rate: {rate:.2%} ({passed}/{total})')
      sys.exit(0 if rate >= 0.90 else 1)
      PY
  artifacts:
    paths:
      - runs/

Using the REST API in CI (alternative)
1) Start server in the job (background)
   - python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
2) Create a run
   - curl -X POST http://127.0.0.1:8000/runs -H "Content-Type: application/json" -d '{"dataset_id":"demo","model_spec":"gemini:gemini-2.5"}'
3) Download results
   - curl http://127.0.0.1:8000/runs/<run_id>/artifacts?type=json > results.json
4) Apply the same pass‑rate check as above

---

## Tips and settings (simple)

- The Settings page (UI) writes a local .env (for development only). Restart the backend after changes.
- Useful env vars: OLLAMA_HOST, GOOGLE_API_KEY, OPENAI_API_KEY, SEMANTIC_THRESHOLD,
  OLLAMA_MODEL, GEMINI_MODEL, OPENAI_MODEL, EMBED_MODEL, INDUSTRY_VERTICAL
- Metrics (which ones to use, their weights) live in configs/metrics.json
- Deterministic runs: the system uses temperature=0.0 and seed=42 by default

Example .env (dev)
OLLAMA_HOST=http://localhost:11434
SEMANTIC_THRESHOLD=0.80
OLLAMA_MODEL=llama3.2:latest
GEMINI_MODEL=gemini-2.5
OPENAI_MODEL=gpt-5.1
EMBED_MODEL=nomic-embed-text

---

## Troubleshooting (quick fixes)

- “No module named backend”
  - Run from repo root or use .\\scripts\\dev.ps1
- Frontend can’t reach backend
  - Ensure backend is at http://localhost:8000, restart dev script
- Ollama errors or missing model
  - Start Ollama, run: ollama pull llama3.2:latest, check OLLAMA_HOST
- Gemini/OpenAI not working
  - Set API keys via env or CI secrets
- Semantic metric failing
  - Test embeddings via GET /embeddings/test

---

## Folder map (where things live)

- backend/ — API + evaluation engine
- frontend/ — Web UI (Vite + React)
- configs/ — Metrics and schemas
- datasets/<vertical>/ — Your datasets and goldens
- runs/<vertical>/ — Outputs (results.json, results.csv, report.html)
- scripts/ — Helpers (dev.ps1, smoke.ps1)
- docs/ — More docs (see docs/GOVERNANCE.md)

Glossary
- Dataset: conversations to test
- Golden: expected answers per turn
- Metric: how we score a reply
- Run: one evaluation job that writes results
- Vertical: a business area (commerce, banking, etc.) used to separate files

---

## About screenshots and GIFs

- The images above are placeholders. Put your own files in docs/images/ with the same names, or rename the links here to match your files.
- Suggested files:
  - docs/images/ui-home.png — app home
  - docs/images/datasets-page.png — datasets screen
  - docs/images/start-run.gif — starting a run and watching progress
  - docs/images/report-view.png — HTML report preview
  - docs/images/settings.png — settings page
  - docs/images/metrics.png — metrics page

---

## Need help?

- See CONTRIBUTING.md and docs/
- License: GPLv3 (see LICENSE)
- Ask in your team chat or open an issue
