# Backend (plain explanation)

This folder is the server. It receives requests from the web app or CLI, runs the model, scores answers, and writes the report.

Run it locally
1) Create a virtual env and install:
	- pip install -r backend/requirements.txt
2) Start the server (from the repo root):
	- python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

What it needs
- A model provider (Ollama, Google Gemini, or OpenAI)
- Optional .env file in the repo root with keys like GOOGLE_API_KEY or OPENAI_API_KEY

Useful endpoints
- GET /health — quick “am I alive?”
- GET /version — shows available defaults
- GET /settings and POST /settings — read/save local dev settings
- GET /embeddings/test — quick semantic check
- GET /datasets — list datasets; POST /datasets/upload — upload your own
- POST /runs — start a test run; then GET results/artifacts
- GET /reports/compare — compare two runs

Where files go
- runs/<vertical>/ — results.json, results.csv, report.html
- datasets/<vertical>/ — your datasets and goldens

Tip
- Restart the server after changing .env in development.

See UserGuide.md for the step‑by‑step flow.
