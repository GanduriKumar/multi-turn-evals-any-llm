# Multi‑Turn LLM Evaluation – in simple words

This project helps you test AI chat models (LLMs) using short, realistic conversations. You give it a set of example chats and what the correct answers should be. It runs the model, scores the replies, and shows an easy report.

What you can do
- Upload or pick a dataset of conversations to test
- Choose a model (Ollama, Google Gemini, or OpenAI)
- Run the evaluation and get a clear pass/fail report
- Compare two runs to see what got better or worse

What you need (basics)
- Windows 10/11
- Python 3.12
- Optional: Node.js (only if you want the web app UI)
- At least one model provider:
  - Ollama (local) or
  - Google Gemini (GOOGLE_API_KEY) or
  - OpenAI (OPENAI_API_KEY)

Two ways to use it
1) With the web app (recommended for beginners)
	- Start: .\scripts\dev.ps1
	- Open: http://localhost:5173
	- Click “Dataset Generator” to create or “Datasets Viewer” to pick one, then start a run. Open the report when it finishes.

2) With the CLI (no UI)
	- python -m backend.cli init
	- python -m backend.cli run --file configs/sample.run.json

Where things live
- backend/ — the API and evaluation engine
- frontend/ — the web app
- datasets/<vertical>/ — your test data
- runs/<vertical>/ — results and reports
- configs/ — settings used by the app

Helpful links
- User Guide: UserGuide.md (step‑by‑step)
- Dataset Generator Guide: docs/DatasetGeneratorGuide.md (optional)

License
- GPL v3 (see LICENSE)
