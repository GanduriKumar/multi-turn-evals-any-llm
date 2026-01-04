# Multi-Turn Evaluation Workbench – Usage Guide

Welcome! This guide walks you, step by step, from zero to running your first multi‑turn evaluations. We’ll create or load datasets, choose metrics, pick an LLM provider (including local Ollama), start a run, watch progress, view charts, and download reports.

If you’re brand new to evals, don’t worry — the instructions use simple language and give you the why, not just the how.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Creating and Loading Datasets](#creating-and-loading-datasets)
3. [Defining and Configuring Metrics](#defining-and-configuring-metrics)
4. [Selecting LLM Providers and Models (incl. Ollama)](#selecting-llm-providers-and-models-incl-ollama)
5. [Starting Evaluation Runs](#starting-evaluation-runs)
6. [Monitoring Run Progress](#monitoring-run-progress)
7. [Viewing Metrics and Charts](#viewing-metrics-and-charts)
8. [Downloading Reports and Artifacts](#downloading-reports-and-artifacts)
9. [Troubleshooting](#troubleshooting)
10. [Provider Quick Reference](#provider-quick-reference)

---

## Quick Start

Think of the system as two parts: a backend API (Python) and a frontend app (Vite/React). You’ll run both locally.

### Prerequisites

- Python 3.9 or newer (for backend)
- Node.js 16 or newer (for frontend)
- On Windows PowerShell, ensure script execution allows venv activation

### Start the Full Stack (first time and daily use)

From the repository root:

```bash
# Terminal 1: Backend API
cd backend
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -e .
python -m eval_server

# Backend runs at http://localhost:8000
# API docs available at http://localhost:8000/docs
```

```bash
# Terminal 2: Frontend app
cd frontend
npm ci
npm run dev

# Frontend runs at http://localhost:5173
```

Open http://localhost:5173 in your browser. You’ll land on the Datasets page.

Tip: If anything fails to start, see the [Troubleshooting](#troubleshooting) section at the end.

---

## Creating and Loading Datasets

You evaluate models using “datasets,” which are small bundles describing a conversation and what a correct response looks like. Each dataset has two files:

1. Conversation file (JSON): A multi-turn dialogue (user and assistant turns)
2. Golden file (YAML/JSON): The expected or acceptable answers for specific turns, plus optional scoring rules

### Minimal Conversation Example

File: `configs/datasets/examples/conversation_001.json`

```json
{
   "conversation_id": "conv-001",
   "metadata": {
      "tags": ["commerce", "support"],
      "difficulty": "medium",
      "version": "1.0.0"
   },
   "turns": [
      { "turn_id": "t1", "role": "user", "content": "What's your return policy?" },
      { "turn_id": "t2", "role": "assistant", "content": "We offer 30-day returns." }
   ]
}
```

### Minimal Golden Example

File: `configs/datasets/examples/conversation_001.golden.yaml`

```yaml
version: "1.0.0"
expectations:
   - conversation_id: conv-001
      turn_id: t2
      expected:
         - variant: "We accept returns within 30 days of purchase."
            weight: 1.0
      thresholds:
         turn_pass: 0.75
```

### Recommended Folder Layout

```
configs/
   datasets/
      examples/
         conversation_001.json
         conversation_001.golden.yaml
      my_domain/
         my_conv.json
         my_conv.golden.yaml
```

### Create Your First Dataset (step by step)

1. Make a folder: `configs/datasets/my_domain`
2. Create both files (conversation JSON and golden YAML) as shown above
3. Keep the same `conversation_id` in both files
4. Add simple metadata like tags and difficulty — these help you filter in the UI

### Load Datasets in the UI

1. Go to the Datasets page (navbar → Datasets)
2. If you don’t see your dataset, click “Refresh”
3. Filter using the Domain (tags) or Difficulty dropdowns
4. Select one or more datasets:
    - Use the checkbox on each card, then click “Load Selected”
    - Or click “Load Dataset” on a single card to jump directly to Run Setup

If nothing shows up, verify the file paths and formats. See Troubleshooting.

---

## Defining and Configuring Metrics

Metrics grade each turn and aggregate into conversation-level scores. In the UI, you’ll choose “metric bundles.” These are preset metric groups suitable for common evaluation needs.

In this workbench, the Run Setup page currently includes these bundles:

- core: correctness and basic quality
- safety: harmful content checks
- reasoning: logical steps and follow-through
- hallucination: incorrect/invented facts

Pick the bundles that match your use case. You can combine multiple bundles.

### Where Do Thresholds Come From?

Thresholds (like “turn must score ≥ 0.75”) usually come from your golden files or optional run-level config. If present, they’ll be applied automatically. If not present, everything still runs; thresholds just won’t be shown in charts.

Example (golden file snippet):

```yaml
expectations:
   - conversation_id: conv-001
      turn_id: t2
      thresholds:
         turn_pass: 0.75
```

Optional run-level thresholds (advanced, YAML):

```yaml
thresholds:
   turn_pass: 0.75
   conversation_pass: 0.80
```

You don’t have to set thresholds to get value — they just make pass/fail clearer.

---

## Selecting LLM Providers and Models (incl. Ollama)

You can evaluate with cloud models (OpenAI, Google, Azure OpenAI, AWS Bedrock) or local models via Ollama.

### Step 1 — Set Provider in Run Setup

In Run Setup, choose a Provider from the dropdown. The Model field will auto-fill a sensible default (you can edit it).

### Step 2 — Configure Credentials (when needed)

Most cloud providers need API keys in your backend environment. Example `.env` values:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Google
GOOGLE_API_KEY=...

# Azure OpenAI
AZURE_OPENAI_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# AWS (Bedrock)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

After editing `.env`, restart the backend:

```bash
python -m eval_server
```

### Ollama (Local models, offline‑friendly)

Ollama lets you run LLMs locally without cloud keys. This is great for privacy, cost, or offline testing.

1) Install Ollama

- Download from https://ollama.com and follow the OS-specific installer

2) Start the Ollama server

```bash
ollama serve
# Default API: http://127.0.0.1:11434
```

3) Pull a model

```bash
ollama pull llama3.2
# or other options: mistral, qwen2.5, phi3, etc.
```

4) In Run Setup

- Set Provider: “Ollama”
- Set Model: the same tag you pulled (e.g., `llama3.2`)

Advanced (optional): If you run Ollama elsewhere, set the endpoint for the backend process:

```env
OLLAMA_HOST=http://localhost:11434
```

Note: Model parameters (temperature, max tokens) are typically supported; defaults usually work fine. Start simple, then adjust.

---

## Starting Evaluation Runs

Once your datasets and model are set, you’re a click away.

1. Go to Run Setup
2. Choose one or more datasets (left side)
3. Choose a provider and model (middle)
4. Pick metric bundles (middle)
5. (Optional) Adjust truncation and max workers (right side)
6. Click “Start Run” (right side Summary card)

You’ll see a green message with the new Run ID and a link to open the Dashboard.

Tips

- For your first test, try the Dummy provider — it’s fast and doesn’t require keys
- Keep max workers low (1–2) while you’re validating
- Use a small dataset first (“hello world”) to confirm everything works

---

## Monitoring Run Progress

The Dashboard shows current status, a progress bar, and conversation cards.

What you’ll see

- Overall status (running, completed, failed, etc.)
- % progress across conversations (nice progress bar)
- A list of conversations with per-item progress
- Buttons for “Cancel Run,” “Open Viewer,” and Artifact downloads

Click any conversation to open its detail page. There you can:

- See each turn’s prompt, the model response, and the expected (golden) text
- View a simple color diff for mismatches
- See per-turn metric scores
- Optionally enter human feedback (rating, notes) and save it back to the run

---

## Viewing Metrics and Charts

The Metrics page helps you spot patterns quickly.

1. Open Metrics (from Dashboard or navbar)
2. Choose a metric (e.g., correctness, safety)
3. Optionally filter to a single conversation
4. Review the bar chart — each bar is a turn’s score
5. If thresholds exist, you’ll see a line on the chart (e.g., 0.75)
6. Use the table below for precise values; it updates with filters

Exporting

- Export CSV of the current table view
- Export PNG of the chart as an image
- From the Artifacts area (Dashboard), you can also download comprehensive reports

---

## Downloading Reports and Artifacts

From the Dashboard “Artifacts” section, download:

- Results (JSON): raw, detailed per-turn and per-conversation scores
- Summary (JSON): quick overview of pass/fail and averages
- CSV: tabular data for spreadsheets
- HTML and Markdown reports: human-readable summaries

You can also combine some into a ZIP (e.g., results + CSV) for easy sharing.

---

## Troubleshooting

Here are the most common first‑day issues and how to fix them.

Backend won’t start

- Ensure the venv is activated and dependencies installed: `pip install -e .`
- Run: `python -m eval_server`
- Check that no other process is using port 8000

Frontend can’t connect to backend

- Make sure backend is running: visit http://localhost:8000/docs
- Restart the frontend dev server (`npm run dev`)

Datasets don’t appear in the UI

- Click “Refresh” on the Datasets page
- Confirm your files exist and are valid (JSON/YAML)
- Ensure conversation_id matches across conversation and golden files
- Add simple metadata (tags/difficulty) if you plan to filter on them

Provider errors (OpenAI, Google, etc.)

- Double‑check API keys in backend `.env`
- Restart backend after editing `.env`

Ollama isn’t working

- Ensure the server is running: `ollama serve`
- Pull the model: `ollama pull llama3.2`
- Test the API: open http://127.0.0.1:11434/api/tags and confirm your model is listed
- In Run Setup, set Provider = Ollama and Model = your pulled tag

No metric lines on chart

- You probably don’t have thresholds in golden or run config — that’s OK. Add them if you want pass/fail visuals.

---

## Provider Quick Reference

This is a handy checklist when switching providers.

OpenAI

- Env: `OPENAI_API_KEY`
- Models: `gpt-4o`, `gpt-4-turbo`, etc.
- Tip: Start with `gpt-4o` for balanced quality/speed

Google (Gemini)

- Env: `GOOGLE_API_KEY`
- Models: `gemini-1.5`, `gemini-2.0`

Azure OpenAI

- Env: `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`
- Models: your deployment names (map to GPT families)

AWS Bedrock

- Env: `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Models: provider‑dependent (e.g., `claude-3`, `mixtral-8x7b` via Bedrock)

Ollama (Local)

- Install Ollama, run `ollama serve`
- Pull models (e.g., `ollama pull llama3.2`)
- Provider: `ollama` in Run Setup
- Model: the tag you pulled (e.g., `llama3.2`)
- Optional env: `OLLAMA_HOST` (default http://127.0.0.1:11434)

---

You’re ready! Start with a tiny dataset and the Dummy or Ollama provider to get comfortable. Then add real models, real keys, and larger datasets. If you get stuck, revisit Troubleshooting or open the backend API docs at http://localhost:8000/docs.
