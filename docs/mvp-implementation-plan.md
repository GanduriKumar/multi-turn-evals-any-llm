# MVP Implementation Plan — Multi‑Turn LLM Evaluation System

This plan follows the Interview‑First approach: clear, atomic prompts you can execute sequentially. Each prompt is self‑contained, test‑driven, and production‑oriented.

Personas: QA evaluator, Product analyst
Key workflows: upload dataset (JSON), select LLM, configure/run evaluation, view per‑turn report, export CSV, submit human feedback, compare runs
Providers/models: Ollama (llama3.2:2b), Gemini (gemini-2.5; auto‑disable if GOOGLE_API_KEY missing)
Embeddings: Ollama nomic-embed-text (cosine threshold 0.80) for semantic similarity
Datasets/goldens: JSON only, multiple acceptable variants per turn, pure text, outcome schema {decision, refund_amount?, reason_code?, next_action?, policy_flags?}
Context policy: state object + last 4 turns
Metrics: per‑turn exact + semantic; consistency; adherence; hallucination; conversation pass/fail via outcome‑first rule; regression deferred
Persistence: filesystem only; run artifacts JSON + CSV
Frontend: React + Vite + TypeScript + Tailwind + React Router; Google brand palette; card layout with borders; NavBar pages (Datasets, Runs, Reports, Settings, Golden Editor, Metrics, Golden Generator)

---

## Implementation Plan

### Prompt 1: Repository Scaffold
- What it implements: Creates top‑level folders and baseline READMEs.
- Dependency: None.
- Prompt:
  """
  Write complete and executable code to scaffold the repository with folders: backend/, frontend/, configs/, datasets/, scripts/, docs/.
  Add minimal README.md files explaining each folder. Include a root README.md summarizing the project and MVP scope.
  Add a scripts/README.md explaining how to run local dev.
  Add a CONTRIBUTING.md with coding standards and testing requirements.
  Include a .editorconfig and basic lint settings for Python and JS/TS.
  Add tests that verify the scaffold exists and required files are present.
  Execute the tests and show results.
  Make sure the code is production‑ready and avoids placeholders or stubs.
  """

### Prompt 2: Backend Scaffold (FastAPI + Settings)
- What it implements: FastAPI app with environment config and health endpoints.
- Dependency: Prompt 1.
- Prompt:
  """
  Write a FastAPI backend scaffold packaged under backend/app with pyproject.toml or requirements.txt.
  Implement settings: read GOOGLE_API_KEY, OLLAMA_HOST (default http://localhost:11434), and default thresholds (semantic=0.80).
  Add /health and /version endpoints. Auto‑disable Gemini features when GOOGLE_API_KEY is missing.
  Provide a uvicorn entrypoint and a make/PowerShell task to run the server.
  Add unit tests for settings and health endpoints. Run tests and show results.
  """

### Prompt 3: JSON Schemas (Datasets, Goldens, Run Config)
- What it implements: Machine‑readable JSON Schemas and validators.
- Dependency: Prompt 2.
- Prompt:
  """
  Define JSON Schemas under configs/schemas/ for:
  - conversation dataset (multi‑turn, roles, text, metadata: domain, difficulty, tags, version, conversation_id)
  - golden references (per‑turn variants, outcome fields: decision [ALLOW|DENY|PARTIAL], refund_amount?, reason_code?, next_action?, policy_flags[]?)
  - run configuration (datasets, models, metrics, thresholds, concurrency=1 by default)
  Implement a Python validator module that validates JSON payloads and returns actionable errors.
  Add tests with valid/invalid examples. Execute tests and show results.
  """

### Prompt 4: Dataset Loader & Repository
- What it implements: Filesystem loader with schema validation and listing.
- Dependency: Prompt 3.
- Prompt:
  """
  Implement a dataset repository that reads JSON datasets and goldens from datasets/.
  Functions: list_datasets(), get_dataset(dataset_id), get_conversation(conversation_id), get_golden(conversation_id).
  Validate all files against schemas with clear error messages. Reject invalid files.
  Add unit tests covering happy paths and error cases. Run tests and show results.
  """

### Prompt 5: Provider Registry (Ollama + Gemini)
- What it implements: Pluggable provider adapters and registry.
- Dependency: Prompt 2.
- Prompt:
  """
  Implement a provider registry with adapters for:
  - Ollama chat: model tag llama3.2:2b
  - Gemini chat: model gemini-2.5 (disabled if GOOGLE_API_KEY missing)
  Define a standard request/response schema capturing prompt, context turns, model metadata, timestamps, and provider response fields (latency, token stats if available, errors).
  Add unit tests using mocked HTTP calls. Execute tests and show results.
  """

### Prompt 6: Embeddings via Ollama (nomic-embed-text)
- What it implements: Local embedding client and cosine similarity.
- Dependency: Prompt 5.
- Prompt:
  """
  Implement an embeddings client using Ollama model nomic-embed-text.
  Provide functions: embed_text(text) and cosine_similarity(vec_a, vec_b).
  Handle batch embedding with backoff/retry. Add unit tests and run them.
  """

### Prompt 7: State Object Extractor
- What it implements: Extracts commerce/banking state fields from turns.
- Dependency: Prompt 3.
- Prompt:
  """
  Implement a deterministic state extractor that produces a compact state object for each conversation turn:
  - Common: user_intent, decision?, next_action?, policy_flags[]?, notes?
  - Commerce: order_id?, items?, totals?, refund_amount?
  - Banking: account_id?, amount?, kyc_status?, limit_flags?
  Use regex/entity rules, not LLMs. Provide unit tests with fixtures. Run tests and show results.
  """

### Prompt 8: Context Builder (State + Last 4 Turns)
- What it implements: Composes model input context per turn.
- Dependency: Prompt 7.
- Prompt:
  """
  Implement a context builder that takes the accumulated state object and the last 4 raw turns, returning a structured prompt for the provider adapters.
  Include safeguards for token budget (simple clipping) and logging for audit.
  Add tests. Execute tests and show results.
  """

### Prompt 9: Turn Runner
- What it implements: Executes one turn with context and captures output.
- Dependency: Prompts 5–8.
- Prompt:
  """
  Implement a turn runner that sends the current user turn along with context (state + last 4 turns) to the selected provider, captures raw output and metadata, and normalizes it into a canonical record.
  Store per‑turn artifacts (prompt, response, latency, errors) in a run folder.
  Add unit tests with provider mocks. Run tests and show results.
  """

### Prompt 10: Orchestrator & Job Queue (Headless)
- What it implements: Sequential run execution with job states.
- Dependency: Prompt 9.
- Prompt:
  """
  Implement an evaluation orchestrator that processes selected conversations sequentially with a simple in‑memory queue.
  Track job states: queued, running, succeeded, failed, cancelled, with progress percentages.
  Support cancellation and deterministic run IDs (dataset version, model, config checksum).
  Add tests for state transitions and cancellation. Run tests and show results.
  """

### Prompt 11: Metrics — Exact & Semantic Similarity
- What it implements: Per‑turn exact match and embedding‑based semantic score.
- Dependency: Prompt 6.
- Prompt:
  """
  Implement metrics for per‑turn scoring:
  - exact_match: case/whitespace‑normalized string equality against any acceptable variant
  - semantic_similarity: cosine(embedding(output), embedding(variant)) and take max across variants; pass if >= 0.80
  Return both raw scores and pass/fail. Add unit tests covering variants. Run tests and show results.
  """

### Prompt 12: Metrics — Consistency, Adherence, Hallucination
- What it implements: Additional per‑turn and final checks.
- Dependency: Prompts 7 and 11.
- Prompt:
  """
  Implement:
  - consistency: fail if final answer contradicts state fields or previous commitments
  - adherence: pass/fail by checking golden business‑rule constraints in dataset metadata
  - hallucination: fail if output introduces unseen critical entities (IDs, policy numbers, amounts) via regex/entity checks
  Provide clear reasons in metric breakdowns. Add unit tests. Run tests and show results.
  """

### Prompt 13: Conversation Outcome & Aggregation
- What it implements: Outcome‑first pass/fail and aggregation.
- Dependency: Prompt 12.
- Prompt:
  """
  Implement conversation‑level scoring:
  - Outcome‑first: require final outcome expectations to pass and no high‑severity violations; otherwise fail
  - Aggregate per‑turn metrics into a summary with weights (default 1.0 each), reporting pass/fail and summary scores
  Add unit tests and execute them.
  """

### Prompt 14: Persistence & Artifacts (JSON + CSV)
- What it implements: Run folder layout and writers.
- Dependency: Prompt 10.
- Prompt:
  """
  Define a run‑specific folder structure under runs/<run_id>/.
  Persist: inputs, per‑turn outputs, scores, logs, a machine‑readable results.json, and a tabular export results.csv.
  Implement writers and readers. Add tests and show results.
  """

### Prompt 15: Reports (Human‑Reviewable)
- What it implements: HTML/JSON report with diffs and evaluator fields.
- Dependency: Prompt 14.
- Prompt:
  """
  Generate a human‑readable HTML report per run including transcript, expected vs actual per turn, metric breakdowns, pass/fail flags, and string diffs.
  Include evaluator fields: rating, notes, override pass/fail. Persist feedback JSON linked to run ID.
  Add tests for report generation. Run tests and show results.
  """

### Prompt 16: REST API (FastAPI)
- What it implements: Endpoints for datasets, runs, jobs, results, feedback.
- Dependency: Prompts 4, 10, 14, 15.
- Prompt:
  """
  Implement REST API endpoints:
  - GET /datasets (list)
  - GET /conversations/{id} (with golden entries)
  - POST /runs (start evaluation)
  - GET /runs/{id}/status (progress)
  - GET /runs/{id}/results (per‑turn outputs, scores)
  - GET /runs/{id}/artifacts (download JSON/CSV/report bundle)
  - POST /runs/{id}/feedback (submit human evaluation)
  - GET /compare?runA=&runB= (simple side‑by‑side summary)
  Add OpenAPI docs and error handling. Include API tests. Run tests and show results.
  """

### Prompt 17: CLI Tools
- What it implements: Initialize workspace and run headless.
- Dependency: Prompts 3, 10.
- Prompt:
  """
  Provide a CLI with commands:
  - init: create default folders and sample files
  - run: execute a run from a run config file (headless)
  Add tests for CLI behavior. Execute tests and show results.
  """

### Prompt 18: Frontend Scaffold (Vite + React + TS + Tailwind)
- What it implements: Frontend project with theme setup.
- Dependency: Prompt 1.
- Prompt:
  """
  Scaffold a Vite React TypeScript app under frontend/ with Tailwind.
  Configure theme with Google brand palette: primary #4285F4, success #0F9D58, warning #F4B400, danger #DB4437.
  Establish a Card component with bordered layout and responsive grid. Add unit tests (Vitest + RTL). Run tests and show results.
  """

### Prompt 19: Frontend Routing & NavBar
- What it implements: React Router and top navigation.
- Dependency: Prompt 18.
- Prompt:
  """
  Implement React Router with pages: Datasets, Runs, Reports, Settings, Golden Editor, Metrics, Golden Generator.
  Create a sticky NavBar using the brand palette. Ensure keyboard accessibility.
  Add component tests and run them.
  """

### Prompt 20: Datasets Page (Upload/List)
- What it implements: JSON upload with schema validation and listing.
- Dependency: Prompts 3, 16, 19.
- Prompt:
  """
  Build a Datasets page that allows uploading JSON datasets/goldens, validates against schemas client‑side, then sends to backend.
  Show list with domain, difficulty, version, counts. Display validation errors.
  Add UI tests and run them.
  """

### Prompt 21: Runs Page (Configure + Progress)
- What it implements: Run configuration and live status.
- Dependency: Prompts 16, 19.
- Prompt:
  """
  Build a Runs page to select dataset(s), model (Ollama llama3.2:2b or Gemini gemini-2.5), thresholds, and start a run.
  Show live progress with job state and per‑conversation status.
  Add UI tests and run them.
  """

### Prompt 22: Reports Page (Per‑Turn + CSV + Compare)
- What it implements: Detailed report viewer and exports.
- Dependency: Prompts 15, 16, 19.
- Prompt:
  """
  Build a Reports page to view a run: transcript, expected vs actual side‑by‑side per turn, metric breakdowns, and pass/fail.
  Provide CSV download and a simple run compare selector for side‑by‑side summary.
  Add UI tests and run them.
  """

### Prompt 23: Settings Page (Providers & Thresholds)
- What it implements: Configure Ollama host and GOOGLE_API_KEY.
- Dependency: Prompt 16.
- Prompt:
  """
  Build a Settings page with forms to set OLLAMA_HOST and GOOGLE_API_KEY (stored in a local dev .env only; do not persist secrets to datasets).
  Allow configuring semantic threshold (default 0.80).
  Add UI tests and run them.
  """

### Prompt 24: Golden Data Generator (Commerce, Banking)
- What it implements: Template generator with coverage model.
- Dependency: Prompts 3, 19.
- Prompt:
  """
  Implement a Golden Generator page that produces conversation/golden JSON templates for Commerce and Banking.
  Ensure coverage across Intent × Outcome (ALLOW/DENY/PARTIAL) and Intent × Difficulty (easy/medium/hard), with pairwise coverage for constraint toggles.
  Generate 5–7 turn conversations with multiple acceptable variants for key turns. Allow download or send to editor.
  Add tests for template validity. Run tests and show results.
  """

### Prompt 25: Golden Set Editor UI
- What it implements: Full JSON editing workflow with versioning.
- Dependency: Prompts 3, 20, 24.
- Prompt:
  """
  Implement a Golden Editor page to create/edit datasets:
  - add/edit/delete turns (role, text)
  - add multiple acceptable variants per turn
  - set final outcome fields (decision, refund_amount, reason_code, next_action, policy_flags)
  - set metadata (domain, difficulty, tags, version)
  Validate via JSON Schema before save; bump version on save; write to filesystem via backend.
  Add UI tests and run them.
  """

### Prompt 26: Metrics Management UI
- What it implements: Manage metric bundle for MVP.
- Dependency: Prompts 11–13, 19.
- Prompt:
  """
  Implement a Metrics page to view and adjust the MVP metric bundle (enable/disable metrics, set weights and thresholds) with server‑side validation and persistence in configs/.
  Add UI tests and run them.
  """

### Prompt 27: Human Feedback Capture
- What it implements: UI form and backend persistence.
- Dependency: Prompts 15, 16, 22.
- Prompt:
  """
  Add a feedback widget on the Reports page to capture evaluator rating, notes, and override pass/fail per turn or for the conversation. Persist as structured JSON linked to run ID without mutating auto‑scores.
  Add tests and show results.
  """

### Prompt 28: Samples & E2E Tests
- What it implements: Sample datasets and end‑to‑end verification.
- Dependency: All prior backend and frontend prompts.
- Prompt:
  """
  Provide sample datasets: at least one 5‑turn Commerce conversation with golden references and a run configuration.
  Implement end‑to‑end tests that execute a run and verify artifact generation and report rendering. Use Playwright or equivalent.
  Run E2E tests and show results.
  """

### Prompt 29: Dev Experience — Single Start Command
- What it implements: One command to run full stack in dev.
- Dependency: Prompts 2 and 18.
- Prompt:
  """
  Implement a single dev command/script to start backend (uvicorn) and frontend (Vite) concurrently with clear logs. Provide Windows‑friendly scripts. Document in root README.
  Add a smoke test to hit /health and open the frontend home. Show results.
  """
