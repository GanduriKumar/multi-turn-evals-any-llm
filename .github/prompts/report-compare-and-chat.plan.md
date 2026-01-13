# Implementation Plan: Report Comparison + Dataset/Report Chat

This plan follows Interview-First guidelines: clarified requirements are summarized, then modular prompts specify complete, testable units. No code is included here; each prompt instructs full implementation with tests and execution.

## Requirements Summary

- Report Comparison (any two runs):
  - Select two available reports from recent runs.
  - Align conversations primarily by `conversation_slug`/title; gracefully handle unmatched items.
  - Compute deltas: overall pass/fail, per-metric, per-domain/behavior, high-severity violations, token usage, and turn-level differences.
  - Visuals: donuts for pass/fail, bars for per-metric and domain/behavior changes, heatmap per metric across conversations/turns, optional Sankey for A→B shifts.
  - Exports: CSV/PDF of diff tables; side-by-side turn diff.
  - Business-friendly note when comparison is not apples-to-apples.

- Dataset Chat (single conversation at a time):
  - Chat over a selected conversation from dataset generator page.
  - Retrieval-context includes that conversation’s turns and metadata.
  - Model selection: per-message with ability to change mid-session (drop-down of configured models).
  - Deterministic defaults: temperature 0.0, seed 42; budgets ~1800 input, 400 completion.

- Report Chat:
  - Chat over a selected report (results.json), answer all possible details: explain failures, summarize metrics, drill into specific turns, list high severity violations, highlight outliers.
  - Model selection per-message with mid-session change allowed.

- Backend/Data:
  - New endpoints for: run listing, comparison diff, dataset conversation chat, report chat.
  - Embeddings/indexing using configured embed model (default `nomic-embed-text` via Ollama), cache indices on disk.

- UX:
  - Reports page: comparison selector (two runs), results with charts and tables, download exports.
  - Dataset generator page: conversation selector + chat panel.
  - Reports page: chat panel for selected run.

- Performance/Privacy:
  - Use provided defaults (semantic/hallucination thresholds, token budgets) and expose settings.
  - Aim for fast responses (≤5s typical); index caching for speed.
  - Store chat transcripts locally with opt-out; respect environment settings for secrets.

---

## Implementation Plan

### Prompt 1: Report Diff Model + Alignment
- **What it implements:** A backend diff engine to align conversations (by `conversation_slug`/title, fallback to ID) across two runs and compute deltas for conversation-level and turn-level metrics.
- **Dependency:** None.
- **Prompt:**
  ```
  Write complete and executable backend code to implement a ReportDiff module that:
  - Loads two run results (`results.json`) and aligns conversations by `conversation_slug` or `conversation_title`, fallback to `conversation_id`.
  - Computes deltas: overall pass/fail counts, per-metric pass rates, per-domain/behavior changes, high-severity violation counts, token usage differences, and per-turn metric diffs.
  - Marks items as unmatched with reason when alignment fails.
  - Returns a JSON diff structure optimized for visualization (rollups + detailed tables).
  Include unit tests covering alignment cases (perfect match, partial match, mismatched datasets) and metric delta calculations. Execute the tests and show results.
  Ensure production-ready code, avoiding placeholders.
  ```

### Prompt 2: Comparison API Endpoint
- **What it implements:** FastAPI endpoint to compare two runs and return diff JSON; optionally render PDF/CSV.
- **Dependency:** Prompt 1.
- **Prompt:**
  ```
  Implement a FastAPI route `GET /reports/compare?runA=...&runB=...` that:
  - Loads the two runs from the vertical-aware runs root.
  - Uses ReportDiff to produce diff JSON with rollups and tables.
  - Supports `type=csv|pdf|json` for export; for PDF, reuse existing rendering infra or provide a simple fallback.
  Add tests for endpoint responses and error handling (missing runs, invalid params). Execute tests and show results.
  ```

### Prompt 3: Comparison UI with Charts
- **What it implements:** Reports page UI for selecting two runs and visualizing diffs via charts and tables.
- **Dependency:** Prompt 2.
- **Prompt:**
  ```
  Implement frontend changes on the Reports page to:
  - Add two run selectors (searchable list of available `run_id`s) and a Compare button.
  - Render visuals: donut charts for pass/fail distribution; bar charts for per-metric and domain/behavior deltas; heatmap matrix for per-metric pass/fail across conversations and turns; optional Sankey for A→B transitions.
  - Include side-by-side turn diff view and CSV/PDF export buttons.
  - Display an explicit banner when comparisons are not apples-to-apples, using business-friendly language.
  Add component-level tests and snapshot tests for UI states. Execute tests and show results.
  ```

### Prompt 4: Diff Exports (CSV/PDF)
- **What it implements:** Export utilities for diff tables.
- **Dependency:** Prompt 2.
- **Prompt:**
  ```
  Implement backend utilities to export diff tables to CSV and render a PDF summary report.
  - CSV: per-conversation and per-turn tables with delta columns.
  - PDF: compact layout of charts, key metrics, and a summary narrative.
  Add tests for CSV content correctness and PDF rendering (skip if dependencies missing, but test structure). Execute tests and show results.
  ```

### Prompt 5: Dataset Conversation Chat Backend
- **What it implements:** Chat endpoint to answer questions using a single selected conversation from the dataset.
- **Dependency:** None.
- **Prompt:**
  ```
  Implement `POST /chat/dataset` that accepts `{ dataset_id, conversation_id, model_spec, messages }` and:
  - Builds retrieval context from the conversation turns and metadata (policy/facts if available).
  - Supports per-message model selection (`provider:model`) with temperature 0.0, seed 42, window_turns from settings.
  - Applies token budgets (~1800 input, ~400 output) and returns provider meta and latency.
  Add tests using stubbed providers to verify context building and response shaping. Execute tests and show results.
  ```

### Prompt 6: Dataset Conversation Chat UI
- **What it implements:** Chat panel on Dataset Generator page with model selector per message.
- **Dependency:** Prompt 5.
- **Prompt:**
  ```
  Add a chat panel to the Dataset Generator page:
  - Conversation selector (dropdown) from the selected dataset.
  - Chat input with per-message model selector (choices from configured models in settings).
  - Message list showing assistant responses, provider meta, and latency.
  - Allow mid-session model changes; persist selection in local storage per vertical.
  Add component tests for selector and chat flow. Execute tests and show results.
  ```

### Prompt 7: Report Chat Backend (RAG over results)
- **What it implements:** Chat endpoint to answer questions using a selected report; supports explain/drilldown.
- **Dependency:** Prompt 1 (for data structs) or standalone.
- **Prompt:**
  ```
  Implement `POST /chat/report` that accepts `{ run_id, model_spec, messages }` and:
  - Loads `results.json` for the run and constructs a lightweight index (in-memory or cached embeddings) over conversations, turns, and metrics.
  - Retrieves relevant snippets (failures, high-severity, outliers) and builds a concise context.
  - Supports per-message model selection and budgets similar to dataset chat.
  Add tests with synthetic `results.json` to validate retrieval and answer composition. Execute tests and show results.
  ```

### Prompt 8: Report Chat UI
- **What it implements:** Chat panel on Reports page for a selected run with model selector.
- **Dependency:** Prompt 7.
- **Prompt:**
  ```
  Add a chat panel to the Reports page:
  - Run selector; chat input with per-message model selector.
  - Responses can link to specific conversations/turns; provide quick chips (e.g., "Show failures", "Summarize metrics").
  - Allow mid-session model changes; show provider meta.
  Add component tests for run selection and chat interactions. Execute tests and show results.
  ```

### Prompt 9: Embeddings + Index Caching
- **What it implements:** Embedding pipeline using Ollama (`EMBED_MODEL`) and caching to speed report chat.
- **Dependency:** Prompt 7.
- **Prompt:**
  ```
  Implement an embedding/index layer that:
  - Uses configured `EMBED_MODEL` and `OLLAMA_HOST` to embed text chunks (titles, turn snippets, failure reasons).
  - Caches per-run index on disk in the runs folder; handles versioning and invalidation.
  - Gracefully degrades if embeddings are unavailable (return informative errors).
  Add tests for index build, cache reuse, and error handling. Execute tests and show results.
  ```

### Prompt 10: Settings + Performance Budgets
- **What it implements:** Integrate settings to control token budgets, thresholds, and UI defaults.
- **Dependency:** Prompts 5, 7.
- **Prompt:**
  ```
  Extend settings endpoints/UI to expose chat-related parameters: input/output token budgets, window_turns, thresholds.
  - Respect existing environment variables and metrics-config.
  - Wire defaults: temperature 0.0, seed 42, ~1800 input, ~400 completion.
  Add tests verifying persistence and runtime application. Execute tests and show results.
  ```

### Prompt 11: Tests: E2E Scenarios
- **What it implements:** End-to-end tests for comparison and chat flows.
- **Dependency:** All prior prompts.
- **Prompt:**
  ```
  Write E2E tests to:
  - Compare two synthetic runs and validate charts/tables presence and diff correctness.
  - Dataset chat over a selected conversation with stub provider.
  - Report chat retrieval of failures and drilldowns.
  Execute tests and show results.
  ```

### Prompt 12: Documentation
- **What it implements:** Update README/UserGuide with feature usage, constraints, and settings.
- **Dependency:** All prior prompts.
- **Prompt:**
  ```
  Update documentation to describe:
  - How to compare reports and interpret charts.
  - How to chat with a dataset conversation or a report, including model selection per message.
  - Performance tips, settings, and limitations.
  Include screenshots or examples where possible.
  ```
