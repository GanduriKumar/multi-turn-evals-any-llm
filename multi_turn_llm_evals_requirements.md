Multi-Turn LLM Evaluation System – Functional Requirements (Enhanced EvalBench‑style)
Scope

A Python/FastAPI backend and a React frontend (using Tailwind CSS) to execute multi‑turn conversations against one or more LLMs, auto‑score outputs, and generate human‑reviewable evaluation reports using configurable golden references stored in JSON/YAML. This version intentionally excludes IAM/RBAC and containerization/deployment concerns but provides additional explicit capabilities requested by stakeholders, such as golden‑set editing and metric management.

Sequential functional requirements (one requirement per line)

Provide a repository scaffold with separate top‑level folders for backend, frontend, configs, datasets, scripts, and docs.

Provide a backend Python package scaffold with environment management (requirements/pyproject) and a runnable entrypoint.

Provide a frontend React scaffold with a standard build/dev workflow, environment‑based configuration and styling implemented with Tailwind CSS.

Provide a shared configuration folder that stores system‑wide settings for models, datasets, scoring, and reporting.

Provide a CLI script to initialize a new evaluation workspace (create default folders and sample files).

Provide a registry to define one or more LLM providers/endpoints (local, hosted, or proxy) using pluggable adapters.

Provide a standard request/response schema for LLM calls that includes prompt, conversation context, model metadata, and timestamps.

Provide a dataset format to store multi‑turn conversations as separate files from expected outputs (goldens).

Support JSON and YAML formats for dataset and golden reference configurations.

Provide a dataset loader that validates schema and rejects invalid datasets with actionable error messages.

Support versioning for datasets and golden references using semantic version fields in metadata.

Support tagging datasets by domain (e.g., search, checkout, returns, support) and by difficulty level.

Store each multi‑turn conversation as an ordered list of turns with explicit speaker roles and optional tool calls.

Store expected responses separately from conversations, keyed by stable conversation IDs and turn IDs.

Allow golden references to specify per‑turn expected outputs and optional final outcome expectations.

Allow golden references to specify multiple acceptable variants per expected output (e.g., paraphrases).

Allow golden references to specify structured expected fields (e.g., decision=ALLOW/DENY, reason_code, next_action).

Allow golden references to specify business‑rule constraints as machine‑readable conditions (e.g., refund_after_ship=false).

Allow golden references to specify evaluation weights per metric and per turn.

Provide a run configuration file (JSON/YAML) that selects dataset(s), model(s), metrics, and thresholds.

Provide an evaluation orchestrator that runs selected conversations sequentially or in parallel with configurable concurrency.

Provide an independent backend execution engine that can run evaluations without the UI (headless mode).

Provide an execution queue that tracks jobs, job states (queued/running/succeeded/failed), and progress percentages.

Provide job cancellation support that safely stops in‑flight conversation runs and marks them as cancelled.

Provide deterministic run IDs for reproducibility, including dataset version, model version, and config checksum.

Provide a turn runner that feeds each user turn to the LLM along with the accumulated prior context.

Provide support for app‑managed memory injection (e.g., summary or state object) as an optional input to each turn.

Provide support for truncation policies (full history, summarized history, windowed history) configurable per run.

Provide support for tool‑call simulation hooks (optional) so turns can include tool outputs when required by the dataset.

Capture raw LLM outputs per turn including tokens, latency, provider response metadata, and errors.

Normalize LLM outputs into a canonical evaluation record for scoring (text, structured fields, tool results).

Provide automatic scoring capability for each turn output against its corresponding golden reference.

Provide automatic scoring capability for the full conversation outcome (end‑to‑end success/failure).

Implement a metric for per‑turn correctness against golden reference (exact/semantic/structured match as configured).

Implement a metric for multi‑turn consistency (detect contradictions vs prior turns or constraints).

Implement a metric for instruction/constraint adherence across turns (budget/eligibility/policy constraints).

Implement a metric for hallucination risk using grounding checks against provided context (optional, if context provided).

Implement a metric for refusal/safety compliance when the dataset defines prohibited actions or restricted info.

Implement a metric for regression detection by comparing current scores against a baseline run.

Allow custom metric plugins to be added via a Python entrypoint mechanism.

Provide threshold rules to mark a turn as pass/fail based on metric scores and weights.

Provide threshold rules to mark a conversation as pass/fail using aggregated metrics across turns.

Provide aggregation functions (mean/min/weighted) configurable per metric for conversation‑level scoring.

Persist all evaluation artifacts (inputs, outputs, scores, logs) in a run‑specific folder structure.

Persist a machine‑readable results file per run in JSON format for downstream automation.

Persist a tabular results export per run in CSV format for analysts and reporting tools.

Generate a human evaluation report per run that includes conversation transcript, model outputs, scores, pass/fail flags, and diffs between expected and actual outputs.

Generate a human evaluation report that includes evaluator fields (rating, notes, override pass/fail) for manual review.

Provide a UI workflow to capture human evaluator feedback and store it back into a structured file linked to run ID.

Support re‑scoring using human overrides while keeping the original auto‑scores immutable.

Provide a golden update workflow that can propose new golden references from approved human‑reviewed outputs.

Provide audit logs that record who ran evaluations, with timestamps and configuration fingerprints.

Expose a REST API layer using FastAPI for all backend capabilities (datasets, runs, jobs, results, reports).

Provide an API endpoint to list available datasets and dataset metadata.

Provide an API endpoint to fetch a specific conversation by ID and its associated golden reference entries.

Provide an API endpoint to start a new evaluation run using a supplied run configuration.

Provide an API endpoint to stream or poll job progress (queued/running/completed) with per‑conversation status and progress.

Provide an API endpoint to retrieve per‑turn outputs, scores, and metric breakdowns for a given run and conversation.

Provide an API endpoint to download run artifacts (JSON results, CSV export, HTML/PDF report bundle).

Provide an API endpoint to submit human evaluation feedback for a specific run/conversation/turn.

Provide an API endpoint to compare two runs and return regression summaries and deltas by dataset/metric.

Provide OpenAPI/Swagger documentation automatically generated from the FastAPI application.

Provide backend validation and error handling that returns consistent error codes and messages for the frontend.

Provide a backend configuration to store secrets (API keys) securely via environment variables and not in dataset files.

Provide React UI screens to browse datasets, select models, choose metrics, and start an evaluation run.

Provide React UI screens to monitor run progress with per‑conversation status and live logs.

Provide React UI screens to view a conversation transcript with side‑by‑side expected vs actual responses per turn.

Provide React UI screens to view metric breakdowns (turn‑level and conversation‑level) and pass/fail thresholds.

Provide React UI screens to enter and save human evaluator ratings/notes and override decisions.

Provide React UI screens to compare runs and visualize regressions by metric and dataset tag.

Provide frontend support for downloading artifacts and reports from backend APIs.

Provide end‑to‑end tests that validate a sample dataset run from UI start to report generation.

Provide backend unit tests for dataset validation, turn runner, metric scoring, and report generation.

Provide a sample dataset and golden references demonstrating at least one 5‑turn commerce conversation.

Provide a sample run configuration demonstrating multiple models evaluated on the same dataset.

Provide a single command (or script) to start the full stack (backend + frontend) for local development.

Provide machine‑readable JSON Schema (or equivalent) definitions for conversation datasets, golden references, and run configuration files, and validate all inputs against these schemas before execution.

Allow each dataset to specify a task_type (e.g., qa, classification, policy_decision, tool_use, rag_qa) in its metadata so that evaluations can be grouped and analyzed by task.

Provide a configuration mechanism to define reusable metric bundles per task_type and allow run configuration files to reference these bundles by name.

Provide per‑provider configuration of rate limits, including maximum requests per time window and concurrency caps, so that the evaluation runner can respect external API limits.

Provide a retry policy configuration per provider that supports exponential backoff and distinguishes between transient errors (e.g., timeouts, 5xx) and permanent errors (e.g., invalid API key, 4xx).

Abstract run artifact storage behind a storage interface so that run outputs (JSON results, CSV exports, reports) can be persisted to local filesystem or remote object storage via pluggable adapters.

Provide clear, versioned sample JSON and YAML templates (with comments) for conversation datasets, golden references, and run configuration files in the repository’s configs and datasets folders.

Additional explicit requirements to address missing or partially stated functionality

Golden dataset generator: Provide a generic golden dataset generator that supports a variety of industry use cases by supplying templates and sample conversation flows for domains such as commerce, healthcare, finance, support, and more. The generator should produce conversation/golden pairs that conform to the system’s schema and include metadata tags for domain and difficulty.

Golden set editor UI: Provide a dedicated web‑based editor that allows authorized users to create, modify, or delete golden references and associated conversation entries. The editor should support versioning, track changes, and enforce schema validation. It should integrate with the golden update workflow and allow committing changes back into the dataset repository.

Metrics management interface: Provide a metrics management interface (UI and API) that allows administrators or evaluators to define new metrics, edit existing metric definitions (including weights, scoring functions, and thresholds), activate or deactivate metrics for future runs, and assign metrics to specific task_type bundles.

Dataset upload/import via UI: Provide a dataset upload/import interface in the frontend that allows users to select or drag‑and‑drop new conversation datasets and associated golden references. Uploaded datasets must be validated against the system’s schemas, tagged by domain/task, assigned a semantic version, and stored in the dataset repository alongside existing datasets.

NavBar and unified navigation: Provide a navigation bar (NavBar) in the frontend UI that organizes links to major sections such as Datasets, Runs, Reports, Metrics, Golden Editor, Configuration, and Settings. The NavBar should allow users to switch between these sections without losing session context.

Evaluation run interface enhancements: Enhance the evaluation run interface so that users can not only select existing datasets but also choose from newly uploaded datasets or import a dataset on the fly. The interface should offer options to pick models, metric bundles, truncation policies, and concurrency settings before launching a run.

Industry‑specific templates and documentation: Provide a collection of industry‑specific templates and documentation to guide users in creating conversation datasets and golden references for new domains, ensuring the evaluation framework can be adopted across various verticals.