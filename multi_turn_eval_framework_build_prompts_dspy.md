# Single‑Step Build Prompts — Enhanced Multi‑Turn LLM Evaluation Framework

(FastAPI backend, React + Tailwind frontend, comprehensive metrics and golden management)

Purpose: This document decomposes the functional requirements of the enhanced multi‑turn LLM evaluation framework into discrete, one‑step prompts. Each [ ] Promptstands on its own and describes a single functional requirement or sub‑requirement. Prompts are ordered to guide development from repository scaffolding through backend services, REST API layers, frontend interfaces and additional features. Every [ ] Promptexplicitly instructs that a full code implementation must be provided, along with tests to validate the implementation.

# PROMPTS
# Repository and scaffolding
# [x] Prompt001 — Create repository scaffold

Prompt:

Initialize a new Git repository for the project. Create the top‑level folders: backend/, frontend/, configs/, datasets/, metrics/, templates/, scripts/ and docs/. Add a root README.md explaining the purpose of the framework, a .gitignore covering Python, Node and environment artifacts, and an Apache‑2.0 LICENSE. Ensure all directories exist and are tracked in version control. You must produce the full code implementation for this [ ] Prompt– no placeholders or skeletal files are allowed.

Expected outputs:

Monorepo directory structure with all specified folders

README.md, .gitignore, and LICENSE committed at the repository root

Tests to add/run now:

Write a test script in the repository root (scripts/test_structure.sh) that asserts the existence of each required folder and file. Run this script with a shell to verify the scaffold. Fail the test if any directory or file is missing.

# [x] Prompt002 — Scaffold backend Python package

Prompt:

Within backend/, create a source‑layout Python package named eval_server. Add a pyproject.toml declaring dependencies (fastapi, uvicorn, pydantic, pydantic-settings, pyyaml, httpx, pytest, pytest-cov) and an entrypoint for running the server. Add a requirements.txt containing the same dependencies for alternative installers. Include an empty __init__.py in backend/src/eval_server to mark it as a package. Provide a complete implementation for all configuration files and package setup – the package must be importable and runnable without missing pieces.

Expected outputs:

backend/pyproject.toml with correct dependencies and package metadata

backend/requirements.txt listing the same dependencies

backend/src/eval_server/__init__.py file

Tests to add/run now:

Write a unit test (backend/tests/test_install.py) that installs the package via pip install -e . and imports eval_server without error.

Run pytest in the backend directory to ensure the package installs and loads successfully.

# [x] Prompt003 — Scaffold frontend React project with Tailwind

Prompt:

Initialize a new React project inside frontend/ using Vite and TypeScript. Configure Tailwind CSS by installing tailwindcss, postcss and autoprefixer, creating a tailwind.config.js with a custom colour palette, and adding Tailwind directives to src/index.css. Organize the React project with directories src/components/, src/pages/, src/hooks/, src/utils/. Create a .gitignore for Node modules inside frontend/. Do not generate placeholder components beyond the project starter. Your output must be a fully bootstrapped and compilable Vite + React + TypeScript project with Tailwind integration, not just scaffolding instructions.

Expected outputs:

A working Vite + React + TypeScript project inside frontend/

Tailwind configured and integrated

Folder structure under frontend/src/ as specified

Tests to add/run now:

Run npm install (or yarn install) in frontend/ and ensure it completes without errors.

Run npm run dev and visit the development server in a test (for example, using httpx or a headless browser) to confirm that the project boots and the default page renders.

# [x] Prompt004 — Create shared configuration folder

Prompt:

Under configs/, create subdirectories for datasets/, runs/, and schemas/. Add a top‑level configs/settings.yaml that contains system‑wide defaults for models, datasets, scoring weights, report settings, and UI options. Document each configuration option with comments. Provide a Python helper in backend/src/eval_server/config_loader.py that can load these YAML files and return typed objects. Implement all functions and configuration files completely – do not leave any part unimplemented.

Expected outputs:

Directory structure configs/datasets/, configs/runs/, configs/schemas/

settings.yaml with documented keys and default values

config_loader.py with a function load_settings() reading YAML into a Pydantic model

Tests to add/run now:

Write a test (backend/tests/test_config_loader.py) that uses load_settings() to read settings.yaml and asserts that all expected keys are present and types match.

Verify that invalid YAML triggers a validation error.

# [x] Prompt005 — Create CLI script to initialize a new evaluation workspace

Prompt:

Develop a command‑line script scripts/init_workspace.py that sets up a fresh evaluation workspace in a user‑specified directory. The script should create default folders (datasets/, runs/, reports/) and copy sample configuration files and dataset templates from the repository into the workspace. It should also generate a sample .env file populated with example values. Provide a console entrypoint in backend/pyproject.toml to run this script with python -m eval_server.init_workspace. Deliver a fully working script and entrypoint; do not rely on external manual steps.

Expected outputs:

scripts/init_workspace.py script that performs workspace initialization

Console entrypoint registered in pyproject.toml

Sample .env generated in the target workspace with placeholder values

Tests to add/run now:

Write an integration test (backend/tests/test_init_workspace.py) that runs the script using subprocess in a temporary directory and verifies that all required folders and files are created.

Verify that running the script twice does not overwrite existing files without confirmation.

# [x] Prompt006 — Implement LLM provider registry and adapter pattern

Prompt:

Create a provider registry in backend/src/eval_server/llm/provider_registry.py that can register and retrieve LLM provider adapters by name. Define an abstract base class LLMProvider with methods initialize(), generate(prompt: str, context: list) -> str, and metadata() -> dict. Implement a DummyProvider in backend/src/eval_server/llm/providers/dummy_provider.py returning deterministic responses for testing. Add a provider entry in the configuration to select providers by name. Ensure the registry supports registration and retrieval of multiple providers with full implementations – avoid incomplete adapter methods.

Expected outputs:

Provider registry module with registration and lookup functions

Abstract LLMProvider base class defined in a module

Concrete DummyProvider implementing all required methods

Tests to add/run now:

Write unit tests (backend/tests/test_provider_registry.py) to register the DummyProvider and retrieve it by name, ensuring that generate() returns the expected deterministic output.

Test that registering two providers with the same name raises an error.

# [x] Prompt007 — Define standard request/response schema for LLM calls

Prompt:

Define a request/response schema for LLM calls using Pydantic models. The request must include a Promptstring, conversation context, model metadata (e.g., provider, model version), and timestamps. The response must capture the generated text, token usage, latency, and any provider metadata. Save these models in backend/src/eval_server/schemas/llm.py. Fully implement all Pydantic models with type validations – partial or placeholder fields are not acceptable.

Expected outputs:

Pydantic request and response models for LLM calls in schemas/llm.py

Tests to add/run now:

Write unit tests (backend/tests/test_llm_schema.py) that instantiate the request and response models with valid data and verify serialization to and from JSON.

Test that invalid inputs (missing fields, wrong types) raise validation errors.

# [x] Prompt008 — Design dataset format for multi‑turn conversations and goldens

Prompt:

Define a dataset format for storing multi‑turn conversations in separate files from their expected outputs (goldens). Each conversation file should contain a unique conversation ID, metadata (tags, difficulty), and a list of turns with speaker roles and optional tool calls. Each golden file should map conversation and turn IDs to expected responses, multiple acceptable variants, structured expected fields, weights, business‑rule constraints, and evaluation conditions. Use JSON and YAML schemas stored in configs/schemas/. Produce complete schema files and examples – the format must be ready for immediate use in the evaluation system.

Expected outputs:

JSON Schema files defining conversation and golden formats

Example dataset and golden files conforming to the schemas

Tests to add/run now:

Write tests (backend/tests/test_dataset_format.py) that validate the example dataset and golden files against the JSON Schema definitions using a validation library.

Test that missing or extra fields in the examples cause validation failures.

# [x] Prompt009 — Support JSON and YAML dataset formats

Prompt:

Ensure that the evaluation system can load both JSON and YAML formats for conversation datasets and golden references. Implement loader functions in backend/src/eval_server/data/loader.py that detect file extensions, parse JSON or YAML, and return typed objects. Reuse the schemas defined previously for validation. Deliver fully implemented loaders with appropriate error handling for invalid formats.

Expected outputs:

Loader functions supporting JSON and YAML

Unit tests covering JSON and YAML parsing

Tests to add/run now:

Write tests (backend/tests/test_loader.py) that load sample JSON and YAML datasets and assert that the parsed objects match expected structures.

Test that loading invalid JSON/YAML raises a clear exception.

# [x] Prompt010 — Implement dataset loader with schema validation and error messages

Prompt:

Develop a dataset loader in backend/src/eval_server/data/validation_loader.py that validates each conversation and golden file against the schemas. On invalid input, produce actionable error messages specifying the location and reason of failure. Integrate the loader with the functions created in # [ ] Prompt009. Write complete validation logic and descriptive error reporting – avoid generic exceptions.

Expected outputs:

validation_loader.py with validation logic and error reporting

Tests verifying schema validation and error messages

Tests to add/run now:

Implement tests (backend/tests/test_validation_loader.py) that provide both valid and invalid datasets to the loader and assert appropriate behaviour and error messages.

# [x] Prompt011 — Add versioning for datasets and golden references

Prompt:

Enhance dataset and golden metadata to include semantic version numbers (major.minor.patch). Update the schemas to require version fields and implement version comparison functions in backend/src/eval_server/data/versioning.py. Use version numbers when naming output files and results directories. Provide complete version management functionality – do not leave version handling unimplemented.

Expected outputs:

Updated schemas with required version fields

versioning.py with functions to parse and compare version strings

Tests to add/run now:

Write tests (backend/tests/test_versioning.py) that parse various semantic version strings and compare them correctly (e.g., 1.0.0 > 0.9.5).

Test that missing or malformed version fields in datasets cause validation errors.

# [x] Prompt012 — Implement tagging for dataset domain and difficulty

Prompt:

Allow datasets to be tagged by domain (e.g., search, checkout, returns, support) and difficulty (e.g., easy, medium, hard). Update the dataset schema to include optional tags and difficulty fields. Provide helper functions to filter datasets by tags and difficulty in backend/src/eval_server/data/filtering.py. Ensure that tag handling and filtering are fully implemented and documented.

Expected outputs:

Updated schema supporting tags and difficulty

Filtering helpers for datasets based on tags and difficulty

Tests to add/run now:

Write tests (backend/tests/test_filtering.py) that filter a list of dataset metadata by various tag and difficulty combinations and assert that the correct datasets are returned.

# [x] Prompt013 — Store multi‑turn conversation turns as ordered lists with speaker roles and tool calls

Prompt:

Define a data model where each conversation is an ordered list of turns, each turn specifying the speaker (user or assistant) and optionally including a tool call (with tool name and arguments). Update the dataset schema accordingly and implement classes in backend/src/eval_server/models/conversation.py to represent conversations and turns. Provide complete type definitions and any necessary helper functions for working with turns and tool calls.

Expected outputs:

Updated dataset schema with ordered turn representation

Conversation and Turn classes with attributes for speaker, content, and tool call

Tests to add/run now:

Write tests (backend/tests/test_conversation_model.py) that construct conversations with and without tool calls and verify ordering and data integrity.

# [x] Prompt014 — Store expected responses separately keyed by conversation and turn IDs

Prompt:

Design the golden reference format so that expected responses are stored separately from the conversations. Each golden entry should be keyed by conversation_id and turn_id. Update the golden schema and implement helper functions in backend/src/eval_server/data/golden_access.py to fetch expected responses and associated metadata. Fully implement the separation and retrieval logic – no part should remain as a stub.

Expected outputs:

Updated golden schema with separate keying fields

Helper functions to retrieve golden entries by conversation and turn

Tests to add/run now:

Write tests (backend/tests/test_golden_access.py) that store and retrieve golden entries by ID and verify correctness.

# [x] Prompt015 — Allow per‑turn expected outputs and optional final outcome expectations

Prompt:

Modify the golden reference format to support expected outputs at each turn and to optionally specify a final outcome for the entire conversation (such as a successfully completed transaction). Update the schema accordingly and implement logic in the evaluation system to store and evaluate these final outcomes. Deliver full support for both per‑turn and conversation‑level expected outcomes.

Expected outputs:

Schema fields for per‑turn and final expected outcomes

Evaluation logic capable of checking final conversation outcomes

Tests to add/run now:

Write tests (backend/tests/test_final_outcome.py) that include final outcome expectations and verify that they are evaluated properly.

# [x] Prompt016 — Support multiple acceptable variants per expected output

Prompt:

Extend the golden reference to allow multiple acceptable output variants for each turn. Implement data structures and evaluation logic to treat any variant as a match during scoring. Update the schema to support a list of acceptable responses. Ensure that variant matching is fully implemented and used by the scoring functions.

Expected outputs:

Schema allowing lists of variants

Scoring logic that iterates through acceptable variants and matches accordingly

Tests to add/run now:

Write tests (backend/tests/test_variants.py) that define multiple variants for a turn and verify that the evaluation accepts any matching variant.

# [x] Prompt017 — Support structured expected fields in golden references

Prompt:

Allow golden references to specify structured expected fields such as decision=ALLOW/DENY, reason_code, and next_action. Update the golden schema to include these structured fields and implement evaluation functions that compare structured outputs. Provide full implementations for structured field comparison – do not leave placeholders.

Expected outputs:

Updated golden schema supporting structured fields

Comparison functions that verify structured outputs match expected values

Tests to add/run now:

Write tests (backend/tests/test_structured_fields.py) for scenarios where structured fields are present and ensure the evaluation compares them correctly.

# [x] Prompt018 — Implement business‑rule constraints in golden references

Prompt:

Enable golden references to define machine‑readable business‑rule constraints (e.g., refund_after_ship=false). Update the schema to include a constraints section with condition expressions. Implement an evaluator in backend/src/eval_server/scoring/constraints.py that interprets these conditions and validates outputs accordingly. Implement a complete parser and evaluator for constraint expressions; do not leave rules partially implemented.

Expected outputs:

Schema supporting business‑rule constraints

Constraint evaluation module parsing and enforcing conditions

Tests to add/run now:

Write tests (backend/tests/test_constraints.py) that define conversations with constraints and verify that scoring fails or passes correctly based on the outputs.

# [x] Prompt019 — Specify evaluation weights per metric and per turn

Prompt:

Allow golden references to specify evaluation weights for each metric and per turn. Update the golden schema to include weight fields and ensure the evaluation system uses these weights when aggregating scores. Provide functions to calculate weighted averages at turn and conversation levels. Fully implement weight handling and ensure it affects scoring results.

Expected outputs:

Updated golden schema with weight fields

Weighted aggregation functions for scores

Tests to add/run now:

Write tests (backend/tests/test_weights.py) that define different weight scenarios and verify that weighted scoring aggregates as expected.

# [x] Prompt020 — Create run configuration file format

Prompt:

Design a run configuration file format (JSON or YAML) that selects dataset(s), model(s), metric bundles, truncation policies, concurrency settings, and thresholds. Store examples in configs/runs/sample_run_config.yaml and update the schemas. Implement a parser in backend/src/eval_server/config/run_config_loader.py that reads and validates run configurations. Ensure the parser and schema are fully implemented, ready for immediate use.

Expected outputs:

Sample run configuration file with multiple models and datasets

Schema for run configuration

Run configuration loader with validation

Tests to add/run now:

Write tests (backend/tests/test_run_config_loader.py) that load sample configurations and validate them.

Test that invalid configurations result in informative errors.

# [x] Prompt021 — Implement evaluation orchestrator with concurrency support

Prompt:

Develop an evaluation orchestrator in backend/src/eval_server/orchestrator.py that reads the run configuration, spawns tasks for each conversation, and manages concurrency (threads or async tasks). Allow sequential or parallel execution based on configuration. Provide hooks for progress reporting and cancellation. Deliver a fully functioning orchestrator – concurrency and cancellation must be properly implemented.

Expected outputs:

orchestrator.py with functions to schedule evaluations, monitor progress, and respect concurrency settings

Integration with dataset loader and scoring modules

Tests to add/run now:

Write tests (backend/tests/test_orchestrator.py) that run a small evaluation with configurable concurrency and assert correct ordering and results.

Test that cancellation stops execution and reports a cancelled status.

# [x] Prompt022 — Provide headless execution engine for backend

Prompt:

Provide an independent execution engine in backend/src/eval_server/headless_engine.py that can run evaluations without the UI. The engine should accept a run configuration path and produce run artifacts to the output directory. Integrate with the orchestrator. Fully implement the engine such that it can be invoked from the command line and run evaluations end to end.

Expected outputs:

Headless execution engine module

Ability to run evaluations via CLI and produce results

Tests to add/run now:

Write integration tests (backend/tests/test_headless_engine.py) that run the headless engine with a sample run config and verify that the run completes and artifacts are created.

# [x] Prompt023 — Implement execution queue tracking job states and progress

Prompt:

Implement an execution queue in backend/src/eval_server/queue.py that tracks jobs, job states (queued, running, succeeded, failed, cancelled), and progress percentages. The queue should support adding jobs, updating their state, querying status, and persistent storage (e.g., in memory for now). Ensure all queue operations are fully implemented with thread‑safe or async‑safe mechanisms.

Expected outputs:

Execution queue with job state and progress tracking

Functions to add, update, and query jobs

Tests to add/run now:

Write tests (backend/tests/test_queue.py) that add jobs, update their states, and verify that progress and states are correctly tracked.

Test concurrent updates to ensure thread safety.

# [x] Prompt024 — Provide job cancellation support

Prompt:

Add job cancellation functionality to the execution queue and orchestrator. Implement a cancel_job(job_id) function that stops any running tasks and marks the job as cancelled. Update the orchestrator to periodically check for cancellation signals. Deliver a working cancellation implementation that safely stops evaluation tasks.

Expected outputs:

Cancellation function integrated into queue and orchestrator

Jobs marked cancelled upon user request

Tests to add/run now:

Write tests (backend/tests/test_cancellation.py) that start a long‑running evaluation, issue a cancellation, and verify that the job status is updated and no further processing occurs.

# [x] Prompt025 — Generate deterministic run IDs for reproducibility

Prompt:

Design a deterministic run ID generation strategy based on the dataset version(s), model version(s), metric bundle(s), and a configuration checksum. Implement a function in backend/src/eval_server/utils/run_id.py that computes a unique ID by hashing these elements. Fully implement the hashing and ensure identical inputs produce the same ID.

Expected outputs:

Function to generate deterministic run IDs

Tests to add/run now:

Write tests (backend/tests/test_run_id.py) that pass identical and different configuration objects and assert that run IDs are consistent or differ accordingly.

# [x] Prompt026 — Build turn runner to feed user turns with accumulated context

Prompt:

Implement a turn runner in backend/src/eval_server/runner/turn_runner.py that feeds each user turn to the LLM along with the accumulated prior context. It should call the LLM provider, handle responses, and maintain context state. Provide a complete turn runner implementation that interacts with LLM providers and manages context; do not leave stubbed methods.

Expected outputs:

Turn runner that builds context and sends requests to the LLM

Tests to add/run now:

Write tests (backend/tests/test_turn_runner.py) that simulate a multi‑turn conversation and verify that the context passed to the LLM grows correctly.

Test that the runner can handle tool calls as part of context when required.

# [x] Prompt027 — Add support for memory injection

Prompt:

Provide support for app‑managed memory injection (e.g., conversation summaries or state objects) as an optional input to each turn. Modify the turn runner to accept an additional memory object and merge it with the prior context before sending to the LLM. Fully implement memory injection; do not leave memory merge logic unimplemented.

Expected outputs:

Updated turn runner accepting and using memory objects

Tests to add/run now:

Write tests (backend/tests/test_memory_injection.py) that provide memory objects and verify that they are incorporated into the context passed to the LLM.

# [x] Prompt028 — Support configurable truncation policies

Prompt:

Implement truncation policies in backend/src/eval_server/utils/truncation.py supporting full history, summarized history, and windowed history. Allow run configurations to specify which policy to use. Update the turn runner to apply the selected policy when building context. Provide full implementation of truncation strategies and ensure they operate correctly in different scenarios.

Expected outputs:

Truncation policy functions for full, summarized, and windowed history

Turn runner integration that applies these policies

Tests to add/run now:

Write tests (backend/tests/test_truncation.py) that feed long conversation histories and verify that the context passed to the LLM follows the specified truncation policy.

# [x] Prompt029 — Support tool‑call simulation hooks

Prompt:

Add support for tool‑call simulation hooks so that turns can include tool outputs when required by the dataset. Implement an optional callback interface in the turn runner that can supply tool outputs for a given tool name and arguments. Provide a default no‑op implementation when no tool is used. Fully implement the hook mechanism and ensure it integrates with the turn runner and evaluation flow.

Expected outputs:

Hook interface for simulating tool outputs

Updated turn runner calling the hook when a tool call is specified in the dataset

Tests to add/run now:

Write tests (backend/tests/test_tool_hooks.py) that define a mock hook returning a sample tool output and verify that the turn runner injects the tool output correctly into the conversation context.

# [ ] Prompt030 — Capture raw LLM outputs per turn

Prompt:

Modify the evaluation system to capture raw LLM outputs per turn including generated tokens, latency, provider response metadata, and errors. Store this information alongside normalized outputs. Update the response schema and data storage accordingly. Ensure full capture of metadata and error handling in the implementation.

Expected outputs:

Extended response schema including tokens, latency, and metadata

Data storage capturing raw outputs for each turn

Tests to add/run now:

Write tests (backend/tests/test_raw_outputs.py) that simulate LLM calls and verify that tokens, latency, and metadata fields are recorded correctly.

Test error scenarios and ensure metadata captures error details.

# [ ] Prompt031 — Normalize LLM outputs into canonical evaluation records

Prompt:

Develop a normalizer in backend/src/eval_server/scoring/normalizer.py that transforms raw LLM outputs into canonical evaluation records containing text, structured fields, and tool results. Apply consistent casing, whitespace normalization, and structured extraction. Fully implement normalization logic; partial transformation is not acceptable.

Expected outputs:

Normalization module that produces canonical evaluation records

Integration with scoring functions

Tests to add/run now:

Write tests (backend/tests/test_normalizer.py) that take raw outputs and verify that the normalized records have consistent structure and content.

# [ ] Prompt032 — Provide automatic scoring per turn against golden reference

Prompt:

Implement a scoring module in backend/src/eval_server/scoring/turn_scoring.py that compares canonical outputs to the golden reference for each turn. The module should calculate metric scores (defined later) and produce per‑turn pass/fail results based on thresholds. Implement the complete scoring logic; do not leave metrics unimplemented at this stage.

Expected outputs:

Turn scoring module returning scores per metric and pass/fail status

Tests to add/run now:

Write tests (backend/tests/test_turn_scoring.py) that compare sample outputs to goldens and verify that scores and pass/fail results are computed correctly.

# [ ] Prompt033 — Provide automatic scoring for full conversation outcomes

Prompt:

Create a scoring module in backend/src/eval_server/scoring/conversation_scoring.py that aggregates per‑turn metrics into an overall conversation score and determines end‑to‑end success or failure. Support configurable aggregation functions (mean/min/weighted). Provide full aggregation logic and ensure conversation‑level results are computed without stubs.

Expected outputs:

Conversation scoring module computing aggregated scores and pass/fail

Tests to add/run now:

Write tests (backend/tests/test_conversation_scoring.py) that feed sequences of per‑turn scores and verify correct aggregation for mean, min, and weighted functions.

# [ ] Prompt034 — Implement per‑turn correctness metric

Prompt:

Implement a metric that measures per‑turn correctness by comparing generated outputs to the golden reference. Support exact match, semantic similarity (e.g., embedding cosine similarity), and structured match as configured. Place the implementation in backend/src/eval_server/metrics/correctness.py. Provide full implementations of all match modes – do not leave incomplete sections.

Expected outputs:

Correctness metric module with functions for exact, semantic, and structured match

Tests to add/run now:

Write tests (backend/tests/test_correctness_metric.py) that evaluate different match modes and assert correct scoring behaviour.

# [ ] Prompt035 — Implement multi‑turn consistency metric

Prompt:

Create a metric that assesses multi‑turn consistency by detecting contradictions or drift relative to prior turns or constraints. Implement this metric in backend/src/eval_server/metrics/consistency.py. Implement complete logic for detecting inconsistency – partial detection is not acceptable.

Expected outputs:

Consistency metric module detecting contradictions across turns

Tests to add/run now:

Write tests (backend/tests/test_consistency_metric.py) that simulate conversations with contradictions and verify that the metric detects them.

# [ ] Prompt036 — Implement instruction/constraint adherence metric

Prompt:

Develop a metric to measure instruction and policy adherence across turns (e.g., budget limits, eligibility criteria). Place the implementation in backend/src/eval_server/metrics/adherence.py. Ensure the metric can access constraints specified in the dataset and evaluate each turn for adherence. Fully implement adherence checks and scoring.

Expected outputs:

Adherence metric module

Tests to add/run now:

Write tests (backend/tests/test_adherence_metric.py) that define constraints and verify that the metric correctly penalizes violations and rewards adherence.

# [ ] Prompt037 — Implement hallucination risk metric

Prompt:

Develop a hallucination risk metric that checks the grounding of LLM outputs against provided context. If context is provided, measure whether the output remains within the context boundaries. Implement this metric in backend/src/eval_server/metrics/hallucination.py. Provide a complete implementation for hallucination detection; do not leave placeholder functions.

Expected outputs:

Hallucination metric module

Tests to add/run now:

Write tests (backend/tests/test_hallucination_metric.py) that supply context and outputs and verify that hallucination risk scores correspond to whether the output is grounded.

# [ ] Prompt038 — Implement refusal/safety compliance metric

Prompt:

Implement a metric that checks for safe behaviour, including refusal to provide prohibited information or perform restricted actions. Place this metric in backend/src/eval_server/metrics/safety.py. Read dataset definitions for prohibited actions and ensure the metric flags unsafe responses. Fully implement safety checks and produce appropriate scoring.

Expected outputs:

Safety compliance metric module

Tests to add/run now:

Write tests (backend/tests/test_safety_metric.py) that define prohibited actions and verify that responses violating safety rules are flagged and scored accordingly.

# [ ] Prompt039 — Implement regression detection metric

Prompt:

Create a regression detection metric that compares current scores against a baseline run. Implement this logic in backend/src/eval_server/metrics/regression.py. The metric should compute the delta between current and baseline scores and flag regressions. Provide a fully working implementation without leaving any parts unaddressed.

Expected outputs:

Regression metric module comparing runs and computing deltas

Tests to add/run now:

Write tests (backend/tests/test_regression_metric.py) that compare sample baseline and new runs and verify that regressions are detected and properly reported.

# [ ] Prompt040 — Implement custom metric plugin mechanism

Prompt:

Allow custom metric plugins to be added via a Python entrypoint mechanism. Define an entrypoint group (e.g., eval_server.metrics) in pyproject.toml and implement plugin loading in backend/src/eval_server/metrics/loader.py. Provide an example plugin. Ensure the plugin system is fully implemented, discoverable, and operational.

Expected outputs:

Entry point definitions in pyproject.toml

Plugin loader module and example metric plugin

Tests to add/run now:

Write tests (backend/tests/test_metric_plugins.py) that register a custom metric plugin and verify that it is discovered and executed correctly.

# [ ] Prompt041 — Implement threshold rules for per‑turn pass/fail

Prompt:

Implement threshold rules to mark a turn as pass/fail based on metric scores and weights. Place this logic in backend/src/eval_server/scoring/thresholds.py. Use configurable thresholds per metric. Fully implement threshold evaluation; incomplete logic is not acceptable.

Expected outputs:

Threshold evaluation module for per‑turn scores

Tests to add/run now:

Write tests (backend/tests/test_thresholds.py) that define thresholds and verify that turns are marked pass or fail correctly based on computed scores and weights.

# [ ] Prompt042 — Implement threshold rules for conversation pass/fail

Prompt:

Extend the threshold mechanism to evaluate conversation‑level pass/fail using aggregated metrics across turns. Implement this in backend/src/eval_server/scoring/thresholds.py or a new module. Provide complete conversation‑level threshold logic.

Expected outputs:

Conversation‑level threshold evaluation functions

Tests to add/run now:

Write tests (backend/tests/test_conversation_thresholds.py) that provide aggregated metric scores and verify that conversation pass/fail outcomes are computed based on thresholds.

# [ ] Prompt043 — Implement configurable aggregation functions for conversation scoring

Prompt:

Provide aggregation functions (mean, minimum, weighted average) configurable per metric for conversation‑level scoring. Implement these functions in backend/src/eval_server/scoring/aggregation.py and integrate them into the conversation scoring module. Ensure all aggregation functions are implemented and selectable via run configuration.

Expected outputs:

Aggregation functions implemented and integrated with scoring

Tests to add/run now:

Write tests (backend/tests/test_aggregation.py) that verify mean, minimum, and weighted averages are computed correctly for sample score sets.

# [ ] Prompt044 — Persist evaluation artifacts and outputs

Prompt:

Implement persistence of evaluation artifacts (inputs, outputs, scores, logs) in a run‑specific folder structure under runs/{run_id}. Ensure that for each run all relevant artifacts are stored, including raw outputs, normalized records, score reports, and logs. Implement complete file writing and directory management; there should be no missing files or uninitialized directories.

Expected outputs:

Directory structure under runs/{run_id} storing all evaluation artifacts

Tests to add/run now:

Write tests (backend/tests/test_artifact_persistence.py) that run an evaluation and verify that all expected artifact files are created with correct content.

# [ ] Prompt045 — Persist machine‑readable results in JSON

Prompt:

Generate a machine‑readable results file in JSON format at the end of each run, capturing per‑turn outputs, metrics, and aggregated results. Implement this in backend/src/eval_server/reporting/results_writer.py. Ensure the JSON structure is fully specified and that all fields are written.

Expected outputs:

JSON results file for each run

Tests to add/run now:

Write tests (backend/tests/test_results_writer.py) that run an evaluation and verify that the JSON file exists and contains all required data.

# [ ] Prompt046 — Persist tabular results export in CSV

Prompt:

Create a CSV export of results per run for analysts and reporting tools. Implement this in backend/src/eval_server/reporting/csv_export.py. Include columns for run ID, conversation ID, turn ID, metrics, scores, pass/fail flags, and other relevant metadata. Provide a complete CSV export implementation.

Expected outputs:

CSV export file per run

Tests to add/run now:

Write tests (backend/tests/test_csv_export.py) that run an evaluation and verify that the CSV export exists, has the correct columns, and contains expected data.

# [ ] Prompt047 — Generate human‑readable evaluation reports

Prompt:

Implement report generation in backend/src/eval_server/reporting/report_generator.py that produces human‑readable reports (e.g., HTML or Markdown) summarizing each conversation with transcripts, model outputs, scores, pass/fail flags, and differences between expected and actual outputs. Include charts or tables as appropriate. Provide a full report generation implementation – do not omit any required sections.

Expected outputs:

HTML or Markdown report per run

Tests to add/run now:

Write tests (backend/tests/test_report_generator.py) that run a small evaluation and verify that the report file exists and contains expected sections and data.

# [ ] Prompt048 — Generate evaluator‑annotated reports with rating, notes, and overrides

Prompt:

Extend the report generator to include evaluator fields such as rating, notes, and override pass/fail decisions. The report should display these fields and integrate human overrides into the scoring summary. Implement full support for including evaluator annotations in reports.

Expected outputs:

Report generation module updated to include evaluator annotations

Tests to add/run now:

Write tests (backend/tests/test_report_generator_annotations.py) that simulate evaluator feedback and verify that the annotations appear in the report and influence re‑scoring appropriately.

# [ ] Prompt049 — Provide UI workflow to capture and store evaluator feedback

Prompt:

Create a UI workflow in the frontend that allows human evaluators to enter ratings, notes, and override decisions per turn. On submission, the feedback should be stored in a structured file linked to the run ID. Integrate with the backend endpoint for feedback submission. Fully implement the feedback UI and backend integration; do not leave any part incomplete.

Expected outputs:

Frontend component to enter evaluator feedback

Backend API call to submit feedback

Tests to add/run now:

Write UI tests that simulate entering feedback and verify that it is sent to the backend and stored correctly.

# [ ] Prompt050 — Support re‑scoring using human overrides

Prompt:

Implement logic in the scoring system to incorporate evaluator overrides while keeping original auto‑scores immutable. When an override is present, it should update pass/fail results and aggregated scores but record the original auto‑score separately. Fully implement re‑scoring logic and ensure that both original and overridden scores are available.

Expected outputs:

Re‑scoring functionality integrated into scoring modules

Tests to add/run now:

Write tests (backend/tests/test_rescoring.py) that simulate evaluator overrides and verify that pass/fail statuses and aggregated scores are updated while original scores remain stored.

# [ ] Prompt051 — Implement golden update workflow

Prompt:

Create a golden update workflow that allows new golden references to be proposed from approved human‑reviewed outputs. Provide a script or tool in scripts/update_golden.py that compares evaluator‑approved outputs with existing goldens and suggests updates. Include mechanisms for approval and version bumping. Deliver a complete update workflow, including CLI and integration with dataset versioning.

Expected outputs:

Script or tool for proposing and approving golden updates

Updated dataset files with new versions when goldens are updated

Tests to add/run now:

Write tests (backend/tests/test_golden_update.py) that propose updates, approve them, and verify that new golden files are created with incremented versions.

# [ ] Prompt052 — Implement audit logging of evaluation activity

Prompt:

Implement audit logging that records who ran evaluations, with timestamps and configuration fingerprints. Store logs in a file or database table. Integrate logging into the orchestrator and API endpoints. Provide a complete audit log implementation – do not omit any part of logging.

Expected outputs:

Audit log file or database table capturing evaluation runs

Integration with orchestrator and API endpoints

Tests to add/run now:

Write tests (backend/tests/test_audit_logging.py) that run evaluations and verify that audit logs are created with correct entries.

# [ ] Prompt053 — Expose REST API layer for backend capabilities

Prompt:

Expose a REST API layer using FastAPI for all backend capabilities (datasets, runs, jobs, results, reports, feedback, comparisons). Organize endpoints under /api/v1/. Implement routers in backend/src/eval_server/api/ for each resource. Fully implement each endpoint with appropriate request/response models and error handling – do not leave stub endpoints.

Expected outputs:

FastAPI routers for datasets, runs, jobs, results, reports, feedback, and comparisons

API server integrated into the backend package

Tests to add/run now:

Write API tests (backend/tests/test_api.py) that call each endpoint using httpx and verify correct responses, including error cases.

# [ ] Prompt054 — Implement dataset listing API endpoint

Prompt:

Create an API endpoint /api/v1/datasets that lists available datasets and their metadata (name, version, tags, difficulty). Query the dataset repository and return a JSON array. Fully implement the endpoint with pagination if necessary.

Expected outputs:

GET /api/v1/datasets endpoint returning dataset metadata

Tests to add/run now:

Write tests (backend/tests/test_api_datasets.py) that call the endpoint and verify that the response includes expected dataset entries and metadata.

# [ ] Prompt055 — Implement API to fetch conversation and golden data by ID

Prompt:

Create an API endpoint /api/v1/datasets/{dataset_id}/conversations/{conversation_id} that returns the conversation data along with associated golden entries. Fully implement route handlers and error handling for missing IDs. Ensure that the endpoint returns complete conversation and golden information.

Expected outputs:

GET endpoint returning conversation and golden details by ID

Tests to add/run now:

Write tests (backend/tests/test_api_conversation.py) that request valid and invalid conversation IDs and assert responses or error messages accordingly.

# [ ] Prompt056 — Implement API to start a new evaluation run

Prompt:

Implement a POST endpoint /api/v1/runs that accepts a run configuration and starts a new evaluation run using the orchestrator. Return the generated run ID and job status. Fully implement run creation, including validation and queueing.

Expected outputs:

POST endpoint to create a new run and return run ID

Tests to add/run now:

Write tests (backend/tests/test_api_start_run.py) that post valid and invalid run configurations and verify correct run creation or errors.

# [ ] Prompt057 — Implement API to poll job progress

Prompt:

Implement an endpoint /api/v1/runs/{run_id}/progress that returns per‑conversation status and progress for a given run. Support polling or streaming (e.g., WebSocket) to update clients. Provide full implementation of progress reporting, including handling of run completion, failure, and cancellation.

Expected outputs:

Endpoint returning job progress and statuses

Tests to add/run now:

Write tests (backend/tests/test_api_progress.py) that run evaluations and poll progress, verifying that statuses update over time.

# [ ] Prompt058 — Implement API to retrieve per‑turn outputs and metric breakdowns

Prompt:

Create an endpoint /api/v1/runs/{run_id}/conversations/{conversation_id} to retrieve per‑turn outputs, scores, and metric breakdowns for a conversation in a given run. Return JSON with raw outputs, normalized outputs, scores, and thresholds. Fully implement the endpoint and ensure data is returned accurately.

Expected outputs:

GET endpoint returning per‑turn outputs and scores

Tests to add/run now:

Write tests (backend/tests/test_api_metrics.py) that run an evaluation and retrieve metric breakdowns, verifying that all data matches stored results.

# [ ] Prompt059 — Implement API to download run artifacts (JSON, CSV, report)

Prompt:

Provide an endpoint /api/v1/runs/{run_id}/artifacts that allows downloading run artifacts such as JSON results, CSV exports, and human‑readable reports. Support query parameters to select which artifact(s) to download. Fully implement artifact retrieval and file streaming.

Expected outputs:

Endpoint to download evaluation artifacts

Tests to add/run now:

Write tests (backend/tests/test_api_artifacts.py) that request each artifact type and verify that the downloaded files match those stored on disk.

# [ ] Prompt060 — Implement API to submit human evaluation feedback

Prompt:

Create a POST endpoint /api/v1/runs/{run_id}/feedback that accepts evaluator feedback for specific conversations and turns. Validate the input and store feedback appropriately. Implement full input validation and error handling – do not leave feedback processing incomplete.

Expected outputs:

Endpoint accepting and storing evaluator feedback

Tests to add/run now:

Write tests (backend/tests/test_api_feedback.py) that submit valid and invalid feedback and verify correct storage or error responses.

# [ ] Prompt061 — Implement API to compare two runs and return regression summaries

Prompt:

Implement a GET endpoint /api/v1/runs/compare that accepts parameters for two run IDs, compares their metrics, and returns regression summaries and deltas by dataset and metric. Provide a complete comparison implementation.

Expected outputs:

API endpoint returning regression summaries

Tests to add/run now:

Write tests (backend/tests/test_api_compare.py) that compare two sample runs and verify that the delta calculations are correct.

# [ ] Prompt062 — Generate OpenAPI/Swagger documentation automatically

Prompt:

Configure the FastAPI application to generate OpenAPI/Swagger documentation automatically. Ensure that all endpoints include correct request and response models and that the docs are accessible via /docs. Provide a fully documented API – no endpoint should lack description or models.

Expected outputs:

OpenAPI documentation accessible from the running server

Tests to add/run now:

Write a test (backend/tests/test_openapi.py) that retrieves the OpenAPI JSON and asserts that all endpoints are documented with correct schemas.

# [ ] Prompt063 — Implement backend validation and error handling

Prompt:

Ensure that all backend modules perform input validation and error handling, returning consistent error codes and messages for the frontend. Define a set of custom exception classes and an error response schema. Integrate error handlers into the FastAPI application. Fully implement validation and error handling across the entire backend.

Expected outputs:

Custom exception classes and error response schemas

Global error handlers integrated with FastAPI

Tests to add/run now:

Write tests (backend/tests/test_error_handling.py) that trigger various error conditions and verify that the API returns consistent error responses and statuses.

# [ ] Prompt064 — Secure secret configuration via environment variables

Prompt:

Ensure that sensitive configuration values (API keys, database URLs, secret keys) are never stored in datasets or code. Load them from environment variables via the settings module. Provide a .env.example file listing all required secrets. Document how to set these variables in deployment environments. Audit the repository to confirm no secrets are committed. Fully implement secret loading and documentation – do not leave any secret undefined.

Expected outputs:

.env.example specifying all secret keys and API keys

Settings module reading secrets from environment variables only

Documentation instructing developers where to set secrets

Tests to add/run now:

Tests (backend/tests/test_secrets.py) verifying that attempting to load the application without required environment variables raises an informative error

Test that secrets are not accidentally persisted in dataset or results files

Frontend user interfaces
# [ ] Prompt065 — Create React UI for browsing datasets, selecting models, choosing metrics, and starting runs

Prompt:

Develop a Datasets page in the frontend that lists available datasets by calling the /api/v1/datasets endpoint. Provide controls to filter by domain and difficulty. Create a Run Setup page that allows the user to select one or more datasets from the list, choose an LLM provider, choose metric bundles, configure truncation and concurrency, and start a run. On submission, send the run configuration to the backend. Display a success message with the run ID and link to the run dashboard. Fully implement these pages and forms using React and Tailwind – no placeholders or incomplete components.

Expected outputs:

Dataset listing page with filters and selection controls

Run setup form that posts to the run creation endpoint

Tests to add/run now:

Frontend integration tests using a testing library (e.g., React Testing Library) that mock the API and verify the dataset list is fetched and displayed

Test that filling out the run setup form and submitting sends the correct payload and handles the response

# [ ] Prompt066 — Create React UI for monitoring run progress

Prompt:

Build a Run Dashboard page that displays all active and completed runs with their status, progress, and key metadata. Use polling or WebSocket connections to update progress in real time via the /api/v1/runs/{run_id}/progress endpoint. Provide options to cancel a run. Clicking a run should navigate to a detailed view (implemented in a later prompt). Fully implement the dashboard and progress updates – no stubbed components.

Expected outputs:

Dashboard listing runs with real‑time status updates

Cancel button connected to the cancellation endpoint

Tests to add/run now:

UI tests ensuring that run statuses update over time and that cancelling a run sends a request and updates the status

Test that runs disappear from the active list when completed or cancelled

# [ ] Prompt067 — Create React UI for viewing conversation transcripts with expected vs actual responses

Prompt:

Develop a Conversation Detail page that displays a conversation transcript with each turn showing the user message, assistant response, expected responses, and evaluation scores. Call the /api/v1/runs/{run_id}/conversations/{conversation_id} endpoint to retrieve data. Use a tabbed or accordion layout to collapse/expand sections. Highlight differences between expected and actual responses (using diff information computed by the backend). Ensure that large transcripts remain performant. Provide a complete implementation of the detail page with diff highlighting and performance considerations.

Expected outputs:

Conversation detail page with side‑by‑side expected vs actual responses and scores

UI elements to expand/collapse turns for readability

Tests to add/run now:

Integration tests mocking the endpoint response and verifying that the page renders correct information for each turn

Test that diff highlights appear where expected and not when responses match exactly

# [ ] Prompt068 — Create React UI for viewing metric breakdowns and thresholds

Prompt:

Create a Metrics Breakdown page that visualizes per‑metric scores across turns and conversations. Fetch aggregated metric data from the backend and render it using charts (e.g., bar or radar charts) and tables. Provide controls to filter by metric or conversation. Display configured thresholds and highlight metrics that fall below the threshold. Allow exporting the view as CSV or image. Implement the complete visualization page with interactive features and export functionality.

Expected outputs:

Metrics breakdown page with interactive charts and tables

Threshold indicators integrated into the visualization

Tests to add/run now:

UI tests verifying that metric data is rendered correctly and that threshold indicators appear when scores are below configured values

Test exporting the view and ensure the downloaded file matches the on‑screen data

# [ ] Prompt069 — Create React UI for entering and saving human evaluator ratings and notes

Prompt:

Develop an Evaluation Feedback component that appears on the conversation detail page. For each turn, display fields to enter a numeric rating, free‑text notes, and an override pass/fail checkbox. When the evaluator submits feedback, send it to the /api/v1/runs/{run_id}/feedback endpoint. Show confirmation messages and disable submission until all required fields are filled. Preserve unsaved feedback locally to prevent data loss when navigating. Fully implement the feedback component with state management and API integration.

Expected outputs:

Feedback form integrated into conversation detail pages

API calls to submit feedback for each turn

Tests to add/run now:

UI tests verifying that feedback can be entered and submitted, and that submission triggers a backend call

Test that unsaved feedback persists in component state when navigating between turns

# [ ] Prompt070 — Create React UI for comparing runs and visualizing regressions

Prompt:

Implement a Run Comparison page where users can select two runs and view a comparison summary. Fetch comparison data from the /api/v1/runs/compare endpoint and render tables or charts showing score deltas per metric and dataset. Allow filtering by dataset tag or metric. Provide a fully functional comparison page – do not leave incomplete charts or controls.

Expected outputs:

Run comparison page with visualized metric deltas

Tests to add/run now:

UI tests verifying that the comparison data is displayed correctly and that filtering works as intended

# [ ] Prompt071 — Provide frontend support for downloading artifacts and reports

Prompt:

Add functionality to the frontend that allows users to download JSON results, CSV exports, and human‑readable reports from the backend via the /api/v1/runs/{run_id}/artifacts endpoint. Provide download buttons on the run detail and reports pages. Fully implement the download feature with progress indication and error handling.

Expected outputs:

Download buttons integrated into the frontend

Functionality to retrieve and download artifacts

Tests to add/run now:

UI tests verifying that clicking download buttons retrieves files and that error messages are shown on failure

# [ ] Prompt072 — Provide end‑to‑end tests for the full UI flow

Prompt:

Write end‑to‑end tests using a tool like Playwright or Cypress that automate the full flow: selecting a dataset, configuring a run, monitoring progress, viewing results, entering feedback, and downloading reports. These tests should run against the real backend and verify that the full system works as intended. Provide fully implemented end‑to‑end tests – do not leave any step untested.

Expected outputs:

E2E test suite covering the full evaluation flow

Tests to add/run now:

Execute the E2E test suite and ensure all tests pass against a fresh instance of the application

# [ ] Prompt073 — Provide backend unit tests for critical components

Prompt:

Develop a comprehensive suite of backend unit tests covering dataset validation, loader functions, turn runner, scoring metrics, report generation, and API endpoints. Aim for high coverage and include tests for edge cases and failure conditions. Provide fully implemented tests, not stubs or placeholders.

Expected outputs:

Test suite covering critical backend modules

Tests to add/run now:

Run the test suite using pytest and ensure all unit tests pass with high coverage

# [ ] Prompt074 — Provide a sample dataset with at least one 5‑turn commerce conversation

Prompt:

Create a sample dataset and golden reference demonstrating at least one 5‑turn commerce conversation. Include metadata tags and difficulty, multiple acceptable variants, structured fields, and constraints. Store the dataset in datasets/sample/ and reference it in example run configurations. Fully implement the dataset and golden files – they must conform to the schemas.

Expected outputs:

Sample conversation and golden files with complete metadata and expected responses

Tests to add/run now:

Write a test (backend/tests/test_sample_dataset.py) that loads the sample dataset using the loader and validates it against the schema

# [ ] Prompt075 — Provide a sample run configuration evaluating multiple models on the same dataset

Prompt:

Create a sample run configuration file in configs/runs/sample_multi_model.yaml that evaluates the sample dataset from # [ ] Prompt074 against multiple models. Specify different metric bundles, truncation policies, and concurrency. Ensure that the configuration is valid and documented. Provide a fully working sample configuration – do not leave missing fields.

Expected outputs:

Sample multi‑model run configuration

Tests to add/run now:

Write a test (backend/tests/test_sample_run_config.py) that loads the configuration and verifies that it is valid and includes multiple models

# [ ] Prompt076 — Provide a single command to start backend and frontend services

Prompt:

Create a Makefile or script that starts both the backend and frontend services concurrently for local development. The make target (e.g., make dev) should start the FastAPI server and the React development server with hot reloading. Ensure that environment variables are loaded correctly for both services. Deliver a fully working make target or script – do not leave any steps manual.

Expected outputs:

Makefile with targets to start backend and frontend concurrently

Tests to add/run now:

Write a test (scripts/test_make_dev.sh) that runs the make target and verifies that both servers start and are accessible at their respective ports

# [ ] Prompt077 — Provide JSON Schema definitions for datasets, goldens, and run configurations

Prompt:

Create machine‑readable JSON Schema (or equivalent) definitions for conversation datasets, golden references, and run configuration files. Store them under configs/schemas/. Update all loaders and validators to use these schemas. Fully implement the schemas with complete field definitions and types.

Expected outputs:

JSON Schema files for datasets, goldens, and run configurations

Updated validators using the schemas

Tests to add/run now:

Write tests (backend/tests/test_schemas.py) that load the schemas and validate sample data against them, ensuring correctness and completeness

# [ ] Prompt078 — Allow task_type metadata in datasets

Prompt:

Allow each dataset to specify a task_type (e.g., qa, classification, policy_decision, tool_use, rag_qa) in its metadata. Use this metadata to group and analyze evaluations. Update the dataset schema and provide helpers to filter datasets by task_type. Fully implement task type handling and ensure filters operate correctly.

Expected outputs:

Updated dataset schema with task_type

Filtering helpers based on task type

Tests to add/run now:

Write tests (backend/tests/test_task_type.py) that add datasets with different task types and verify grouping and filtering functions

# [ ] Prompt079 — Provide reusable metric bundles per task_type

Prompt:

Provide a configuration mechanism to define reusable metric bundles per task_type. Store metric bundle definitions in configs/metrics/, and allow run configuration files to reference bundles by name. Load bundles in the orchestrator. Fully implement bundle loading and referencing.

Expected outputs:

Metric bundle configuration files

Code to load and apply metric bundles based on task_type

Tests to add/run now:

Write tests (backend/tests/test_metric_bundles.py) that define metric bundles and verify that run configurations referencing them load the correct metrics

# [ ] Prompt080 — Implement per‑provider rate limit configuration

Prompt:

Provide per‑provider configuration of rate limits, including maximum requests per time window and concurrency caps, so that the evaluation runner can respect external API limits. Implement a rate limiting module in backend/src/eval_server/providers/rate_limiter.py. Integrate the limiter with the orchestrator and provider adapters. Fully implement rate limiting; partial stubs are not acceptable.

Expected outputs:

Rate limiter implementation

Integration with provider calls

Tests to add/run now:

Write tests (backend/tests/test_rate_limiter.py) that simulate high request volumes and verify that the rate limiter enforces limits correctly

# [ ] Prompt081 — Implement per‑provider retry policies

Prompt:

Add support for a retry policy configuration per provider that supports exponential backoff and distinguishes between transient errors (e.g., timeouts, 5xx) and permanent errors (e.g., invalid API key, 4xx). Implement this in backend/src/eval_server/providers/retry.py and integrate it with provider calls. Implement retry logic completely, handling backoff and error classification.

Expected outputs:

Retry policy implementation

Integration with provider calls

Tests to add/run now:

Write tests (backend/tests/test_retry_policy.py) that simulate transient and permanent errors and verify that the retry mechanism behaves correctly

# [ ] Prompt082 — Abstract storage behind pluggable adapters

Prompt:

Abstract run artifact storage behind a storage interface so that run outputs (JSON results, CSV exports, reports) can be persisted to local filesystem or remote object storage via pluggable adapters. Define a storage interface and implement a local filesystem adapter and a stub for remote storage. Fully implement the storage abstraction and the local adapter.

Expected outputs:

Storage interface definition and local filesystem adapter

Stub for remote storage adapter with clear extension points

Tests to add/run now:

Write tests (backend/tests/test_storage_adapter.py) that write and retrieve artifacts via the storage interface and verify data integrity

# [ ] Prompt083 — Provide versioned sample JSON and YAML templates

Prompt:

Provide clear, versioned sample JSON and YAML templates (with comments) for conversation datasets, golden references, and run configuration files in the repository’s configs and datasets folders. Include comments explaining each field. Fully implement the templates with correct versions and comprehensive comments.

Expected outputs:

Versioned JSON and YAML templates stored in the repository

Tests to add/run now:

Write tests (backend/tests/test_templates.py) that load each template, strip comments, and validate against the appropriate schema

Additional features and UI enhancements
# [ ] Prompt084 — Implement generic golden dataset generator

Prompt:

Develop a command‑line utility scripts/generate_golden_dataset.py that can produce generic conversation and golden reference pairs for multiple industries (commerce, healthcare, finance, support). Use domain‑specific templates and randomization to generate realistic conversations and expected responses. Allow specifying domain and number of conversations via command‑line arguments. Save generated datasets to datasets/generated/<domain>/ with appropriate metadata tags and versions. Implement the generator completely – generated datasets must conform to the schemas.

Expected outputs:

Dataset generation script producing domain‑specific datasets

Generated dataset files saved in the repository

Tests to add/run now:

Tests (backend/tests/test_dataset_generator.py) invoking the script with different domains and verifying that generated datasets conform to the schema and include correct tags

Test that running the generator multiple times produces unique versions

# [ ] Prompt085 — Create a dedicated golden set editor UI

Prompt:

Develop a Golden Editor page in the frontend that allows authorized users to create, edit, and delete golden entries. Provide a list of golden entries keyed by dataset, conversation ID, and turn ID. Implement forms for editing expected responses, structured fields, constraints, and weights. Display version history and change logs. Call the backend golden management endpoints to persist changes. Restrict access using a simple role check (e.g., local flag) to simulate authorization. Fully implement the editor UI and its integration with the backend; no stubbed actions are allowed.

Expected outputs:

Golden editor UI allowing CRUD operations on golden entries

Integration with backend golden APIs and version history display

Tests to add/run now:

UI tests verifying that golden entries can be created, edited, and deleted and that version numbers increment accordingly

Test that unauthorized access is blocked (if implemented)

# [ ] Prompt086 — Create a metrics management interface (UI)

Prompt:

Add a Metric Management page to the frontend where administrators can view, create, edit, activate, and deactivate metric definitions. List built‑in and custom metrics, show their weights and thresholds, and allow uploading new metric plugins. Provide forms to edit weights and thresholds. Call the backend metrics API for CRUD operations. Display validation errors returned by the server. Provide a fully functioning metrics management interface with complete CRUD operations.

Expected outputs:

Metrics management UI with full CRUD capabilities

Integration with backend metrics API endpoints

Tests to add/run now:

UI tests verifying that metrics can be created, edited, and toggled active/inactive and that changes are persisted

Test handling of invalid metric uploads and server‑side validation errors

# [ ] Prompt087 — Create dataset upload/import interface in the frontend

Prompt:

Develop a Dataset Upload page that allows users to upload new conversation datasets and golden references via drag‑and‑drop or file picker. Validate files client‑side (e.g., check JSON/YAML format) before submitting to the backend. On successful upload, call the dataset upload API, display validation results, and add the dataset to the available list. Show error messages if validation fails. Fully implement upload UI, validation, API integration, and feedback handling.

Expected outputs:

Dataset upload UI with file validation and API integration

User feedback on success or error

Tests to add/run now:

UI tests verifying that uploading valid files results in dataset creation and that invalid files are rejected with appropriate messages

Test that newly uploaded datasets appear in the dataset listing page

# [ ] Prompt088 — Implement NavBar and unified navigation

Prompt:

Create a responsive navigation bar component used across the frontend. The NavBar should include links to Datasets, Upload, Runs, Metrics, Golden Editor, Reports, Settings, and Docs. Implement active link highlighting and a collapsible menu for small screens. Use React Router to handle navigation. Place the NavBar inside a top‑level layout component so it appears on every page. Fully implement the NavBar with responsive design and route integration.

Expected outputs:

NavBar component with links and responsive design

Integration into the application layout

Tests to add/run now:

Component tests ensuring that NavBar renders all links and highlights the active route

Test that clicking each link navigates to the correct page without a full reload

# [ ] Prompt089 — Enhance evaluation run interface for selecting or importing datasets

Prompt:

Update the Run Setup page to allow users to select datasets from the existing list or import a new dataset on the fly via the upload interface. Provide a button or link to launch the dataset upload component. After uploading, automatically populate the dataset selector with the new dataset. Ensure that the run configuration includes newly uploaded datasets seamlessly. Fully implement these enhancements with dynamic refresh and integration of newly uploaded datasets.

Expected outputs:

Run setup page enhanced to support selecting or uploading datasets in a single workflow

Automatic refresh of dataset choices after upload

Tests to add/run now:

UI tests verifying that uploading a dataset during run setup adds it to the selection list and allows the run to proceed

Test that run configurations include newly uploaded datasets when submitted

# [ ] Prompt090 — Provide industry‑specific templates and documentation

Prompt:

Create domain‑specific documentation and templates under docs/templates/ for industries such as commerce, healthcare, finance, and support. Each document should describe typical conversation flows, common metrics, and example golden references. Provide starter templates for building datasets in these domains. Link these docs in the main README and in the generator script help messages. Fully implement the documentation and templates; they must be complete and accurate.

Expected outputs:

Industry‑specific documentation and templates stored in the repository

README references to these templates and guidance on creating new datasets

Tests to add/run now:

Validate that the templates conform to the dataset schema using a test (backend/tests/test_industry_templates.py)

Review documentation for completeness and accuracy