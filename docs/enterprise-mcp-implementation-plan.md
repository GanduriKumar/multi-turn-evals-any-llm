# Enterprise Agent Evaluation Toolkit: Implementation Plan

Status: Proposed

This document defines the target architecture and the first implementation
sequence for converting the current multi-turn LLM evaluation application into
an enterprise toolkit exposed through MCP. The existing frontend remains as a
test console for the MCP server; it is not the owner of evaluation logic.

- docs/engineering-harness-and-loop-plan.md
- docs/runtime-harness-implementation-plan.md
- docs/enterprise-mcp-granular-delivery-plan-part1.md
- docs/enterprise-mcp-granular-delivery-plan-part2.md
- docs/reference-pack-builder-plan.md

Authoritative execution documents are the engineering-harness plan, runtime-harness plan, granular delivery plans, and reference-pack builder plan. The high-level phase document is a map only.

## 1. Target outcome

The toolkit ships with a versioned built-in reference pack based on LLM Wiki and Google OKF files. A user selects a business use case, persona, risk area, and
a connection to an existing agent. The toolkit generates datasets and goldens,
runs evaluations through selected evaluation frameworks, stores normalized
results, and exposes reports and metrics through MCP.

Supported agent entry points:

- A2A endpoint
- HTTP or WebSocket API
- Local Python function
- CLI process or executable
- Container command
- Browser-accessible application when no machine-readable interface exists

The evaluated agent must not be changed to install a toolkit adapter.

## 2. Current repository boundaries

Useful implementation pieces already exist, but responsibilities are mixed:

- backend/app.py is a large API composition point.
- backend/orchestrator.py combines job control, execution, scoring, and storage.
- backend/turn_runner.py assumes an LLM/provider-oriented execution model.
- backend/coverage_builder_v2.py and backend/convgen_v2.py contain useful
  generation behavior but are commerce-specific.
- backend/metrics.py, metrics_extra.py, and conversation_scoring.py own scoring
  that should be delegated or isolated behind framework adapters.
- backend/reporter.py and the report template can remain useful renderers after
  results are normalized.
- frontend/ remains, but becomes an MCP test client and demonstration UI.

Preserve the current application until replacement behavior passes compatibility
tests. Do not delete working modules during the early refactor.

## 3. Ownership rules

1. MCP handlers validate transport input, call one service, and return data.
2. Generators create datasets and goldens. They do not run agents or reports.
3. Target drivers communicate with agents. They do not score responses.
4. Framework adapters invoke one external framework. They do not own storage.
5. Result services normalize and summarize. They do not invoke agents.
6. Stores read and write artifacts. They do not apply business rules.
7. The frontend owns presentation state only and calls the MCP boundary.
8. One function has one clear reason to change.
9. Prefer sequential, understandable logic until measured demand requires more.
10. Split production files before they exceed approximately 200-240 lines.

## 4. Stable contracts

Add versioned contracts under configs/schemas/ for:

- domain_pack: reference profile, use cases, personas, risks, constraints, and
  tags.
- dataset: scenarios, turns, variables, and metadata.
- golden_case: expected outcomes, acceptable responses, required actions,
  forbidden actions, and metric hints.
- target_config: A2A, HTTP, function, CLI, container, or browser target.
- run_config: dataset, target, framework, metrics, timeouts, and limits.
- normalized_result: observations, scores, errors, traces, and artifacts.
- report_summary: run status, metrics, failures, and artifact references.

Use Pydantic models at application boundaries and JSON Schema for stored
artifacts. Include schema versions and actionable validation errors. The new
schemas must not require commerce-only fields.

The normalized result should preserve:

- run and scenario identifiers,
- input turns and final output,
- target type and framework name,
- metrics and pass/fail status,
- tool calls or artifacts when externally observable,
- timing, errors, and trace references.

## 5. Built-in reference boundary

The first release uses only the versioned LLM Wiki and Google OKF files shipped with the toolkit. It does not connect to arbitrary enterprise policy repositories or accept company policy documents as a source.

## 6. Target package layout

Create focused packages with one public responsibility per module:

  backend/engineering_harness/
  backend/runtime_harness/

  backend/toolkit/
    domain_pack.py
    dataset_generator.py
    golden_generator.py
    dataset_validator.py
    exporters.py

  backend/execution/
    target.py
    a2a_driver.py
    http_driver.py
    websocket_driver.py
    function_driver.py
    cli_driver.py
    container_driver.py

  backend/evaluation/
    framework.py
    deepeval_adapter.py
    evalbench_adapter.py
    framework_registry.py

  backend/results/
    result_schema.py
    result_normalizer.py
    metrics_summary.py
    report_service.py
    compare_service.py

  backend/storage/
    dataset_store.py
    run_store.py
    artifact_store.py

  backend/mcp/
    server.py
    dataset_tools.py
    run_tools.py
    report_tools.py
    resources.py

## 7. Generation responsibilities

The generation flow is:

1. Load and validate the built-in reference pack.
2. Select a business use case and risk profile.
3. Build scenario requirements from the selected profile.
4. Generate candidate cases.
5. Generate expected outcomes and golden constraints.
6. Validate every case.
7. Export the dataset without executing it.

Move reusable behavior from the current coverage and conversation modules
behind these services. Keep commerce as one selectable profile, not the generator's
hidden default. Add deterministic seeds and reference provenance. Do not add arbitrary source connectors,
automatic policy discovery or a plugin marketplace in the first release.

## 8. Standalone-agent execution

Every target driver receives a normalized turn request and returns an
observation containing response text, status, latency, errors, and artifacts.

- A2A discovers the agent card, sends messages, and handles task status and
  streaming.
- HTTP and WebSocket drivers use configured request and response mappings.
- Function drivers call a local callable with a documented input contract.
- CLI and container drivers capture stdout, stderr, exit codes, and timeouts.
- Browser execution is a last-resort driver.

Support black-box output, externally observed output with logs/files/network
events, and optional instrumentation already provided by the agent owner.
Never claim to measure hidden reasoning or hidden tool calls.

## 9. Evaluation and storage boundary

An evaluation adapter receives a dataset, observations, and metric settings. It
invokes DeepEval, EvalBench, or another selected framework and maps its output
to normalized results. Framework dependencies remain optional, and missing
frameworks produce clear errors.

Use the filesystem store first. Keep JSON as the source of truth; generate CSV,
HTML, and PDF as derived artifacts. Record framework, target, schema versions,
timestamps, and configuration hashes. Reports must be regenerable from stored
normalized results, and run comparisons must not rerun agents.

The engineering harness H0.1-H0.8 and runtime harness R0-R7 are prerequisites for the granular delivery gates.
Use the granular delivery plans and harness plans listed at the start of this document.
