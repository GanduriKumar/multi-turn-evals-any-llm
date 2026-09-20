# Enterprise MCP Toolkit: Implementation Phases

This document is a high-level phase map. The detailed execution order and gates
are defined by the following documents:

- docs/enterprise-mcp-granular-delivery-plan-part1.md
- docs/enterprise-mcp-granular-delivery-plan-part2.md
- docs/engineering-harness-and-loop-plan.md
- docs/runtime-harness-implementation-plan.md
- docs/reference-pack-builder-plan.md


## Phase 0: Baseline and compatibility

Tasks:

- Record current CLI and REST workflows.
- Select representative datasets, goldens, run artifacts, and reports.
- Add compatibility tests for loading datasets and reading reports.
- Record syntax and backend/frontend test results.
- Mark old modules as compatibility code in documentation.

Exit criteria:

- A known baseline command and result are recorded.
- One current run is a compatibility fixture.
- Every refactor has a replacement test or fixture.

## Phase 1: Built-in reference pack and use cases

Package versioned LLM Wiki and Google OKF reference files as read-only toolkit inputs. Implement the domain pack, dataset, golden, and validation contracts. Add
valid and invalid fixtures, including one non-commerce example.

Acceptance test:

- Load the built-in reference pack.
- Select a business use case and risk profile.
- Generate a dataset and golden set.
- Validate both without calling an agent.
- Export portable JSON with version and seed metadata.

## Phase 2: Dataset and golden services

Move reusable behavior from coverage_builder_v2.py and convgen_v2.py behind the
toolkit services. Keep commerce-specific rules in a selectable commerce profile and record the built-in reference revision and hash.

Do not combine generation with file storage, target execution, scoring, or
report rendering. Ensure generation can run without network access.

## Phase 3: Standalone-agent drivers

Implement A2A, HTTP, and local function drivers first. Add WebSocket, CLI, and
container drivers after the common target contract is stable.

Required failure behavior:

- connection failure,
- timeout,
- malformed response,
- partial A2A task,
- process crash,
- oversized output.

Acceptance test:

- Run a fake A2A target.
- Run a fake HTTP target.
- Run a local function.
- Confirm no evaluated agent contains toolkit-specific code.

## Phase 4: External evaluation adapters

Define the framework adapter interface and registry. Implement DeepEval and
EvalBench adapters as optional integrations. Keep the current metric code only
as a temporary compatibility adapter while equivalent coverage is verified.

Acceptance test:

- Execute a small fixture through each installed framework.
- Normalize each result.
- Report a structured missing-dependency error when a framework is unavailable.
- Confirm adapters do not write files directly.

## Phase 5: Run control, results, and storage

Split backend/orchestrator.py into run state, execution coordination,
persistence, and report requests. Create dataset, run, and artifact stores.

Required run states:

- queued
- running
- succeeded
- failed
- cancelled

Acceptance test:

- Start a run.
- Stop or restart the service.
- Read status and results from disk.
- Regenerate a report.
- Compare two stored runs without rerunning agents.

## Phase 6: MCP server

Expose these initial tools:

- generate_dataset
- validate_dataset
- list_datasets
- start_eval_run
- get_run_status
- get_run_results
- get_metrics_summary
- get_report
- compare_runs

Expose read-only resources for large datasets, reports, and result files.
Return identifiers and references instead of placing large artifacts in tool
responses.

Security requirements:

- allow-listed target hosts and commands,
- no arbitrary shell execution from untrusted requests,
- timeouts and output-size limits,
- request and run identifiers in every response,
- secrets from environment or secret configuration only.

Acceptance test:

- Generate, validate, execute, inspect, and compare through MCP only.
- Confirm structured errors for invalid input and unavailable dependencies.

## Phase 7: Frontend MCP test console

Keep frontend/ and replace direct backend assumptions with an MCP client
boundary. Required screens:

- MCP connection and server health,
- domain pack and dataset generation,
- dataset validation,
- agent target configuration,
- run creation and status,
- results and metrics,
- report and artifact retrieval,
- run comparison.

Frontend tests mock MCP operations. A local smoke test uses the real MCP
server. No frontend path calls private backend implementation modules.

## Phase 8: Compatibility cleanup

Only after the replacement flow passes:

- split backend/app.py into composition and compatibility routes,
- move shared request models out of route modules,
- remove duplicate metric calculations with equivalent tests,
- keep old artifact readers until migration is documented,
- add package READMEs describing ownership and public interfaces.

Do not combine this cleanup with a broad rename or formatting rewrite.

## Verification matrix

Unit tests cover schemas, generators, drivers, adapters, stores, and summaries.

Contract tests cover MCP inputs, outputs, structured errors, and normalized
results.

Integration tests cover fake A2A/HTTP/function targets and one installed
evaluation framework.

Frontend tests cover mocked MCP calls, loading states, failure states, and
report rendering. One end-to-end smoke test uses the real server.

## Definition of done

- Policies and use cases produce portable datasets and goldens.
- A2A, HTTP, and function targets work without agent modification.
- Execution is delegated through an evaluation-framework adapter.
- MCP exposes generation, execution, status, results, metrics, reports, and
  comparison.
- The frontend tests the public MCP interface end to end.
- Results survive a server restart.
- Core functions have one responsibility.
- Changed production files remain below the agreed size limit.
- Documentation and tests cover every public capability.

## First-release exclusions

- automatic discovery of every agent protocol,
- hidden chain-of-thought capture,
- a new custom evaluation language,
- distributed workers,
- multi-tenant billing and identity,
- a replacement analytics product.

These require evidence from real usage before being planned.
