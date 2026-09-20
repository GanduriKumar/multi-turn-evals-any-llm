# Enterprise MCP Toolkit: Granular Gated Delivery Plan (Part 2)

This document continues the sequential gated delivery plan. Complete Part 1 before starting these phases.

## Phase 5: Evaluation-framework adapters

### 5.1 Define the adapter interface

Accept observations, dataset cases, metric settings, and framework options.
Return normalized results without writing files.

Unit gate: adapter contract and error tests.
Regression gate: target-driver suites pass.

### 5.2 Add the DeepEval adapter

Map supported test cases and metrics to DeepEval. Keep the dependency optional.

Unit gate: mocked DeepEval result mapping.
Regression gate: normalized-result and driver suites pass.

### 5.3 Add the EvalBench adapter

Map supported test cases and metrics to EvalBench. Keep the dependency
optional and report missing installation clearly.

Unit gate: mocked EvalBench result mapping.
Regression gate: DeepEval and normalized-result suites pass.

### 5.4 Add framework selection

Register adapters by name and version. Reject unknown or unavailable
frameworks before an agent run starts.

Unit gate: registry and missing-dependency tests.
Regression gate: every adapter test passes.

Phase gate: at least one installed framework executes a fixture and produces
the normalized result format.

## Phase 6: Runs, results, and storage

### 6.1 Add dataset, run, and artifact stores

Implement filesystem read/write operations only. Stores do not execute agents or
apply business rules.

Unit gate: read, write, missing, and corrupt-artifact tests.
Regression gate: all pack and generation suites pass.

### 6.2 Add run state transitions

Implement queued, running, succeeded, failed, and cancelled states.

Unit gate: valid transitions and invalid-transition tests.
Regression gate: store and adapter suites pass.

### 6.3 Add the execution coordinator

Connect one dataset case, one target driver, and one framework adapter. Persist
normalized results.

Unit gate: mocked driver and adapter orchestration tests.
Regression gate: complete previous-phase suite passes.

### 6.4 Add report generation

Generate reports only from stored normalized results.

Unit gate: report regeneration and failure-summary tests.
Regression gate: existing report fixtures remain readable.

### 6.5 Add run comparison

Compare two stored runs without executing either agent again.

Unit gate: equal, improved, regressed, and incompatible-run tests.
Regression gate: result and report suites pass.

Phase gate: a run survives restart, produces a report, and can be compared.

## Phase 7: MCP server

### 7.1 Add server bootstrap

Start the MCP server and expose health and capability discovery.

Unit gate: server startup and capability-list tests.
Regression gate: service and storage suites pass.

### 7.2 Add dataset tools

Expose generation, validation, and dataset listing through MCP.

Unit gate: tool input, output, and structured-error tests.
Regression gate: generation flow passes through service APIs.

### 7.3 Add run tools

Expose run creation, status, results, and metric summaries.

Unit gate: tool-to-service contract tests.
Regression gate: complete run flow works without private REST routes.

### 7.4 Add reports and resources

Expose report retrieval, comparison, and large-artifact resources by reference.

Unit gate: resource URI and missing-artifact tests.
Regression gate: report and comparison suites pass.

### 7.5 Add execution safety

Enforce target allow-lists, timeouts, output limits, and secret references.

Unit gate: rejection and redaction tests.
Regression gate: all MCP and execution tests pass.

Phase gate: generate, execute, inspect, report, and compare through MCP only.

## Phase 8: Frontend MCP test console

### 8.1 Add the MCP client boundary

Create one frontend client for MCP calls. Do not call private backend modules.

Unit gate: mocked connection, success, loading, and error tests.
Regression gate: existing frontend tests pass.

### 8.2 Add configuration screens

Add reference-pack/use-case, target, framework, and run configuration screens.

Unit gate: form validation and request-payload tests.
Regression gate: MCP contract tests pass.

### 8.3 Add run and report screens

Add status, results, metrics, reports, artifacts, and comparison views.

Unit gate: loading, failure, empty, and success-state tests.
Regression gate: all frontend tests pass.

### 8.4 Add one real-server smoke flow

Run the frontend against the local MCP server for generation, execution, and
report retrieval.

Smoke gate: complete flow succeeds with a fake agent and fixture pack.

Phase gate: the frontend validates the same public MCP workflow used by
external clients.

## Phase 9: Cleanup and release

### 9.1 Isolate compatibility routes

Separate backend/app.py composition from old compatibility endpoints.

Unit gate: route and service tests.
Regression gate: all backend tests pass.

### 9.2 Remove proven duplication

Remove old custom runtime or metric paths only after equivalent framework and
normalized-result tests pass.

Unit gate: replacement tests cover removed behavior.
Regression gate: compatibility fixtures and full suite pass.

### 9.3 Enforce maintainability checks

Check file size, function responsibility, documentation, and ownership rules.

Unit gate: lint or static checks where available.
Regression gate: complete backend, frontend, and smoke suites pass.

### 9.4 Publish the implementation record

Document commands, configuration, supported targets, pack creation, MCP tools,
known limitations, and the final test results.

Release gate: every phase gate is recorded as passed and the staged diff is
reviewed.
