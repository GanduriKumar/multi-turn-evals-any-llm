# Enterprise MCP Toolkit: Granular Gated Delivery Plan

Status: Proposed

The engineering-harness H0 steps and runtime-harness R0-R7 steps are cross-cutting prerequisites. They must be registered in the same manifest and pass before dependent product steps proceed.

This document breaks the implementation into small sequential steps. A step
must pass its unit tests and regression tests before the next step starts. A
phase must pass its phase gate before the next phase starts.

## Delivery rules

For every step:

1. Read the step scope and existing ownership rules.
2. Implement only that step.
3. Add or update focused unit tests.
4. Run the regression suite for all completed steps.
5. Run formatting, type, and syntax checks where configured.
6. Record the result and review the diff.
7. Commit the step only after the gate passes.

Do not hide pre-existing failures. Record them in the baseline and confirm that
the step did not add new failures.

Every phase gate requires:

- all step gates in the phase to pass,
- the full regression suite to pass,
- no duplicate ownership introduced,
- documentation updated,
- production files within the 200-240 line limit.

## Phase 0: Baseline

### 0.1 Inventory current behavior

Record current CLI, REST, frontend, dataset, run, report, and provider flows.

Unit gate: test the inventory/config readers that already exist.
Regression gate: existing tests pass with no source changes.

### 0.2 Capture compatibility fixtures

Select representative datasets, goldens, run artifacts, reports, and failure
cases. Copy only stable test fixtures, not credentials or private data.

Unit gate: fixtures load through current readers.
Regression gate: current report and dataset tests pass.

### 0.3 Establish the test commands

Document backend unit, frontend unit, integration, and smoke-test commands.
Add a small test-runner note for environments missing optional dependencies.

Unit gate: each configured command can discover its intended tests.
Regression gate: the recorded baseline is reproducible.

Phase gate: baseline fixtures and test results are recorded.

## Phase 1: Contracts

### 1.1 Define reference-pack and domain-profile schemas

Define the built-in LLM Wiki and OKF pack manifest, use cases, personas, risks,
and generation configuration.

Unit gate: valid and invalid schema fixtures.
Regression gate: existing schema tests still pass.

### 1.2 Define dataset and golden schemas

Define scenarios, turns, variables, expected outcomes, constraints, and
acceptable response variants.

Unit gate: validation errors identify exact fields.
Regression gate: current datasets remain readable or have an explicit adapter.

### 1.3 Define target and run schemas

Define A2A, HTTP, function, CLI, and container target configuration, run
configuration, timeouts, limits, and framework selection.

Unit gate: every target type rejects missing required fields.
Regression gate: current run configuration fixtures remain readable.

### 1.4 Define normalized results and reports

Define observations, outputs, metrics, errors, artifacts, provenance, and
report references.

Unit gate: result round-trip serialization.
Regression gate: existing report readers remain functional.

Phase gate: all versioned contracts and fixtures pass.

## Phase 2: Reference-pack builder

### 2.1 Add source-location configuration

Accept local files/directories, Git locations, and HTTP(S) locations. This
utility creates the built-in pack; evaluation runs do not fetch live sources.

Unit gate: configuration validation and source-type selection.
Regression gate: repository configuration tests pass.

### 2.2 Add raw-source snapshotting

Read sources, preserve raw content, resolve revisions, and calculate hashes.

Unit gate: local source, missing source, duplicate source, and hash tests.
Regression gate: no existing dataset or artifact path changes.

### 2.3 Add provenance manifest

Write source URI, revision, hash, retrieval time, license note, and builder
version to a manifest.

Unit gate: manifest fields and deterministic serialization.
Regression gate: artifact persistence tests pass.

### 2.4 Generate LLM Wiki files

Create stable Markdown pages with identifiers, summaries, links, and citations
back to raw sources. Raw sources remain immutable.

Unit gate: page generation, citation, and stable-ID tests.
Regression gate: pack loading tests pass.

### 2.5 Generate OKF files

Create OKF records from the same extracted source records used by the wiki
writer. Do not create a second extraction pipeline.

Unit gate: required metadata and relationship tests.
Regression gate: reference-pack schema tests pass.

### 2.6 Validate and publish a pack

Validate links, metadata, references, and hashes. Reject partial or invalid
packs. Write a pack version and validation report.

Unit gate: valid, invalid, and partial-pack tests.
Regression gate: all builder tests and contract tests pass.

### 2.7 Verify offline consumption

Build a pack, disable network access, and use only that pack for generation.

Unit gate: offline pack-reader tests.
Regression gate: complete Phase 1 and Phase 2 suites pass.

Phase gate: a versioned LLM Wiki/OKF pack is reproducible and offline-readable.

## Phase 3: Dataset and golden generation

### 3.1 Add domain-profile loading

Load the built-in pack and select a business use case, persona, and risk
profile. Do not add arbitrary enterprise policy connectors.

Unit gate: profile selection and invalid-profile tests.
Regression gate: pack-builder and schema suites pass.

### 3.2 Add scenario planning

Convert selected use cases and risks into explicit scenario requirements.

Unit gate: deterministic planning tests.
Regression gate: previous generation fixtures remain stable.

### 3.3 Add dataset generation

Generate multi-turn scenarios without executing an agent or writing run
artifacts.

Unit gate: seed, count, metadata, and no-side-effect tests.
Regression gate: all contract and pack suites pass.

### 3.4 Add golden generation

Generate expected outcomes, acceptable variants, required actions, forbidden
actions, and metric hints.

Unit gate: golden validation and traceability tests.
Regression gate: generated datasets validate successfully.

### 3.5 Add dataset exporters

Export versioned JSON and the supported portable formats. Record pack
provenance and generation configuration.

Unit gate: export/import round-trip tests.
Regression gate: current dataset readers remain compatible.

Phase gate: a non-commerce and commerce profile produce valid datasets and
goldens without contacting an agent.

## Phase 4: Standalone-agent drivers

### 4.1 Define the driver interface

Define normalized turn requests, observations, errors, artifacts, and
observation modes.

Unit gate: interface and serialization tests.
Regression gate: schema and generation suites pass.

### 4.2 Add the function driver

Invoke a configured local callable without changing the callable.

Unit gate: success, exception, timeout, and malformed-result tests.
Regression gate: driver contract tests pass.

### 4.3 Add the HTTP driver

Support request and response field mappings, authentication references, and
timeouts.

Unit gate: fake HTTP success, error, timeout, and malformed-response tests.
Regression gate: function-driver and contract suites pass.

### 4.4 Add the A2A driver

Support agent-card lookup, messages, task status, streaming, artifacts, and
errors.

Unit gate: fake A2A task lifecycle tests.
Regression gate: all previous driver tests pass.

### 4.5 Add optional process drivers

Add CLI and container execution only after the common driver contract is
stable. Capture stdout, stderr, exit code, and timeout.

Unit gate: fake process success, failure, and timeout tests.
Regression gate: all target-driver tests pass.

Phase gate: fake A2A, HTTP, and function agents run without agent changes.
