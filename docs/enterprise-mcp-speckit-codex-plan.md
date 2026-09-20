# Enterprise Agent Evaluation Toolkit: Spec Kit + Codex Plan

Status: Proposed alternative delivery process

This document describes how to implement the migration with GitHub Spec Kit
and Codex. It is a process plan, not a second architecture. The architecture
and completion criteria remain in
docs/enterprise-mcp-implementation-plan.md.

Spec Kit provides a sequence of persistent artifacts: specify, clarify, plan,
tasks, implement, and converge. Codex uses the Spec Kit integration files under
.agents/skills and invokes the installed skills with the $speckit-* form.
Verify the local Spec Kit version before starting because command names and
installation details can change.

References:

- Spec Kit quickstart: https://github.github.com/spec-kit/quickstart.html
- Spec Kit Codex integration: https://github.github.com/spec-kit/reference/integrations.html
- Spec Kit core commands: https://github.github.com/spec-kit/reference/core.html
- Codex repository instructions: https://developers.openai.com/codex/guides/agents-md

## 1. Prepare the repository safely

Before initialization:

1. Create a branch for the migration.
2. Record the current test result and working tree state.
3. Read the existing README files and plans.
4. Decide where Spec Kit artifacts will live. Use the default specs/ layout
   unless the team already has a documented alternative.
5. Do not run a force initialization over uncommitted work without reviewing
   the files it will merge.

For an existing project, the expected initialization shape is:

  uv tool install specify-cli
  specify init --here --force --non-interactive --integration codex

Use the repository's approved Spec Kit version in CI. If installation is not
available, the workflow can be followed manually with the same artifacts, but
the commands below will not be available.

## 2. Establish the project constitution

Run the Spec Kit constitution step once and record the rules that Codex must
follow. Include these project-specific rules:

- MCP is the public control-plane interface.
- The frontend is an MCP test console, not a second backend.
- Standalone agents must not require code changes.
- A2A, HTTP, and local function targets are first-release requirements.
- External evaluation frameworks own execution and scoring where supported.
- Stored results use one versioned normalized schema.
- Functions and production files should stay below 200-240 lines.
- Each module has one responsibility and clear documentation.
- No speculative protocol, worker, or plugin abstractions.
- Every implementation task requires focused tests and a verification record.

Review the constitution manually. It is a quality gate, not a substitute for
technical design.

## 3. Divide the migration into feature specifications

Do not create one giant Spec Kit feature. Use small features with an explicit
dependency order:

1. 001-contracts-and-fixtures
2. 002-domain-pack-generation
3. 003-target-drivers
4. 004-evaluation-framework-adapters
5. 005-run-results-and-storage
6. 006-mcp-server
7. 007-frontend-mcp-console
8. 008-compatibility-cleanup

Each feature should be independently reviewable and should leave the repository
in a tested state. The first feature establishes contracts that later features
consume; it should not attempt to implement the entire system.

## 4. Repeat the Spec Kit loop for every feature

For each feature, use this sequence:

1. $speckit-specify: describe the user-visible capability and boundaries.
2. $speckit-clarify: resolve only questions that affect implementation or
   acceptance criteria. Record decisions in the spec.
3. $speckit-plan: map the feature to existing files, new modules, schemas,
   dependencies, and tests.
4. $speckit-checklist: create a review checklist for contracts, security,
   compatibility, documentation, and tests.
5. $speckit-tasks: produce small ordered tasks. Each task should be one
   responsibility and normally one focused change.
6. $speckit-analyze: check that the spec, plan, and tasks agree and that no
   task reintroduces duplicated ownership.
7. $speckit-implement: implement only the selected feature tasks.
8. $speckit-converge: run tests, inspect the diff, update the spec with actual
   behavior, and record unresolved risks.

Use the full quality-gate sequence for the contract, target-driver, framework,
and MCP features. A smaller loop is acceptable for documentation-only changes.

## 5. Feature-specific instructions

### Feature 001: contracts and fixtures

Specify schemas for the built-in LLM Wiki and OKF reference pack, domain profiles, datasets, goldens, target configurations,
run configurations, normalized results, and reports. Require valid and invalid
fixtures. Plan the migration from current commerce schemas without changing
existing readers until compatibility tests exist.

The acceptance test is a schema-only flow: load the built-in reference pack and select a use case, validate a
dataset, and read a normalized result without calling an agent.

### Feature 002: domain-pack generation

Specify generation from the built-in references, use cases, personas, and risks. Preserve
commerce behavior as a selectable profile. Require deterministic seeds, validation,
and JSON export. Explicitly exclude arbitrary enterprise source connectors, execution, report rendering, and automatic
policy discovery.

The acceptance test generates one commerce and one non-commerce fixture.

### Feature 003: target drivers

Specify the external driver contract and A2A, HTTP, and function targets first.
Add CLI/container drivers only after the common contract works. Define timeout,
malformed response, task status, and error behavior before implementation.

The acceptance test runs fake A2A and HTTP services plus a local callable. The
agents contain no toolkit-specific adapter code.

### Feature 004: framework adapters

Specify the adapter interface and normalized result mapping. Keep DeepEval and
EvalBench dependencies optional. Define behavior when a framework is missing or
returns an unsupported metric shape. Existing custom metrics become a temporary
compatibility adapter, not a third permanent runtime.

The acceptance test proves that two framework result shapes produce the same
normalized fields where their meanings are equivalent.

### Feature 005: results and storage

Specify run state, artifact layout, restart behavior, report regeneration, and
run comparison. Keep filesystem storage as the first implementation. Require
that stores do not validate business semantics or execute agents.

The acceptance test stops and restarts the service between execution and report
retrieval, then compares two stored runs.

### Feature 006: MCP server

Specify tools for dataset generation, validation, run execution, status,
results, metrics, reports, and comparisons. Specify resource behavior for large
artifacts and structured errors. Define allowed target hosts/commands,
timeouts, output limits, and secret handling.

The acceptance test performs the complete flow through MCP only.

### Feature 007: frontend MCP console

Specify the frontend as a client of the MCP contract. Include connection
health, dataset generation, target setup, run status, results, reports, and
comparison. Mock MCP in unit tests and use one real-server smoke test.

The acceptance test proves that the UI uses public MCP operations and does not
call private backend implementation modules.

### Feature 008: compatibility cleanup

Specify removal or isolation of old route and runtime paths only after all
replacement tests pass. Keep readers for existing artifacts and document any
intentional breaking changes.

The acceptance test runs the baseline fixture through the new path and confirms
that existing reports remain readable.

## 6. Codex execution discipline

For every Codex implementation session:

- Start by reading the active feature's spec.md, plan.md, and tasks.md.
- Read the nearest AGENTS.md and the package README before editing.
- Work on one task at a time.
- Inspect existing code before introducing a new helper.
- Use apply_patch for manual edits.
- Keep changes scoped to the active feature.
- Run the smallest relevant tests after each task.
- Run the full backend/frontend checks at feature convergence.
- Update task status and record test results in the feature artifacts.

If the implementation exposes an ambiguity, stop and update the feature
specification before writing a speculative abstraction. Do not solve future
protocols or providers just because a generic interface could be imagined.

## 7. Review gates before merging a feature

The reviewer should verify:

- Every acceptance criterion has an automated or explicitly documented test.
- Ownership is clear: generator, driver, adapter, result service, store, MCP,
  and frontend each have separate responsibilities.
- No large function hides multiple workflows.
- No new code requires modifying the evaluated agent.
- Errors, timeouts, and missing dependencies are visible to the caller.
- Documentation describes the public contract and actual behavior.
- Existing compatibility fixtures still pass.

If the feature cannot satisfy a gate, record the reason in converge output
and create a follow-up feature instead of silently weakening the requirement.

## 8. Suggested branch and commit structure

Use one branch per feature, or one branch for the migration with commits that
match the feature numbers. Prefer commits such as:

- contracts: add versioned dataset and result schemas
- generation: add domain-pack dataset service
- execution: add A2A and HTTP target drivers
- evaluation: add framework adapter boundary
- mcp: expose run and report tools
- frontend: consume MCP test-console contract

Avoid mixing a formatting rewrite with an architectural change. This makes
Codex output easier to review and makes rollback possible.

## 9. Spec Kit completion definition

The Spec Kit migration process is complete when:

- Every major capability has a persistent specification and plan.
- Tasks show what was implemented and what was verified.
- Convergence artifacts record test results and known limitations.
- The final code meets the architecture plan's definition of done.
- The frontend proves the public MCP flow.
- The repository explains how future contributors add a target driver or
  evaluation-framework adapter without duplicating existing logic.
