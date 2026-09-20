---
name: plan-locked-implementation
description: Implement an approved software plan through a gated, traceable loop so a coding agent changes only the current step, proves it with tests, and does not silently deviate. Use when a repository has a phased plan, specification, task list, or architecture contract.
---

# Plan-Locked Implementation

Treat the approved plan, not the coding agent's preferences, as the source of
truth. The goal is verified progress, not the largest possible code change.

## Establish control before feature work

Read the plan, repository instructions, current tests, and existing decision
records. Create or verify:

- a machine-readable step registry;
- prerequisites and allowed paths for each step;
- acceptance criteria;
- focused unit-test commands;
- regression-test commands;
- architecture and security checks;
- a requirement-to-evidence traceability record;
- status and verification artifacts.

The first implementation task should build this control layer when it does not
already exist.

## Select exactly one step

Before editing:

1. Choose the next eligible step.
2. Confirm every prerequisite passed.
3. Load only the relevant plan context.
4. List the files the step may change.
5. Identify the tests that prove the step.
6. Record unresolved decisions before implementation.

Do not start a later step because it appears easy or related. Do not expand the
allowed file set without an explicit decision.

## Required implementation loop

1. Inspect existing code and tests.
2. Implement the smallest change satisfying the current step.
3. Add or update focused unit tests.
4. Run the focused tests.
5. Run regression tests for every completed step.
6. Run architecture, security, dependency, scope, and size checks.
7. Compare the diff with the step and traceability record.
8. Record commands, exit codes, results, and evidence.
9. Request human review for contract, security, architecture, or plan changes.
10. Mark the step complete only after every required gate passes.
11. Unlock the next step only after completion is recorded.

The harness must execute the checks. A statement that a test passed is not
evidence unless the command ran and its result was captured.

## Non-deviation rules

Stop and report when:

- a test or regression check fails;
- an unapproved file changes;
- a requirement has no test or documented decision;
- an existing behavior changes without a compatibility test;
- a new dependency or integration is outside the plan;
- a security boundary, timeout, or permission rule is weakened;
- a file or function exceeds the repository limit;
- the plan is ambiguous or internally inconsistent.

Do not skip a failing test, weaken an acceptance criterion, hide an error,
silently change the plan, or implement speculative future functionality.

## Traceability record

Every completed step records:

- step ID and plan version;
- files changed;
- requirements addressed;
- tests and checks executed;
- pass/fail results and exit codes;
- compatibility impact;
- decisions and human approvals;
- known limitations;
- the next eligible step.

Keep records append-only when possible. Preserve failures rather than replacing
them with a later passing result.

## Review gates

Human approval is required before:

- changing a public contract or schema;
- changing metric meaning or thresholds;
- changing the runtime or security boundary;
- removing compatibility behavior;
- adding a new external framework or protocol;
- changing the approved phase or step order.

The agent may propose a plan change, but it must not apply that change and
continue implementation without approval.

## Definition of done

A step is complete when focused tests, regression tests, static checks,
architecture checks, security checks, file-size checks, documentation, and
traceability all pass.

A phase is complete only when all of its steps pass and the full regression
suite passes.

The implementation is complete only when the final plan checklist, end-to-end
tests, and release evidence pass. Report what was verified and what was not.


## Runtime-harness requirement

When the approved plan builds an AI or agent solution, verify that the plan
includes a runtime harness. At minimum, require:

- a session and target lifecycle contract;
- explicit turn, duration, retry, output, and stop limits;
- state, checkpoints, and structured event history;
- tool, network, permission, and secret boundaries;
- observability for latency, cost, outputs, artifacts, and failures;
- cancellation, recovery, replay, and bounded retry behavior;
- integration tests with fake targets.

If the runtime harness is missing, make it the next plan step before feature
work. Add its acceptance criteria, allowed paths, tests, and regression scope to
the same step registry. Do not mark the solution complete when only the coding
workflow is controlled but the running system is unbounded or unobservable.