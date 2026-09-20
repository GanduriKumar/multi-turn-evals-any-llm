# Engineering Harness and Loop-Engineering Plan

Status: Proposed

This harness controls coding-agent changes to the repository. It is separate
from the runtime harness that evaluates external agents.

The approach uses one router, ordered stages, durable artifacts, traceable
handoffs, and human decisions at gates. This adapts the artifact-chain approach
described in the referenced article:
https://medium.com/@julian.oczkowski/18-claude-skills-that-turn-a-product-idea-into-a-launch-f96a32055800

## 1. Purpose

The engineering harness:

- selects one approved implementation step;
- loads only that step's context and allowed paths;
- runs unit, regression, static, and architecture checks;
- verifies the diff against the plan;
- records evidence;
- blocks the next step when a gate fails.

The evaluated-agent runtime harness is separate. It connects to A2A, HTTP,
function, CLI, or container agents and runs evaluation datasets.

## 2. Durable artifacts

Create a numbered engineering dossier:

  engineering/
    constitution.md
    plan-manifest.yaml
    step-status.json
    traceability.yaml
    verification/
    decisions/
    known-failures/

Each step record contains the step ID, acceptance criteria, allowed files,
tests, checks, decisions, evidence, and final status. Markdown is for humans;
JSON and YAML are for automated checks.

## 3. Router and loop

Provide one entry point:

  python -m backend.engineering_harness route --request "implement step 4.3"

The router confirms prerequisites, identifies the next eligible step, and
produces an implementation brief. It does not implement code or mark a step
complete.

Every step follows this loop:

1. Select one approved plan step.
2. Read acceptance criteria and allowed paths.
3. Inspect relevant code and tests.
4. Record open decisions.
5. Implement the smallest satisfying change.
6. Run focused unit tests.
7. Run regression tests for completed steps.
8. Run static, architecture, security, and file-size checks.
9. Compare the diff with the plan and traceability manifest.
10. Write a verification record.
11. Require the gate to pass.
12. Unlock the next step.

The harness executes commands itself and records exit codes. Agent claims are
not evidence.

## 4. Harness build steps

### H0.1: Manifest and registry

Record every implementation step with ID, prerequisites, allowed paths, tests,
regression scope, and acceptance criteria.

Gate: manifest tests pass and every granular-plan step has an entry.

### H0.2: Status and artifact store

Store status, verification records, decisions, failures, and evidence with
atomic writes.

Gate: round-trip, interrupted-write, and invalid-status tests pass.

### H0.3: Step router

Resolve only the next eligible step. Reject skipped prerequisites and active
step conflicts.

Gate: ordering, prerequisite, and blocked-step tests pass.

### H0.4: Test runner

Run focused unit tests, phase regression tests, and the full regression suite.
Record command, environment summary, exit code, duration, and result.

Gate: success, failure, missing-command, and timeout tests pass.

### H0.5: Scope and architecture checks

Check changed files, module ownership, imports, file/function size, duplicate
implementations, and dependency rules.

Gate: invalid fixtures fail and valid fixtures pass.

### H0.6: Traceability checker

Verify that every acceptance criterion maps to a test, static check, or human
decision, and every changed file is justified by the active step.

Gate: missing, stale, and complete mapping tests pass.

### H0.7: Gate command and CI

Provide:

  python -m backend.engineering_harness verify --step 4.3

The command runs all checks, writes evidence, and returns non-zero on failure.
CI and local runs must use the same manifest and commands.

Gate: passing fixtures complete; failing fixtures remain blocked.

## 5. Completion criteria

A step completes only when these applicable checks pass:

- no unapproved files changed;
- contracts match the plan;
- focused unit tests pass;
- all completed-step regression tests pass;
- architecture ownership remains intact;
- size and responsibility rules pass;
- security checks pass;
- documentation is updated;
- identical inputs reproduce the verification result;
- every requirement has evidence.

When a gate fails, preserve the output, mark the step blocked, do not start the
next step, and rerun the full gate after correction.

Human approval is required for public-contract changes, evaluation meaning or
threshold changes, security-boundary changes, architecture deviations,
compatibility removal, and acceptance-criteria changes.

The harness is complete when every granular step has a registry entry, the
router enforces order, checks execute automatically, evidence is stored, failed
steps block progression, and local and CI verification agree.
