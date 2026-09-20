# Getting Started: Build the Toolkit with Its Harness

## Purpose

This guide is the starting point for implementing the enterprise evaluation
toolkit. The toolkit must contain a runtime harness for evaluating external
agents. The repository must also contain an engineering harness that controls
how the coding agent implements the plan.

The coding agent must follow the plan in small steps. It must not implement
later features early, skip tests, or silently change the architecture.

## Read these documents first

1. enterprise-mcp-implementation-plan.md
2. runtime-harness-implementation-plan.md
3. engineering-harness-and-loop-plan.md
4. enterprise-mcp-granular-delivery-plan-part1.md
5. enterprise-mcp-granular-delivery-plan-part2.md
6. reference-pack-builder-plan.md

The older MVP and commerce plans are historical. They are useful for
understanding existing behavior, but they are not the new implementation plan.

## The two harnesses

### Runtime harness

This becomes part of the final toolkit. It makes external agents:

- connectable through A2A, HTTP, functions, CLI, or containers;
- bounded by time, turn, retry, and output limits;
- observable through events, traces, artifacts, and errors;
- safe through allow-lists, isolation, and secret handling;
- recoverable through checkpoints and durable run state;
- replayable without contacting the live agent.

### Engineering harness

This controls the coding agent. It:

- selects one approved step;
- limits the allowed files;
- runs unit and regression tests;
- checks architecture and security rules;
- verifies requirement-to-test traceability;
- records evidence;
- blocks the next step when a gate fails.

## Correct starting order

### Step 1: Build the engineering harness

Implement H0.1 through H0.8 from the engineering harness plan:

- plan manifest and step registry;
- status and artifact store;
- step router;
- test runner;
- scope and architecture checks;
- traceability checker;
- gate command;
- CI and Codex integration.

Do not start the evaluation toolkit features until these gates pass.

### Step 2: Define the runtime harness

Implement R0 through R7 from the runtime harness plan:

- session contracts;
- target lifecycle;
- orchestration limits;
- state and event logs;
- security boundaries;
- observability and framework handoff;
- durable asynchronous execution;
- MCP and frontend integration.

The runtime harness is product functionality. It is not only a test utility.

### Step 3: Follow the granular implementation plan

Continue in order through the granular phases:

1. Baseline and compatibility
2. Contracts
3. LLM Wiki and OKF reference-pack builder
4. Dataset and golden generation
5. Standalone-agent drivers
6. Evaluation-framework adapters
7. Runs, results, and storage
8. MCP server
9. Frontend MCP test console
10. Cleanup and release

## Required loop for every step

For each step:

1. The harness selects exactly one eligible step.
2. The coding agent reads only that step's requirements and allowed paths.
3. The coding agent inspects existing code before editing.
4. The coding agent implements the smallest required change.
5. Focused unit tests are added or updated.
6. Focused unit tests are run by the harness.
7. All previous regression tests are run by the harness.
8. Architecture, security, size, and scope checks are run.
9. Every requirement is mapped to evidence.
10. A verification record is written.
11. A human reviews contract, security, and architecture changes.
12. The next step is unlocked only after the gate passes.

Suggested commands:

    python -m backend.engineering_harness route --request "next step"
    python -m backend.engineering_harness verify --step <step-id>

The harness, not the coding agent, decides whether a step passed.

## Stop conditions

Stop immediately when:

- a required test fails;
- a regression appears;
- an unapproved file changes;
- a requirement has no evidence;
- a target or framework is added outside the plan;
- a security boundary is weakened;
- a function or file exceeds the agreed size;
- the coding agent cannot explain why a change is needed.

Record the failure. Fix the current step or update the plan through explicit
review. Never skip the step or weaken a test just to continue.

## First implementation session

The first coding session should only:

1. Confirm the repository and baseline.
2. Create the engineering-harness manifest.
3. Add the first harness tests.
4. Implement the step router.
5. Run the harness tests and baseline regression tests.
6. Record the verification result.

Do not build the MCP server, frontend changes, evaluation adapters, or agent
drivers in the first session.

## Final success condition

The solution is complete only when the runtime harness is part of the toolkit,
the engineering harness controls implementation, every plan step has evidence,
and the full regression suite passes at the final release gate.


## Copy/paste starter prompt

Use this prompt to start the coding-agent implementation:

```text
You are the implementation agent for this repository.

Treat these documents as the source of truth:
- docs/enterprise-mcp-implementation-plan.md
- docs/runtime-harness-implementation-plan.md
- docs/engineering-harness-and-loop-plan.md
- docs/enterprise-mcp-granular-delivery-plan-part1.md
- docs/enterprise-mcp-granular-delivery-plan-part2.md
- docs/reference-pack-builder-plan.md
- docs/getting-started-implementation-guide.md

Your first task is only to build the engineering harness that will control
future implementation work. Do not implement the MCP server, frontend changes,
agent drivers, evaluation adapters, or reference-pack builder yet.

Start by:
1. Inspecting the repository and existing tests.
2. Creating the engineering plan manifest and step registry.
3. Adding status, verification, and traceability artifact storage.
4. Implementing the step router.
5. Adding focused unit tests for the engineering harness.
6. Running the baseline regression tests.

After the engineering-harness gate passes, the harness must unlock the runtime
harness phases R0 through R7. Only after the runtime-harness gate passes may it
unlock the granular product phases: contracts, reference-pack builder, dataset
and golden generation, agent drivers, evaluation adapters, storage, MCP,
frontend, and cleanup. Do not skip or reorder this sequence.

Follow the engineering harness loop exactly:
- work on one approved step at a time;
- change only files allowed by that step;
- write or update focused unit tests;
- run focused unit tests;
- run all completed-step regression tests;
- run architecture, security, scope, and file-size checks;
- record command results and evidence;
- stop if any gate fails.

Do not claim that a check passed unless you executed it. Do not skip a failed
test, weaken an acceptance criterion, add speculative abstractions, or move to
the next step without a passing gate. Ask for human review before changing a
public contract, security boundary, architecture rule, or plan requirement.

At the end, report:
- files changed;
- tests and checks executed;
- exact pass or failure results;
- evidence file locations;
- the next eligible step only if the current gate passed.
```
