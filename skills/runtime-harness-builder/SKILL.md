---
name: runtime-harness-builder
description: Design and implement the runtime harness around an AI model or agent so the system is bounded, observable, secure, recoverable, and testable. Use when building or refactoring an AI solution that takes actions, uses tools, maintains state, or runs multi-step work.
---

# Runtime Harness Builder

Build the surrounding system that makes an AI model or agent usable in the real
world. Treat the model as one component, not the whole system.

## Start with the system boundary

Before writing code, identify:

- the user or system interface;
- the agent or model target;
- the tools and external systems it can reach;
- the state that must survive between turns or restarts;
- the actions that may have side effects;
- the latency, cost, and reliability limits;
- what can actually be observed from outside the target.

Write these decisions down as a small architecture note and test contract.

## Build the harness in this order

1. Define a session contract for inputs, outputs, status, errors, artifacts,
   timestamps, and identifiers.
2. Define the target lifecycle: start, send, observe, finish, cancel, timeout,
   and cleanup.
3. Add explicit orchestration limits: maximum turns, duration, retries,
   output size, and a safe stop condition.
4. Add state and an append-only event history. Persist checkpoints for work
   that can outlive one process.
5. Add tool and network boundaries. Allow-list hosts, commands, tools, and
   workspaces.
6. Add permission and secret handling. Enforce authority on the server or
   harness; never rely on prompts as security controls.
7. Add structured observability for requests, turns, tool events, retries,
   timeouts, costs, latency, artifacts, and failures.
8. Add cancellation, bounded retries, circuit breakers, and graceful
   degradation.
9. Add replay from recorded inputs and responses. Live replay must be explicit.
10. Expose status, evidence, and artifacts through the public interface.

Use a proven orchestration or evaluation framework when it covers the need.
Add custom orchestration only when the gap is demonstrated by a test or
requirement.

## Agent compatibility

The evaluated agent must not need toolkit-specific code. Use an external
adapter, proxy, process boundary, or protocol client. Support the interfaces
that the project actually requires, such as A2A, HTTP, function calls, CLI,
WebSocket, or containers.

Capture only observable information. Mark hidden reasoning and unavailable tool
traces as unavailable; never infer them or treat missing data as a pass.

## Safety requirements

Default to read-only or fake tools. Require explicit configuration for
side effects. Enforce:

- target and command allow-lists;
- isolated or disposable workspaces where practical;
- short-lived credentials or secret references;
- timeouts and output limits;
- cancellation and kill behavior;
- redaction before persistence;
- no automatic retry of non-idempotent actions.

## Verification requirements

For each harness capability, add:

- unit tests for valid and invalid inputs;
- failure tests for timeout, cancellation, malformed output, and crashes;
- integration tests with a fake target;
- restart or recovery tests for durable state;
- replay tests;
- security tests for rejected targets and redacted secrets;
- regression tests for all previously completed capabilities.

Do not report a capability as complete until its tests execute successfully.
Preserve the command, exit code, and evidence for each gate.

## Keep the design understandable

Give each module one responsibility: session contract, target lifecycle,
orchestration limits, event store, security policy, observability, replay, or
interface. Avoid a universal manager that owns all of them.

Do not add routers, model tiers, caches, distributed workers, or protocol
connectors without a current requirement and a test that justifies them.


## Engineering loop requirement

When this skill is used inside a planned implementation, follow the plan-locked
loop as well as the runtime design steps:

1. Select one approved plan step and its allowed files.
2. Define the runtime-harness acceptance criteria before editing.
3. Implement only the current runtime capability.
4. Add focused unit, failure, integration, and regression tests.
5. Run the checks through the engineering harness and record evidence.
6. Stop on a failed gate or an unapproved architecture, security, or scope
   change.
7. Unlock the next runtime step only after the current gate passes.

The runtime harness must be represented in the implementation plan and the
engineering harness manifest. Do not build the runtime behavior as an
untracked helper or a one-off test fixture.