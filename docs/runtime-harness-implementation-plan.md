# Runtime Harness Implementation Plan

Status: Proposed

This plan builds the runtime harness around the evaluation toolkit. It is
based on the article provided by the user:

https://medium.com/@roanmonteiro/ai-architecture-and-system-design-the-guide-i-wish-id-read-before-putting-an-agent-in-production-d8f90383004e

The runtime harness is the boundary that makes an agent testable, observable,
bounded, recoverable, and safe. It does not require changes to the evaluated
agent.

## 1. Runtime responsibilities

The toolkit supplies:

- interface and target-driver lifecycle;
- session state and event history;
- orchestration limits and stop conditions;
- tool, network, and artifact observation;
- permissions, sandboxing, and secret handling;
- tracing, metrics, evaluation handoff, and cost telemetry;
- durable execution, cancellation, retry, and recovery;
- MCP and frontend status/report access.

The evaluated agent keeps its own model, tools, and implementation. The
toolkit observes it externally through A2A, HTTP, function, CLI, container, or
other approved drivers.

## 2. Architecture layers

### Interface

MCP is the primary control interface. CLI and the retained frontend use the
same services. Long-running evaluations expose status and artifacts rather than
holding a fragile synchronous request open.

### Orchestration

Use the selected external evaluation framework or an established execution
framework before writing custom orchestration. The harness enforces maximum
turns, maximum duration, retry limits, cancellation, and a safe stop condition.
An agent is allowed to stop with an incomplete or failed result.

### Model and agent boundary

The toolkit does not assume one model or one agent framework. It records the
target and provider metadata when available, but evaluates the agent through
the target-driver contract.

### Tools and MCP

Treat agent tools and MCP servers as untrusted dependencies. Allow-list hosts,
commands, tools, and workspaces. Record tool calls when externally observable.
Do not treat a prompt as a security boundary.

### State and durable sessions

Persist run state, turn history, task status, observations, and artifacts.
Long-running work must resume from the last durable checkpoint rather than
starting from the beginning.

### Security and permissions

Run targets in an isolated or disposable environment where possible. Use
short-lived credentials or secret references. Default to read-only or fake
tools. Require explicit configuration for side effects.

### Observability and evaluation

Emit structured events for every session, turn, response, tool observation,
retry, timeout, cancellation, artifact, and failure. Record latency, token
usage, cost when available, and metric results. Every report must link back to
the recorded run and its evidence.

## 3. Cross-cutting controls

Implement:

- circuit breakers for repeatedly failing targets or frameworks;
- graceful degradation when optional telemetry is unavailable;
- bounded retries with no automatic retry of non-idempotent actions;
- feature flags for new drivers and observation modes;
- async execution for long-running runs;
- cancellation and kill behavior;
- output, duration, and resource limits;
- replay from recorded inputs and responses.

These controls are part of the first usable runtime, not future enhancements.

## 4. Implementation phases

### R0: Session contract

Define run, session, turn, event, artifact, error, and checkpoint schemas.

Gate: valid, invalid, and round-trip fixtures pass.

### R1: Target lifecycle

Wrap each target driver with start, send, observe, finish, timeout, cancel, and
cleanup operations.

Gate: fake function, HTTP, A2A, and process lifecycle tests pass.

### R2: Orchestration limits

Add turn caps, duration budgets, retry rules, safe stop conditions, and
cancellation.

Gate: limit, stop, retry, cancellation, and non-idempotent-action tests pass.

### R3: State and event log

Persist append-only events, checkpoints, task status, bounded output, and
artifact references.

Gate: ordering, restart, redaction, size-limit, and recovery tests pass.

### R4: Security boundary

Enforce target allow-lists, isolated workspaces, command restrictions, secret
references, and explicit side-effect permissions.

Gate: unsafe target fixtures are rejected and approved fake targets pass.

### R5: Observability and framework handoff

Emit traces and cost/latency events. Pass normalized observations and trace
references to DeepEval, EvalBench, or the selected framework.

Gate: framework fixtures consume the same observation contract and reports link
metrics to evidence.

### R6: Durable async execution

Run long evaluations asynchronously, expose status through MCP, resume from
checkpoints, and support circuit breakers and graceful degradation.

Gate: interrupted-run recovery, repeated-failure, cancellation, and status
polling tests pass.

### R7: MCP and frontend integration

Expose session status, event references, results, reports, failures, and
replay references through MCP. Show the same data in the retained frontend.

Gate: a full fake-agent run succeeds through MCP and the frontend without
private backend calls.

## 5. Coding-agent enforcement

Every runtime phase and step must be entered in the engineering harness
manifest with:

- allowed files;
- prerequisites;
- unit tests;
- regression tests;
- security checks;
- observability checks;
- acceptance criteria;
- evidence location.

The coding agent must complete the engineering loop before advancing:

1. Load one step.
2. Implement only its scope.
3. Run unit tests.
4. Run all previous regression tests.
5. Run architecture, security, and file-size checks.
6. Verify requirement-to-test traceability.
7. Record evidence.
8. Stop on failure.
9. Require human approval for contract, security, or architecture changes.

## 6. Runtime definition of done

The harness is complete when target sessions are bounded, observable, secured,
durable, cancellable, replayable, and connected to normalized evaluation
results. The evaluated agent remains unchanged, and the coding-agent harness
can prove each requirement with recorded evidence.
