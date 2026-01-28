# Implementation Plan

### 🔹 Prompt 1: `A2A Agent Servers (Dataset, Scoring, Reports)`

- **What it implements:** Add three separate A2A-compliant HTTP+JSON servers (one process each) for dataset creation, scoring/metrics, and report generation. Include Agent Card discovery and a `/health` endpoint for each agent.
- **Dependency:** None.
- **Prompt:**
  ```
  Write complete and executable code to add three A2A-compliant HTTP+JSON servers in Python using a2a-sdk: DatasetCreationAgent, ScoringMetricsAgent, and ReportGenerationAgent.  
  Each agent runs as a separate process with its own Agent Card at /.well-known/agent-card.json and a /health endpoint.  
  Implement minimal agent logic stubs that call into existing backend functions for dataset creation, metrics/scoring, and report generation (wire to current backend modules; do not change their behavior).  
  Add unit tests that verify each agent responds to /health and exposes the Agent Card endpoint.  
  Execute the tests and show results.  
  Make sure the code is production-ready and avoids placeholders or stubs.
  ```

### 🔹 Prompt 2: `Standardized A2A Result Schema`

- **What it implements:** Define a shared result schema for all A2A agents and ensure each agent returns this payload consistently.
- **Dependency:** Prompt 1.
- **Prompt:**
  ```
  Write complete and executable code to define a standardized result schema used by all three A2A agents.  
  Include fields such as: status, task_id, agent_name, run_id, attempt, max_attempts, started_at, completed_at, duration_ms, artifacts (paths), output (agent-specific payload), metrics (if any), warnings, and errors.  
  Update each agent to return this schema for both success and failure paths.  
  Add unit tests to validate schema fields and ensure error cases populate errors/warnings correctly.  
  Execute the tests and show results.  
  Make sure the code is production-ready and avoids placeholders or stubs.
  ```

### 🔹 Prompt 3: `Orchestrator Async Queue + Polling`

- **What it implements:** Modify the backend orchestrator to enqueue work to A2A agents asynchronously and poll for completion (5s interval). Add per-task timeout (default 60s) and retry attempts (3) with configurable backoff.
- **Dependency:** Prompts 1–2.
- **Prompt:**
  ```
  Write complete and executable code to update the backend orchestrator to submit tasks to the three A2A agents asynchronously and poll for completion every 5 seconds.  
  Implement per-task timeout (default 60s) and retry attempts (default 3) with a configurable backoff strategy.  
  Ensure failure isolation between agents so one agent failure does not halt others.  
  Persist A2A task state under runs/ for polling and recovery.  
  Add unit tests for enqueueing, polling, retry logic, and timeout handling.  
  Execute the tests and show results.  
  Make sure the code is production-ready and avoids placeholders or stubs.
  ```

### 🔹 Prompt 4: `Persist A2A Results into results.json/results.csv`

- **What it implements:** Save standardized A2A agent results into existing `results.json` and `results.csv` outputs.
- **Dependency:** Prompts 2–3.
- **Prompt:**
  ```
  Write complete and executable code to persist the standardized A2A result schema into results.json and results.csv alongside existing run outputs.  
  Ensure the schema fields are mapped consistently in both JSON and CSV outputs.  
  Add unit tests that verify the persistence and field mapping in both files.  
  Execute the tests and show results.  
  Make sure the code is production-ready and avoids placeholders or stubs.
  ```

### 🔹 Prompt 5: `Settings Panel: A2A Configuration`

- **What it implements:** Extend the existing settings panel to configure A2A endpoints (host/port per agent), timeout, retries, and backoff strategy.
- **Dependency:** Prompt 3.
- **Prompt:**
  ```
  Write complete and executable code to extend the existing settings panel to configure A2A agent endpoints (host/port for each agent), per-task timeout (default 60s), retry attempts (default 3), and backoff strategy.  
  Persist settings using the existing settings flow (.env or configs) consistently with current behavior.  
  Add frontend tests (or backend tests where applicable) that validate settings persistence and defaults.  
  Execute the tests and show results.  
  Make sure the code is production-ready and avoids placeholders or stubs.
  ```

### 🔹 Prompt 6: `Per-Agent Health View`

- **What it implements:** Add a UI status/health view for each A2A agent, refreshing at the same cadence as the runs page progress chart and using /health ping.
- **Dependency:** Prompt 1.
- **Prompt:**
  ```
  Write complete and executable code to add a per-agent health/status view in the UI.  
  Use each agent’s /health endpoint and refresh on the same interval as the runs page progress chart.  
  Display status, last-checked time, and any error details.  
  Add tests to verify the health view renders and updates correctly.  
  Execute the tests and show results.  
  Make sure the code is production-ready and avoids placeholders or stubs.
  ```
