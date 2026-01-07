# App Consistency Repair — NO-CODE Interview + Implementation-Plan Prompts (VSCode + GitHub Copilot)

Save as: `.github/app_consistency_repair.interview_and_plan.prompts.md`

This prompt-file forces a strict sequence for **Copilot Chat / Agent mode**:
1) **Review** the full workspace to identify inconsistencies (UI styling, FE↔BE integration, persistence, navigation/edit flows).
2) Run a **structured interview** with **ONE question at a time**.
3) **Only after** the interview is complete, generate a **separate implementation plan prompts document** (markdown) — **and do not change any code**.

---

## 0) HARD GATES — YOU MUST OBEY (Copilot Agent)

### Gate G0 — NO-CODE MODE (mandatory)
You are in **NO-CODE MODE** until explicitly told otherwise.

**Allowed in NO-CODE MODE**
- Read and analyze files
- Search workspace
- Produce markdown documents (audit report + interview log + implementation plan prompt doc)
- Suggest fixes **only as plan prompts** (not as code changes)

**Forbidden in NO-CODE MODE**
- **Do NOT modify any source code or config**
- Do NOT run generators that write files
- Do NOT reformat files
- Do NOT create PRs
- Do NOT implement “quick fixes”

If you are about to edit any file: **STOP** and instead add that fix as a new prompt in the plan.

### Gate G1 — Required deliverables BEFORE any coding can start
Before leaving NO-CODE MODE, you must produce **both**:
1) `docs/audit_snapshot.md`
2) `docs/implementation_plan.prompts.md`

If either file is missing, you must not implement anything.

### Gate G2 — Explicit user command required to start implementation
Only after G1 is completed, you may implement **one prompt at a time** when the user says exactly:

> **Start Prompt N**

Until then, remain in NO-CODE MODE.

---

## 1) Repo Scan Instructions (do this BEFORE interview)

### Step S1 — Workspace map
- List top-level folders and key packages (frontend, backend, shared, packages, etc.).
- Identify frameworks and tools:
  - Frontend (React/Next/Vite), router, Tailwind/UI library
  - Backend (FastAPI/Flask/Node), DB/ORM
  - Auth/session mechanism (if any)
  - API client (fetch/axios), OpenAPI usage (if any)
  - State management (Context/Redux/Zustand), persistence (localStorage/DB)

### Step S2 — Quick health check (read-only)
- Identify existing tests and commands
- Identify env/config files and base URL wiring
- Identify routing/menu structure
- Identify where state is stored and why it might be lost

### Output required (to `docs/audit_snapshot.md`)
Create `docs/audit_snapshot.md` with sections:
- **Findings Snapshot** (max 20 lines)
- **UI Uniformity Hotspots** (page/component paths)
- **FE↔BE Contract Mismatches** (endpoint map + mismatch notes)
- **Persistence Gaps** (what is lost, where, expected behavior)
- **Navigation & Editability Issues** (routes/menu items: expected vs actual)
- **Test/CI Status** (what exists, what’s missing)

Do not implement fixes — only report.

---

## 2) Interview — ask ONE question at a time

### Interview Operating Rules
- Ask **exactly one question**.
- Wait for user’s answer.
- After each answer:
  - Summarize learnings (3–6 bullets)
  - Update “Open Assumptions” list
  - Ask the next question

### Interview Exit Criteria (must satisfy before generating plan)
You must confirm:
1) **Source of truth** for shared app state (client/server/hybrid)
2) **Auth/session** model (or “none”)
3) Expected **persistence behavior** for each setting/selection
4) Canonical **UI system** (Tailwind conventions/design tokens/components)
5) API contract boundary and **error-handling** approach
6) **Test strategy** and the commands that should run in CI

### Questions (ask sequentially)
**Q1.** What is the app stack and repo structure? (frontend framework + backend framework + main folders)

**Q2.** What are the top 3 pages where UI styling looks most inconsistent? (name pages/routes)

**Q3.** Do you have a design system baseline to enforce?
- Tailwind-only, shadcn/ui, MUI/AntD, custom components?
Also any brand colors/typography rules?

**Q4.** Which selections/settings must persist across pages? List each item and the pages where it should appear.

**Q5.** Where should persistence live?
- client-only (localStorage)
- server DB (per user/workspace)
- hybrid (client cache + server)
Pick the intended approach.

**Q6.** Is there authentication?
- If yes: how does the frontend authenticate (cookie/JWT/header)?
- If no: should persistence be per browser only?

**Q7.** What is the intended API style?
- REST only?
- REST + OpenAPI?
Where is API base URL configured for dev/prod?

**Q8.** Give 3 concrete examples of FE↔BE integration inconsistency (endpoint mismatch, payload mismatch, CORS, error shape, etc.).

**Q9.** List 5–10 navigation/menu items that must be verified for clickability + correct routing.

**Q10.** What is the expected edit/save UX?
- inline edit?
- edit page with Save/Cancel?
- autosave?
Any validation rules?

**Q11.** What testing exists today?
- Frontend: Vitest/Jest + RTL? Playwright/Cypress?
- Backend: pytest?
Provide the commands.

**Q12.** Any constraints?
- avoid breaking API?
- must keep URLs?
- must keep DB schema?

**Q13.** Priority order (rank): UI uniformity, API correctness, persistence, navigation/editability, tests/CI hardening.

**Q14.** Definition of Done:
- What outputs demonstrate success (passing tests, demo script, screenshots)?

---

## 3) Implementation Plan — MUST be generated as a markdown document

### Mandatory Output
After the interview exit criteria is met, generate:
- `docs/implementation_plan.prompts.md`

**Important:** This plan is a sequence of prompts for Copilot. It is NOT the code.  
Remain in NO-CODE MODE after generating this plan until the user says “Start Prompt N”.

### Global plan constraints
- **One prompt = one specific fix + its related integration + its tests**
- Every prompt must require:
  - complete, executable production-ready code (no stubs/placeholders)
  - unit tests (and integration/E2E where applicable)
  - running tests and showing results
- Prompts must be small and sequential, with explicit dependencies.

### Recommended ordering (adapt based on audit)
1) Tooling baseline (lint/test scripts, consistent commands)
2) FE↔BE contract normalization (schemas/types + error shape)
3) API client consolidation (base URL, interceptors, one client)
4) Global state + persistence (store/provider + storage/server)
5) UI system unification (components/tokens/Tailwind conventions)
6) Navigation + edit flows (routes/menu, save/cancel, validation)
7) Regression suite (add coverage, E2E smoke tests)
8) Docs updates (README, troubleshooting)

---

## 4) 🔹 Implementation Plan Format (use exactly this in `docs/implementation_plan.prompts.md`)

### Implementation Plan

### 🔹 Prompt N: `[Title of Feature or Requirement]`

- **What it implements:** 1–2 sentence description of this unit’s functionality.
- **Dependency:** Reference any earlier or later prompt it connects to.
- **Prompt:**
  ```
  Write complete and executable code to implement [describe this requirement].
  Include all relevant function or class definitions.
  Add tests that verify correctness using a test framework.
  Execute the tests and show results.
  Make sure the code is production-ready and avoids placeholders or stubs.
  ```

---

## 5) “No-code driver” message you should paste into Copilot Chat (recommended)

Copy/paste this as your first message in Copilot Chat when starting:

```
Follow `.github/app_consistency_repair.interview_and_plan.prompts.md`.

HARD RULE: NO-CODE MODE.
- Do not modify any code or config files.
- First generate `docs/audit_snapshot.md`.
- Then ask the interview questions ONE at a time.
- After interview exit criteria is met, generate `docs/implementation_plan.prompts.md` using the specified format.
- Stop. Do not implement fixes until I say: "Start Prompt N".
```

---

## 6) Start Here (Copilot Agent)
1) Perform Repo Scan Steps S1–S2 (read-only).  
2) Create `docs/audit_snapshot.md`.  
3) Ask **Q1 only**.  
