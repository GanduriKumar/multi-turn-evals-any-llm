# Status: Historical and superseded for the MCP migration.
# Use the enterprise MCP plans instead.

# Multi-Turn Commerce Evals — Implementation Plan (v2)

This plan encodes the clarified requirements for a Commerce merchant app. Each prompt is atomic, self-contained, and test-driven. Do not overwrite existing files; add new modules/files where reasonable.

---

## 🔹 Prompt 1: Commerce Taxonomy + Risk Tiers
- What it implements: Defines domains, behaviors, axis bins, and risk tiers tailored to Commerce.
- Dependency: None.
- Prompt:
```
Write complete and executable code to define a Commerce taxonomy and risk tiers for evaluation dataset generation.
Include:
- Domains: Orders & Returns; Payments & Refunds/Chargebacks; Promotions & Pricing; Inventory & Availability; Shipping & Logistics; Account & Security; Restricted Items & Compliance; Marketplace Policy/Disputes.
- Behaviors: Refund/Exchange/Cancellation; Price match/Discount/Coupon stacking; Post-purchase modifications; Availability workarounds; Restricted items or cross-border compliance; PII access/deletion; Chargeback disputes; Adversarial/trap attempts.
- Axes with bins: price_sensitivity; brand_bias; availability; policy_boundary (include near_edge_allowed; exclude over_edge in this suite).
- Risk tiers: High (as confirmed set), Medium (everything else not Low), Low (low-stakes bins/domains as confirmed).
Provide JSON/YAML plus a loader with schema validation and unit tests.
Execute the tests and show results.
```

## 🔹 Prompt 2: Risk-Weighted Stratified Sampler (100 per behavior)
- What it implements: Selection algorithm with domain floors and pair coverage guarantee.
- Dependency: Prompt 1.
- Prompt:
```
Implement a sampler that:
- Targets 100 scenarios per behavior (configurable), min 3 per domain, ≥90% axes pair coverage within the selected set.
- Allocates ~50 High, ~35 Medium, ~15 Low risk.
- Uses a stable RNG (seed from config) for reproducibility.
- Outputs a manifest with scenario IDs, risk tier, domain, and covered axis pairs.
Add tests for allocation, domain floors, pair coverage estimate, and determinism.
Execute tests and show results.
```

## 🔹 Prompt 3: Policy and Scenario Facts Storage
- What it implements: Per-domain policy excerpts and per-scenario facts with constraints.
- Dependency: Prompt 1.
- Prompt:
```
Create storage and models for:
- Per-domain policy text at configs/policies/<domain>.md (short, evaluable excerpts).
- Per-scenario facts generated synthetically from axis templates (deterministic via seed), constrained to 2–5 bullets, ≤120 words total, include 1–2 numbers/dates, no real PII.
Add loaders/validators and tests for completeness and constraints.
Execute tests and show results.
```

## 🔹 Prompt 4: Unified System Prompt Builder (policy + facts only)
- What it implements: Single system message containing policy and facts; standardized decoding params.
- Dependency: Prompts 1 and 3.
- Prompt:
```
Implement a SystemPromptBuilder that:
- Accepts domain, behavior, axes, policy_text, and facts_text.
- Renders one system message with sections: Role, Safety/Policy, Scenario Facts, Output Requirements.
- Applies decoding defaults: temperature=0.2, max_tokens=512, top_p=1.0, presence_penalty=0, frequency_penalty=0.
Include sanitization and truncation; add unit tests for exact rendering and parameter packing.
Execute tests and show results.
```

## 🔹 Prompt 5: Conversation Generator (clarify → correct)
- What it implements: Generates U1 and U2 only; A1/A2 produced at run-time; canonical A2 auto-composed.
- Dependency: Prompts 1 and 3.
- Prompt:
```
Implement a conversation generator that for each scenario produces:
- messages.u1: opening with domain/axes flavor (price sensitivity, brand bias, availability, policy boundary).
- messages.u2: correction/change referencing axes after an assumed A1 clarification.
- expected: { outcome: "Allowed", a2_canonical } where a2_canonical is auto-composed from behavior-specific templates using facts/policy.
Provide deterministic IDs and exports. Add unit tests for structure and template filling.
Execute tests and show results.
```

## 🔹 Prompt 6: Canonical A2 Template Library
- What it implements: Behavior-specific templates that synthesize a short, actionable A2 from facts/policy.
- Dependency: Prompts 3 and 5.
- Prompt:
```
Create a template library that:
- Maps behavior → template with placeholders for key fact/policy fields.
- Produces concise, policy-compliant, actionable canonical A2 answers.
- Validates output length and presence of required fields.
Add unit tests using sample facts/policy for multiple behaviors.
Execute tests and show results.
```

## 🔹 Prompt 7: Provider Adapters (OpenAI, Gemini, Ollama) with System-Only Prompting
- What it implements: Unified interface across providers; standardized decoding params; system message only.
- Dependency: Prompt 4.
- Prompt:
```
Implement or update provider adapters to:
- Accept a single system message (policy+facts) and user messages (u1, u2) for two-turn execution.
- Apply standardized decoding params.
- Return normalized responses (text, tokens, latency, error) and per-turn usage.
Add integration tests with mocked HTTP clients ensuring proper system role usage and params.
Execute tests and show results.
```

## 🔹 Prompt 8: Orchestrator Multi-Turn Flow (all-assistant-turns gating)
- What it implements: Message assembly for two turns; persistence; success judged on final outcome AND no severe violations on any assistant turn.
- Dependency: Prompts 4–7.
- Prompt:
```
Enhance the orchestrator to:
- Turn 1: [system(policy+facts), U1] → A1 (clarification).
- Turn 2: [system(policy+facts), U1, A1, U2] → A2 (final resolution).
- Persist turn artifacts with text/usage/latency.
- Mark conversation success only if:
	- Final outcome matches expected (from canonical A2/state), and
	- No high-severity violations (adherence/hallucination) occur on any assistant turn (A1 or A2).
- Keep semantic similarity applied to A2 vs canonical A2 only.
Add tests with stub providers to verify ordering and persistence.
Execute tests and show results.
```

## 🔹 Prompt 9: Embeddings + Semantic Similarity (0.8 threshold)
- What it implements: Cosine similarity scoring using Ollama nomic-embed-text.
- Dependency: Prompt 6.
- Prompt:
```
Implement a semantic similarity scorer that:
- Uses Ollama embeddings model: nomic-embed-text.
- Computes cosine(A2, a2_canonical) with vector normalization; pass/fail at threshold 0.8 (configurable).
- Handles retries/timeouts and caches embeddings.
Add unit tests for cosine math, thresholding, and caching.
Execute tests and show results.
```

## 🔹 Prompt 10: Policy Adherence (Allowed)
- What it implements: Detects incorrect refusal and major deviations from policy constraints.
- Dependency: Prompts 3, 5, and 8.
- Prompt:
```
Implement a lightweight policy adherence checker for Allowed outcomes that:
- Flags refusal phrases and mismatches with key policy constraints (pattern-based checks).
- Returns pass/fail with flags and rationale.
Add unit tests covering allowed vs incorrect refusal scenarios.
Execute tests and show results.
```

## 🔹 Prompt 11: Hallucination Heuristic (facts/policy support)
- What it implements: Heuristic scoring without an LLM judge.
- Dependency: Prompt 3.
- Prompt:
```
Create a heuristic hallucination scorer that:
- Extracts key entities/numbers from A2 via simple NP/NER heuristics or regex.
- Scores support by overlap/containment against facts/policy.
- Produces a normalized score and pass/fail using a configurable threshold.
Add tests with positive/negative cases and edge conditions.
Execute tests and show results.
```

## 🔹 Prompt 12: Aggregation and Reporting (Conversation-level + Turn breakdown)
- What it implements: Rollups by domain/behavior/risk/axes; exports.
- Dependency: Prompts 8–11.
- Prompt:
```
Implement aggregation/reporting that:
- Computes conversation-level success (A2) and shows turn-level metrics.
- Provides rollups by domain, behavior, risk tier, and axis bins; trend by run.
- Exports CSV/JSON with stable schemas and records run-level token totals (input_tokens_total, output_tokens_total) aggregated across turns.
- Applies the gating rule: conversation passes only if final outcome passes AND no high-severity violations on any assistant turn.
Add tests for aggregation correctness and schema stability.
Execute tests and show results.
```

## 🔹 Prompt 13: Settings + Strategy Controls
- What it implements: API/UI to configure risk-weighted sampling (bundle M), thresholds, seed, and dataset paths.
- Dependency: Prompts 1–2 and 9–11.
- Prompt:
```
Add settings to control:
- Strategy: Risk-weighted (M: 100/behavior), RNG seed (default 42), domain floor (min 3/domain).
- Semantic threshold (default 0.8) and hallucination threshold.
- Context window: last 5 turns; input budget ~1800 tokens; max completion tokens ~400.
- Dataset path convention: datasets/commerce/<behavior>/<version>/<slug>.json and versioning (default 1.0.0).
Include validation and persistence; add tests for API and UI.
Execute tests and show results.
```

## 🔹 Prompt 14: Dataset Generation/Regeneration (JSON array schema)
- What it implements: Endpoints/CLI to generate combined datasets in one JSON array with agreed schema.
- Dependency: Prompts 2, 3, 5, and 6.
- Prompt:
```
Implement generation/regeneration endpoints/CLI that output a single JSON array with fields:
- scenario_id, version, domain, behavior, risk_tier, axes{price_sensitivity, brand_bias, availability, policy_boundary}, policy_text, facts_text,
- messages{u1, u2}, expected{ outcome:"Allowed", a2_canonical }, metadata{ seed, created_ts }.
Support overwrite/version bump; return manifest preview and counts by risk tier.
Add tests for schema emission and overwrite safeguards.
Execute tests and show results.
```

## 🔹 Prompt 15: CI + Docs (v1 scope)
- What it implements: CI for lint/tests and docs for usage.
- Dependency: All previous.
- Prompt:
```
Set up CI to run tests and lint on push/PR. Add documentation covering:
- Taxonomy, risk tiers, dataset generation strategy (risk-weighted M), and seeding.
- Unified system prompt, multi-turn flow (clarify → correct), metrics (similarity 0.8, policy adherence, hallucination heuristic), token accounting (input/output totals), and report visualizations (donuts + token card).
- Dataset schema, paths, and versioning.
Defer cross-provider comparison to a later iteration.
Run CI locally and show results.
```

---

Conventions (confirmed):
- Embeddings: Ollama nomic-embed-text
- Similarity threshold: 0.8
- Outcome: Allowed (A2 should resolve)
- System message: policy + facts (single system role)
- Multi-turn: U1 → A1(clarify) → U2 → A2(final); pass requires final outcome pass + no severe violations on any assistant turn
- Dataset path: datasets/commerce/<behavior>/<version>/<slug>.json
- scenario_id: deterministic slug with hash; version 1.0.0; seed 42

Golden Matching Rules (explicit):
- Scope golden by dataset_id and conversation_id.
- Prefer entry.final_outcome and entry.constraints; fall back to top-level when missing.
- Map assistant turn index from user turn index with preference order: 2*uidx+1 → uidx+1 → uidx.
- Use variants for exact/semantic checks on the mapped assistant index only.
- Enforce outcome-first gating with high-severity violations across all assistant turns.

Download link:
file:///c:/Users/kumar.gn/PycharmProjects/Testproject/docs/multi_turn_evals_implementation_plan_v2.md
