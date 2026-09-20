# Status: Historical and superseded for the MCP migration.
# Use the enterprise MCP plans instead.

# Dataset 100% Coverage Generation – Implementation Plan

This plan defines how to generate datasets with 100% coverage based on Domains × Behavioral Classes × Variation axes, with approved thresholds and exclusion rules.

## Final requirements (confirmed)
- Domains (14):
  1) Product Discovery & Search  2) Product Details & Content  3) Recommendations & Ranking  4) Pricing, Offers & Promotions  5) Cart Management  6) Checkout & Payments  7) Order Management  8) Shipping & Delivery  9) Returns, Refunds & Exchanges  10) Customer Support & Escalations  11) User Account & Preferences  12) Trust, Safety & Fraud  13) Merchant / Seller Operations  14) System Awareness & Failure Handling
- Behavioral classes (6): Happy Path; Constraint-heavy Queries; Ambiguous Queries; Multi-turn Conversations; User Corrections; Adversarial/Trap Queries
- Variation axes (discretized bins):
  - Price sensitivity: low (≤10%), medium (10–30%), high (≥30%)
  - Brand bias: none, soft (prefers a brand but accepts others), hard (brand‑exclusive)
  - Availability: in-stock, low-stock, backorder, out-of-stock
  - Policy boundary: within-policy, near-limit, out-of-policy
- Coverage target: Full Cartesian across the above axes per Domain × Behavioral class (3×3×4×3 = 108 scenarios), subject to exclusions below.
- Exclusions (approved baselines):
  - Physical impossibility: Shipping & Delivery (Constraint-heavy): exclude availability ∈ {backorder, out-of-stock} when request implies immediate pickup/same-day.
  - Business irrelevance: System Awareness & Failure Handling; Trust, Safety & Fraud: brand bias = none only (exclude soft/hard).
  - Redundancy reduction (Happy Path): when availability=in-stock and policy=within-policy, keep 3 scenarios (price ∈ {low, med, high} × brand=none), drop other brand biases.
  - Low-risk deprioritization: in Happy Path, cap remaining in-stock+within-policy to 12 per behavior (balanced across price bins).
  - Regulatory/policy blocks: exclude out-of-policy in Happy Path; allow in Adversarial/Trap only (retain 1 per price bin).
  - Skew control: limit extreme combos (high price × out-of-policy × out-of-stock × hard brand) to 1 per Domain × Behavior.

Notes: Support generating one combined dataset+golden covering all scenarios per domain and/or globally; use `difficulty: "mixed"` when combining. Store under per-vertical paths (e.g., datasets/commerce/...). Use a stable RNG seed and record it in metadata for reproducibility.

---

## Prompts

### Prompt 1: Coverage Taxonomy Config
- What it implements: Declarative config files listing Domains, Behavioral classes, and the discretized variation axes; plus an exclusions config with rule predicates.
- Dependency: None.
- Prompt:
```
Write complete and executable code to define a configuration module and JSON/YAML files for coverage taxonomy: domains (14), behaviors (6), and variation axes with bins as specified. Add an exclusions configuration supporting per-domain, per-behavior rules and generic predicates (e.g., physical-impossibility, business-irrelevance, redundancy, low-risk-cap, regulatory-blocks, skew-control). Include loaders, schema validation, and typed accessors.
Add tests verifying config parsing, defaults, and schema validation. Execute tests and show results.
Ensure production-ready code with no placeholders.
```

### Prompt 2: Schema Updates for Combined Datasets
- What it implements: Extend dataset schema to allow `difficulty: "mixed"` (or an array of difficulties) and ensure golden supports constraints/policy flags.
- Dependency: Prompt 1.
- Prompt:
```
Update dataset and golden JSON Schemas to support combined datasets: dataset.metadata.difficulty allows "mixed" (and optionally arrays), and golden entries include constraints/policy flags. Migrate validators and ensure backward compatibility.
Add tests that validate example single and combined datasets/goldens. Run tests and show results.
```

### Prompt 3: Rule Engine for Exclusions
- What it implements: Functional engine that enumerates and filters scenarios using the approved exclusion rules and any future rules from config.
- Dependency: Prompts 1–2.
- Prompt:
```
Implement a rule engine that generates the Cartesian product of variation axes and filters with exclusions from config. Support domain/behavior scoping, caps (e.g., max 12), and one-off extreme-case retention. Provide deterministic seeding for tie-breaking.
Add unit tests that prove each rule works and aggregate counts match expectations before/after filtering. Run tests and show results.
```

### Prompt 4: Scenario Enumerator & Manifest
- What it implements: End-to-end scenario enumeration for every Domain × Behavior into a manifest (counts per axis, filtered totals, excluded reasons).
- Dependency: Prompt 3.
- Prompt:
```
Implement a scenario enumerator that produces a JSON manifest for all Domain × Behavior pairs, including: total raw combinations, exclusions applied with counts by rule, and final kept scenarios with stable IDs. Expose functions to query by domain/behavior.
Add tests verifying manifest totals and IDs stability across runs. Execute tests and show results.
```

### Prompt 5: Conversation Template Generator
- What it implements: Deterministic templating from a scenario to a multi-turn conversation (dataset) and corresponding golden, including outcomes and policy flags.
- Dependency: Prompt 4.
- Prompt:
```
Implement a parametric conversation generator mapping a scenario (domain, behavior, axes bins) to a dataset conversation and a golden entry. Support multi-turn variants, user corrections, and adversarial prompts. Ensure final outcome decisions align with policy boundary and availability. Include few-shot fixtures per domain. Record golden constraints needed for adherence checks.
Add tests validating schema compliance and golden alignment for sampled scenarios. Run tests and show results.
```

### Prompt 6: Combined Dataset/Golden Builder
- What it implements: Builder that collates generated conversations into one or many datasets and matching golden files with versioning, IDs, and dedupe.
- Dependency: Prompt 5.
- Prompt:
```
Implement builders to assemble: (a) per-behavior datasets, (b) per-domain combined datasets, and (c) global combined datasets. Use dataset_id conventions, version bumping, and difficulty="mixed" for combined. Ensure golden alignment (scoped by dataset_id), no orphan entries, and deduplicate by stable scenario ID. Emit a corresponding golden with entry.final_outcome precedence and constraints for adherence.
Add tests verifying file contents and referential integrity. Run tests and show results.
```

### Prompt 7: Backend API for Coverage Generation
- What it implements: FastAPI endpoints to trigger generation with options: domains, behaviors, combined/split, dry-run, save/overwrite, and progress.
- Dependency: Prompts 4–6.
- Prompt:
```
Add FastAPI endpoints: POST /coverage/generate with payload {domains, behaviors, combined, dry_run, save, overwrite}. Stream or poll progress, return manifest and saved file paths under datasets/. Implement input validation and error handling.
Add API tests (pytest + httpx) covering dry-run vs save, combined vs split, and overwrite behavior. Run tests and show results.
```

### Prompt 8: CLI Tool for Batch Generation
- What it implements: Command-line tool to run generation offline, shard by domain/behavior, and resume.
- Dependency: Prompts 4–6.
- Prompt:
```
Create a CLI command `coverage generate` with flags to select domains/behaviors, combined/split, output dir, overwrite, and concurrency. Support resume and sharding. Provide human-readable summary tables.
Add CLI tests using pytest. Execute tests and show results.
```

### Prompt 9: Frontend UI for Coverage Generation
- What it implements: Configuration UI to pick domains/behaviors, see counts and exclusions, and trigger combined or split generation with save.
- Dependency: Prompt 7.
- Prompt:
```
Build a frontend page "Coverage Generator": selectors for domains/behaviors, live counts from /coverage/generate?dry_run=true, and actions to generate+save combined or split. Show exclusion breakdown and download links. Add component tests.
Run tests and show results.
```

### Prompt 10: Validation & QA Tests
- What it implements: Comprehensive tests for schemas, enumerator counts, exclusion correctness, and end-to-end generation.
- Dependency: Prompts 2–7.
- Prompt:
```
Add QA tests: (1) schema compliance for all outputs, (2) counts per Domain × Behavior match expected after exclusions, (3) golden aligns with dataset turns and outcomes, (4) manifests include exclusion rationale. Run full test suite and show results.
```

### Prompt 11: Performance & Limits
- What it implements: Efficient generation with chunking, memory control, and deterministic batching.
- Dependency: Prompts 4–6.
- Prompt:
```
Optimize generation for large counts: batch size controls, streaming writes, and deterministic chunking. Provide metrics logging and guardrails to prevent OOM. Add performance tests using smaller seeded runs and assert time/memory bounds.
```

### Prompt 12: Reporting & Coverage Dashboards
- What it implements: Summaries of kept vs excluded counts, gaps, and per-axis heatmaps; CSV/HTML exports.
- Dependency: Prompt 4.
- Prompt:
```
Implement reporting utilities that compute coverage stats and export CSV and HTML dashboards, including per-domain/behavior heatmaps and exclusion breakdowns. Include token/cost budget estimators based on average turns and configured model rates if provided. Add snapshot tests for report contents. Run tests and show results.
```

### Prompt 13: Governance & Docs
- What it implements: Versioned configs, rule docs, and contribution guidelines.
- Dependency: Prompt 1.
- Prompt:
```
Document taxonomy, bins, and exclusion rules. Version the config files and add a linter to catch invalid or overlapping rules. Provide CONTRIBUTING.md updates and examples. Add doc generation tests where applicable.
```

---

Download this plan: `docs/dataset_coverage_implementation_plan.md`
