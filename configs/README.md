# Configs

Holds JSON Schemas and server-side configuration (metric bundles, thresholds, dataset generation strategy).

- `schemas/` for JSON Schemas (datasets, goldens, run config)
- Metrics configuration persisted as `configs/metrics.json` (used by UI and orchestrator)
- Coverage/Dataset generation strategy in `configs/coverage.json` (mode, t, sampler, dataset_paths)
