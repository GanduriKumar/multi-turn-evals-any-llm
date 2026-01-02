multi-turn-evals-any-llm
=================================

Purpose
-------
This repository is a monorepo scaffold for building a multi-turn evaluation framework that can run against any Large Language Model (LLM). It organizes backend services, frontend UI, datasets, metrics, configurations, templates, and automation scripts into a single, cohesive structure.

Top-level structure
-------------------
- backend/   — services, APIs, and evaluation orchestration logic
- frontend/  — web UI for configuring runs, visualizing results, and comparisons
- configs/   — runtime configuration (YAML/JSON/TOML) for eval jobs, providers, and pipelines
- datasets/  — prompts, conversations, and gold references used in evaluations
- metrics/   — metric definitions and evaluators (exact match, BLEU, ROUGE, custom rubric, etc.)
- templates/ — prompt templates and system messages (jinja/handlebars or similar)
- scripts/   — automation utilities, CI hooks, and test tooling
- docs/      — documentation, design notes, and how-tos

Quick start
-----------
1) Clone and create your virtual environment for backend work (optional):
	 - Python: create .venv and install your packages
	 - Node: for frontend if/when you add a web app

2) Place your datasets, configs, and templates into their respective folders.

3) Add metric implementations under `metrics/` and evaluation drivers under `backend/`.

Repository health check
-----------------------
Run the structure test to verify the scaffold:

	bash scripts/test_structure.sh

The script fails if any required top-level directory or file is missing.

License
-------
This project is licensed under the Apache License, Version 2.0. See `LICENSE` for details.