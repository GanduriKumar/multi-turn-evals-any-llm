````markdown
# Dataset Generator - Beginner's Guide

This guide explains how to generate evaluation datasets (and matching goldens) using the Dataset Generator page.

---

## Start the app

Backend
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Frontend
```powershell
cd frontend
npm run dev
```

Open http://localhost:5173 and click “Dataset Generator” (first item in the top navigation).

---

## Section 1: Dataset Generation Strategy (server)

Configure generation strategy (pairwise vs exhaustive), sampler budget, and seed. Click Save to persist to `configs/coverage.json`.

Key knobs
- Mode: pairwise (recommended) or exhaustive
- t (pairwise depth): 2 (recommended)
- per_behavior_budget: e.g., 120
- sampler: rng_seed (reproducibility), per_behavior_total, min_per_domain

---

## Section 2: Dataset Generator

1) Select Domains/Behaviors (optional). Leave empty to cover all.
2) Options
   - Combined (per-domain + global) — recommended
   - Save to server — check when ready to write files
   - Overwrite — only if updating existing datasets
3) Preview coverage — shows counts per Domain × Behavior
4) Generate — dry-run (without Save) or write files (with Save)

Where files are written
- Per selected vertical under `datasets/<vertical>/` (e.g., `datasets/commerce/`)
- File naming: `coverage-<domain-slug>-combined-<version>.dataset.json` and matching `.golden.json`
- Hierarchical mode (optional): `datasets/<vertical>/<behavior>/<version>/...`

---

## Understanding formats (important)

The generator can emit two different formats depending on the API options. Most users use the Dataset + Golden format for running evaluations.

1) Dataset + Golden (used for testing)
- Generated when calling POST `/coverage/generate` with default options (as_array=false)
- Two files per dataset id:
  - `<id>.dataset.json` — the conversations to evaluate
  - `<id>.golden.json` — expected answers and outcomes

Dataset example (.dataset.json)
```json
{
  "dataset_id": "coverage-orders-returns-combined-1.0.0",
  "version": "1.0.0",
  "metadata": { "domain": "Orders & Returns", "difficulty": "mixed" },
  "conversations": [
    {
      "conversation_id": "ord-001",
      "title": "Refund request for damaged item",
      "turns": [
        { "role": "user", "text": "My order arrived damaged, I want a refund." },
        { "role": "assistant", "text": "I'm sorry to hear that. Can you share your order ID?" }
      ],
      "metadata": { "behavior": "Refund/Exchange/Cancellation" }
    }
  ]
}
```

Golden example (.golden.json)
```json
{
  "dataset_id": "coverage-orders-returns-combined-1.0.0",
  "version": "1.0.0",
  "entries": [
    {
      "conversation_id": "ord-001",
      "turns": [
        {
          "turn_index": 1,
          "expected": { "variants": [
            "I'm sorry to hear that. Can you share your order ID?",
            "Please provide your order ID so I can help."
          ]}
        }
      ],
      "final_outcome": { "decision": "ALLOW", "next_action": "issue_refund" }
    }
  ]
}
```

2) Combined Array (for batch/analysis use, not the test runner)
- Generated when calling POST `/coverage/generate` with `as_array: true`
- Saved under `datasets/<vertical>/arrays/combined_array-<version>.json`
- Structure is a flat array of scenario items with fields like `domain`, `behavior`, and `messages.u1/u2`.
- Example below is for arrays and is NOT the `.dataset.json` used by the runner:

```json
{
  "schema": "combined_array.v1",
  "version": "1.0.0",
  "items": [
    {
      "domain": "Orders & Returns",
      "behavior": "Refund/Exchange/Cancellation",
      "messages": {
        "u1": "I need help with my recent order.",
        "u2": "Actually, can I get store credit instead?"
      },
      "expected": { "outcome": "Allowed", "a2_canonical": "I will verify your order and process a refund..." }
    }
  ]
}
```

---

## Tips and troubleshooting

- Vertical selection: Use the header selector; files are written under that vertical.
- Flat vs Hierarchical paths: Configure in `configs/coverage.json` under `dataset_paths`.
- Duplicated vertical folder: Recent fix avoids `datasets/<vertical>/<vertical>/...` nesting.
- After changing .env via Settings, restart backend.

---

## Next steps

- Start a run from the Runs page using a generated dataset id
- View results in Reports; download JSON/CSV/HTML

---

## References

- User Guide: `../UserGuide.md`
- API: `/coverage/generate`, `/coverage/manifest`, `/coverage/settings`
````