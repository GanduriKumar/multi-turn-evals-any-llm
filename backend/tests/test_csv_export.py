from __future__ import annotations

import csv
import json
from pathlib import Path

from eval_server.headless_engine import run_headless
from eval_server.reporting.csv_export import export_results_csv


def test_csv_export_from_results(tmp_path: Path):
    # Run headless to produce artifacts including results.json
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    assert results_path.exists(), "results.json must exist after headless run"

    # Export to CSV
    csv_path = export_results_csv(results_path)
    assert csv_path.exists(), "CSV export not created"

    # Basic column validation
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        # Required columns
        required = {
            "run_id",
            "dataset_id",
            "conversation_id",
            "model_name",
            "conv_score",
            "conv_pass",
            "turn_id",
            "weighted_score",
            "passed",
            "prompt",
            "response",
        }
        for col in required:
            assert col in headers, f"Missing required column: {col}"

        # There should be at least one metric_ column
        metric_cols = [h for h in headers if h.startswith("metric_")]
        assert metric_cols, "Expected at least one metric column"

        # Read a couple of rows and sanity check data presence
        rows = list(reader)
        assert rows, "CSV should contain at least one row"
        first = rows[0]
        assert first.get("run_id"), "run_id should be populated"
        assert first.get("dataset_id"), "dataset_id should be populated"
        assert first.get("conversation_id") is not None
        assert first.get("turn_id") is not None
        # Verify metric values are present (may be numeric strings)
        any_metric_value_present = any(first.get(m) not in (None, "") for m in metric_cols)
        assert any_metric_value_present, "Expected some metric values populated"
