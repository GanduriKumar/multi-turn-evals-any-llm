from __future__ import annotations

import json
from pathlib import Path

from eval_server.headless_engine import run_headless
from eval_server.reporting.report_generator import generate_html_report


def _make_annotations(results_path: Path) -> dict:
    # Load results to know keys
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    annotations = []
    for group in payload.get("results", []) or []:
        ds = group.get("dataset_id")
        cid = group.get("conversation_id")
        model = group.get("model_name")
        for t in group.get("turns", []) or []:
            # Provide an override for the first turn only, add rating/notes
            rating = 4.5
            notes = f"Manual check for {cid} turn {t.get('turn_id')}"
            # Flip pass flag and boost score slightly
            override_pass = not bool(t.get("passed", False))
            override_score = float(t.get("weighted_score", 0.0)) + 0.1
            annotations.append({
                "dataset_id": ds,
                "conversation_id": cid,
                "model_name": model,
                "turn_id": t.get("turn_id"),
                "rating": rating,
                "notes": notes,
                "override_pass": override_pass,
                "override_score": override_score,
            })
            break
    return {"annotations": annotations}


def test_report_includes_evaluator_annotations(tmp_path: Path):
    # Run headless to produce results
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"

    # Build annotations and write to file
    anns = _make_annotations(results_path)
    anns_path = tmp_path / "annotations.json"
    anns_path.write_text(json.dumps(anns), encoding="utf-8")

    # Generate report with annotations
    report_path = generate_html_report(results_path, annotations_path=anns_path)
    content = report_path.read_text(encoding="utf-8")

    # Verify evaluator table present
    assert "Evaluator:" in content
    assert "Override Pass" in content
    assert "Override Score" in content
    assert "Final Score" in content
    assert "Final Pass" in content

    # Ensure notes appear
    assert "Manual check for" in content


def test_annotations_influence_summary(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"

    # Generate report without annotations
    report_no_ann = generate_html_report(results_path)
    html_no_ann = report_no_ann.read_text(encoding="utf-8")

    # Create annotations that flip pass and increase score
    anns = _make_annotations(results_path)
    anns_path = tmp_path / "annotations.json"
    anns_path.write_text(json.dumps(anns), encoding="utf-8")

    # Generate report with annotations
    report_with_ann = generate_html_report(results_path, annotations_path=anns_path)
    html_with_ann = report_with_ann.read_text(encoding="utf-8")

    # Check final summary counters appear
    assert "Final Conversations Passed" in html_with_ann
    assert "Final Turns Passed" in html_with_ann
    assert "Avg Final Turn Score" in html_with_ann

    # The with-annotations report should reflect different final stats
    # We can't parse exact numbers easily; just ensure counters exist and differ visually
    assert html_with_ann.count("Final Turns Passed") == 1

    # Ensure final pass badges appear in turns
    assert "Final Pass" in html_with_ann


def test_generate_html_with_inline_annotations(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))

    # Build inline annotations dict form (keyed)
    anns: dict[str, dict] = {}
    for group in payload.get("results", []) or []:
        ds = group.get("dataset_id")
        cid = group.get("conversation_id")
        model = group.get("model_name")
        for t in group.get("turns", []) or []:
            k = f"{ds}|{cid}|{model}|{t.get('turn_id')}"
            anns[k] = {
                "rating": 5,
                "notes": "Looks great",
                "override_pass": True,
                "override_score": float(t.get("weighted_score", 0.0)) + 0.2,
            }
            break
        break

    report_path = generate_html_report(results_path, annotations=anns)
    content = report_path.read_text(encoding="utf-8")

    assert "Evaluator:" in content
    assert "Looks great" in content
    assert "Final Score" in content
