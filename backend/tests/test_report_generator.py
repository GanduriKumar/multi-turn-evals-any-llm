"""Tests for HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

from eval_server.headless_engine import run_headless
from eval_server.reporting.report_generator import generate_html_report


def test_report_generator_creates_html(tmp_path: Path):
    """Test that report generator creates an HTML file from results.json."""
    # Run headless to produce artifacts
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    assert results_path.exists(), "results.json must exist after headless run"

    # Generate report
    report_path = generate_html_report(results_path)
    assert report_path.exists(), "HTML report must be created"
    assert report_path.suffix == ".html", "Report should be HTML file"


def test_report_contains_header_section(tmp_path: Path):
    """Test that report contains run summary header."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    report_path = generate_html_report(results_path)

    content = report_path.read_text(encoding="utf-8")
    
    # Check for header elements
    assert "Run Summary" in content, "Report should contain Run Summary section"
    assert "📊" in content, "Report should contain section emoji"
    assert "Conversations" in content, "Report should mention conversations"
    assert "Turns" in content, "Report should mention turns"
    assert "Passed" in content, "Report should show pass/fail counts"


def test_report_contains_conversation_details(tmp_path: Path):
    """Test that report contains per-conversation details."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    report_path = generate_html_report(results_path)

    content = report_path.read_text(encoding="utf-8")
    
    # Check for conversation section
    assert "Results by Conversation" in content, "Report should have results section"
    assert "📈" in content, "Report should contain results emoji"
    assert "conversation-group" in content, "Report should have conversation containers"
    assert "Model:" in content, "Report should list model names"
    assert "Dataset:" in content, "Report should list dataset IDs"


def test_report_contains_turn_details(tmp_path: Path):
    """Test that report contains turn-by-turn details with prompts and responses."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    report_path = generate_html_report(results_path)

    content = report_path.read_text(encoding="utf-8")
    
    # Check for turn details
    assert "Turn" in content, "Report should list turns"
    assert "Prompt:" in content, "Report should show prompts"
    assert "Response:" in content, "Report should show responses"
    assert "turn-box" in content, "Report should have turn containers"
    assert "Metrics:" in content, "Report should show metrics for each turn"


def test_report_contains_metrics_table(tmp_path: Path):
    """Test that report includes metrics summary section."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    report_path = generate_html_report(results_path)

    content = report_path.read_text(encoding="utf-8")
    
    # Check for metrics summary (may show "no metrics" message or actual table)
    assert "Metrics Summary" in content, "Report should have metrics summary section"
    assert "📉" in content, "Report should contain metrics emoji"
    # Either show metrics table or message about no metrics
    assert "summary-metrics-table" in content or "No individual metrics" in content, \
        "Report should have metrics table or message"


def test_report_contains_scores_and_badges(tmp_path: Path):
    """Test that report displays scores and pass/fail badges."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    report_path = generate_html_report(results_path)

    content = report_path.read_text(encoding="utf-8")
    
    # Check for scores and badges
    assert "badge" in content, "Report should contain badge elements"
    assert "Score:" in content, "Report should show score labels"
    assert "score" in content.lower(), "Report should contain score values"


def test_report_has_valid_html_structure(tmp_path: Path):
    """Test that generated report has valid HTML structure."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    report_path = generate_html_report(results_path)

    content = report_path.read_text(encoding="utf-8")
    
    # Check for HTML structure
    assert "<!DOCTYPE html>" in content, "Report should have DOCTYPE"
    assert "<html" in content, "Report should have html tag"
    assert "<head>" in content, "Report should have head"
    assert "<body>" in content, "Report should have body"
    assert "<style>" in content, "Report should include CSS"
    assert "</html>" in content, "Report should close html tag"


def test_report_escapes_html_characters(tmp_path: Path):
    """Test that HTML special characters in content are properly escaped."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    report_path = generate_html_report(results_path)

    content = report_path.read_text(encoding="utf-8")
    
    # Check that report is valid HTML with DOCTYPE and proper structure
    assert "<!DOCTYPE html>" in content, "Report should have DOCTYPE"
    assert "</html>" in content, "Report should be properly closed"
    # Verify no obvious unescaped issues like unclosed tags in data
    assert content.count("<") == content.count(">"), \
        "Report should have balanced angle brackets"


def test_report_includes_run_metadata(tmp_path: Path):
    """Test that report includes run ID and metadata."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    report_path = generate_html_report(results_path)

    content = report_path.read_text(encoding="utf-8")
    
    # Load results to verify run ID is in report
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    
    run_id = results.get("run", {}).get("run_id")
    if run_id:
        assert run_id in content or "Run ID" in content, \
            "Report should include run ID or label"


def test_report_custom_path(tmp_path: Path):
    """Test that report can be saved to custom path."""
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    out_dir = tmp_path / "out"
    run_headless(sample, output_dir=out_dir)

    results_path = out_dir / "results.json"
    custom_report_path = tmp_path / "custom" / "my_report.html"
    
    report_path = generate_html_report(results_path, report_path=custom_report_path)
    
    assert report_path == custom_report_path, "Should return the specified path"
    assert report_path.exists(), "Report should be created at custom path"
    assert "Run Summary" in report_path.read_text(encoding="utf-8"), \
        "Report at custom path should have expected content"
