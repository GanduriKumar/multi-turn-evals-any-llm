"""Tests for all report generation improvements."""

from __future__ import annotations

import json
from pathlib import Path

from eval_server.headless_engine import run_headless
from eval_server.reporting.report_generator import generate_html_report
from eval_server.reporting.markdown_export import generate_markdown_report
from eval_server.reporting.comparison_generator import generate_comparison_report
from eval_server.reporting.style_config import get_theme, THEMES
from eval_server.reporting.pdf_export import generate_pdf_report


class TestThemingAndStyling:
    """Tests for custom theming functionality."""

    def test_themes_available(self):
        """Test that all themes are available."""
        assert "default" in THEMES
        assert "dark" in THEMES
        assert "compact" in THEMES

    def test_get_theme_default(self):
        """Test getting default theme."""
        theme = get_theme("default")
        assert theme.name == "default"
        assert theme.primary_color == "#667eea"

    def test_get_theme_dark(self):
        """Test getting dark theme."""
        theme = get_theme("dark")
        assert theme.name == "dark"
        assert theme.text_color == "#f3f4f6"

    def test_get_theme_compact(self):
        """Test getting compact theme."""
        theme = get_theme("compact")
        assert theme.name == "compact"

    def test_get_theme_invalid_defaults_to_default(self):
        """Test that invalid theme name defaults to default."""
        theme = get_theme("nonexistent")
        assert theme.name == "default"


class TestHTMLReportWithTheme:
    """Tests for HTML report generation with theme support."""

    def test_html_report_with_default_theme(self, tmp_path: Path):
        """Test HTML report with default theme."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"
        run_headless(sample, output_dir=out_dir)

        results_path = out_dir / "results.json"
        report_path = generate_html_report(results_path, theme="default")

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "default" in content

    def test_html_report_with_dark_theme(self, tmp_path: Path):
        """Test HTML report with dark theme."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"
        run_headless(sample, output_dir=out_dir)

        results_path = out_dir / "results.json"
        report_path = generate_html_report(results_path, theme="dark")

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "dark" in content or "#1f2937" in content

    def test_html_report_includes_theme_name(self, tmp_path: Path):
        """Test that theme name is displayed in report."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"
        run_headless(sample, output_dir=out_dir)

        results_path = out_dir / "results.json"
        report_path = generate_html_report(results_path, theme="compact")

        content = report_path.read_text(encoding="utf-8")
        assert "Theme:" in content


class TestMarkdownReportExport:
    """Tests for Markdown report generation."""

    def test_markdown_report_created(self, tmp_path: Path):
        """Test that Markdown report is created."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"
        run_headless(sample, output_dir=out_dir)

        results_path = out_dir / "results.json"
        report_path = generate_markdown_report(results_path)

        assert report_path.exists()
        assert report_path.suffix == ".md"

    def test_markdown_report_has_sections(self, tmp_path: Path):
        """Test that Markdown report has all required sections."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"
        run_headless(sample, output_dir=out_dir)

        results_path = out_dir / "results.json"
        report_path = generate_markdown_report(results_path)

        content = report_path.read_text(encoding="utf-8")

        # Check for sections
        assert "# Evaluation Report" in content
        assert "## 📊 Summary" in content
        assert "## 📈 Results by Conversation" in content
        assert "| Metric | Value |" in content

    def test_markdown_report_has_metrics(self, tmp_path: Path):
        """Test that Markdown report includes metrics tables."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"
        run_headless(sample, output_dir=out_dir)

        results_path = out_dir / "results.json"
        report_path = generate_markdown_report(results_path)

        content = report_path.read_text(encoding="utf-8")

        # Should have some table content
        assert "|" in content  # Markdown table character
        assert "✓" in content or "✗" in content  # Pass/fail indicators

    def test_markdown_report_custom_path(self, tmp_path: Path):
        """Test Markdown report with custom output path."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"
        run_headless(sample, output_dir=out_dir)

        results_path = out_dir / "results.json"
        custom_path = tmp_path / "custom_report.md"
        report_path = generate_markdown_report(results_path, report_path=custom_path)

        assert report_path == custom_path
        assert report_path.exists()


class TestComparisonReports:
    """Tests for comparison report generation."""

    def test_comparison_report_created(self, tmp_path: Path):
        """Test that comparison report is created."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"

        # Create baseline run
        baseline_dir = tmp_path / "baseline"
        run_headless(sample, output_dir=baseline_dir)
        baseline_results = baseline_dir / "results.json"

        # Create current run
        current_dir = tmp_path / "current"
        run_headless(sample, output_dir=current_dir)
        current_results = current_dir / "results.json"

        # Generate comparison
        report_path = generate_comparison_report(current_results, baseline_results)

        assert report_path.exists()
        assert report_path.suffix == ".html"

    def test_comparison_report_has_content(self, tmp_path: Path):
        """Test that comparison report contains expected content."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"

        # Create baseline run
        baseline_dir = tmp_path / "baseline"
        run_headless(sample, output_dir=baseline_dir)
        baseline_results = baseline_dir / "results.json"

        # Create current run
        current_dir = tmp_path / "current"
        run_headless(sample, output_dir=current_dir)
        current_results = current_dir / "results.json"

        # Generate comparison
        report_path = generate_comparison_report(current_results, baseline_results)

        content = report_path.read_text(encoding="utf-8")

        # Check for expected sections
        assert "Comparison Report" in content
        assert "Run Comparison" in content
        assert "Score Changes by Conversation" in content
        assert "Baseline" in content
        assert "Current" in content

    def test_comparison_report_shows_changes(self, tmp_path: Path):
        """Test that comparison report shows score changes."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"

        # Create baseline and current runs (same config for deterministic comparison)
        baseline_dir = tmp_path / "baseline"
        run_headless(sample, output_dir=baseline_dir)
        baseline_results = baseline_dir / "results.json"

        current_dir = tmp_path / "current"
        run_headless(sample, output_dir=current_dir)
        current_results = current_dir / "results.json"

        # Generate comparison
        report_path = generate_comparison_report(current_results, baseline_results)

        content = report_path.read_text(encoding="utf-8")

        # Should have comparison data
        assert "↑" in content or "↓" in content or "→" in content  # Delta indicators


class TestPDFExport:
    """Tests for PDF report generation."""

    def test_pdf_generation_graceful_without_tools(self, tmp_path: Path):
        """Test that PDF generation fails gracefully without wkhtmltopdf."""
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"
        run_headless(sample, output_dir=out_dir)

        results_path = out_dir / "results.json"

        try:
            report_path = generate_pdf_report(results_path)
            # If PDF tools are available, verify file was created
            assert report_path.exists()
        except RuntimeError as e:
            # If tools not available, should raise informative error
            assert "No PDF generation tool" in str(e) or "failed" in str(e).lower()


class TestCLIIntegration:
    """Tests for CLI integration of report generation."""

    def test_cli_with_generate_report_flag(self, tmp_path: Path):
        """Test CLI with --generate-report flag."""
        from eval_server.headless_engine import main

        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"

        # Run with --generate-report flag
        args = [str(sample), "--output", str(out_dir), "--generate-report"]
        result = main(args)

        assert result == 0
        # Check that HTML report was created
        assert (out_dir / "report.html").exists()

    def test_cli_with_generate_markdown_flag(self, tmp_path: Path):
        """Test CLI with --generate-markdown flag."""
        from eval_server.headless_engine import main

        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"

        # Run with --generate-markdown flag
        args = [str(sample), "--output", str(out_dir), "--generate-markdown"]
        result = main(args)

        assert result == 0
        # Check that Markdown report was created
        assert (out_dir / "report.md").exists()

    def test_cli_with_theme_flag(self, tmp_path: Path):
        """Test CLI with --theme flag."""
        from eval_server.headless_engine import main

        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"

        # Run with --theme flag
        args = [str(sample), "--output", str(out_dir), "--generate-report", "--theme", "dark"]
        result = main(args)

        assert result == 0
        # Check that HTML report was created with dark theme
        report = out_dir / "report.html"
        assert report.exists()
        content = report.read_text(encoding="utf-8")
        assert "dark" in content

    def test_cli_with_comparison_flag(self, tmp_path: Path):
        """Test CLI with --generate-comparison flag."""
        from eval_server.headless_engine import main

        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"

        # Create baseline run
        baseline_dir = tmp_path / "baseline"
        args = [str(sample), "--output", str(baseline_dir)]
        main(args)

        # Create current run and generate comparison
        current_dir = tmp_path / "current"
        baseline_results = baseline_dir / "results.json"
        args = [str(sample), "--output", str(current_dir), "--generate-comparison", str(baseline_results)]
        result = main(args)

        assert result == 0
        # Check that comparison report was created
        assert (current_dir / "comparison.html").exists()

    def test_cli_multiple_reports_generation(self, tmp_path: Path):
        """Test CLI generating multiple report types at once."""
        from eval_server.headless_engine import main

        repo = Path(__file__).resolve().parents[2]
        sample = repo / "configs" / "runs" / "sample_run_config.yaml"
        out_dir = tmp_path / "out"

        # Run with multiple report flags
        args = [
            str(sample),
            "--output", str(out_dir),
            "--generate-report",
            "--generate-markdown",
            "--theme", "compact",
        ]
        result = main(args)

        assert result == 0
        # Check that both reports were created
        assert (out_dir / "report.html").exists()
        assert (out_dir / "report.md").exists()
