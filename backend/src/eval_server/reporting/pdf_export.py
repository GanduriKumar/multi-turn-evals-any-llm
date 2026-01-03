"""PDF report generation using wkhtmltopdf or HTML-to-PDF libraries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import json
import subprocess
import sys


def _check_wkhtmltopdf_available() -> bool:
    """Check if wkhtmltopdf is available on the system."""
    try:
        subprocess.run(
            ["wkhtmltopdf", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _try_import_pdfkit() -> bool:
    """Try to import pdfkit library."""
    try:
        import pdfkit  # type: ignore
        return True
    except ImportError:
        return False


def generate_pdf_report(
    results_json_path: Path,
    report_path: Path | None = None,
    theme: str = "default",
    html_report_path: Path | None = None
) -> Path:
    """Generate a PDF report from evaluation results.

    This function first generates an HTML report, then converts it to PDF using wkhtmltopdf
    or pdfkit if available.

    Args:
        results_json_path: Path to results.json produced by headless engine.
        report_path: Optional output path for PDF report. Defaults to report.pdf.
        theme: Theme name (default, dark, compact).
        html_report_path: Optional path to use for intermediate HTML report.

    Returns:
        Path to generated PDF report (or HTML if PDF generation unavailable).

    Raises:
        ImportError: If no PDF generation tool is available.
    """
    # First, generate HTML report
    from .report_generator import generate_html_report
    
    if html_report_path is None:
        html_report_path = results_json_path.with_name("report_temp.html")
    
    html_path = generate_html_report(results_json_path, report_path=html_report_path, theme=theme)
    
    # Determine PDF output path
    if report_path is None:
        report_path = results_json_path.with_name("report.pdf")
    
    # Try wkhtmltopdf first
    if _check_wkhtmltopdf_available():
        try:
            subprocess.run(
                ["wkhtmltopdf", str(html_path), str(report_path)],
                capture_output=True,
                timeout=30,
                check=True,
            )
            # Clean up temporary HTML
            try:
                html_path.unlink()
            except Exception:
                pass
            return report_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"wkhtmltopdf failed: {e.stderr.decode()}") from e
    
    # Try pdfkit with different backends
    if _try_import_pdfkit():
        try:
            import pdfkit  # type: ignore
            
            pdfkit.from_file(str(html_path), str(report_path))
            # Clean up temporary HTML
            try:
                html_path.unlink()
            except Exception:
                pass
            return report_path
        except Exception as e:
            raise RuntimeError(f"pdfkit failed: {e}") from e
    
    # Try weasyprint
    try:
        from weasyprint import HTML  # type: ignore
        
        HTML(str(html_path)).write_pdf(str(report_path))
        # Clean up temporary HTML
        try:
            html_path.unlink()
        except Exception:
            pass
        return report_path
    except ImportError:
        pass
    except Exception as e:
        raise RuntimeError(f"weasyprint failed: {e}") from e
    

    
    # No PDF generator available
    raise RuntimeError(
        "No PDF generation tool is available. Please install one of:\n"
        "  - wkhtmltopdf (system package)\n"
        "  - pdfkit: pip install pdfkit\n"
        "  - weasyprint: pip install weasyprint\n"
        f"HTML report has been generated at: {html_path}"
    )


__all__ = ["generate_pdf_report"]
