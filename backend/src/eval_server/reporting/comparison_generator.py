"""Comparison report generation for comparing two evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import json
from datetime import datetime


def _compare_scores(baseline: float, current: float) -> Tuple[str, str]:
    """Compare two scores and return delta and direction."""
    delta = current - baseline
    direction = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "→"
    color = "green" if delta > 0.001 else "red" if delta < -0.001 else "gray"
    return f"{direction} {delta:+.4f}", color


def _generate_comparison_html(baseline_payload: Mapping[str, Any], current_payload: Mapping[str, Any]) -> str:
    """Generate HTML comparison report."""
    baseline_run = baseline_payload.get("run", {})
    current_run = current_payload.get("run", {})
    baseline_results = baseline_payload.get("results", []) or []
    current_results = current_payload.get("results", []) or []
    
    # Index results for easier lookup
    baseline_map = {
        f"{r.get('dataset_id')}|{r.get('conversation_id')}|{r.get('model_name')}": r
        for r in baseline_results
    }
    current_map = {
        f"{r.get('dataset_id')}|{r.get('conversation_id')}|{r.get('model_name')}": r
        for r in current_results
    }
    
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparison Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.6; color: #333; background: #f5f7fa; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; border-radius: 8px; margin-bottom: 30px; }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .section {{ background: white; padding: 30px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
        .section h2 {{ font-size: 1.8em; margin-bottom: 20px; color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .run-info {{ background: #f9fafb; padding: 20px; border-radius: 6px; border-left: 4px solid #667eea; }}
        .run-info h3 {{ margin-bottom: 10px; }}
        .run-info p {{ margin: 5px 0; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 10px; }}
        th {{ background: #f3f4f6; padding: 10px; text-align: left; font-weight: 600; border-bottom: 2px solid #e5e7eb; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
        tr:hover {{ background: #f9fafb; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 600; }}
        .improved {{ background: #d1fae5; color: #065f46; }}
        .degraded {{ background: #fee2e2; color: #991b1b; }}
        .unchanged {{ background: #f3f4f6; color: #374151; }}
        .arrow-up {{ color: #10b981; }}
        .arrow-down {{ color: #ef4444; }}
        .arrow-right {{ color: #999; }}
        footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Comparison Report</h1>
            <p><strong>Generated:</strong> {generated_at}</p>
        </header>
        
        <section class="section">
            <h2>Run Comparison</h2>
            <div class="comparison-grid">
                <div class="run-info">
                    <h3>Baseline</h3>
                    <p><strong>Run ID:</strong> <code>{baseline_run.get("run_id", "N/A")}</code></p>
                    <p><strong>Models:</strong> {", ".join(baseline_run.get("models", []))}</p>
                </div>
                <div class="run-info">
                    <h3>Current</h3>
                    <p><strong>Run ID:</strong> <code>{current_run.get("run_id", "N/A")}</code></p>
                    <p><strong>Models:</strong> {", ".join(current_run.get("models", []))}</p>
                </div>
            </div>
        </section>
        
        <section class="section">
            <h2>Score Changes by Conversation</h2>
            <table>
                <thead>
                    <tr>
                        <th>Conversation</th>
                        <th>Model</th>
                        <th>Baseline Score</th>
                        <th>Current Score</th>
                        <th>Change</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Compare results
    for key in sorted(set(baseline_map.keys()) | set(current_map.keys())):
        baseline_entry = baseline_map.get(key)
        current_entry = current_map.get(key)
        
        if baseline_entry and current_entry:
            baseline_score = baseline_entry.get("aggregate", {}).get("score", 0.0)
            current_score = current_entry.get("aggregate", {}).get("score", 0.0)
            
            delta, color = _compare_scores(baseline_score, current_score)
            
            parts = key.split("|")
            conv_id, model = parts[1], parts[2]
            
            status_class = "improved" if current_score > baseline_score + 0.001 else \
                          "degraded" if current_score < baseline_score - 0.001 else \
                          "unchanged"
            
            html += f"""
                    <tr>
                        <td>{conv_id}</td>
                        <td>{model}</td>
                        <td>{baseline_score:.4f}</td>
                        <td>{current_score:.4f}</td>
                        <td><span class="badge {status_class}">{delta}</span></td>
                        <td>{"✓ Improved" if status_class == "improved" else "✗ Degraded" if status_class == "degraded" else "→ Same"}</td>
                    </tr>
"""
        elif baseline_entry:
            baseline_score = baseline_entry.get("aggregate", {}).get("score", 0.0)
            parts = key.split("|")
            conv_id, model = parts[1], parts[2]
            html += f"""
                    <tr>
                        <td>{conv_id}</td>
                        <td>{model}</td>
                        <td>{baseline_score:.4f}</td>
                        <td>—</td>
                        <td><span class="badge degraded">↓ Missing</span></td>
                        <td>✗ Removed</td>
                    </tr>
"""
        elif current_entry:
            current_score = current_entry.get("aggregate", {}).get("score", 0.0)
            parts = key.split("|")
            conv_id, model = parts[1], parts[2]
            html += f"""
                    <tr>
                        <td>{conv_id}</td>
                        <td>{model}</td>
                        <td>—</td>
                        <td>{current_score:.4f}</td>
                        <td><span class="badge improved">↑ New</span></td>
                        <td>✓ Added</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </section>
        
        <footer>
            <p>Comparison report generated by eval-server</p>
        </footer>
    </div>
</body>
</html>
"""
    return html


def generate_comparison_report(current_results_path: Path, baseline_results_path: Path, report_path: Path | None = None) -> Path:
    """Generate a comparison report between two evaluation runs.

    Args:
        current_results_path: Path to current results.json.
        baseline_results_path: Path to baseline results.json.
        report_path: Optional output path for comparison report. Defaults to comparison.html.

    Returns:
        Path to generated comparison report.
    """
    # Load both results
    with current_results_path.open("r", encoding="utf-8") as f:
        current_payload = json.load(f)
    
    with baseline_results_path.open("r", encoding="utf-8") as f:
        baseline_payload = json.load(f)

    # Generate HTML
    html_content = _generate_comparison_html(baseline_payload, current_payload)

    if report_path is None:
        report_path = current_results_path.with_name("comparison.html")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path


__all__ = ["generate_comparison_report"]
