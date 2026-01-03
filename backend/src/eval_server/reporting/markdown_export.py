"""Markdown report generation for documentation and version control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping
import json
from datetime import datetime


def _escape_markdown(text: str | None) -> str:
    """Escape Markdown special characters."""
    if text is None:
        return ""
    text = str(text)
    # Escape common markdown chars
    for char in r"\`*_{}[]()#+-.!":
        text = text.replace(char, f"\\{char}")
    return text


def _format_value(val: Any) -> str:
    """Format a value for markdown display."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, bool):
        return "✓" if val else "✗"
    return str(val)


def _generate_markdown_content(payload: Mapping[str, Any]) -> str:
    """Generate markdown report content from results."""
    run = payload.get("run", {})
    results = payload.get("results", []) or []
    
    run_id = run.get("run_id", "unknown")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate statistics
    total_convs = len(results)
    total_turns = sum(len(r.get("turns", [])) for r in results)
    conv_passes = sum(1 for r in results if r.get("aggregate", {}).get("passed", False))
    turn_passes = sum(
        1 for r in results
        for t in r.get("turns", [])
        if t.get("passed", False)
    )
    
    conv_scores = [r.get("aggregate", {}).get("score", 0.0) for r in results]
    avg_conv_score = sum(conv_scores) / len(conv_scores) if conv_scores else 0.0
    
    turn_scores = [t.get("weighted_score", 0.0) for r in results for t in r.get("turns", [])]
    avg_turn_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
    
    datasets = run.get("datasets", [])
    models = run.get("models", [])
    
    md = f"""# Evaluation Report

**Run ID:** `{run_id}`  
**Generated:** {generated_at}

---

## 📊 Summary

| Metric | Value |
|--------|-------|
| Total Conversations | {total_convs} |
| Total Turns | {total_turns} |
| Conversations Passed | {conv_passes}/{total_convs} ({100*conv_passes//max(total_convs, 1)}%) |
| Turns Passed | {turn_passes}/{total_turns} ({100*turn_passes//max(total_turns, 1)}%) |
| Avg Conversation Score | {avg_conv_score:.4f} |
| Avg Turn Score | {avg_turn_score:.4f} |
| Datasets | {len(datasets)} |
| Models | {len(models)} |

### Datasets

"""
    for ds in datasets:
        md += f"- `{ds}`\n"
    
    md += f"\n### Models\n\n"
    for model in models:
        md += f"- {model}\n"
    
    md += "\n---\n\n## 📈 Results by Conversation\n\n"
    
    for group in results:
        ds = group.get("dataset_id", "?")
        conv_id = group.get("conversation_id", "?")
        model = group.get("model_name", "?")
        agg = group.get("aggregate", {})
        conv_score = agg.get("score", 0.0)
        conv_passed = agg.get("passed", False)
        turns = group.get("turns", []) or []
        
        status = "✓ PASS" if conv_passed else "✗ FAIL"
        md += f"### {_escape_markdown(conv_id)} ({status})\n\n"
        md += f"**Dataset:** `{_escape_markdown(ds)}`  \n"
        md += f"**Model:** {_escape_markdown(model)}  \n"
        md += f"**Score:** {conv_score:.4f}\n\n"
        
        # Turn details
        for turn in turns:
            turn_id = turn.get("turn_id", "?")
            prompt = turn.get("prompt", "")
            response = turn.get("response", "")
            weighted_score = turn.get("weighted_score", 0.0)
            turn_passed = turn.get("passed", False)
            metrics = turn.get("metrics", {})
            
            turn_status = "✓" if turn_passed else "✗"
            md += f"#### Turn {turn_id} {turn_status}\n\n"
            md += f"**Score:** {weighted_score:.4f}\n\n"
            
            md += "**Prompt:**\n```\n"
            md += prompt[:500]  # Truncate long prompts
            if len(prompt) > 500:
                md += "\n... (truncated)"
            md += "\n```\n\n"
            
            md += "**Response:**\n```\n"
            md += response[:500]  # Truncate long responses
            if len(response) > 500:
                md += "\n... (truncated)"
            md += "\n```\n\n"
            
            if metrics:
                md += "**Metrics:**\n\n"
                md += "| Metric | Score |\n"
                md += "|--------|-------|\n"
                for metric_name, metric_val in metrics.items():
                    md += f"| {_escape_markdown(metric_name)} | {_format_value(metric_val)} |\n"
                md += "\n"
    
    # Metrics summary table
    metric_names: set[str] = set()
    for group in results:
        for turn in group.get("turns", []) or []:
            metric_names.update(turn.get("metrics", {}).keys())
    
    if metric_names:
        md += "\n---\n\n## 📉 Metrics Summary\n\n"
        md += "| Conversation | Model | Turn | "
        for metric in sorted(metric_names):
            md += f"{_escape_markdown(metric)} | "
        md += "Weighted Score | Pass |\n"
        md += "|" + "---|" * (len(metric_names) + 5) + "\n"
        
        for group in results:
            conv_id = group.get("conversation_id", "?")
            model = group.get("model_name", "?")
            for turn in group.get("turns", []) or []:
                turn_id = turn.get("turn_id", "?")
                metrics = turn.get("metrics", {})
                weighted_score = turn.get("weighted_score", 0.0)
                passed = turn.get("passed", False)
                
                md += f"| {_escape_markdown(conv_id)} | {_escape_markdown(model)} | {_escape_markdown(str(turn_id))} | "
                for metric in sorted(metric_names):
                    val = metrics.get(metric)
                    md += f"{_format_value(val)} | "
                md += f"{_format_value(weighted_score)} | {'✓' if passed else '✗'} |\n"
    
    md += "\n---\n\n*Report generated by eval-server*\n"
    return md


def generate_markdown_report(results_json_path: Path, report_path: Path | None = None) -> Path:
    """Generate a Markdown report from consolidated results.json.

    Args:
        results_json_path: Path to results.json produced by headless engine.
        report_path: Optional output path for Markdown report. Defaults to report.md.

    Returns:
        Path to generated Markdown report.
    """
    with results_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    md_content = _generate_markdown_content(payload)

    if report_path is None:
        report_path = results_json_path.with_name("report.md")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    return report_path


__all__ = ["generate_markdown_report"]
