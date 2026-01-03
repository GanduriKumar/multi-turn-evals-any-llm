"""Report generation module producing human-readable HTML reports from evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import json
from datetime import datetime
from .style_config import get_theme, generate_css_for_theme


def _key(ds: str | None, cid: str | None, model: str | None, turn_id: str | int | None) -> str:
    return f"{ds}|{cid}|{model}|{turn_id}"


def _load_annotations(annotations_path: Path | None, annotations: Mapping[str, Any] | None) -> dict[str, dict]:
    """Load evaluator annotations from a file path or mapping.

    Supported formats:
    - List[ {dataset_id, conversation_id, model_name, turn_id, rating?, notes?, override_pass?, override_score?} ]
    - Dict[key -> {...}] where key == "dataset|conversation|model|turn"
    - Dict{"annotations": [...]} wrapper
    Returns a dict mapping the composite key to the annotation dict.
    """
    data: Any = None
    if annotations is not None:
        data = annotations
    elif annotations_path is not None and Path(annotations_path).exists():
        try:
            data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
        except Exception:
            data = None
    if data is None:
        return {}

    ann_map: dict[str, dict] = {}
    source = data.get("annotations") if isinstance(data, dict) and "annotations" in data else data
    if isinstance(source, list):
        for item in source:
            if not isinstance(item, dict):
                continue
            k = _key(item.get("dataset_id"), item.get("conversation_id"), item.get("model_name"), item.get("turn_id"))
            ann_map[k] = {
                "rating": item.get("rating"),
                "notes": item.get("notes"),
                "override_pass": item.get("override_pass"),
                "override_score": item.get("override_score"),
            }
    elif isinstance(source, dict):
        for k, v in source.items():
            if isinstance(v, dict):
                ann_map[str(k)] = {
                    "rating": v.get("rating"),
                    "notes": v.get("notes"),
                    "override_pass": v.get("override_pass"),
                    "override_score": v.get("override_score"),
                }
    return ann_map


def _apply_annotations(payload: dict[str, Any], ann_map: dict[str, dict]) -> dict[str, Any]:
    """Apply evaluator annotations into a copy of payload.

    Adds per-turn fields: evaluator, final_pass, final_weighted_score.
    Adds per-conversation aggregate_final: {avg_turn_score, passed_all}.
    """
    updated = json.loads(json.dumps(payload))  # deep copy
    for group in updated.get("results", []) or []:
        ds = group.get("dataset_id")
        cid = group.get("conversation_id")
        model = group.get("model_name")
        final_scores: list[float] = []
        final_passes: list[bool] = []
        for turn in group.get("turns", []) or []:
            k = _key(ds, cid, model, turn.get("turn_id"))
            ann = ann_map.get(k)
            original_pass = bool(turn.get("passed", False))
            original_score = float(turn.get("weighted_score", 0.0) or 0.0)
            final_pass = original_pass
            final_score = original_score
            evaluator: dict[str, Any] = {}
            if ann:
                evaluator = {
                    "rating": ann.get("rating"),
                    "notes": ann.get("notes"),
                    "override_pass": ann.get("override_pass"),
                    "override_score": ann.get("override_score"),
                }
                if ann.get("override_pass") is not None:
                    final_pass = bool(ann.get("override_pass"))
                override_score = ann.get("override_score")
                if override_score is not None:
                    try:
                        final_score = float(override_score)
                    except Exception:
                        pass
            turn["evaluator"] = evaluator
            turn["final_pass"] = final_pass
            turn["final_weighted_score"] = final_score
            final_scores.append(final_score)
            final_passes.append(final_pass)
        if final_scores:
            avg_final = sum(final_scores) / max(1, len(final_scores))
        else:
            avg_final = 0.0
        group["aggregate_final"] = {
            "avg_turn_score": avg_final,
            "passed_all": all(final_passes) if final_passes else bool(group.get("aggregate", {}).get("passed", False)),
        }
    return updated


def _escape_html(text: str | None) -> str:
    """Escape HTML special characters for safe display."""
    if text is None:
        return ""
    text = str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _format_metric_value(val: Any) -> str:
    """Format a metric value for display."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, bool):
        return "✓" if val else "✗"
    return str(val)


def _pass_fail_badge(passed: bool) -> str:
    """Generate a pass/fail badge HTML."""
    if passed:
        return '<span class="badge badge-pass">PASS</span>'
    else:
        return '<span class="badge badge-fail">FAIL</span>'


def _metric_bar(value: float, max_val: float = 1.0) -> str:
    """Generate a simple progress bar for metric values."""
    if max_val <= 0:
        percentage = 0
    else:
        percentage = min(100, int((value / max_val) * 100))
    
    bar_color = "green" if percentage >= 70 else "orange" if percentage >= 50 else "red"
    return f'<div class="metric-bar" style="width: {percentage}%; background-color: {bar_color};">{percentage}%</div>'


def _build_summary_section(payload: Mapping[str, Any]) -> str:
    """Build the summary section with run overview and statistics."""
    run = payload.get("run", {})
    summary = payload.get("summary", {})
    results = payload.get("results", []) or []

    run_id = run.get("run_id", "unknown")
    total_convs = len(results)
    total_turns = sum(len(r.get("turns", [])) for r in results)

    # Count passes (original)
    conv_passes = sum(1 for r in results if r.get("aggregate", {}).get("passed", False))
    turn_passes = sum(
        1 for r in results
        for t in r.get("turns", [])
        if t.get("passed", False)
    )
    # Count passes (final with overrides if present)
    final_turn_passes = sum(
        1 for r in results
        for t in r.get("turns", [])
        if t.get("final_pass", t.get("passed", False))
    )
    final_conv_passes = sum(
        1 for r in results
        if (r.get("aggregate_final", {}) or {}).get("passed_all", r.get("aggregate", {}).get("passed", False))
    )

    # Average scores
    conv_scores = [r.get("aggregate", {}).get("score", 0.0) for r in results]
    avg_conv_score = sum(conv_scores) / len(conv_scores) if conv_scores else 0.0

    turn_scores = [t.get("weighted_score", 0.0) for r in results for t in r.get("turns", [])]
    avg_turn_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0

    # Final averages
    final_turn_scores = [t.get("final_weighted_score", t.get("weighted_score", 0.0)) for r in results for t in r.get("turns", [])]
    avg_final_turn_score = sum(final_turn_scores) / len(final_turn_scores) if final_turn_scores else 0.0

    datasets = run.get("datasets", [])
    models = run.get("models", [])

    html = f"""
    <section id="summary" class="section">
        <h2>📊 Run Summary</h2>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Run ID</h3>
                <p class="mono">{_escape_html(run_id)}</p>
            </div>
            <div class="summary-card">
                <h3>Total Conversations</h3>
                <p class="stat-value">{total_convs}</p>
            </div>
            <div class="summary-card">
                <h3>Total Turns</h3>
                <p class="stat-value">{total_turns}</p>
            </div>
            <div class="summary-card">
                <h3>Conversations Passed</h3>
                <p class="stat-value">{conv_passes}/{total_convs}</p>
            </div>
            <div class="summary-card">
                <h3>Turns Passed</h3>
                <p class="stat-value">{turn_passes}/{total_turns}</p>
            </div>
            <div class="summary-card">
                <h3>Final Conversations Passed</h3>
                <p class="stat-value">{final_conv_passes}/{total_convs}</p>
            </div>
            <div class="summary-card">
                <h3>Final Turns Passed</h3>
                <p class="stat-value">{final_turn_passes}/{total_turns}</p>
            </div>
            <div class="summary-card">
                <h3>Avg Conversation Score</h3>
                <p class="stat-value">{avg_conv_score:.4f}</p>
            </div>
            <div class="summary-card">
                <h3>Avg Turn Score</h3>
                <p class="stat-value">{avg_turn_score:.4f}</p>
            </div>
            <div class="summary-card">
                <h3>Avg Final Turn Score</h3>
                <p class="stat-value">{avg_final_turn_score:.4f}</p>
            </div>
            <div class="summary-card">
                <h3>Datasets</h3>
                <p>{len(datasets)} dataset(s)</p>
            </div>
            <div class="summary-card">
                <h3>Models</h3>
                <p>{len(models)} model(s): {", ".join(models)}</p>
            </div>
        </div>
        
        <div class="summary-details">
            <h3>Datasets</h3>
            <ul>
"""
    for ds in datasets:
        html += f"                <li><code>{_escape_html(str(ds))}</code></li>\n"
    html += "            </ul>\n        </div>\n    </section>\n"
    
    return html


def _build_results_by_conversation(payload: Mapping[str, Any]) -> str:
    """Build detailed results grouped by conversation."""
    results = payload.get("results", []) or []
    
    html = """
    <section id="results" class="section">
        <h2>📈 Results by Conversation</h2>
"""
    
    for group in results:
        ds = group.get("dataset_id", "unknown")
        conv_id = group.get("conversation_id", "unknown")
        model = group.get("model_name", "unknown")
        agg = group.get("aggregate", {})
        conv_score = agg.get("score", 0.0)
        conv_passed = agg.get("passed", False)
        turns = group.get("turns", []) or []

        html += f"""
        <div class="conversation-group">
            <div class="conversation-header">
                <h3>Conversation: {_escape_html(conv_id)}</h3>
                <div class="conversation-meta">
                    <span class="meta-item"><strong>Dataset:</strong> {_escape_html(ds)}</span>
                    <span class="meta-item"><strong>Model:</strong> {_escape_html(model)}</span>
                    <span class="meta-item"><strong>Score:</strong> {conv_score:.4f}</span>
                    <span class="meta-item">{_pass_fail_badge(conv_passed)}</span>
                </div>
            </div>

            <div class="turns-container">
"""
        
        # Build turn-by-turn details
        for turn in turns:
            turn_id = turn.get("turn_id", "?")
            prompt = turn.get("prompt", "")
            response = turn.get("response", "")
            weighted_score = turn.get("weighted_score", 0.0)
            turn_passed = turn.get("passed", False)
            metrics = turn.get("metrics", {})
            weights = turn.get("weights", {})

            html += f"""
                <div class="turn-box">
                    <div class="turn-header">
                        <h4>Turn {_escape_html(str(turn_id))}</h4>
                        <div class="turn-badges">
                            <span class="turn-score">Score: {weighted_score:.4f}</span>
                            {_pass_fail_badge(turn_passed)}
                        </div>
                    </div>

                    <div class="turn-content">
                        <div class="exchange">
                            <div class="message prompt">
                                <strong>Prompt:</strong>
                                <p>{_escape_html(prompt)}</p>
                            </div>
                            <div class="message response">
                                <strong>Response:</strong>
                                <p>{_escape_html(response)}</p>
                            </div>
                        </div>

                        <div class="turn-metrics">
                            <strong>Metrics:</strong>
                            <table class="metrics-table">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Score</th>
                                        <th>Weight</th>
                                        <th>Contribution</th>
                                    </tr>
                                </thead>
                                <tbody>
"""
            
            # Metrics detail rows
            for metric_name, metric_val in metrics.items():
                weight = weights.get(metric_name, 1.0)
                contribution = float(metric_val) * float(weight) if metric_val is not None else 0.0
                html += f"""
                                    <tr>
                                        <td><strong>{_escape_html(metric_name)}</strong></td>
                                        <td>{_format_metric_value(metric_val)}</td>
                                        <td>{_format_metric_value(weight)}</td>
                                        <td>{_format_metric_value(contribution)}</td>
                                    </tr>
"""
            
            html += f"""
                                </tbody>
                            </table>
                        </div>

                        <div class="turn-metrics">
                            <strong>Evaluator:</strong>
                            <table class="metrics-table">
                                <thead>
                                    <tr>
                                        <th>Rating</th>
                                        <th>Notes</th>
                                        <th>Override Pass</th>
                                        <th>Override Score</th>
                                        <th>Final Score</th>
                                        <th>Final Pass</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>{_format_metric_value((turn.get('evaluator') or {}).get('rating'))}</td>
                                        <td>{_escape_html((turn.get('evaluator') or {}).get('notes'))}</td>
                                        <td>{_format_metric_value((turn.get('evaluator') or {}).get('override_pass'))}</td>
                                        <td>{_format_metric_value((turn.get('evaluator') or {}).get('override_score'))}</td>
                                        <td>{_format_metric_value(turn.get('final_weighted_score', weighted_score))}</td>
                                        <td>{_pass_fail_badge(bool(turn.get('final_pass', turn_passed)))}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </section>
"""
    
    return html


def _build_metrics_summary(payload: Mapping[str, Any]) -> str:
    """Build a summary table of all metrics across all turns."""
    results = payload.get("results", []) or []
    
    # Collect all metrics
    metric_names: set[str] = set()
    for group in results:
        for turn in group.get("turns", []) or []:
            metric_names.update(turn.get("metrics", {}).keys())
    
    metric_names_list = sorted(list(metric_names))

    # Always show the metrics section
    html = """
    <section id="metrics" class="section">
        <h2>📉 Metrics Summary</h2>
"""
    
    # If no metrics, show a message
    if not metric_names_list:
        html += "        <p><em>No individual metrics recorded for this evaluation.</em></p>\n"
    else:
        # Build table if metrics exist
        html += """        <table class="summary-metrics-table">
            <thead>
                <tr>
                    <th>Conversation</th>
                    <th>Model</th>
                    <th>Turn</th>
"""
        
        for metric in metric_names_list:
            html += f"                    <th>{_escape_html(metric)}</th>\n"
        
        html += """                    <th>Weighted Score</th>
                    <th>Pass</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for group in results:
            conv_id = group.get("conversation_id", "?")
            model = group.get("model_name", "?")
            for turn in group.get("turns", []) or []:
                turn_id = turn.get("turn_id", "?")
                metrics = turn.get("metrics", {})
                weighted_score = turn.get("weighted_score", 0.0)
                passed = turn.get("passed", False)
                
                html += f"""
                <tr>
                    <td>{_escape_html(conv_id)}</td>
                    <td>{_escape_html(model)}</td>
                    <td>{_escape_html(str(turn_id))}</td>
"""
                
                for metric in metric_names_list:
                    val = metrics.get(metric)
                    html += f"                    <td>{_format_metric_value(val)}</td>\n"
                
                html += f"""                    <td>{_format_metric_value(weighted_score)}</td>
                    <td>{_pass_fail_badge(passed)}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
    
    html += """    </section>
"""
    
    return html


def _build_html_document(payload: Mapping[str, Any], theme_name: str = "default") -> str:
    """Build the complete HTML document with theme support."""
    theme = get_theme(theme_name)
    theme_css = generate_css_for_theme(theme)
    run = payload.get("run", {})
    run_id = run.get("run_id", "evaluation-report")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report - {_escape_html(run_id)}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header .meta {{
            font-size: 0.95em;
            opacity: 0.9;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        }}
        
        .section h2 {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .section h3 {{
            font-size: 1.3em;
            margin: 15px 0 10px 0;
            color: #444;
        }}
        
        .section h4 {{
            font-size: 1.1em;
            margin: 10px 0 5px 0;
            color: #555;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .summary-card h3 {{
            font-size: 0.9em;
            font-weight: 600;
            opacity: 0.9;
            margin-bottom: 10px;
            color: white;
        }}
        
        .summary-card p {{
            margin: 0;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 0;
        }}
        
        .mono {{
            font-family: "Monaco", "Courier New", monospace;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 0 5px 0 0;
        }}
        
        .badge-pass {{
            background-color: #10b981;
            color: white;
        }}
        
        .badge-fail {{
            background-color: #ef4444;
            color: white;
        }}
        
        .conversation-group {{
            background: #f9fafb;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}
        
        .conversation-header {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        .conversation-header h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
        }}
        
        .conversation-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            font-size: 0.95em;
        }}
        
        .meta-item {{
            background: white;
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid #e5e7eb;
        }}
        
        .turns-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .turn-box {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 15px;
        }}
        
        .turn-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #f3f4f6;
        }}
        
        .turn-header h4 {{
            margin: 0;
        }}
        
        .turn-badges {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .turn-score {{
            background: #f3f4f6;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: 600;
        }}
        
        .turn-content {{
            margin-bottom: 20px;
        }}
        
        .exchange {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        @media (max-width: 768px) {{
            .exchange {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .message {{
            padding: 15px;
            border-radius: 6px;
            background: #f9fafb;
        }}
        
        .message.prompt {{
            border-left: 4px solid #667eea;
        }}
        
        .message.response {{
            border-left: 4px solid #10b981;
        }}
        
        .message strong {{
            display: block;
            margin-bottom: 10px;
            color: #333;
        }}
        
        .message p {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .turn-metrics {{
            background: #f9fafb;
            padding: 15px;
            border-radius: 6px;
        }}
        
        .turn-metrics strong {{
            display: block;
            margin-bottom: 10px;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }}
        
        .metrics-table th {{
            background: #f3f4f6;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e5e7eb;
        }}
        
        .metrics-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        .metrics-table tr:hover {{
            background: #f9fafb;
        }}
        
        .summary-metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            max-height: 600px;
            overflow-y: auto;
        }}
        
        .summary-metrics-table th {{
            background: #f3f4f6;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e5e7eb;
            position: sticky;
            top: 0;
        }}
        
        .summary-metrics-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        .summary-metrics-table tr:hover {{
            background: #f9fafb;
        }}
        
        .summary-details {{
            margin-top: 20px;
            padding: 15px;
            background: #f9fafb;
            border-radius: 6px;
        }}
        
        .summary-details ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        
        .summary-details li {{
            margin: 5px 0;
            font-family: "Monaco", "Courier New", monospace;
            font-size: 0.9em;
        }}
        
        .metric-bar {{
            display: inline-block;
            height: 20px;
            border-radius: 3px;
            color: white;
            font-size: 0.8em;
            line-height: 20px;
            text-align: center;
            min-width: 30px;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            margin-top: 40px;
        }}
        
        .table-responsive {{
            overflow-x: auto;
        }}
        
        code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Monaco", "Courier New", monospace;
            font-size: 0.9em;
        }}
        
        {theme_css}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📋 Evaluation Report</h1>
            <div class="meta">
                <p><strong>Run ID:</strong> {_escape_html(run_id)}</p>
                <p><strong>Generated:</strong> {generated_at}</p>
                <p><strong>Theme:</strong> {theme.name}</p>
            </div>
        </header>

        {_build_summary_section(payload)}
        {_build_results_by_conversation(payload)}
        {_build_metrics_summary(payload)}
        
        <footer>
            <p>Report generated by eval-server on {generated_at}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    return html


def generate_html_report(
    results_json_path: Path,
    report_path: Path | None = None,
    theme: str = "default",
    *,
    annotations_path: Path | None = None,
    annotations: Mapping[str, Any] | None = None,
) -> Path:
    """Generate an HTML report from consolidated results.json.

    Args:
        results_json_path: Path to results.json produced by headless engine.
        report_path: Optional output path for HTML report. Defaults to results.html in same directory.
        theme: Theme name (default, dark, compact).

    Returns:
        Path to generated HTML report.
    """
    # Load results
    with results_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Load and apply evaluator annotations if provided
    ann_map = _load_annotations(annotations_path, annotations)
    if ann_map:
        payload = _apply_annotations(payload, ann_map)

    # Generate HTML with theme
    html_content = _build_html_document(payload, theme_name=theme)

    # Determine output path
    if report_path is None:
        report_path = results_json_path.with_name("report.html")
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path


__all__ = ["generate_html_report"]
