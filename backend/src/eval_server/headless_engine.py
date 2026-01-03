from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, List

from .config.run_config_loader import RunConfig, load_run_config
from .orchestrator import OrchestratorSummary, evaluate_run
from .utils.run_id import compute_run_id
from .runner.turn_runner import TurnRunner
from .data.loader import load_conversation, load_golden
from .data.golden_access import index_golden
from .scoring.normalizer import normalize_turn_output
from .scoring.turn_scoring import score_turn_canonical
from .scoring.thresholds import ThresholdPolicy, evaluate_turn_thresholds, evaluate_conversation_thresholds
from .evaluation.weights import aggregate_conversation
from .metrics.regression import score_regressions
from .reporting.results_writer import assemble_results, write_results_json
from .reporting.report_generator import generate_html_report
from .reporting.markdown_export import generate_markdown_report
from .reporting.comparison_generator import generate_comparison_report
from .reporting.pdf_export import generate_pdf_report


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_output_dir(rc: RunConfig, override: Optional[str | Path]) -> Path:
    if override:
        return Path(override)
    if rc.output_dir:
        # Embed run id in output dir for uniqueness
        rid = compute_run_id(rc)
        return Path(rc.output_dir) / rid
    # Fallback
    rid = compute_run_id(rc)
    return Path("runs") / rid


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _serialize_summary(summary: OrchestratorSummary) -> Dict[str, Any]:
    # dataclasses.asdict converts nested dataclasses to plain dicts
    return asdict(summary)


def _build_score_map(summary_json: Dict[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for r in (summary_json.get("results") or []):
        ds = r.get("dataset_id")
        cid = r.get("conversation_id")
        model = r.get("model_name")
        key = f"{ds}|{cid}|{model}"
        try:
            scores[key] = float(r.get("score", 0.0))
        except Exception:
            scores[key] = 0.0
    return scores


def _load_baseline_scores(path: Union[str, Path]) -> Optional[Dict[str, float]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _build_score_map(data)
    except Exception:
        return None


def run_headless(run: RunConfig | str | Path, *, output_dir: str | Path | None = None, baseline_summary: str | Path | None = None) -> Path:
    """Run evaluations from a run configuration and write artifacts.

    Returns the output directory path used.
    """
    rc = load_run_config(run) if isinstance(run, (str, Path)) else run
    out_dir = _resolve_output_dir(rc, output_dir)
    _ensure_dir(out_dir)

    # Optional: adjust concurrency via CLI override is handled in main()
    cancel = threading.Event()

    # Capture progress events for logs
    progress_events: List[Dict[str, Any]] = []
    def on_progress(evt: Dict[str, Any]) -> None:
        progress_events.append(evt)

    summary: OrchestratorSummary = evaluate_run(rc, cancel_event=cancel, on_progress=on_progress)
    summary_json = _serialize_summary(summary)
    # store summary and a simple manifest
    summary_path = Path(out_dir) / "summary.json"
    _write_json(summary_path, summary_json)
    _write_json(Path(out_dir) / "manifest.json", {"run_id": compute_run_id(rc), "datasets": [d.conversation for d in rc.datasets]})

    # Prepare subdirectories for artifacts
    inputs_dir = Path(out_dir) / "inputs"
    logs_dir = Path(out_dir) / "logs"
    scores_dir = Path(out_dir) / "scores"
    for d in (inputs_dir, logs_dir, scores_dir):
        _ensure_dir(d)

    # Persist run config used (best effort)
    try:
        _write_json(inputs_dir / "run_config.json", asdict(rc))
    except Exception:
        pass
    # Persist dataset definitions for traceability
    try:
        _write_json(inputs_dir / "datasets.json", {"datasets": [asdict(ds) for ds in rc.datasets]})
    except Exception:
        pass

    # Additionally, generate and store per-turn raw outputs as JSONL using TurnRunner
    raw_rows: list[Dict[str, Any]] = []
    normalized_rows: list[Dict[str, Any]] = []
    turn_scores_rows: list[Dict[str, Any]] = []
    conversation_scores: list[Dict[str, Any]] = []

    # Build thresholds for scoring
    rt = rc.thresholds.__dict__ if rc.thresholds else {}
    metric_thresholds: Dict[str, float] = dict((rt.get("metric") or {}))
    default_turn_thr = float(rt.get("turn_pass") or 0.5)
    default_conv_thr = float(rt.get("conversation_pass") or default_turn_thr)
    tpolicy = ThresholdPolicy(require="all", pass_ratio=1.0, default_threshold=default_turn_thr)
    for ds in rc.datasets:
        conv = load_conversation(ds.conversation)
        conv_id = str(conv.get("conversation_id"))
        gold = load_golden(ds.golden)
        expected_index = index_golden(gold)
        for m in rc.models:
            runner = TurnRunner(m.provider, **(m.params or {}))
            policy = rc.truncation.strategy if rc.truncation else None
            trunc_params: Dict[str, int] = {}
            results = runner.run(conv, truncation_policy=policy, truncation_params=trunc_params)
            # Collect for conversation aggregation
            per_turn_weighted: Dict[str, float] = {}
            per_turn_weights: Dict[str, float] = {}
            for r in results:
                row: Dict[str, Any] = {
                    "dataset_id": ds.id,
                    "conversation_id": conv_id,
                    "turn_id": r.turn_id,
                    "model_name": m.name,
                    "provider": m.provider,
                    "prompt": r.prompt,
                    "response": r.response,
                    "context": r.context,
                    "error": r.error,
                }
                if r.llm_request is not None:
                    row["llm_request"] = r.llm_request.model_dump(mode="json")
                if r.llm_response is not None:
                    row["llm_response"] = r.llm_response.model_dump(mode="json")
                raw_rows.append(row)

                # Normalized canonical record
                canonical = normalize_turn_output(r.response or "", r.context or [])
                normalized_rows.append({
                    "dataset_id": ds.id,
                    "conversation_id": conv_id,
                    "turn_id": r.turn_id,
                    "model_name": m.name,
                    "canonical": canonical,
                })

                # Per-turn scoring and threshold pass
                expected = expected_index.get((conv_id, str(r.turn_id)), {})
                scores_by_metric, _ = score_turn_canonical(canonical, expected, thresholds=metric_thresholds)
                weights = expected.get("weights") or {}
                passed, _ = evaluate_turn_thresholds(scores_by_metric, metric_thresholds, weights=weights, policy=tpolicy)
                # Weighted score
                acc = 0.0
                totalw = 0.0
                for k, v in scores_by_metric.items():
                    w = float(weights.get(k, 1.0))
                    acc += w * float(v)
                    totalw += w
                weighted_score = (acc / totalw) if totalw > 0 else 0.0
                turn_scores_rows.append({
                    "dataset_id": ds.id,
                    "conversation_id": conv_id,
                    "turn_id": r.turn_id,
                    "model_name": m.name,
                    "scores": scores_by_metric,
                    "weights": weights,
                    "weighted_score": weighted_score,
                    "passed": passed,
                })
                per_turn_weighted[str(r.turn_id)] = weighted_score
                per_turn_weights[str(r.turn_id)] = float(expected.get("turn_weight", 1.0) or 1.0)

            # Conversation-level aggregation and thresholds
            conv_score = aggregate_conversation(per_turn_weighted, per_turn_weights if per_turn_weights else None)
            cpol = ThresholdPolicy(require="all", pass_ratio=1.0, default_threshold=default_conv_thr)
            conv_pass, _ = evaluate_conversation_thresholds({"score": conv_score}, thresholds=metric_thresholds, policy=cpol)
            conversation_scores.append({
                "dataset_id": ds.id,
                "conversation_id": conv_id,
                "model_name": m.name,
                "score": conv_score,
                "passed": conv_pass,
            })

    _write_jsonl(Path(out_dir) / "raw_outputs.jsonl", raw_rows)
    _write_jsonl(Path(out_dir) / "normalized.jsonl", normalized_rows)
    _write_jsonl(Path(scores_dir) / "turn_scores.jsonl", turn_scores_rows)
    _write_json(Path(scores_dir) / "conversation_scores.json", {"results": conversation_scores})
    # Logs
    _write_jsonl(Path(logs_dir) / "progress.jsonl", progress_events)
    with (Path(logs_dir) / "run.log").open("w", encoding="utf-8") as lf:
        lf.write(f"events={len(progress_events)}\n")

    # Machine-readable consolidated results
    run_info = {"run_id": compute_run_id(rc), "datasets": [d.conversation for d in rc.datasets], "models": [m.name for m in rc.models]}
    consolidated = assemble_results(
        run_info=run_info,
        summary=summary_json,
        raw_rows=raw_rows,
        normalized_rows=normalized_rows,
        turn_scores_rows=turn_scores_rows,
        conversation_scores=conversation_scores,
    )
    write_results_json(Path(out_dir) / "results.json", consolidated)

    # Optional regression comparison if baseline provided
    if baseline_summary:
        baseline_scores = _load_baseline_scores(baseline_summary)
        if baseline_scores is not None:
            current_scores = _build_score_map(summary_json)
            reg_score, reg_info = score_regressions(baseline_scores, current_scores, threshold=0.0, direction="higher_is_better")
            _write_json(Path(out_dir) / "regression.json", {"score": reg_score, **reg_info, "baseline": str(baseline_summary)})
    return out_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run headless evaluation from a config file")
    parser.add_argument("config", help="Path to run configuration (YAML or JSON)")
    parser.add_argument("--output", help="Override output directory", default=None)
    parser.add_argument("--max-workers", type=int, default=None, help="Override max workers (global)")
    parser.add_argument("--baseline", type=str, default=None, help="Path to a baseline summary.json to compare for regressions")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML report after run")
    parser.add_argument("--generate-markdown", action="store_true", help="Generate Markdown report after run")
    parser.add_argument("--generate-comparison", type=str, default=None, help="Generate comparison report vs another results.json")
    parser.add_argument("--generate-pdf", action="store_true", help="Generate PDF report (requires wkhtmltopdf)")
    parser.add_argument("--theme", type=str, default="default", choices=["default", "dark", "compact"], help="Report theme")

    args = parser.parse_args(argv)

    # Load config to optionally override concurrency
    rc = load_run_config(args.config)
    if args.max_workers is not None and args.max_workers > 0:
        # Rebuild RunConfig with overridden concurrency
        conc = None
        if rc.concurrency is not None:
            conc = type(rc.concurrency)(max_workers=args.max_workers, per_model=rc.concurrency.per_model)
        else:
            # Create a simple dataclass with expected fields
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class _Conc:
                max_workers: int | None = None
                per_model: int | None = None

            conc = _Conc(max_workers=args.max_workers, per_model=None)

        rc = type(rc)(
            version=rc.version,
            datasets=rc.datasets,
            models=rc.models,
            run_id=rc.run_id,
            name=rc.name,
            description=rc.description,
            output_dir=args.output or rc.output_dir,
            random_seed=rc.random_seed,
            metric_bundles=rc.metric_bundles,
            truncation=rc.truncation,
            concurrency=conc,
            thresholds=rc.thresholds,
        )

    out_dir = run_headless(rc, output_dir=args.output, baseline_summary=args.baseline)
    print(f"Headless run complete. Artifacts written to: {out_dir}")
    
    # Generate reports if requested
    results_path = Path(out_dir) / "results.json"
    
    if args.generate_report:
        try:
            report_path = generate_html_report(results_path, theme=args.theme)
            print(f"HTML report generated: {report_path}")
        except Exception as e:
            print(f"Warning: Failed to generate HTML report: {e}")
    
    if args.generate_markdown:
        try:
            md_path = generate_markdown_report(results_path)
            print(f"Markdown report generated: {md_path}")
        except Exception as e:
            print(f"Warning: Failed to generate Markdown report: {e}")
    
    if args.generate_comparison:
        try:
            comp_path = generate_comparison_report(results_path, Path(args.generate_comparison))
            print(f"Comparison report generated: {comp_path}")
        except Exception as e:
            print(f"Warning: Failed to generate comparison report: {e}")
    
    if args.generate_pdf:
        try:
            pdf_path = generate_pdf_report(results_path, theme=args.theme)
            print(f"PDF report generated: {pdf_path}")
        except Exception as e:
            print(f"Warning: Failed to generate PDF report: {e}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
