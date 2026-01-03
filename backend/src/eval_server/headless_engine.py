from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .config.run_config_loader import RunConfig, load_run_config
from .orchestrator import OrchestratorSummary, evaluate_run
from .utils.run_id import compute_run_id
from .runner.turn_runner import TurnRunner
from .data.loader import load_conversation
from .metrics.regression import score_regressions


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

    summary: OrchestratorSummary = evaluate_run(rc, cancel_event=cancel)
    summary_json = _serialize_summary(summary)
    # store summary and a simple manifest
    summary_path = Path(out_dir) / "summary.json"
    _write_json(summary_path, summary_json)
    _write_json(Path(out_dir) / "manifest.json", {"run_id": compute_run_id(rc), "datasets": [d.conversation for d in rc.datasets]})

    # Additionally, generate and store per-turn raw outputs as JSONL using TurnRunner
    raw_rows: list[Dict[str, Any]] = []
    for ds in rc.datasets:
        conv = load_conversation(ds.conversation)
        conv_id = str(conv.get("conversation_id"))
        for m in rc.models:
            runner = TurnRunner(m.provider, **(m.params or {}))
            policy = rc.truncation.strategy if rc.truncation else None
            trunc_params: Dict[str, int] = {}
            results = runner.run(conv, truncation_policy=policy, truncation_params=trunc_params)
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

    _write_jsonl(Path(out_dir) / "raw_outputs.jsonl", raw_rows)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
