from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .config.run_config_loader import RunConfig, load_run_config
from .orchestrator import OrchestratorSummary, evaluate_run


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_output_dir(rc: RunConfig, override: Optional[str | Path]) -> Path:
    if override:
        return Path(override)
    if rc.output_dir:
        return Path(rc.output_dir)
    # Fallback
    return Path("runs") / "_headless_output"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _serialize_summary(summary: OrchestratorSummary) -> Dict[str, Any]:
    # dataclasses.asdict converts nested dataclasses to plain dicts
    return asdict(summary)


def run_headless(config_path: str | Path, *, output_dir: str | Path | None = None) -> Path:
    """Run evaluations from a run configuration and write artifacts.

    Returns the output directory path used.
    """
    rc = load_run_config(config_path)
    out_dir = _resolve_output_dir(rc, output_dir)
    _ensure_dir(out_dir)

    # Optional: adjust concurrency via CLI override is handled in main()
    cancel = threading.Event()

    summary: OrchestratorSummary = evaluate_run(rc, cancel_event=cancel)
    summary_json = _serialize_summary(summary)

    _write_json(Path(out_dir) / "summary.json", summary_json)
    return out_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run headless evaluation from a config file")
    parser.add_argument("config", help="Path to run configuration (YAML or JSON)")
    parser.add_argument("--output", help="Override output directory", default=None)
    parser.add_argument("--max-workers", type=int, default=None, help="Override max workers (global)")

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

    out_dir = run_headless(rc, output_dir=args.output)
    print(f"Headless run complete. Artifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
