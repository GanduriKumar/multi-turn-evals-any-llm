from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from .orchestrator import Orchestrator
    from .schemas import SchemaValidator
    from .reporter import Reporter
    from .coverage_builder_v2 import (
        build_per_behavior_datasets_v2,
        build_domain_combined_datasets_v2,
        build_global_combined_dataset_v2,
    )
    from .export_prompts import export_prompts_to_csv
except ImportError:
    from backend.orchestrator import Orchestrator
    from backend.schemas import SchemaValidator
    from backend.reporter import Reporter
    from backend.coverage_builder_v2 import (
        build_per_behavior_datasets_v2,
        build_domain_combined_datasets_v2,
        build_global_combined_dataset_v2,
    )
    from backend.export_prompts import export_prompts_to_csv


DEMO_DATASET = {
    "dataset_id": "demo",
    "version": "1.0.0",
    "metadata": {"domain": "commerce", "difficulty": "easy", "tags": ["sample"]},
    "conversations": [
        {
            "conversation_id": "conv1",
            "turns": [
                {"role": "user", "text": "I want a refund for order A1"},
                {"role": "assistant", "text": "Please share the order ID."},
                {"role": "user", "text": "Order ID is A1."}
            ]
        }
    ]
}

DEMO_GOLDEN = {
    "dataset_id": "demo",
    "version": "1.0.0",
    "entries": [
        {
            "conversation_id": "conv1",
            "turns": [
                {"turn_index": 1, "expected": {"variants": ["Please share the order ID.", "Could you provide your order ID?"]}}
            ],
            "final_outcome": {"decision": "ALLOW", "next_action": "issue_refund"},
            "constraints": {"refund_after_ship": False, "max_refund": 10}
        }
    ]
}


def cmd_init(root: Path) -> int:
    root = Path(root)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)

    ds_path = root / "datasets" / "demo.dataset.json"
    gd_path = root / "datasets" / "demo.golden.json"
    if not ds_path.exists():
        ds_path.write_text(json.dumps(DEMO_DATASET, indent=2), encoding="utf-8")
    if not gd_path.exists():
        gd_path.write_text(json.dumps(DEMO_GOLDEN, indent=2), encoding="utf-8")

    sample_run = {
        "run_id": "run-demo",
        "datasets": ["demo"],
        "models": ["gemini:gemini-2.5"],
        "metrics": ["exact"],
        "thresholds": {"semantic": 0.80},
    }
    rc_path = root / "configs" / "sample.run.json"
    if not rc_path.exists():
        rc_path.write_text(json.dumps(sample_run, indent=2), encoding="utf-8")

    print(f"Initialized workspace at {root}")
    print(f" - datasets/: demo.dataset.json, demo.golden.json")
    print(f" - configs/: sample.run.json")
    print(f" - runs/: (created)")
    return 0


def cmd_run(root: Path, file: Path, no_semantic: bool = False) -> int:
    root = Path(root)
    cfg_path = Path(file)
    if not cfg_path.exists():
        print(f"Run config not found: {cfg_path}", file=sys.stderr)
        return 2
    try:
        run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 2

    # Validate against run_config schema if available
    try:
        sv = SchemaValidator()
        errs = sv.validate("run_config", run_cfg)
        if errs:
            print("run_config validation errors:")
            for e in errs:
                print(" -", e)
            return 3
    except Exception:
        # If schema is unavailable, proceed
        pass

    datasets: List[str] = list(run_cfg.get("datasets") or [])
    models: List[str] = list(run_cfg.get("models") or [])
    metrics: List[str] = list(run_cfg.get("metrics") or [])
    if no_semantic and "semantic" in metrics:
        metrics = [m for m in metrics if m != "semantic"]
    thresholds = run_cfg.get("thresholds") or {}

    if not datasets or not models:
        print("No datasets or models specified", file=sys.stderr)
        return 2

    orch = Orchestrator(datasets_dir=root / "datasets", runs_root=root / "runs")
    run_ids: List[str] = []
    for d in datasets:
        for m in models:
            job = orch.submit(dataset_id=d, model_spec=m, config={"metrics": metrics, "thresholds": thresholds})
            # Run the job inline without requiring an event loop
            import asyncio
            asyncio.run(orch.run_job(job.job_id))
            print(f"Run completed: job={job.job_id} state={job.state} run_id={job.run_id}")
            run_ids.append(job.run_id)
            # Generate HTML report per run
            try:
                runs_dir = root / "runs" / job.run_id
                results_path = runs_dir / "results.json"
                if results_path.exists():
                    results = json.loads(results_path.read_text(encoding="utf-8"))
                    templates_dir = Path(__file__).resolve().parent / "templates"
                    rep = Reporter(templates_dir)
                    out_html = runs_dir / "report.html"
                    rep.write_html(results, out_html)
                    print(f"Report: {out_html}")
            except Exception as e:
                print(f"Report generation failed: {e}")

    print("All runs:", ", ".join(run_ids))
    return 0


def cmd_export_prompts_csv(root: Path, vertical: str, dataset_id: Optional[str] = None, output: Optional[str] = None) -> int:
    """Export prompts and golden data to CSV."""
    try:
        csv_content = export_prompts_to_csv(vertical, dataset_id)
        
        # Determine output file path
        if output:
            out_path = Path(output)
        else:
            filename = f"prompts-export-{vertical}"
            if dataset_id:
                filename += f"-{dataset_id}"
            filename += ".csv"
            out_path = root / filename
        
        # Write to file
        out_path.write_text(csv_content, encoding="utf-8")
        print(f"Export successful: {out_path}")
        return 0
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-eval-cli", description="LLM Eval CLI")
    p.add_argument("command", choices=["init", "run", "coverage", "export-prompts-csv"], help="CLI command")
    p.add_argument("--root", dest="root", default=str(Path.cwd()), help="Workspace root (default: CWD)")
    # run
    p.add_argument("--file", dest="file", default=None, help="Run config file (for run)")
    p.add_argument("--no-semantic", dest="no_semantic", action="store_true", help="Disable semantic metric for this run")
    # coverage generate options
    p.add_argument("--combined", dest="combined", action="store_true", help="Generate combined datasets (per-domain + global)")
    p.add_argument("--split", dest="split", action="store_true", help="Generate split per-behavior datasets")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="Do not write files, only print summary")
    p.add_argument("--save", dest="save", action="store_true", help="Write generated dataset/golden files to datasets/")
    p.add_argument("--overwrite", dest="overwrite", action="store_true", help="Allow overwriting existing files")
    p.add_argument("--version", dest="version", default="1.0.0", help="Dataset version to assign")
    p.add_argument("--domains", dest="domains", nargs="*", default=None, help="Subset of domains to include (exact match)")
    p.add_argument("--behaviors", dest="behaviors", nargs="*", default=None, help="Subset of behaviors to include (exact match)")
    p.add_argument("--out", dest="out", default=None, help="Output directory (default: <root>/datasets)")
    p.add_argument("--shards", dest="shards", type=int, default=1, help="Total shards for generation")
    p.add_argument("--shard-index", dest="shard_index", type=int, default=0, help="This shard index [0..shards-1]")
    # export-prompts-csv options
    p.add_argument("--vertical", dest="vertical", default=None, help="Vertical subfolder (required for export-prompts-csv)")
    p.add_argument("--dataset-id", dest="dataset_id", default=None, help="Specific dataset to export (optional for export-prompts-csv)")
    p.add_argument("--output", dest="output", default=None, help="Output CSV file path (optional for export-prompts-csv)")
    return p


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "init":
        return cmd_init(root)
    if args.command == "run":
        if not args.file:
            print("--file is required for run", file=sys.stderr)
            return 2
        return cmd_run(root, Path(args.file), no_semantic=args.no_semantic)
    if args.command == "coverage":
        return cmd_coverage_generate(
            root=root,
            combined=(True if args.combined else not args.split),
            dry_run=args.dry_run or not args.save,
            save=args.save,
            overwrite=args.overwrite,
            version=args.version,
            domains=args.domains,
            behaviors=args.behaviors,
            out_dir=Path(args.out) if args.out else None,
            shards=args.shards,
            shard_index=args.shard_index,
            v2=True,
        )
    if args.command == "export-prompts-csv":
        if not args.vertical:
            print("--vertical is required for export-prompts-csv", file=sys.stderr)
            return 2
        return cmd_export_prompts_csv(root, args.vertical, args.dataset_id, args.output)
    parser.print_help()
    return 2


# ---------------- Coverage Generate Implementation ----------------

def _should_take(index: int, shards: int, shard_index: int) -> bool:
    if shards <= 1:
        return True
    if shard_index < 0 or shard_index >= shards:
        return True
    return (index % shards) == shard_index


def _print_summary(rows: List[Tuple[str, int, int]]) -> None:
    # rows: (dataset_id, conv_count, golden_count)
    if not rows:
        print("No outputs")
        return
    w = max(len(r[0]) for r in rows)
    header = f"{'DATASET ID'.ljust(w)}  CONVS  GOLDEN"
    print(header)
    print("-" * len(header))
    for ds_id, c, g in rows:
        print(f"{ds_id.ljust(w)}  {str(c).rjust(5)}  {str(g).rjust(6)}")


def cmd_coverage_generate(
    root: Path,
    *,
    combined: bool,
    dry_run: bool,
    save: bool,
    overwrite: bool,
    version: str,
    domains: Optional[List[str]],
    behaviors: Optional[List[str]],
    out_dir: Optional[Path],
    shards: int,
    shard_index: int,
    v2: bool = False,
) -> int:
    root = Path(root)
    out_dir = out_dir or (root / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build outputs using v2 pipeline only
    if combined:
        outputs = build_domain_combined_datasets_v2(domains=domains, behaviors=behaviors, version=version)
        # Append global combined as last element
        outputs.append(build_global_combined_dataset_v2(domains=domains, behaviors=behaviors, version=version))
    else:
        outputs = build_per_behavior_datasets_v2(domains=domains, behaviors=behaviors, version=version)

    # Shard selection by index over outputs
    selected: List[Tuple[dict, dict]] = []
    for idx, pair in enumerate(outputs):
        if _should_take(idx, shards, shard_index):
            selected.append(pair)

    # Validate and optionally save
    sv = SchemaValidator()
    summary_rows: List[Tuple[str, int, int]] = []
    for ds, gd in selected:
        ds_id = ds["dataset_id"]
        ds_err = sv.validate("dataset", ds)
        gd_err = sv.validate("golden", gd)
        if ds_err or gd_err:
            print(f"Validation errors for {ds_id}")
            for e in ds_err:
                print(" dataset:", e)
            for e in gd_err:
                print(" golden:", e)
            return 3
        summary_rows.append((ds_id, len(ds["conversations"]), len(gd["entries"])) )
        if save and not dry_run:
            ds_path = out_dir / f"{ds_id}.dataset.json"
            gd_path = out_dir / f"{ds_id}.golden.json"
            # Ensure parent folders exist if dataset_id encodes subfolders (e.g., "domain/behavior-...")
            ds_path.parent.mkdir(parents=True, exist_ok=True)
            gd_path.parent.mkdir(parents=True, exist_ok=True)
            if not overwrite and (ds_path.exists() or gd_path.exists()):
                print(f"Exists (skip): {ds_id}")
                continue
            ds_path.write_text(json.dumps(ds, indent=2), encoding="utf-8")
            gd_path.write_text(json.dumps(gd, indent=2), encoding="utf-8")

    _print_summary(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
