from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    # <repo>/backend/src/eval_server/init_workspace.py -> parents[3] = <repo>
    return Path(__file__).resolve().parents[3]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path, *, overwrite: bool = False) -> bool:
    """Copy a single file if it doesn't exist or overwrite=True.

    Returns True if a copy occurred, False if skipped.
    """
    _ensure_dir(dst.parent)
    if dst.exists() and not overwrite:
        return False
    shutil.copy2(src, dst)
    return True


def _copy_tree(src_dir: Path, dst_dir: Path, *, overwrite: bool = False, include_extensions: Iterable[str] | None = None) -> tuple[int, int]:
    """Copy directory tree with optional extension filtering.

    Returns (copied_count, skipped_count).
    """
    copied = 0
    skipped = 0
    if not src_dir.exists():
        return (0, 0)
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        if include_extensions:
            if path.suffix.lower() not in [e.lower() for e in include_extensions]:
                continue
        rel = path.relative_to(src_dir)
        dst = dst_dir / rel
        if _copy_file(path, dst, overwrite=overwrite):
            copied += 1
        else:
            skipped += 1
    return (copied, skipped)


def _write_env(dst: Path, *, overwrite: bool = False) -> bool:
    if dst.exists() and not overwrite:
        return False
    content = (
        "# Sample environment variables for eval workspace\n"
        "# Populate with your actual secrets/values as needed.\n"
        "MODEL_PROVIDER=openai\n"
        "MODEL_ID=gpt-4o-mini\n"
        "OPENAI_API_KEY=sk-xxxxx\n"
        "TIMEOUT_SEC=60\n"
        "DATASETS_DIR=./datasets\n"
        "RUNS_DIR=./runs\n"
        "REPORTS_DIR=./reports\n"
    )
    _ensure_dir(dst.parent)
    dst.write_text(content, encoding="utf-8")
    return True


def init_workspace(target_dir: Path, *, force: bool = False, verbose: bool = True) -> None:
    repo = _repo_root()
    target_dir = target_dir.resolve()

    # Create base folders
    for folder in ("datasets", "runs", "reports"):
        d = target_dir / folder
        _ensure_dir(d)
        if verbose:
            print(f"Ensured folder: {d}")

    # Copy sample configs
    src_configs = repo / "configs"
    dst_configs = target_dir / "configs"
    copied, skipped = _copy_tree(src_configs, dst_configs, overwrite=force)
    if verbose:
        print(f"Configs copied: {copied}, skipped: {skipped}")

    # Copy dataset readme/template
    src_ds = repo / "datasets"
    dst_ds = target_dir / "datasets"
    copied_ds, skipped_ds = _copy_tree(src_ds, dst_ds, overwrite=force, include_extensions=None)
    if verbose:
        print(f"Dataset templates copied: {copied_ds}, skipped: {skipped_ds}")

    # Create .env
    env_path = target_dir / ".env"
    created = _write_env(env_path, overwrite=force)
    if verbose:
        print(f".env {'created' if created else 'exists, skipped'} at {env_path}")

    if verbose:
        print(f"Workspace initialized at: {target_dir}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize an evaluation workspace")
    parser.add_argument("target", help="Target workspace directory to initialize")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--quiet", action="store_true", help="Suppress informational logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    target = Path(ns.target)
    try:
        init_workspace(target, force=ns.force, verbose=not ns.quiet)
        return 0
    except Exception as e:  # pragma: no cover - bubbled as non-zero exit
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
