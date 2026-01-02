#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # Defer import so this script can be used without installation, relying on repo layout
    try:
        from eval_server.init_workspace import main as _main
    except ModuleNotFoundError:
        # Add backend/src to sys.path when running from repo
        repo_root = Path(__file__).resolve().parents[1]
        backend_src = repo_root / "backend" / "src"
        sys.path.insert(0, str(backend_src))
        from eval_server.init_workspace import main as _main
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
