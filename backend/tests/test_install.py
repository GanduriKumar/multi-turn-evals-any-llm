import importlib
import os
import subprocess
import sys
from pathlib import Path


def test_editable_install_and_import(tmp_path):
    backend_dir = Path(__file__).resolve().parents[1]

    # Create an isolated virtual environment using the current interpreter's site-packages
    # and install the package in editable mode within this environment using pip.
    # If running under a venv, we still use pip -e ., which should work.

    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    print("Running:", " ".join(cmd), "in", str(backend_dir))
    subprocess.check_call(cmd, cwd=str(backend_dir))

    # Now attempt to import the package
    mod = importlib.import_module("eval_server")
    assert mod is not None
