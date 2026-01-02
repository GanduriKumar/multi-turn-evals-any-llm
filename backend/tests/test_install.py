import subprocess
import sys
from pathlib import Path


def test_editable_install_and_import(tmp_path):
    backend_dir = Path(__file__).resolve().parents[1]

    # Install the package in editable mode
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    print("Running:", " ".join(cmd), "in", str(backend_dir))
    subprocess.check_call(cmd, cwd=str(backend_dir))

    # Validate import in a fresh Python interpreter (ensures .pth is loaded)
    check_cmd = [
        sys.executable,
        "-c",
        "import eval_server, sys; print(eval_server.__file__)",
    ]
    subprocess.check_call(check_cmd)
