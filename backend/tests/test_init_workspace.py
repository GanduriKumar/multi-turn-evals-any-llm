import os
import subprocess
import sys
from pathlib import Path


def run_cli(tmpdir: Path, *args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "init_workspace.py"
    cmd = [sys.executable, str(script), str(tmpdir), *args]
    return subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_workspace_initialized(tmp_path):
    out = run_cli(tmp_path)
    assert out.returncode == 0, out.stderr

    # Required folders
    for folder in ("datasets", "runs", "reports"):
        assert (tmp_path / folder).is_dir(), f"Missing folder: {folder}"

    # Configs copied
    assert (tmp_path / "configs" / "settings.yaml").is_file()

    # Dataset templates or README
    assert (tmp_path / "datasets" / "README.md").is_file()

    # .env created
    env = tmp_path / ".env"
    assert env.is_file()
    content = env.read_text(encoding="utf-8")
    assert "MODEL_PROVIDER" in content


def test_idempotent_without_force(tmp_path):
    # First run
    out1 = run_cli(tmp_path)
    assert out1.returncode == 0, out1.stderr

    # Capture mtimes
    env_path = tmp_path / ".env"
    configs_settings = tmp_path / "configs" / "settings.yaml"
    env_mtime_1 = env_path.stat().st_mtime
    cfg_mtime_1 = configs_settings.stat().st_mtime

    # Second run without --force should not overwrite
    out2 = run_cli(tmp_path)
    assert out2.returncode == 0, out2.stderr

    env_mtime_2 = env_path.stat().st_mtime
    cfg_mtime_2 = configs_settings.stat().st_mtime

    assert env_mtime_2 == env_mtime_1
    assert cfg_mtime_2 == cfg_mtime_1
