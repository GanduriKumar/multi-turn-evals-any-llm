from pathlib import Path
import pytest

from eval_server.config_loader import load_settings, Settings


def test_load_settings_success():
    # Use default repository settings.yaml
    settings = load_settings()
    assert isinstance(settings, Settings)

    # Validate top-level sections
    assert settings.version == 1
    assert settings.models.provider == "dummy"
    assert settings.models.model_id == "dummy-1.0"
    assert 0.0 <= settings.models.temperature <= 1.0
    assert settings.models.max_tokens > 0
    assert settings.models.timeout_sec > 0

    assert settings.datasets.root_dir
    # Ensure extensions were normalized and include canonical forms
    assert ".json" in settings.datasets.allowed_extensions
    assert ".yaml" in settings.datasets.allowed_extensions
    assert ".yml" in settings.datasets.allowed_extensions

    # Scoring
    w = settings.scoring.weights
    assert abs((w.correctness + w.consistency + w.adherence + w.hallucination) - 1.0) < 1e-6
    assert 0.0 <= settings.scoring.pass_threshold <= 1.0
    assert settings.scoring.aggregation in {"mean", "min", "weighted"}

    # Report
    assert settings.report.output_dir
    assert "html" in settings.report.formats
    assert settings.report.include_raw is True

    # UI
    assert settings.ui.enabled in {True, False}
    assert settings.ui.theme in {"light", "dark", "system"}
    assert settings.ui.page_size > 0


def test_invalid_yaml_raises_validation_error(tmp_path: Path):
    bad_yaml = tmp_path / "settings.yaml"
    # Missing required sections, wrong types
    bad_yaml.write_text(
        """
        version: "one"
        models:
          provider: 123
          model_id: {}
          temperature: -1
          max_tokens: -5
          timeout_sec: 0
        """,
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_settings(bad_yaml)
