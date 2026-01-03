from __future__ import annotations

from pathlib import Path

from eval_server.config.run_config_loader import load_run_config
from eval_server.utils.run_id import compute_run_id


def _load_sample_rc():
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "configs" / "runs" / "sample_run_config.yaml"
    return load_run_config(sample)


def test_run_id_deterministic():
    rc1 = _load_sample_rc()
    rc2 = _load_sample_rc()

    id1 = compute_run_id(rc1)
    id2 = compute_run_id(rc2)
    assert id1 == id2


def test_run_id_changes_on_model_or_metrics(tmp_path: Path):
    rc = _load_sample_rc()

    id_base = compute_run_id(rc)

    # Change model name or model id
    rc_model_changed = type(rc)(
        version=rc.version,
        datasets=rc.datasets,
        models=[
            type(rc.models[0])(
                name=rc.models[0].name,
                provider=rc.models[0].provider,
                model=rc.models[0].model + "-x",  # change
                params=rc.models[0].params,
                concurrency=rc.models[0].concurrency,
            )
        ],
        run_id=rc.run_id,
        name=rc.name,
        description=rc.description,
        output_dir=rc.output_dir,
        random_seed=rc.random_seed,
        metric_bundles=rc.metric_bundles,
        truncation=rc.truncation,
        concurrency=rc.concurrency,
        thresholds=rc.thresholds,
    )

    id_model_changed = compute_run_id(rc_model_changed)
    assert id_model_changed != id_base

    # Change metric bundles
    rc_metrics_changed = type(rc)(
        version=rc.version,
        datasets=rc.datasets,
        models=rc.models,
        run_id=rc.run_id,
        name=rc.name,
        description=rc.description,
        output_dir=rc.output_dir,
        random_seed=rc.random_seed,
        metric_bundles=["basic_text"],  # change
        truncation=rc.truncation,
        concurrency=rc.concurrency,
        thresholds=rc.thresholds,
    )

    id_metrics_changed = compute_run_id(rc_metrics_changed)
    assert id_metrics_changed != id_base
