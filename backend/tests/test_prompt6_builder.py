from backend.coverage_builder_v2 import (
    build_per_behavior_datasets_v2 as build_per_behavior_datasets,
    build_domain_combined_datasets_v2 as build_domain_combined_datasets,
    build_global_combined_dataset_v2 as build_global_combined_dataset,
)
from backend.schemas import SchemaValidator


def _validate(ds, gd):
    sv = SchemaValidator()
    assert sv.validate("dataset", ds) == []
    assert sv.validate("golden", gd) == []
    # referential integrity
    ds_ids = {c["conversation_id"] for c in ds["conversations"]}
    gd_ids = {e["conversation_id"] for e in gd["entries"]}
    assert ds_ids == gd_ids


def test_per_behavior_datasets_build_and_validate():
    outputs = build_per_behavior_datasets(version="1.0.0")
    # Build only a few to keep test quick; but by default it will iterate all configured
    # Here we just validate the first three if present
    for ds, gd in outputs[:3]:
        _validate(ds, gd)


def test_domain_combined_datasets_build_and_validate():
    outputs = build_domain_combined_datasets(version="1.0.0")
    for ds, gd in outputs[:2]:
        _validate(ds, gd)


def test_global_combined_dataset_build_and_validate():
    ds, gd = build_global_combined_dataset(version="1.0.0")
    _validate(ds, gd)
