from backend.coverage_builder_v2 import (
    build_domain_combined_datasets_v2 as build_domain_combined_datasets,
    build_global_combined_dataset_v2 as build_global_combined_dataset,
    build_per_behavior_datasets_v2 as build_per_behavior_datasets,
)
from backend.coverage_engine import CoverageEngine
from backend.coverage_manifest import CoverageManifestor
from backend.schemas import SchemaValidator


def _validate_docs(ds, gd):
    sv = SchemaValidator()
    assert sv.validate("dataset", ds) == []
    assert sv.validate("golden", gd) == []
    ds_ids = {c["conversation_id"] for c in ds["conversations"]}
    gd_ids = {e["conversation_id"] for e in gd["entries"]}
    assert ds_ids == gd_ids


def test_schema_compliance_for_domain_combined_and_global():
    # Validate all domain-combined outputs and the global combined
    domain_outputs = build_domain_combined_datasets(version="1.0.0")
    for ds, gd in domain_outputs:
        _validate_docs(ds, gd)
    ds, gd = build_global_combined_dataset(version="1.0.0")
    _validate_docs(ds, gd)


def test_expected_counts_for_key_pairs():
    eng = CoverageEngine()
    # Shipping & Delivery × Constraint-heavy Queries => 54 after removing 2 availability bins
    sd = eng.scenarios_for("Shipping & Delivery", "Constraint-heavy Queries")
    assert len(sd) == 54

    # System Awareness & Failure Handling × Happy Path
    # brand bias => none (3*1*4*3=36), regulatory block removes out-of-policy (12) => 24
    saf = eng.scenarios_for("System Awareness & Failure Handling", "Happy Path")
    assert len(saf) == 24

    # Trust, Safety & Fraud × Ambiguous Queries => brand bias none => 36
    tsf = eng.scenarios_for("Trust, Safety & Fraud", "Ambiguous Queries")
    assert len(tsf) == 36


def _parse_axes_from_id(cid: str) -> dict:
    # id format: domain|behavior|price_sensitivity=..|brand_bias=..|availability=..|policy_boundary=..
    parts = cid.split("|")
    axes = {}
    for kv in parts[2:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            axes[k] = v
    return axes


def test_golden_alignment_and_denials_for_out_of_policy():
    # Build per-behavior for a couple of pairs that include out-of-policy
    outputs = build_per_behavior_datasets(
        domains=["Checkout & Payments"],
        behaviors=["Adversarial/Trap Queries", "Ambiguous Queries"],
        version="1.0.0",
    )
    for ds, gd in outputs:
        _validate_docs(ds, gd)
        # For conversations whose id encodes out-of-policy, decision must be DENY
        for entry in gd["entries"]:
            axes = _parse_axes_from_id(entry["conversation_id"])
            if axes.get("policy_boundary") == "out-of-policy":
                assert entry["final_outcome"]["decision"] == "DENY"


def test_manifest_breakdown_sums_match_final_total():
    cm = CoverageManifestor()
    manifest = cm.build(seed=123)
    pair = cm.get_pair(manifest, "System Awareness & Failure Handling", "Happy Path")
    assert pair is not None
    removed = sum((b.get("removed_exclude", 0) + b.get("removed_cap", 0)) for b in pair["breakdown"])
    assert pair["raw_total"] - removed == pair["final_total"]
    # Expect known removal numbers here: 72 excludes for brand bias + 12 regulatory => 84, final 24
    assert pair["raw_total"] == 108
    assert removed == 84
    assert pair["final_total"] == 24
