from backend.coverage_engine import CoverageEngine
from backend.convgen_v2 import build_records
from backend.schemas import SchemaValidator


def test_generated_conversation_and_golden_schema_compliance():
    eng = CoverageEngine()
    tax = eng.taxonomy
    # Pick a deterministic scenario: first from Returns domain, Happy Path
    scenarios = eng.scenarios_for("Returns, Refunds & Exchanges", "Happy Path")
    assert scenarios, "No scenarios generated"
    sc = scenarios[0]

    axes = dict(sc.axes)
    dataset_doc, golden_doc = build_records(
        domain=sc.domain,
        behavior=sc.behavior,
        axes=axes,
        version="1.0.0",
        seed=42,
    )

    sv = SchemaValidator()
    assert sv.validate("dataset", dataset_doc) == []
    assert sv.validate("golden", golden_doc) == []

    # Outcome alignment: if out-of-policy then DENY
    if axes["policy_boundary"] == "out-of-policy":
        assert golden_doc["entries"][0]["final_outcome"]["decision"] == "DENY"
