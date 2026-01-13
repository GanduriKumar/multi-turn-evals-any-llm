from backend.coverage_reports import coverage_summary_csv, coverage_heatmap_csv, per_turn_csv


def test_summary_and_heatmap_csv_nonempty():
    s = coverage_summary_csv()
    h = coverage_heatmap_csv()
    assert s.startswith("domain,behavior,raw_total")
    assert h.startswith("Domain/Behavior,")


def test_per_turn_csv_from_generated_sample():
    # Build a tiny sample via v2 generator through coverage builder per-behavior
    from backend.coverage_builder_v2 import build_per_behavior_datasets_v2 as build_per_behavior_datasets
    ds, gd = build_per_behavior_datasets(domains=["Returns, Refunds & Exchanges"], behaviors=["Happy Path"], version="1.0.0")[0]
    csv_str = per_turn_csv(ds, gd)
    assert "dataset_id,conversation_id,turn_index" in csv_str
    # Should include a final decision row (-1 index)
    assert ",-1," in csv_str
