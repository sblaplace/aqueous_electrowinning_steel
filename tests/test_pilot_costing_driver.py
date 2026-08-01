"""Test that the pilot costing driver runs without errors and produces output."""

import json
from models.run_pilot_costing import main


def test_driver_main(tmp_path, monkeypatch):
    """Run the full driver main() and verify JSON report + figures."""
    import models.run_pilot_costing as drv

    # Redirect outputs to tmp_path
    monkeypatch.setattr(drv, "OUTPUT_DIR", tmp_path / "figures")
    monkeypatch.setattr(drv, "REPORT_DIR", tmp_path / "data")
    (tmp_path / "figures").mkdir()
    (tmp_path / "data").mkdir()

    main()

    # Verify figures
    for name in ["capex_by_scale.png", "capex_category.png", "opex_breakdown_pilot.png", "capex_tornado.png"]:
        assert (tmp_path / "figures" / name).exists(), f"Missing {name}"

    # Verify JSON report
    report_path = tmp_path / "data" / "pilot_costing_report.json"
    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)
    assert len(report["equipment_table"]) == 16
    assert report["capex"]["pilot"]["total_capex"] > 0
    assert report["capex"]["lab"]["total_capex"] < report["capex"]["pilot"]["total_capex"]
    assert report["capex"]["pilot"]["total_capex"] < report["capex"]["production"]["total_capex"]
    assert report["opex_pilot"]["Total OPEX ($/yr)"] > 0
    assert "base" in report["sensitivity"]
