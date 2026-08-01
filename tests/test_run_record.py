"""Tests for the versioned run-record data contract and ledger report."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from models.run_record import (
    DataContractError,
    build_qa_report,
    load_campaign_manifest,
    load_run_record,
    load_voltammetry,
    normalize_plating_timeseries,
    validate_metadata,
)


def _manifest(**overrides):
    data = {
        "schema_version": "1.0",
        "record_status": "complete",
        "run_id": "beaker-20260801-001",
        "date": "2026-08-01",
        "operator": "tester",
        "experiment_type": "beaker_galvanostatic",
        "bath_batch": "B-20260801-001",
        "equipment": {
            "power_supply": {"model": "PS", "asset_id": "PS-01"},
            "cell": {"type": "beaker", "volume_mL": 1000},
        },
        "setup": {
            "anode": {"material": "graphite"},
            "cathode": {"material": "316L", "area_cm2": 10.0},
        },
        "measurement_conventions": {"cathodic_sign": "negative"},
        "video": {"recording_status": "complete"},
        "gate_evidence": [
            {
                "candidate_id": "candidate-1",
                "gate_id": "chemistry-1",
                "metric": "apparent_fe",
                "value_from": "apparent_faradaic_efficiency",
                "unit": "fraction",
            }
        ],
    }
    data.update(overrides)
    return data


def _bath_batch():
    return {
        "schema_version": "1.0",
        "batch_id": "B-20260801-001",
        "date_mixed": "2026-08-01",
        "operator": "tester",
        "composition": {
            "fe2_g_L": 55.845,
            "h3bo3_g_L": 30.0,
            "pH": 3.0,
            "volume_mL": 1000.0,
        },
        "source_chemicals": {"water_source": "DI"},
        "storage": {"container": "glass", "cover": "sealed"},
    }


def _metadata():
    return {
        "sample_id": "beaker-20260801-001",
        "operator": "tester",
        "instrument": "PS-01 and DMM-01",
        "calibration_date": "2026-07-29",
        "electrolyte_id": "B-20260801-001",
        "working_electrode": "316L, 10 cm2",
        "counter_electrode": "graphite",
        "reference_electrode": "none for galvanostatic run",
        "temperature_C": 25.0,
        "agitation": "magnetic, 300 rpm",
        "preparation": "acetone, DI rinse, dry to constant mass",
    }


def _write_complete_run(tmp_path, *, hull_aliases=True, with_energy=True):
    root = tmp_path / "beaker-20260801-001"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    (root / "bath_batch.json").write_text(json.dumps(_bath_batch()), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")

    if hull_aliases:
        trace = pd.DataFrame({
            "timestamp_s": [0.0, 1.0, 2.0],
            "current_A": [-2.0, -2.0, -2.0],
            "cell_voltage_V": [2.5, 2.5, 2.5],
            "temperature_C": [25.0, 25.0, 25.0],
        })
    else:
        trace = pd.DataFrame({
            "timestamp_s": [0.0, 1.0, 2.0],
            "current_actual_A": [-2.0, -2.0, -2.0],
            "voltage_V": [2.5, 2.5, 2.5],
        })
    trace.to_csv(root / "timeseries.csv", index=False)

    # 80% apparent FE for a 4 C trace.
    theoretical_mass = 4.0 * 55.845 / (2.0 * 96485.3321)
    pd.DataFrame({
        "mass_before_g": [50.0],
        "mass_after_g": [50.0 + 0.8 * theoretical_mass],
        "blank_mass_change_g": [0.0],
        "mass_uncertainty_g": [0.0001],
        "blank_mass_uncertainty_g": [0.0001],
    }).to_csv(root / "mass_log.csv", index=False)

    pd.DataFrame({
        "run_id": ["beaker-20260801-001", "beaker-20260801-001"],
        "coupon_id": ["C-01", "C-01"],
        "characterization_id": ["EDS-01", "EDS-01"],
        "technique": ["SEM_EDS", "SEM_EDS"],
        "analyte": ["Fe", "O"],
        "value": [98.0, 2.0],
        "unit": ["wt%", "wt%"],
        "uncertainty": [0.5, 0.2],
        "basis": ["area average", "area average"],
        "instrument": ["SEM-01", "SEM-01"],
        "calibration_date": ["2026-07-29", "2026-07-29"],
        "analysis_file": ["raw/eds.csv", "raw/eds.csv"],
    }).to_csv(root / "characterization.csv", index=False)

    if with_energy:
        pd.DataFrame({
            "component": ["pumps", "heating", "cooling", "gas_handling", "drying", "other_auxiliary"],
            "energy_Wh": [0.1, 0.2, 0.0, 0.05, 0.1, 0.01],
            "uncertainty_Wh": [0.01] * 6,
            "measurement_method": ["meter"] * 6,
        }).to_csv(root / "energy_log.csv", index=False)
    return root


def test_complete_run_normalizes_hull_trace_and_builds_ledgers(tmp_path):
    root = _write_complete_run(tmp_path)
    report = build_qa_report(root)

    assert report["valid"] is True
    assert report["ready_for_analysis"] is True
    assert report["column_mapping"] == {"current_A": "current_actual_A", "cell_voltage_V": "voltage_V"}
    assert report["metrics"]["charge_C"]["value"] == pytest.approx(4.0)
    assert report["metrics"]["apparent_faradaic_efficiency"]["value"] == pytest.approx(0.8)
    assert report["ledgers"]["energy"]["status"] == "closed"
    assert report["ledgers"]["charge"]["status"] == "partial_with_fe_deposit"
    assert report["ledgers"]["iron"]["status"] == "partial"
    assert report["gate_evidence"]["status"] == "ready_for_gate_evaluation"
    assert report["gate_evidence"]["records"][0]["source"] == "experimental"

    run = load_run_record(root)
    assert run.timeseries.loc[0, "current_actual_A"] == -2.0
    assert run.derived.energy_Wh == pytest.approx(2.5 * 2.0 * 2.0 / 3600.0)


def test_canonical_plating_trace_does_not_need_mapping(tmp_path):
    root = _write_complete_run(tmp_path, hull_aliases=False, with_energy=False)
    report = build_qa_report(root)
    assert report["valid"] is True
    assert report["column_mapping"] == {}
    assert report["ledgers"]["energy"]["status"] == "partial"


def test_incomplete_planned_run_is_reported_without_throwing(tmp_path):
    root = tmp_path / "planned"
    root.mkdir()
    manifest = _manifest(record_status="planned")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = build_qa_report(root)
    assert report["valid"] is True
    assert report["ready_for_analysis"] is False
    assert any(item["severity"] == "warning" for item in report["issues"])
    with pytest.raises(DataContractError, match="not ready"):
        load_run_record(root)


def test_ambiguous_plating_aliases_are_rejected():
    frame = pd.DataFrame({
        "timestamp_s": [0.0, 1.0],
        "current_actual_A": [-1.0, -1.0],
        "current_A": [-1.0, -1.0],
        "voltage_V": [2.0, 2.0],
    })
    with pytest.raises(DataContractError, match="ambiguous"):
        normalize_plating_timeseries(frame)


def test_voltammetry_contract_remains_separate():
    frame = load_voltammetry("experiments/data/voltammetry_template.csv")
    assert "potential_V_vs_ref" in frame
    assert "current_density_mA_cm2" in frame
    assert "voltage_V" not in frame


def test_campaign_manifest_contract_and_metadata_validation(tmp_path):
    campaign = pd.DataFrame({
        "schema_version": ["1.0"],
        "run_id": ["P1-001"],
        "phase": ["I"],
        "technique": ["LSV"],
        "status": ["planned"],
        "raw_file": ["raw.csv"],
        "processed_file": ["processed.csv"],
        "metadata_file": ["metadata.json"],
    })
    path = tmp_path / "campaign.csv"
    campaign.to_csv(path, index=False)
    loaded = load_campaign_manifest(path)
    assert loaded.loc[0, "run_id"] == "P1-001"

    report = validate_metadata({"operator": "tester"})
    assert not report.valid
    assert any(issue.path == "instrument" for issue in report.errors)


def test_invalid_gate_metric_is_a_qa_error(tmp_path):
    root = _write_complete_run(tmp_path)
    manifest = _manifest()
    manifest["gate_evidence"][0]["value_from"] = "not_a_metric"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = build_qa_report(root)
    assert report["valid"] is False
    assert any("unknown measured metric" in item["message"] for item in report["issues"])


def test_file_paths_must_stay_inside_run_directory(tmp_path):
    root = _write_complete_run(tmp_path)
    manifest = _manifest(files={"timeseries_csv": "../outside.csv"})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = build_qa_report(root)
    assert report["valid"] is False
    assert any("inside the run directory" in item["message"] for item in report["issues"])
