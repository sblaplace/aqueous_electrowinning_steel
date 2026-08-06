"""Tests for the immutable reference-cell spec and its run_record consumer.

Covers two deliverables of D1 ``NEXT_STEPS §1``:
1. the machine spec module (``models/reference_cell_spec``) — freeze/verify/load
   with tamper-evident content hashing; and
2. the versioned raw-linked data record consumer (``models.run_record``) that
   validates a ``reference_cell.json`` sidecar against the frozen spec and the
   ``docs/REFERENCE_CELL_SPEC.md`` §9 invariants.
"""
from __future__ import annotations

import json

import pytest

from models.reference_cell_spec import (
    CANONICAL_SPEC_RELATIVE,
    SCHEMA_NAME,
    freeze_spec,
    load_spec,
    spec_hash,
    verify_run_pin,
    verify_spec,
)
from models.run_record import (
    build_qa_report,
    validate_reference_cell_record,
)


# --- helpers ----------------------------------------------------------------

def _frozen_sha() -> str:
    """Current sha256 of the canonical frozen spec (the pinned target)."""
    spec = load_spec(str(CANONICAL_SPEC_RELATIVE))
    return spec["sha256"]


def _valid_reference_cell(sha: str | None = None) -> dict:
    sha = sha or _frozen_sha()
    return {
        "schema_version": "1.0",
        "run_id": "refcell-20260801-001",
        "reference_cell_spec": {
            "spec_version": "1.0",
            "configuration_id": "RC-1",
            "spec_sha256": sha,
            "spec_file": "processes/reference_cell_spec.v1.json",
        },
        "as_built_deviations": [],
        "rectifier": {
            "applied_mode": "constant_current",
            "current_setpoint_A": 3.0,
            "current_density_mA_cm2_bound": [100, 300],
            "sync_timestamp_source": "DAQ-101",
            "trace_file": "timeseries.csv",
        },
        "samples": {
            "before": [
                {"sample_id": "RC1-001-BC", "timestamp_s": 0.0,
                 "loop": "catholyte", "sample_point": "SP-101", "file": "sample_log.csv"}
            ],
            "after": [
                {"sample_id": "RC1-001-AC", "timestamp_s": 7200.0,
                 "loop": "catholyte", "sample_point": "SP-101", "file": "sample_log.csv"}
            ],
        },
        "deposit_metrology": {
            "mass":             {"file": "mass_log.csv",             "required": True},
            "thickness_map":    {"file": "deposit/thickness_map.csv", "required": True},
            "composition":      {"file": "deposit/composition.csv",   "required": True},
            "morphology":       {"file": "deposit/morphology.csv",    "required": False},
            "porosity":         {"file": "deposit/porosity.csv",      "required": False},
            "adhesion":         {"file": "deposit/adhesion.csv",      "required": False},
            "hydrogen_content": {"file": "deposit/hydrogen.csv",      "required": False},
        },
    }


def _manifest(**overrides):
    data = {
        "schema_version": "1.0",
        "record_status": "complete",
        "run_id": "refcell-20260801-001",
        "date": "2026-08-01",
        "operator": "tester",
        "experiment_type": "divided_cell",
        "bath_batch": "B-20260801-001",
        "equipment": {
            "power_supply": {"model": "PS", "asset_id": "PS-01", "voltage_V": 30, "current_A": 10},
            "cell": {"type": "divided", "membrane": "Nafion_N117"},
        },
        "setup": {"anode": {"material": "OER"}, "cathode": {"material": "316L", "area_cm2": 10.0}},
        "measurement_conventions": {"cathodic_sign": "negative"},
        "video": {"recording_status": "complete"},
    }
    data.update(overrides)
    return data


def _write_refcell_run(tmp_path, *, reference_cell=None, experiment_type="divided_cell",
                       include_sidecar=True, with_deposit_channels=True):
    root = tmp_path / "refcell-20260801-001"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(_manifest(experiment_type=experiment_type)), encoding="utf-8"
    )
    (root / "bath_batch.json").write_text(json.dumps({
        "schema_version": "1.0",
        "batch_id": "B-20260801-001",
        "date_mixed": "2026-08-01",
        "operator": "tester",
        "composition": {"fe2_g_L": 55.845, "h3bo3_g_L": 30.0, "pH": 2.0, "volume_mL": 1000.0},
        "source_chemicals": {"water_source": "DI"},
        "storage": {"container": "glass", "cover": "sealed"},
    }), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps({
        "sample_id": "refcell-20260801-001",
        "operator": "tester",
        "instrument": "PS-01, DMM-01, CT-01",
        "calibration_date": "2026-07-29",
        "electrolyte_id": "B-20260801-001",
        "working_electrode": "316L, 10 cm2",
        "counter_electrode": "OER",
        "reference_electrode": "none for galvanostatic run",
        "temperature_C": 60.0,
        "agitation": "recirculation, 0.5 L/min",
        "preparation": "ground Ra<=0.8um, acetone, DI rinse, dry to constant mass",
    }), encoding="utf-8")
    import pandas as pd
    pd.DataFrame({
        "timestamp_s": [0.0, 1.0, 2.0],
        "current_actual_A": [-3.0, -3.0, -3.0],
        "voltage_V": [6.0, 6.0, 6.0],
        "temperature_C": [60.0, 60.0, 60.0],
    }).to_csv(root / "timeseries.csv", index=False)
    pd.DataFrame({
        "mass_before_g": [50.0], "mass_after_g": [50.5],
        "blank_mass_change_g": [0.0], "mass_uncertainty_g": [0.0001],
        "blank_mass_uncertainty_g": [0.0001],
    }).to_csv(root / "mass_log.csv", index=False)
    pd.DataFrame({"component": ["pumps"], "energy_Wh": [0.1],
                  "uncertainty_Wh": [0.01], "measurement_method": ["meter"]}
                 ).to_csv(root / "energy_log.csv", index=False)
    if with_deposit_channels:
        (root / "deposit").mkdir(exist_ok=True)
        for name, cols in {
            "thickness_map.csv": ["x_mm", "y_mm", "thickness_um"],
            "composition.csv": ["analyte", "value", "unit"],
            "morphology.csv": ["field", "note"],
            "porosity.csv": ["region", "porosity_pct"],
            "adhesion.csv": ["force_N"],
            "hydrogen.csv": ["content_ppm"],
        }.items():
            if not (root / "deposit" / name).exists():
                pd.DataFrame({c: [] for c in cols}).to_csv(root / "deposit" / name, index=False)
    if include_sidecar:
        (root / "reference_cell.json").write_text(
            json.dumps(reference_cell if reference_cell is not None else _valid_reference_cell()),
            encoding="utf-8",
        )
    return root


# --- spec module: freeze / verify / tamper ---------------------------------


def test_load_spec_accepts_canonical_schema():
    spec = load_spec(str(CANONICAL_SPEC_RELATIVE))
    assert spec["schema"] == SCHEMA_NAME
    assert spec["spec_version"] == "1.0"
    assert spec["configuration_id"] == "RC-1"
    # All D1-mandated top-level groups are present.
    for group in ("cell_stack", "batch", "flow_mixing_gas_thermal", "rectifier",
                  "sampling", "deposit_metrology", "immutable_rule"):
        assert group in spec, f"spec is missing required group {group!r}"


def test_canonical_spec_is_frozen_and_self_consistent():
    ok, msg = verify_spec(str(CANONICAL_SPEC_RELATIVE))
    assert ok, msg


def test_spec_hash_is_independent_of_whitespace():
    spec = load_spec(str(CANONICAL_SPEC_RELATIVE))
    compact = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spaced = json.dumps(spec, sort_keys=True, indent=4)
    assert spec_hash(json.loads(compact)) == spec_hash(json.loads(spaced))


def test_tamper_changes_the_hash(tmp_path):
    spec = load_spec(str(CANONICAL_SPEC_RELATIVE))
    untampered = spec_hash(spec)
    tampered = dict(spec)
    tampered["cell_stack"]["cathode_material"] = "tampered"
    assert spec_hash(tampered) != untampered


def test_freeze_verify_roundtrip_and_tamper_detection(tmp_path):
    target = tmp_path / "spec.json"
    spec = load_spec(str(CANONICAL_SPEC_RELATIVE))
    spec.pop("sha256", None)  # simulate an unfrozen file
    target.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    ok, _ = verify_spec(target)
    assert not ok  # not yet frozen

    digest = freeze_spec(target)
    assert digest == spec_hash(load_spec(str(target)))
    ok, _ = verify_spec(target)
    assert ok  # now frozen and self-consistent

    # Freezing is idempotent.
    assert freeze_spec(target) == digest

    # A content edit breaks the pin.
    edited = load_spec(str(target))
    edited["batch"]["recipe_targets"]["bulk_pH"] = 3.0
    target.write_text(json.dumps(edited, indent=2), encoding="utf-8")
    ok, _ = verify_spec(target)
    assert not ok


def test_load_spec_rejects_unknown_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "nope", "sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_spec(bad)


# --- run_record consumer: validate_reference_cell_record --------------------


def test_refcell_record_valid_against_canonical_spec():
    report = validate_reference_cell_record(
        _valid_reference_cell(), spec_file=CANONICAL_SPEC_RELATIVE
    )
    assert report.valid, [i.to_dict() for i in report.issues]


def test_refcell_record_valid_when_spec_unreachable_is_warning_only():
    report = validate_reference_cell_record(_valid_reference_cell(), spec_file=None)
    # Format is enforced; content check downgraded to a warning, never an error.
    assert report.valid
    assert any(issue.severity == "warning" for issue in report.issues)


def test_refcell_record_rejects_bad_sha_format():
    rec = _valid_reference_cell(sha="not-a-digest")
    report = validate_reference_cell_record(rec, spec_file=CANONICAL_SPEC_RELATIVE)
    assert not report.valid
    assert any(issue.path == "reference_cell_spec.spec_sha256" for issue in report.errors)


def test_refcell_record_rejects_sha_mismatch_against_resolvable_spec():
    rec = _valid_reference_cell(sha="0" * 64)  # valid format, wrong content
    report = validate_reference_cell_record(rec, spec_file=CANONICAL_SPEC_RELATIVE)
    assert not report.valid
    assert any("does not match the canonical spec" in issue.message
               for issue in report.errors)


def test_refcell_record_requires_before_and_after_samples():
    rec = _valid_reference_cell()
    del rec["samples"]["before"]
    report = validate_reference_cell_record(rec, spec_file=CANONICAL_SPEC_RELATIVE)
    assert not report.valid
    assert any(issue.path == "samples.before" for issue in report.errors)


def test_refcell_record_requires_core_metrology():
    rec = _valid_reference_cell()
    rec["deposit_metrology"].pop("composition", None)
    report = validate_reference_cell_record(rec, spec_file=CANONICAL_SPEC_RELATIVE)
    assert not report.valid
    assert any("deposit_metrology.composition" in issue.path for issue in report.errors)


def test_refcell_record_missing_complete_metrology_is_warning():
    rec = _valid_reference_cell()
    for key in ("morphology", "porosity", "adhesion", "hydrogen_content"):
        rec["deposit_metrology"].pop(key, None)
    report = validate_reference_cell_record(rec, spec_file=CANONICAL_SPEC_RELATIVE)
    assert report.valid  # core metrology still present
    assert any(issue.severity == "warning" and "porosity" in issue.path
               for issue in report.issues)


def test_refcell_record_validates_as_built_deviations():
    rec = _valid_reference_cell()
    rec["as_built_deviations"] = [
        {"path": "cell_stack.electrode_to_membrane_gap_mm", "as_built": 3.2,
         "authorized_by": "RC1-ops"}
    ]
    report = validate_reference_cell_record(rec, spec_file=CANONICAL_SPEC_RELATIVE)
    assert report.valid

    rec["as_built_deviations"][0].pop("authorized_by", None)
    report = validate_reference_cell_record(rec, spec_file=CANONICAL_SPEC_RELATIVE)
    assert not report.valid
    assert any("as_built_deviations[0].authorized_by" in issue.path
               for issue in report.errors)


def test_refcell_record_accepts_planned_status_without_complete_gates():
    rec = _valid_reference_cell()
    del rec["samples"]["after"]
    rec["deposit_metrology"].pop("thickness_map", None)
    report = validate_reference_cell_record(rec, spec_file=CANONICAL_SPEC_RELATIVE,
                                            record_status="in_progress")
    assert report.valid  # not complete -> before/after & metrology not required yet


# --- run_record consumer: build_qa_report integration ------------------------


def test_build_qa_report_accepts_valid_refcell_run(tmp_path):
    root = _write_refcell_run(tmp_path)
    report = build_qa_report(root)
    assert report["valid"] is True, report["issues"]
    comp = report["components"]["reference_cell"]
    assert comp["valid"] is True
    assert report["files"]["reference_cell_json"]["exists"] is True


def test_build_qa_report_flags_divided_cell_missing_sidecar(tmp_path):
    root = _write_refcell_run(tmp_path, include_sidecar=False)
    report = build_qa_report(root)
    assert any(issue["path"] == "reference_cell"
               and "requires a reference_cell.json" in issue["message"]
               for issue in report["issues"])


def test_build_qa_report_rejects_tampered_pin(tmp_path):
    rec = _valid_reference_cell(sha="0" * 64)
    root = _write_refcell_run(tmp_path, reference_cell=rec)
    report = build_qa_report(root)
    assert report["valid"] is False
    assert any("reference_cell_spec.spec_sha256" in issue["path"]
               for issue in report["issues"])


def test_build_qa_report_non_refcell_run_with_sidecar_still_validates(tmp_path):
    # A beaker run is unaffected, but if it ships a reference_cell.json it is
    # still validated (backward compatible; optional sidecar).
    root = tmp_path / "beaker-20260801-001"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(
        _manifest(experiment_type="beaker_galvanostatic",
                  run_id="beaker-20260801-001")), encoding="utf-8")
    (root / "reference_cell.json").write_text(
        json.dumps(_valid_reference_cell(sha="0" * 64)), encoding="utf-8")
    report = build_qa_report(root)
    # Missing required core files are errors, but the refcell sidecar tamper is
    # surfaced AND the run is not an error just for being a beaker run.
    assert any("reference_cell_spec.spec_sha256" in issue["path"]
               for issue in report["issues"])


def test_verify_run_pin_roundtrip(tmp_path):
    # The spec module's pin wrapper: format check, then content check against
    # the canonical file when reachable.
    pin = _frozen_sha()
    ok, msg = verify_run_pin(pin, spec_file=CANONICAL_SPEC_RELATIVE)
    assert ok, msg
    ok, msg = verify_run_pin("0" * 64, spec_file=CANONICAL_SPEC_RELATIVE)
    assert not ok
    assert "does not match" in msg
