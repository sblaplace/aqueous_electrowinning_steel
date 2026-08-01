"""Tests for plating data infrastructure.

Covers: manifest validation, bath batch loading, experiment manifest
loading, timeseries loading, FE computation from raw mass + charge,
anomaly detection (current drift, temperature drift, missing data),
video index loading, and the full load_plating_run integration.
"""
import json

import numpy as np
import pandas as pd
import pytest

from models.run_manifest import (
    VALID_EXPERIMENT_TYPES,
    load_bath_batch,
    validate_experiment_manifest,
)
from models.plating_data import (
    PlatingRun,
    compute_derived,
    detect_anomalies,
    load_plating_run,
    load_timeseries,
)


# ── Helpers ─────────────────────────────────────────────────────────

FARADAY = 96485.3321
M_FE = 55.845
Z_FE = 2


def _make_manifest(
    experiment_type="beaker_galvanostatic",
    video_status="complete",
    **overrides,
):
    data = {
        "run_id": "test-20260801-001",
        "date": "2026-08-01",
        "operator": "test",
        "experiment_type": experiment_type,
        "bath_batch": "B-20260801-001",
        "equipment": {
            "power_supply": {"model": "Korad KA3005P", "asset_id": "PS-01"},
            "cell": {"type": "beaker", "volume_mL": 500},
        },
        "setup": {
            "anode": {"material": "DSA"},
            "cathode": {"material": "304 SS", "area_cm2": 10.0},
        },
        "video": {"recording_status": video_status},
    }
    data.update(overrides)
    return data


def _make_timeseries(
    n=100,
    dt=1.0,
    current=-2.0,
    voltage=2.5,
    drift=False,
    temp_drift=False,
    gap_at=None,
):
    time = np.arange(n, dtype=float) * dt
    cur = np.full(n, current, dtype=float)
    vol = np.full(n, voltage, dtype=float)
    temp = np.full(n, 25.0, dtype=float)

    if drift:
        # Introduce 10% drift at t=50
        cur[50:] = current * 1.10
    if temp_drift:
        # Introduce 5 °C drift at t=50
        temp[50:] = 30.0

    data = {
        "timestamp_s": time,
        "current_actual_A": cur,
        "voltage_V": vol,
        "temperature_C": temp,
    }
    if gap_at is not None:
        # Insert a gap by doubling the timestamp at gap_at
        time = np.concatenate([time[:gap_at], [time[gap_at - 1] + 20.0], time[gap_at:]])
        cur = np.concatenate([cur[:gap_at], [cur[gap_at - 1]], cur[gap_at:]])
        vol = np.concatenate([vol[:gap_at], [vol[gap_at - 1]], vol[gap_at:]])
        temp = np.concatenate([temp[:gap_at], [temp[gap_at - 1]], temp[gap_at:]])
        data = {
            "timestamp_s": time,
            "current_actual_A": cur,
            "voltage_V": vol,
            "temperature_C": temp,
        }
    return pd.DataFrame(data)


def _make_mass_log(mass_before=50.0, mass_gain=0.5, blank=0.0):
    data = {"mass_before_g": [mass_before], "mass_after_g": [mass_before + mass_gain]}
    if blank != 0.0:
        data["blank_mass_change_g"] = [blank]
    return pd.DataFrame(data)


def _write_experiment_dir(tmp_path, *, include_mass_log=True, **ts_kwargs):
    """Create a minimal experiment directory and return the path."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    manifest = _make_manifest()
    (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ts = _make_timeseries(**ts_kwargs)
    ts.to_csv(exp_dir / "timeseries.csv", index=False)
    if include_mass_log:
        mass_log = _make_mass_log()
        mass_log.to_csv(exp_dir / "mass_log.csv", index=False)
    return exp_dir


# ═══════════════════════════════════════════════════════════════════════
#  1. experiment_manifest.json schema validates a complete example
# ═══════════════════════════════════════════════════════════════════════

def test_experiment_manifest_schema_validates_complete_example():
    """The schema accepts a fully populated manifest."""
    data = _make_manifest()
    report = validate_experiment_manifest(data)
    assert report.valid, f"Expected valid, got: {report.summary()}"
    assert report.errors == []


def test_experiment_manifest_schema_rejects_missing_required_keys():
    """Missing top-level keys are caught."""
    data = _make_manifest()
    del data["experiment_type"]
    report = validate_experiment_manifest(data)
    assert not report.valid
    assert any("experiment_type" in i.message for i in report.errors)


def test_experiment_manifest_schema_rejects_unknown_experiment_type():
    """Unknown experiment_type is rejected."""
    data = _make_manifest(experiment_type="electroplating_unknown")
    report = validate_experiment_manifest(data)
    assert not report.valid
    assert any("experiment_type" in i.path for i in report.errors)


# ═══════════════════════════════════════════════════════════════════════
#  2. timeseries_columns.md documents all fields — covered by loader
# ═══════════════════════════════════════════════════════════════════════

def test_timeseries_loader_requires_minimum_columns(tmp_path):
    """Loader rejects CSV missing required columns (current_actual_A)."""
    df = pd.DataFrame({"timestamp_s": [0, 1], "voltage_V": [2.5, 2.5]})
    csv = tmp_path / "bad.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_timeseries(csv)


# ═══════════════════════════════════════════════════════════════════════
#  3. plating_data.py loads manifest + timeseries + mass_log
# ═══════════════════════════════════════════════════════════════════════

def test_load_plating_run_loads_all_files(tmp_path):
    """Full integration: load_plating_run reads all three files."""
    exp_dir = _write_experiment_dir(tmp_path)
    run = load_plating_run(exp_dir, cathode_area_cm2=10.0)
    assert isinstance(run, PlatingRun)
    assert run.manifest["run_id"] == "test-20260801-001"
    assert len(run.timeseries) == 100
    assert run.mass_log is not None
    assert len(run.mass_log) == 1


# ═══════════════════════════════════════════════════════════════════════
#  4. FE computed from raw mass + charge (not user-entered FE)
# ═══════════════════════════════════════════════════════════════════════

def test_fe_computed_from_raw_mass_and_charge():
    """FE = (net deposit mass) / (charge × M / (z × F))  — from first principles."""
    n, dt = 100, 1.0
    current = -2.0  # cathodic
    charge_C = abs(current) * (n - 1) * dt  # simple rectangle ≈ trapezoid
    theoretical_mass = charge_C * M_FE / (Z_FE * FARADAY)
    # Choose mass gain = 80% of theoretical
    mass_gain = 0.80 * theoretical_mass
    ts = _make_timeseries(n=n, dt=dt, current=current)
    mass_log = _make_mass_log(mass_before=50.0, mass_gain=mass_gain)

    derived = compute_derived(ts, mass_log, cathodic_sign="negative")
    assert derived.faradaic_efficiency is not None
    assert abs(derived.faradaic_efficiency - 0.80) < 0.02  # within 2%
    assert derived.net_deposit_mass_g == pytest.approx(mass_gain, rel=1e-6)
    assert derived.theoretical_fe_mass_g == pytest.approx(theoretical_mass, rel=1e-6)


def test_fe_not_capped_at_100_percent():
    """FE > 100% is allowed (retained salts, balance error)."""
    ts = _make_timeseries(n=100, dt=1.0, current=-2.0)
    # Artificially high mass gain (150% of theoretical)
    charge = 2.0 * 99.0
    theoretical = charge * M_FE / (Z_FE * FARADAY)
    mass_log = _make_mass_log(mass_before=50.0, mass_gain=theoretical * 1.5)
    derived = compute_derived(ts, mass_log, cathodic_sign="negative")
    assert derived.faradaic_efficiency_percent is not None
    assert derived.faradaic_efficiency_percent > 150.0


# ═══════════════════════════════════════════════════════════════════════
#  5. plating_data.py flags current drift > 5%
# ═══════════════════════════════════════════════════════════════════════

def test_detects_current_drift_above_threshold():
    """Current drift > 5% triggers a flag."""
    ts = _make_timeseries(drift=True)
    ts["current_setpoint_A"] = -2.0  # constant setpoint
    flags = detect_anomalies(ts, current_drift_threshold=0.05)
    kinds = [f.kind for f in flags]
    assert "current_drift" in kinds


def test_no_current_drift_flag_when_within_threshold():
    """No current-drift flag when deviation < 5%."""
    ts = _make_timeseries(drift=False)
    ts["current_setpoint_A"] = -2.0
    flags = detect_anomalies(ts, current_drift_threshold=0.05)
    kinds = [f.kind for f in flags]
    assert "current_drift" not in kinds


# ═══════════════════════════════════════════════════════════════════════
#  6. plating_data.py flags temperature drift > 2 °C
# ═══════════════════════════════════════════════════════════════════════

def test_detects_temperature_drift_above_threshold():
    """Temperature drift > 2°C triggers a flag."""
    ts = _make_timeseries(temp_drift=True)
    flags = detect_anomalies(ts, temperature_drift_threshold_C=2.0)
    kinds = [f.kind for f in flags]
    assert "temperature_drift" in kinds


def test_no_temperature_drift_flag_when_within_threshold():
    """No temperature-drift flag when drift < 2°C."""
    ts = _make_timeseries(temp_drift=False)
    flags = detect_anomalies(ts, temperature_drift_threshold_C=2.0)
    kinds = [f.kind for f in flags]
    assert "temperature_drift" not in kinds


# ═══════════════════════════════════════════════════════════════════════
#  7. plating_data.py flags missing timestamps (gaps)
# ═══════════════════════════════════════════════════════════════════════

def test_detects_missing_timestamps():
    """A timestamp gap > 2× median triggers a missing_timestamps flag."""
    ts = _make_timeseries(gap_at=50)
    flags = detect_anomalies(ts)
    kinds = [f.kind for f in flags]
    assert "missing_timestamps" in kinds


# ═══════════════════════════════════════════════════════════════════════
#  8. run_manifest.py accepts plating experiment types
# ═══════════════════════════════════════════════════════════════════════

def test_all_plating_experiment_types_accepted():
    """Every plating experiment type passes validation."""
    for etype in VALID_EXPERIMENT_TYPES:
        data = _make_manifest(experiment_type=etype)
        report = validate_experiment_manifest(data)
        assert report.valid, f"Type '{etype}' rejected: {report.summary()}"


# ═══════════════════════════════════════════════════════════════════════
#  9. bath_batch.json schema validates a complete example
# ═══════════════════════════════════════════════════════════════════════

def test_bath_batch_schema_validates_complete_example(tmp_path):
    """A fully populated bath_batch.json loads without error."""
    batch = {
        "batch_id": "B-20260801-001",
        "date_mixed": "2026-08-01",
        "operator": "test",
        "composition": {
            "fe2_g_L": 50.0,
            "h3bo3_g_L": 30.0,
            "ascorbic_acid_g_L": 1.0,
            "pH": 3.2,
            "temperature_C": 25.0,
            "volume_mL": 500,
        },
        "source_chemicals": {
            "feso4_lot": "LOT-001",
            "water_source": "DI, 18 MΩ·cm",
        },
        "storage": {
            "container": "HDPE carboy",
            "cover": "sealed",
            "temperature_C": 22.0,
        },
    }
    path = tmp_path / "bath_batch.json"
    path.write_text(json.dumps(batch), encoding="utf-8")
    loaded = load_bath_batch(path)
    assert loaded["batch_id"] == "B-20260801-001"
    assert loaded["composition"]["fe2_g_L"] == 50.0


def test_bath_batch_rejects_missing_composition_key(tmp_path):
    """Missing composition fields are caught."""
    batch = {
        "batch_id": "B-20260801-001",
        "date_mixed": "2026-08-01",
        "operator": "test",
        "composition": {"fe2_g_L": 50.0},  # missing h3bo3_g_L, pH, volume_mL
        "storage": {"container": "HDPE"},
    }
    path = tmp_path / "bath_batch.json"
    path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ValueError, match="composition missing required key"):
        load_bath_batch(path)


# ═══════════════════════════════════════════════════════════════════════
# 10. Video index loading
# ═══════════════════════════════════════════════════════════════════════

def test_video_index_loaded_when_present(tmp_path):
    """load_plating_run picks up video_index.csv when present."""
    exp_dir = _write_experiment_dir(tmp_path)
    vi = pd.DataFrame({
        "timestamp_s": [0.0, 100.0],
        "camera": ["overhead", "panel"],
        "filename": ["overhead_0001.mp4", "panel_0001.mp4"],
        "event": ["start", "sample"],
    })
    vi.to_csv(exp_dir / "video_index.csv", index=False)
    run = load_plating_run(exp_dir)
    assert run.video_index is not None
    assert len(run.video_index) == 2


def test_video_index_none_when_absent(tmp_path):
    """load_plating_run returns video_index=None when file is absent."""
    exp_dir = _write_experiment_dir(tmp_path)
    run = load_plating_run(exp_dir)
    assert run.video_index is None


# ═══════════════════════════════════════════════════════════════════════
# 11. Video recording status validation
# ═══════════════════════════════════════════════════════════════════════

def test_video_status_none_requires_justification():
    """recording_status='none' without notes is rejected."""
    data = _make_manifest(video_status="none")
    report = validate_experiment_manifest(data)
    assert not report.valid
    assert any("video.notes" in i.path for i in report.errors)


def test_video_status_none_with_justification_accepted():
    """recording_status='none' with justification is accepted."""
    data = _make_manifest()
    data["video"] = {"recording_status": "none", "notes": "Camera battery died before start"}
    report = validate_experiment_manifest(data)
    assert report.valid


# ═══════════════════════════════════════════════════════════════════════
# 12. Charge and energy computation
# ═══════════════════════════════════════════════════════════════════════

def test_charge_computed_correctly():
    """Q = I × t for constant current (no mass_log needed)."""
    ts = _make_timeseries(n=50, dt=2.0, current=-3.0, voltage=2.0)
    derived = compute_derived(ts, cathodic_sign="negative")
    expected_charge = 3.0 * 49 * 2.0  # |I| × (n-1) × dt
    assert abs(derived.charge_C - expected_charge) / expected_charge < 0.02


def test_energy_computed_from_voltage_and_current():
    """Energy = ∫ V*|I| dt / 3600 for constant V and I."""
    ts = _make_timeseries(n=50, dt=2.0, current=-3.0, voltage=2.0)
    derived = compute_derived(ts, cathodic_sign="negative")
    expected_energy_Wh = 2.0 * 3.0 * 49 * 2.0 / 3600.0
    assert abs(derived.energy_Wh - expected_energy_Wh) / expected_energy_Wh < 0.02
