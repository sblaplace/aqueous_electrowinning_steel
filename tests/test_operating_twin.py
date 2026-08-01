"""Tests for the safety boundary around the operating twin."""

import pytest

from models.operating_twin import (
    ControlCommand,
    OperatingTwin,
    SensorSnapshot,
    TwinConfig,
    TwinMode,
)


@pytest.fixture
def config():
    return TwinConfig(
        cell_id="REF-CELL-001",
        max_current_A=10.0,
        max_current_density_mA_cm2=200.0,
        max_voltage_V=5.0,
        min_temperature_C=20.0,
        max_temperature_C=80.0,
        min_fe2_M=0.2,
        max_fe2_M=2.0,
        min_pH=1.0,
        max_pH=5.0,
        target_current_A=5.0,
        target_temperature_C=50.0,
        current_ramp_A_per_s=1.0,
    )


def snap(t=0.0, current=0.0, voltage=2.0, temperature=50.0,
         pH=2.0, fe2=1.0, quality=None):
    return SensorSnapshot(
        timestamp_s=t, current_A=current, voltage_V=voltage,
        temperature_C=temperature, pH=pH, fe2_M=fe2,
        cathode_area_cm2=100.0,
        sensor_quality=quality or {}, source_run_id="R-001",
    )


def test_default_is_advisory_and_never_actuates(config):
    twin = OperatingTwin(config)
    twin.update(snap())
    command = twin.command()
    assert command.mode is TwinMode.ADVISORY
    assert command.current_A == 0.0
    assert "actuation_not_armed" in command.reasons


def test_arming_requires_exact_qualified_cell(config):
    twin = OperatingTwin(config)
    with pytest.raises(PermissionError):
        twin.arm_actuation("wrong-cell")
    twin.arm_actuation("REF-CELL-001")
    twin.update(snap(t=1.0, current=0.0))
    command = twin.command(now_s=1.0)
    assert command.mode is TwinMode.ACTUATION
    # 1 s ramp limit, not an instantaneous jump to the target.
    assert command.current_A == pytest.approx(1.0)


def test_hard_limit_latches_trip_and_zeroes_command(config):
    twin = OperatingTwin(config)
    twin.arm_actuation("REF-CELL-001")
    state = twin.update(snap(t=0.0, voltage=6.0))
    assert state.mode is TwinMode.TRIPPED
    assert "voltage_limit" in state.trip_reasons
    command = twin.command()
    assert command.mode is TwinMode.TRIPPED
    assert command.current_A == 0.0


def test_stale_snapshot_trips_when_clock_is_advanced(config):
    twin = OperatingTwin(config)
    twin.arm_actuation("REF-CELL-001")
    state = twin.update(snap(t=0.0), now_s=10.0)
    assert state.mode is TwinMode.TRIPPED
    assert "stale_snapshot" in state.trip_reasons


def test_bad_quality_and_timestamp_regression_trip(config):
    twin = OperatingTwin(config)
    twin.update(snap(t=2.0))
    state = twin.update(snap(t=1.0, quality={"temperature": "invalid"}))
    assert state.mode is TwinMode.TRIPPED
    assert state.trip_reasons == ("timestamp_regression",)


def test_trip_requires_safe_fresh_snapshot_and_operator_to_clear(config):
    twin = OperatingTwin(config)
    twin.update(snap(t=0.0, voltage=6.0))
    with pytest.raises(RuntimeError):
        twin.clear_trip("operator-1", snap(t=1.0, voltage=6.0))
    twin.clear_trip("operator-1", snap(t=1.0))
    assert twin.mode is TwinMode.ADVISORY
    assert twin.command().current_A == 0.0


def test_charge_and_theoretical_mass_ledger(config):
    twin = OperatingTwin(config)
    twin.update(snap(t=0.0, current=2.0))
    state = twin.update(snap(t=10.0, current=2.0))
    assert state.charge_cathodic_C == pytest.approx(20.0)
    assert state.theoretical_fe_mass_g > 0
    assert state.update_count == 2
