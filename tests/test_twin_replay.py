"""Safety invariants for the scripted composed-twin replay."""

import pytest

from models.operating_twin import OperatingTwin, SensorSnapshot, ShutdownRequest, TwinConfig
from models.twin_replay import run_replay

pytestmark = pytest.mark.slow


def test_full_fault_matrix_degrades_safely():
    report = run_replay()
    assert report["all_passed"]
    assert len(report["scenarios"]) == 9
    assert all(row["shutdown_request"] is not None for row in report["scenarios"])
    assert all(row["operating_twin"]["command"]["current_A"] == 0.0
               for row in report["scenarios"])


def test_shutdown_request_is_request_only():
    config = TwinConfig(
        cell_id="contract", max_current_A=10, max_current_density_mA_cm2=200,
        max_voltage_V=5, min_temperature_C=0, max_temperature_C=80,
        min_fe2_M=.2, max_fe2_M=2, min_pH=.5, max_pH=5,
        max_wind_gust_m_s=40,
    )
    twin = OperatingTwin(config)
    snapshot = SensorSnapshot(
        timestamp_s=0, current_A=1, voltage_V=2.5, temperature_C=25,
        pH=2, fe2_M=1, cathode_area_cm2=100, wind_gust_m_s=50,
        source_run_id="contract-test",
    )
    request = twin.shutdown_request(snapshot)
    assert isinstance(request, ShutdownRequest)
    assert request.action == "storm_mode_hold_high_wind"
    assert "execute" not in dir(request)
    twin.update(snapshot)
    command = twin.command()
    assert command.current_A == 0
    assert command.mode.value == "tripped"
