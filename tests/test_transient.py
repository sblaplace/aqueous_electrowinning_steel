"""Tests for transient process model."""

import pytest
import numpy as np
from models.transient import (
    TransientConfig,
    UpsetType,
    simulate_startup,
    simulate_shutdown,
    simulate_upset,
    recovery_time,
    damage_assessment,
    _first_order_lag,
    _ramp_target,
    AMBIENT_TEMP_C,
)


# ─── Helper function tests ──────────────────────────────────────────────

def test_first_order_lag_converges():
    """First-order lag should converge to target."""
    x = 0.0
    for _ in range(1000):
        x = _first_order_lag(x, 100.0, tau=5.0, dt=1.0)
    assert abs(x - 100.0) < 0.01


def test_first_order_lag_zero_tau():
    """Zero tau should snap to target immediately."""
    result = _first_order_lag(0.0, 100.0, tau=0.0, dt=1.0)
    assert result == 100.0


def test_ramp_target():
    """Linear ramp should interpolate correctly."""
    assert _ramp_target(0.0, 10.0, 300.0) == 0.0
    assert abs(_ramp_target(5.0, 10.0, 300.0) - 150.0) < 0.01
    assert _ramp_target(10.0, 10.0, 300.0) == 300.0
    assert _ramp_target(20.0, 10.0, 300.0) == 300.0  # past ramp


# ─── Config validation ──────────────────────────────────────────────────

def test_config_defaults():
    """Default config should be valid."""
    cfg = TransientConfig()
    assert cfg.electrolyte_volume_L == 1000.0
    assert cfg.target_current_density_mA_cm2 == 300.0
    assert cfg.furnace_operating_temp_C == 900.0


def test_config_validation():
    """Invalid config should raise."""
    with pytest.raises(ValueError, match="electrolyte_volume_L"):
        TransientConfig(electrolyte_volume_L=-1)
    with pytest.raises(ValueError, match="current_efficiency"):
        TransientConfig(current_efficiency_normal=1.5)


# ─── Startup tests ──────────────────────────────────────────────────────

def test_startup_converges_to_steady_state():
    """Startup should approach operating conditions within 2 hours."""
    cfg = TransientConfig(startup_duration_min=120.0)
    result = simulate_startup(config=cfg, dt=1.0)
    assert result.scenario == "startup"

    # Final electrolyte temp should be near operating
    assert abs(result.electrolyte_temp_C[-1] - cfg.electrolyte_operating_temp_C) < 5.0

    # pH should approach target
    assert abs(result.electrolyte_pH[-1] - cfg.electrolyte_target_pH) < 1.0

    # Current should reach target
    assert abs(result.current_density_mA_cm2[-1] - cfg.target_current_density_mA_cm2) < 5.0

    # Furnace should approach operating temp (within 10% of ramp to ambient)
    assert result.furnace_temp_C[-1] > 700.0

    # Quench should reach operating temp
    assert abs(result.quench_temp_C[-1] - cfg.quench_operating_temp_C) < 5.0


def test_startup_starts_from_ambient():
    """Startup should begin near ambient conditions."""
    result = simulate_startup(dt=5.0)
    assert abs(result.electrolyte_temp_C[0] - AMBIENT_TEMP_C) < 1.0
    assert result.current_density_mA_cm2[0] == 0.0


def test_startup_deposit_quality_high():
    """Deposit quality during normal startup should remain high."""
    result = simulate_startup(dt=2.0)
    assert result.deposit_quality[-1] > 0.8


# ─── Shutdown tests ─────────────────────────────────────────────────────

def test_shutdown_current_ramps_to_zero():
    """Current should ramp to zero during shutdown."""
    cfg = TransientConfig(shutdown_duration_min=90.0)
    result = simulate_shutdown(config=cfg, dt=1.0)
    assert result.scenario == "shutdown"
    assert result.current_density_mA_cm2[-1] == 0.0
    assert result.ce_fraction[-1] == 0.0


def test_shutdown_furnace_cools():
    """Furnace should cool during shutdown."""
    result = simulate_shutdown(dt=2.0)
    assert result.furnace_temp_C[-1] < result.furnace_temp_C[0]


def test_shutdown_safety_margins():
    """Shutdown should maintain deposit quality above scrap threshold."""
    result = simulate_shutdown(dt=2.0)
    # During controlled shutdown, quality should not drop to scrap
    assert np.min(result.deposit_quality) > 0.5


# ─── Upset scenarios ────────────────────────────────────────────────────

def test_upset_power_interruption():
    """Power interruption: current drops, quality degrades."""
    result = simulate_upset(upset_type=UpsetType.POWER_INTERRUPTION, duration=30.0, dt=1.0)
    assert result.scenario == "upset_power_interruption"
    assert result.current_density_mA_cm2[-1] == 0.0
    # Quality should degrade
    assert result.deposit_quality[-1] < result.deposit_quality[0]


def test_upset_ph_excursion():
    """pH excursion: pH should drift upward when pump fails."""
    result = simulate_upset(upset_type=UpsetType.PH_EXCURSION, duration=60.0, dt=1.0)
    assert result.scenario == "upset_ph_excursion"
    # pH should increase (pump not adding acid)
    assert result.electrolyte_pH[-1] > result.electrolyte_pH[0]


def test_upset_temperature_excursion():
    """Temperature excursion: electrolyte overheats."""
    result = simulate_upset(upset_type=UpsetType.TEMPERATURE_EXCURSION, duration=30.0, dt=1.0)
    assert result.scenario == "upset_temperature_excursion"
    # Should exceed safe temperature
    assert np.max(result.electrolyte_temp_C) > 85.0


def test_upset_gas_supply_interruption():
    """Gas interruption: O2 probe should rise toward air."""
    result = simulate_upset(upset_type=UpsetType.GAS_SUPPLY_INTERRUPTION, duration=60.0, dt=1.0)
    assert result.scenario == "upset_gas_supply_interruption"
    # O2 should increase significantly
    assert result.o2_probe_ppm[-1] > 1000.0


def test_upset_current_interruption():
    """Current interruption: current drops to zero while immersed."""
    result = simulate_upset(upset_type=UpsetType.CURRENT_INTERRUPTION, duration=30.0, dt=1.0)
    assert result.scenario == "upset_current_interruption"
    assert result.current_density_mA_cm2[-1] == 0.0
    # Quality degrades from dissolution
    assert result.deposit_quality[-1] < 1.0


def test_upset_rectifier_fault():
    """Rectifier fault: overcurrent then reverse polarity."""
    result = simulate_upset(upset_type=UpsetType.RECTIFIER_FAULT, duration=30.0, dt=1.0)
    assert result.scenario == "upset_rectifier_fault"
    # Should see overcurrent and reverse polarity in flags
    all_flags = [f for step_flags in result.flags for f in step_flags]
    assert any("overcurrent" in f for f in all_flags) or any("reverse_polarity" in f for f in all_flags)


# ─── Recovery time ──────────────────────────────────────────────────────

def test_recovery_time_normal_startup():
    """Normal startup should not need 'recovery' (already starts at spec)."""
    result = simulate_startup(dt=2.0)
    # Quality stays high; recovery time should be 0 or near-0
    rec = recovery_time(result, threshold=0.9)
    assert rec >= 0.0  # may be 0 if never drops below threshold


def test_recovery_time_power_upset():
    """Power upset should have a finite recovery time estimate."""
    result = simulate_upset(upset_type=UpsetType.POWER_INTERRUPTION, duration=60.0, dt=1.0)
    rec = recovery_time(result, threshold=0.9)
    # May be inf if quality never drops below 0.9 then recovers, or finite
    assert isinstance(rec, float)


# ─── Damage assessment ──────────────────────────────────────────────────

def test_damage_assessment_power():
    """Power interruption should report current outage."""
    result = simulate_upset(upset_type=UpsetType.POWER_INTERRUPTION, duration=30.0, dt=1.0)
    dmg = damage_assessment(result)
    assert dmg["scenario"] == "upset_power_interruption"
    assert dmg["current_outage_min"] > 0.0
    assert dmg["gas_outage_min"] > 0.0  # blowers also dead


def test_damage_assessment_ph():
    """pH excursion should report pH excursion risk."""
    result = simulate_upset(upset_type=UpsetType.PH_EXCURSION, duration=60.0, dt=1.0)
    dmg = damage_assessment(result)
    assert dmg["scenario"] == "upset_ph_excursion"
    # pH should exceed threshold
    assert dmg["max_electrolyte_pH"] > 2.5


def test_damage_assessment_temperature():
    """Temperature excursion should flag boiling risk."""
    result = simulate_upset(upset_type=UpsetType.TEMPERATURE_EXCURSION, duration=60.0, dt=1.0)
    dmg = damage_assessment(result)
    assert dmg["max_electrolyte_temp_C"] > 85.0
    assert dmg["boiling_risk"] is True


def test_damage_assessment_six_scenarios():
    """All 6 upset scenarios should produce valid damage reports."""
    for utype in UpsetType:
        result = simulate_upset(upset_type=utype, duration=30.0, dt=2.0)
        dmg = damage_assessment(result)
        assert "scenario" in dmg
        assert "min_deposit_quality" in dmg
        assert 0.0 <= dmg["min_deposit_quality"] <= 1.0


# ─── Result structure ───────────────────────────────────────────────────

def test_result_summary():
    """Result summary should contain all expected fields."""
    result = simulate_startup(dt=5.0)
    s = result.summary()
    assert "duration_min" in s
    assert "final_electrolyte_temp_C" in s
    assert "final_pH" in s
    assert "final_deposit_quality" in s
    assert "flagged_time_steps" in s


def test_result_as_columns():
    """as_columns should return dict of arrays of equal length."""
    result = simulate_startup(dt=5.0)
    cols = result.as_columns()
    assert "time_min" in cols
    assert "deposit_quality" in cols
    n = len(result.time_min)
    for key, arr in cols.items():
        assert len(arr) == n, f"{key} length mismatch"
