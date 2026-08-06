"""Tests for cell heat balance and thermal management model."""

import pytest
from models.thermal_balance import CellThermalParams, simulate_thermal_transient, evaporative_heat_loss_W


def test_evaporative_loss():
    """Evaporative loss should be positive for T > T_amb and zero for T <= T_amb."""
    q_low = evaporative_heat_loss_W(T_C=20.0, T_amb_C=22.0, A_surface_m2=0.04, RH=0.5)
    q_high = evaporative_heat_loss_W(T_C=60.0, T_amb_C=22.0, A_surface_m2=0.04, RH=0.5)

    assert q_low == 0.0
    assert q_high > 5.0


def test_uncooled_cell_temperature_rise():
    """Uncooled cell operating at 10 A, 2.5 V should experience temperature rise."""
    p = CellThermalParams(V_cell=2.5, current_A=10.0, volume_L=2.0, T_init_C=22.0, T_amb_C=22.0, cooling_active=False)
    res = simulate_thermal_transient(p, t_end_hr=2.0)

    assert res["T_ss_C"] > p.T_init_C
    assert res["heat_gen_power_W"] > 0.0
    assert res["cooling_duty_50C_W"] >= 0.0


def test_active_jacket_reduces_temperature():
    """Jacket cooling should reduce steady-state temperature compared to uncooled."""
    p_uncooled = CellThermalParams(V_cell=2.5, current_A=10.0, T_init_C=22.0, T_amb_C=22.0, cooling_active=False)
    p_cooled = CellThermalParams(V_cell=2.5, current_A=10.0, T_init_C=22.0, T_amb_C=22.0, cooling_active=True, T_cool_in_C=15.0)

    res_u = simulate_thermal_transient(p_uncooled, t_end_hr=2.0)
    res_c = simulate_thermal_transient(p_cooled, t_end_hr=2.0)

    assert res_c["T_ss_C"] < res_u["T_ss_C"]


def test_configured_target_temperature_is_reported_separately_from_legacy_50c_field():
    """The integrated pipeline must be able to request the RC-1 60 °C target."""
    params = CellThermalParams(
        V_cell=2.5,
        current_A=10.0,
        T_init_C=25.0,
        T_amb_C=25.0,
        T_target_C=60.0,
    )
    result = simulate_thermal_transient(params, t_end_hr=0.05, dt_s=10.0)
    assert result["T_target_C"] == 60.0
    assert result["cooling_duty_target_W"] >= 0.0
    assert result["cooling_duty_50C_W"] >= 0.0


# ── 2026-08: FE-weighted thermoneutral heat generation ────────────────────


def test_legacy_constant_etherm_path_unchanged():
    """With current_efficiency=None, heat = I(V − E_THERM_FE) as before."""
    from models.thermal_balance import heat_generation_W
    # 10 A × (2.5 − 1.28) V = 12.2 W
    assert heat_generation_W(2.5, 10.0, None) == pytest.approx(12.2, abs=1e-6)


def test_all_her_generates_more_heat_than_fe_plating():
    """HER has E_therm ≈ 0, so all-HER operation dissipates I·V as heat."""
    from models.thermal_balance import heat_generation_W
    q_fe = heat_generation_W(2.5, 10.0, current_efficiency=1.0)
    q_her = heat_generation_W(2.5, 10.0, current_efficiency=0.0)
    assert q_her > q_fe
    # all HER: reversible power ~0 → Q = I·V = 25 W
    assert q_her == pytest.approx(25.0, abs=0.5)


def test_heat_falls_monotonically_with_fe():
    """Higher Faradaic efficiency means less irreversible heat per coulomb."""
    from models.thermal_balance import heat_generation_W
    qs = [heat_generation_W(2.5, 10.0, fe) for fe in (0.0, 0.5, 0.9, 1.0)]
    for a, b in zip(qs, qs[1:]):
        assert a > b


def test_fe_aware_thermal_transient_runs():
    """The transient integration accepts and uses current_efficiency."""
    p = CellThermalParams(
        V_cell=2.5, current_A=10.0, T_init_C=22.0, T_amb_C=22.0,
        current_efficiency=0.5, cooling_active=False,
    )
    res = simulate_thermal_transient(p, t_end_hr=0.1, dt_s=30.0)
    assert res["heat_gen_power_W"] > 0.0
    assert res["T_ss_C"] > 22.0
