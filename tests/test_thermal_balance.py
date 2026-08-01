"""Tests for cell heat balance and thermal management model."""

from models.thermal_balance import CellThermalParams, simulate_thermal_transient, evaporative_heat_loss_W


def test_evaporative_loss():
    """Evaporative loss should be positive for T > T_amb and zero for T <= T_amb."""
    q_low = evaporative_heat_loss_W(T_C=20.0, T_amb_C=22.0, A_surf_m2=0.04, RH=0.5)
    q_high = evaporative_heat_loss_W(T_C=60.0, T_amb_C=22.0, A_surf_m2=0.04, RH=0.5)

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
