"""
Unit tests for membrane core Ohmic heating and trans-membrane thermal profile model.
"""

import pytest
from models.membrane_thermal import (
    MembraneThermalParams,
    solve_membrane_temperature_profile,
)


def test_zero_current_isothermal():
    """At j = 0 mA/cm², there is no Ohmic heating and membrane remains isothermal."""
    params = MembraneThermalParams()

    res = solve_membrane_temperature_profile(
        j_mA_cm2=0.0,
        t_cath_bulk_C=60.0,
        t_ano_bulk_C=60.0,
        params=params,
    )

    assert res.p_mem_W_m2 == 0.0
    assert pytest.approx(res.t_cath_surface_C, abs=1e-3) == 60.0
    assert pytest.approx(res.t_ano_surface_C, abs=1e-3) == 60.0
    assert pytest.approx(res.t_peak_core_C, abs=1e-3) == 60.0
    assert pytest.approx(res.crossover_acceleration_factor, abs=1e-3) == 1.0
    assert res.is_thermally_safe


def test_zero_current_temperature_gradient():
    """At j = 0 mA/cm² with different bulk temperatures, a linear conduction profile is established."""
    params = MembraneThermalParams()

    res = solve_membrane_temperature_profile(
        j_mA_cm2=0.0,
        t_cath_bulk_C=70.0,
        t_ano_bulk_C=50.0,
        params=params,
    )

    assert res.p_mem_W_m2 == 0.0
    # Convective cooling creates a gradient, surfaces should be between bulk T
    assert 50.0 < res.t_ano_surface_C < res.t_cath_surface_C < 70.0
    assert pytest.approx(res.mean_membrane_temp_C, abs=0.1) == (res.t_cath_surface_C + res.t_ano_surface_C) / 2.0


def test_high_current_heating():
    """At 300 mA/cm², Ohmic heating creates a peak temperature strictly above bulk."""
    params = MembraneThermalParams()

    res = solve_membrane_temperature_profile(
        j_mA_cm2=300.0,
        t_cath_bulk_C=60.0,
        t_ano_bulk_C=60.0,
        params=params,
    )

    assert res.p_mem_W_m2 > 1000.0  # (3000 A/m²)^2 * 3e-4 ohm m² = 2700 W/m²
    assert res.t_peak_core_C > 60.0
    assert res.t_cath_surface_C > 60.0
    assert res.t_ano_surface_C > 60.0
    # Ferric crossover is accelerated by peak temperature and Arrhenius kinetics
    assert res.crossover_acceleration_factor > 1.0


def test_cooling_coefficient_impact():
    """Slowing electrolyte flow (lower heat transfer h) increases peak core temperature."""
    params_well_cooled = MembraneThermalParams(h_cath_W_m2K=2000.0, h_ano_W_m2K=2000.0)
    params_poorly_cooled = MembraneThermalParams(h_cath_W_m2K=200.0, h_ano_W_m2K=200.0)

    res_well = solve_membrane_temperature_profile(
        j_mA_cm2=300.0,
        t_cath_bulk_C=60.0,
        t_ano_bulk_C=60.0,
        params=params_well_cooled,
    )

    res_poor = solve_membrane_temperature_profile(
        j_mA_cm2=300.0,
        t_cath_bulk_C=60.0,
        t_ano_bulk_C=60.0,
        params=params_poorly_cooled,
    )

    assert res_poor.t_peak_core_C > res_well.t_peak_core_C
    assert res_poor.crossover_acceleration_factor > res_well.crossover_acceleration_factor
