"""
Unit tests for Mullins-Sekerka morphological wave instability and adatom surface diffusion.
"""

import pytest
from models.mullins_sekerka import (
    MullinsSekerkaParams,
    get_surface_diffusivity_m2_s,
    analyze_morphological_stability,
    analyze_pulse_stability,
)


def test_get_surface_diffusivity():
    """Verify surface diffusivity is Arrhenius-dependent and suppressed by additives."""
    params = MullinsSekerkaParams()

    # Arrhenius check: T-dependency
    d_s_25 = get_surface_diffusivity_m2_s(temperature_C=25.0, params=params)
    d_s_60 = get_surface_diffusivity_m2_s(temperature_C=60.0, params=params)

    assert d_s_60 > d_s_25
    assert d_s_25 > 1e-20

    # Additive suppression check
    d_s_60_blocked = get_surface_diffusivity_m2_s(
        temperature_C=60.0,
        additive_coverage_fraction=0.5,
        params=params,
    )
    assert d_s_60_blocked < d_s_60


def test_analyze_morphological_stability_trends():
    """Verify that current density and temperature trends on dendritic onset are physical."""
    params = MullinsSekerkaParams()

    # 1. Higher current density accelerates dendritic onset (smaller critical thickness h_crit)
    res_low_j = analyze_morphological_stability(
        j_Fe_mA_cm2=50.0,
        temperature_C=60.0,
        params=params,
    )
    res_high_j = analyze_morphological_stability(
        j_Fe_mA_cm2=300.0,
        temperature_C=60.0,
        params=params,
    )

    assert res_high_j.v_dep_nm_s > res_low_j.v_dep_nm_s
    assert res_high_j.critical_transition_thickness_um < res_low_j.critical_transition_thickness_um

    # 2. Higher temperature accelerates surface diffusion, delaying dendritic onset (higher h_crit)
    res_low_T = analyze_morphological_stability(
        j_Fe_mA_cm2=150.0,
        temperature_C=25.0,
        params=params,
    )
    res_high_T = analyze_morphological_stability(
        j_Fe_mA_cm2=150.0,
        temperature_C=60.0,
        params=params,
    )

    assert res_high_T.critical_transition_thickness_um > res_low_T.critical_transition_thickness_um


def test_wavelength_and_growth_bounds():
    """Verify that the dominant dendrite wavelength and growth rates are in physical ranges."""
    params = MullinsSekerkaParams()

    res = analyze_morphological_stability(
        j_Fe_mA_cm2=150.0,
        temperature_C=60.0,
        params=params,
    )

    # Dendrite spacing (λ_max) should typically be in the micrometer to tens of micrometers scale
    assert 0.01 <= res.dominant_wavelength_um <= 1000.0
    assert res.critical_wavelength_um < res.dominant_wavelength_um
    assert res.max_growth_rate_1_s > 0.0


def test_analyze_pulse_stability():
    """Verify that pulse plating delays dendritic onset compared to continuously running peak current."""
    params = MullinsSekerkaParams()

    res = analyze_pulse_stability(
        j_peak_mA_cm2=300.0,
        duty_cycle=0.20,
        temperature_C=60.0,
        params=params,
    )

    # Average deposition velocity under pulse is less than peak
    assert res.average_deposition_velocity_nm_s < 300.0 * 10.0 # and is proportional to average current
    # Pulse-plating (matched at lower average current) improves h_crit over continuously plating at high peak current
    assert res.improvement_factor > 1.0
    assert res.pc_critical_thickness_um > res.dc_critical_thickness_um
