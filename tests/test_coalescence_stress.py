"""
Unit tests for crystallite coalescence stress, Hall-Petch strength, and Griffith cracking.
"""

from models.coalescence_stress import (
    CoalescenceStressParams,
    get_temperature_dependent_youngs_modulus_Pa,
    analyze_coalescence_stress,
)


def test_temperature_dependent_youngs_modulus():
    """Verify that Young's Modulus decreases slightly at elevated temperatures."""
    params = CoalescenceStressParams()

    e_25 = get_temperature_dependent_youngs_modulus_Pa(temperature_C=25.0, params=params)
    e_60 = get_temperature_dependent_youngs_modulus_Pa(temperature_C=60.0, params=params)

    assert e_60 < e_25
    assert e_25 > 1e11  # Around 200 GPa = 2e11 Pa


def test_analyze_coalescence_stress_grain_size_trends():
    """Verify that smaller grains drive higher tensile stress and lower safe cracking thickness."""
    params = CoalescenceStressParams()

    # Compare a fine grain (0.1 µm) vs a coarse grain (1.0 µm)
    res_fine = analyze_coalescence_stress(grain_size_um=0.1, temperature_C=60.0, params=params)
    res_coarse = analyze_coalescence_stress(grain_size_um=1.0, temperature_C=60.0, params=params)

    # 1. Stress scales inversely with L (tensile stress increases for fine grains)
    assert res_fine.coalescence_stress_MPa > res_coarse.coalescence_stress_MPa

    # 2. Yield strength scales inversely with sqrt(L) (Hall-Petch strength increases)
    assert res_fine.hall_petch_yield_strength_MPa > res_coarse.hall_petch_yield_strength_MPa

    # 3. Griffith cracking limit is smaller for fine grains because stored energy increases quadratically with stress
    assert res_fine.critical_crack_thickness_um < res_coarse.critical_crack_thickness_um
    assert res_fine.stored_strain_energy_per_um_J_m2 > res_coarse.stored_strain_energy_per_um_J_m2


def test_plastic_deformation_transition():
    """Verify that extremely small grains can trigger plastic deformation where stress exceeds strength."""
    params = CoalescenceStressParams()

    # Very small grains (e.g. 10 nm) generate immense stress exceeding Hall-Petch limits
    res_nanoscale = analyze_coalescence_stress(grain_size_um=0.01, temperature_C=60.0, params=params)
    assert res_nanoscale.is_plastically_deformed

    # Coarser grains remain elastic
    res_macroscale = analyze_coalescence_stress(grain_size_um=5.0, temperature_C=60.0, params=params)
    assert not res_macroscale.is_plastically_deformed
