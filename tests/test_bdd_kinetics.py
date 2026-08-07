"""
Unit tests for Bockris–Dražic–Despic (BDD) multi-step iron deposition microkinetics.
"""

import pytest
from models.bdd_kinetics import (
    BDDKineticParams,
    feoh_plus_concentration_mol_L,
    solve_bdd_kinetics,
)


def test_feoh_concentration_increases_with_ph():
    """Verify that hydrolysis produces more electroactive FeOH+ at higher pH."""
    c_feoh_ph2 = feoh_plus_concentration_mol_L(1.5, ph=2.0)
    c_feoh_ph3 = feoh_plus_concentration_mol_L(1.5, ph=3.0)
    c_feoh_ph4 = feoh_plus_concentration_mol_L(1.5, ph=4.0)

    assert c_feoh_ph3 > c_feoh_ph2 * 8.0  # Approx 10x per pH unit
    assert c_feoh_ph4 > c_feoh_ph3 * 8.0


def test_bdd_dual_tafel_slopes():
    """Verify BDD low-overpotential (~40 mV/dec) vs high-overpotential (~120 mV/dec) regime."""
    # Low overpotential (40 mV): Step 2 is rate-determining
    res_low = solve_bdd_kinetics(overpotential_V=0.040, ph=2.5, fe2_mol_L=1.5)
    assert res_low.intermediate_coverage_theta < 0.35
    assert 30.0 <= res_low.apparent_tafel_slope_mV_dec <= 65.0

    # High overpotential (350 mV): Step 1 is rate-determining, theta approaches 1
    res_high = solve_bdd_kinetics(overpotential_V=0.350, ph=2.5, fe2_mol_L=1.5)
    assert res_high.intermediate_coverage_theta > 0.70
    assert 85.0 <= res_high.apparent_tafel_slope_mV_dec <= 150.0


def test_bdd_ph_acceleration():
    """Verify that higher surface pH accelerates Fe deposition rate (positive reaction order in OH-)."""
    res_acidic = solve_bdd_kinetics(overpotential_V=0.100, ph=2.0, fe2_mol_L=1.5)
    res_mild = solve_bdd_kinetics(overpotential_V=0.100, ph=3.0, fe2_mol_L=1.5)

    assert res_mild.cathodic_current_density_A_m2 > res_acidic.cathodic_current_density_A_m2
    assert 0.0 < res_mild.reaction_order_oh_minus <= 1.0
