"""Tests for multicomponent electrolyte speciation and activity model."""

import pytest
import numpy as np
from models.speciation import SolutionComposition, solve_speciation, speciation_temperature_sweep, davies_A, davies_gamma


def test_davies_A_temperature_scaling():
    """A parameter should increase slightly with temperature as dielectric constant drops."""
    A_25 = davies_A(25.0)
    A_60 = davies_A(60.0)
    assert 0.49 < A_25 < 0.52
    assert A_60 > A_25


def test_davies_gamma():
    """Activity coefficient gamma for z=2 should be lower than for z=1 at same I > 0."""
    gamma1 = davies_gamma(z=1, I=0.5, A=0.509)
    gamma2 = davies_gamma(z=2, I=0.5, A=0.509)
    assert 0.0 < gamma2 < gamma1 < 1.0


def test_speciation_baseline():
    """Solve baseline 1 M FeSO4 + 0.5 M Na2SO4 solution."""
    comp = SolutionComposition(c_FeSO4=1.0, c_Na2SO4=0.5, c_H2SO4=0.01, T_C=50.0)
    res = solve_speciation(comp)
    
    assert res["ionic_strength_M"] > 1.0
    assert 0.0 < res["gamma_Fe2"] < 1.0
    assert 0.0 < res["c_Fe2_free_M"] <= 1.0
    assert res["c_FeSO4_pair_M"] > 0.0
    assert -0.6 < res["E_rev_Fe_V_SHE"] < -0.4
    assert 5.0 < res["pH_precip_Fe_OH2"] < 8.0
    assert res["conductivity_S_m"] > 0.0


def test_speciation_temperature_sweep():
    """Speciation sweep over temperature range 20-80 °C."""
    comp = SolutionComposition(c_FeSO4=1.0, c_Na2SO4=0.5)
    sweep = speciation_temperature_sweep(comp, T_min=20.0, T_max=80.0, num=5)
    
    assert len(sweep["temperature_C"]) == 5
    # Electrical conductivity should increase with temperature
    assert sweep["conductivity_S_m"][-1] > sweep["conductivity_S_m"][0]
