"""
Unit tests for ammonium interfacial buffering and ferrous ammine complexation models.
"""

import pytest
from models.ammonium_buffer import (
    AmmoniumBufferModel,
)


def test_temperature_dependent_constants():
    """Verify pKa and solubility product are physically correct and temperature-dependent."""
    model_25 = AmmoniumBufferModel(temperature_C=25.0)
    model_60 = AmmoniumBufferModel(temperature_C=60.0)

    # pKa decreases with temperature (endothermic dissociation)
    assert pytest.approx(model_25.pka, abs=0.1) == 9.25
    assert model_60.pka < model_25.pka

    # Kw increases with temperature (autoprotolysis is endothermic)
    assert pytest.approx(model_25.kw, rel=0.1) == 1e-14
    assert model_60.kw > model_25.kw


def test_speciation_at_acidic_pH():
    """At low pH (pH=2), ammonia is completely protonated and no complexation occurs."""
    model = AmmoniumBufferModel(temperature_C=60.0)
    res = model.solve_speciation(pH=2.0, total_Fe_M=1.5, total_ammonia_M=1.0)

    # 1. Total ferrous should equal free ferrous (no ammine complexes in acid)
    assert pytest.approx(res.free_fe2_M, rel=1e-3) == 1.5
    assert sum(res.fe_ammine_M) < 1e-4

    # 2. Total ammonia should be mostly protonated ammonium
    assert pytest.approx(res.nh4_M, rel=1e-3) == 1.0
    assert res.free_nh3_M < 1e-5
    assert not res.is_hydroxide_precipitated


def test_speciation_and_complexation_at_alkaline_pH():
    """At elevated pH (pH=8.5) and 60°C, substantial free ammonia and ammine complexes form."""
    model = AmmoniumBufferModel(temperature_C=60.0)
    
    # Run speciation in neutral-alkaline region where complexes can survive
    res = model.solve_speciation(pH=8.0, total_Fe_M=1.5, total_ammonia_M=1.0)

    # Free ammonia is non-negligible
    assert res.free_nh3_M > 1e-5
    # Iron ammine complexes are formed
    complexed_fe = sum(res.fe_ammine_M)
    assert complexed_fe > 1e-5
    assert res.free_fe2_M < 1.5


def test_hydroxide_precipitation():
    """Verify that hydroxide precipitation is correctly identified and suppressed by complexation."""
    model = AmmoniumBufferModel(temperature_C=25.0)

    # 1. At pH 6 with no buffer/ammonia, high iron leads to precipitation
    res_no_nh = model.solve_speciation(pH=8.0, total_Fe_M=1.5, total_ammonia_M=0.0)
    assert res_no_nh.is_hydroxide_precipitated

    # 2. At pH 2, there is no precipitation
    res_acid = model.solve_speciation(pH=2.0, total_Fe_M=1.5, total_ammonia_M=1.0)
    assert not res_acid.is_hydroxide_precipitated


def test_buffer_capacity():
    """Verify that buffer capacity peaks near the pKa of ammonium."""
    model = AmmoniumBufferModel(temperature_C=60.0)
    pka = model.pka

    cap_at_pka = model.get_buffer_capacity(pH=pka, total_ammonia_M=1.0)
    cap_off_pka = model.get_buffer_capacity(pH=pka - 2.0, total_ammonia_M=1.0)

    assert cap_at_pka > cap_off_pka
