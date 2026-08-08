"""
Unit tests for Marangoni / electrocapillary surface flows (Round 5, D1).
"""

import pytest

from models.marangoni import (
    effective_diffusion_layer_um,
    marangoni_velocity_m_s,
    surface_tension_N_m,
    surface_tension_gradient_N_m2,
)


def test_surface_tension_decreases_with_temperature():
    """Higher temperature -> lower surface tension."""
    cold = surface_tension_N_m(temperature_C=40.0)
    hot = surface_tension_N_m(temperature_C=90.0)
    assert hot < cold
    assert cold > 0.0


def test_additive_lowers_surface_tension():
    """Additive coverage lowers surface tension (surfactant-like)."""
    clean = surface_tension_N_m(additive_coverage_fraction=0.0)
    loaded = surface_tension_N_m(additive_coverage_fraction=0.8)
    assert loaded < clean


def test_gradient_drives_velocity():
    """A temperature gradient produces a non-zero Marangoni velocity."""
    v0 = marangoni_velocity_m_s(temperature_gradient_K_m=0.0)
    v1 = marangoni_velocity_m_s(temperature_gradient_K_m=10.0)
    assert v0 == pytest.approx(0.0)
    assert v1 > 0.0


def test_marangoni_thins_boundary_layer():
    """Marangoni stirring reduces the effective diffusion-layer thickness."""
    calm = effective_diffusion_layer_um(temperature_gradient_K_m=0.0)
    stirred = effective_diffusion_layer_um(temperature_gradient_K_m=20.0)
    assert stirred["delta_effective_um"] < calm["delta_effective_um"]
    assert stirred["delta_effective_um"] > 0.0


def test_forced_flow_adds_to_mixing():
    """Adding forced flow further reduces delta."""
    base = effective_diffusion_layer_um(temperature_gradient_K_m=5.0,
                                        forced_flow_velocity_m_s=0.0)
    forced = effective_diffusion_layer_um(temperature_gradient_K_m=5.0,
                                          forced_flow_velocity_m_s=0.01)
    assert forced["delta_effective_um"] < base["delta_effective_um"]


def test_potential_gradient_contributes():
    """An electrocapillary potential gradient also stirs the surface."""
    grad = surface_tension_gradient_N_m2(potential_gradient_V_m=5.0)
    assert grad != pytest.approx(0.0)
