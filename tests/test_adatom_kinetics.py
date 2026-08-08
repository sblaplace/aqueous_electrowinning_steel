"""
Unit tests for adatom surface-diffusion / kink incorporation (Round 5, C3).
"""


from models.adatom_kinetics import (
    crystallization_overpotential_V,
    off_time_healing_length_m,
    surface_diffusivity_m2_s,
    surface_exchange_current_A_m2,
)


def test_surface_diffusivity_rises_with_temperature():
    """Higher temperature -> higher surface diffusivity."""
    cool = surface_diffusivity_m2_s(40.0)
    warm = surface_diffusivity_m2_s(90.0)
    assert warm > cool
    assert cool > 0.0


def test_additive_suppresses_surface_diffusion():
    """Additive coverage lowers surface diffusivity."""
    clean = surface_diffusivity_m2_s(60.0, additive_coverage_fraction=0.0)
    blocked = surface_diffusivity_m2_s(60.0, additive_coverage_fraction=0.8)
    assert blocked < clean


def test_crystallization_overpotential_rises_with_current():
    """Higher iron current -> higher crystallization overpotential."""
    lo = crystallization_overpotential_V(1000.0, 60.0)
    hi = crystallization_overpotential_V(10000.0, 60.0)
    assert hi["crystallization_overpotential_V"] > lo["crystallization_overpotential_V"]


def test_additive_raises_crystallization_overpotential():
    """Additives slow surface diffusion -> higher crystallization eta."""
    clean = crystallization_overpotential_V(3000.0, 60.0, additive_coverage_fraction=0.0)
    blocked = crystallization_overpotential_V(3000.0, 60.0, additive_coverage_fraction=0.7)
    assert blocked["crystallization_overpotential_V"] > clean["crystallization_overpotential_V"]


def test_off_time_healing_length_positive():
    """Off-time healing length grows with time and is positive."""
    short = off_time_healing_length_m(0.001, 60.0)
    long = off_time_healing_length_m(0.1, 60.0)
    assert long > short
    assert short > 0.0


def test_surface_exchange_current_positive():
    """Surface exchange current is positive and finite."""
    assert surface_exchange_current_A_m2(60.0) > 0.0
