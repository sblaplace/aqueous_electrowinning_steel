"""
Unit tests for nucleation-density -> grain-size -> Hall-Petch (Round 5, C1).
"""


from models.nucleation_grain import (
    grain_size_um,
    hall_petch_yield_MPa,
    nucleation_density_1_m2,
    recipe_to_grain_and_strength,
)


def test_nucleation_density_rises_with_overpotential():
    """Higher overpotential -> higher nucleation-site density."""
    lo = nucleation_density_1_m2(0.05, 60.0)
    hi = nucleation_density_1_m2(0.30, 60.0)
    assert hi > lo


def test_grain_fines_with_overpotential():
    """Higher overpotential -> finer grain (smaller d)."""
    coarse = grain_size_um(0.05, 60.0)
    fine = grain_size_um(0.30, 60.0)
    assert fine < coarse


def test_grain_fines_with_additive():
    """Leveler/brightener coverage refines the grain (higher effective N0)."""
    plain = grain_size_um(0.15, 60.0, additive_coverage_fraction=0.0)
    blocked = grain_size_um(0.15, 60.0, additive_coverage_fraction=0.5)
    assert blocked < plain
    assert blocked > 0.0


def test_hall_petch_finer_grains_stronger():
    """Hall-Petch: finer grains -> higher yield strength."""
    weak = hall_petch_yield_MPa(5.0)
    strong = hall_petch_yield_MPa(0.2)
    assert strong > weak


def test_recipe_to_strength_monotonic():
    """Higher overpotential recipe -> finer grain -> higher YS."""
    lo = recipe_to_grain_and_strength(0.05, 60.0)
    hi = recipe_to_grain_and_strength(0.40, 60.0)
    assert hi["grain_size_um"] < lo["grain_size_um"]
    assert hi["hall_petch_yield_MPa"] > lo["hall_petch_yield_MPa"]


def test_grain_size_in_sane_range():
    """Predicted grain size stays within a physically reasonable range."""
    d = grain_size_um(0.2, 60.0)
    assert 0.01 <= d <= 20.0  # µm
