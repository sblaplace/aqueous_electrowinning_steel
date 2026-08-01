"""Tests for the life cycle assessment model."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.lca import (
    ElectricityMix,
    ChemicalSources,
    LCAResult,
    ComparisonTable,
    REFERENCE_ROUTES,
    compute_lca,
    compare_routes,
    sensitivity_to_electricity,
    breakeven_renewable_fraction,
)


# ── 1. Renewable electricity gives GWP below 0.5 kg CO₂/kg target ────
def test_renewable_gwp_below_target():
    """With 100% renewable electricity, GWP should be below 0.5 kg CO₂-eq/kg."""
    result = compute_lca(2.6, electricity_mix=ElectricityMix(renewable=1.0))
    assert result.gwp_kgCO2eq < 0.5, (
        f"Expected GWP < 0.5 with renewables, got {result.gwp_kgCO2eq:.4f}"
    )
    assert result.gwp_kgCO2eq > 0.0, "GWP must be positive"


# ── 2. Coal electricity raises GWP substantially ──────────────────────
def test_coal_gwp_exceeds_target():
    """With 100% coal, GWP should exceed 0.5 and approach BOF range."""
    result = compute_lca(2.6, electricity_mix=ElectricityMix(coal=1.0, renewable=0.0))
    assert result.gwp_kgCO2eq > 0.5, (
        f"Expected GWP > 0.5 with coal, got {result.gwp_kgCO2eq:.4f}"
    )


# ── 3. Electricity GWP scales linearly with energy input ──────────────
def test_gwp_scales_with_energy():
    """Doubling energy input should (approximately) double the electricity GWP."""
    r1 = compute_lca(2.0, electricity_mix=ElectricityMix(renewable=1.0))
    r2 = compute_lca(4.0, electricity_mix=ElectricityMix(renewable=1.0))
    # The non-electricity parts (chemicals, heat treatment, waste) are fixed,
    # so total GWP ratio should be less than 2 but electricity GWP ratio ≈ 2.
    elec_ratio = r2.electricity_gwp / max(r1.electricity_gwp, 1e-12)
    assert 1.9 < elec_ratio < 2.1, f"Expected ~2x electricity GWP, got {elec_ratio:.2f}"


# ── 4. Compare routes returns all reference routes ─────────────────────
def test_compare_routes_returns_all():
    """Comparison table should include all reference routes."""
    result = compute_lca(2.6, electricity_mix=ElectricityMix(renewable=1.0))
    table = compare_routes(result)
    assert len(table.routes) == len(REFERENCE_ROUTES)
    route_names = {r.route for r in table.routes}
    assert route_names == set(REFERENCE_ROUTES.keys())


# ── 5. Sensitivity to electricity returns expected mixes ───────────────
def test_sensitivity_returns_all_mixes():
    """Sensitivity function should return a result for each input mix."""
    mixes = {
        "coal": ElectricityMix(coal=1.0, renewable=0.0),
        "renewable": ElectricityMix(renewable=1.0),
    }
    results = sensitivity_to_electricity(2.6, mixes)
    assert set(results.keys()) == {"coal", "renewable"}
    assert isinstance(results["coal"], LCAResult)
    # Coal should have higher GWP than renewable
    assert results["coal"].gwp_kgCO2eq > results["renewable"].gwp_kgCO2eq


# ── 6. Breakeven fraction is between 0 and 1 ──────────────────────────
def test_breakeven_fraction_range():
    """Breakeven renewable fraction should be in [0, 1]."""
    frac = breakeven_renewable_fraction(2.6, target_co2_kg_per_kg=0.5)
    assert 0.0 <= frac <= 1.0, f"Expected fraction in [0,1], got {frac}"


# ── 7. Breakeven with impossible target returns 1.0 ────────────────────
def test_breakeven_impossible_target():
    """If target is below even 100% renewable GWP, breakeven is 1.0."""
    result_renewable = compute_lca(2.6, electricity_mix=ElectricityMix(renewable=1.0))
    impossible_target = result_renewable.gwp_kgCO2eq - 0.01
    frac = breakeven_renewable_fraction(2.6, target_co2_kg_per_kg=impossible_target)
    assert frac == 1.0, f"Expected 1.0 for impossible target, got {frac}"


# ── 8. Water consumption is higher with coal mix than renewable ────────
def test_water_higher_with_coal():
    """Coal electricity has higher water intensity than renewable."""
    coal = compute_lca(2.6, electricity_mix=ElectricityMix(coal=1.0, renewable=0.0))
    renew = compute_lca(2.6, electricity_mix=ElectricityMix(renewable=1.0))
    assert coal.water_L > renew.water_L, (
        f"Coal water={coal.water_L:.2f} should exceed renewable water={renew.water_L:.2f}"
    )


# ── 9. Chemical recycling reduces GWP ──────────────────────────────────
def test_chemical_recycling_reduces_gwp():
    """Acid recycling should lower the chemicals GWP component."""
    no_recycle = compute_lca(2.6, chemical_sources=ChemicalSources(acid_recycling_fraction=0.0))
    with_recycle = compute_lca(2.6, chemical_sources=ChemicalSources(acid_recycling_fraction=0.5))
    assert with_recycle.chemicals_gwp < no_recycle.chemicals_gwp


# ── 10. ElectricityMix rejects invalid fractions ───────────────────────
def test_invalid_mix_raises():
    """Mix fractions not summing to 1.0 should raise ValueError."""
    with pytest.raises(ValueError):
        ElectricityMix(coal=0.5, renewable=0.3)  # sums to 0.8


# ── 11. LCAResult to_dict has all expected keys ────────────────────────
def test_lca_result_to_dict():
    """to_dict should include all five impact categories."""
    result = compute_lca(2.6)
    d = result.to_dict()
    expected_keys = [
        "GWP (kg CO₂-eq/kg)",
        "Acidification (kg SO₂-eq/kg)",
        "Eutrophication (kg PO₄-eq/kg)",
        "Water consumption (L/kg)",
        "Land use (m²/kg)",
    ]
    for key in expected_keys:
        assert key in d, f"Missing key: {key}"
