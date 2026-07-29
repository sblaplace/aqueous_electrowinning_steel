"""Tests for membrane fouling model — CSTR loop degradation."""

import numpy as np
import pytest

from models.membrane_fouling import (
    CleaningAgent,
    CleaningParams,
    CleaningResult,
    CSTRFoulingCoupling,
    FluxDeclineResult,
    FoulingRateParams,
    FoulingSimulationResult,
    HermiaModel,
    MembraneFoulingModel,
    MembraneParams,
    hermia_flux,
    membrane_replacement_cost,
)


# ── Fixtures ───────────────────────────────────────────────────────────────
def default_model(**overrides) -> MembraneFoulingModel:
    """Return a baseline model with high-ish fouling for test visibility."""
    return MembraneFoulingModel(**overrides)


def high_fouling_model() -> MembraneFoulingModel:
    """Model with aggressive fouling (high pH, hard water, long idle)."""
    return MembraneFoulingModel(
        operating_pH=5.0,
        hardness_mg_L=400.0,
        idle_time_hr=48.0,
    )


# ── Test 1: flux declines monotonically ────────────────────────────────────
def test_flux_declines_monotonically():
    """J(t) must never increase in the absence of cleaning."""
    model = high_fouling_model()
    result = model.simulate_flux_decline(duration_hr=2000, dt_hr=1.0)
    diffs = np.diff(result.flux_L_m2_hr)
    assert np.all(diffs <= 0), "Flux must decline monotonically without cleaning"


# ── Test 2: cleaning restores partial flux ─────────────────────────────────
def test_cleaning_restores_partial_flux():
    """After a cleaning event flux should jump upward."""
    model = high_fouling_model()
    result = model.simulate_with_cleaning(
        duration_hr=500, dt_hr=1.0, cleaning_interval_hr=100.0,
    )
    # Find the first cleaning event — flux should increase at that point
    events = result.cleaning_events
    assert len(events) > 0, "Should have at least one cleaning event"
    clean_time = events[0][0]
    idx = np.searchsorted(result.time_hr, clean_time)
    if idx > 0 and idx < len(result.flux_L_m2_hr) - 1:
        assert result.flux_L_m2_hr[idx] > result.flux_L_m2_hr[idx - 1], \
            "Flux should increase after cleaning"


# ── Test 3: higher impurity concentration accelerates fouling ──────────────
def test_higher_impurity_accelerates_fouling():
    """Model with higher coupling sensitivity should foul faster."""
    low = MembraneFoulingModel(
        coupling=CSTRFoulingCoupling(rejection_fouling_coupling=0.1),
        operating_pH=5.0,
    )
    high = MembraneFoulingModel(
        coupling=CSTRFoulingCoupling(rejection_fouling_coupling=2.0),
        operating_pH=5.0,
    )
    r_low = low.simulate_flux_decline(duration_hr=1000, dt_hr=1.0)
    r_high = high.simulate_flux_decline(duration_hr=1000, dt_hr=1.0)
    # Higher coupling → higher rejection → higher impurity accumulation
    assert r_high.impurity_M[-1] >= r_low.impurity_M[-1]


# ── Test 4: optimal cleaning interval is positive and finite ───────────────
def test_optimal_cleaning_interval_positive_and_finite():
    model = high_fouling_model()
    interval = model.find_optimal_cleaning_interval(
        duration_hr=2000, n_points=20,
    )
    assert interval > 0, "Optimal interval must be positive"
    assert np.isfinite(interval), "Optimal interval must be finite"


# ── Test 5: integration with closed-loop model ─────────────────────────────
def test_integration_with_closed_loop_model():
    """MembraneReplacementCost feeds the techno-economic model."""
    mem = MembraneParams(area_m2=10.0, membrane_cost_per_m2=200.0)
    cost = membrane_replacement_cost(mem, n_replacements=3)
    assert cost["total_membrane_cost"] == pytest.approx(
        3 * (10 * 200 + 500)
    )
    assert cost["membrane_cost_per_event"] == pytest.approx(2500.0)


# ── Test 6: Hermia model variants implemented ─────────────────────────────
def test_hermia_model_variants():
    """All four Hermia models produce sensible flux curves."""
    t = np.linspace(0, 1000, 100)
    J0 = 100.0
    K = 1e-3

    for model in HermiaModel:
        J = hermia_flux(t, J0, K, model)
        assert J.shape == t.shape
        assert np.all(J > 0), f"{model.value}: flux must be positive"
        assert np.all(J <= J0), f"{model.value}: flux must not exceed J₀"
        # Must be monotonically non-increasing
        assert np.all(np.diff(J) <= 1e-10), f"{model.value}: must be non-increasing"


# ── Test 7: parameter validation ───────────────────────────────────────────
def test_parameter_validation():
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        MembraneParams(area_m2=0)
    with pytest.raises(ValueError, match="positive"):
        MembraneParams(clean_water_flux_L_m2_hr=-1)
    with pytest.raises(ValueError, match="non-negative"):
        FoulingRateParams(fe_oh3_base_rate=-0.1)
    with pytest.raises(ValueError, match="must lie"):
        CSTRFoulingCoupling(base_rejection=0.0)
    with pytest.raises(ValueError, match="positive"):
        MembraneFoulingModel().simulate_flux_decline(duration_hr=0)
    with pytest.raises(ValueError, match="positive"):
        MembraneFoulingModel().simulate_flux_decline(dt_hr=-1)


# ── Test 8: resistance accumulates and rejection increases ─────────────────
def test_resistance_accumulates_and_rejection_increases():
    model = high_fouling_model()
    result = model.simulate_flux_decline(duration_hr=500, dt_hr=1.0)
    assert result.total_resistance[-1] > result.total_resistance[0]
    assert result.rejection[-1] >= result.rejection[0]


# ── Test 9: cleaning efficiency is mechanism-specific ──────────────────────
def test_cleaning_efficiency_mechanism_specific():
    model = default_model()
    eff_acid = model.cleaning_efficiency(CleaningAgent.ACID_WASH)
    eff_naoh = model.cleaning_efficiency(CleaningAgent.NAOH_WASH)
    # Acid is best for Fe(OH)₃, NaOH for organics
    assert eff_acid["fe_oh3"] > eff_naoh["fe_oh3"]
    assert eff_naoh["organic"] > eff_acid["organic"]


# ── Test 10: full simulate returns all components ──────────────────────────
def test_full_simulate_returns_all_components():
    model = high_fouling_model()
    result = model.simulate(duration_hr=500, dt_hr=2.0)
    assert isinstance(result, FoulingSimulationResult)
    assert isinstance(result.flux_decline, FluxDeclineResult)
    assert isinstance(result.cleaning, CleaningResult)
    assert "total_throughput_L" in result.economics
    summary = result.summary()
    assert "flux_decline" in summary
    assert "n_cleanings" in summary


# ── Test 11: zero fouling rates preserve initial flux ──────────────────────
def test_zero_fouling_preserves_flux():
    model = MembraneFoulingModel(
        fouling=FoulingRateParams(
            fe_oh3_base_rate=0.0,
            caso4_base_rate=0.0,
            organic_base_rate=0.0,
            biofilm_base_rate=0.0,
        ),
        operating_pH=2.0,
        hardness_mg_L=0.0,
        idle_time_hr=0.0,
    )
    result = model.simulate_flux_decline(duration_hr=500, dt_hr=1.0)
    assert np.allclose(result.flux_L_m2_hr, model.membrane.clean_water_flux_L_m2_hr)


# ── Test 12: biofilm activates after idle threshold ───────────────────────
def test_biofilm_activates_after_idle():
    short_idle = MembraneFoulingModel(
        idle_time_hr=0.0,
        fouling=FoulingRateParams(
            fe_oh3_base_rate=0.0, caso4_base_rate=0.0, organic_base_rate=0.0,
        ),
    )
    long_idle = MembraneFoulingModel(
        idle_time_hr=48.0,
        fouling=FoulingRateParams(
            fe_oh3_base_rate=0.0, caso4_base_rate=0.0, organic_base_rate=0.0,
        ),
    )
    assert short_idle._biofilm_rate() == 0.0
    assert long_idle._biofilm_rate() > 0.0
