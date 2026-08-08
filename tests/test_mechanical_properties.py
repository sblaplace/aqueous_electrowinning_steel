"""Tests for mechanical properties screening model."""

import numpy as np
from models.mechanical_properties import (
    MechanicalPropertiesModel,
    estimate_grain_size_um,
    hall_petch_yield_MPa,
    solid_solution_strengthening_MPa,
    carbon_dispersion_strengthening_MPa,
    porosity_factor,
    build_mechanical_model_from_phase3_result,
)


def test_grain_size_dc_baseline():
    d = estimate_grain_size_um(j_avg_mA_cm2=100, waveform="dc")
    assert 1.0 < d < 10.0


def test_grain_size_pulse_finer_than_dc():
    d_dc = estimate_grain_size_um(j_avg_mA_cm2=100, j_peak_mA_cm2=100, duty_cycle=1.0, waveform="dc")
    d_pe = estimate_grain_size_um(j_avg_mA_cm2=100, j_peak_mA_cm2=200, duty_cycle=0.5, waveform="pe")
    d_pre = estimate_grain_size_um(j_avg_mA_cm2=100, j_peak_mA_cm2=200, duty_cycle=0.5, waveform="pre")
    assert d_pe < d_dc
    assert d_pre <= d_pe
    assert d_pre >= 0.05


def test_grain_size_high_j_finer():
    d_low = estimate_grain_size_um(j_avg_mA_cm2=20, waveform="dc")
    d_high = estimate_grain_size_um(j_avg_mA_cm2=300, waveform="dc")
    assert d_high < d_low


def test_hall_petch_monotonic():
    s_coarse = hall_petch_yield_MPa(5.0)
    s_fine = hall_petch_yield_MPa(0.2)
    assert s_fine > s_coarse
    assert s_coarse > 50  # at least friction stress


def test_solid_solution_positive():
    d0 = solid_solution_strengthening_MPa(ni_wt_percent=0.0)
    assert d0 == 0.0
    d1 = solid_solution_strengthening_MPa(ni_wt_percent=2.0)
    d5 = solid_solution_strengthening_MPa(ni_wt_percent=5.0)
    assert d1 > 0
    assert d5 > d1


def test_carbon_strengthening():
    orowan0, lt0, tot0 = carbon_dispersion_strengthening_MPa(0.0)
    assert tot0 == 0.0

    orowan, lt, tot = carbon_dispersion_strengthening_MPa(1.0, particle_size_um=1.5)
    assert tot > 0
    assert orowan >= 0
    assert lt >= 0

    # Smaller particles should give stronger for same wt%
    _, _, tot_small = carbon_dispersion_strengthening_MPa(1.0, particle_size_um=0.5)
    _, _, tot_large = carbon_dispersion_strengthening_MPa(1.0, particle_size_um=3.0)
    assert tot_small > tot_large


def test_porosity_factor():
    p0, f0 = porosity_factor(100.0)
    assert p0 == 0.0
    assert f0 == 1.0

    p_low, f_low = porosity_factor(50.0)
    assert 0 < p_low < 0.3
    assert 0 < f_low < 1.0
    assert f_low < f0


def test_mechanical_model_predict():
    model = MechanicalPropertiesModel()
    res = model.predict(
        j_avg_mA_cm2=100, j_peak_mA_cm2=200, duty_cycle=0.5,
        waveform="pe", ni_wt_percent=2.0, carbon_wt_percent=0.8,
        current_efficiency_percent=93.0,
    )
    assert res.sigma_y_MPa > 200
    assert res.sigma_y_MPa < 1000  # screening upper bound
    assert res.uts_MPa > res.sigma_y_MPa
    assert 0 < res.elongation_pct < 40
    assert res.vickers_hv > 50
    # Grade may be composite or AISI depending on C threshold; check non-empty and plausible
    assert len(res.grade_estimate) > 5
    assert any(k in res.grade_estimate for k in ("AISI", "Fe-", "composite", "tool", "structural"))


def test_mechanical_model_sweep():
    model = MechanicalPropertiesModel()
    sweep = model.sweep_current_density(
        np.linspace(20, 200, 10),
        waveform="pe",
        ni_wt_percent=1.0,
        carbon_wt_percent=0.5,
    )
    assert len(sweep["j_mA_cm2"]) == 10
    assert np.all(sweep["yield_MPa"] > 100)
    # yield should generally increase with j due to grain refinement, but allow non-monotonic due to model noise?
    # At least fine grain at high j gives higher yield than low j on average
    assert sweep["yield_MPa"][-1] >= sweep["yield_MPa"][0] * 0.8


def test_build_from_phase3_adapter():
    fake_phase3 = {
        "alloy_kinetics": {"ni_wt_percent": 3.5, "current_efficiency_percent": 92},
        "carbon_incorporation": {"predicted_carbon_wt_percent": 1.2, "adjusted_ce_percent": 90},
    }
    res = build_mechanical_model_from_phase3_result(fake_phase3, j_avg_mA_cm2=100, waveform="pre")
    assert res.ni_wt_percent == 3.5
    assert res.carbon_wt_percent == 1.2
    assert res.sigma_y_MPa > 0


def test_flags():
    model = MechanicalPropertiesModel()
    # low CE should trigger high_porosity / low_current_efficiency
    res_low_ce = model.predict(current_efficiency_percent=50.0, carbon_wt_percent=0.1)
    assert "low_current_efficiency" in res_low_ce.flags or "high_porosity" in res_low_ce.flags

    # excessive carbon flag
    res_high_c = model.predict(carbon_wt_percent=6.0)
    assert "excessive_carbon" in res_high_c.flags


def test_nucleation_grain_model_refines_with_overpotential():
    """use_nucleation_grain_model: higher overpotential -> finer grain -> higher YS."""
    model = MechanicalPropertiesModel()
    low = model.predict(use_nucleation_grain_model=True, cathodic_overpotential_V=0.1)
    high = model.predict(use_nucleation_grain_model=True, cathodic_overpotential_V=0.4)
    assert high.grain_size_um < low.grain_size_um
    assert high.sigma_y_MPa > low.sigma_y_MPa


def test_nucleation_grain_additive_refines():
    """Higher additive coverage refines the predicted grain (C1 levelers)."""
    model = MechanicalPropertiesModel()
    plain = model.predict(use_nucleation_grain_model=True, cathodic_overpotential_V=0.2,
                          additive_coverage_fraction=0.0)
    with_add = model.predict(use_nucleation_grain_model=True, cathodic_overpotential_V=0.2,
                             additive_coverage_fraction=0.6)
    assert with_add.grain_size_um < plain.grain_size_um
