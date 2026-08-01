"""Tests for the thermomechanical (cold-roll + recrystallization) model."""

import numpy as np
import pytest

from models.thermomechanical import (
    ThermomechanicalModel,
    ThermomechanicalParams,
    RollingSchedule,
    jmak_fraction_recrystallized,
    jmak_rate_constant_1_s,
    recrystallized_grain_size_um,
    grain_growth_um,
    time_for_fraction,
)


# ── Rolling / strain ────────────────────────────────────────────────────
def test_true_strain_increases_with_reduction():
    low = RollingSchedule(total_reduction=0.2, n_passes=2).total_true_strain
    high = RollingSchedule(total_reduction=0.7, n_passes=2).total_true_strain
    assert high > low > 0


def test_true_strain_total_matches_ln_ratio():
    r = RollingSchedule(total_reduction=0.5, n_passes=3)
    assert r.total_true_strain == pytest.approx(np.log(1.0 / 0.5), rel=1e-9)
    assert len(r.per_pass_reductions) == 3


def test_true_strain_explicit_passes():
    r = RollingSchedule(reductions=[0.25, 0.25, 0.25])
    expected = sum(np.log(1.0 / (1.0 - 0.25)) for _ in range(3))
    assert r.total_true_strain == pytest.approx(expected, rel=1e-9)


def test_invalid_reduction_rejected():
    with pytest.raises(ValueError):
        RollingSchedule(total_reduction=0.0)
    with pytest.raises(ValueError):
        RollingSchedule(total_reduction=0.99)
    with pytest.raises(ValueError):
        RollingSchedule(total_reduction=0.5, n_passes=0)


# ── Recrystallization kinetics ──────────────────────────────────────────
def test_jmak_rate_increases_with_temperature():
    params = ThermomechanicalParams()
    k_low = jmak_rate_constant_1_s(600, 0.5, params)
    k_high = jmak_rate_constant_1_s(800, 0.5, params)
    assert k_high > k_low > 0


def test_jmak_rate_increases_with_strain():
    params = ThermomechanicalParams()
    k_low = jmak_rate_constant_1_s(700, 0.2, params)
    k_high = jmak_rate_constant_1_s(700, 0.8, params)
    assert k_high > k_low


def test_fraction_recrystallized_trends():
    params = ThermomechanicalParams()
    t = np.linspace(0, 3600, 200)
    x = jmak_fraction_recrystallized(t, 700, 0.5, params)
    assert x[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(x) >= 0)
    assert 0 <= x[-1] <= 1
    assert x[-1] > 0.9  # 1 h at 700 °C fully recrystallizes


def test_fraction_monotonic_in_time_at_short_times():
    params = ThermomechanicalParams()
    t = np.array([0.0, 30.0, 60.0, 120.0, 600.0])
    x = jmak_fraction_recrystallized(t, 700, 0.5, params)
    assert np.all(np.diff(x) > 0)


def test_time_for_fraction_scales_inversely_with_temperature():
    params = ThermomechanicalParams()
    t50_hot = time_for_fraction(0.5, 750, 0.5, params)
    t50_cold = time_for_fraction(0.5, 600, 0.5, params)
    assert 0 < t50_hot < t50_cold


def test_recrystallized_grain_finer_with_more_strain():
    params = ThermomechanicalParams()
    d_lo = recrystallized_grain_size_um(1.0, 0.2, 700, params)
    d_hi = recrystallized_grain_size_um(1.0, 0.9, 700, params)
    assert d_hi < d_lo


def test_recrystallized_grain_coarser_with_starting_grain():
    params = ThermomechanicalParams()
    d_fine = recrystallized_grain_size_um(0.5, 0.5, 700, params)
    d_coarse = recrystallized_grain_size_um(5.0, 0.5, 700, params)
    assert d_coarse > d_fine


def test_grain_growth_increases_with_time_and_temperature():
    params = ThermomechanicalParams()
    d0 = 12.0
    d_short = grain_growth_um(d0, 60.0, 700, params)
    d_long = grain_growth_um(d0, 3600.0, 700, params)
    d_hot = grain_growth_um(d0, 60.0, 900, params)
    assert d_long >= d_short >= d0
    assert d_hot > d_short


# ── End-to-end predict ──────────────────────────────────────────────────
def test_predict_annealed_softer_and_more_ductile_than_deposit():
    model = ThermomechanicalModel()
    res = model.predict()
    # fine-grained electrodeposit is strong but brittle; anneal recovers ductility
    assert res.annealed_yield_MPa < res.deposit_yield_MPa
    assert res.annealed_elongation_pct > res.deposit_elongation_pct
    assert res.final_grain_um > res.deposit_grain_um


def test_predict_fully_recrystallized_at_defaults():
    res = ThermomechanicalModel().predict()
    assert res.fraction_recrystallized >= 0.99
    assert res.flags == [] or "incomplete_recrystallization" not in res.flags


def test_incomplete_recrystallization_flag_low_temperature():
    params = ThermomechanicalParams(anneal_temperature_C=500, anneal_time_min=10)
    res = ThermomechanicalModel(params).predict()
    assert res.fraction_recrystallized < 0.99
    assert "incomplete_recrystallization" in res.flags


def test_time_series_shapes():
    res = ThermomechanicalModel().predict()
    assert len(res.time_s) == len(res.fraction_recrystallized_series)
    assert len(res.time_s) == len(res.grain_size_series_um)
    assert res.time_s[0] == 0.0
    assert np.all(np.diff(res.grain_size_series_um) >= 0)


def test_summary_has_expected_keys():
    res = ThermomechanicalModel().predict()
    s = res.summary()
    for k in ("annealed_yield_MPa", "annealed_grade", "final_grain_um",
              "annealing_energy_kWh_per_kg", "fraction_recrystallized"):
        assert k in s


def test_sweep_temperature():
    out = ThermomechanicalModel().sweep_temperature()
    assert len(out["T_C"]) == len(out["D_final_um"])
    assert len(out["frac_rx"]) == len(out["yield_MPa"])
    # higher T -> coarser final grain once fully recrystallized
    assert out["D_final_um"][-1] > out["D_final_um"][0]


def test_sweep_reduction():
    out = ThermomechanicalModel().sweep_reduction()
    # more reduction -> finer recrystallized grains
    assert out["D_rx_um"][-1] < out["D_rx_um"][0]


def test_sweep_time():
    out = ThermomechanicalModel().sweep_time()
    assert len(out["time_min"]) == len(out["yield_MPa"])
    assert np.all(np.diff(out["frac_rx"]) >= 0)


# ── Validation ──────────────────────────────────────────────────────────
def test_invalid_params_rejected():
    for kwargs in (
        {"anneal_temperature_C": 200},
        {"anneal_temperature_C": 1100},
        {"anneal_time_min": 0},
        {"deposit_grain_size_um": 0.01},
        {"deposit_grain_size_um": 200},
        {"furnace_efficiency": 1.5},
    ):
        with pytest.raises(ValueError):
            ThermomechanicalParams(**kwargs)


def test_anneal_energy_zero_delta_at_ambient():
    params = ThermomechanicalParams(anneal_temperature_C=400, anneal_time_min=5)
    model = ThermomechanicalModel(params)
    assert model.anneal_energy_kWh_per_kg() > 0
