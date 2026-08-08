"""Fast contracts for V6 §7.1 N → Cottrell → Lüders screen."""

import pytest

from models.strain_aging import (
    cottrell_time_hours,
    deposit_nitrogen_ppm,
    evaluate_strain_aging,
    luders_strain_pct,
    model_scope,
    nitrogen_diffusivity_m2_s,
    sweep_ammonium,
    yield_return_mpa,
)


def test_nitrogen_uptake_rises_with_ammonium_and_clears_at_zero():
    assert deposit_nitrogen_ppm(total_ammonium_M=0.0) == pytest.approx(6.0)
    low = deposit_nitrogen_ppm(total_ammonium_M=0.5, pH=5.5)
    high = deposit_nitrogen_ppm(total_ammonium_M=2.0, pH=5.5)
    assert high > low
    # higher free NH3 at same total -> more N (higher pH)
    acid = deposit_nitrogen_ppm(total_ammonium_M=1.0, pH=3.0)
    mild = deposit_nitrogen_ppm(total_ammonium_M=1.0, pH=5.5)
    assert mild >= acid


def test_diffusivity_and_cottrell_time_arrhenius_and_concentration():
    d_cold = nitrogen_diffusivity_m2_s(20.0)
    d_hot = nitrogen_diffusivity_m2_s(100.0)
    assert d_hot > 10 * d_cold
    # higher T -> smaller t*
    assert cottrell_time_hours(20.0, temperature_C=60.0) < cottrell_time_hours(20.0, temperature_C=20.0)
    # higher C -> smaller t*
    assert cottrell_time_hours(40.0, temperature_C=20.0) < cottrell_time_hours(10.0, temperature_C=20.0)


def test_yield_return_grows_with_time_and_saturates_and_skin_pass_erases():
    c = 25.0
    d_0 = yield_return_mpa(c, aging_hours=0.0)
    d_24 = yield_return_mpa(c, aging_hours=24.0, temperature_C=20.0)
    d_500 = yield_return_mpa(c, aging_hours=500.0, temperature_C=20.0)
    assert d_0 == pytest.approx(0.0)
    assert 10 < d_24 < 45
    assert d_500 <= 60.0
    assert d_500 > d_24
    # 2 % skin-pass suppresses return
    prestrained = yield_return_mpa(c, aging_hours=500.0, temperature_C=20.0, pre_strain_pct=2.0)
    assert prestrained == pytest.approx(0.0, abs=1e-9)
    half = yield_return_mpa(c, aging_hours=24.0, temperature_C=20.0, pre_strain_pct=1.0)
    assert half == pytest.approx(d_24 * 0.5, rel=0.01)


def test_luders_scales_with_delta_sigma_and_grain_size():
    assert luders_strain_pct(0.0) == pytest.approx(0.0)
    base = luders_strain_pct(30.0, grain_size_um=20.0)
    assert 0.5 < base < 3.0
    coarse = luders_strain_pct(30.0, grain_size_um=80.0)
    assert coarse > base


def test_evaluate_verdict_clears_to_fails_with_ammonium_or_time():
    borate = evaluate_strain_aging(total_ammonium_M=0.0, storage_hours=24.0)
    assert borate.verdict == "clears"
    light = evaluate_strain_aging(total_ammonium_M=1.0, storage_hours=24.0, storage_temperature_C=20.0)
    heavy = evaluate_strain_aging(total_ammonium_M=2.0, storage_hours=240.0, storage_temperature_C=20.0)
    # more N and longer storage -> larger Δσ, larger Lüders
    assert heavy.delta_sigma_mpa >= light.delta_sigma_mpa
    assert heavy.luders_strain_pct >= light.luders_strain_pct
    # hot storage accelerates return (smaller t*)
    hot = evaluate_strain_aging(total_ammonium_M=1.0, storage_hours=24.0, storage_temperature_C=80.0)
    assert hot.delta_sigma_mpa > light.delta_sigma_mpa


def test_sweep_and_scope():
    rows = sweep_ammonium([0.0, 1.0, 2.0])
    assert len(rows) == 3
    assert rows[0]["c_n_ppm"] < rows[-1]["c_n_ppm"]
    scope = model_scope()
    assert scope["screening_flag"] == "unvalidated (L1)"
    assert any("ammonium_buffer" in s for s in scope["live_derivations"])
    assert any("scavenger" in s for s in scope["out_of_scope"])
