"""Tests for carburization screening model."""

import pytest
import numpy as np
from models.carburization import (
    CarburizationModel,
    CarburizationParams,
    carbon_diffusivity_m2_s,
    hardness_from_carbon_wt,
    tempered_hardness,
    estimate_carburizing_time_for_case_depth,
)


def test_diffusivity_increases_with_T():
    D_low, _ = carbon_diffusivity_m2_s(800, phase="austenite")
    D_high, _ = carbon_diffusivity_m2_s(950, phase="austenite")
    assert D_high > D_low
    assert D_low > 0


def test_diffusivity_ferrite_vs_austenite():
    D_fer, ph_f = carbon_diffusivity_m2_s(700, phase="ferrite")
    D_aust, ph_a = carbon_diffusivity_m2_s(900, phase="austenite")
    # At lower T ferrite can be faster than austenite at higher T? Actually ferrite faster per atom, but check positive
    assert D_fer > 0 and D_aust > 0
    assert ph_f == "ferrite" and ph_a == "austenite"


def test_hardness_monotonic():
    hv_low = hardness_from_carbon_wt(0.1)
    hv_mid = hardness_from_carbon_wt(0.5)
    hv_high = hardness_from_carbon_wt(0.8)
    assert hv_mid > hv_low
    assert hv_high >= hv_mid
    assert hv_high <= 900


def test_hardness_quench_rate():
    hv_fast = hardness_from_carbon_wt(0.5, quench_rate_C_s=200.0)
    hv_slow = hardness_from_carbon_wt(0.5, quench_rate_C_s=5.0)
    # slow quench → more bainite → softer
    assert hv_fast >= hv_slow


def test_tempered_softening():
    hv_q = hardness_from_carbon_wt(0.6, quench_rate_C_s=200)
    hv_t_low = tempered_hardness(hv_q, temper_T_C=200, temper_t_hr=1.0)
    hv_t_high = tempered_hardness(hv_q, temper_T_C=600, temper_t_hr=2.0)
    assert hv_t_low < hv_q
    assert hv_t_high < hv_t_low


def test_profile_at_time():
    params = CarburizationParams(temperature_C=900, surface_carbon_wt_percent=1.1, sheet_thickness_um=1000)
    model = CarburizationModel(params)
    prof = model.profile_at_time(t_hr=1.0, n_points=100)
    assert len(prof.x_um) == 100
    assert len(prof.c_wt_percent) == 100
    # surface should be near Cs, both sides enriched for finite slab
    assert prof.c_wt_percent[0] >= 1.0
    assert prof.c_wt_percent[-1] >= 1.0  # far surface also enriched (both sides)
    # midplane should be lower than surface at short time
    mid = len(prof.c_wt_percent)//2
    assert prof.c_wt_percent[mid] <= prof.c_wt_percent[0] + 1e-9
    assert prof.c_wt_percent.min() >= params.initial_carbon_wt_percent - 1e-6


def test_simulate_time_series():
    params = CarburizationParams(temperature_C=900, surface_carbon_wt_percent=1.1, sheet_thickness_um=1000)
    model = CarburizationModel(params)
    result = model.simulate(duration_hr=2.0, dt_hr=1.0, n_x=100, save_profiles_every_hr=1.0)
    assert len(result.time_hr) == 3  # 0,1,2
    # case depth should grow with time
    assert result.effective_case_depth_035_um[-1] >= result.effective_case_depth_035_um[0]
    assert result.carbon_uptake_g_m2[-1] >= result.carbon_uptake_g_m2[0]
    assert len(result.profiles) >= 2


def test_time_estimate():
    t = estimate_carburizing_time_for_case_depth(
        target_case_depth_um=500,
        temperature_C=900,
        surface_c_wt=1.1,
        threshold_c_wt=0.35,
        initial_c_wt=0.02,
    )
    assert 0.1 < t < 20.0  # hours screening range
    # Deeper case needs longer time
    t2 = estimate_carburizing_time_for_case_depth(
        target_case_depth_um=1000,
        temperature_C=900,
        surface_c_wt=1.1,
        threshold_c_wt=0.35,
        initial_c_wt=0.02,
    )
    assert t2 > t


def test_composite_strength():
    params = CarburizationParams(temperature_C=900, sheet_thickness_um=1000)
    model = CarburizationModel(params)
    comp = model.composite_strength_estimate(case_depth_um=200, core_yield_MPa=300)
    assert 0 < comp["case_fraction"] <= 1.0
    assert comp["sigma_composite_035_MPa"] >= comp["core_yield_MPa"]
