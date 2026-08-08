"""Tests for the oxygen-in-iron deposit-quality screening model."""

import numpy as np
import pytest

from models.oxygen_in_iron import (
    OxygenInIronModel,
    OxygenInIronParams,
    SCREENING_FLAG,
    oxygen_mass_rate_kg_m2_s,
    oxygen_wt_percent,
    oxygen_ppm,
    deposit_density_kg_m3,
    oxygen_strengthening_MPa,
    cold_rollability,
    precipitation_flux_from_pulse,
    RHO_FE,
    M_O_KG,
    N_O_PER_FE,
)


# ── Oxygen budget ──────────────────────────────────────────────────────────


def test_oxygen_mass_rate_zero_at_zero_flux():
    assert oxygen_mass_rate_kg_m2_s(0.0) == 0.0
    assert oxygen_mass_rate_kg_m2_s(-5.0) == 0.0


def test_oxygen_mass_rate_scales_with_flux_and_capture():
    r1 = oxygen_mass_rate_kg_m2_s(1e-4, capture_fraction=0.1)
    r2 = oxygen_mass_rate_kg_m2_s(2e-4, capture_fraction=0.1)
    r3 = oxygen_mass_rate_kg_m2_s(1e-4, capture_fraction=0.2)
    # Doubling the flux doubles the O mass rate
    assert abs(r2 - 2.0 * r1) < 1e-20
    # Doubling the capture fraction doubles the O mass rate
    assert abs(r3 - 2.0 * r1) < 1e-20
    # Unit sanity: flux (mol/m²/s) * 2 O * M_O gives kg/m²/s
    expected = 1e-4 * 0.1 * N_O_PER_FE * M_O_KG
    assert abs(r1 - expected) < 1e-24


def test_oxygen_ppm_monotonic_in_flux_and_deposition():
    # More precipitation → more O
    a = oxygen_ppm(1e-5, j_mA_cm2=100, current_efficiency_percent=90)
    b = oxygen_ppm(1e-4, j_mA_cm2=100, current_efficiency_percent=90)
    assert b > a > 0
    # More Fe deposition (higher j) dilutes O
    c = oxygen_ppm(1e-4, j_mA_cm2=200, current_efficiency_percent=90)
    assert c < b
    # O ppm rises with capture fraction
    d = oxygen_ppm(1e-4, j_mA_cm2=100, capture_fraction=0.2)
    assert d > b


def test_oxygen_wt_percent_ppm_consistency():
    wt = oxygen_wt_percent(1e-4, j_mA_cm2=100, current_efficiency_percent=90)
    ppm = oxygen_ppm(1e-4, j_mA_cm2=100, current_efficiency_percent=90)
    assert abs(ppm - wt * 1e4) < 1e-6


def test_oxygen_level_is_physically_reasonable():
    # A screening-heavy precipitation flux must land in the 1–2.5 wt% band,
    # positive and monotonic (never negative, never absurd).
    lo = oxygen_ppm(1e-5, j_mA_cm2=100, current_efficiency_percent=90)
    hi = oxygen_ppm(1e-3, j_mA_cm2=100, current_efficiency_percent=90)
    assert 0.1 < lo < hi < 2.5e4


# ── Density ────────────────────────────────────────────────────────────────


def test_density_is_bulk_fe_at_zero_o():
    assert deposit_density_kg_m3(0.0) == RHO_FE


def test_density_decreases_with_o():
    d0 = deposit_density_kg_m3(0)
    d500 = deposit_density_kg_m3(500)
    d2000 = deposit_density_kg_m3(2000)
    assert d500 < d0
    assert d2000 < d500
    # Still physically dense (a few % below bulk Fe even at 2000 ppm)
    assert d2000 > 7000.0


# ── Strengthening ──────────────────────────────────────────────────────────


def test_oxygen_strengthening_zero_and_monotonic():
    assert oxygen_strengthening_MPa(0.0) == 0.0
    s1 = oxygen_strengthening_MPa(500)
    s2 = oxygen_strengthening_MPa(1000)
    s3 = oxygen_strengthening_MPa(2000)
    assert s3 > s2 > s1 > 0
    # ~100 MPa per 1000 ppm order of magnitude
    assert 50.0 < s2 < 300.0


# ── Cold-roll gate ─────────────────────────────────────────────────────────


def test_cold_rollability_regimes():
    free = cold_rollability(200)
    assert free["rollable"] is True and free["status"] == "free"
    marginal = cold_rollability(700)
    assert marginal["rollable"] is False and marginal["status"] == "marginal"
    forbidden = cold_rollability(1200)
    assert forbidden["rollable"] is False and forbidden["status"] == "forbidden"


def test_cold_rollability_honesty_flag():
    r = cold_rollability(800)
    assert r["flag"] == SCREENING_FLAG


# ── Pulse coupling ─────────────────────────────────────────────────────────


def test_precipitation_flux_from_pulse_rises_with_pH():
    # Higher bulk pH drives surface pH up → more Fe(OH)₂ supersaturation → flux
    f_acid = precipitation_flux_from_pulse(
        100, 200, 0.5, "pe", bath_pH=3.0, temperature_C=60.0)[0]
    f_less_acid = precipitation_flux_from_pulse(
        100, 200, 0.5, "pe", bath_pH=5.0, temperature_C=60.0)[0]
    assert f_less_acid >= f_acid


def test_precipitation_flux_from_pulse_nonnegative():
    f, ph = precipitation_flux_from_pulse(
        50, 100, 0.5, "pe", bath_pH=3.0, temperature_C=60.0)
    assert f >= 0.0
    assert ph >= 3.0


# ── Model class ────────────────────────────────────────────────────────────


def test_model_predict_returns_expected_keys():
    r = OxygenInIronModel().predict(
        j_avg_mA_cm2=100, waveform="pe", bath_pH=3.5,
        current_efficiency_percent=90,
    )
    for k in ["o_ppm", "deposit_density_kg_m3", "delta_oxygen_strength_MPa",
              "cold_rollable", "cold_roll_status", "flag"]:
        assert k in r
    assert r["flag"] == SCREENING_FLAG
    assert r["o_ppm"] >= 0.0


def test_model_accepts_external_precipitation_flux():
    # Explicit flux from a diffusion-layer solve path is used verbatim
    r = OxygenInIronModel().predict(
        j_avg_mA_cm2=100, waveform="dc",
        precipitation_flux_mol_m2_s=1e-4, current_efficiency_percent=90,
    )
    assert r["precipitation_flux_mol_m2_s"] == 1e-4


def test_model_upper_bound_yield_with_base():
    r = OxygenInIronModel().predict(
        j_avg_mA_cm2=100, waveform="pe",
        precipitation_flux_mol_m2_s=1e-4,
        yield_MPa=250.0,
    )
    assert r["yield_upper_bound_MPa"] == pytest.approx(
        250.0 + r["delta_oxygen_strength_MPa"])


def test_model_params_validation():
    with pytest.raises(ValueError):
        OxygenInIronParams(capture_fraction=1.5)
    with pytest.raises(ValueError):
        OxygenInIronParams(free_o_ppm=500, forbidden_o_ppm=400)


def test_model_waveform_ranges_are_sane():
    m = OxygenInIronModel()
    # Sweep DC/PE/PRE — all must return finite, non-negative O
    for wf in ["dc", "pe", "pre"]:
        r = m.predict(j_avg_mA_cm2=100, waveform=wf, bath_pH=3.5)
        assert np.isfinite(r["o_ppm"])
        assert r["o_ppm"] >= 0.0


def test_edge_effect_wiring_changes_roll_gate():
    """include_edge_effect uses edge O for the cold-roll gate (Round 5 D2)."""
    m = OxygenInIronModel()
    base = m.predict(j_avg_mA_cm2=150, bath_pH=3.5)
    edge = m.predict(j_avg_mA_cm2=150, bath_pH=3.5, include_edge_effect=True)
    assert "edge_effect" in edge
    assert edge["edge_effect"]["edge_o_ppm"] > edge["o_ppm"]
    assert edge["edge_effect"]["roll_gate_o_ppm"] > base["o_ppm"]
