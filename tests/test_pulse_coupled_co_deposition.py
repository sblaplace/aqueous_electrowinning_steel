"""Tests for pulse-coupled co-deposition coupling."""

import pytest
from models.co_deposition import (
    AnomalousFeNiKinetics,
    PhaseIIICoDeposition,
    build_phase3_model,
    surface_pH_from_current,
    surface_pH_from_pulse,
    effective_mass_transport_enhancement,
)


def test_surface_pH_pulse_recovery():
    bulk = 3.5
    pH_dc = surface_pH_from_current(100, bulk)
    pH_pe = surface_pH_from_pulse(50, 100, 0.5, bulk, waveform="pe")
    pH_pre = surface_pH_from_pulse(50, 100, 0.5, bulk, waveform="pre")
    # Pulse off-time should lower surface pH vs DC at peak
    assert pH_pe <= surface_pH_from_current(100, bulk) + 0.01
    assert pH_pe >= bulk
    assert pH_pre <= pH_pe  # PRE extra H+ generation
    assert pH_pre >= bulk


def test_mass_transport_enhancement():
    base = 5e-5
    eff_dc = effective_mass_transport_enhancement(100, 100, 1.0, "dc", base)
    eff_pe = effective_mass_transport_enhancement(50, 100, 0.5, "pe", base)
    eff_pre = effective_mass_transport_enhancement(50, 100, 0.5, "pre", base)
    assert eff_dc == base
    assert eff_pe < base
    assert eff_pre <= eff_pe  # PRE more efficient


def test_kinetics_pulsed_methods():
    kin = AnomalousFeNiKinetics(pH=3.5, temperature_C=60)
    pH_avg = kin.surface_pH(50)
    pH_pulsed = kin.surface_pH_pulsed(50, 100, 0.5, "pe")
    assert pH_pulsed >= kin.pH
    delta_eff = kin.effective_boundary_layer_pulsed(50, 100, 0.5, "pe")
    assert delta_eff <= kin.boundary_layer_m


def test_phase3_pulsed_run():
    model = build_phase3_model(mechanism_fe_ni="hydroxide_suppression")
    dc = model.run_at_current(50.0)
    pe = model.run_at_current_pulsed(50.0, 100.0, duty_cycle=0.5, waveform="pe")
    pre = model.run_at_current_pulsed(50.0, 100.0, duty_cycle=0.5, waveform="pre")

    for res in (dc, pe, pre):
        assert "fe_wt_percent" in res["alloy_kinetics"]
        assert "ni_wt_percent" in res["alloy_kinetics"]

    # Pulsed should have pH diagnostics
    assert "pulsed_surface_pH" in pe["alloy_kinetics"]
    assert pe["alloy_kinetics"]["pulsed_surface_pH"] >= model.pH

    # Sweep pulsed
    sweep = model.run_sweep_pulsed([20, 50, 100], j_peak_factor=2.0, duty_cycle=0.5, waveform="pe")
    assert len(sweep["j_avg_mA_cm2"]) == 3
    assert len(sweep["fe_wt_percent"]) == 3
