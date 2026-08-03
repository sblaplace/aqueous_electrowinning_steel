"""Tests for the cathode-channel gas hold-up / two-phase model.

Covers:
  - Faradaic gas stoichiometry and wet-gas expansion
  - Bubble mechanics: Fritz, shear detachment, Stokes/Harmathy rise
  - Drift-flux void fraction limits and monotonicity
  - Bruggeman conductivity and surface coverage
  - Bubble microconvection and effective boundary layer
  - Axial hold-up profile accumulation and current redistribution
  - Coupled gas <-> current <-> FE fixed point (with a surrogate FE model)
  - Hydrogen safety: LFL margin, dilution requirement, accumulation time
  - Screening sweeps, measurement protocol and model_scope contract
"""

import json
import math

import numpy as np
import pytest

from models.electrochemistry import FARADAY
from models.gas_holdup import (
    BRUGGEMAN_EXPONENT,
    CONTACT_ANGLE_H2_DEG,
    G_ACCEL,
    LFL_H2_VOL_FRAC,
    MU_LIQUID_PA_S,
    P_ATM_PA,
    RHO_LIQUID_KG_M3,
    THETA_MAX_H2,
    UFL_H2_VOL_FRAC,
    Z_H2,
    Z_O2,
    ChannelGeometry,
    bruggeman_conductivity,
    combined_boundary_layer_m,
    current_density_sweep,
    departure_diameter_m,
    drift_flux_void_fraction,
    faradaic_gas_flow_mol_s,
    fritz_detachment_diameter_m,
    gas_volumetric_flow_m3_s,
    height_scaling_screen,
    holdup_profile,
    hydrogen_flow_L_h,
    hydrogen_safety,
    measurement_protocol,
    model_scope,
    oxygen_flow_L_h,
    shear_detachment_diameter_m,
    solve_coupled,
    solve_current_distribution,
    surface_coverage_fraction,
    terminal_rise_velocity_m_s,
    vogt_mass_transfer_coefficient_m_s,
    water_vapor_pressure_Pa,
)


# ─── Gas generation ───────────────────────────────────────────────────

def test_faraday_stoichiometry_hydrogen():
    """1 A of pure HER makes exactly I/(2F) mol/s of H2."""
    assert faradaic_gas_flow_mol_s(1.0, Z_H2, 1.0) == pytest.approx(1.0 / (2 * FARADAY))


def test_hydrogen_and_oxygen_are_two_to_one():
    """Water splitting gives 2 volumes H2 per volume O2 at equal current."""
    h2 = hydrogen_flow_L_h(1.0, current_efficiency=0.0, temperature_C=60.0)
    o2 = oxygen_flow_L_h(1.0, oer_fraction=1.0, temperature_C=60.0)
    assert h2 / o2 == pytest.approx(2.0, rel=1e-9)


def test_faradaic_efficiency_scales_hydrogen():
    """Hydrogen tracks (1 - FE): at 100 % FE there is no cathodic gas."""
    assert hydrogen_flow_L_h(3.0, 1.0) == pytest.approx(0.0)
    q85 = hydrogen_flow_L_h(3.0, 0.85)
    q70 = hydrogen_flow_L_h(3.0, 0.70)
    assert q70 / q85 == pytest.approx(0.30 / 0.15, rel=1e-9)


def test_water_vapor_pressure_reference_points():
    """Buck equation hits the textbook anchors."""
    assert water_vapor_pressure_Pa(0.0) == pytest.approx(611.2, rel=0.01)
    assert water_vapor_pressure_Pa(100.0) == pytest.approx(101325.0, rel=0.02)
    assert water_vapor_pressure_Pa(60.0) == pytest.approx(19940.0, rel=0.02)


def test_wet_gas_is_larger_than_dry_and_matches_ratio():
    """Saturation expands the stream by P/(P - p_sat)."""
    dry = gas_volumetric_flow_m3_s(1e-5, 60.0, water_saturated=False)
    wet = gas_volumetric_flow_m3_s(1e-5, 60.0, water_saturated=True)
    p_sat = water_vapor_pressure_Pa(60.0)
    assert wet / dry == pytest.approx(P_ATM_PA / (P_ATM_PA - p_sat), rel=1e-9)
    assert wet > dry


def test_wet_gas_at_temperature_exceeds_dry_at_25C():
    """The design-basis 1.37 L/h dry-at-25C scalar understates cell-condition gas.

    reference_cell_design.hydrogen_rate_L_h reports dry gas at 25 C/1 atm; a
    vent sees hot saturated gas. Both effects push the same way.
    """
    from models.reference_cell_design import hydrogen_rate_L_h as dry_25C

    dry = dry_25C(3.0, 1.0)
    wet_hot = hydrogen_flow_L_h(3.0, current_efficiency=0.0, temperature_C=60.0)
    assert dry == pytest.approx(1.37, abs=0.05)
    assert wet_hot > dry
    # Thermal expansion (333/298) x saturation (101325/81380)
    assert wet_hot / dry == pytest.approx(
        (333.15 / 298.15) * P_ATM_PA / (P_ATM_PA - water_vapor_pressure_Pa(60.0)), rel=0.02
    )


def test_gas_flow_rejects_bad_input():
    with pytest.raises(ValueError):
        faradaic_gas_flow_mol_s(-1.0, Z_H2)
    with pytest.raises(ValueError):
        faradaic_gas_flow_mol_s(1.0, Z_H2, faradaic_fraction=1.5)
    with pytest.raises(ValueError):
        faradaic_gas_flow_mol_s(1.0, 0)


# ─── Bubble mechanics ─────────────────────────────────────────────────

def test_fritz_matches_closed_form():
    d = fritz_detachment_diameter_m(contact_angle_deg=3.0, sigma_N_m=0.070,
                                    rho_liquid=1200.0, rho_gas=0.07)
    expected = 0.0208 * 3.0 * math.sqrt(0.070 / (G_ACCEL * (1200.0 - 0.07)))
    assert d == pytest.approx(expected, rel=1e-12)


def test_electrolytic_bubbles_are_micron_scale():
    """The apparent contact angle must give tens-to-hundreds of µm, not mm."""
    d = fritz_detachment_diameter_m(contact_angle_deg=CONTACT_ANGLE_H2_DEG)
    assert 20e-6 < d < 500e-6


def test_shear_detachment_shrinks_bubbles_with_velocity():
    """Faster crossflow strips smaller bubbles; zero flow is no constraint."""
    assert shear_detachment_diameter_m(0.0) == math.inf
    fast = shear_detachment_diameter_m(1.0)
    slow = shear_detachment_diameter_m(0.1)
    assert fast < slow
    # d_shear scales as u^-2
    assert slow / fast == pytest.approx(100.0, rel=1e-9)


def test_departure_diameter_takes_the_binding_mechanism():
    quiescent = departure_diameter_m(0.0)
    sheared = departure_diameter_m(5.0)
    assert quiescent == pytest.approx(fritz_detachment_diameter_m(CONTACT_ANGLE_H2_DEG))
    assert sheared < quiescent
    assert sheared == pytest.approx(shear_detachment_diameter_m(5.0))


def test_stokes_branch_for_small_bubbles():
    d = 50e-6
    u = terminal_rise_velocity_m_s(d)
    stokes = G_ACCEL * (RHO_LIQUID_KG_M3 - 0.07) * d ** 2 / (18.0 * MU_LIQUID_PA_S)
    assert u == pytest.approx(stokes, rel=1e-9)


def test_harmathy_caps_large_bubbles():
    """Above ~1 mm the rise velocity stops growing with diameter."""
    u_1mm = terminal_rise_velocity_m_s(1e-3)
    u_5mm = terminal_rise_velocity_m_s(5e-3)
    assert u_5mm == pytest.approx(u_1mm, rel=1e-9)
    assert u_5mm < 1.0


def test_rise_velocity_monotonic_in_stokes_regime():
    ds = np.linspace(10e-6, 200e-6, 20)
    us = [terminal_rise_velocity_m_s(float(d)) for d in ds]
    assert all(b > a for a, b in zip(us, us[1:]))


# ─── Drift flux and conductivity ──────────────────────────────────────

def test_void_fraction_zero_without_gas():
    assert drift_flux_void_fraction(0.0, 0.07, 0.02) == 0.0


def test_void_fraction_below_homogeneous_value():
    """Slip and the distribution parameter both hold void below no-slip."""
    jg, jl = 0.01, 0.07
    eps = drift_flux_void_fraction(jg, jl, 0.02)
    homogeneous = jg / (jg + jl)
    assert 0.0 < eps < homogeneous


def test_void_fraction_monotonic_in_gas_flux():
    prev = -1.0
    for jg in (0.0, 0.001, 0.01, 0.05, 0.2):
        eps = drift_flux_void_fraction(jg, 0.07, 0.02)
        assert eps > prev
        prev = eps


def test_faster_liquid_flushes_gas_out():
    slow = drift_flux_void_fraction(0.01, 0.02, 0.02)
    fast = drift_flux_void_fraction(0.01, 0.50, 0.02)
    assert fast < slow


def test_bruggeman_limits_and_exponent():
    assert bruggeman_conductivity(13.5, 0.0) == pytest.approx(13.5)
    assert bruggeman_conductivity(13.5, 0.10) == pytest.approx(
        13.5 * 0.9 ** BRUGGEMAN_EXPONENT
    )
    assert bruggeman_conductivity(13.5, 0.5) < bruggeman_conductivity(13.5, 0.1)


def test_surface_coverage_saturates_and_starts_at_zero():
    assert surface_coverage_fraction(0.0) == pytest.approx(0.0)
    assert surface_coverage_fraction(1e5) == pytest.approx(THETA_MAX_H2, rel=1e-6)
    assert 0.0 < surface_coverage_fraction(150.0) < THETA_MAX_H2


def test_surface_coverage_monotonic():
    vals = [surface_coverage_fraction(j) for j in (10, 50, 100, 300, 600)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


# ─── Mass transfer ────────────────────────────────────────────────────

def test_no_gas_means_no_bubble_mass_transfer():
    assert vogt_mass_transfer_coefficient_m_s(0.0, 150e-6, 7.2e-10, 5.8e-7) == 0.0


def test_bubble_mass_transfer_grows_with_gas_flux():
    k1 = vogt_mass_transfer_coefficient_m_s(1e-4, 150e-6, 7.2e-10, 5.8e-7)
    k2 = vogt_mass_transfer_coefficient_m_s(1e-3, 150e-6, 7.2e-10, 5.8e-7)
    assert 0.0 < k1 < k2
    # Sh ~ Re^0.5, so a 10x flux gives sqrt(10)
    assert k2 / k1 == pytest.approx(math.sqrt(10.0), rel=1e-6)


def test_bubbles_only_thin_the_boundary_layer():
    delta0 = 50e-6
    D = 7.2e-10
    assert combined_boundary_layer_m(delta0, 0.0, D) == pytest.approx(delta0)
    thinned = combined_boundary_layer_m(delta0, 1e-5, D)
    assert 0.0 < thinned < delta0


def test_boundary_layer_quadrature_superposition():
    delta0, D, kb = 50e-6, 7.2e-10, 2e-5
    expected = D / math.hypot(D / delta0, kb)
    assert combined_boundary_layer_m(delta0, kb, D) == pytest.approx(expected, rel=1e-12)


# ─── Geometry ─────────────────────────────────────────────────────────

def test_rc1_geometry_matches_the_yaml_configuration():
    """Defaults must track processes/reference_cell_rc1.yaml."""
    g = ChannelGeometry()
    assert g.electrode_area_cm2 == pytest.approx(10.0)
    assert g.height_m == pytest.approx(0.050)
    assert g.width_m == pytest.approx(0.020)
    assert g.depth_m == pytest.approx(0.003)


def test_superficial_velocity_from_flow_and_cross_section():
    g = ChannelGeometry(liquid_flow_L_min=0.25)
    expected = (0.25 / 1000.0 / 60.0) / (0.020 * 0.003)
    assert g.superficial_liquid_velocity_m_s == pytest.approx(expected)


def test_geometry_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError):
        ChannelGeometry(height_m=0.0)
    with pytest.raises(ValueError):
        ChannelGeometry(depth_m=-1.0)


# ─── Axial profile ────────────────────────────────────────────────────

def test_gas_accumulates_monotonically_upward():
    prof = holdup_profile(300.0, 0.15, n_segments=16)
    eps = prof.void_fraction
    assert all(b >= a for a, b in zip(eps, eps[1:]))
    assert eps[0] < eps[-1]


def test_conductivity_falls_where_gas_accumulates():
    prof = holdup_profile(300.0, 0.15, n_segments=16)
    k = prof.kappa_eff_S_m
    assert all(b <= a for a, b in zip(k, k[1:]))
    assert prof.conductivity_penalty > 1.0


def test_perfect_efficiency_makes_no_cathodic_gas():
    prof = holdup_profile(300.0, 0.0, n_segments=8)
    assert np.allclose(prof.void_fraction, 0.0)
    assert np.allclose(prof.surface_coverage, 0.0)
    assert prof.conductivity_penalty == pytest.approx(1.0)
    assert np.allclose(prof.delta_eff_m, 50e-6)


def test_holdup_grows_with_current_density():
    lo = holdup_profile(100.0, 0.15, n_segments=8).outlet_void_fraction
    hi = holdup_profile(400.0, 0.15, n_segments=8).outlet_void_fraction
    assert hi > lo


def test_holdup_accepts_per_segment_arrays():
    j = [100.0, 200.0, 300.0, 400.0]
    prof = holdup_profile(j, [0.15] * 4, n_segments=4)
    assert np.allclose(prof.j_mA_cm2, j)
    assert prof.void_fraction[-1] > prof.void_fraction[0]


def test_holdup_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        holdup_profile([100.0, 200.0], 0.15, n_segments=4)
    with pytest.raises(ValueError):
        holdup_profile(100.0, 1.4, n_segments=4)


def test_taller_channel_accumulates_more_gas():
    short = holdup_profile(300.0, 0.15, geometry=ChannelGeometry(height_m=0.05),
                           n_segments=12).outlet_void_fraction
    tall = holdup_profile(300.0, 0.15, geometry=ChannelGeometry(height_m=0.50),
                          n_segments=12).outlet_void_fraction
    assert tall > short


def test_profile_serialises_to_json():
    payload = holdup_profile(300.0, 0.15, n_segments=6).to_dict()
    json.dumps(payload)
    assert payload["outlet_void_fraction"] > 0.0
    assert len(payload["y_mm"]) == 6


# ─── Current redistribution ───────────────────────────────────────────

def test_uniform_conductivity_gives_uniform_current():
    j = solve_current_distribution(300.0, [13.5] * 8)
    assert np.allclose(j, 300.0, rtol=1e-6)


def test_current_avoids_the_resistive_segments():
    """Lower local conductivity must draw less current."""
    kappa = [13.5, 13.0, 12.0, 10.0]
    j = solve_current_distribution(300.0, kappa)
    assert all(b < a for a, b in zip(j, j[1:]))


def test_redistribution_conserves_mean_current():
    j = solve_current_distribution(250.0, [13.5, 12.0, 10.0, 8.0])
    assert float(np.mean(j)) == pytest.approx(250.0, rel=1e-9)


def test_bubble_coverage_pushes_current_away():
    """Blanketed area is less attractive even at equal conductivity."""
    kappa = [13.5] * 4
    j = solve_current_distribution(300.0, kappa, surface_coverage=[0.0, 0.1, 0.2, 0.3])
    assert all(b < a for a, b in zip(j, j[1:]))
    assert float(np.mean(j)) == pytest.approx(300.0, rel=1e-9)


def test_redistribution_rejects_bad_input():
    with pytest.raises(ValueError):
        solve_current_distribution(300.0, [])
    with pytest.raises(ValueError):
        solve_current_distribution(300.0, [13.5, -1.0])
    with pytest.raises(ValueError):
        solve_current_distribution(-5.0, [13.5])
    with pytest.raises(ValueError):
        solve_current_distribution(300.0, [13.5, 13.5], surface_coverage=[0.1])


# ─── Coupled solve ────────────────────────────────────────────────────

def _surrogate_fe(j_mA_cm2, delta_m, temperature_C, fe_conc_M, pH_bulk):
    """Cheap monotone stand-in for the 1-D diffusion-layer engine.

    FE falls with current density and rises as the diffusion layer thins,
    which is the qualitative behaviour the real engine has. Keeps the
    coupling tests fast and independent of the FE engine's calibration.
    """
    transport_limit = 1000.0 * (50e-6 / delta_m)
    return float(max(0.05, min(0.99, 1.0 - 0.5 * (j_mA_cm2 / transport_limit))))


def test_coupled_solve_converges_and_conserves_current():
    res = solve_coupled(300.0, n_segments=4, fe_model=_surrogate_fe, max_iterations=25)
    assert res.converged
    assert float(np.mean(res.profile.j_mA_cm2)) == pytest.approx(300.0, rel=1e-6)
    assert 0.0 < res.area_average_FE < 1.0


def test_coupled_solve_needs_more_than_one_pass_to_claim_convergence():
    """Convergence must be a fixed-point test, not distance from the guess."""
    res = solve_coupled(300.0, n_segments=3, fe_model=_surrogate_fe, max_iterations=1)
    assert not res.converged
    assert res.iterations == 1


def test_coupled_reports_baseline_and_shift():
    res = solve_coupled(300.0, n_segments=4, fe_model=_surrogate_fe, max_iterations=25)
    baseline = _surrogate_fe(300.0, 50e-6, 60.0, 1.0, 2.0)
    assert res.FE_no_bubbles == pytest.approx(baseline)
    assert res.FE_shift == pytest.approx((res.area_average_FE - baseline) * 100.0)


def test_bubble_microconvection_raises_fe_against_the_baseline():
    """With a transport-limited FE law, self-stirring is a net FE gain."""
    res = solve_coupled(400.0, n_segments=4, fe_model=_surrogate_fe, max_iterations=25)
    assert res.area_average_FE > res.FE_no_bubbles


def test_coupled_ohmic_penalty_is_positive_and_gas_free_reference_correct():
    res = solve_coupled(300.0, n_segments=4, fe_model=_surrogate_fe, max_iterations=25)
    geom = ChannelGeometry()
    expected_free = 300.0 * 10.0 * geom.interelectrode_gap_m / 13.5
    assert res.ohmic_gas_free_V == pytest.approx(expected_free, rel=1e-9)
    assert res.ohmic_penalty_V > 0.0


def test_coupled_hydrogen_flow_matches_the_standalone_calculation():
    res = solve_coupled(300.0, n_segments=4, fe_model=_surrogate_fe, max_iterations=25)
    geom = ChannelGeometry()
    current_A = 300.0 * 10.0 * geom.electrode_area_m2
    assert res.hydrogen_flow_L_h == pytest.approx(
        hydrogen_flow_L_h(current_A, res.area_average_FE, 60.0), rel=1e-9
    )


def test_coupled_serialises_to_json():
    res = solve_coupled(300.0, n_segments=3, fe_model=_surrogate_fe, max_iterations=25)
    json.dumps(res.to_dict())


def test_coupled_rejects_bad_relaxation():
    with pytest.raises(ValueError):
        solve_coupled(300.0, relaxation=0.0, fe_model=_surrogate_fe)
    with pytest.raises(ValueError):
        solve_coupled(300.0, relaxation=1.5, fe_model=_surrogate_fe)


@pytest.mark.slow
def test_coupled_solve_with_the_real_fe_engine():
    """End-to-end against diffusion_layer_1d — the gating FE model."""
    res = solve_coupled(300.0, n_segments=3, max_iterations=6)
    assert 0.5 < res.area_average_FE < 1.0
    assert res.profile.outlet_void_fraction > 0.0


# ─── Hydrogen safety ──────────────────────────────────────────────────

def test_undiluted_headspace_is_pure_hydrogen():
    r = hydrogen_safety(3.0, 0.85, dilution_flow_L_h=0.0)
    assert r.hydrogen_vol_fraction == pytest.approx(1.0)
    assert not r.acceptable


def test_required_dilution_achieves_the_target():
    r = hydrogen_safety(3.0, 0.85, dilution_flow_L_h=0.0, target_fraction_of_LFL=0.25)
    check = hydrogen_safety(3.0, 0.85, dilution_flow_L_h=r.required_dilution_flow_L_h,
                            target_fraction_of_LFL=0.25)
    assert check.hydrogen_vol_fraction == pytest.approx(0.25 * LFL_H2_VOL_FRAC, rel=1e-6)
    assert check.acceptable
    assert check.fraction_of_LFL == pytest.approx(0.25, rel=1e-6)


def test_flammability_band_detected():
    """A mixture between 4 % and 75 % H2 must be flagged flammable."""
    q = hydrogen_flow_L_h(3.0, 0.85)
    # Dilute to ~10 vol %: inside the band.
    r = hydrogen_safety(3.0, 0.85, dilution_flow_L_h=q * 9.0)
    assert r.hydrogen_vol_fraction == pytest.approx(0.10, rel=1e-6)
    assert LFL_H2_VOL_FRAC < r.hydrogen_vol_fraction < UFL_H2_VOL_FRAC
    assert r.flammable
    assert not r.acceptable


def test_below_lfl_is_not_flammable():
    r = hydrogen_safety(3.0, 0.85, dilution_flow_L_h=10000.0)
    assert not r.flammable
    assert r.acceptable


def test_time_to_lfl_in_sealed_headspace():
    r = hydrogen_safety(3.0, 0.0, headspace_L=0.5)
    expected = (LFL_H2_VOL_FRAC * 0.5) / r.hydrogen_flow_L_h * 60.0
    assert r.time_to_LFL_min == pytest.approx(expected)
    # A worst-case bench cell reaches the LFL in minutes, not hours.
    assert r.time_to_LFL_min < 5.0


def test_higher_fe_needs_less_dilution():
    lo = hydrogen_safety(3.0, 0.70).required_dilution_flow_L_h
    hi = hydrogen_safety(3.0, 0.95).required_dilution_flow_L_h
    assert hi < lo


def test_no_gas_no_hazard():
    r = hydrogen_safety(3.0, 1.0, headspace_L=0.5)
    assert r.hydrogen_flow_L_h == pytest.approx(0.0)
    assert r.required_dilution_flow_L_h == pytest.approx(0.0)
    assert not r.flammable
    assert r.time_to_LFL_min is None


def test_safety_rejects_bad_input():
    with pytest.raises(ValueError):
        hydrogen_safety(3.0, 0.85, target_fraction_of_LFL=0.0)
    with pytest.raises(ValueError):
        hydrogen_safety(3.0, 0.85, dilution_flow_L_h=-1.0)
    with pytest.raises(ValueError):
        hydrogen_safety(3.0, 0.85, headspace_L=0.0)


def test_safety_serialises_to_json():
    json.dumps(hydrogen_safety(3.0, 0.85, headspace_L=0.5).to_dict())


# ─── Sweeps ───────────────────────────────────────────────────────────

def test_current_density_sweep_is_monotone_in_holdup():
    rows = current_density_sweep()
    voids = [r["outlet_void_fraction"] for r in rows]
    unis = [r["current_uniformity"] for r in rows]
    assert all(b > a for a, b in zip(voids, voids[1:]))
    assert all(b < a for a, b in zip(unis, unis[1:]))


def test_rc1_is_not_holdup_limited_at_the_kill_criterion():
    """The headline result: bench scale is safe from hold-up effects."""
    rows = current_density_sweep(j_values_mA_cm2=(300.0,))
    r = rows[0]
    assert r["outlet_void_fraction"] < 0.05
    assert r["conductivity_penalty"] < 1.05
    assert r["current_uniformity"] > 0.95


def test_height_screen_finds_a_scale_up_limit():
    """Uniformity must degrade with height and cross the floor somewhere."""
    rows = height_scaling_screen()
    unis = [r["current_uniformity"] for r in rows]
    assert all(b < a for a, b in zip(unis, unis[1:]))
    assert rows[0]["passes_uniformity_floor"]
    assert not rows[-1]["passes_uniformity_floor"]


def test_height_screen_reports_area_scale():
    rows = height_scaling_screen(heights_m=(0.05, 0.10))
    assert rows[0]["area_scale_vs_RC1"] == pytest.approx(1.0)
    assert rows[1]["area_scale_vs_RC1"] == pytest.approx(2.0)


# ─── Contracts ────────────────────────────────────────────────────────

def test_model_scope_is_honest_about_level_zero():
    scope = model_scope()
    assert scope["level"] == 0
    assert "no gas hold-up" in scope["provenance"].lower()
    assert scope["computes"] and scope["does_not_compute"]
    assert scope["dominant_uncertainty"]
    assert scope["replaced_by"].startswith("measurement_protocol")
    json.dumps(scope)


def test_model_scope_names_the_gaps_it_closes():
    gaps = " ".join(model_scope()["gap_closed"])
    assert "NEXT_STEPS" in gaps
    assert "REFERENCE_CELL_DESIGN_BASIS" in gaps
    assert "SIM_THEORY_CONFIDENCE" in gaps


def test_measurement_protocol_is_actionable():
    p = measurement_protocol()
    assert p["estimated_cost_usd"] < 2000
    assert len(p["measurements"]) >= 4
    for m in p["measurements"]:
        assert m["method"] and m["calibrates"] and m["resolution_required"]
    assert set(p["decision_rules"]) == {"confirm", "recalibrate", "escalate"}
    json.dumps(p)


def test_protocol_escalation_rule_matches_the_design_basis():
    """The CFD trigger must point at the design-basis clause it implements."""
    assert "CFD" in measurement_protocol()["decision_rules"]["escalate"]
