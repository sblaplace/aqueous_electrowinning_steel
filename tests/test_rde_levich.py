"""Tests for the RDE kinetics/transport separation model (Levich + Koutecky-Levich).

Covers:
  - Levich limiting current scaling with omega^(1/2)
  - Diffusivity recovery from a Levich slope (round trip)
  - Nernst film thickness decreasing with rotation rate
  - Koutecky-Levich kinetic-current correction
  - End-to-end recovery of D, delta, Fe and HER Tafel kinetics from a synthetic
    RDE polarization set
  - Automatic plateau / kinetic-window / HER-window recommendation
  - Experimental-design matrix and gate rules
  - model_scope contract
"""

import math

import numpy as np
import pytest

from models.rde_levich import (
    KIN_VISC_WATER_25,
    RDEBranch,
    analyze_rde_polarization,
    diffusivity_Arrhenius,
    diffusivity_from_levich_B,
    kinematic_viscosity_water_m2_s,
    koutecky_levich_kinetic,
    levich_constant_B,
    levich_limiting_current,
    model_scope,
    nernst_layer_thickness_m,
    recommend_windows_from_polarization,
    rde_experiment_design,
    rpm_to_rad_per_s,
    simulate_rde_polarization,
)

D_REF = 7.2e-10
C_REF = 1.0
NU_REF = KIN_VISC_WATER_25


def _default_fe():
    return RDEBranch(i0_A_m2=500.0, tafel_V_decade=0.120, E_eq_V=-0.440)


def _default_her():
    return RDEBranch(i0_A_m2=0.02, tafel_V_decade=0.150, E_eq_V=-0.1184)


def _synthetic(her_i0=0.02):
    fe = _default_fe()
    her = _default_her()
    if her_i0 != 0.02:
        her = RDEBranch(her_i0, 0.150, -0.1184)
    E = np.linspace(-0.50, -1.05, 96)
    omegas = np.array([400.0, 900.0, 1600.0, 2500.0])
    return simulate_rde_polarization(
        E, omegas, fe=fe, her=her, D_m2_s=D_REF, C_fe_M=C_REF, nu_m2_s=NU_REF
    )


class TestLevichTransport:
    def test_limiting_current_scales_sqrt_omega(self):
        omega = np.array([400.0, 900.0, 1600.0, 2500.0])
        i_lim = levich_limiting_current(omega, z=2, D_m2_s=D_REF, C_bulk_M=C_REF, nu_m2_s=NU_REF)
        ratio = i_lim / np.sqrt(rpm_to_rad_per_s(omega))
        # i_lim / sqrt(omega) should be the constant B
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
        assert np.all(np.diff(i_lim) > 0)  # faster rotation -> higher limit

    def test_levich_constant_equals_limiting_over_sqrt_omega(self):
        omega = 1600.0
        B = levich_constant_B(z=2, D_m2_s=D_REF, C_bulk_M=C_REF, nu_m2_s=NU_REF)
        i_lim = levich_limiting_current(omega, z=2, D_m2_s=D_REF, C_bulk_M=C_REF, nu_m2_s=NU_REF)
        assert i_lim == pytest.approx(B * np.sqrt(rpm_to_rad_per_s(omega)), rel=1e-12)

    def test_diffusivity_round_trip_from_B(self):
        B = levich_constant_B(z=2, D_m2_s=D_REF, C_bulk_M=C_REF, nu_m2_s=NU_REF)
        D_back = diffusivity_from_levich_B(B, z=2, C_bulk_M=C_REF, nu_m2_s=NU_REF)
        assert D_back == pytest.approx(D_REF, rel=1e-9)

    def test_nernst_layer_thickness_decreases_with_omega(self):
        omega = np.array([400.0, 2500.0])
        delta = nernst_layer_thickness_m(omega, D_m2_s=D_REF, nu_m2_s=NU_REF)
        assert delta[0] > delta[1]
        assert np.all(delta > 0)

    def test_temperature_correlations_sensible(self):
        nu25 = kinematic_viscosity_water_m2_s(25.0)
        nu60 = kinematic_viscosity_water_m2_s(60.0)
        assert 8e-7 < nu25 < 1.0e-6   # ~8.9e-7 m^2/s at 25 C
        assert nu60 < nu25            # viscosity drops with temperature
        D60 = diffusivity_Arrhenius(D_REF, 60.0)
        assert D60 > D_REF           # diffusivity rises with temperature


class TestKouteckyLevich:
    def test_kinetic_current_recovery(self):
        # If i is exactly i_lim, the kinetic current is infinite (no valid K-L)
        i_k = koutecky_levich_kinetic(np.array([100.0]), np.array([200.0]))
        assert i_k[0] == pytest.approx(200.0, rel=1e-9)  # 1/ik = 1/100 - 1/200

    def test_kinetic_current_nan_at_limit(self):
        i_k = koutecky_levich_kinetic(np.array([200.0]), np.array([200.0]))
        assert math.isnan(i_k[0])

    def test_kl_composition_inverse_of_parallel(self):
        # 1/i = 1/i_k + 1/i_lim (series resistance analogue)
        i_k = 300.0
        i_lim = 600.0
        i_total = 1.0 / (1.0 / i_k + 1.0 / i_lim)
        i_k_back = koutecky_levich_kinetic(np.array([i_total]), np.array([i_lim]))
        assert i_k_back[0] == pytest.approx(i_k, rel=1e-6)


class TestFullAnalysis:
    def test_end_to_end_recovers_D_and_fe_kinetics(self):
        data = _synthetic()
        res = analyze_rde_polarization(
            data["frame"], C_fe_M=C_REF, pH=2.0, D_ref_m2_s=D_REF
        )
        chk = res["recovery_checks"]
        assert abs(chk["relative_error_pct"]) < 5.0
        # Nernst film thickness at 1600 rpm should be in the tens-of-microns range
        assert 5e-6 < res["nernst_layer_m_at_1600rpm"] < 30e-6
        # Fe kinetics
        assert res["fe_tafel"]["tafel_slope_V_decade"] == pytest.approx(0.120, rel=0.05)
        assert res["fe_tafel"]["i0_A_m2"] == pytest.approx(500.0, rel=0.15)
        # HER kinetics (slope recovers well; i0 has more transport-subtraction bias)
        assert res["her_tafel"]["tafel_slope_V_decade"] == pytest.approx(0.150, rel=0.15)
        assert res["her_tafel"]["i0_A_m2"] < 4 * 0.02

    def test_recommend_windows_places_plateau_in_fe_limited_region(self):
        data = _synthetic()
        rec = recommend_windows_from_polarization(data["frame"])
        assert rec["plateau_E_V"] is not None
        # The kinetic window sits at potentials ABOVE (less cathodic than) the
        # plateau, and the HER window BELOW (more cathodic than) it.
        assert rec["kinetic_window_V"][1] >= rec["kinetic_window_V"][0]
        assert rec["kinetic_window_V"][0] > rec["plateau_E_V"]
        assert rec["her_window_V"][0] < rec["her_window_V"][1]
        assert rec["her_window_V"][1] < rec["plateau_E_V"]
        assert rec["her_window_V"][0] < -0.9

    def test_her_absent_gives_no_her_window(self):
        # With a vanishingly small HER exchange current, no HER Tafel is returned
        data = _synthetic(her_i0=1e-9)
        res = analyze_rde_polarization(data["frame"], C_fe_M=C_REF, pH=2.0)
        assert math.isnan(res["her_tafel"]["tafel_slope_V_decade"])


class TestDesignAndScope:
    def test_experiment_design_matrix(self):
        design = rde_experiment_design()
        omegas = design["rotation_matrix_rpm"]
        assert len(omegas) == 6
        assert omegas[0] < omegas[-1]
        assert design["i_lim_spread_ratio"] > 1.5
        assert len(design["gates"]) >= 3

    def test_model_scope_contract(self):
        scope = model_scope()
        assert "computes" in scope
        assert "does_not_compute" in scope
        assert "calibration_required" in scope
        assert "limitations" in scope
        assert any("Levich" in c for c in scope["computes"])
