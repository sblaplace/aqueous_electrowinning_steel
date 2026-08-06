"""Tests for the deposit internal stress (Stoney / bent-strip) model.

Covers:
  - Stoney forward/inverse and bent-strip cantilever deflection round trips
  - Exact two-layer laminate curvature and finite-thickness correction
  - GUM standard-uncertainty budget and resolution limits
  - Stress decomposition from plating conditions (intrinsic, H, thermal)
  - Empirical additive corrections (saccharin relief, chloride bath shift)
  - Stress evolution through deposit thickness (local vs average Stoney)
  - End-to-end integration with the adhesion and peel module
  - Coupon curvature experimental protocol and model_scope contract
"""

import json
import math

import numpy as np
import pytest

from models.internal_stress import (
    CHLORIDE_SHIFT_MPa,
    COUPON_E_GPA,
    COUPON_NU,
    E_FE_GPA,
    HOFFMAN_DELTA_M,
    NU_FE,
    _two_layer_curvature_per_stress,
    bent_strip_stress_MPa,
    cantilever_deflection_m,
    coupon_curvature_protocol,
    curvature_from_stress_MPa,
    deflection_resolution_for_stress_um,
    deposit_stress_from_conditions,
    equivalent_grain_and_hydrogen,
    finite_thickness_correction,
    model_scope,
    peel_verdict_from_conditions,
    stoney_stress_finite_thickness_MPa,
    stoney_stress_MPa,
    stress_evolution,
    stress_profile,
    stress_resolution_MPa,
    stress_uncertainty_MPa,
)


class TestMeasurementTheory:
    def test_stoney_forward_inverse_round_trip(self):
        sigma = 150.0
        h_f = 25e-6
        h_s = 0.4e-3
        kappa = curvature_from_stress_MPa(
            sigma, h_f, h_s, substrate_E_GPa=COUPON_E_GPA, substrate_nu=COUPON_NU
        )
        sigma_rt = stoney_stress_MPa(
            kappa, h_s, h_f, substrate_E_GPa=COUPON_E_GPA, substrate_nu=COUPON_NU
        )
        assert sigma_rt == pytest.approx(sigma)

    def test_bent_strip_forward_inverse_round_trip(self):
        sigma = 200.0
        h_f = 30e-6
        h_s = 0.4e-3
        deflection = cantilever_deflection_m(
            sigma, film_thickness_m=h_f, substrate_thickness_m=h_s
        )
        sigma_rt = bent_strip_stress_MPa(
            deflection, film_thickness_m=h_f, substrate_thickness_m=h_s
        )
        assert sigma_rt == pytest.approx(sigma)

    def test_stoney_is_linear_in_stress(self):
        k100 = curvature_from_stress_MPa(100.0, 25e-6, 0.4e-3)
        k200 = curvature_from_stress_MPa(200.0, 25e-6, 0.4e-3)
        assert k200 == pytest.approx(2.0 * k100)

    def test_deflection_scales_with_gauge_length_squared(self):
        d60 = cantilever_deflection_m(100.0, gauge_length_m=0.060, film_thickness_m=25e-6)
        d120 = cantilever_deflection_m(100.0, gauge_length_m=0.120, film_thickness_m=25e-6)
        assert d120 == pytest.approx(4.0 * d60)

    def test_rejects_nonpositive_geometry(self):
        with pytest.raises(ValueError):
            curvature_from_stress_MPa(100.0, -1e-6, 0.4e-3)
        with pytest.raises(ValueError):
            cantilever_deflection_m(100.0, gauge_length_m=-0.06)
        with pytest.raises(ValueError):
            bent_strip_stress_MPa(1e-5, film_thickness_m=0.0)


class TestFiniteThicknessCorrection:
    def test_correction_approaches_one_for_thin_film(self):
        c = finite_thickness_correction(
            film_thickness_m=1e-6, substrate_thickness_m=0.4e-3
        )
        assert 1.0 <= c < 1.01

    def test_correction_is_monotonic(self):
        ratios = [0.001, 0.05, 0.1, 0.25]
        corrs = [
            finite_thickness_correction(r * 0.4e-3, 0.4e-3) for r in ratios
        ]
        assert corrs == sorted(corrs)

    def test_laminate_curvature_matches_thin_film_in_limit(self):
        k_lam = _two_layer_curvature_per_stress(100.0, 1e-6, 0.4e-3)
        k_thin = curvature_from_stress_MPa(100.0, 1e-6, 0.4e-3)
        assert k_lam == pytest.approx(k_thin, rel=0.015)

    def test_stoney_finite_thickness_stress_is_higher_than_thin_film(self):
        h_f = 40e-6
        h_s = 0.4e-3
        k = curvature_from_stress_MPa(100.0, h_f, h_s)
        s_thin = stoney_stress_MPa(k, h_s, h_f)
        s_fin = stoney_stress_finite_thickness_MPa(k, h_s, h_f)
        assert s_fin > s_thin

    def test_rejects_nonpositive_thicknesses(self):
        with pytest.raises(ValueError):
            finite_thickness_correction(0.0, 0.4e-3)
        with pytest.raises(ValueError):
            finite_thickness_correction(10e-6, -0.1e-3)


class TestUncertaintyBudget:
    def test_uncertainty_budget_math(self):
        d = cantilever_deflection_m(250.0, film_thickness_m=25e-6, substrate_thickness_m=0.4e-3)
        unc = stress_uncertainty_MPa(
            d,
            u_deflection_m=10e-6,
            film_thickness_m=25e-6,
            substrate_thickness_m=0.4e-3,
        )
        assert unc["sigma_MPa"] == pytest.approx(250.0)
        expected_rel = math.sqrt(sum((c / 250.0) ** 2 for c in unc["contributions_MPa"].values()))
        assert unc["relative_uncertainty"] == pytest.approx(expected_rel)

    def test_dominant_uncertainty_is_identified(self):
        d = cantilever_deflection_m(20.0, film_thickness_m=25e-6, substrate_thickness_m=0.4e-3)
        unc = stress_uncertainty_MPa(
            d,
            u_deflection_m=15e-6,
            film_thickness_m=25e-6,
            substrate_thickness_m=0.4e-3,
        )
        assert unc["dominant_uncertainty"] == "deflection"

    def test_stress_resolution_and_deflection_resolution(self):
        res = stress_resolution_MPa(deflection_resolution_m=10e-6, gauge_length_m=0.060, film_thickness_m=25e-6)
        res120 = stress_resolution_MPa(deflection_resolution_m=10e-6, gauge_length_m=0.120, film_thickness_m=25e-6)
        assert res120 == pytest.approx(res / 4.0)
        def_res = deflection_resolution_for_stress_um(target_stress_resolution_MPa=10.0, gauge_length_m=0.060, film_thickness_m=25e-6)
        assert def_res > 0.0

    def test_rejects_zero_deflection(self):
        with pytest.raises(ValueError):
            stress_uncertainty_MPa(0.0, 10e-6)


class TestDepositStressFromConditions:
    def test_reference_story_reproduction(self):
        res = deposit_stress_from_conditions(
            j_mA_cm2=100.0,
            current_efficiency_percent=85.0,
            deposition_time_s=900.0,
        )
        comp = res["components"]
        # Hydrogen is the dominant term (~370 of ~415 MPa) at ~240 ppm —
        # the same screening story as before, now produced by the IPZ
        # surface-kinetic H-entry model rather than the empirical 5% factor.
        assert comp["total_MPa"] == pytest.approx(414.0, abs=3.0)
        assert comp["hydrogen_MPa"] == pytest.approx(372.0, abs=3.0)
        assert res["dominant_mechanism"] == "hydrogen"
        assert res["derived"]["thickness_um"] == pytest.approx(28.0, abs=1.0)
        assert res["derived"]["C_H_diffusible_ppm"] == pytest.approx(240.0, abs=5.0)

    def test_saccharin_relieves_intrinsic_stress(self):
        res_0 = deposit_stress_from_conditions(saccharin_g_L=0.0)
        res_sac = deposit_stress_from_conditions(saccharin_g_L=1.5)
        assert res_sac["components"]["intrinsic_MPa"] < res_0["components"]["intrinsic_MPa"]

    def test_chloride_bath_shifts_stress_compressive(self):
        res_sulf = deposit_stress_from_conditions(chloride_bath=False)
        res_chlor = deposit_stress_from_conditions(chloride_bath=True)
        diff = res_sulf["components"]["total_MPa"] - res_chlor["components"]["total_MPa"]
        assert diff == pytest.approx(CHLORIDE_SHIFT_MPa)

    def test_pulse_waveform_reduces_stress(self):
        res_dc = deposit_stress_from_conditions(j_mA_cm2=100.0, current_efficiency_percent=85.0)
        res_pre = deposit_stress_from_conditions(
            waveform="pre",
            duty_cycle=0.3,
            j_peak_mA_cm2=333.3,
            current_efficiency_percent=95.0,
        )
        assert res_pre["components"]["total_MPa"] < res_dc["components"]["total_MPa"]


class TestStressEvolution:
    def test_evolution_local_and_average_at_zero(self):
        ev0 = stress_evolution(
            plateau_stress_MPa=400.0,
            thickness_um=1e-6,
            interface_stress_MPa=-40.0,
            characteristic_thickness_um=10.0,
        )
        assert ev0["local_MPa"] == pytest.approx(-40.0, abs=1.0)
        assert ev0["average_MPa"] == pytest.approx(-40.0, abs=1.0)

    def test_evolution_local_approaches_plateau(self):
        ev = stress_evolution(
            plateau_stress_MPa=400.0,
            thickness_um=100.0,
            interface_stress_MPa=-40.0,
            characteristic_thickness_um=10.0,
        )
        assert ev["local_MPa"] == pytest.approx(400.0, abs=0.1)
        assert ev["average_MPa"] < 400.0

    def test_stress_profile_returns_arrays(self):
        prof = stress_profile(400.0, 50.0, n_points=50)
        assert len(prof["h_um"]) == 50
        assert len(prof["local_MPa"]) == 50
        assert len(prof["average_MPa"]) == 50
        assert np.all(prof["h_um"] > 0)


class TestPeelVerdictFromConditions:
    def test_dc_baseline_spontaneous_delamination(self):
        pv = peel_verdict_from_conditions(
            j_mA_cm2=100.0, current_efficiency_percent=85.0, deposition_time_s=900.0
        )
        assert pv["peel"]["outcome"] == "spontaneous_delamination"
        assert pv["verdict"] == "spontaneous_delamination"

    def test_equivalent_grain_and_hydrogen(self):
        d_eff, ch_eff = equivalent_grain_and_hydrogen(10.0, 200.0)
        assert d_eff > 0.0
        assert ch_eff > 0.0

    def test_to_dict_is_json_safe(self):
        pv = peel_verdict_from_conditions()
        json_str = json.dumps(pv)
        assert len(json_str) > 50


class TestCouponProtocol:
    def test_protocol_structure_and_cost(self):
        cp = coupon_curvature_protocol()
        assert "title" in cp
        assert cp["budget_usd"]["total"] == pytest.approx(200.0)
        assert len(cp["coupons"]) == 3
        assert len(cp["decision_rules"]) >= 3

    def test_protocol_is_json_safe(self):
        cp = coupon_curvature_protocol()
        json_str = json.dumps(cp)
        assert "Bent-strip" in json_str


class TestModelScope:
    def test_lists_what_it_does_not_compute(self):
        scope = model_scope()
        assert len(scope["does_not_compute"]) >= 5

    def test_limitations_declare_no_repository_data(self):
        scope = model_scope()
        assert "no iron internal-stress data exists in this repository" in scope["limitations"].lower()

    def test_names_key_uncertainty_and_calibration(self):
        scope = model_scope()
        assert len(scope["key_uncertainty"]) >= 1
        assert len(scope["calibration_required"]) >= 4


class TestCrossModelConsistency:
    def test_iron_constants_match_adhesion_peel(self):
        from models import adhesion_peel as ap
        assert E_FE_GPA == pytest.approx(ap.E_FE_GPA)
        assert NU_FE == pytest.approx(ap.NU_FE)
        assert HOFFMAN_DELTA_M == pytest.approx(ap.HOFFMAN_DELTA_M)

    def test_residual_stress_model_scope_complementary(self):
        scope = model_scope()
        computes = " ".join(scope["computes"]).lower()
        assert "stoney" in computes or "stress" in computes
