"""Tests for the Q3 RDE + volumetric-H2 Fe/HER separation module.

Covers:
  - HER equilibrium potential helper
  - Fe-free bath HER fit recovers b_her and i0_her (the #34 first measurement)
  - Fe-bath constrained fit with HER HELD FIXED recovers i0_fe and b_fe
    (non-degenerate Fe/HER separation)
  - volumetric H2 ideal-gas conversion and charge-ledger closure
  - end-to-end L0 self-test (all PASS)
  - measurement_spec / model_scope contracts
"""

import math

import numpy as np
import pytest

from models.rde_volumetric_h2 import (
    GAS_CORR,
    fit_fe_given_her_on_rde,
    fit_fe_kinetics_given_her,
    fit_her_from_free_bath,
    h2_moles_from_volume,
    h2_volume_from_moles,
    her_equilibrium_potential,
    measurement_spec,
    model_scope,
    self_test,
    simulate_fe_bath_rde_polarization,
    simulate_her_free_bath_polarization,
    volumetric_h2_closure,
)
from models.electrochemistry import FARADAY, M_FE, Z_FE

B_HER = 0.140
I0_HER = 2.0e-3
B_FE = 0.120
I0_FE = 50.0
PH = 2.0
T_C = 25.0
EQ_HER = her_equilibrium_potential(PH, T_C)


class TestHERFreeBath:
    def test_her_equilibrium_potential_matches_pourbaix(self):
        from models.pourbaix import her_line
        assert EQ_HER == pytest.approx(her_line(PH, T_C + 273.15), rel=1e-12)

    def test_fit_recovers_her_branch(self):
        E = np.linspace(EQ_HER - 0.80, EQ_HER - 0.05, 60)
        free = simulate_her_free_bath_polarization(
            E, i0_her_A_m2=I0_HER, b_her_V_dec=B_HER, E_eq_her_V=EQ_HER)
        fit = fit_her_from_free_bath(free["potentials_V"], free["i_her_A_m2"],
                                     E_eq_her_V=EQ_HER)
        assert fit["converged"]
        assert fit["b_her_V_dec"] == pytest.approx(B_HER, rel=0.05)
        # exchange current on a log scale
        assert abs(math.log10(fit["i0_her_A_m2"]) - math.log10(I0_HER)) < 0.3
        assert fit["r_squared"] > 0.99
        assert fit["n_points"] >= 8

    def test_fit_recovers_her_under_noise(self):
        E = np.linspace(EQ_HER - 0.80, EQ_HER - 0.05, 80)
        free = simulate_her_free_bath_polarization(
            E, i0_her_A_m2=I0_HER, b_her_V_dec=0.150, E_eq_her_V=EQ_HER,
            noise_rel_fraction=0.03, seed=3)
        fit = fit_her_from_free_bath(free["potentials_V"], free["i_her_A_m2"],
                                     E_eq_her_V=EQ_HER)
        assert fit["b_her_V_dec"] == pytest.approx(0.150, rel=0.10)
        assert abs(math.log10(fit["i0_her_A_m2"]) - math.log10(I0_HER)) < 0.4


class TestFeGivenHER:
    def _fe_bath(self):
        E = np.linspace(-0.50, -1.05, 80)
        omegas = np.array([400.0, 900.0, 1600.0, 2500.0])
        return simulate_fe_bath_rde_polarization(
            E, omegas, fe_i0_A_m2=I0_FE, fe_tafel_V=B_FE, fe_E_eq_V=-0.440,
            b_her_V_dec=B_HER, i0_her_A_m2=I0_HER, E_eq_her_V=EQ_HER,
            fe_conc_M=1.0, D_m2_s=7.2e-10)

    def test_constrained_fit_recovers_fe_given_true_her(self):
        bath = self._fe_bath()
        df = bath["frame"]
        fit = fit_fe_kinetics_given_her(
            df["potential_V"].to_numpy(float), df["i_total_A_m2"].to_numpy(float),
            i_lim_A_m2=df["i_lim_A_m2"].to_numpy(float),
            b_her_V_dec=B_HER, i0_her_A_m2=I0_HER, E_eq_her_V=EQ_HER,
            E_eq_fe_V=-0.440)
        assert fit["converged"]
        assert fit["fe_tafel_V_dec"] == pytest.approx(B_FE, rel=0.06)
        assert abs(math.log10(fit["fe_i0_A_m2"]) - math.log10(I0_FE)) < 0.3
        assert fit["r_squared"] > 0.98

    def test_fe_fit_uses_recovered_her_from_free_bath(self):
        # Full two-step: HER measured first (Fe-free), then Fe fitted with it fixed.
        bath = self._fe_bath()
        df = bath["frame"]
        E_free = np.linspace(EQ_HER - 0.80, EQ_HER - 0.05, 60)
        free = simulate_her_free_bath_polarization(
            E_free, i0_her_A_m2=I0_HER, b_her_V_dec=B_HER, E_eq_her_V=EQ_HER)
        fit_h = fit_her_from_free_bath(free["potentials_V"], free["i_her_A_m2"],
                                       E_eq_her_V=EQ_HER)
        fit = fit_fe_kinetics_given_her(
            df["potential_V"].to_numpy(float), df["i_total_A_m2"].to_numpy(float),
            i_lim_A_m2=df["i_lim_A_m2"].to_numpy(float),
            b_her_V_dec=fit_h["b_her_V_dec"], i0_her_A_m2=fit_h["i0_her_A_m2"],
            E_eq_her_V=EQ_HER, E_eq_fe_V=-0.440)
        assert fit["fe_tafel_V_dec"] == pytest.approx(B_FE, rel=0.08)
        assert abs(math.log10(fit["fe_i0_A_m2"]) - math.log10(I0_FE)) < 0.35

    def test_rde_convenience_wrapper(self):
        bath = self._fe_bath()
        E = np.linspace(-0.50, -1.05, 80)
        omegas = np.array([400.0, 900.0, 1600.0, 2500.0])
        # Frame is omega-major (outer loop omega, inner loop E), so row-aligned
        # E is E-major tiled across omegas and omega is omega-major repeated.
        df = bath["frame"]  # natural order, no re-sort
        E_row = np.tile(E, len(omegas))
        omega_row = np.repeat(omegas, len(E))
        fit = fit_fe_given_her_on_rde(
            E_row, df["i_total_A_m2"].to_numpy(float), omega_row,
            fe_conc_M=1.0, b_her_V_dec=B_HER, i0_her_A_m2=I0_HER,
            E_eq_her_V=EQ_HER, E_eq_fe_V=-0.440)
        assert fit["fe_tafel_V_dec"] == pytest.approx(B_FE, rel=0.08)


class TestVolumetricClosure:
    def test_h2_volume_moles_roundtrip(self):
        n = 1e-4  # mol
        V = h2_volume_from_moles(n)
        n_back = h2_moles_from_volume(V)
        assert n_back == pytest.approx(n, rel=1e-9)

    def test_closure_closes_when_fe_and_her_sum_to_total(self):
        # synthetic 60 s run at 100 A/m2 on 1e-4 m2 → charge 0.6 C
        j, area, t = 100.0, 1e-4, 60.0
        Q = j * area * t
        # split 80/20 HER/Fe
        q_her = 0.8 * Q
        q_fe = 0.2 * Q
        n_h2 = q_her / (Z_FE * FARADAY)
        fe_kg = q_fe * M_FE / (Z_FE * FARADAY)
        vol = volumetric_h2_closure(
            h2_moles=n_h2, fe_deposit_kg=fe_kg,
            j_A_m2=j, electrode_area_m2=area, run_time_s=t)
        assert vol["FE_her_gas"] == pytest.approx(0.8, abs=1e-6)
        assert vol["FE_fe_mass"] == pytest.approx(0.2, abs=1e-6)
        assert vol["closure"] == pytest.approx(1.0, abs=1e-6)

    def test_closure_branch_residual(self):
        j, area, t, i_her = 100.0, 1e-4, 60.0, 80.0
        q_her = i_her * area * t  # perfectly consistent with branch
        n_h2 = q_her / (Z_FE * FARADAY)
        vol = volumetric_h2_closure(
            h2_moles=n_h2, j_A_m2=j, electrode_area_m2=area, run_time_s=t,
            i_her_at_run_A_m2=i_her)
        assert vol["her_branch_residual"] == pytest.approx(0.0, abs=1e-9)

    def test_missing_inputs_raise(self):
        with pytest.raises(ValueError):
            volumetric_h2_closure()  # no h2, no charge


class TestSelfTestAndContracts:
    def test_self_test_all_pass(self):
        res = self_test(seed=7, verbose=False)
        assert res["all_pass"]
        assert all(res["verdict"].values())

    def test_measurement_spec_first_step_is_her(self):
        spec = measurement_spec()
        assert spec["sequence"][0]["step"] == 1
        assert "Fe-free" in spec["sequence"][0]["title"]
        assert "FeSO4" in spec["sequence"][0]["bath"]  # HER measured with Fe absent
        assert len(spec["gates"]) >= 4

    def test_model_scope_contract(self):
        scope = model_scope()
        assert "computes" in scope
        assert "does_not_compute" in scope
        assert "calibration_required" in scope
        assert "limitations" in scope
        assert any("HER" in c for c in scope["computes"])
        assert GAS_CORR  # non-empty gas-correction note
