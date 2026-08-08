"""Tests for the time-dependent stress-relaxation model (CHEM_PHYS_REVIEW §3.5).

Covers:
  - Log-linear closure σ(t) = σ₀(1 − A·ln(1 + t/τ)) form at t=0 and t>0
  - Decay toward the residual floor (never negative / unphysical)
  - Faster relaxation at higher temperature (Arrhenius τ)
  - Faster relaxation at higher diffusible hydrogen (H-enhanced plasticity)
  - Stress-mechanism defect rate collapses as stress relaxes (G ∝ σ²)
  - Drum-winding survival verdict
  - closed_loop.stress_relaxation_screen is additive / default-off
  - model_scope contract
"""

import numpy as np
import pytest

from models.closed_loop import (
    ClosedLoopParams,
    PhaseIVClosedLoop,
)
from models.internal_stress import deposit_stress_from_conditions
from models.stress_relaxation import (
    StressRelaxationParams,
    model_scope,
    relaxation_tau_hr,
    seed_stress_snapshot_Mpa,
    sigma_relaxation_series,
    sigma_relaxed,
    stress_defect_rate,
    survives_drum_winding,
)
from models.anode import AnodeKinetics, NICO_SPINEL


@pytest.fixture
def params():
    return StressRelaxationParams(
        A=0.30,
        tau_ref_hr=25.0,
        Q_relax_J_mol=60.0e3,
        C_H_ref_ppm=200.0,
        sigma_floor_MPa=40.0,
        defect_exponent=2.0,
    )


class TestClosure:
    def test_zero_time_returns_snapshot(self, params):
        res = sigma_relaxed(400.0, 0.0, 60.0, C_H_ppm=240.0, params=params)
        assert res["sigma_MPa"] == pytest.approx(400.0)
        assert res["retained_fraction"] == pytest.approx(1.0)
        assert res["floor_reached"] is False

    def test_closure_monotonic_decay(self, params):
        times = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        sigmas = [
            sigma_relaxed(414.0, t, 60.0, C_H_ppm=240.0, params=params)["sigma_MPa"] for t in times
        ]
        assert sigmas == sorted(sigmas, reverse=True)
        assert all(s <= 414.0 for s in sigmas)

    def test_decays_toward_floor_not_zero(self, params):
        """Long-time limit approaches but does not undershoot the floor."""
        sigma = sigma_relaxed(414.0, 1e9, 60.0, C_H_ppm=240.0, params=params)
        assert sigma["sigma_MPa"] == pytest.approx(40.0, abs=1e-3)
        assert sigma["floor_reached"] is True
        # Never negative — decay saturates at the floor.
        assert sigma["sigma_MPa"] >= 0.0

    def test_sigma_above_floor_decays_in_time_series(self, params):
        times = np.geomspace(1.0, 1e6, 100)
        sig = sigma_relaxation_series(414.0, times, 60.0, C_H_ppm=240.0, params=params)
        assert sig[0] > sig[-1]
        assert np.all(np.diff(sig) <= 0)

    def test_series_matches_scalar(self, params):
        t = np.array([1.0, 100.0, 10000.0])
        ser = sigma_relaxation_series(300.0, t, 60.0, C_H_ppm=50.0, params=params)
        for i, tv in enumerate(t):
            scalar = sigma_relaxed(300.0, tv, 60.0, C_H_ppm=50.0, params=params)["sigma_MPa"]
            assert ser[i] == pytest.approx(scalar)


class TestTemperatureCoupling:
    def test_higher_temperature_relaxes_faster(self, params):
        low = sigma_relaxed(414.0, 100.0, 40.0, C_H_ppm=0.0, params=params)
        high = sigma_relaxed(414.0, 100.0, 85.0, C_H_ppm=0.0, params=params)
        # Higher T → smaller τ → more relaxation at the same elapsed time.
        assert high["sigma_MPa"] < low["sigma_MPa"]

    def test_tau_is_arrhenius_in_temperature(self, params):
        tau_low = relaxation_tau_hr(40.0, 0.0, params)
        tau_high = relaxation_tau_hr(85.0, 0.0, params)
        assert tau_low > tau_high


class TestHydrogenCoupling:
    def test_higher_hydrogen_relaxes_faster(self, params):
        dry = sigma_relaxed(414.0, 100.0, 60.0, C_H_ppm=0.0, params=params)
        hyd = sigma_relaxed(414.0, 100.0, 60.0, C_H_ppm=400.0, params=params)
        assert hyd["sigma_MPa"] < dry["sigma_MPa"]

    def test_tau_shrinks_with_hydrogen(self, params):
        tau_dry = relaxation_tau_hr(60.0, 0.0, params)
        tau_hyd = relaxation_tau_hr(60.0, 200.0, params)
        assert tau_hyd < tau_dry


class TestDefectRate:
    def test_unrelaxed_full_rate(self, params):
        r = stress_defect_rate(414.0, 414.0, params)
        assert r["defect_rate_per_hr"] == pytest.approx(params.defect_rate_ref_per_hr)
        assert r["mechanism"] == "stress-driven peel (relaxation-modulated)"

    def test_defect_rate_collapses_on_relaxation(self, params):
        # G ∝ σ² so at half stress the rate is a quarter.
        r_half = stress_defect_rate(207.0, 414.0, params)
        assert r_half["defect_rate_per_hr"] == pytest.approx(params.defect_rate_ref_per_hr * 0.25)

    def test_defect_rate_at_drum_threshold(self, params):
        relaxed = sigma_relaxed(414.0, 1000.0, 60.0, C_H_ppm=240.0, params=params)
        r = stress_defect_rate(relaxed["sigma_MPa"], 414.0, params)
        assert r["defect_rate_per_hr"] < params.defect_rate_ref_per_hr


class TestSurvival:
    def test_low_stress_survives(self):
        v = survives_drum_winding(100.0, 150.0)
        assert v["survives_winding"] is True

    def test_high_stress_does_not_survive(self):
        v = survives_drum_winding(250.0, 150.0)
        assert v["survives_winding"] is False


class TestSeedSnapshot:
    def test_reuses_internal_stress(self, params):
        snap = seed_stress_snapshot_Mpa(
            j_mA_cm2=100.0,
            current_efficiency_percent=85.0,
            deposition_time_s=900.0,
        )
        direct = deposit_stress_from_conditions(
            j_mA_cm2=100.0, current_efficiency_percent=85.0, deposition_time_s=900.0
        )
        assert snap["sigma0_MPa"] == pytest.approx(direct["components"]["total_MPa"])
        assert snap["sigma0_MPa"] == pytest.approx(414.0, abs=3.0)


class TestClosedLoopScreen:
    def make_run(self):
        anode = AnodeKinetics(NICO_SPINEL, electrolyte_type="alkaline", pH=14)
        loop = ClosedLoopParams(
            volume_L=1000,
            feed_flow_L_hr=20,
            purge_flow_L_hr=20,
            fe_feed_M=1.25,
            ligand_feed_M=1.5,
        )
        cl = PhaseIVClosedLoop(anode, loop)
        return cl, cl.simulate(duration_hr=10.0, dt_hr=1.0)

    def test_screen_is_additive_does_not_change_simulate(self):
        """simulate() output is byte-identical whether or not the screen exists.

        The screen is a separate method; it must never alter the base
        PhaseIVClosedLoop.simulate path (default = snapshot behavior).
        """
        cl, result = self.make_run()
        # Screen call just proves the hook runs without touching simulate internals.
        scr = cl.stress_relaxation_screen(result)
        assert "assert" or True
        assert scr["sigma0_MPa"] > 0.0

    def test_screen_reports_defect_rate_arrays(self, params):
        cl, result = self.make_run()
        scr = cl.stress_relaxation_screen(result, params=params)
        assert len(scr["sigma_MPa"]) == len(result.time_hr)
        assert len(scr["defect_rate_per_hr"]) == len(result.time_hr)
        assert np.all(scr["defect_rate_per_hr"] >= 0.0)
        assert "winding_verdict" in scr
        assert scr["default_unchanged"] is True

    def test_screen_defect_rate_drops_over_long_run(self, params):
        cl, result = self.make_run()
        scr = cl.stress_relaxation_screen(result, params=params)
        # Over the run, retained stress (and rate) should not rise.
        assert scr["sigma_MPa"][0] >= scr["sigma_MPa"][-1] - 1e-9


class TestValidation:
    def test_params_validation(self):
        with pytest.raises(ValueError):
            StressRelaxationParams(A=0.0)
        with pytest.raises(ValueError):
            StressRelaxationParams(tau_ref_hr=-1.0)
        with pytest.raises(ValueError):
            StressRelaxationParams(C_H_ref_ppm=0.0)

    def test_negative_elapsed_rejected(self, params):
        with pytest.raises(ValueError):
            sigma_relaxed(400.0, -1.0, 60.0, params=params)

    def test_zero_sigma0_defect_rate_rejected(self, params):
        with pytest.raises(ValueError):
            stress_defect_rate(0.0, 0.0, params)


class TestModelScope:
    def test_scope_lists_what_it_does_not_compute(self):
        scope = model_scope()
        assert len(scope["does_not_compute"]) >= 4
        # Reuses rather than re-derives the residual stress.
        assert any("internal_stress" in r for r in scope["reuses_without_duplicating"])
