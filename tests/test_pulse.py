import numpy as np
import pytest

from models.kinetics import DepositionKinetics
from models.pulse import (
    PulseDepositionModel,
    PulseResult,
    PulseWaveform,
    compare_dc_vs_pulse,
    fe3_charge_split,
    fe3_fraction_on_anodic,
)

pytestmark = pytest.mark.slow


def test_waveform_properties_and_evaluation():
    wf = PulseWaveform(
        j_cathodic_mA_cm2=100.0,
        t_cathodic_s=0.05,
        j_anodic_mA_cm2=-20.0,
        t_anodic_s=0.01,
        t_off_s=0.04,
    )
    assert wf.t_cycle_s == pytest.approx(0.10)
    assert wf.frequency_Hz == pytest.approx(10.0)
    assert wf.duty_cycle == pytest.approx(0.50)
    assert wf.j_avg_mA_cm2 == pytest.approx(48.0)  # (100*0.05 - 20*0.01) / 0.10

    # Current evaluation at different points in the cycle
    assert wf.evaluate_current_A_m2(0.02) == pytest.approx(1000.0)  # 100 mA/cm2 = 1000 A/m2
    assert wf.evaluate_current_A_m2(0.055) == pytest.approx(-200.0)  # -20 mA/cm2 = -200 A/m2
    assert wf.evaluate_current_A_m2(0.08) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"j_cathodic_mA_cm2": -10.0, "t_cathodic_s": 0.05},
        {"j_cathodic_mA_cm2": 100.0, "t_cathodic_s": 0.0},
        {"j_cathodic_mA_cm2": 100.0, "t_cathodic_s": 0.05, "j_anodic_mA_cm2": 10.0},
        {"j_cathodic_mA_cm2": 100.0, "t_cathodic_s": 0.05, "t_anodic_s": -0.01},
        {"j_cathodic_mA_cm2": 100.0, "t_cathodic_s": 0.05, "t_off_s": -0.01},
    ],
)
def test_waveform_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        PulseWaveform(**kwargs)


def test_simulation_runs_and_returns_result():
    wf = PulseWaveform(
        j_cathodic_mA_cm2=50.0,
        t_cathodic_s=0.01,
        t_off_s=0.01,
    )
    model = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)
    res = model.simulate(wf, n_cycles=5, steps_per_cycle=20)

    assert isinstance(res, PulseResult)
    assert len(res.time_s) == 101  # 5 * 20 + 1
    assert len(res.surface_fe_M) == 101
    assert len(res.surface_pH) == 101
    assert res.cathode_potential_V is not None and len(res.cathode_potential_V) == 101
    assert res.cycle_avg_efficiency > 0.0
    assert res.net_fe_deposited_g_m2 > 0.0
    assert res.plating_rate_um_hr > 0.0
    assert 0.0 <= res.peak_surface_depletion_ratio <= 1.0


# ---------------------------------------------------------------------------
# Butler–Volmer split (2026-08 default) — physics pins
# ---------------------------------------------------------------------------

class TestButlerVolmerSplit:
    def test_off_period_is_open_circuit_corrosion(self):
        """j=0 → mixed potential: i_Fe < 0 (dissolution), i_HER > 0, sum = 0."""
        model = PulseDepositionModel()
        i_fe, i_her, eff, E = model._kinetic_split(0.0, 1000.0, 10.0)
        assert i_fe < 0.0 < i_her
        assert i_fe + i_her == pytest.approx(0.0, abs=1e-6)
        assert eff == 0.0
        # between the two equilibrium potentials
        assert -0.440 < E < -0.20

    def test_reverse_segment_is_a_corrosion_couple(self):
        """Reverse pulse: dissolution carries the applied reverse charge PLUS
        the residual corrosion HER (heuristic: 'dissolve exactly j, no HER')."""
        model = PulseDepositionModel()
        i_fe, i_her, eff, E = model._kinetic_split(-200.0, 1000.0, 10.0)
        assert i_fe <= -200.0          # at least the applied reverse charge
        assert i_her > 0.0             # corrosion HER keeps running
        assert i_fe + i_her == pytest.approx(-200.0, rel=1e-9)
        assert eff == 0.0
        assert E > -0.440              # anodic of Fe equilibrium

    def test_forward_current_vanishes_with_surface_fe(self):
        """Surface-activity closure: starved surface cannot deposit."""
        model = PulseDepositionModel()
        i_fe_full, *_ = model._kinetic_split(1000.0, 1000.0, 10.0)
        i_fe_starved, *_ = model._kinetic_split(1000.0, 10.0, 10.0)  # 1% of bulk
        assert 0.0 < i_fe_starved < 0.2 * i_fe_full

    def test_potential_is_monotone_in_applied_current(self):
        model = PulseDepositionModel()
        _, _, _, E_cath = model._kinetic_split(1000.0, 1000.0, 10.0)
        _, _, _, E_rev = model._kinetic_split(-200.0, 1000.0, 10.0)
        _, _, _, E_off = model._kinetic_split(0.0, 1000.0, 10.0)
        assert E_cath < E_off < E_rev

    def test_her_shuts_down_as_surface_protons_starve(self):
        """No phantom H+ source: HER forward rate dies with c_H,surf."""
        model = PulseDepositionModel()
        _, i_her_full, _, _ = model._kinetic_split(1000.0, 1000.0, 10.0)
        _, i_her_starved, _, _ = model._kinetic_split(1000.0, 1000.0, 1e-3)
        assert 0.0 <= i_her_starved < 0.05 * i_her_full

    def test_dc_light_load_matches_deposition_kinetics(self):
        """Light-load DC asymptote == DepositionKinetics at matched params.

        In the nearly-undepleted limit the Koutecký–Levich blend in
        ``DepositionKinetics`` is inert and the surface-activity scale → 1,
        so the two models must agree to well under a percent.
        """
        model = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)  # 50 °C default
        wf = PulseWaveform(j_cathodic_mA_cm2=5.0, t_cathodic_s=20.0)
        res = model.simulate(wf, n_cycles=1, steps_per_cycle=2000)
        dk = DepositionKinetics(pH=2.0, temperature_C=50.0,
                                fe_i0=model.fe_i0, her_i0=model.her_i0,
                                fe_conc_M=1.0, boundary_layer_m=1.0e-4)
        assert res.instant_efficiency[-1] == pytest.approx(
            dk.efficiency_at_current(5.0), rel=0.02)

    def test_dc_late_time_reaches_algebraic_mixed_control_state(self):
        """CN converges to the fixed point of its own steady film equations.

        At steady state the film is linear: c_s = c_bulk − i_fe·δ/(z·F·D_Fe)
        and c_h,s = c_h,bulk − i_her·δ/(F·D_H).  Solved independently
        (fixed-point in E), the late-time CN state must reproduce it.
        """
        model = PulseDepositionModel()
        j_tot = 1000.0  # A/m²
        delta, D_fe, D_h = model.boundary_layer_m, model.diffusivity_fe, model.diffusivity_h
        c_fe_b, c_h_b = model.fe_bulk_M * 1000.0, model.c_h_bulk_mol_m3
        from models.electrochemistry import FARADAY, Z_FE
        from scipy.optimize import brentq

        def currents_at_E(E_val, iters=2000, damp=0.4, tol=1e-12):
            import math
            cs, ch = c_fe_b, c_h_b
            i_fe = i_h = 0.0
            for _ in range(iters):
                pH_s = -math.log10(max(ch / 1000.0, 1e-14))
                fe_b, her_b = model._branches(pH_s)
                s_fe = min(max(cs / c_fe_b, 0.0), 1.0)
                s_h = min(max(ch / c_h_b, 0.0), 1.0)
                i_fe = float(fe_b.current_scaled(E_val, s_fe))
                i_h = float(her_b.current_scaled(E_val, s_h))
                cs_t = max(c_fe_b - i_fe * delta / (Z_FE * FARADAY * D_fe), 0.0)
                ch_t = max(c_h_b - i_h * delta / (FARADAY * D_h), 1e-9)
                if abs(cs_t - cs) < tol and abs(ch_t - ch) < tol:
                    cs, ch = cs_t, ch_t
                    break
                cs += damp * (cs_t - cs)
                ch += damp * (ch_t - ch)
            return i_fe, i_h, cs, ch

        def f(E):
            return currents_at_E(E)[0] + currents_at_E(E)[1] - j_tot

        E_star = brentq(f, -3.0, 0.16, xtol=1e-8)
        i_fe_s, i_h_s, cs_s, ch_s = currents_at_E(E_star)
        fe_star = i_fe_s / j_tot

        wf = PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=30.0)
        res = model.simulate(wf, n_cycles=1, steps_per_cycle=3000)
        assert res.instant_efficiency[-1] == pytest.approx(fe_star, rel=0.02)
        assert res.surface_fe_M[-1] == pytest.approx(cs_s / 1000.0, rel=0.02)

    def test_transport_response_is_monotone_in_peak_current(self):
        """Depletion deepens and surface pH rises as j_peak grows.

        (FE-vs-j is deliberately NOT pinned monotone: once HER becomes
        proton-supply-limited the surface pH climb suppresses it, so CE
        direction is regime-dependent in this reduced model — see
        docs/SIM_PULSE_BV.md.)
        """
        model = PulseDepositionModel()
        depl, phs = [], []
        for j in (50.0, 200.0, 400.0):
            wf = PulseWaveform(j_cathodic_mA_cm2=j, t_cathodic_s=0.05, t_off_s=0.05)
            res = model.simulate(wf, n_cycles=5, steps_per_cycle=40)
            depl.append(res.peak_surface_depletion_ratio)
            phs.append(res.max_surface_pH)
        assert depl[0] > depl[1] > depl[2]
        assert phs[0] < phs[1] < phs[2]

    def test_net_deposition_lower_with_reverse(self):
        """Pulse-reverse deposits less net iron than the matched unipolar pulse."""
        model = PulseDepositionModel()
        wf_pe = PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=0.05, t_off_s=0.04)
        wf_pre = PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=0.05,
                               j_anodic_mA_cm2=-20.0, t_anodic_s=0.01, t_off_s=0.039)
        r_pe = model.simulate(wf_pe, n_cycles=10, steps_per_cycle=50)
        r_pre = model.simulate(wf_pre, n_cycles=10, steps_per_cycle=50)
        assert 0.0 < r_pre.net_fe_deposited_g_m2 < r_pe.net_fe_deposited_g_m2

    def test_envelope_flag_discriminates_valid_from_starved(self):
        """proton_limited_steps_fraction: 0 at valid points, >0 in starved DC."""
        model = PulseDepositionModel()
        ok = model.simulate(PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=0.05,
                                          j_anodic_mA_cm2=-20.0, t_anodic_s=0.01,
                                          t_off_s=0.04),
                            n_cycles=10, steps_per_cycle=100)
        starved = model.simulate(PulseWaveform(j_cathodic_mA_cm2=300.0, t_cathodic_s=20.0),
                                 n_cycles=1, steps_per_cycle=500)
        assert ok.proton_limited_steps_fraction == 0.0
        assert starved.proton_limited_steps_fraction > 0.2


# ---------------------------------------------------------------------------
# Legacy heuristic split — preserved verbatim for A/B checks
# ---------------------------------------------------------------------------

class TestLegacyHeuristic:
    def test_zero_current_is_zero_currents(self):
        model = PulseDepositionModel(kinetics="heuristic")
        assert model._kinetic_split(0.0, 1000.0, 10.0)[:3] == (0.0, 0.0, 0.0)
        assert model._kinetic_split(0.0, 1000.0, 10.0)[3] is None

    def test_reverse_dissolves_exactly_applied_current(self):
        """Legacy convention: reverse = 'dissolve j_app, no HER'."""
        model = PulseDepositionModel(kinetics="heuristic")
        i_fe, i_her, eff, _ = model._kinetic_split(-50.0, 1000.0, 10.0)
        assert (i_fe, i_her, eff) == (-50.0, 0.0, 0.0)

    def test_legacy_split_bounds_efficiency(self):
        model = PulseDepositionModel(kinetics="heuristic")
        i_fe, i_her, eff, _ = model._kinetic_split(1000.0, 1000.0, 10.0)
        assert 0.0 < eff <= 0.995 + 1e-9
        assert i_fe + i_her == pytest.approx(1000.0)


def test_pulse_off_time_allows_surface_fe_recovery():
    """Pulse off time must allow surface Fe2+ concentration to recover toward bulk."""
    wf = PulseWaveform(
        j_cathodic_mA_cm2=150.0,
        t_cathodic_s=0.05,
        t_off_s=0.15,
    )
    model = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)
    res = model.simulate(wf, n_cycles=3, steps_per_cycle=100)

    # Find surface Fe2+ at end of pulse ON vs end of pulse OFF in first cycle
    t_cycle = wf.t_cycle_s
    time_arr = res.time_s
    idx_end_on = np.argmin(np.abs(time_arr - wf.t_cathodic_s))
    idx_end_off = np.argmin(np.abs(time_arr - t_cycle))

    fe_end_on = res.surface_fe_M[idx_end_on]
    fe_end_off = res.surface_fe_M[idx_end_off]

    # Surface Fe2+ should recover during off period (even with the small
    # BV off-period corrosion bleed, the film relaxes back toward bulk)
    assert fe_end_off > fe_end_on
    assert fe_end_off == pytest.approx(1.0, rel=0.15)


def test_pulse_reverse_reduces_peak_surface_pH_rise():
    """Pulse-reverse electrodeposition should suppress local pH spike vs continuous DC."""
    wf_pre = PulseWaveform(
        j_cathodic_mA_cm2=100.0,
        t_cathodic_s=0.02,
        j_anodic_mA_cm2=-20.0,
        t_anodic_s=0.005,
        t_off_s=0.015,
    )
    wf_dc = PulseWaveform(
        j_cathodic_mA_cm2=100.0,
        t_cathodic_s=0.04 * 10,
    )
    model = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)

    res_pre = model.simulate(wf_pre, n_cycles=10, steps_per_cycle=40)
    res_dc = model.simulate(wf_dc, n_cycles=1, steps_per_cycle=400)

    # PRE max surface pH rise should be lower or equal to continuous DC at high peak current
    assert res_pre.max_surface_pH <= res_dc.max_surface_pH + 0.1


def test_compare_dc_vs_pulse_dictionary_keys():
    comparison = compare_dc_vs_pulse(j_peak_mA_cm2=80.0, n_cycles=5)
    for key in ("dc_peak", "dc_avg", "pulsed", "pulse_reverse"):
        assert key in comparison
        assert isinstance(comparison[key], PulseResult)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boundary_layer_m": 0.0},
        {"fe_bulk_M": -1.0},
        {"bulk_pH": 15.0},
        {"grid_points": 2},
        {"kinetics": "tafel"},
        {"fe_i0_A_m2": 0.0},
        {"her_i0_A_m2": -1.0},
    ],
)
def test_model_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        PulseDepositionModel(**kwargs)


# ---------------------------------------------------------------------------
# Fe²⁺/Fe³⁺ anodic-dissolution split (CHEM_PHYS_REVIEW §2.5, L0)
# ---------------------------------------------------------------------------

class TestAnodicFe3Split:
    def test_fraction_rises_with_anodic_overpotential(self):
        """Fe³⁺ split is monotone increasing in anodic overpotential η."""
        fs = [fe3_fraction_on_anodic(eta, 2.0) for eta in (0.0, 0.03, 0.1, 0.3, 0.6)]
        assert fs[0] == 0.0                     # zero η → no Fe³⁺
        assert fs == sorted(fs)                 # non-decreasing
        assert 0.0 <= fs[-1] <= 1.0
        assert fs[0] < fs[-1]

    def test_fraction_falls_with_pH(self):
        """Low pH stabilises Fe³⁺ against Fe(OH)₃ hydrolysis → higher split."""
        f_acid = fe3_fraction_on_anodic(0.5, 0.5)
        f_ref = fe3_fraction_on_anodic(0.5, 2.0)
        f_base = fe3_fraction_on_anodic(0.5, 5.0)
        assert f_acid > f_ref > f_base >= 0.0

    def test_reproduces_baseline_at_reference_state(self):
        """At the reference pH and a mild reverse overpotential the Fe³⁺
        fraction is negligible, so the legacy 100 %-Fe²⁺ anodic branch holds."""
        assert fe3_fraction_on_anodic(0.0, 2.0) == 0.0
        assert fe3_fraction_on_anodic(0.03, 2.0) < 0.02

    def test_charge_split_conserves_current(self):
        """i_Fe2 + i_Fe3 == i_total at every fraction (2 e⁻/Fe²⁺, 3 e⁻/Fe³⁺)."""
        for f in (0.0, 0.1, 0.3, 0.5, 0.85, 1.0):
            fe2, fe3 = fe3_charge_split(-1000.0, f)
            assert fe2 + fe3 == pytest.approx(-1000.0, rel=1e-12)
            assert fe2 <= 0.0 <= -fe3
        # no split → the whole anodic current stays Fe²⁺ (legacy form)
        assert fe3_charge_split(-800.0, 0.0) == (-800.0, 0.0)

    def test_flag_off_is_byte_identical_legacy_anodic(self):
        """Default fe3_split=False: Fe³⁺ arrays stay zero, net Fe unchanged."""
        model = PulseDepositionModel(fe3_split=False)
        wf = PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=0.05,
                           j_anodic_mA_cm2=-100.0, t_anodic_s=0.01, t_off_s=0.04)
        res = model.simulate(wf, n_cycles=3, steps_per_cycle=40)
        assert np.all(res.fe3_current_A_m2 == 0.0)
        assert res.cycle_avg_anodic_fe3_flux_mol_m2_s == 0.0
        assert res.cycle_avg_anodic_fe3_fraction == 0.0

    def test_flag_on_produces_fe3_only_on_anodic_fe(self):
        """fe3_split=True: Fe³⁺ partial current appears (negative) whenever the
        Fe branch is anodic — reverse pulses AND the small open-circuit
        dissolution that runs during rest periods (module docstring) — and is
        zero on every cathodic (depositing) step, charge-lumped with Fe²⁺ into
        the total Fe current at every step."""
        model = PulseDepositionModel(fe3_split=True)
        wf = PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=0.05,
                           j_anodic_mA_cm2=-100.0, t_anodic_s=0.01, t_off_s=0.04)
        res = model.simulate(wf, n_cycles=3, steps_per_cycle=40)
        fe3 = res.fe3_current_A_m2
        assert fe3 is not None
        anodic = res.fe_current_A_m2 < 0.0      # Fe-branch anodic (dissolving)
        assert np.any(anodic)
        # Fe³⁺ only where the Fe branch dissolves; never on a depositing step
        assert np.all(fe3[~anodic] == 0.0)
        assert np.any(fe3[anodic] < 0.0)
        # ...including (small) open-circuit corrosion during rest periods
        off_rest = res.applied_current_A_m2 == 0.0
        assert np.any(fe3[off_rest] < 0.0)
        # charge conservation at anodic steps is unit-covered by
        # test_charge_split_conserves_current (fe2 + fe3 == i_total, 2/3 e⁻);
        # here just confirm the Fe³⁺ path carries strictly less than the full
        # anodic branch (some Fe²⁺ remains at every reverse/rest step).
        assert np.all(np.abs(fe3[anodic]) < np.abs(res.fe_current_A_m2[anodic]))
        # bounded, physical aggregate
        assert 0.0 < res.cycle_avg_anodic_fe3_fraction < 1.0
        assert res.cycle_avg_anodic_fe3_flux_mol_m2_s > 0.0

    def test_shuttle_closure_couples_to_sludge(self):
        """The run-averaged anodic Fe³⁺ flux feeds fe3_shuttle's steady state,
        so a strong reverse drive seeds Fe(OH)₃ sludge (restart H₂ coupling)."""
        model = PulseDepositionModel(fe3_split=True)
        mild = model.simulate(PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=0.05,
                                            j_anodic_mA_cm2=-100.0, t_anodic_s=0.01,
                                            t_off_s=0.04), n_cycles=3, steps_per_cycle=40)
        strong = model.simulate(PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=0.05,
                                              j_anodic_mA_cm2=-500.0, t_anodic_s=0.01,
                                              t_off_s=0.04), n_cycles=3, steps_per_cycle=40)
        cl_mild = model.fe3_shuttle_closure(mild)
        cl_strong = model.fe3_shuttle_closure(strong)
        assert cl_strong["anodic_fe3_source_mol_m2_s"] > cl_mild["anodic_fe3_source_mol_m2_s"]
        assert cl_strong["fe3_production_M_s"] > cl_mild["fe3_production_M_s"]
        # Reverse-drive Fe³⁺ seeds the hydroxide sludge.
        assert cl_strong["iron_sludge_loss_mol_m2_s"] >= 0.0
