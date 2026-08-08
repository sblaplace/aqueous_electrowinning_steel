"""Tests for the potential-/double-layer-aware charge-transfer kinetics
(CHEM_PHYS_REVIEW.md §2.3): Marcus-like α_eff(η, ψ₁), the heat-of-activation
correction, and leveler/aditive adsorption as an organic ψ₁ dipole shift.

The two deliverables the card pins are asserted directly:
  * α_eff is monotone-decreasing in cathodic overpotential (the "alpha drop")
    and responds to ψ₁ (Frumkin).
  * a leveler/aditive Γ·μ dipole layer shifts ψ₁, and shifting ψ₁ changes α_eff.
And the "default path unchanged" constraint: use_frumkin_alpha_eff=False leaves
the base BV branch byte-identical.
"""
import numpy as np
import pytest

from models.frumkin import (
    DEBYE,
    FrumkinCorrectedBV,
    FrumkinParams,
    alpha_eff,
    heat_of_activation,
    organic_psi1_shift,
    tafel_slope_from_alpha,
)
from models.kinetics import (
    ButlerVolmerBranch,
    DepositionKinetics,
    TafelBranch,
)

FE_I0 = 1.0e-2
FE_E_EQ = -0.440


# ─── α_eff: Marcus-like potential-dependent symmetry factor ──────────
class TestAlphaEff:
    def test_monotone_decreasing_in_cathodic_eta(self):
        """The 'alpha drop': α_eff falls as cathodic |η| grows."""
        etas = np.linspace(0.0, 0.6, 13)
        vals = [alpha_eff(e) for e in etas]
        assert vals[0] == pytest.approx(0.5)
        # strictly decreasing on (0, 0.6)
        diffs = np.diff(vals)
        assert np.all(diffs < 0.0), f"α_eff must decrease with η: {vals}"
        # well past 200 mV the drop is material
        assert alpha_eff(0.20) < 0.5
        assert alpha_eff(0.40) < alpha_eff(0.20)

    def test_clamp_keeps_alpha_positive(self):
        """Hard floor applies at very high η (no nonphysical negative α)."""
        assert alpha_eff(5.0) == pytest.approx(0.05)
        assert alpha_eff(5.0, alpha_min=0.03) == pytest.approx(0.03)

    def test_frumkin_psi1_shifts_alpha(self):
        """More negative ψ₁ raises η_eff → further α drop (Frumkin)."""
        a0 = alpha_eff(0.30, psi_1_V=0.0)
        a_neg = alpha_eff(0.30, psi_1_V=-0.15)
        assert a_neg < a0
        # a positive ψ₁ (cation-dipole-up) suppresses the drop
        assert alpha_eff(0.30, psi_1_V=0.15) > a0

    def test_n_and_lambda_trend(self):
        """More electrons / smaller λ → faster α drop."""
        assert alpha_eff(0.25, n=2) < alpha_eff(0.25, n=1)
        assert (
            alpha_eff(0.25, lambda_J_mol=1.0 * 96485.0)
            < alpha_eff(0.25, lambda_J_mol=3.0 * 96485.0)
        )

    def test_slope_consistent_with_tafel_form_at_low_eta(self):
        """At η→0, tafel_slope_from_alpha(α₀) recovers the BV slope, and the
        local slope grows as α drops."""
        b_low = tafel_slope_from_alpha(0.5, n=1, T_K=298.15)
        b_high = tafel_slope_from_alpha(0.3, n=1, T_K=298.15)
        assert b_high > b_low  # smaller α → steeper slope (larger b)
        # screening sanity: α=0.5, n=1 at 298.15 → ~118 mV/dec
        assert b_low == pytest.approx(0.118, abs=0.005)


# ─── Heat-of-activation correction ──────────────────────────────────
class TestHeatOfActivation:
    def test_unity_at_equilibrium(self):
        assert heat_of_activation(0.0) == pytest.approx(1.0)

    def test_below_unity_and_monotone_at_high_eta(self):
        """f_act ≤ 1 and falls as |η| grows — the high-η bending."""
        assert heat_of_activation(0.30) < 1.0
        assert heat_of_activation(0.40) < heat_of_activation(0.30)
        assert heat_of_activation(0.02) == pytest.approx(1.0, abs=5e-3)

    def test_psi1_enters_through_eta_eff(self):
        # a negative ψ₁ enlarges η_eff → more quadratic barrier → smaller f
        assert heat_of_activation(0.30, psi_1_V=-0.15) < heat_of_activation(0.30)


# ─── Leveler / aditive adsorption as an organic ψ₁ dipole shift ─────
class TestOrganicPsi1Shift:
    def test_shift_is_negative_for_dipole_down(self):
        """A positive Γ·μ (dipole-down) organic shifts ψ₁ negative."""
        shift = organic_psi1_shift(gamma_organic_mol_m2=1e-5, mu_dipole_C_m=2.0 * DEBYE)
        assert shift < 0.0

    def test_magnitude_grows_with_gamma_and_mu(self):
        """The mechanism is 'ψ₁ shifted by Γ_organic × μ_dipole' — linear in both."""
        g1 = organic_psi1_shift(1e-5, 2.0 * DEBYE)
        g2 = organic_psi1_shift(2e-5, 2.0 * DEBYE)
        m2 = organic_psi1_shift(1e-5, 4.0 * DEBYE)
        assert abs(g2) / abs(g1) == pytest.approx(2.0, rel=1e-6)
        assert abs(m2) / abs(g1) == pytest.approx(2.0, rel=1e-6)
        # ~0.1–0.3 V screening scale for a monolayer organic
        assert 0.01 < abs(g1) < 1.0

    def test_higher_dielectric_reduces_shift(self):
        s_lo = organic_psi1_shift(1e-5, 2.0 * DEBYE, eps_r=6.0)
        s_hi = organic_psi1_shift(1e-5, 2.0 * DEBYE, eps_r=20.0)
        assert abs(s_hi) < abs(s_lo)

    def test_shift_levers_alpha_eff(self):
        """Ψ₁ from a leveler flows through to α_eff (the practical point)."""
        shift = organic_psi1_shift(1e-5, 2.0 * DEBYE)
        alpha_plain = alpha_eff(0.30, psi_1_V=0.0)
        alpha_lev = alpha_eff(0.30, psi_1_V=shift)
        assert alpha_lev < alpha_plain


# ─── FrumkinCorrectedBV branch wiring ───────────────────────────────
class TestFrumkinCorrectedBV:
    @pytest.fixture()
    def fbv(self):
        return FrumkinCorrectedBV(
            FE_I0,
            FE_E_EQ,
            FrumkinParams(n=2, T_K=298.15),
            i_lim=None,
        )

    def test_zero_at_equilibrium(self, fbv):
        assert fbv.current(FE_E_EQ) == pytest.approx(0.0, abs=1e-12)

    def test_cathodic_positive_anodic_signed(self, fbv):
        assert fbv.current(FE_E_EQ - 0.3) > 0.0
        assert fbv.current(FE_E_EQ + 0.1) < 0.0

    def test_reproduces_constant_alpha_bv_at_low_eta(self):
        """With α₀ matching the repo's low-η slope *and* a large λ (so the
        Marcus heat/drop corrections are negligible in the window), the Marcus
        cathodic arm coincides with constant-α BV's cathodic arm (TafelBranch
        is cathodic-only).  This isolates the α₀↔Tafel-slope calibration — the
        'explore over the kinetics Tafel slopes' constraint.  (Compare arms,
        not net currents: near E_eq the net is a difference of two ~i0 terms,
        so the anodic-slope choice would dominate a net-current comparison.)"""
        # repo Fe slope 0.120 V/dec, n=2 → α₀ = 2.303RT/(0.120·2F)
        alpha0_match = 2.303 * 8.314 * 298.15 / (0.120 * 2 * 96485.0)
        marcus = FrumkinCorrectedBV(
            FE_I0, FE_E_EQ,
            FrumkinParams(n=2, T_K=298.15, alpha0=alpha0_match,
                          lambda_J_mol=20.0 * 96485.0),
        )
        tafel = TafelBranch(FE_I0, 0.120, FE_E_EQ)
        for eta in (0.02, 0.05, 0.08):
            arm_c, _ = marcus._arm_c(eta)
            i_t = tafel.current(FE_E_EQ - eta)
            assert arm_c == pytest.approx(i_t, rel=5e-2), f"η={eta}"

    def test_high_eta_bends_below_constant_alpha_bv(self):
        """The Marcus drop + heat correction bends the >200 mV branch down."""
        alpha0_match = 2.303 * 8.314 * 298.15 / (0.120 * 2 * 96485.0)
        marcus = FrumkinCorrectedBV(
            FE_I0, FE_E_EQ, FrumkinParams(n=2, T_K=298.15, alpha0=alpha0_match)
        )
        b_const = ButlerVolmerBranch(FE_I0, 0.120, FE_E_EQ, None, 0.0392)
        for eta in (0.25, 0.40, 0.60):
            i_m = marcus.current(FE_E_EQ - eta)
            i_b = b_const.current(FE_E_EQ - eta)
            assert i_m < i_b, f"η={eta}: Marcus {i_m} should be below BV {i_b}"

    def test_i_lim_blends_only_cathodic_arm(self):
        fbv_lim = FrumkinCorrectedBV(
            FE_I0, FE_E_EQ, FrumkinParams(n=2, T_K=298.15), i_lim=100.0
        )
        assert fbv_lim.current(FE_E_EQ - 0.5) < 100.0
        # anodic arm unchanged by i_lim
        no_lim = FrumkinCorrectedBV(
            FE_I0, FE_E_EQ, FrumkinParams(n=2, T_K=298.15), i_lim=None
        )
        assert fbv_lim.current(FE_E_EQ + 0.1) == pytest.approx(
            no_lim.current(FE_E_EQ + 0.1)
        )


# ─── DepositionKinetics flag: default path unchanged ────────────────
class TestDepositionKineticsFlag:
    def test_off_default_matches_constant_alpha_bv(self):
        """use_frumkin_alpha_eff=False (default) is the unmodified BV path."""
        k = DepositionKinetics(temperature_C=50.0)
        E = np.linspace(-0.60, -0.44, 9)
        i_fe, i_h, i_tot = k.partial_currents(E)
        i_fe_ref = k.fe_branch_bv.current(E)
        i_h_ref = k.her_branch_bv.current(E)
        np.testing.assert_allclose(i_fe, i_fe_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(i_h, i_h_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(i_tot, i_fe_ref + i_h_ref, rtol=0, atol=1e-12)

    def test_on_calibrates_slope_via_alpha0(self):
        """With a large λ (Marcus corrections negligible in-window), the
        flag-on wired branch's cathodic arm reproduces the repo Fe Tafel slope
        — the α₀↔slope calibration flows through the constructor."""
        k_on = DepositionKinetics(
            temperature_C=50.0, use_frumkin_alpha_eff=True,
            frumkin_lambda_J_mol=50.0 * 96485.0,
        )
        tafel = TafelBranch(k_on.fe_i0_T, k_on.fe_tafel_V, k_on.fe_E_eq)
        for eta in (0.02, 0.06, 0.10):
            arm_c, _ = k_on.fe_frumkin_branch()._arm_c(eta)
            i_t = tafel.current(k_on.fe_E_eq - eta)
            assert arm_c == pytest.approx(i_t, rel=5e-2), f"η={eta}"

    def test_on_without_psi1_changes_only_high_eta(self):
        """Flag on, no leveler, realistic λ: the Fe branch bends below
        constant-α BV at high |η| (the honest Marcus high-|η| correction)
        while remaining positive/cathodic."""
        k_off = DepositionKinetics(temperature_C=50.0)
        k_on = DepositionKinetics(
            temperature_C=50.0,
            use_frumkin_alpha_eff=True,
            frumkin_lambda_J_mol=3.0 * 96485.0,
        )
        E_hi = np.array([-0.70, -0.80, -0.90])
        i_on = k_on.partial_currents(E_hi)[0]
        i_off = k_off.partial_currents(E_hi)[0]
        assert np.all(i_on > 0.0)                       # still cathodic
        assert np.all(i_on < i_off), f"Marcus should bend below: {i_on} vs {i_off}"

    def test_leveler_psi1_in_deposition_kinetics(self):
        """An organic leveler (Γ·μ) resolves into a more-negative ψ₁ that
        flows through the wired Frumkin branch into α_eff (the §2.3 (b)
        chain: leveler adsorption ↔ 'ψ₁ shifted by Γ_organic × μ_dipole')."""
        k_plain = DepositionKinetics(
            temperature_C=50.0, use_frumkin_alpha_eff=True,
            frumkin_lambda_J_mol=3.0 * 96485.0,
        )
        k_lev = DepositionKinetics(
            temperature_C=50.0, use_frumkin_alpha_eff=True,
            frumkin_lambda_J_mol=3.0 * 96485.0,
            organic_gamma_mol_m2=1e-5,
            organic_mu_dipole_C_m=2.0 * DEBYE,
        )
        # the leveler dipole layer shifts ψ₁ negative
        assert k_lev.resolved_psi1_V < k_plain.resolved_psi1_V
        assert k_plain.resolved_psi1_V == pytest.approx(0.0, abs=1e-12)
        # and that shift flows into the branch's α_eff at a cathodic η
        from models.frumkin import alpha_eff as _a
        fe_plain = k_plain.fe_frumkin_branch()
        fe_lev = k_lev.fe_frumkin_branch()
        eta = 0.30
        a_plain = _a(eta, fe_plain.params.psi_1_V, n=fe_plain.params.n,
                     lambda_J_mol=fe_plain.params.lambda_J_mol,
                     alpha0=fe_plain.params.alpha0)
        a_lev = _a(eta, fe_lev.params.psi_1_V, n=fe_lev.params.n,
                   lambda_J_mol=fe_lev.params.lambda_J_mol,
                   alpha0=fe_lev.params.alpha0)
        assert a_lev < a_plain
