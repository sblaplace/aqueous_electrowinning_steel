"""Tests for Fe-H₂O Pourbaix thermodynamics.

These are the thermodynamic foundations the rest of the model set assumes:
where iron can deposit, and by how much the HER beats it.  The headline claim
in the README — that Fe deposition lies below the HER line at every pH, with
the penalty narrowing from ~440 mV in acid to ~47 mV in alkali — is pinned
here so it cannot drift silently.
"""

import numpy as np
import pytest

from models.pourbaix import (
    E0_FE2_FE,
    E0_FE3_FE2,
    FePourbaix,
    her_line,
    nernst_pH_line,
    oer_line,
)

NERNST_SLOPE_25C = 0.05916  # V per pH unit at 298.15 K


class TestNernstLines:
    def test_pH_independent_couple_is_flat(self):
        """Fe²⁺/Fe involves no protons, so E must not vary with pH."""
        E = nernst_pH_line(E0_FE2_FE, np.array([0.0, 7.0, 14.0]), n_h=0, n_e=2)
        assert np.allclose(E, E0_FE2_FE)

    def test_unit_activity_reduces_to_standard_potential(self):
        assert nernst_pH_line(E0_FE3_FE2, 0.0, n_h=0, n_e=1, log_activity=0.0) == (
            pytest.approx(E0_FE3_FE2)
        )

    def test_proton_coupled_slope_is_59_mV_per_pH(self):
        """One proton per electron gives −59.16 mV/pH at 25 °C."""
        E0_, E14 = nernst_pH_line(0.0, np.array([0.0, 14.0]), n_h=1, n_e=1)
        assert (E14 - E0_) / 14.0 == pytest.approx(-NERNST_SLOPE_25C, abs=1e-4)

    def test_two_proton_two_electron_slope_also_59_mV(self):
        """The slope depends on n_H/n_e, so 2H⁺/2e⁻ matches 1H⁺/1e⁻."""
        E = nernst_pH_line(0.0, np.array([0.0, 10.0]), n_h=2, n_e=2)
        assert (E[1] - E[0]) / 10.0 == pytest.approx(-NERNST_SLOPE_25C, abs=1e-4)


class TestWaterWindow:
    def test_her_passes_through_origin(self):
        assert her_line(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_her_slope_is_59_mV_per_pH(self):
        assert (her_line(14.0) - her_line(0.0)) / 14.0 == pytest.approx(
            -NERNST_SLOPE_25C, abs=1e-4
        )

    def test_oer_standard_potential_at_pH_zero(self):
        assert oer_line(0.0) == pytest.approx(1.229, abs=1e-3)

    def test_water_window_is_1_23_V_at_every_pH(self):
        """OER − HER = 1.229 V independent of pH; both lines share a slope."""
        pH = np.linspace(0, 14, 15)
        assert np.allclose(oer_line(pH) - her_line(pH), 1.229, atol=1e-3)

    def test_higher_temperature_steepens_the_her_slope(self):
        cold = her_line(10.0, T=298.15)
        hot = her_line(10.0, T=363.15)
        assert hot < cold

    def test_hydrogen_partial_pressure_shifts_her(self):
        assert her_line(7.0, p_H2=0.1) > her_line(7.0, p_H2=1.0)


class TestFePourbaix:
    def test_defaults_construct(self):
        p = FePourbaix()
        assert p.T == pytest.approx(298.15, abs=0.1)

    def test_log_activity_matches_activity(self):
        assert FePourbaix(activity=1e-6).log_a == pytest.approx(-6.0)

    def test_deposition_potential_is_negative(self):
        """Iron is less noble than hydrogen; E_Fe/Fe²⁺ ≈ −0.44 V vs SHE."""
        assert FePourbaix(activity=1.0).deposition_potential(2.0) < 0.0

    def test_deposition_potential_near_standard_at_unit_activity(self):
        assert FePourbaix(activity=1.0).deposition_potential(1.0) == pytest.approx(
            E0_FE2_FE, abs=0.02
        )

    def test_lower_activity_makes_deposition_harder(self):
        dilute = FePourbaix(activity=1e-6).deposition_potential(2.0)
        conc = FePourbaix(activity=1.0).deposition_potential(2.0)
        assert dilute < conc

    def test_her_margin_is_positive_at_every_pH(self):
        """The core thermodynamic obstacle: HER is favoured over Fe everywhere,
        so iron deposition always carries an HER penalty."""
        p = FePourbaix(activity=1.0, temperature_C=60.0)
        for pH in [0, 2, 4, 7, 10, 12, 14]:
            assert p.her_margin(pH) > 0

    def test_her_margin_narrows_toward_alkaline(self):
        """README claim: ~440 mV penalty in strong acid, ~47 mV in alkali.
        This is precisely why alkaline routes are attractive."""
        p = FePourbaix(activity=1.0, temperature_C=60.0)
        assert p.her_margin(0.0) > p.her_margin(14.0)
        assert p.her_margin(0.0) == pytest.approx(0.44, abs=0.06)
        assert p.her_margin(14.0) < 0.15

    def test_hydrolysis_boundaries_ordered_and_in_range(self):
        p = FePourbaix(activity=1.0)
        assert 0.0 < p.pH_Fe3_FeOH3 < p.pH_Fe2_FeOH2 < 14.0

    def test_ferric_hydrolyzes_before_ferrous(self):
        """Fe³⁺ precipitates at pH 3-4 while Fe²⁺ stays soluble — the basis of
        the hydrolysis purification stage."""
        p = FePourbaix(activity=1.0)
        assert p.pH_Fe3_FeOH3 < 4.5
        assert p.pH_Fe2_FeOH2 > 5.0

    def test_more_concentrated_iron_precipitates_sooner(self):
        assert FePourbaix(activity=1.0).pH_Fe2_FeOH2 < FePourbaix(
            activity=1e-6
        ).pH_Fe2_FeOH2

    def test_deposition_potential_is_continuous_across_boundaries(self):
        """The branch switch between Fe²⁺, Fe(OH)₂ and HFeO₂⁻ domains must not
        introduce a discontinuity large enough to be unphysical."""
        p = FePourbaix(activity=1.0)
        pH = np.linspace(0.5, 15.0, 300)
        E = np.array([p.deposition_potential(x) for x in pH])
        assert np.max(np.abs(np.diff(E))) < 0.15

    def test_dominant_species_identifies_metal_below_deposition(self):
        p = FePourbaix(activity=1.0)
        assert p.dominant_species(2.0, p.deposition_potential(2.0) - 0.3) == "Fe(s)"

    def test_dominant_species_returns_a_label_across_the_diagram(self):
        p = FePourbaix(activity=1.0)
        for pH in [1.0, 7.0, 13.0]:
            for E in [-1.2, -0.2, 0.5, 1.2]:
                assert isinstance(p.dominant_species(pH, E), str)

    def test_temperature_shifts_the_margin(self):
        cold = FePourbaix(activity=1.0, temperature_C=25.0).her_margin(2.0)
        hot = FePourbaix(activity=1.0, temperature_C=80.0).her_margin(2.0)
        assert cold != hot


# ── Fe(OH)₂ solubility product temperature dependence ──────────────────────


def test_ksp_feoh2_recovers_25c_anchor():
    """At 25 °C the van 't Hoff form must return the tabulated Ksp."""
    from models.pourbaix import ksp_feoh2, LOGKSP_FEOH2
    ksp25 = ksp_feoh2(298.15)
    assert np.log10(ksp25) == pytest.approx(LOGKSP_FEOH2, abs=1e-9)


def test_ksp_feoh2_rises_with_temperature():
    """Fe(OH)₂ dissolution is endothermic: Ksp increases with T."""
    from models.pourbaix import ksp_feoh2
    cold = ksp_feoh2(298.15)
    hot = ksp_feoh2(333.15)   # 60 °C
    assert hot > cold
    # ~2.5× between 25 and 60 °C with ΔH ≈ 22 kJ/mol (screening value).
    assert 2.0 < hot / cold < 3.0


def test_diffusion_layer_uses_temperature_corrected_ksp():
    """The FE engine's film profile must carry the T-corrected Ksp."""
    from models.diffusion_layer_1d import DiffusionLayer1D, KSP_FEOH2

    cold = DiffusionLayer1D(temperature_C=25.0)
    hot = DiffusionLayer1D(temperature_C=60.0)
    assert cold.Ksp == pytest.approx(KSP_FEOH2, rel=1e-6)
    assert hot.Ksp > cold.Ksp
    r = hot.solve(50.0)
    assert r.profile.Ksp_FeOH2 == pytest.approx(hot.Ksp, rel=1e-9)
