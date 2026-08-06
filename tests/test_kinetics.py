"""Tests for Butler-Volmer Fe/HER competition kinetics.

Current efficiency is the single number this program lives or dies by, and
``kinetics.py`` is where it is computed.  These tests pin the Tafel and Levich
arithmetic against closed forms and verify the four qualitative claims the
README's results table rests on:

* an active cathode in acid gives terrible FE,
* suppressing HER exchange current is the dominant lever,
* transport limits cap the iron branch, and
* agitation (thinner δ) recovers efficiency.
"""

import numpy as np
import pytest

from models.electrochemistry import FARADAY, Z_FE
from models.kinetics import (
    DepositionKinetics,
    TafelBranch,
    limiting_current_density,
)


class TestLimitingCurrentDensity:
    def test_matches_levich_form(self):
        """i_lim = zFDC/δ."""
        C, D, delta = 1000.0, 7.2e-10, 5e-5
        expected = Z_FE * FARADAY * D * C / delta
        assert limiting_current_density(C, D, delta) == pytest.approx(expected)

    def test_linear_in_concentration(self):
        assert limiting_current_density(2000.0) == pytest.approx(
            2.0 * limiting_current_density(1000.0)
        )

    def test_inverse_in_boundary_layer(self):
        """Agitation thins δ and raises the transport ceiling."""
        thick = limiting_current_density(1000.0, boundary_layer_m=200e-6)
        thin = limiting_current_density(1000.0, boundary_layer_m=20e-6)
        assert thin == pytest.approx(10.0 * thick)

    def test_zero_concentration_gives_no_current(self):
        assert limiting_current_density(0.0) == 0.0


class TestTafelBranch:
    def test_current_equals_i0_at_equilibrium(self):
        b = TafelBranch(i0=1e-2, tafel_slope_V=0.12, E_eq=-0.44)
        assert float(b.current(-0.44)) == pytest.approx(1e-2)

    def test_decade_per_tafel_slope(self):
        """One Tafel slope of extra overpotential must give 10x the current."""
        b = TafelBranch(i0=1e-2, tafel_slope_V=0.12, E_eq=-0.44)
        i1 = float(b.current(-0.44 - 0.12))
        i2 = float(b.current(-0.44 - 0.24))
        assert i1 == pytest.approx(1e-1, rel=1e-6)
        assert i2 == pytest.approx(1.0, rel=1e-6)

    def test_current_increases_with_cathodic_overpotential(self):
        b = TafelBranch(i0=1e-3, tafel_slope_V=0.12, E_eq=-0.44)
        E = np.linspace(-1.2, -0.44, 25)
        i = b.current(E)
        assert np.all(np.diff(i) < 0)  # less negative E → lower current

    def test_limiting_current_caps_the_branch(self):
        b = TafelBranch(i0=1e-2, tafel_slope_V=0.12, E_eq=-0.44, i_lim=100.0)
        assert float(b.current(-3.0)) <= 100.0

    def test_koutecky_levich_mixing(self):
        """1/i = 1/i_kin + 1/i_lim."""
        b = TafelBranch(i0=1e-2, tafel_slope_V=0.12, E_eq=-0.44, i_lim=100.0)
        E = -0.44 - 0.48        # 4 decades → i_kin = 100 A/m²
        assert float(b.current(E)) == pytest.approx(50.0, rel=1e-3)

    def test_unlimited_branch_exceeds_limited_branch(self):
        free = TafelBranch(1e-2, 0.12, -0.44)
        capped = TafelBranch(1e-2, 0.12, -0.44, i_lim=50.0)
        assert float(free.current(-1.5)) > float(capped.current(-1.5))


class TestDepositionKinetics:
    def test_defaults_construct_and_expose_branches(self):
        """At the kinetics reference temperature (50 °C) the branches expose
        the configured i0; elsewhere they are Arrhenius-scaled."""
        k = DepositionKinetics(temperature_C=50.0)
        assert k.fe_branch.i0 == pytest.approx(k.fe_i0)
        assert k.her_branch.i0 == pytest.approx(k.her_i0)

    def test_exchange_currents_are_arrhenius_scaled(self):
        """2026-08 physics change: i0 now carries an Arrhenius temperature
        dependence (previously constant).  HER (Ea≈60 kJ/mol) grows faster
        than Fe deposition (Ea≈50 kJ/mol) with temperature."""
        cold = DepositionKinetics(temperature_C=30.0)
        hot = DepositionKinetics(temperature_C=70.0)
        assert cold.fe_branch.i0 < cold.fe_i0 < hot.fe_branch.i0
        assert cold.her_branch.i0 < cold.her_i0 < hot.her_branch.i0
        ratio_fe = hot.fe_branch.i0 / cold.fe_branch.i0
        ratio_her = hot.her_branch.i0 / cold.her_branch.i0
        assert ratio_her > ratio_fe  # HER Ea is larger

    def test_current_efficiency_temperature_trend_is_physical(self):
        """With HER Ea > Fe Ea, galvanostatic FE falls with temperature at
        fixed current density — the direction long known for Zn/Fe
        electrowinning, where CE peaks at moderate temperature because the
        hydrogen branch is more strongly activated than metal deposition."""
        cold = DepositionKinetics(temperature_C=30.0, her_i0=1e-4)
        hot = DepositionKinetics(temperature_C=70.0, her_i0=1e-4)
        assert hot.efficiency_at_current(100.0) < cold.efficiency_at_current(100.0)

    def test_temperature_property(self):
        assert DepositionKinetics(temperature_C=60.0).T == pytest.approx(333.15)

    def test_her_equilibrium_tracks_pH(self):
        """More alkaline → more negative HER equilibrium potential."""
        acid = DepositionKinetics(pH=1.0).her_branch.E_eq
        base = DepositionKinetics(pH=13.0).her_branch.E_eq
        assert base < acid

    def test_partial_currents_sum_to_total(self):
        k = DepositionKinetics()
        i_fe, i_h, i_tot = k.partial_currents(-0.8)
        assert i_tot == pytest.approx(i_fe + i_h)

    def test_current_efficiency_is_a_fraction(self):
        k = DepositionKinetics()
        for E in [-1.4, -1.0, -0.7, -0.5]:
            ce = float(k.current_efficiency(E))
            assert 0.0 <= ce <= 1.0

    def test_potential_at_current_inverts_partial_currents(self):
        k = DepositionKinetics()
        E = k.potential_at_current(100.0)
        assert k.partial_currents(E)[2] == pytest.approx(1000.0, rel=1e-6)

    def test_active_cathode_in_acid_has_poor_efficiency(self):
        """README case 1: i0,H = 1e-2 A/m² gives single-digit FE."""
        k = DepositionKinetics(pH=2.0, temperature_C=60.0, her_i0=1e-2)
        assert k.efficiency_at_current(100.0) < 0.30

    def test_her_suppression_is_the_dominant_lever(self):
        """README case 2: dropping i0,H to 1e-5 lifts FE above 90%."""
        active = DepositionKinetics(pH=2.0, temperature_C=60.0, her_i0=1e-2)
        blocked = DepositionKinetics(pH=2.0, temperature_C=60.0, her_i0=1e-5)
        assert blocked.efficiency_at_current(100.0) > 0.90
        assert blocked.efficiency_at_current(100.0) > active.efficiency_at_current(100.0)

    def test_agitation_recovers_efficiency_in_a_dilute_bath(self):
        """README cases 4-5: same bath, δ 200 µm → 20 µm, FE rises sharply."""
        stagnant = DepositionKinetics(
            pH=2.0, temperature_C=60.0, fe_conc_M=0.1,
            her_i0=1e-3, boundary_layer_m=200e-6,
        )
        stirred = DepositionKinetics(
            pH=2.0, temperature_C=60.0, fe_conc_M=0.1,
            her_i0=1e-3, boundary_layer_m=20e-6,
        )
        assert stirred.efficiency_at_current(100.0) > stagnant.efficiency_at_current(100.0)

    def test_iron_branch_cannot_exceed_its_transport_limit(self):
        k = DepositionKinetics(fe_conc_M=0.1, boundary_layer_m=200e-6)
        i_fe, _, _ = k.partial_currents(-2.0)
        assert float(i_fe) <= k.i_lim * 1.001

    def test_efficiency_falls_as_current_passes_the_transport_limit(self):
        """Past i_lim the extra current has nowhere to go but hydrogen."""
        k = DepositionKinetics(fe_conc_M=0.1, boundary_layer_m=100e-6, her_i0=1e-4)
        j_lim_mA = k.i_lim / 10.0
        below = k.efficiency_at_current(max(j_lim_mA * 0.3, 1.0))
        above = k.efficiency_at_current(j_lim_mA * 3.0)
        assert above < below

    def test_richer_bath_supports_higher_efficiency(self):
        lean = DepositionKinetics(fe_conc_M=0.05, her_i0=1e-4)
        rich = DepositionKinetics(fe_conc_M=2.0, her_i0=1e-4)
        assert rich.efficiency_at_current(200.0) > lean.efficiency_at_current(200.0)

    def test_polarization_curve_shapes(self):
        k = DepositionKinetics()
        E, i_fe, i_h, i_tot, ce = k.polarization_curve()
        assert len(E) == len(i_fe) == len(i_h) == len(i_tot) == len(ce)
        assert np.all(i_tot >= i_fe - 1e-12)
        # 2026-08: with full Butler–Volmer branches the sweep tail runs
        # anodic of E_eq(Fe), where Fe dissolves (i_fe < 0) and CE as a
        # ratio is undefined; the [0,1] bound holds wherever both partial
        # currents are cathodic (the galvanostatic CE regime).
        both_cathodic = (i_fe > 0.0) & (i_h > 0.0)
        assert both_cathodic.any()
        assert np.all((ce[both_cathodic] >= 0.0) & (ce[both_cathodic] <= 1.0))

    def test_efficiency_sweep_returns_paired_arrays(self):
        js, ces = DepositionKinetics().efficiency_sweep([10.0, 50.0, 100.0])
        assert len(js) == len(ces) == 3
        assert np.all((ces >= 0) & (ces <= 1))

    def test_deposition_rate_positive_and_monotonic(self):
        k = DepositionKinetics(her_i0=1e-5)
        assert k.deposition_rate_um_hr(50.0) > 0
        assert k.deposition_rate_um_hr(200.0) > k.deposition_rate_um_hr(50.0)

    def test_deposition_rate_matches_faraday(self):
        """At near-unit FE the rate must equal jM/(zFρ)."""
        k = DepositionKinetics(her_i0=1e-9, fe_conc_M=2.0, boundary_layer_m=20e-6)
        j = 100.0
        ce = k.efficiency_at_current(j)
        expected = (j * 10.0) * ce * 55.845e-3 / (2 * FARADAY) / 7874.0 * 3600e6
        assert k.deposition_rate_um_hr(j) == pytest.approx(expected, rel=1e-6)

    def test_hydrogen_flux_falls_when_her_is_suppressed(self):
        active = DepositionKinetics(her_i0=1e-2).hydrogen_flux_mol_m2_hr(100.0)
        blocked = DepositionKinetics(her_i0=1e-6).hydrogen_flux_mol_m2_hr(100.0)
        assert blocked < active

    def test_summary_reports_expected_keys(self):
        s = DepositionKinetics().summary(100.0)
        for key in (
            "j applied (mA/cm²)",
            "Current efficiency (%)",
            "Deposition rate (µm/hr)",
            "i_lim,Fe (A/m²)",
        ):
            assert key in s

    def test_summary_efficiency_is_a_percentage(self):
        v = DepositionKinetics().summary(100.0)["Current efficiency (%)"]
        assert 0.0 <= v <= 100.0
