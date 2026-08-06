"""Assertions for the full Butler–Volmer branches (2026-08 addition).
The repo's kinetics used to be Tafel-only, which has two artefacts:
i(E_eq) = i0 ≠ 0 and no representation of dissolution anodic of E_eq.
These tests pin the BV corrections without expecting any change at
operating overpotentials (the reverse term is 10^-3..10^-8 there).
"""
from math import isfinite
import numpy as np
import pytest
from models.kinetics import (
    FE_ANODIC_SLOPE_V,
    HER_ANODIC_SLOPE_V,
    ButlerVolmerBranch,
    DepositionKinetics,
    TafelBranch,
)
from models.transport import NernstPlanckFilm

FE_I0 = 1.0e-2
FE_E_EQ = -0.440


@pytest.fixture()
def fe_bv():
    return ButlerVolmerBranch(
        FE_I0, 0.120, FE_E_EQ, i_lim=None, anodic_slope_V=FE_ANODIC_SLOPE_V
    )


def test_current_is_zero_at_equilibrium(fe_bv):
    """The defining BV property absent in Tafel-only form."""
    assert fe_bv.current(FE_E_EQ) == pytest.approx(0.0, abs=1e-12)
    her = ButlerVolmerBranch(1e-6, 0.140, -0.12, None, HER_ANODIC_SLOPE_V)
    assert her.current(-0.12) == pytest.approx(0.0, abs=1e-12)


def test_cathodic_side_recovers_tafel_limit(fe_bv):
    """At |η| ≥ 150 mV the reverse term is ≲1e-4 of the forward one."""
    tafel = TafelBranch(FE_I0, 0.120, FE_E_EQ)
    for eta in (0.15, 0.25, 0.40, 0.70):
        E = FE_E_EQ - eta
        i_bv = fe_bv.current(E)
        i_tf = tafel.current(E)
        assert abs(i_bv - i_tf) / i_tf < 1e-4, f"η={eta}: {i_bv} vs {i_tf}"


def test_anodic_side_is_signed_and_grows(fe_bv):
    """Anodic of E_eq the branch is net-oxidation (Fe dissolves)."""
    i1 = fe_bv.current(FE_E_EQ + 0.10)
    i2 = fe_bv.current(FE_E_EQ + 0.20)
    assert i1 < 0.0 and i2 < 0.0
    # One decade per anodic slope unit: ~0.0392 V/dec.
    ratio = abs(i2) / abs(i1)
    assert ratio == pytest.approx(10.0 ** (0.10 / FE_ANODIC_SLOPE_V), rel=0.05)


def test_koutecky_levich_only_caps_the_cathodic_arm():
    """i_lim must blend the cathodic arm and leave dissolution alone."""
    b = ButlerVolmerBranch(1.0, 0.120, FE_E_EQ, i_lim=100.0,
                           anodic_slope_V=FE_ANODIC_SLOPE_V)
    i_cat = b.current(FE_E_EQ - 0.5)
    assert i_cat < 100.0  # capped by transport
    i_an = b.current(FE_E_EQ + 0.15)
    i_an_no_lim = ButlerVolmerBranch(
        1.0, 0.120, FE_E_EQ, None, FE_ANODIC_SLOPE_V
    ).current(FE_E_EQ + 0.15)
    assert i_an == pytest.approx(i_an_no_lim)


def test_anodic_slope_bookkeeping_values():
    """α_a·n = n − α_c·n with α_c·n read from the 25 °C cathodic slope."""
    from models.electrochemistry import FARADAY as F_REPO, R_GAS as R_REPO
    b25 = 2.303 * R_REPO * 298.15 / F_REPO
    assert FE_ANODIC_SLOPE_V == pytest.approx(
        b25 / (2.0 - b25 / 0.120), rel=1e-9
    )
    assert HER_ANODIC_SLOPE_V == pytest.approx(
        b25 / (1.0 - b25 / 0.140), rel=1e-9
    )


def test_deposition_kinetics_matches_tafel_only_at_operating_points():
    """Galvanostatic answers must not move at screening precision."""
    k_bv = DepositionKinetics(temperature_C=50.0)
    k_tf = DepositionKinetics(temperature_C=50.0, use_butler_volmer=False)
    for j in (10.0, 50.0, 100.0, 200.0, 400.0):
        assert k_bv.efficiency_at_current(j) == pytest.approx(
            k_tf.efficiency_at_current(j), abs=1e-4
        )
        assert k_bv.potential_at_current(j) == pytest.approx(
            k_tf.potential_at_current(j), abs=1e-4
        )


def test_deposition_kinetics_fe_branch_zero_at_equilibrium():
    k = DepositionKinetics(temperature_C=50.0)
    i_fe, i_h, i_tot = k.partial_currents(k.fe_E_eq)
    assert float(i_fe) == pytest.approx(0.0, abs=1e-9)
    assert float(i_h) > 0.0  # HER still runs cathodically at E_eq(Fe)
    assert isfinite(float(i_tot))


def test_polarization_curve_shows_dissolution_anodic_of_Eeq():
    k = DepositionKinetics(temperature_C=50.0)
    E = np.linspace(-0.50, -0.30, 41)
    _, i_fe, _, i_tot, _ = k.polarization_curve(E)
    assert float(i_fe[0]) > 0.0
    assert float(i_fe[-1]) < 0.0  # Fe dissolving at the anodic end


@pytest.mark.parametrize("j", [25.0, 100.0, 250.0])
def test_nernst_planck_solve_is_tafel_compatible(j):
    """Film-model galvanostatic solve: BV must reproduce Tafel answers."""
    film = NernstPlanckFilm(temperature_C=50.0)
    state = film.solve(j)
    assert state.converged
    assert 0.9 < state.current_efficiency <= 1.0
    assert state.applied_current_A_m2 == pytest.approx(j * 10.0, rel=1e-4)
