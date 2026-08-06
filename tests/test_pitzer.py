"""Tests for the Pitzer ion-interaction activity model (models/pitzer.py).

Validation anchors are published literature values (see module docstring):
  * NaCl γ±: the canonical 1–1 electrolyte (Harvie et al. 1984 / Robinson &
    Stokes tabulation) — validates the full multicomponent machinery.
  * Na2SO4 γ±/φ: 1–2 electrolyte (PHREEQC pitzer.dat set).
  * FeSO4 γ±(0.1 m) = 0.164 (Kobylin et al. 2011 assessment;
    Reardon & Beckie 1987 used 0.161) — the bath-relevant 2–2 electrolyte.
  * MgSO4: independent 2–2 sulfate cross-check (Pitzer 1991 tabulation).
"""

import math

import pytest

from models.pitzer import (
    A_phi,
    PITZER_BINARY,
    PITZER_T_REF_K,
    PitzerPair,
    mean_activity_coefficient_pure,
    solve_pitzer,
)

# ─── Debye–Hückel slope ──────────────────────────────────────────────


def test_A_phi_25C_matches_literature():
    assert A_phi(25.0) == pytest.approx(0.3915, abs=0.002)


def test_A_phi_increases_with_temperature():
    # dielectric constant falls with T → stronger electrostatics
    assert A_phi(60.0) > A_phi(25.0)
    assert A_phi(90.0) > A_phi(60.0)


# ─── Canonical pure-electrolyte anchors ──────────────────────────────


@pytest.mark.parametrize(
    "m, expected",
    [(0.001, 0.965), (0.01, 0.903), (0.1, 0.778), (0.5, 0.681), (1.0, 0.657)],
)
def test_nacl_mean_gamma(m, expected):
    """NaCl γ± — the standard implementation check, ±1 %."""
    assert mean_activity_coefficient_pure("Na+", "Cl-", m) == pytest.approx(expected, rel=0.01)


def test_feso4_gamma_anchor_kobylin():
    """γ±(FeSO4, 0.1 m) ≈ 0.164 (Kobylin 2011) / 0.161 (Reardon & Beckie 1987)."""
    g = mean_activity_coefficient_pure("Fe2+", "SO4-2", 0.1)
    assert 0.14 < g < 0.18


def test_feso4_gamma_concentrated():
    """γ±(FeSO4, 1 m) ≈ 0.04–0.05 (Reardon & Beckie); 2–2 electrolytes dive."""
    g = mean_activity_coefficient_pure("Fe2+", "SO4-2", 1.0)
    assert 0.03 < g < 0.07


def test_feso4_gamma_dilute_limit_and_minimum():
    """γ± rises toward 1 in the dilute limit and passes through the
    characteristic 2–2-sulfate minimum before rising near saturation."""
    g_dilute = mean_activity_coefficient_pure("Fe2+", "SO4-2", 0.001)
    g_01 = mean_activity_coefficient_pure("Fe2+", "SO4-2", 0.1)
    g_1 = mean_activity_coefficient_pure("Fe2+", "SO4-2", 1.0)
    g_sat = mean_activity_coefficient_pure("Fe2+", "SO4-2", 3.58)  # copperas saturation
    assert g_dilute > g_01 > 0.04
    assert g_1 < g_01
    assert g_dilute > 0.7
    # minimum near ~2 m, upturn at the solubility limit (Kobylin 2011 behaviour)
    assert g_sat > g_1


def test_mgso4_gamma_crosscheck():
    """MgSO4 0.1 m: published γ± ≈ 0.163 (Robinson & Stokes family)."""
    assert 0.14 < mean_activity_coefficient_pure("Mg2+", "SO4-2", 0.1) < 0.19


def test_na2so4_gamma_and_osmotic():
    """Na2SO4: γ±(1 m) ≈ 0.204; osmotic coefficient φ(1 m) ≈ 0.66."""
    g = mean_activity_coefficient_pure("Na+", "SO4-2", 1.0)
    assert 0.17 < g < 0.24
    sol = solve_pitzer({"Na+": 2.0, "SO4-2": 1.0})
    assert 0.60 < sol.osmotic_coefficient < 0.72
    assert 0.95 < sol.water_activity < 0.99


# ─── Mixture behaviour ───────────────────────────────────────────────


def test_bath_mixture_gamma_fe_honest_small():
    """At the reference-bath molality the Fe2+ gamma must be the honest
    small 2–2-salt value, not the physically wrong Davies number (≈0.68)
    that produced phantom 97 % ion pairing."""
    sol = solve_pitzer({"Fe2+": 1.0, "Na+": 1.0, "H+": 0.02, "SO4-2": 1.5, "HSO4-": 0.02}, T_C=50.0)
    assert sol.gamma["Fe2+"] < 0.2
    assert sol.gamma["Na+"] > sol.gamma["Fe2+"]  # charge scaling preserved
    assert 0.0 < sol.gamma["SO4-2"] < 1.0
    assert sol.water_activity < 1.0


def test_trace_species_get_gamma():
    """Zero-molality species are reported (trace gamma), not dropped."""
    sol = solve_pitzer({"Fe2+": 1.0, "Na+": 0.0, "H+": 0.0,
                        "SO4-2": 1.0, "HSO4-": 0.0})
    assert "Na+" in sol.gamma
    assert "HSO4-" in sol.gamma
    assert sol.activity["Na+"] == 0.0


def test_nonelectroneutral_input_warns():
    with pytest.warns(UserWarning):
        solve_pitzer({"Fe2+": 1.0, "Na+": 3.0, "SO4-2": 0.1})


def test_2minus2_convention_alpha1():
    """2–2 electrolytes use the Pitzer α1 = 1.4 convention."""
    assert PITZER_BINARY[("Fe2+", "SO4-2")].alpha1 == 1.4
    assert PITZER_BINARY[("Fe2+", "SO4-2")].alpha2 == 12.0
    assert PITZER_BINARY[("Fe2+", "SO4-2")].beta2 < 0.0  # association slope


# ─── Temperature-dependence framework (2026-08) ─────────────────────
#
# The shipped binary tables carry all-zero T-coefficients (frozen 25 °C
# set).  These tests pin (a) that frozen tables are byte-identical under
# at_T at any T, (b) the EQ3/6-Sandia polynomial form, and (c) that the
# solver actually routes pairs through at_T.


def test_shipped_tables_are_frozen_and_byte_identical():
    """Every shipped pair must still ship frozen parameters: at_T is the
    identity object at any T, so default results cannot drift."""
    for key, pair in PITZER_BINARY.items():
        assert all(c == 0.0 for row in pair.t_coeffs for c in row), key
        for T_C in (5.0, 25.0, 50.0, 90.0):
            assert pair.at_T(T_C) is pair, key


def test_at_T_polynomial_form_and_anchor():
    """p(T) = a + c1(1/T − 1/Tr) + c2 ln(T/Tr) + c3(T − Tr), Tr = 298.15 K;
    at the anchor T, p(Tr) = a exactly."""
    row0 = (0.10, 5.0e2, -2.0e-2, 1.0e-5)
    pair = PitzerPair(
        0.05, 0.20, 0.0, 0.01,
        t_coeffs=(row0, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
    )
    anchored = pair.at_T(25.0)
    assert anchored.beta0 == pytest.approx(row0[0])  # a supersedes base at Tr
    T = 40.0 + 273.15
    Tr = PITZER_T_REF_K
    a, c1, c2, c3 = row0
    expect = a + c1 * (1.0 / T - 1.0 / Tr) + c2 * math.log(T / Tr) + c3 * (T - Tr)
    assert pair.at_T(40.0).beta0 == pytest.approx(expect, rel=1e-12)
    # zero rows fall back to the base field
    assert pair.at_T(40.0).beta1 == pytest.approx(0.20)
    assert pair.at_T(40.0).beta2 == pytest.approx(0.0)
    assert pair.at_T(40.0).Cphi == pytest.approx(0.01)
    # alpha parameters belong to the functional form — never T-evolved
    assert pair.at_T(40.0).alpha1 == pair.alpha1


def test_at_T_returns_new_object_and_leaves_original():
    pair = PitzerPair(0.05, 0.20, 0.0, 0.01,
                      t_coeffs=((0.10, 1.0e3, 0.0, 0.0),) + ((0.0,) * 4,) * 3)
    evolved = pair.at_T(60.0)
    assert evolved is not pair
    assert pair.beta0 == pytest.approx(0.05)      # original untouched
    assert evolved.beta0 != pytest.approx(0.05)   # copy carries the shift


def test_solve_pitzer_routes_pairs_through_at_T():
    """Swap in a T-evolving Na+/Cl- pair and confirm γ±(T) actually moves;
    at the anchor temperature it must coincide with the frozen result."""
    frozen = PITZER_BINARY[("Na+", "Cl-")]
    evolving = PitzerPair(
        frozen.beta0, frozen.beta1, frozen.beta2, frozen.Cphi,
        ref=frozen.ref,
        t_coeffs=((frozen.beta0, 3.0e2, 0.0, 0.0),) + ((0.0,) * 4,) * 3,
    )
    try:
        PITZER_BINARY[("Na+", "Cl-")] = evolving
        g25 = mean_activity_coefficient_pure("Na+", "Cl-", 1.0, T_C=25.0)
        g50 = mean_activity_coefficient_pure("Na+", "Cl-", 1.0, T_C=50.0)
        g90 = mean_activity_coefficient_pure("Na+", "Cl-", 1.0, T_C=90.0)
    finally:
        PITZER_BINARY[("Na+", "Cl-")] = frozen
    frozen25 = mean_activity_coefficient_pure("Na+", "Cl-", 1.0, T_C=25.0)
    frozen50 = mean_activity_coefficient_pure("Na+", "Cl-", 1.0, T_C=50.0)
    frozen90 = mean_activity_coefficient_pure("Na+", "Cl-", 1.0, T_C=90.0)
    # Anchor invariance: at Tr the evolved pair IS the frozen pair.
    assert g25 == pytest.approx(frozen25, rel=1e-12)
    assert frozen25 == pytest.approx(0.6540, abs=0.002)  # NaCl literature anchor
    assert g50 != pytest.approx(frozen50, rel=1e-6)
    assert g90 != pytest.approx(frozen90, rel=1e-6)


def test_extreme_T_still_solves():
    """Outside the 10–60 °C window the solver still evaluates (flagging is
    the caller's job); the frozen tables keep the numbers well-defined."""
    sol = solve_pitzer({"Fe2+": 1.0, "SO4-2": 1.0}, T_C=95.0)
    assert 0.0 < sol.gamma["Fe2+"] < 1.0
    assert 0.0 < sol.water_activity < 1.0
