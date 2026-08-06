"""Tests for the Pitzer ion-interaction activity model (models/pitzer.py).

Validation anchors are published literature values (see module docstring):
  * NaCl γ±: the canonical 1–1 electrolyte (Harvie et al. 1984 / Robinson &
    Stokes tabulation) — validates the full multicomponent machinery.
  * Na2SO4 γ±/φ: 1–2 electrolyte (PHREEQC pitzer.dat set).
  * FeSO4 γ±(0.1 m) = 0.164 (Kobylin et al. 2011 assessment;
    Reardon & Beckie 1987 used 0.161) — the bath-relevant 2–2 electrolyte.
  * MgSO4: independent 2–2 sulfate cross-check (Pitzer 1991 tabulation).
"""

import pytest

from models.pitzer import (
    A_phi,
    PITZER_BINARY,
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
