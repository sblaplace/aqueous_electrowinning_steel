"""
Tests for the Fe(OH)₂ passivation-film thickness ODE (models/feoh2_film.py).

CHEM_PHYS_REVIEW Tier 1.3: the Nernst–Planck solve omits the passivation film.
This module turns the ``precipitation_sink`` flux into a real coupled film:
growth from precipitation minus acid + Fe²⁺-promoted reductive dissolution,
feeding an ohmic surface overpotential (10 s of mV).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.feoh2_film import (
    FILM_DEPOSITION_FRACTION,
    FEOH2_KAPPA_S_M,
    acid_dissolution_m_s,
    film_diagnostics,
    film_growth_rate_m_s,
    film_ode,
    film_overpotential_V,
    integrate_film,
    reductive_dissolution_m_s,
    steady_state_thickness_m,
)


# ─── Growth / dissolution mechanics ─────────────────────────────────────

def test_growth_rate_linear_in_flux():
    """Growth scales linearly with precipitation flux and deposition fraction."""
    g1 = film_growth_rate_m_s(1e-2)
    g2 = film_growth_rate_m_s(2e-2)
    assert g2 == pytest.approx(2 * g1, rel=1e-9)
    # Raising the deposition fraction raises growth commensurately.
    g_hi = film_growth_rate_m_s(1e-2, deposition_fraction=2 * FILM_DEPOSITION_FRACTION)
    assert g_hi == pytest.approx(2 * g1, rel=1e-9)
    # All-zero when there is no precipitation.
    assert film_growth_rate_m_s(0.0) == 0.0


def test_dissolution_first_order_in_thickness_and_species():
    """Both removal routes are linear in δ and in their driving species."""
    delta = 1e-7
    # Acid term rises with c_H; zero when no H⁺.
    assert acid_dissolution_m_s(delta, c_h_surf_mol_m3=0.0) == 0.0
    a_lo = acid_dissolution_m_s(delta, c_h_surf_mol_m3=50.0)
    a_hi = acid_dissolution_m_s(delta, c_h_surf_mol_m3=100.0)
    assert a_hi == pytest.approx(2 * a_lo, rel=1e-9)
    # Reductive term rises with c_Fe2; zero when no Fe²⁺.
    assert reductive_dissolution_m_s(delta, c_fe2_surf_mol_m3=0.0) == 0.0
    r_lo = reductive_dissolution_m_s(delta, c_fe2_surf_mol_m3=500.0)
    r_hi = reductive_dissolution_m_s(delta, c_fe2_surf_mol_m3=1000.0)
    assert r_hi == pytest.approx(2 * r_lo, rel=1e-9)
    # Both are zero on a zero-thickness film.
    assert acid_dissolution_m_s(0.0, c_h_surf_mol_m3=100.0) == 0.0
    assert reductive_dissolution_m_s(0.0, c_fe2_surf_mol_m3=1000.0) == 0.0


def test_ode_is_growth_minus_dissolution():
    """dδ/dt = growth − (acid + reductive) dissolution."""
    precip = 1e-3
    c_fe2 = 1000.0
    c_h = 1e-3
    delta = 1e-7
    expected = film_growth_rate_m_s(precip) - (
        acid_dissolution_m_s(delta, c_h)
        + reductive_dissolution_m_s(delta, c_fe2)
    )
    assert film_ode(0.0, delta, precip, c_h, c_fe2) == pytest.approx(expected, rel=1e-9)


# ─── Steady-state thickness & the 10 s-of-mV overpotential ────────────

def test_steady_state_thickness_plausible_and_finite():
    """At a reference precipitating flux the film is sub-µm with a small η."""
    # Reference passivating point (Fe²⁺-rich neutral surface: reductive
    # dissolution dominates, acid negligible).
    precip = 0.12          # mol/m²/s — ~ the pH-6 precipitating sink
    c_fe2 = 1000.0         # mol/m³
    c_h = 1e-3             # mol/m³ (neutral surface)
    delta = steady_state_thickness_m(precip, c_h, c_fe2)
    # Screening target: 0.1–1 µm coherent film (bulk sludge excluded).
    assert 0.05e-6 <= delta <= 5e-6, f"δ={delta:.3e} m out of plausible passivation range"

    # Overpotential at a realistic 100–300 mA/cm², in the 10 s-of-mV band.
    eta = film_overpotential_V(3000.0, delta, FEOH2_KAPPA_S_M)
    assert 0.005 <= eta <= 0.30, f"η_film={eta:.3f} V outside the 10 s-of-mV band"


def test_overpotential_ohm_law():
    """η_film = j·δ/κ — linear in j and δ, inverse in κ."""
    delta = 1e-7
    j = 3000.0
    eta = film_overpotential_V(j, delta, FEOH2_KAPPA_S_M)
    assert eta == pytest.approx(j * delta / FEOH2_KAPPA_S_M, rel=1e-9)
    # Doubling j doubles η.
    assert film_overpotential_V(2 * j, delta, FEOH2_KAPPA_S_M) == pytest.approx(2 * eta)
    # Zero current → zero overpotential.
    assert film_overpotential_V(0.0, delta) == pytest.approx(0.0)


def test_steady_state_zero_when_no_precipitation():
    assert steady_state_thickness_m(0.0, 1e-3, 1000.0) == 0.0


def test_steady_state_infinite_without_removal():
    """With no dissolution route the equilibrium is unbounded (honesty flag)."""
    # No H⁺ (acid off) and no Fe²⁺ (reductive off) on a precipitating flux.
    assert np.isinf(steady_state_thickness_m(1e-2, c_h_surf_mol_m3=0.0, c_fe2_surf_mol_m3=0.0))


# ─── film_diagnostics one-stop ─────────────────────────────────────────

def test_diagnostics_reports_reference_scenario():
    """film_diagnostics yields a coherent sub-µm film with 10 s-of-mV η."""
    diag = film_diagnostics(
        0.12,
        c_h_surf_mol_m3=1e-3,
        c_fe2_surf_mol_m3=1000.0,
        current_density_A_m2=3000.0,
    )
    assert 0.05e-6 <= diag.thickness_m <= 5e-6
    assert 0.005 <= diag.film_overpotential_V <= 0.30
    assert diag.growth_rate_m_s > 0.0
    assert diag.dissolution_rate_m_s > 0.0
    # Steady state: growth balances dissolution.
    assert diag.growth_rate_m_s == pytest.approx(diag.dissolution_rate_m_s, rel=1e-6)
    assert diag.flag == "unvalidated (L1)"


def test_diagnostics_zero_without_current():
    """Without a current the overpotential is 0 but the film chemistry stands."""
    diag = film_diagnostics(0.12, c_h_surf_mol_m3=1e-3, c_fe2_surf_mol_m3=1000.0)
    assert diag.film_overpotential_V == 0.0
    assert diag.thickness_m > 0.0


# ─── Transient integration ─────────────────────────────────────────────

def test_integrate_film_reaches_steady_state():
    """Time-stepping the ODE converges to the closed-form steady state."""
    precip, c_fe2, c_h = 0.12, 1000.0, 1e-3
    sol = integrate_film(
        lambda t: precip,
        lambda t: c_h,
        lambda t: c_fe2,
        thickness_0_m=0.0,
        t_span_s=(0.0, 3600.0),
    )
    assert sol.success
    final = float(sol.y[0, -1])
    ss = steady_state_thickness_m(precip, c_h, c_fe2)
    assert final == pytest.approx(ss, rel=0.05)
