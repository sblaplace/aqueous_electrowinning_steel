"""Tests for the steady 1-D Nernst-Planck cathode film model."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.electrochemistry import FARADAY  # noqa: E402
from models.transport import (  # noqa: E402
    KW_SI,
    NernstPlanckFilm,
    compare_support_levels,
)


# ─── Governing equations ──────────────────────────────────────────────
def test_electroneutrality_holds_across_the_film():
    film = NernstPlanckFilm(bulk_pH=3.0, support_conc_M=0.3)
    p = film.integrate(500.0, 100.0)
    charge = 2 * p.fe_M + p.h_M + p.na_M - p.oh_M - 2 * p.so4_M
    scale = np.max(2 * p.fe_M + p.na_M)
    assert np.max(np.abs(charge)) / scale < 1e-4


def test_water_equilibrium_is_enforced_pointwise():
    p = NernstPlanckFilm(bulk_pH=5.0).integrate(300.0, 80.0)
    # concentrations reported in mol/L; KW_SI is in (mol/m^3)^2
    assert np.allclose(p.h_M * p.oh_M * 1e6, KW_SI, rtol=1e-8)


def test_iron_flux_is_conserved_along_the_film():
    """Nernst-Planck flux -D dC/dx - 2 D (F/RT) C dphi/dx must be constant."""
    film = NernstPlanckFilm(bulk_pH=3.0, support_conc_M=0.3)
    i_fe = 500.0
    p = film.integrate(i_fe, 100.0)
    c = p.fe_M * 1000.0
    flux = -film.diffusivity_fe_m2_s * np.gradient(c, p.x_m) - (
        2.0 * film.diffusivity_fe_m2_s * film.f_RT * c * np.gradient(p.potential_V, p.x_m)
    )
    expected = -i_fe / (2.0 * FARADAY)
    assert np.allclose(flux, expected, rtol=1e-2)


def test_bulk_boundary_conditions_are_recovered():
    film = NernstPlanckFilm(bulk_pH=4.0, fe_conc_M=0.5, support_conc_M=0.2)
    p = film.integrate(200.0, 50.0)
    assert p.x_m[0] == pytest.approx(0.0)
    assert p.x_m[-1] == pytest.approx(film.boundary_layer_m)
    assert p.fe_M[-1] == pytest.approx(film.fe_conc_M, rel=1e-6)
    assert p.pH[-1] == pytest.approx(film.bulk_pH, abs=1e-6)
    assert p.potential_V[-1] == pytest.approx(0.0)


def test_zero_current_gives_a_flat_film():
    film = NernstPlanckFilm(bulk_pH=3.0, support_conc_M=0.5)
    p = film.integrate(0.0, 0.0)
    assert np.allclose(p.fe_M, film.fe_conc_M, rtol=1e-9)
    assert np.allclose(p.pH, film.bulk_pH, atol=1e-9)
    assert np.allclose(p.potential_V, 0.0, atol=1e-12)


# ─── Migration physics ────────────────────────────────────────────────
def test_unsupported_binary_electrolyte_doubles_the_limiting_current():
    """For a symmetric z:z binary salt the exact result is i_lim = 2 i_diff.

    Migration supplies exactly as much Fe2+ as diffusion when the only ions
    present are Fe2+ and SO4^2-, so the enhancement factor is 1 + z+/|z-| = 2.
    """
    film = NernstPlanckFilm(bulk_pH=7.0, fe_conc_M=1.0, support_conc_M=0.0)
    ratio = film.transport_limit_A_m2() / film.diffusion_limit_A_m2
    assert ratio == pytest.approx(2.0, rel=2e-3)


def test_supporting_electrolyte_collapses_migration():
    """Excess inert salt must recover the pure-diffusion (Levich) limit."""
    supported = NernstPlanckFilm(bulk_pH=7.0, support_conc_M=20.0)
    ratio = supported.transport_limit_A_m2() / supported.diffusion_limit_A_m2
    assert ratio == pytest.approx(1.0, abs=0.03)


def test_transport_limit_decreases_monotonically_with_support():
    ratios = []
    for c_s in (0.0, 0.5, 2.0, 10.0):
        film = NernstPlanckFilm(bulk_pH=7.0, support_conc_M=c_s)
        ratios.append(film.transport_limit_A_m2() / film.diffusion_limit_A_m2)
    assert all(a > b for a, b in zip(ratios, ratios[1:]))
    assert ratios[0] > 1.9


def test_transference_number_falls_with_supporting_electrolyte():
    bare = NernstPlanckFilm(bulk_pH=7.0, support_conc_M=0.0)
    salty = NernstPlanckFilm(bulk_pH=7.0, support_conc_M=5.0)
    assert bare.fe_transference_number > salty.fe_transference_number
    assert 0.0 < salty.fe_transference_number < 0.1


def test_migration_supplies_a_positive_share_of_the_iron_flux():
    state = NernstPlanckFilm(bulk_pH=3.0, her_i0=1e-4).solve(100.0)
    assert 0.0 < state.migration_flux_fraction < 1.0
    assert state.migration_enhancement > 1.0


def test_film_potential_drop_shrinks_with_supporting_electrolyte():
    bare = NernstPlanckFilm(bulk_pH=3.0, her_i0=1e-4).solve(100.0)
    salty = NernstPlanckFilm(
        bulk_pH=3.0, her_i0=1e-4, support_conc_M=5.0
    ).solve(100.0)
    assert abs(salty.film_potential_drop_V) < abs(bare.film_potential_drop_V)


# ─── Coupled kinetics ─────────────────────────────────────────────────
def test_solve_reproduces_the_applied_current():
    for j in (1.0, 20.0, 100.0):
        state = NernstPlanckFilm(her_i0=1e-4).solve(j)
        assert state.applied_current_A_m2 == pytest.approx(j * 10.0, rel=1e-3)
        assert state.converged


def test_partial_currents_sum_and_efficiency_is_bounded():
    state = NernstPlanckFilm(her_i0=1e-4).solve(100.0)
    total = state.fe_current_A_m2 + state.her_current_A_m2
    assert total == pytest.approx(state.applied_current_A_m2, rel=1e-9)
    assert 0.0 <= state.current_efficiency <= 1.0


def test_surface_pH_rises_above_bulk_and_scales_with_her():
    suppressed = NernstPlanckFilm(bulk_pH=2.0, her_i0=1e-7).solve(100.0)
    active = NernstPlanckFilm(bulk_pH=2.0, her_i0=1e-3).solve(100.0)
    assert active.local_pH_rise > suppressed.local_pH_rise
    assert active.surface_pH > active.bulk_pH


def test_proton_transport_limits_the_local_pH_rise_in_strong_acid():
    """Inward H+ migration/diffusion buffers the surface far better than a
    diffusion-only film predicts, so a 1 M-acid bath stays acidic."""
    state = NernstPlanckFilm(bulk_pH=0.0, her_i0=1e-4).solve(100.0)
    assert state.surface_pH < 2.0


def test_agitation_reduces_depletion_and_pH_rise():
    stagnant = NernstPlanckFilm(boundary_layer_m=2e-4, her_i0=1e-4).solve(100.0)
    agitated = NernstPlanckFilm(boundary_layer_m=2e-5, her_i0=1e-4).solve(100.0)
    assert agitated.local_pH_rise < stagnant.local_pH_rise
    assert agitated.surface_fe_M > stagnant.surface_fe_M


def test_higher_current_depletes_the_surface_further():
    film = NernstPlanckFilm(fe_conc_M=0.5, her_i0=1e-6)
    low = film.solve(10.0)
    high = film.solve(150.0)
    assert high.surface_fe_M < low.surface_fe_M
    assert high.surface_fe_M > 0.0


def test_her_suppression_raises_current_efficiency():
    active = NernstPlanckFilm(her_i0=1e-2).solve(100.0)
    inhibited = NernstPlanckFilm(her_i0=1e-6).solve(100.0)
    assert inhibited.current_efficiency > active.current_efficiency


def test_efficiency_sweep_shape():
    js, ce = NernstPlanckFilm(her_i0=1e-4).efficiency_sweep([10.0, 50.0, 100.0])
    assert js.shape == ce.shape == (3,)
    assert np.all((ce >= 0.0) & (ce <= 1.0))


# ─── Precipitation diagnostics ────────────────────────────────────────
def test_precipitation_flags_in_alkaline_bath():
    state = NernstPlanckFilm(bulk_pH=9.0, fe_conc_M=0.05, her_i0=1e-4).solve(20.0)
    assert state.precipitation_active
    assert state.feoh2_supersaturation >= 1.0


def test_no_precipitation_in_strong_acid():
    state = NernstPlanckFilm(bulk_pH=1.0, her_i0=1e-6).solve(50.0)
    assert not state.precipitation_active
    assert state.feoh2_supersaturation < 1.0


# ─── Profiles, helpers and validation ─────────────────────────────────
def test_profile_grid_size_and_monotonic_iron():
    film = NernstPlanckFilm(grid_points=41, her_i0=1e-4)
    p = film.solve(100.0).profile
    assert len(p.x_m) == 41
    assert np.all(np.diff(p.x_m) > 0.0)
    # Fe2+ is consumed at the electrode, so it increases toward the bulk.
    assert np.all(np.diff(p.fe_M) >= -1e-12)


def test_summary_reports_expected_keys():
    s = NernstPlanckFilm(her_i0=1e-4).summary(100.0)
    for key in (
        "Current efficiency (%)",
        "Surface pH",
        "i_lim diffusion (A/m²)",
        "i_lim with migration (A/m²)",
        "Migration enhancement (×)",
        "t_Fe²⁺ (bulk)",
    ):
        assert key in s


def test_compare_support_levels_is_ordered():
    rows = compare_support_levels([0.0, 2.0], j_mA_cm2=50.0, her_i0=1e-4)
    assert len(rows) == 2
    assert rows[0]["t_Fe"] > rows[1]["t_Fe"]
    assert rows[0]["migration_enhancement"] > rows[1]["migration_enhancement"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fe_conc_M": 0.0},
        {"fe_conc_M": -1.0},
        {"support_conc_M": -0.1},
        {"boundary_layer_m": 0.0},
        {"grid_points": 2},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        NernstPlanckFilm(**kwargs)


def test_nonpositive_current_and_negative_flux_are_rejected():
    with pytest.raises(ValueError):
        NernstPlanckFilm().solve(0.0)
    with pytest.raises(ValueError):
        NernstPlanckFilm().integrate(-1.0, 0.0)
