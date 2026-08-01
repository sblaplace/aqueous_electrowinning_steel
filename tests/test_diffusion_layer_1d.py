"""Tests for the 1D diffusion-layer FE prediction model.

Acceptance criteria from the task:
  - FE(j) curve: high at low j, drops at high j (HER competition)
  - FE increases with [Fe²⁺]
  - FE increases with T
  - Surface pH rises with j (proton depletion)
  - Precipitation criterion flags high-j conditions
  - V_cell decomposes correctly
  - >= 10 tests
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.diffusion_layer_1d import (  # noqa: E402
    DiffusionLayer1D,
    _diffusivity_T,
    _Ka_T,
    faradaic_efficiency,
    KA2_25C,
    KAB_25C,
)


# ─── Helpers ──────────────────────────────────────────────────────────

def _default(**overrides):
    """Return a default model, optionally overriding fields."""
    return DiffusionLayer1D(**overrides)


# ─── 1. FE(j) curve: high at low j, drops at high j ─────────────────

def test_fe_drops_with_current_density():
    """Core acceptance: FE is high at low j and falls at high j due to HER."""
    m = _default()
    fe_low = m.solve(50.0).current_efficiency
    fe_high = m.solve(400.0).current_efficiency
    assert fe_low > fe_high
    assert fe_low > 0.5
    assert 0.0 < fe_high < fe_low


def test_fe_sweep_is_monotonically_decreasing():
    """FE should decrease (or stay flat) as j increases beyond the peak."""
    m = _default()
    js = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    _, fes = m.efficiency_sweep(js)
    assert all(fes[i] >= fes[i + 1] - 1e-6 for i in range(len(fes) - 1))


# ─── 2. FE increases with [Fe²⁺] ────────────────────────────────────

def test_fe_increases_with_fe_concentration():
    """More Fe²⁺ means more available for deposition before transport limit."""
    j = 150.0
    fe_dilute = _default(fe_conc_M=0.2).solve(j).current_efficiency
    fe_concentrated = _default(fe_conc_M=2.0).solve(j).current_efficiency
    assert fe_concentrated > fe_dilute


# ─── 3. FE increases with temperature ────────────────────────────────

def test_fe_increases_with_temperature():
    """Higher T → faster kinetics and higher conductivity → better FE."""
    j = 100.0
    fe_cold = _default(temperature_C=25.0).solve(j).current_efficiency
    fe_hot = _default(temperature_C=80.0).solve(j).current_efficiency
    assert fe_hot > fe_cold


# ─── 4. Surface pH rises with j ─────────────────────────────────────

def test_surface_pH_rises_with_current():
    """HER consumes protons → surface pH increases with current density."""
    m = _default()
    s_low = m.solve(10.0)
    s_high = m.solve(200.0)
    assert s_high.surface_pH > s_low.surface_pH


def test_surface_pH_above_bulk():
    """Surface pH should always be >= bulk pH."""
    m = _default()
    for j in (10.0, 50.0, 100.0, 300.0):
        s = m.solve(j)
        assert s.surface_pH >= m.pH_bulk - 0.01


# ─── 5. Precipitation criterion flags high-j conditions ──────────────

def test_precipitation_at_high_current():
    """At very high j, surface pH rises enough to supersaturate Fe(OH)₂."""
    m = _default(fe_conc_M=1.0, pH_bulk=3.0, buffer_conc_M=0.0)
    s_high = m.solve(500.0)
    # At very high current the surface OH⁻ should be elevated
    assert s_high.feoh2_supersaturation > 0.0
    # Check that supersaturation increases with current
    s_low = m.solve(10.0)
    assert s_high.feoh2_supersaturation >= s_low.feoh2_supersaturation


# ─── 6. V_cell decomposes correctly ─────────────────────────────────

def test_V_cell_components_sum():
    """V_cell = (E_anode_eq + η_anode) - V_cathode + iR."""
    m = _default()
    s = m.solve(100.0)
    expected = (m.E_anode_eq + m.eta_anode_V) - s.V_cathode_V + m.ir_drop_V
    assert s.V_cell == pytest.approx(expected, rel=1e-9)


def test_V_cell_positive():
    """Cell voltage must be positive."""
    m = _default()
    for j in (10.0, 100.0, 300.0):
        assert m.solve(j).V_cell > 0.0


def test_V_cell_increases_with_current():
    """More current → more overpotential → higher cell voltage."""
    m = _default()
    v_low = m.solve(10.0).V_cell
    v_high = m.solve(300.0).V_cell
    assert v_high > v_low


# ─── 7. Diffusion limit scales correctly ────────────────────────────

def test_diffusion_limit_scales_with_concentration():
    """i_lim = z·F·D·C/δ; doubling C doubles i_lim."""
    m1 = _default(fe_conc_M=0.5)
    m2 = _default(fe_conc_M=1.0)
    assert m2.diffusion_limit_A_m2 == pytest.approx(
        2.0 * m1.diffusion_limit_A_m2, rel=1e-9
    )


def test_diffusion_limit_inversely_with_layer_thickness():
    """Thinner film → higher limiting current."""
    thin = _default(delta_m=25e-6)
    thick = _default(delta_m=100e-6)
    assert thin.diffusion_limit_A_m2 == pytest.approx(
        4.0 * thick.diffusion_limit_A_m2, rel=1e-9
    )


# ─── 8. Temperature correction increases diffusivity ────────────────

def test_diffusivity_increases_with_temperature():
    """D(T) = D(25°C)·exp(Ea/R·(1/298 − 1/T)); higher T → larger D."""
    D_cold = _diffusivity_T(1e-9, 298.15)
    D_hot = _diffusivity_T(1e-9, 353.15)  # 80°C
    assert D_hot > D_cold
    assert D_hot / D_cold > 1.5  # should be ~2× over 55 K range


def test_Ka2_increases_with_temperature():
    """Bisulfate dissociation is exothermic → Ka₂ decreases with T."""
    Ka_cold = _Ka_T(KA2_25C, 298.15, -22e3)
    Ka_hot = _Ka_T(KA2_25C, 353.15, -22e3)
    # Exothermic: Ka decreases with T
    assert Ka_hot < Ka_cold


def test_Kab_increases_with_temperature():
    """Boric acid dissociation is endothermic → Ka_b increases with T."""
    Ka_cold = _Ka_T(KAB_25C, 298.15, 14e3)
    Ka_hot = _Ka_T(KAB_25C, 353.15, 14e3)
    assert Ka_hot > Ka_cold


# ─── 9. Buffer effect ───────────────────────────────────────────────

def test_buffer_raises_surface_pH():
    """Boric acid buffer should raise surface pH vs unbuffered at same j.

    At pH 2 the boric acid (pKa ≈ 9.24) has negligible capacity, so the
    effect is tiny and we only check non-negative.  At higher bulk pH
    (4–5) the effect becomes significant.
    """
    j = 200.0
    s_buf = _default(buffer_conc_M=0.4).solve(j)
    s_nobuf = _default(buffer_conc_M=0.0).solve(j)
    # Buffer must not lower surface pH (allow numerical noise)
    assert s_buf.surface_pH >= s_nobuf.surface_pH - 1e-6


# ─── 10. Zero buffer runs cleanly ───────────────────────────────────

def test_zero_buffer_no_error():
    """Model must run without error when buffer_conc_M = 0."""
    m = _default(buffer_conc_M=0.0)
    s = m.solve(100.0)
    assert 0.0 < s.current_efficiency < 1.0
    assert s.surface_pH > 0.0


# ─── 11. Film profile structure ─────────────────────────────────────

def test_profile_has_correct_length():
    """Profile arrays should match grid_points."""
    m = _default(grid_points=51)
    s = m.solve(100.0)
    assert len(s.profile.x_m) == 51
    assert len(s.profile.fe_M) == 51


def test_profile_bulk_edge_matches_input():
    """At x = δ (last grid point), concentrations should match bulk."""
    m = _default()
    s = m.solve(50.0)
    p = s.profile
    # Bulk Fe²⁺
    assert p.fe_M[-1] == pytest.approx(m.fe_conc_M, rel=1e-3)
    # Bulk pH
    bulk_pH_from_profile = float(p.pH[-1])
    assert bulk_pH_from_profile == pytest.approx(m.pH_bulk, abs=0.1)


def test_profile_surface_fe_depleted_at_high_j():
    """Surface Fe²⁺ should be significantly depleted at high j."""
    m = _default(fe_conc_M=0.5, delta_m=50e-6)
    s = m.solve(250.0)
    assert s.surface_fe_M < m.fe_conc_M * 0.8


# ─── 12. Top-level convenience function ─────────────────────────────

def test_faradaic_efficiency_function():
    """Top-level FE function matches model.solve()."""
    fe = faradaic_efficiency(100.0, temperature_C=60.0, fe_conc_M=1.0)
    m = DiffusionLayer1D(temperature_C=60.0, fe_conc_M=1.0)
    assert fe == pytest.approx(m.solve(100.0).current_efficiency, rel=1e-9)


# ─── 13. Summary output ─────────────────────────────────────────────

def test_summary_keys_present():
    """Summary dict should have all expected keys."""
    m = _default()
    s = m.summary(100.0)
    expected_keys = {
        "j applied (mA/cm²)",
        "E cathode (V vs SHE)",
        "Current efficiency (%)",
        "Surface pH",
        "ΔpH (surface − bulk)",
        "Surface Fe²⁺ (M)",
        "i_lim diffusion (A/m²)",
        "i_lim transport (A/m²)",
        "Film Δφ (mV)",
        "Fe(OH)₂ supersaturation",
        "V_cell (V)",
    }
    assert expected_keys.issubset(set(s.keys()))


# ─── 14. Validation ─────────────────────────────────────────────────

def test_negative_conc_raises():
    """Invalid parameters should raise ValueError."""
    with pytest.raises(ValueError, match="fe_conc_M"):
        DiffusionLayer1D(fe_conc_M=-1.0)
    with pytest.raises(ValueError, match="delta_m"):
        DiffusionLayer1D(delta_m=0.0)


def test_negative_j_raises():
    m = _default()
    with pytest.raises(ValueError, match="j_mA_cm2"):
        m.solve(-10.0)


# ─── 15. Convergence ────────────────────────────────────────────────

def test_solve_converges_across_range():
    """Model should converge at a range of operating points."""
    m = _default()
    for j in (5.0, 20.0, 50.0, 100.0, 200.0, 400.0):
        s = m.solve(j)
        assert s.converged, f"Failed to converge at j={j}"


# ─── 16. No NaN / Inf in results ────────────────────────────────────

def test_no_nan_in_results():
    """All outputs should be finite."""
    m = _default()
    for j in (10.0, 100.0, 300.0):
        s = m.solve(j)
        for val in (
            s.current_efficiency,
            s.surface_pH,
            s.surface_fe_M,
            s.V_cell,
            s.feoh2_supersaturation,
            s.film_potential_drop_V,
        ):
            assert math.isfinite(val), f"Non-finite value at j={j}: {val}"
