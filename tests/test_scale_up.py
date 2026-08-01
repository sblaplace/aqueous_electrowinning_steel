"""Tests for the scale-up model: current distribution, mass transport, thermal."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.scale_up import (
    boundary_layer_thickness,
    mass_transport_scaling,
    optimize_geometry,
    primary_current_distribution,
    scale_up_analysis,
    secondary_current_distribution,
    thermal_management,
    uniformity_index,
    wagner_number,
)
from models.transport import D_FE


# ─── 1. Current distribution: uniformity at small Wa ─────────────────
def test_current_distribution_is_uniform_at_small_wa():
    """At high Wagner number (small electrode), distribution is nearly uniform."""
    L = 0.001  # 1 mm electrode
    j_avg = 1000.0  # A/m²
    Wa = wagner_number(kappa=10.0, j_ref=j_avg, L=L)
    assert Wa > 1.0  # should be uniform
    x = np.linspace(-L / 2 * 0.9, L / 2 * 0.9, 201)
    j = primary_current_distribution(x, L, j_avg, kappa=10.0)
    uni = uniformity_index(j)
    assert uni > 0.95, f"Expected uniformity > 0.95 at small Wa={Wa:.2f}, got {uni:.3f}"


# ─── 2. Edge effects emerge at large L ───────────────────────────────
def test_edge_effects_emerge_at_large_l():
    """At low Wagner number (large electrode), edge effects reduce uniformity."""
    L = 0.3  # 30 cm electrode
    j_avg = 3000.0  # A/m², high current
    Wa = wagner_number(kappa=10.0, j_ref=j_avg, L=L)
    assert Wa < 1.0  # should show edge effects
    x = np.linspace(-L / 2 * 0.95, L / 2 * 0.95, 501)
    j = primary_current_distribution(x, L, j_avg, kappa=10.0)
    # Current near edges should be higher than average
    assert np.max(j) > 1.1 * j_avg
    # Uniformity should be worse than small electrode
    uni = uniformity_index(j)
    assert uni < 0.95, f"Expected reduced uniformity at large L, got {uni:.3f}"


# ─── 3. j_lim decreases with increasing L ────────────────────────────
def test_j_lim_decreases_with_increasing_l():
    """Transport-limited current drops as electrode length grows (thicker δ)."""
    j_lim_short = []
    j_lim_long = []
    for L in [0.01, 0.1]:
        x = np.linspace(0.0, L, 101)
        j_local = np.full_like(x, 1000.0)
        mt = mass_transport_scaling(L, j_local, D=D_FE, v=0.1)
        if L == 0.01:
            j_lim_short = mt.j_lim
        else:
            j_lim_long = mt.j_lim

    # The minimum j_lim at the downstream end should be lower for longer electrode
    assert np.min(j_lim_long) < np.min(j_lim_short), (
        f"j_lim should decrease with L: short={np.min(j_lim_short):.0f}, "
        f"long={np.min(j_lim_long):.0f}"
    )


# ─── 4. Temperature rises with current density ──────────────────────
def test_temperature_rises_with_current_density():
    """Higher current density → more Joule heating → higher temperature."""
    thermal_low = thermal_management(
        j_avg=500.0, area_m2=0.1, gap_m=0.01, kappa=10.0, V_cell=2.0,
    )
    thermal_high = thermal_management(
        j_avg=3000.0, area_m2=0.1, gap_m=0.01, kappa=10.0, V_cell=2.0,
    )
    assert thermal_high.T_cell_C > thermal_low.T_cell_C
    assert thermal_high.Q_gen_W > thermal_low.Q_gen_W


# ─── 5. Optimal geometry is physically reasonable ────────────────────
def test_optimal_geometry_is_physically_reasonable():
    """Geometry optimizer returns sensible dimensions and meets constraints."""
    result = optimize_geometry(
        total_current_A=1000.0,  # 1000 A pilot cell
        area_m2=0.1,  # 1000 cm²
        kappa=10.0,
        V_cell=2.0,
        current_efficiency=0.85,
    )
    assert result.area_m2 > 0
    assert result.gap_m >= 0.001  # gap >= 1 mm
    assert result.gap_m < 0.05   # gap < 50 mm
    assert result.j_avg > 0
    assert 0.0 < result.uniformity <= 1.0
    assert result.energy_kWh_per_kg > 0


# ─── 6. Secondary current distribution solves and returns valid data ─
def test_secondary_current_distribution_solves():
    """The 1-D Poisson + BV solver converges and returns physical values."""
    result = secondary_current_distribution(
        L=0.01,  # 10 mm gap
        j_target=1000.0,  # A/m²
        kappa=10.0,
    )
    assert len(result.x) > 10
    assert len(result.j_local) == len(result.x)
    assert result.j_avg > 0
    assert result.uniformity > 0.0
    assert result.uniformity <= 1.0
    # Current density should be positive (cathodic)
    assert np.all(result.j_local > 0)


# ─── 7. Boundary layer grows with position ──────────────────────────
def test_boundary_layer_grows_with_position():
    """δ = sqrt(D·x/v) increases downstream."""
    delta_1mm = boundary_layer_thickness(0.001, D=D_FE, v=0.1)
    delta_100mm = boundary_layer_thickness(0.1, D=D_FE, v=0.1)
    assert delta_100mm > delta_1mm
    # Check scaling: δ ∝ sqrt(L)
    ratio = delta_100mm / delta_1mm
    assert ratio == pytest.approx(np.sqrt(100.0), rel=0.01)


# ─── 8. Thermal result flags boiling risk at extreme conditions ─────
def test_thermal_boiling_risk_flagged():
    """At extreme current densities, boiling risk should be flagged."""
    # Very high current, poor cooling
    result = thermal_management(
        j_avg=10000.0, area_m2=1.0, gap_m=0.01, kappa=10.0, V_cell=2.0,
        h_conv=50.0,  # poor cooling
    )
    assert result.boiling_risk
    assert result.T_cell_C > 80.0


# ─── 9. Full scale-up analysis runs end-to-end ──────────────────────
def test_scale_up_analysis_end_to_end():
    """The unified analysis completes for both lab and pilot scales."""
    # Lab scale: 10 cm²
    lab = scale_up_analysis(area_m2=0.001, total_current_A=10.0)
    assert lab.wagner > 0
    assert lab.primary_uniformity > 0.0
    assert lab.thermal.T_cell_C > 0
    assert lab.mass_transport.j_lim_min > 0

    # Pilot scale: 1000 cm²
    pilot = scale_up_analysis(area_m2=0.1, total_current_A=1000.0)
    assert pilot.wagner > 0
    assert pilot.primary_uniformity > 0.0
    assert pilot.thermal.T_cell_C > 0

    # Lab should be more uniform than pilot
    assert lab.primary_uniformity >= pilot.primary_uniformity


# ─── 10. Mass transport flags transport-limited regions ─────────────
def test_mass_transport_flags_limited_regions():
    """When local current exceeds j_lim, regions are flagged."""
    L = 0.1  # 10 cm
    x = np.linspace(0.0, L, 201)
    # Artificially high current that exceeds limit downstream
    j_local = np.full_like(x, 5000.0)  # very high
    mt = mass_transport_scaling(L, j_local, D=D_FE, v=0.01)  # slow flow
    # At the downstream end with slow flow, transport limit should be exceeded
    assert mt.fraction_limited > 0.0, "Should flag some transport-limited regions"


# ─── 11. Wagner number scaling is correct ────────────────────────────
def test_wagner_number_math():
    """Wa = κ / (j_ref * L) matches hand calculation."""
    Wa = wagner_number(kappa=10.0, j_ref=1000.0, L=0.1)
    assert Wa == pytest.approx(10.0 / (1000.0 * 0.1))
    assert Wa == pytest.approx(0.1)
