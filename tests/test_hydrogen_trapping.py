"""
Unit tests for hierarchical hydrogen trapping and McNabb–Foster bakeout model.
"""

import pytest
from models.hydrogen_trapping import (
    default_trap_hierarchy,
    lattice_diffusivity_m2_s,
    effective_trapped_diffusivity_m2_s,
    compute_bakeout_schedule,
)


def test_effective_diffusivity_smaller_than_lattice():
    """Trap occupancy slows down apparent hydrogen diffusion compared to pure lattice."""
    d_lattice = lattice_diffusivity_m2_s(25.0)
    d_eff = effective_trapped_diffusivity_m2_s(25.0)

    assert d_lattice > 0.0
    assert d_eff > 0.0
    assert d_eff < d_lattice * 0.50  # Trapping slows diffusion significantly at 25 °C


def test_effective_diffusivity_increases_with_temperature():
    """At elevated bakeout temperature (190 °C), thermal energy detraps hydrogen."""
    d_eff_25 = effective_trapped_diffusivity_m2_s(25.0)
    d_eff_190 = effective_trapped_diffusivity_m2_s(190.0)

    assert d_eff_190 > d_eff_25 * 10.0


def test_bakeout_schedule_calculation():
    """Verify ASTM F519 bakeout time calculation for 100 µm foil."""
    res_100um = compute_bakeout_schedule(
        foil_thickness_um=100.0,
        total_initial_H_ppm_wt=5.0,
        bake_temperature_C=190.0,
        target_diffusible_H_ppm_wt=0.10,
    )

    assert res_100um.is_embrittlement_safe
    assert 1.0 <= res_100um.required_bake_time_hours <= 12.0
    assert res_100um.irreversible_trapped_H_ppm_wt > 0.0

    # Thicker foil requires longer bakeout
    res_300um = compute_bakeout_schedule(
        foil_thickness_um=300.0,
        total_initial_H_ppm_wt=5.0,
        bake_temperature_C=190.0,
        target_diffusible_H_ppm_wt=0.10,
    )
    assert res_300um.required_bake_time_hours > res_100um.required_bake_time_hours
