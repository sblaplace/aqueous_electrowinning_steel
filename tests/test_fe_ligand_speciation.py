"""
Unit tests for Fe(II)-ligand speciation (Round 5, E2).
"""

import pytest

from models.fe_ligand_speciation import (
    deposition_potential_shift_V,
    formation_constant,
    free_Fe2_concentration_M,
    ligand_window_summary,
    precipitation_pH,
)


def test_free_fe2_lowered_by_ligand():
    """A strong ligand reduces free Fe²⁺ (binds it)."""
    free_none = free_Fe2_concentration_M(0.5, "none", 0.0)
    free_edta = free_Fe2_concentration_M(0.5, "EDTA", 0.5)
    assert free_edta["free_fe2_M"] < free_none["free_fe2_M"]
    assert free_edta["bound_fraction"] > 0.0


def test_precipitation_pH_raised_by_ligand():
    """Ligand raises the Fe(OH)₂ precipitation pH (wider window)."""
    base = precipitation_pH(0.5, "none", 0.0)
    chelated = precipitation_pH(0.5, "EDTA", 0.5)
    assert chelated > base


def test_deposition_potential_shift_negative():
    """Complexation shifts Fe²⁺/Fe reduction potential negative (<=0)."""
    shift = deposition_potential_shift_V(0.5, "EDTA", 0.5)
    assert shift <= 0.0
    base = deposition_potential_shift_V(0.5, "none", 0.0)
    assert base == pytest.approx(0.0, abs=1e-6)


def test_formation_constant_monotonic_with_beta():
    """Stronger ligand (higher log beta) -> larger formation constant."""
    assert formation_constant("EDTA", 1) > formation_constant("glycine", 1)


def test_window_summary_coherent():
    """Summary fields are internally consistent and non-negative."""
    res = ligand_window_summary(total_Fe_M=0.5, ligand="citrate", total_ligand_M=0.5)
    assert res["precipitation_pH"] > 0.0
    assert res["pH_window_widening"] >= 0.0
    assert res["deposition_potential_shift_V"] <= 0.0
    assert 0.0 <= res["bound_fraction"] <= 1.0
