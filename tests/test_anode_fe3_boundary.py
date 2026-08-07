"""
Tests for the anode Fe³⁺ boundary layer (models/anode_fe3_boundary.py + the
AnodeKinetics opt-in wiring).

CHEM_PHYS_REVIEW Tier 1.3: the anode module models OER/CER but not the local
Fe³⁺ accumulation when anolyte Fe²⁺ is oxidised.  Fe³⁺ hydrolyses, lowering
the local pH and raising the OER overpotential (10 s of mV) — the anode end of
the Fe(OH)₃ / fe³-shuttle story.  Must be OFF by default so the bare-OER anode
is byte-identical.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anode import AnodeKinetics, DSA_IRO2_TA2O5
from models.anode_fe3_boundary import (
    E0_FE3_FE2,
    fe3_boundary_analysis,
    mass_transfer_coeff,
)


# ─── Module-level unit tests ───────────────────────────────────────────

def test_zero_fe2_gives_inert_result():
    """Fe²⁺-free anolyte → all-zero boundary layer (safe to wire in)."""
    r = fe3_boundary_analysis(0.0, ph_bulk=2.0)
    assert not r.active
    assert r.fe2_oxidation_flux_mol_m2_s == 0.0
    assert r.oer_overpotential_raise_V == 0.0
    assert r.feoh3_sludge_flux_mol_m2_s == 0.0
    assert r.shuttle_source_flux_mol_m2_s == 0.0
    assert r.surface_pH == 2.0


def test_fe2_oxidation_is_mass_transfer_limited():
    """Q = k_m·[Fe²⁺]_bulk with k_m = D/δ; i_ox = F·Q (1 e⁻ per Fe²⁺)."""
    fe2_M = 1.0
    delta = 5e-5
    d = 5.5e-10
    km = mass_transfer_coeff(d, delta)
    r = fe3_boundary_analysis(fe2_M, ph_bulk=2.0, boundary_layer_m=delta,
                              d_fe2_m2_s=d)
    assert r.fe2_oxidation_flux_mol_m2_s == pytest.approx(km * fe2_M * 1000.0)
    # i_ox = F·Q
    from models.electrochemistry import FARADAY
    assert r.fe2_oxidation_current_A_m2 == pytest.approx(
        FARADAY * r.fe2_oxidation_flux_mol_m2_s
    )


def test_higher_fe2_gives_larger_drop_and_raise():
    """More anolyte Fe²⁺ → more oxidation → bigger pH drop & OER raise."""
    low = fe3_boundary_analysis(0.5, ph_bulk=2.0)
    high = fe3_boundary_analysis(2.0, ph_bulk=2.0)
    assert high.ph_drop > low.ph_drop
    assert high.oer_overpotential_raise_V > low.oer_overpotential_raise_V
    assert high.feoh3_sludge_flux_mol_m2_s > low.feoh3_sludge_flux_mol_m2_s


def test_reference_scenario_10s_of_mV():
    """1 M Fe²⁺ anolyte at a DSA: ~1-unit pH drop, ~10 s-of-mV OER raise."""
    r = fe3_boundary_analysis(1.0, ph_bulk=2.0, boundary_layer_m=5e-5,
                              temperature_C=60.0)
    assert 0.3 <= r.ph_drop <= 3.0, f"ΔpH={r.ph_drop:.2f} outside plausible band"
    assert 0.02 <= r.oer_overpotential_raise_V <= 0.20, (
        f"Δη={r.oer_overpotential_raise_V:.3f} V outside the 10 s-of-mV band"
    )
    assert r.surface_pH < 2.0  # local pH drops below bulk
    assert r.feoh3_sludge_flux_mol_m2_s > 0.0
    assert r.shuttle_source_flux_mol_m2_s > 0.0


def test_hydrolysis_fraction_splits_sludge_and_shuttle():
    """f_hyd precipitates to sludge; the complement leaves as the shuttle."""
    f = 0.25
    r = fe3_boundary_analysis(1.0, ph_bulk=2.0, fraction_hydrolysed=f)
    total = r.fe2_oxidation_flux_mol_m2_s
    assert r.feoh3_sludge_flux_mol_m2_s == pytest.approx(f * total)
    assert r.shuttle_source_flux_mol_m2_s == pytest.approx((1 - f) * total)


def test_fe3_equilibrium_well_below_oer():
    """E0(Fe³⁺/Fe²⁺)=0.771 V is far below the OER potential → oxidation mass-ltd."""
    assert 0.771 == pytest.approx(E0_FE3_FE2)


# ─── AnodeKinetics opt-in integration ──────────────────────────────────

def test_fe3_boundary_off_by_default_unchanged():
    """Default AnodeKinetics (flag off) shows no Fe³⁺ effect."""
    a = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    r = a.overpotential_at_current(300.0)
    assert r["fe3_boundary_oer_raise_V"] == 0.0
    assert r["fe3_boundary_layer"] is None
    # total is exactly the bare-OER decomposition (no additive term).
    assert r["total_V"] == pytest.approx(
        r["eta_activation_V"] + r["eta_concentration_V"] + r["eta_bubble_V"]
    )


def test_fe3_boundary_flag_on_raises_anode_overpotential():
    """With the flag on + Fe²⁺ anolyte, OER overpotential rises 10 s of mV."""
    off = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    on = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0,
        fe3_boundary_layer=True, anolyte_fe2_M=1.0,
    )
    r_off = off.overpotential_at_current(300.0)
    r_on = on.overpotential_at_current(300.0)
    raise_ = r_on["fe3_boundary_oer_raise_V"]
    assert raise_ > 0.0
    assert 0.02 <= raise_ <= 0.20, f"Δη={raise_:.3f} V outside 10 s-of-mV band"
    assert r_on["total_V"] == pytest.approx(r_off["total_V"] + raise_)
    assert r_on["fe3_boundary_layer"] is not None
    assert r_on["fe3_boundary_layer"].surface_pH < 2.0


def test_fe3_boundary_needs_fe2_anolyte():
    """Flag on but no anolyte Fe²⁺ → layer inactive, bare-OER unchanged."""
    a = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0,
        fe3_boundary_layer=True, anolyte_fe2_M=0.0,
    )
    r = a.overpotential_at_current(300.0)
    assert r["fe3_boundary_oer_raise_V"] == 0.0
    assert r["fe3_boundary_layer"] is None


def test_fe3_boundary_soluble_anode_unchanged():
    """The Fe³⁺ layer is an OER-anode feature; a soluble Fe anode ignores it."""
    a = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0,
        anode_chemistry="soluble", fe3_boundary_layer=True, anolyte_fe2_M=1.0,
    )
    r = a.overpotential_at_current(300.0)
    assert r["fe3_boundary_oer_raise_V"] == 0.0
    assert r["fe3_boundary_layer"] is None
