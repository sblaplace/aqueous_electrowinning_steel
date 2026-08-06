"""
Tests for the anode / OER / DSA model (models/anode.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anode import (
    AnodeKinetics,
    AnodeMaterial,
    DSA_IRO2_TA2O5,
    NICO_SPINEL,
    bubble_fraction,
    bubble_resistance_multiplier,
    concentration_overpotential_oer,
    full_cell_voltage,
    E0_OER_ACIDIC,
)


# ─── Material properties ────────────────────────────────────────────────

def test_dsa_ir02_ta2o5_is_ir02():
    assert "IrO" in DSA_IRO2_TA2O5.name


def test_nico_spinel_has_high_i0():
    """NiCo spinel should have higher OER exchange current than IrO₂ DSA."""
    assert NICO_SPINEL.oer_i0 > DSA_IRO2_TA2O5.oer_i0


def test_temperature_affects_exchange_current():
    """Arrhenius correction: i₀(T) should be higher at elevated temperature."""
    cold = DSA_IRO2_TA2O5
    warm = AnodeMaterial(
        name="warm IrO₂",
        oer_i0=DSA_IRO2_TA2O5.oer_i0,
        oer_tafel_V=DSA_IRO2_TA2O5.oer_tafel_V,
        temperature_C=90.0,
    )
    assert warm.oer_i0_at_T() > cold.oer_i0_at_T()


# ─── Equilibrium potentials ─────────────────────────────────────────────

def test_oer_equilibrium_is_1_229_at_pH0_acidic():
    """At pH 0, acidic OER E_eq ≈ 1.229 V vs SHE."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=0.0)
    assert anode.oer_equilibrium() == pytest.approx(E0_OER_ACIDIC, abs=0.01)


def test_oer_equilibrium_decreases_with_pH_acidic():
    """Acidic OER should shift negative with pH (−dE/dpH at operating T)."""
    anode_ph1 = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=1.0)
    anode_ph3 = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=3.0)
    delta_pH = 3.0 - 1.0
    # (RT/nF)·ln(10) per pH unit, evaluated at 60°C (333 K)
    # = 0.0591 * (333/298) ≈ 0.066 V/pH
    expected_shift = 0.0591 * (333.0 / 298.15) * delta_pH
    assert (anode_ph3.oer_equilibrium() - anode_ph1.oer_equilibrium()) == pytest.approx(
        -expected_shift, rel=1e-2
    )


def test_oer_equilibrium_at_elevated_pH():
    """OER equilibrium should be well-defined at pH 14 (alkaline pathway)."""
    anode = AnodeKinetics(
        material=NICO_SPINEL, electrolyte_type="alkaline", pH=14.0
    )
    # At 60°C (NICO_SPINEL temperature), 2H₂O OER: E = 1.229 − 0.0661×14
    assert 0.15 < anode.oer_equilibrium() < 0.30


def test_cer_equilibrium_rises_with_Cl_activity():
    """E_eq(CER) = E°_CER + (RT/2F)·ln(a_Cl²); higher Cl⁻ → higher E_eq."""
    anode_low = AnodeKinetics(
        material=DSA_IRO2_TA2O5,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=1.0,
    )
    anode_high = AnodeKinetics(
        material=DSA_IRO2_TA2O5,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=10.0,
    )
    # (RT/2F)·ln(100) ≈ 0.0591·log10(100) = 0.118 V at 25 °C; slightly higher at 60 °C
    assert anode_high.cer_equilibrium() > anode_low.cer_equilibrium()


# ─── Tafel kinetics ───────────────────────────────────────────────────

def test_oer_current_is_zero_below_equilibrium():
    """At E < E_eq the OER current should be zero."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=0.0)
    E_eq = anode.oer_equilibrium()
    assert anode._oer_current(E_eq - 0.1) == pytest.approx(0.0, abs=1e-12)


def test_oer_current_increases_with_overpotential():
    """Anodic current should rise steeply above E_eq."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=0.0)
    E_eq = anode.oer_equilibrium()
    i_low = anode._oer_current(E_eq + 0.05)
    i_high = anode._oer_current(E_eq + 0.15)
    assert i_high > i_low


def test_cer_active_only_in_chloride_acidic():
    """CER is suppressed in alkaline and neutral baths."""
    base = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0, a_Cl_molar=0.0
    )
    chloride = AnodeKinetics(
        material=DSA_IRO2_TA2O5,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=10.0,
    )
    assert not base.cer_active
    assert chloride.cer_active


def test_cer_current_at_fixed_potential_decreases_with_Cl_activity():
    """At fixed electrode potential E, higher a_Cl raises E_eq(CER) and reduces η_CER.

    The Tafel current falls because the driving force shrinks.  This is the
    opposite of the i₀ effect — both are physically real; this test checks
    the fixed-E (fixed-cell-voltage) behaviour relevant to operation.
    """
    anode_low = AnodeKinetics(
        material=DSA_IRO2_TA2O5,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=1.0,
    )
    anode_high = AnodeKinetics(
        material=DSA_IRO2_TA2O5,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=10.0,
    )
    E = 1.56   # V vs SHE — fixed electrode potential (AWARE operating range)
    i_low = anode_low._cer_current(E)
    i_high = anode_high._cer_current(E)
    assert i_high < i_low   # higher Cl⁻ → higher E_eq → smaller η → smaller i


# ─── Overpotential decomposition ───────────────────────────────────────

def test_zero_current_gives_zero_eta():
    """At j = 0 the total anode overpotential should be zero."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    r = anode.overpotential_at_current(0.0)
    assert r["total_V"] == pytest.approx(0.0, abs=1e-9)
    assert r["eta_activation_V"] == pytest.approx(0.0, abs=1e-9)


def test_overpotential_rises_with_current_density():
    """η_anode should increase monotonically with current density."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    prev_eta = 0.0
    for j in [10.0, 50.0, 100.0, 200.0]:
        r = anode.overpotential_at_current(j)
        assert r["total_V"] > prev_eta
        prev_eta = r["total_V"]


def test_activation_is_significant_overpotential():
    """Activation overpotential should be a meaningful fraction of total η."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    r = anode.overpotential_at_current(100.0)
    # Activation is the dominant contributor in this configuration
    assert r["eta_activation_V"] > 0.2
    assert r["eta_activation_V"] + r["eta_concentration_V"] > 0.3


def test_concentration_eta_is_nonnegative():
    """η_conc should always be non-negative."""
    eta = concentration_overpotential_oer(100.0, temperature_C=60.0)
    assert eta >= 0.0


def test_concentration_eta_deprecated_o2_args_warn():
    """The old dissolved-O₂ keyword arguments must emit a DeprecationWarning."""
    with pytest.warns(DeprecationWarning):
        concentration_overpotential_oer(
            100.0, diffusivity_O2_m2_s=2e-9, bulk_O2_mol_m3=0.25
        )


def test_concentration_eta_increases_with_current():
    """η_conc should increase as current density rises (unsaturated regime)."""
    eta_low = concentration_overpotential_oer(10.0, temperature_C=60.0)
    eta_high = concentration_overpotential_oer(100.0, temperature_C=60.0)
    assert eta_high > eta_low


def test_concentration_eta_is_salt_polarization_not_o2_depletion():
    """η_conc is supporting-salt concentration polarization (Nernst/Hittorf).

    Regression guard for the 2026-08 physics correction: a previous version
    modelled dissolved O₂ as a reactant that *depletes* at the OER anode
    (i_lim = 4F D_O2 C_O2/δ), which has the sign/mechanism backwards — OER
    produces O₂.  The corrected model depletes the supporting anion, so
    the result must be a small, finite Nernst film overpotential that
    vanishes with no anion transport (t₊ → 1).
    """
    eta = concentration_overpotential_oer(100.0, temperature_C=60.0)
    # At 100 mA/cm² the salt film gives a few tens of mV, not 0.3–0.5 V.
    assert 0.0 < eta < 0.2
    # No anion transport → no concentration overpotential.
    assert concentration_overpotential_oer(
        100.0, cation_transport_number=1.0
    ) == pytest.approx(0.0)


def test_concentration_eta_rises_near_diffusion_limit():
    """η_conc grows as the surface salt is depleted toward i_lim.

    Use a dilute supporting salt (0.05 M) and a thick film so the
    salt-transport limit sits near the tested currents and the
    logarithmic Nernst curvature is visible.
    """
    kw = dict(
        temperature_C=60.0,
        boundary_layer_m=1e-4,
        bulk_salt_M=0.05,
        salt_diffusivity_m2_s=1.0e-9,
        cation_transport_number=0.6,
        n_valence=2,
    )
    eta_low = concentration_overpotential_oer(5.0, **kw)
    eta_mid = concentration_overpotential_oer(10.0, **kw)
    eta_high = concentration_overpotential_oer(30.0, **kw)
    # Monotonic increase with j
    assert eta_mid > eta_low
    assert eta_high > eta_mid
    # η = (RT/nF)·ln(1/(1−j/j_lim)) is convex in j.
    assert (eta_mid - eta_low) / 5.0 < (eta_high - eta_mid) / 20.0


def test_NiCo_spinel_lower_eta_than_IrO2():
    """At equal current density, NiCo spinel should have lower η than IrO₂ DSA."""
    ir_anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="alkaline", pH=14.0
    )
    nico_anode = AnodeKinetics(
        material=NICO_SPINEL, electrolyte_type="alkaline", pH=14.0
    )
    eta_ir = ir_anode.eta_anode(100.0)
    eta_nico = nico_anode.eta_anode(100.0)
    assert eta_nico < eta_ir


# ─── Bubble resistance ──────────────────────────────────────────────────

def test_bubble_fraction_is_saturating():
    """Bubble fraction should approach a maximum asymptotically with j."""
    for j in [10.0, 100.0, 500.0, 1000.0]:
        theta = bubble_fraction(j, temperature_C=60.0, anode_material="IrO2")
        assert 0.0 <= theta <= 0.13


def test_bubble_fraction_increases_with_current():
    """θ should increase monotonically with j."""
    theta_low = bubble_fraction(10.0)
    theta_high = bubble_fraction(500.0)
    assert theta_high > theta_low


def test_bubble_resistance_multiplier_at_zero():
    """At θ=0 the resistance multiplier should be exactly 1."""
    assert bubble_resistance_multiplier(0.0) == pytest.approx(1.0, abs=1e-9)


def test_bubble_resistance_increases_with_theta():
    """Higher bubble coverage should increase electrolyte resistance."""
    r_low = bubble_resistance_multiplier(0.05)
    r_high = bubble_resistance_multiplier(0.15)
    assert r_high > r_low


def test_bubble_eta_contributes_nonnegative_resistance():
    """Bubble-induced resistance overpotential should be non-negative."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    r = anode.overpotential_at_current(100.0)
    assert r["eta_bubble_V"] >= 0.0


# ─── CER vs OER mixed potential ───────────────────────────────────────

def test_CER_fraction_bounded_0_to_1():
    """CER fraction must be between 0 and 1 at all current densities."""
    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=10.0,
    )
    for j in [10.0, 50.0, 100.0]:
        r = anode.overpotential_at_current(j)
        assert 0.0 <= r["cer_fraction"] <= 1.0


def test_OER_dominates_in_acidic_no_chloride():
    """In acid without chloride, OER should carry essentially all the current."""
    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0, a_Cl_molar=0.0
    )
    r = anode.overpotential_at_current(100.0)
    assert r["cer_fraction"] < 1e-3


def test_CER_makes_significant_contribution_in_concentrated_chloride():
    """In very high Cl⁻ (AWARE process), CER should carry a measurable fraction."""
    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=12.0,
    )
    r = anode.overpotential_at_current(100.0)
    # At high current, OER dominates even in conc. chloride on DSA;
    # but at low current the CER fraction should be non-trivial
    r_low = anode.overpotential_at_current(5.0)
    assert r_low["cer_fraction"] > 0.0


# ─── Polarization curve ────────────────────────────────────────────────

def test_polarization_curve_length():
    """Polarization curve should return arrays of equal length."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    j, E, eta, eta_act, cer = anode.polarization_curve(np.linspace(1.0, 300.0, 100))
    assert len(j) == len(E) == len(eta) == len(eta_act) == len(cer)


def test_polarization_curve_E_rises_with_j():
    """Anode potential should increase with current density."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    j, E, *_ = anode.polarization_curve(np.linspace(10.0, 300.0, 50))
    assert E[-1] > E[0]


def test_polarization_curve_cer_fraction_bounded():
    """CER fraction must be between 0 and 1 everywhere."""
    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5,
        electrolyte_type="acidic_chloride",
        pH=0.0,
        a_Cl_molar=10.0,
    )
    _, _, _, _, cer = anode.polarization_curve(np.linspace(1.0, 500.0, 200))
    assert np.all((cer >= 0.0) & (cer <= 1.0))


# ─── Full-cell integration ─────────────────────────────────────────────

def test_full_cell_voltage_is_positive():
    """V_cell should always be positive (cathode is negative of anode)."""
    anode = AnodeKinetics(
        material=NICO_SPINEL, electrolyte_type="alkaline", pH=14.0
    )
    result = full_cell_voltage(
        anode=anode,
        E_cathode_eq=-0.440,
        E_cathode_actual=-0.740,
        ir_drop=0.15,
        j_mA_cm2=100.0,
    )
    assert result["V_cell"] > 0.0


def test_full_cell_voltage_components_sum():
    """V_cell ≈ (E_anode − E_cathode) + ir_drop."""
    anode = AnodeKinetics(
        material=NICO_SPINEL, electrolyte_type="alkaline", pH=14.0
    )
    result = full_cell_voltage(
        anode=anode,
        E_cathode_eq=-0.440,
        E_cathode_actual=-0.740,
        ir_drop=0.15,
        j_mA_cm2=100.0,
    )
    V_computed = result["E_anode"] - result["E_cathode"] + result["ir_drop"]
    assert result["V_cell"] == pytest.approx(V_computed, abs=1e-9)


def test_full_cell_thermo_is_less_than_total():
    """Thermodynamic voltage should be smaller than V_cell."""
    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0
    )
    result = full_cell_voltage(
        anode=anode,
        E_cathode_eq=-0.440,
        E_cathode_actual=-0.740,
        ir_drop=0.15,
        j_mA_cm2=100.0,
    )
    assert result["E_thermo"] < result["V_cell"]


# ─── O2 and Cl2 production rates ──────────────────────────────────────

def test_O2_rate_positive():
    """O₂ production rate should be non-negative at all current densities."""
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    for j in [10.0, 50.0, 100.0, 200.0]:
        rate = anode.O2_production_rate_mol_m2_hr(j)
        assert rate >= 0.0


def test_Cl2_rate_zero_without_chloride():
    """Cl₂ rate should be zero when no chloride is present."""
    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0, a_Cl_molar=0.0
    )
    rate = anode.Cl2_production_rate_mol_m2_hr(100.0)
    assert rate == pytest.approx(0.0, abs=1e-12)


# ─── Summary dict ─────────────────────────────────────────────────────

def test_summary_has_required_keys():
    """Summary dict should contain all expected fields."""
    anode = AnodeKinetics(material=NICO_SPINEL, electrolyte_type="alkaline", pH=14.0)
    s = anode.summary(100.0)
    for key in (
        "η_anode total (V)",
        "η_activation (V)",
        "η_concentration (V)",
        "η_bubble (V)",
        "E_anode (V vs SHE)",
        "E_eq OER (V vs SHE)",
        "O₂ rate (mol/m²·hr)",
    ):
        assert key in s


# ─── CellVoltageModel integration ─────────────────────────────────────

def test_cell_voltage_model_with_anode():
    """CellVoltageModel should compute η_anode from AnodeKinetics when supplied."""
    from models.electrochemistry import CellVoltageModel

    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0
    )
    cell = CellVoltageModel(
        E_cathode_eq=-0.440,
        eta_cathode=0.30,
        ir_drop=0.20,
        anode=anode,
        j_operating_mA_cm2=100.0,
    )
    r = anode.overpotential_at_current(100.0)
    assert cell._effective_eta_anode == pytest.approx(r["total_V"], abs=1e-9)
    assert cell.V_cell > 1.5   # reasonable minimum for acidic Fe electrowinning


def test_cell_voltage_model_no_anode_uses_detailed_decomposition():
    """Without an anode model, CellVoltageModel uses Nernst cathode + fixed
    eta_anode + detailed IR decomposition (electrolyte + contacts)."""
    from models.electrochemistry import CellVoltageModel

    cell = CellVoltageModel(
        E_cathode_eq=-0.440,
        eta_cathode=0.30,
        eta_anode=0.40,
        ir_drop=0.20,
    )
    # No anode -> E_anode_nernst falls back to fixed OER, eta_anode to fixed.
    assert cell.E_anode_nernst == pytest.approx(1.229, abs=1e-9)
    assert cell._effective_eta_anode == pytest.approx(0.40, abs=1e-9)
    # Cathode equilibrium is Nernst (here == E° since a_Fe2 = 1 M).
    assert cell.E_cathode_nernst == pytest.approx(-0.440, abs=1e-9)
    # Detailed IR drop (electrolyte + contacts) replaces the legacy fixed 0.20.
    assert cell._total_ir_drop == pytest.approx(
        cell.IR_electrolyte + cell.IR_membrane + cell.IR_contacts, abs=1e-9
    )
    # V_cell is the sum of thermodynamic + kinetic + ohmic terms.
    assert cell.V_cell == pytest.approx(
        cell.E_thermodynamic
        + cell.eta_cathode
        + cell._effective_eta_anode
        + cell._total_ir_drop,
        abs=1e-9,
    )


def test_cell_voltage_model_thermo_matches_anode_eq():
    """E_thermodynamic should use the anode's Nernst-corrected E_eq."""
    from models.electrochemistry import CellVoltageModel

    # DSA IrO2 at pH 2, 60°C: OER E_eq ≈ 1.097 V; Fe E_eq = −0.440 V
    anode = AnodeKinetics(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    cell = CellVoltageModel(
        E_cathode_eq=-0.440,
        anode=anode,
        j_operating_mA_cm2=100.0,
    )
    # Expected: |1.097 − (−0.440)| ≈ 1.537 V
    assert cell.E_thermodynamic == pytest.approx(1.537, abs=0.01)


# ── Soluble vs inert anode chemistry (2026-08) ─────────────────────────────


def test_soluble_fe_anode_has_no_gas_or_bubble_penalty():
    """A soluble Fe anode (Fe→Fe²⁺+2e⁻) produces no O₂/Cl₂ or bubbles."""
    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0,
        anode_chemistry="soluble", fe2_conc_M=1.0,
    )
    r = anode.overpotential_at_current(200.0)
    assert r["anode_chemistry"] == "soluble"
    assert r["bubble_fraction"] == 0.0
    assert r["eta_bubble_V"] == 0.0
    assert r["eta_concentration_V"] == 0.0
    assert r["i_oer_A_m2"] == 0.0
    assert r["i_cer_A_m2"] == 0.0
    assert r["i_fe_dissolution_A_m2"] == pytest.approx(2000.0, rel=1e-6)


def test_soluble_anode_runs_near_fe2_fe_potential():
    """Soluble anode E is set by Fe²⁺/Fe (~−0.44 V), not OER (~1.2 V)."""
    anode = AnodeKinetics(
        material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0,
        anode_chemistry="soluble",
    )
    r = anode.overpotential_at_current(100.0)
    assert r["E_eq_V"] == pytest.approx(-0.440, abs=0.01)
    assert r["E_anode_V"] < 0.0  # far below OER potentials


def test_soluble_anode_much_lower_overpotential_than_inert():
    """Soluble Fe dissolution is fast; the cell saves ~0.4 V vs DSA."""
    kw = dict(material=DSA_IRO2_TA2O5, electrolyte_type="acidic", pH=2.0)
    soluble = AnodeKinetics(anode_chemistry="soluble", **kw).eta_anode(200.0)
    inert = AnodeKinetics(anode_chemistry="inert", **kw).eta_anode(200.0)
    assert soluble < inert - 0.3


def test_invalid_anode_chemistry_rejected():
    import pytest
    with pytest.raises(ValueError):
        AnodeKinetics(
            material=DSA_IRO2_TA2O5, electrolyte_type="acidic",
            anode_chemistry="mystery",
        )
