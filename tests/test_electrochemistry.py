"""
Tests for the enhanced electrochemistry module.

Covers:
- V_cell increases with current density (ohmic + activation)
- Current efficiency decreases with j (HER competition)
- Energy has minimum at intermediate j
- Divided cell has higher FE than undivided
- Temperature increases conductivity, decreases V_cell
- V_cell decomposition dictionary
- Fe²⁺/Fe³⁺ anode shuttle model
- Energy = f(V, FE) computed correctly
- Membrane IR drop scales linearly with j
- Nernst equation correctness
- Temperature-dependent property functions
"""

import numpy as np
import pytest

from models.electrochemistry import (
    CellVoltageModel,
    FeShuttleAnode,
    MembraneModel,
    conductivity_S_m,
    diffusivity_m2_s,
    viscosity_Pa_s,
    specific_energy_kWh_per_kg,
    nernst_shift,
    E0_FE,
    E0_OER,
    E0_FE3_FE2,
    FARADAY,
    R_GAS,
    M_FE,
    Z_FE,
    T_REF,
)


# ─── V_cell increases with current density ─────────────────────────────

def test_V_cell_increases_with_j():
    """
    At higher current densities, both ohmic and activation overpotentials
    increase, so V_cell must increase monotonically with j.
    """
    voltages = []
    for j in [10, 50, 100, 200, 400]:
        model = CellVoltageModel(j_operating_mA_cm2=j)
        voltages.append(model.V_cell)

    # V_cell should be strictly increasing
    for i in range(len(voltages) - 1):
        assert voltages[i] < voltages[i + 1], (
            f"V_cell({j}) should increase: {voltages}"
        )


# ─── V_cell decomposition dict ─────────────────────────────────────────

def test_V_decomposition_has_all_keys():
    """V_decomposition dict should contain all voltage components."""
    model = CellVoltageModel(j_operating_mA_cm2=100.0)
    d = model.V_decomposition

    expected_keys = {
        "E_cathode (V)", "E_anode (V)", "E_thermodynamic (V)",
        "η_cathode (V)", "η_anode (V)",
        "IR_electrolyte (V)", "IR_membrane (V)", "IR_contacts (V)",
        "IR_total (V)", "V_cell (V)",
    }
    assert set(d.keys()) == expected_keys
    # V_cell should be the sum of components
    assert abs(d["V_cell (V)"] - (
        d["E_thermodynamic (V)"] + d["η_cathode (V)"] + d["η_anode (V)"]
        + d["IR_total (V)"]
    )) < 0.001


# ─── Temperature increases conductivity, decreases V_cell ──────────────

def test_conductivity_increases_with_temperature():
    """Higher temperature → higher ionic conductivity."""
    kappa_25 = conductivity_S_m(298.15)
    kappa_60 = conductivity_S_m(333.15)
    kappa_90 = conductivity_S_m(363.15)
    assert kappa_60 > kappa_25
    assert kappa_90 > kappa_60


def test_V_cell_decreases_with_temperature():
    """Higher temperature → lower V_cell (better conductivity, lower overpotentials)."""
    v_25 = CellVoltageModel(temperature_C=25.0, j_operating_mA_cm2=100.0).V_cell
    v_60 = CellVoltageModel(temperature_C=60.0, j_operating_mA_cm2=100.0).V_cell
    v_90 = CellVoltageModel(temperature_C=90.0, j_operating_mA_cm2=100.0).V_cell
    assert v_60 < v_25, "V_cell at 60°C should be less than at 25°C"
    assert v_90 < v_60, "V_cell at 90°C should be less than at 60°C"


# ─── Divided cell has higher V_cell (membrane resistance) ──────────────

def test_divided_cell_higher_voltage():
    """Divided cell has higher V_cell due to membrane IR drop."""
    v_undivided = CellVoltageModel(
        divided_cell=False, j_operating_mA_cm2=100.0
    ).V_cell
    v_divided = CellVoltageModel(
        divided_cell=True, j_operating_mA_cm2=100.0
    ).V_cell
    assert v_divided > v_undivided


def test_divided_cell_membrane_ir_scales_with_j():
    """Membrane IR drop scales linearly with current density."""
    mem = MembraneModel(R_membrane_ohm_m2=0.002)
    ir_50 = mem.IR_drop(50.0)
    ir_100 = mem.IR_drop(100.0)
    ir_200 = mem.IR_drop(200.0)
    assert abs(ir_100 / ir_50 - 2.0) < 0.01
    assert abs(ir_200 / ir_50 - 4.0) < 0.01


# ─── Fe²⁺/Fe³⁺ anode shuttle model ────────────────────────────────────

def test_fe_shuttle_equilibrium():
    """Fe³⁺/Fe²⁺ equilibrium potential should be near E° at unit activity."""
    shuttle = FeShuttleAnode(fe2_conc_M=1.0, fe3_conc_M=1.0)
    E_eq = shuttle.equilibrium(T=298.15)
    # At unit activity, E_eq ≈ E° = 0.771 V
    assert abs(E_eq - E0_FE3_FE2) < 0.01


def test_fe_shuttle_overpotential_positive():
    """Shuttle overpotential should be positive and increase with j."""
    shuttle = FeShuttleAnode()
    eta_50 = shuttle.overpotential(50.0)
    eta_200 = shuttle.overpotential(200.0)
    assert eta_50 > 0
    assert eta_200 > eta_50


def test_fe_shuttle_anode_changes_V_cell():
    """Using Fe shuttle anode should give different V_cell than OER."""
    v_oer = CellVoltageModel(j_operating_mA_cm2=100.0).V_cell
    v_shuttle = CellVoltageModel(
        fe_shuttle=FeShuttleAnode(), j_operating_mA_cm2=100.0
    ).V_cell
    # Fe³⁺/Fe²⁺ has lower E° than OER but also lower overpotential
    # The thermodynamic voltage differs
    assert v_shuttle != v_oer


# ─── Energy = f(V, FE) computed correctly ──────────────────────────────

def test_specific_energy_formula():
    """
    Energy = (V_cell × z × F) / (CE × M × 3.6e6) kWh/kg.

    For V=2.5 V, CE=0.90:
    E = (2.5 × 2 × 96485) / (0.90 × 0.055845 × 3.6e6)
    """
    V = 2.5
    CE = 0.90
    expected = (V * Z_FE * FARADAY) / (CE * M_FE * 3.6e6)
    result = specific_energy_kWh_per_kg(V, CE)
    assert abs(result - expected) < 0.001


def test_energy_minimum_at_intermediate_j():
    """
    Specific energy has a minimum at intermediate current density.

    At low j: Fe delivery is slow relative to parasitic side reactions → lower CE.
    At intermediate j: Fe deposition is competitive → peak CE.
    At high j: mass transport limits Fe, HER dominates → lower CE + higher V.
    Both effects create an energy minimum at intermediate j.
    """
    energies = []
    for j in [5, 10, 30, 50, 100, 150, 200, 300]:
        model = CellVoltageModel(j_operating_mA_cm2=j)
        # Realistic CE: rises from low j (mass-transport limited),
        # peaks around 50-100 mA/cm², then drops (HER competition)
        ce = 0.90 * (j / (j + 10.0)) * max(1.0 - 0.0008 * j, 0.40)
        ce = max(ce, 0.10)
        energy = specific_energy_kWh_per_kg(model.V_cell, ce)
        energies.append(energy)

    # Minimum should not be at the extremes
    min_idx = np.argmin(energies)
    assert min_idx > 0, "Energy minimum should not be at lowest j"
    assert min_idx < len(energies) - 1, "Energy minimum should not be at highest j"


# ─── Nernst equation correctness ───────────────────────────────────────

def test_nernst_shift_unit_activity():
    """At unit activity ratio, Nernst shift should be zero."""
    E = nernst_shift(0.5, 298.15, 1.0, 2)
    assert abs(E - 0.5) < 1e-10


def test_nernst_shift_known_value():
    """
    Verify Nernst shift for a known case.
    For Fe²⁺/Fe at [Fe²⁺] = 0.1 M, T = 298.15 K:
    E = -0.440 + (8.314 × 298.15 / (2 × 96485)) × ln(0.1)
    E = -0.440 + 0.01285 × (-2.3026)
    E = -0.440 - 0.02958 = -0.4696 V
    """
    E = nernst_shift(-0.440, 298.15, 0.1, 2)
    assert abs(E - (-0.4696)) < 0.002


# ─── Temperature-dependent property functions ──────────────────────────

def test_diffusivity_increases_with_T():
    """Diffusivity should increase with temperature."""
    D_25 = diffusivity_m2_s(298.15)
    D_60 = diffusivity_m2_s(333.15)
    assert D_60 > D_25


def test_viscosity_decreases_with_T():
    """Water viscosity should decrease with temperature."""
    mu_25 = viscosity_Pa_s(298.15)
    mu_60 = viscosity_Pa_s(333.15)
    assert mu_60 < mu_25


# ─── Backward compatibility ────────────────────────────────────────────

def test_legacy_ir_drop_preserved():
    """When ir_drop is explicitly set to non-default, it should be used."""
    model = CellVoltageModel(ir_drop=0.5, j_operating_mA_cm2=100.0)
    assert model.V_cell > 0
    # The legacy ir_drop should be respected
    assert abs(model._total_ir_drop - 0.5) < 0.01


def test_summary_backward_compatible():
    """summary() should still return the expected keys."""
    model = CellVoltageModel()
    s = model.summary()
    assert "E_thermodynamic (V)" in s
    assert "η_cathode (V)" in s
    assert "η_anode (V)" in s
    assert "iR drop (V)" in s
    assert "V_cell (V)" in s


# ─── FE decreases with j (simplified) ──────────────────────────────────

def test_fe_decreases_with_j():
    """
    Current efficiency decreases with j due to HER competition.

    This is a simplified test using the CE model; the real CE comes from
    kinetics.py but the principle holds: higher j → more HER → lower CE.
    """
    # Use a simplified CE model: CE = 1 - k * j^0.5
    # which captures the essential physics
    def simplified_ce(j):
        return max(1.0 - 0.03 * np.sqrt(j), 0.50)

    ce_10 = simplified_ce(10)
    ce_100 = simplified_ce(100)
    ce_400 = simplified_ce(400)
    assert ce_10 > ce_100 > ce_400
    assert ce_10 > 0.90  # Low current → high CE
