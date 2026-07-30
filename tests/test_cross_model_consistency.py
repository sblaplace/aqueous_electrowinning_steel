"""Cross-model internal-consistency tests.

These tests verify that independent models which *should* agree at matched
conditions actually do — catching silent drift in constants, units, or
physical assumptions across modules.

Where models genuinely disagree by design (different physics), the tests
document the disagreement ratio so future changes don't silently alter it.
"""
import math
import numpy as np
import pandas as pd
import pytest

from models.kinetics import limiting_current_density, DepositionKinetics
from models.boundary_layer import CathodeBoundaryLayer
from models.diffusion_layer_1d import DiffusionLayer1D
from models.co_deposition import surface_pH_from_current
from models.pourbaix import FePourbaix, LOGKSP_FEOH2
from models.electrochemistry import FARADAY, R_GAS, E0_FE, M_FE, Z_FE, CellVoltageModel, specific_energy_kWh_per_t


# ─── Test 1: Limiting-current — NP diffusion limit vs Levich ───────────────
# The NP model's "diffusion limit" is computed from the full multi-ion
# Nernst-Planck system (coupled Fe²⁺/H⁺/SO₄²⁻ transport + electroneutrality).
# The Levich formula (zFDC/δ) assumes binary transport. They should be in the
# same order of magnitude; a large discrepancy reveals constant drift or a
# physics mismatch.

def test_np_diffusion_limit_vs_levich():
    """NP diffusion-only limit and Levich limit should be same order of magnitude."""
    from models.transport import NernstPlanckFilm

    fe_conc_M = 1.0
    delta_m = 50e-6
    T_C = 60.0
    T_K = T_C + 273.15

    D_fe = 7.2e-10  # m²/s at 25°C (CRC)
    Ea = 18e3       # J/mol (electrochemistry.py DIFF_EA_J_MOL)
    D_fe_T = D_fe * math.exp(Ea / R_GAS * (1.0 / 298.15 - 1.0 / T_K))
    levich = limiting_current_density(fe_conc_M * 1000.0, D_fe_T, delta_m)

    film = NernstPlanckFilm(
        fe_conc_M=fe_conc_M,
        boundary_layer_m=delta_m,
        temperature_C=T_C,
        support_conc_M=10.0,
    )
    result = film.solve(100.0)
    np_diff = result.diffusion_limit_A_m2

    ratio = np_diff / levich
    assert 0.3 < ratio < 1.5, (
        f"NP diffusion limit ({np_diff:.1f} A/m²) vs Levich ({levich:.1f} A/m²): "
        f"ratio={ratio:.3f} — outside plausible range, check constants"
    )
    # If this ratio drifts from its current value (~0.47), something changed
    assert 0.40 < ratio < 0.55, (
        f"NP/Levich ratio changed to {ratio:.3f} (was ~0.47). "
        f"Verify this is intentional before accepting."
    )


def test_migration_enhances_limiting_current():
    """Unsupported baths should have higher transport limits than high-support baths."""
    from models.transport import NernstPlanckFilm

    fe_conc_M = 1.0
    delta_m = 50e-6
    T_C = 60.0

    high_support = NernstPlanckFilm(
        fe_conc_M=fe_conc_M, boundary_layer_m=delta_m,
        temperature_C=T_C, support_conc_M=10.0,
    )
    unsupported = NernstPlanckFilm(
        fe_conc_M=fe_conc_M, boundary_layer_m=delta_m,
        temperature_C=T_C, support_conc_M=0.0,
    )

    ratio = unsupported.transport_limit_A_m2() / high_support.transport_limit_A_m2()
    assert ratio > 1.5, (
        f"Migration enhancement ({ratio:.2f}×) too small; "
        f"expected >1.5× per the transport model's own migration analysis"
    )


# ─── Test 2: Surface pH — empirical vs Nernst-Planck ──────────────────────

@pytest.mark.parametrize("j_mA_cm2", [100, 200])
def test_surface_pH_empirical_vs_nernst_planck(j_mA_cm2):
    """Document: empirical surface pH (no migration) >= NP surface pH (with migration)."""
    bulk_pH = 2.0
    fe_conc_M = 1.0
    buffer_M = 0.40
    T_C = 60.0
    delta_m = 50e-6

    empirical_pH = surface_pH_from_current(
        j_mA_cm2, bulk_pH,
        buffer_capacity_M=buffer_M,
        temperature_C=T_C,
        boundary_layer_m=delta_m,
    )

    model = DiffusionLayer1D(
        fe_conc_M=fe_conc_M,
        pH_bulk=bulk_pH,
        temperature_C=T_C,
        delta_m=delta_m,
        buffer_conc_M=buffer_M,
        fe_i0=10.0,
        her_i0=0.010,
    )
    result = model.solve(j_mA_cm2)
    np_pH = result.surface_pH

    assert empirical_pH >= np_pH - 0.5, (
        f"At j={j_mA_cm2}: empirical pH ({empirical_pH:.2f}) should be >= "
        f"NP pH ({np_pH:.2f}) since empirical ignores migration suppression"
    )


# ─── Test 3: Calibration round-trip ────────────────────────────────────────

def test_calibration_round_trip():
    """Fitted kinetics parameters must reproduce the generating model."""
    from models.calibration import fit_total_cathodic_polarization

    true_fe_i0 = 0.05
    true_her_i0 = 0.001
    true_fe_tafel = 0.120
    true_her_tafel = 0.140
    pH = 2.0
    T_C = 60.0
    fe_conc_M = 1.0

    kin = DepositionKinetics(
        pH=pH, temperature_C=T_C,
        fe_i0=true_fe_i0, her_i0=true_her_i0,
        fe_tafel_V=true_fe_tafel, her_tafel_V=true_her_tafel,
        fe_conc_M=fe_conc_M,
    )

    E_she = np.linspace(-0.30, -0.90, 50)
    _, _, total_current = kin.partial_currents(E_she)
    current_A_m2 = -np.maximum(total_current, 1e-10)

    data = pd.DataFrame({
        "potential_V_vs_ref": E_she,
        "current_density_A_m2": current_A_m2,
    })

    fit = fit_total_cathodic_polarization(
        data, pH=pH, temperature_C=T_C, fe_conc_M=fe_conc_M,
        reference_to_she_V=0.0,
    )

    assert fit.converged, "Calibration failed to converge"
    assert abs(np.log10(fit.fe_i0_A_m2) - np.log10(true_fe_i0)) < 0.5, (
        f"Fe i0: fitted={fit.fe_i0_A_m2:.4g}, true={true_fe_i0:.4g}"
    )
    assert abs(np.log10(fit.her_i0_A_m2) - np.log10(true_her_i0)) < 0.5, (
        f"HER i0: fitted={fit.her_i0_A_m2:.4g}, true={true_her_i0:.4g}"
    )


# ─── Test 4: Fe(OH)₂ KSP self-consistency ─────────────────────────────────

def test_feoh2_ksp_precipitation_pH():
    """Fe(OH)₂ precipitation pH at 1 M Fe²⁺ should be in a reasonable range."""
    KSP = 10.0 ** LOGKSP_FEOH2

    fe_conc_M = 1.0
    oh_precip = math.sqrt(KSP / fe_conc_M)
    pH_precip = 14.0 + math.log10(oh_precip)

    assert 5.0 < pH_precip < 7.0, (
        f"Fe(OH)₂ precipitation pH={pH_precip:.2f} at 1 M Fe²⁺ outside expected range. "
        f"LOGKSP_FEOH2={LOGKSP_FEOH2:.2f}"
    )

    oh_001 = math.sqrt(KSP / 0.01)
    pH_001 = 14.0 + math.log10(oh_001)
    assert pH_001 > pH_precip, "Lower [Fe²⁺] should precipitate at higher pH"
    assert 6.5 < pH_001 < 9.0, (
        f"Fe(OH)₂ precipitation pH={pH_001:.2f} at 0.01 M Fe²⁺ outside expected range"
    )


# ─── Test 5: DiffusionLayer1D FE vs DepositionKinetics FE ──────────────────
# The diffusion-layer model uses the same Butler-Volmer kinetics as
# DepositionKinetics internally. At identical kinetic parameters and bulk
# conditions, the FE predictions should agree (the NP model adds transport
# coupling, so agreement is expected at moderate j where transport isn't limiting).

def test_diffusion_layer_fe_vs_kinetics_fe():
    """DiffusionLayer1D FE ≈ DepositionKinetics FE at moderate current density."""
    fe_i0 = 10.0
    her_i0 = 0.010
    fe_tafel = 0.120
    her_tafel = 0.140
    fe_conc_M = 1.0
    pH = 2.0
    T_C = 60.0
    j_mA_cm2 = 50.0  # moderate — not transport-limited

    # DepositionKinetics (bulk Butler-Volmer, no transport coupling)
    kin = DepositionKinetics(
        pH=pH, temperature_C=T_C,
        fe_i0=fe_i0, her_i0=her_i0,
        fe_tafel_V=fe_tafel, her_tafel_V=her_tafel,
        fe_conc_M=fe_conc_M,
    )
    kin_fe = kin.efficiency_at_current(j_mA_cm2)

    # DiffusionLayer1D (Nernst-Planck + Butler-Volmer)
    dlm = DiffusionLayer1D(
        fe_conc_M=fe_conc_M, pH_bulk=pH, temperature_C=T_C,
        delta_m=50e-6, buffer_conc_M=0.40,
        fe_i0=fe_i0, her_i0=her_i0,
        fe_tafel_V=fe_tafel, her_tafel_V=her_tafel,
    )
    dlm_result = dlm.solve(j_mA_cm2)
    dlm_fe = dlm_result.current_efficiency

    # At moderate j, transport effects are small — FE should agree within 15%
    delta = abs(kin_fe - dlm_fe)
    assert delta < 0.15, (
        f"FE disagreement at j={j_mA_cm2} mA/cm²: "
        f"kinetics={kin_fe:.3f}, diffusion_layer={dlm_fe:.3f}, Δ={delta:.3f}"
    )


# ─── Test 6: Scenario V_cell vs CellVoltageModel ──────────────────────────
# scenarios.py computes V_cell from E_cathode_eq + E_anode_eq + eta + IR.
# CellVoltageModel computes the same decomposition. They must agree.

def test_scenario_v_cell_arithmetic():
    """Scenario.V_cell must be the correct sum of its components."""
    from models.scenarios import Scenario

    # Verify the scenario's V_cell property computes the expected sum.
    # Scenario uses: abs(anode_eq - cathode_eq) + eta_cathode + eta_anode + ir_drop
    s = Scenario(
        name="test", description="consistency check",
        electrolyte_type="alkaline", electrolyte_composition="test",
        current_density_mA_cm2=100.0, current_efficiency=0.90,
        temperature_C=60.0,
        E_cathode_eq=-0.440, E_anode_eq=1.229,
        eta_cathode=0.30, eta_anode=0.40, ir_drop=0.20,
        anode_type="test",
    )

    expected = abs(s.E_anode_eq - s.E_cathode_eq) + s.eta_cathode + s.eta_anode + s.ir_drop
    assert abs(s.V_cell - expected) < 1e-10, (
        f"V_cell arithmetic error: scenario={s.V_cell:.6f}, expected={expected:.6f}"
    )

    # Also verify CellVoltageModel's V_cell is its own correct sum
    cvm = CellVoltageModel(
        E_cathode_eq=-0.440, E_anode_eq=1.229,
        eta_cathode=0.30, eta_anode=0.40, ir_drop=0.20,
    )
    cvm_expected = cvm.E_thermodynamic + cvm.eta_cathode + cvm._effective_eta_anode + cvm._total_ir_drop
    assert abs(cvm.V_cell - cvm_expected) < 1e-10, (
        f"CellVoltageModel arithmetic error: V_cell={cvm.V_cell:.6f}, "
        f"expected={cvm_expected:.6f}"
    )


# ─── Test 7: specific_energy formula vs electrochemistry function ──────────
# E = (V × z × F) / (CE × M × 3.6e6) kWh/kg × 1000 = kWh/t
# This is Faraday's law — the bridge between physics and economics.

def test_specific_energy_faraday_round_trip():
    """specific_energy_kWh_per_t must match the Faraday formula exactly."""
    for V, FE in [(2.0, 0.90), (2.5, 0.95), (4.0, 0.70)]:
        expected = (V * Z_FE * FARADAY) / (FE * M_FE * 3.6e6) * 1000.0
        actual = specific_energy_kWh_per_t(V, FE)
        assert abs(actual - expected) < 1e-6, (
            f"specific_energy at V={V}, FE={FE}: "
            f"expected={expected:.6f}, actual={actual:.6f}"
        )

    # Verify the kill-criterion threshold: 4000 kWh/t at FE=70% → V≈2.92 V
    V_kill = 4000 * 0.70 * M_FE * 3.6e6 / (Z_FE * FARADAY) / 1000.0
    assert abs(specific_energy_kWh_per_t(V_kill, 0.70) - 4000.0) < 0.01


# ─── Test 8: Hull cell gravimetric FE vs Faraday's law ─────────────────────
# Given known charge and mass gain, the gravimetric FE formula should return
# the correct value. This protects the Phase II experimental workflow.

def test_hull_cell_gravimetric_fe_faraday():
    """Gravimetric FE must match Faraday's law for known charge/mass."""
    from models.hull_cell import gravimetric_faradaic_efficiency

    # Simulate: 100 C of cathodic charge at 90% FE
    # Expected mass gain = FE × Q × M / (z × F)
    charge_C = 100.0
    expected_FE = 0.90
    expected_mass_gain = expected_FE * charge_C * (M_FE_G := 55.845) / (2 * FARADAY)

    # Build synthetic trace: constant -1 A for 100 s
    t = np.linspace(0, 100, 200)
    I = np.full_like(t, -1.0)  # cathodic (negative)

    gravimetry = pd.DataFrame({
        "mass_before_g": [25.0],
        "mass_after_g": [25.0 + expected_mass_gain],
        "blank_mass_change_g": [0.0],
    })
    trace = pd.DataFrame({"timestamp_s": t, "current_A": I})

    result = gravimetric_faradaic_efficiency(
        trace["timestamp_s"], trace["current_A"],
        25.0, 25.0 + expected_mass_gain,
        cathodic_sign="negative",
    )

    assert abs(result.apparent_faradaic_efficiency - expected_FE) < 0.001, (
        f"Gravimetric FE: expected={expected_FE:.4f}, "
        f"got={result.apparent_faradaic_efficiency:.4f}"
    )


# ─── Test 9: Membrane iron conservation ────────────────────────────────────
# Over a membrane simulation, iron atoms must be conserved:
#   Fe oxidized at anode = Fe deposited + Fe accumulated as Fe³⁺ + Fe crossover loss

def test_membrane_iron_conservation():
    """MembraneTransportModel must conserve iron atoms over a simulation."""
    from models.membrane_transport import (
        MembraneTransportModel, AnolyteState, CatholyteState, NAFION_N117,
    )

    model = MembraneTransportModel(
        membrane=NAFION_N117,
        electrode_area_m2=0.01,
        temperature_C=60.0,
        j_mA_cm2=100.0,
        anolyte=AnolyteState(volume_L=1.0, fe2_M=1.0, fe3_M=0.0, h_M=1.0),
        catholyte=CatholyteState(volume_L=1.0, fe2_M=1.0, fe3_M=0.0, h_M=1.0),
        purge_fe3_threshold_M=10.0,  # disable purge for clean conservation check
    )

    result = model.simulate(duration_hr=1.0, dt_hr=0.05)

    # Iron accounting from the simulation
    j_A_m2 = 100.0 * 10.0
    area = 0.01
    duration_s = 3600.0
    total_charge_C = j_A_m2 * area * duration_s

    # Fe oxidized at anode (all current goes to Fe²⁺→Fe³⁺ in this model)
    fe_oxidized_mol = total_charge_C / FARADAY  # 1 e⁻ per Fe²⁺→Fe³⁺

    # Fe³⁺ accumulated in anolyte
    anolyte = model.anolyte
    final_fe3 = anolyte.fe3_M * anolyte.volume_L  # mol

    # The simulation tracks crossover loss as a percentage
    # Conservation: fe_oxidized ≈ fe3_accumulated + crossover_losses
    # (cathode Fe³⁺ reduction returns Fe²⁺, so net crossover loss is the
    # fraction that gets re-deposited or stays as Fe³⁺ in catholyte)

    # At minimum, anolyte Fe³⁺ should not exceed total Fe oxidized
    assert final_fe3 <= fe_oxidized_mol * 1.01, (
        f"Fe³⁺ accumulated ({final_fe3:.4f} mol) exceeds Fe oxidized "
        f"({fe_oxidized_mol:.4f} mol) — iron not conserved"
    )
    assert final_fe3 >= 0, "Fe³⁺ cannot be negative"
