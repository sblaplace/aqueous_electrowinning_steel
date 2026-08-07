"""Tests for the cell-physics × gas-hold-up coupling layer.

Locks the claims of ``models/coupled_cell_physics.py``:

  - **Closure**: with the gas switched off (Faradaic gas fraction ≈ 0) the
    coupled energy reproduces the uncoupled ``cell_physics`` value — the
    coupling is faithful, not double-counting.
  - **Ordering**: with gas on, the coupled V_cell and coupled energy are ≥ the
    uncoupled ones and the gas ohmic penalty is strictly positive.
  - **Reachability**: ``coupled_energy_gate_reachable`` reports both flags and
    both minima over the same joint space as #40, so the delta is directly
    comparable.  The honest values are asserted; the verdict is derived, never
    forced.
  - **Reuse**: the coupling calls ``gas_holdup``'s own machinery
    (``bruggeman_conductivity``, ``solve_current_distribution``,
    ``solve_coupled``/``CoupledGasResult``) rather than reimplementing it.

Every assert checks a number against a threshold, a closure tolerance or an
ordering. All predicted values are Level-0 and are NOT gate evidence.
"""

from __future__ import annotations

import dataclasses

import pytest

from models import coupled_cell_physics as ccp
from models import gas_holdup
from models.cell_physics import CellPhysics
from models.economics_from_physics import (
    FE_TARGET_MIN,
    SPECIFIC_ENERGY_TARGET_MAX_KWH_T,
)
from models.electrochemistry import specific_energy_kWh_per_t
from models.gas_holdup import ChannelGeometry

#: #40's reachable operating point, the one this brief re-derives.
J_REF = 150.0
GAP_REF = 1.5e-3
CONTACT_REF = 1.0e-4
#: The coupled screen's current uncoupled minimum-energy result, kWh/t Fe.
# The DSA IrO2-Ta2O5 first-principles anode (Trasatti) and the per-surface
# Pitzer gamma are now the default.  This is a Level-0 regression pin for
# the current implementation, not a universal physical constant.
UNCOUPLED_MIN_ENERGY_40 = 3385.13


@pytest.fixture(scope="module")
def cell():
    return ccp.coupled_reference_cell(gap_m=GAP_REF, contact_resistance_ohm_m2=CONTACT_REF)


@pytest.fixture(scope="module")
def point(cell):
    return ccp.coupled_operating_point(cell, J_REF)


# ─── Construction ─────────────────────────────────────────────────────

def test_coupled_reference_cell_shares_rc1_geometry():
    c = ccp.coupled_reference_cell()
    # 10 cm² electrode = 50 mm x 20 mm, 3 mm deep channel (RC-1 design basis).
    assert c.channel.electrode_area_cm2 == pytest.approx(10.0, rel=1e-12)
    assert c.channel.height_m == pytest.approx(0.050, rel=1e-12)
    assert c.channel.width_m == pytest.approx(0.020, rel=1e-12)
    assert c.channel.depth_m == pytest.approx(0.003, rel=1e-12)
    assert c.electro.cathode_area_cm2 == pytest.approx(10.0, rel=1e-12)
    # The gap is the *same* number in both physics — that is the coupling point.
    assert c.channel.interelectrode_gap_m == pytest.approx(
        c.electro.geometry.interelectrode_gap_m, rel=1e-12
    )


def test_gap_and_contact_overrides_propagate_to_both_models():
    c = ccp.coupled_reference_cell(gap_m=GAP_REF, contact_resistance_ohm_m2=CONTACT_REF)
    assert c.electro.geometry.interelectrode_gap_m == pytest.approx(GAP_REF, rel=1e-12)
    assert c.electro.interelectrode_gap_m == pytest.approx(GAP_REF, rel=1e-12)
    assert c.channel.interelectrode_gap_m == pytest.approx(GAP_REF, rel=1e-12)
    assert c.electro.geometry.contact_resistance_ohm_m2 == pytest.approx(CONTACT_REF, rel=1e-12)


@pytest.mark.parametrize("kwargs", [{"gap_m": 0.0}, {"gap_m": -1e-3},
                                    {"contact_resistance_ohm_m2": -1.0}])
def test_invalid_construction_rejected(kwargs):
    with pytest.raises(ValueError):
        ccp.coupled_reference_cell(**kwargs)


def test_non_positive_current_density_rejected(cell):
    with pytest.raises(ValueError):
        ccp.coupled_operating_point(cell, 0.0)


# ─── Reuse of gas_holdup's own machinery (drift guard) ────────────────

def test_coupling_reuses_gas_holdup_functions():
    """The coupling must *bridge* gas_holdup, not reimplement it."""
    assert ccp.bruggeman_conductivity is gas_holdup.bruggeman_conductivity
    assert ccp.solve_current_distribution is gas_holdup.solve_current_distribution
    assert ccp.solve_coupled is gas_holdup.solve_coupled
    assert ccp.holdup_profile is gas_holdup.holdup_profile
    assert ccp.drift_flux_void_fraction is gas_holdup.drift_flux_void_fraction
    assert ccp.CoupledGasResult is gas_holdup.CoupledGasResult
    assert ccp.ChannelGeometry is gas_holdup.ChannelGeometry


def test_coupled_gas_state_returns_upstream_result_type(cell):
    physics = CellPhysics(cell.electro.bath, cell.electro.geometry, cell.electro.conditions)
    kappa = physics.solve_at_j(J_REF).conductivity_S_m
    state = ccp.coupled_gas_state(cell, J_REF, kappa_S_m=kappa)
    assert isinstance(state, gas_holdup.CoupledGasResult)
    assert isinstance(state.profile, gas_holdup.HoldupProfile)
    # The upstream fixed point actually converged, inside its own tolerance.
    assert state.converged is True
    assert 1 <= state.iterations <= ccp.MAX_ITERATIONS
    assert state.profile.void_fraction.size == ccp.N_SEGMENTS


def test_bruggeman_penalty_matches_upstream_function(cell):
    impact = ccp.gas_impact_summary(cell, J_REF)
    expected = gas_holdup.bruggeman_conductivity(
        impact["kappa_bulk_S_m"], impact["mean_void_fraction"]
    )
    assert impact["kappa_eff_at_mean_void_S_m"] == pytest.approx(expected, rel=1e-12)
    assert impact["kappa_eff_at_mean_void_S_m"] < impact["kappa_bulk_S_m"]


# ─── Closure: no gas ⇒ coupled == uncoupled ───────────────────────────

def test_null_gas_case_reproduces_uncoupled_energy(cell):
    """Faithfulness: with the Faradaic gas fraction zero, coupling is a no-op."""
    off = ccp.coupled_operating_point(cell, J_REF, gas_off=True)
    assert off["mean_void_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert off["outlet_void_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert off["conductivity_penalty"] == pytest.approx(1.0, abs=1e-9)
    assert abs(off["gas_ohmic_penalty_V"]) < 1e-6            # < 1 µV
    assert off["V_cell_coupled"] == pytest.approx(off["V_cell_uncoupled"], abs=1e-6)
    # Coupled energy converges to the uncoupled cell_physics value within 0.01 kWh/t.
    assert off["coupled_specific_energy_kWh_t"] == pytest.approx(
        off["uncoupled_specific_energy_kWh_t"], abs=1e-2
    )


def test_voltage_ledger_closes(cell):
    """V_uncoupled + ohmic_penalty − V_coupled ≈ 0: added once, nothing else."""
    closure = ccp.coupling_closure(cell, J_REF)
    assert closure["voltage_residual_V"] < 1e-12
    assert abs(closure["null_gas_ohmic_penalty_V"]) < 1e-6
    assert closure["null_gas_energy_residual_kWh_t"] < 1e-2
    assert closure["null_gas_void_fraction"] == pytest.approx(0.0, abs=1e-12)


def test_uncoupled_branch_matches_cell_physics_exactly(cell, point):
    """The uncoupled numbers reported are cell_physics's own, unmodified."""
    physics = CellPhysics(cell.electro.bath, cell.electro.geometry, cell.electro.conditions)
    ref = physics.solve_at_j(J_REF)
    assert point["V_cell_uncoupled"] == pytest.approx(ref.V_cell, rel=1e-12)
    assert point["FE_uncoupled"] == pytest.approx(ref.current_efficiency, rel=1e-12)
    assert point["uncoupled_specific_energy_kWh_t"] == pytest.approx(
        ref.specific_energy_kWh_t, rel=1e-12
    )
    # And the program energy identity E = 959.9 * V / FE holds on both branches.
    assert point["coupled_specific_energy_kWh_t"] == pytest.approx(
        specific_energy_kWh_per_t(point["V_cell_coupled"], point["FE_uncoupled"]), rel=1e-12
    )


# ─── Ordering: gas only ever hurts ────────────────────────────────────

def test_gas_on_penalises_voltage_and_energy(point):
    assert point["gas_ohmic_penalty_V"] > 0.0
    assert point["V_cell_coupled"] >= point["V_cell_uncoupled"]
    assert point["coupled_specific_energy_kWh_t"] >= point["uncoupled_specific_energy_kWh_t"]
    assert point["energy_delta_kWh_t"] >= 0.0
    assert point["delta_V"] == pytest.approx(point["gas_ohmic_penalty_V"], rel=1e-12)
    # Void fraction is physical and the axial profile accumulates upward.
    assert 0.0 < point["mean_void_fraction"] < 1.0
    assert point["outlet_void_fraction"] > point["mean_void_fraction"]
    # Bruggeman can only reduce conductivity, and redistribution can only spread j.
    assert point["conductivity_penalty"] > 1.0
    assert 0.0 < point["current_uniformity"] <= 1.0


def test_gas_penalty_grows_with_current_density(cell):
    low = ccp.coupled_operating_point(cell, 150.0)
    high = ccp.coupled_operating_point(cell, 300.0)
    assert high["gas_ohmic_penalty_V"] > low["gas_ohmic_penalty_V"]
    assert high["mean_void_fraction"] > low["mean_void_fraction"]
    assert high["energy_delta_kWh_t"] > low["energy_delta_kWh_t"]


def test_gas_penalty_grows_with_channel_height(cell):
    """The 1-D axial term is a scale-up effect: RC-1 is short, a plant cell is not."""
    tall = dataclasses.replace(
        cell,
        channel=ChannelGeometry(
            height_m=1.0,
            width_m=1.0,
            depth_m=cell.channel.depth_m,
            interelectrode_gap_m=cell.channel.interelectrode_gap_m,
            liquid_flow_L_min=cell.channel.liquid_flow_L_min,
        ),
    )
    short_pt = ccp.coupled_operating_point(cell, 300.0)
    tall_pt = ccp.coupled_operating_point(tall, 300.0)
    assert tall_pt["outlet_void_fraction"] > 10.0 * short_pt["outlet_void_fraction"]
    assert tall_pt["gas_ohmic_penalty_V"] > 10.0 * short_pt["gas_ohmic_penalty_V"]
    assert tall_pt["energy_delta_kWh_t"] > short_pt["energy_delta_kWh_t"]
    assert tall_pt["current_uniformity"] < short_pt["current_uniformity"]


def test_transport_limit_is_honoured(point):
    assert point["transport_limit_mA_cm2"] > point["j_mA_cm2"]
    assert point["transport_margin"] == pytest.approx(
        point["transport_limit_mA_cm2"] / point["j_mA_cm2"], rel=1e-12
    )
    assert point["valid"] is True


# ─── Gas impact summary ───────────────────────────────────────────────

def test_gas_impact_summary_numbers(cell):
    impact = ccp.gas_impact_summary(cell, J_REF)
    assert 0.0 < impact["mean_void_fraction"] < 0.05
    assert impact["outlet_void_fraction"] > impact["mean_void_fraction"]
    assert impact["conductivity_penalty"] > 1.0
    assert impact["ohmic_penalty_V"] > 0.0
    assert impact["ohmic_penalty_mV"] == pytest.approx(
        impact["ohmic_penalty_V"] * 1000.0, rel=1e-12
    )
    assert 0.0 < impact["gas_share_of_V_cell_pct"] < 100.0
    assert impact["energy_contribution_kWh_t"] >= 0.0
    assert impact["hydrogen_flow_L_h_wet"] > 0.0
    # Bubble sizing is the dominant lever; the departure diameter must be
    # physical (tens to hundreds of µm) for the penalty to mean anything.
    assert 1.0 < impact["bubble_diameter_um"] < 1000.0
    assert impact["flag"] == "unvalidated (L0)"
    assert "NOT gate evidence" in impact["boundary_note"]


def test_fe_stays_a_derived_output(point):
    """FE is CellPhysics's, shifted only by the model's own bubble coupling."""
    assert 0.0 < point["FE_uncoupled"] <= 1.0
    assert 0.0 < point["FE_coupled"] <= 1.0
    assert point["FE_shift_pp"] == pytest.approx(
        (point["FE_coupled"] - point["FE_uncoupled"]) * 100.0, abs=0.05
    )
    assert abs(point["FE_shift_pp"]) < 5.0     # a screening-scale shift, not a tuning knob
    assert point["FE_coupled"] >= FE_TARGET_MIN


def test_headline_coupled_energy_is_the_conservative_one(point):
    """The headline uses the uncoupled FE, so gas can never flatter the gate."""
    assert point["coupled_specific_energy_kWh_t"] >= point["uncoupled_specific_energy_kWh_t"]
    assert point["coupled_energy_with_FE_shift_kWh_t"] == pytest.approx(
        specific_energy_kWh_per_t(point["V_cell_coupled"], point["FE_coupled"]), rel=1e-12
    )


# ─── The headline: reachability under coupling ────────────────────────

@pytest.fixture(scope="module")
def reach():
    return ccp.coupled_energy_gate_reachable()


@pytest.mark.slow
def test_coupled_window_covers_the_same_joint_space_as_40(reach):
    window = reach["window"]
    assert len(window) == len(ccp.J_VALUES_MA_CM2) * len(ccp.GAP_VALUES_M) \
        * len(ccp.MEMBRANE_VALUES_OHM_M2) * len(ccp.CONTACT_VALUES_OHM_M2)
    assert {r["j_mA_cm2"] for r in window} == set(ccp.J_VALUES_MA_CM2)
    assert {r["interelectrode_gap_m"] for r in window} == set(ccp.GAP_VALUES_M)
    assert {r["contact_resistance_ohm_m2"] for r in window} == set(ccp.CONTACT_VALUES_OHM_M2)
    for row in window:
        assert row["flag"] == "unvalidated (L0)"
        if row["valid"]:
            assert row["coupled_specific_energy_kWh_t"] >= row["uncoupled_specific_energy_kWh_t"]
            assert row["gas_ohmic_penalty_V"] > 0.0
            assert row["transport_margin"] > 1.0


@pytest.mark.slow
def test_coupled_energy_gate_reachable_reports_both_flags_and_minima(reach):
    for key in ("uncoupled_min_energy", "coupled_min_energy",
                "reachable_uncoupled", "reachable_coupled", "energy_delta"):
        assert key in reach
    assert isinstance(reach["reachable_uncoupled"], bool)
    assert isinstance(reach["reachable_coupled"], bool)
    assert reach["energy_gate_kWh_t"] == pytest.approx(SPECIFIC_ENERGY_TARGET_MAX_KWH_T)

    # Coupling can only degrade the minimum, never improve it.
    assert reach["coupled_min_energy"] >= reach["uncoupled_min_energy"]
    assert reach["energy_delta"] == pytest.approx(
        reach["coupled_min_energy"] - reach["uncoupled_min_energy"], rel=1e-9
    )
    assert reach["energy_delta"] >= 0.0

    # The flags are consistent with the numbers against the 4,000 kWh/t gate.
    assert reach["reachable_coupled"] == (
        reach["coupled_min_energy"] <= SPECIFIC_ENERGY_TARGET_MAX_KWH_T
    )
    assert reach["reachable_uncoupled"] == (
        reach["uncoupled_min_energy"] <= SPECIFIC_ENERGY_TARGET_MAX_KWH_T
    )
    # Coupling can never turn a failing verdict into a passing one.
    if reach["reachable_coupled"]:
        assert reach["reachable_uncoupled"]


@pytest.mark.slow
def test_uncoupled_branch_reproduces_the_40_headline(reach):
    """The uncoupled leg of the comparison is #40's own number, unshifted."""
    assert reach["uncoupled_min_energy"] == pytest.approx(UNCOUPLED_MIN_ENERGY_40, abs=1.0)
    assert reach["optimizer_min_energy_reference"] == pytest.approx(
        reach["uncoupled_min_energy"], abs=1.0
    )
    assert reach["optimizer_reachable_reference"] is True


@pytest.mark.slow
def test_coupled_verdict_is_the_honest_derived_answer(reach):
    """The #40 point survives coupling in this model — asserted, not forced.

    The coupled minimum sits at 150 mA/cm², 1.5 mm gap, 1e-4 Ω·m² contact, and
    the gas penalty at RC-1's 50 mm channel is sub-kWh/t: the reachability
    verdict is unchanged, and the delta is small *because the channel is
    short*, not because the penalty was tuned away.
    """
    assert reach["reachable_coupled"] is True
    assert reach["coupled_min_energy"] <= SPECIFIC_ENERGY_TARGET_MAX_KWH_T
    assert reach["coupled_min_energy"] == pytest.approx(UNCOUPLED_MIN_ENERGY_40, abs=5.0)
    assert 0.0 < reach["energy_delta"] < 5.0
    assert 0.0 < reach["energy_delta_pct"] < 1.0

    best = reach["best_combination_coupled"]
    assert best["j_mA_cm2"] == pytest.approx(J_REF)
    assert best["interelectrode_gap_m"] == pytest.approx(GAP_REF)
    assert best["contact_resistance_ohm_m2"] == pytest.approx(CONTACT_REF)
    assert best["FE_uncoupled"] >= FE_TARGET_MIN
    assert "SURVIVES" in reach["verdict"]
    assert reach["flag"] == "unvalidated (L0)"


@pytest.mark.slow
def test_main_runs_and_returns_consistent_report():
    report = ccp.main()
    reach = report["reachability"]
    impact = report["gas_impact"]
    closure = report["closure"]
    assert reach["coupled_min_energy"] >= reach["uncoupled_min_energy"]
    assert impact["ohmic_penalty_V"] > 0.0
    assert closure["voltage_residual_V"] < 1e-12
