"""Tests for the reference-cell theory-confidence (chain-of-claims) simulation.

These lock the screening claims: the reference operating point satisfies every
screening acceptance target, the thermal balance stays under the thermal limit
with active cooling, the three ledgers close within tolerance (and report
``partial`` honestly when a fixture stream is omitted), and the chain-of-claims
table covers all six NEXT_STEPS claims with claim 5 marked NOT COVERED.

Everything here is Level-0 screening: the assertions compare a predicted number
against a stated screening target, not gate evidence.
"""
from __future__ import annotations

import pandas as pd
import pytest

from models.cell_physics import CellPhysics
from models.electrochemistry import FARADAY, M_FE_G, Z_FE
from models.plating_data import PlatingDerived
from models.run_record import compute_ledgers
from models.theory_confidence import (
    REFERENCE_MIN_FE,
    RUN_DURATION_S,
    SCREENING_FLAG,
    close_ledgers,
    reference_cell,
    robustness_sweep,
    solve_reference,
    thermal_balance,
    chain_of_claims,
)


@pytest.fixture(scope="module")
def rc():
    return reference_cell()


@pytest.fixture(scope="module")
def op(rc):
    cp = CellPhysics(rc.bath, rc.geometry, rc.conditions)
    point = cp.find_optimal_j(min_FE=REFERENCE_MIN_FE)
    assert point is not None
    return point


@pytest.fixture(scope="module")
def solve(rc):
    return solve_reference(rc)


@pytest.fixture(scope="module")
def thermal(rc, op):
    return thermal_balance(rc, op)


@pytest.fixture(scope="module")
def ledger(rc, op):
    return close_ledgers(rc, op)


def _build_derived(rc, op):
    current_A = op.j_mA_cm2 * 1e-3 * rc.cathode_area_cm2
    charge_C = current_A * RUN_DURATION_S
    theoretical = charge_C * M_FE_G / (Z_FE * FARADAY)
    return PlatingDerived(
        charge_C=charge_C,
        duration_s=RUN_DURATION_S,
        mean_cathodic_current_A=current_A,
        mean_voltage_V=op.V_cell,
        energy_Wh=op.V_cell * current_A * RUN_DURATION_S / 3600.0,
        current_density_mA_cm2=op.j_mA_cm2,
        faradaic_efficiency=op.current_efficiency,
        faradaic_efficiency_percent=op.current_efficiency * 100.0,
        theoretical_fe_mass_g=theoretical,
        net_deposit_mass_g=op.current_efficiency * theoretical,
    )


@pytest.fixture(scope="module")
def derived(rc, op):
    return _build_derived(rc, op)


# ── 1. Reference operating point ────────────────────────────────────

class TestReferenceOperatingPoint:
    def test_all_screening_verdicts_pass(self, solve):
        assert solve["flag"] == SCREENING_FLAG
        assert solve["all_pass"] is True
        assert solve["transport_converged"] is True

    def test_fe_meets_screening_floor(self, solve, rc):
        v = solve["verdicts"]["fe"]
        assert v["pass"] is True
        assert solve["current_efficiency"] >= rc.targets.fe_min

    def test_v_cell_within_screening_window(self, solve, rc):
        v = solve["verdicts"]["v_cell"]
        assert v["pass"] is True
        assert rc.targets.v_cell_min <= solve["V_cell"] <= rc.targets.v_cell_max

    def test_specific_energy_under_route_threshold(self, solve, rc):
        v = solve["verdicts"]["specific_energy"]
        assert v["pass"] is True
        assert solve["specific_energy_kWh_t"] <= rc.targets.specific_energy_max_kWh_t

    def test_transport_limit_not_binding(self, solve, rc):
        v = solve["verdicts"]["transport_limit"]
        assert v["pass"] is True
        margin = solve["transport_limit_mA_cm2"] / solve["current_density_mA_cm2"]
        assert margin >= rc.targets.transport_margin_min
        assert solve["transport_limit_mA_cm2"] > solve["current_density_mA_cm2"]

    def test_deposition_rate_in_screening_window(self, solve, rc):
        v = solve["verdicts"]["deposition_rate"]
        assert v["pass"] is True
        assert (
            rc.targets.deposit_rate_min_um_hr
            <= solve["deposition_rate_um_hr"]
            <= rc.targets.deposit_rate_max_um_hr
        )

    def test_fe_is_a_fraction(self, solve):
        assert 0.0 <= solve["current_efficiency"] <= 1.0

    def test_reference_cell_is_immutable_and_explicit(self, rc):
        assert rc.cathode_area_cm2 == 200.0
        assert rc.geometry.membrane is True  # divided cell
        assert rc.bath.c_FeSO4_M == 1.0
        assert rc.conditions.temperature_C == 50.0


# ── 2. Thermal balance ──────────────────────────────────────────────

class TestThermalBalance:
    def test_steady_state_under_thermal_limit_with_active_cooling(self, thermal, rc):
        v = thermal["verdict"]
        assert v["pass"] is True
        assert thermal["steady_state_T_C"] <= rc.targets.thermal_limit_C
        assert thermal["steady_state_T_C"] <= thermal["max_T_C"]

    def test_heat_generation_is_positive(self, thermal):
        assert thermal["heat_gen_power_W"] > 0.0

    def test_joule_heat_dominates_over_activation(self, thermal):
        # IR/ohmic heat must exceed cathode+anode activation heat.
        assert thermal["joule_heat_W"] > thermal["activation_heat_W"]

    def test_cooling_reduces_steady_state(self, thermal):
        # Active jacket holds the cell cooler than the passive case.
        assert thermal["steady_state_T_C"] < thermal["steady_state_uncooled_T_C"]


# ── 3. Ledger closure ───────────────────────────────────────────────

class TestLedgerClosure:
    def test_charge_ledger_is_fe_specific_and_closes(self, ledger, rc):
        c = ledger["charge"]
        assert c["status"] == "partial_with_fe_deposit"
        assert c["missing"] == []
        assert c["residual_fraction"] is not None
        assert c["residual_fraction"] <= rc.targets.charge_residual_frac_tol
        # residual expressed as a fraction of applied charge
        assert c["unresolved_charge_C"] >= 0

    def test_iron_ledger_closes_within_tolerance(self, ledger, rc):
        i = ledger["iron"]
        assert i["status"] == "closed"
        assert i["residual_fraction"] is not None
        assert i["residual_fraction"] <= rc.targets.iron_residual_frac_tol
        assert abs(i["unaccounted_fe_mol"]) < 1e-6

    def test_energy_ledger_has_no_missing_components(self, ledger):
        e = ledger["energy"]
        assert e["status"] == "closed"
        assert e["missing_components"] == []
        assert e["stack_Wh"] > 0
        assert e["total_Wh"] > e["stack_Wh"]

    def test_all_ledgers_pass(self, ledger):
        assert ledger["all_pass"] is True


# ── 3b. Negative checks: the machinery reports partial honestly ─────

class TestLedgerHonesty:
    """Omitting a fixture stream must yield `partial` + non-empty `missing`.

    This guards against silently zero-filling unmeasured streams.
    """

    def test_missing_energy_log_reports_partial(self, derived):
        ledgers = compute_ledgers(derived)
        e = ledgers["energy"]
        assert e["status"] == "partial"
        assert e["missing_components"]  # non-empty

    def test_missing_bath_analysis_reports_iron_partial(self, derived):
        # Complete composition but missing the post-run analysis field.
        bath_batch = {
            "composition": {"fe2_g_L": 55.845, "volume_mL": 3000.0},
            "analysis": {"solids_fe_mol": 0.0, "other_fe_mol": 0.0},
        }
        characterization = pd.DataFrame(
            {"analyte": ["Fe"], "unit": ["wt%"], "technique": ["ICP"], "value": [100.0]}
        )
        ledgers = compute_ledgers(
            derived, bath_batch=bath_batch, characterization=characterization
        )
        i = ledgers["iron"]
        assert i["status"] == "partial"
        assert i["missing"]  # non-empty

    def test_energy_log_fixture_is_valid_per_contract(self, ledger):
        # The energy fixture used in the passing case passes run-record validation
        # (no unknown components / negative energy).
        assert ledger["energy"]["status"] == "closed"


# ── 4. Chain of claims ──────────────────────────────────────────────

class TestChainOfClaims:
    def test_renders_a_row_for_every_claim(self):
        rows = chain_of_claims()
        claims = {row["claim"] for row in rows}
        assert claims == {1, 2, 3, 4, 5, 6}

    def test_claim_5_is_not_covered(self):
        rows = chain_of_claims()
        row5 = next(r for r in rows if r["claim"] == 5)
        assert "NOT COVERED" in row5["verdict"]
        assert row5["predicted_value"] == "—"

    def test_every_row_has_full_schema(self):
        for row in chain_of_claims():
            for key in ("claim", "claim_text", "substantiated_by", "predicted_value",
                        "acceptance", "verdict"):
                assert key in row


# ── 5. Robustness (bonus, slow) ─────────────────────────────────────

class TestRobustnessSweep:
    @pytest.mark.slow
    def test_usable_window_is_wide_at_screening(self, rc):
        # Reduced grid (3×3) keeps the slow Nernst-Planck solve bounded.
        result = robustness_sweep(rc)
        assert result["n_total"] == 9
        assert result["usable_fraction"] >= 1 / 3  # screening, not a hard gate
