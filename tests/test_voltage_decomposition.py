"""Tests for the additive Level-0 voltage-decomposition screen.

These tests protect closure and ordering invariants in the transparent model;
they do not turn any predicted value into measurement or gate evidence.
"""

import pytest

from models.voltage_decomposition import (
    buy_next_measurement,
    decompose_at,
    lever_sensitivity,
    rank_levers,
    reference_cell,
)

J_REFERENCE = 300.0
ENERGY_GATE_KWH_T = 4000.0


@pytest.fixture(scope="module")
def cell():
    return reference_cell()


@pytest.fixture(scope="module")
def decomposition(cell):
    return decompose_at(cell, J_REFERENCE)


@pytest.fixture(scope="module")
def sensitivities(cell):
    return lever_sensitivity(cell, J_REFERENCE)


def test_decomposition_closes_and_energy_identity_holds(decomposition):
    components = (
        decomposition["E_thermodynamic"]
        + decomposition["eta_cathode"]
        + decomposition["eta_anode"]
        + decomposition["IR_total"]
    )
    assert abs(components - decomposition["V_cell"]) <= 1e-6
    assert decomposition["IR_total"] == pytest.approx(
        decomposition["IR_electrolyte"]
        + decomposition["IR_membrane"]
        + decomposition["IR_contacts"],
        abs=1e-6,
    )
    assert decomposition["V_cell"] > 0.0
    assert decomposition["FE"] > 0.0
    assert decomposition["specific_energy_kWh_t"] > 0.0
    assert decomposition["specific_energy_kWh_t"] == pytest.approx(
        959.9 * decomposition["V_cell"] / decomposition["FE"],
        rel=1e-4,
    )


def test_ohmic_drop_and_contacts_are_the_dominant_screening_components(decomposition):
    kinetic_and_thermo = (
        decomposition["E_thermodynamic"]
        + decomposition["eta_cathode"]
        + decomposition["eta_anode"]
    )
    assert decomposition["IR_total"] >= kinetic_and_thermo
    assert decomposition["IR_contacts"] > decomposition["IR_membrane"]
    assert decomposition["IR_contacts"] > decomposition["IR_electrolyte"]
    assert decomposition["IR_contacts"] > 1.0
    assert decomposition["IR_membrane"] > 0.5
    assert decomposition["IR_electrolyte"] > 0.1


def test_single_lever_scenarios_reduce_voltage_and_energy(sensitivities, decomposition):
    expected_levers = {
        "contact resistance": (5.0e-4, 1.0e-4),
        "membrane area resistance": (3.0e-4, 1.5e-4),
        "electrode gap": (3.0e-3, 1.5e-3),
        "anode bubble fraction": (0.10, 0.05),
        # The anode lever is modeled within the DSA kinetics: current_value is
        # the DSA's effective overpotential at the reference point, proposed_value
        # is the derived overpotential of the preferred-OER-catalyst anode
        # (100x the DSA exchange current density, 0.1 A/m2).
        "anode overpotential": (0.680616297488156, 0.560616297486031),
    }
    assert len(sensitivities) == len(expected_levers)
    for row in sensitivities:
        assert row["lever"] in expected_levers
        current, proposed = expected_levers[row["lever"]]
        assert row["current_value"] == pytest.approx(current, abs=1e-12)
        assert row["proposed_value"] == pytest.approx(proposed, abs=1e-12)
        assert row["delta_V"] > 0.0
        assert row["V_after"] < decomposition["V_cell"]
        assert row["energy_after"] < decomposition["specific_energy_kWh_t"]
        assert isinstance(row["gate_pass_after"], bool)
        assert row["FE_after"] == pytest.approx(decomposition["FE"], abs=1e-12)
        assert row["energy_after"] == pytest.approx(
            959.9 * row["V_after"] / row["FE_after"], rel=1e-4
        )


def test_gate_verdict_is_honest_for_the_contact_single_lever(sensitivities):
    contact = next(row for row in sensitivities if row["lever"] == "contact resistance")
    # The stated contact improvement is the largest single saving, but does
    # not round a >4,000 kWh/t prediction into a gate pass.
    assert contact["energy_after"] > ENERGY_GATE_KWH_T
    assert contact["gate_pass_after"] is False

    passed = [row for row in sensitivities if row["gate_pass_after"]]
    if not passed:
        # This is the expected result at the stated single-lever improvements;
        # any future change must report a real flip rather than fabricate one.
        assert all(row["energy_after"] > ENERGY_GATE_KWH_T for row in sensitivities)
    else:
        assert all(row["energy_after"] <= ENERGY_GATE_KWH_T for row in passed)


def test_contact_resistance_ranks_first(sensitivities, cell):
    ranked = rank_levers(cell, J_REFERENCE)
    assert ranked[0] == "contact resistance"
    rows_by_name = {row["lever"]: row for row in sensitivities}
    assert rows_by_name[ranked[0]]["delta_V"] == max(
        row["delta_V"] for row in sensitivities
    )


def test_buy_next_measurement_is_specific_and_models_a_lever(cell, sensitivities):
    recommendation = buy_next_measurement(cell, J_REFERENCE)
    modeled_levers = {row["lever"] for row in sensitivities}
    assert isinstance(recommendation["recommendation"], str)
    assert recommendation["recommendation"]
    assert recommendation["lever"] in modeled_levers
    assert recommendation["rank"] == 1
    assert recommendation["predicted_delta_V"] > 0.0
    assert "contact" in recommendation["recommendation"].lower()
    assert "contact" in recommendation["reason"].lower()
