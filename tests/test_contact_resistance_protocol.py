"""Tests for the contact resistance measurement protocol (Deliverable A)."""

from __future__ import annotations

from models.contact_resistance_protocol import (
    protocol_overview,
    expected_contact_resistance_range,
    impact_if_measured,
    instrument_requirements,
    reference_cell,
)


def test_protocol_overview_structure():
    overview = protocol_overview()
    assert isinstance(overview, dict)
    assert len(overview) > 0
    assert "method" in overview
    assert "target_interfaces" in overview
    assert overview["status"] == "unvalidated (L0)"


def test_instrument_requirements_structure():
    reqs = instrument_requirements()
    assert isinstance(reqs, list)
    assert len(reqs) >= 4
    for req in reqs:
        assert isinstance(req, str)
        assert len(req) > 0


def test_expected_contact_resistance_range_consistency():
    rng = expected_contact_resistance_range()
    assert isinstance(rng, dict)
    assert rng["status"] == "expected, not measured"
    assert rng["flag"] == "unvalidated (L0)"

    min_val = rng["min"]["value"]
    typ_val = rng["typical"]["value"]
    max_val = rng["max"]["value"]

    assert 0.0 < min_val <= typ_val <= max_val
    assert rng["unit"] == "Ω·m²"


def test_impact_if_measured():
    cell = reference_cell()
    impact = impact_if_measured(cell, j_mA_cm2=300.0)
    assert isinstance(impact, dict)
    assert "scenarios" in impact
    scenarios = impact["scenarios"]
    assert "min" in scenarios
    assert "typical" in scenarios
    assert "max" in scenarios

    typ_res = scenarios["typical"]
    assert typ_res["contact_resistance_ohm_m2"] == 5.0e-4
    assert typ_res["V_cell"] > 4.0
    assert typ_res["specific_energy_kWh_t"] > 4000.0
    assert typ_res["gate_pass"] is False
    assert impact["flag"] == "unvalidated (L0)"

    min_res = scenarios["min"]
    max_res = scenarios["max"]
    assert min_res["specific_energy_kWh_t"] < typ_res["specific_energy_kWh_t"]
    assert max_res["specific_energy_kWh_t"] > typ_res["specific_energy_kWh_t"]
