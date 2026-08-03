"""Tests for the joint operating-point optimizer (Deliverable B)."""

from __future__ import annotations

from models.operating_point_optimizer import (
    solve_window,
    energy_gate_reachable,
    best_operating_point,
    sweep_table,
    reference_cell,
)
from models.contact_resistance_protocol import expected_contact_resistance_range


def test_solve_window_transport_limit_enforced():
    window = solve_window()
    assert isinstance(window, list)
    assert len(window) > 0

    valid_count = 0
    invalid_count = 0

    for row in window:
        assert "transport_margin" in row
        assert "j_mA_cm2" in row
        assert "valid" in row
        assert row["flag"] == "unvalidated (L0)"

        if row["valid"]:
            valid_count += 1
            assert row["transport_margin"] > 1.0
            assert row["transport_limit_mA_cm2"] > row["j_mA_cm2"]
        else:
            invalid_count += 1
            assert row["transport_margin"] <= 1.0 or row["LCOFe_usd_per_t"] is None


def test_energy_gate_reachable():
    reach = energy_gate_reachable()
    assert isinstance(reach, dict)
    assert "reachable" in reach
    assert isinstance(reach["reachable"], bool)
    assert "min_energy_kWh_t" in reach
    assert isinstance(reach["min_energy_kWh_t"], float)
    assert reach["flag"] == "unvalidated (L0)"

    if reach["reachable"]:
        assert reach["min_energy_kWh_t"] <= 4000.0
    else:
        assert reach["min_energy_kWh_t"] > 4000.0
        assert reach["best_combination"] is not None
        assert reach["best_combination"]["specific_energy_kWh_t"] == reach["min_energy_kWh_t"]


def test_best_operating_point():
    best = best_operating_point()
    assert isinstance(best, dict)
    assert "energy_gate_pass" in best
    assert isinstance(best["energy_gate_pass"], bool)
    assert "LCOFe_usd_per_t" in best
    assert isinstance(best["LCOFe_usd_per_t"], float)
    assert best["LCOFe_usd_per_t"] > 0.0
    assert best["flag"] == "unvalidated (L0)"
    assert "verdict" in best


def test_protocol_optimizer_wiring():
    cell = reference_cell()
    proto_min = expected_contact_resistance_range()["min"]["value"]

    default_best = best_operating_point(cell, contact_resistance_ohm_m2=5.0e-4)
    wired_best = best_operating_point(cell, contact_resistance_ohm_m2=proto_min)

    assert wired_best["specific_energy_kWh_t"] <= default_best["specific_energy_kWh_t"]
    assert wired_best["LCOFe_usd_per_t"] <= default_best["LCOFe_usd_per_t"]


def test_closure_and_ordering_invariant():
    """Energy strictly decreases (or holds) as any resistive lever is reduced at fixed j."""
    cell = reference_cell()
    window_high_contact = solve_window(cell, contact_resistance_ohm_m2=5.0e-4)
    window_low_contact = solve_window(cell, contact_resistance_ohm_m2=1.0e-4)

    row_high = next(r for r in window_high_contact if r["j_mA_cm2"] == 150.0 and r["interelectrode_gap_m"] == 1.5e-3 and r["valid"])
    row_low = next(r for r in window_low_contact if r["j_mA_cm2"] == 150.0 and r["interelectrode_gap_m"] == 1.5e-3 and r["valid"])

    assert row_low["specific_energy_kWh_t"] < row_high["specific_energy_kWh_t"]
    assert row_low["V_cell"] < row_high["V_cell"]


def test_sweep_table():
    table = sweep_table()
    assert isinstance(table, list)
    assert len(table) > 0
    for row in table:
        assert "j_mA_cm2" in row
        assert "cost_optimal_LCOFe" in row
        assert "energy_optimal_kWh_t" in row
        assert row["cost_optimal_LCOFe"] > 0.0
        assert row["energy_optimal_kWh_t"] > 0.0
        assert row["flag"] == "unvalidated (L0)"
