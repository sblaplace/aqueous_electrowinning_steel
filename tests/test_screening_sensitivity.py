"""Numerical locks for the Level-0 screening uncertainty budget.

The assertions compare the synthetic budget to the explicit #32 screening
thresholds.  They are not measurements and do not turn the sensitivity screen
into gate evidence.
"""
from __future__ import annotations

import math

import pytest

from models.screening_sensitivity import (
    define_ranges,
    perturb_and_solve,
    ranked_calibration_priority,
    sensitivity_profile,
)
from models.theory_confidence import reference_cell, solve_reference


@pytest.fixture(scope="module")
def rc():
    return reference_cell()


@pytest.fixture(scope="module")
def central(rc):
    return solve_reference(rc)


@pytest.fixture(scope="module")
def profile(rc):
    return sensitivity_profile(rc)


def test_ranges_are_explicit_factor_screens(rc):
    ranges = define_ranges()
    expected = {
        "fe_i0",
        "her_i0",
        "fe_tafel_V",
        "her_tafel_V",
        "boundary_layer_m",
        "c_FeSO4_M",
        "c_Na2SO4_M",
        "membrane_area_resistance_ohm_m2",
        "temperature_C",
    }
    assert expected <= ranges.keys()
    for name, definition in ranges.items():
        low, high = definition["range"]
        assert definition["value"] > 0.0
        assert 0.0 < low < 1.0 < high
        assert high / low > 1.0
        assert definition["source"].strip()
        assert definition["maps_to"].strip()
        # The range really maps to the reference-cell central value.
        assert math.isclose(
            definition["value"],
            getattr(rc.conditions, name)
            if hasattr(rc.conditions, name)
            else getattr(rc.bath, name)
            if hasattr(rc.bath, name)
            else getattr(rc.geometry, name),
        )


def test_central_reference_passes_all_screening_targets(central, rc):
    assert central["all_pass"] is True
    assert central["current_efficiency"] >= rc.targets.fe_min
    assert rc.targets.v_cell_min <= central["V_cell"] <= rc.targets.v_cell_max
    assert central["specific_energy_kWh_t"] <= rc.targets.specific_energy_max_kWh_t
    transport_margin = (
        central["transport_limit_mA_cm2"] / central["current_density_mA_cm2"]
    )
    assert transport_margin >= rc.targets.transport_margin_min
    assert (
        rc.targets.deposit_rate_min_um_hr
        <= central["deposition_rate_um_hr"]
        <= rc.targets.deposit_rate_max_um_hr
    )


@pytest.mark.slow
class TestScreeningProfile:
    def test_budget_has_a_non_vacuous_decision_threat(self, profile, central, rc):
        assert any(
            entry["delta_fe"] > central["current_efficiency"] - rc.targets.fe_min
            or entry["delta_v_cell"]
            > min(
                central["V_cell"] - rc.targets.v_cell_min,
                rc.targets.v_cell_max - central["V_cell"],
            )
            for entry in profile.values()
        )
        dominant = profile["her_tafel_V"]
        assert dominant["delta_fe"] > central["current_efficiency"] - rc.targets.fe_min
        assert dominant["delta_v_cell"] > rc.targets.v_cell_max - central["V_cell"]
        assert dominant["flips_pass_at_reference"] is True
        assert dominant["min_margin_across_window"] >= 0.0

    def test_profile_entries_report_numeric_influence_metrics(self, profile):
        required = {
            "delta_fe",
            "delta_v_cell",
            "delta_specific_energy",
            "flips_pass_at_reference",
            "min_margin_across_window",
        }
        for entry in profile.values():
            assert required <= entry.keys()
            for key in (
                "delta_fe",
                "delta_v_cell",
                "delta_specific_energy",
                "min_margin_across_window",
            ):
                assert math.isfinite(entry[key])
            assert entry["delta_fe"] >= 0.0
            assert entry["delta_v_cell"] >= 0.0
            assert entry["delta_specific_energy"] >= 0.0

    def test_priority_is_deterministic_and_influence_ordered(self, profile):
        priority = ranked_calibration_priority(profile)
        assert priority
        assert priority == ranked_calibration_priority(dict(reversed(list(profile.items()))))
        assert priority[0] == "her_tafel_V"
        scores = [profile[name]["influence_score"] for name in priority]
        assert scores == sorted(scores, reverse=True)
        assert profile[priority[0]]["flips_pass_at_reference"] is True

    def test_top_range_brackets_central_fe_and_voltage_and_flips(self, rc, central, profile):
        top = ranked_calibration_priority(profile)[0]
        low_factor, high_factor = define_ranges()[top]["range"]
        low = perturb_and_solve(rc, top, low_factor)
        high = perturb_and_solve(rc, top, high_factor)

        fe_values = (low["current_efficiency"], high["current_efficiency"])
        voltage_values = (low["V_cell"], high["V_cell"])
        assert min(fe_values) <= central["current_efficiency"] <= max(fe_values)
        assert min(voltage_values) <= central["V_cell"] <= max(voltage_values)
        assert low["all_pass"] is False or high["all_pass"] is False
        assert profile[top]["flips_pass_at_reference"] is True
