"""Assertions for the Level-0 operating-window confidence screen.

These are numerical screening assertions against the target bundle from
``theory_confidence``.  They are not measurements and are not gate evidence.
"""
from __future__ import annotations

import math

import pytest

from models.operating_window_confidence import (
    margins,
    reference_is_interior,
    sweep_window,
    usable_fraction,
    window_boundary,
)
from models.theory_confidence import reference_cell, solve_reference


@pytest.fixture(scope="module")
def rc():
    return reference_cell()


@pytest.fixture(scope="module")
def result(rc):
    reference_j = solve_reference(rc)["current_density_mA_cm2"]
    return sweep_window(
        rc,
        t_grid=(40.0, 50.0, 60.0),
        fe_grid=(0.75, 1.0, 1.25),
        j_grid=(30.0, reference_j, 140.0, 180.0, 240.0),
    )


@pytest.mark.slow
class TestOperatingWindowConfidence:
    def test_usable_fraction_is_a_nontrivial_screening_window(self, result):
        fraction = usable_fraction(result)
        assert isinstance(fraction, float)
        assert fraction > 1.0 / 3.0

    def test_reference_is_strictly_interior_to_every_target_bound(self, result, rc):
        interior, point_margins = reference_is_interior(result, rc)
        assert interior is True
        assert set(point_margins) == {
            "fe", "v_cell", "specific_energy", "transport_limit", "deposition_rate", "thermal"
        }
        for margin in point_margins.values():
            assert isinstance(margin, float)
            assert margin > 0.0

    def test_specific_energy_ceiling_holds_everywhere_usable(self, result, rc):
        energies = [
            row["specific_energy_kWh_t"]
            for row in result["results"]
            if row["all_pass"]
        ]
        assert energies
        assert max(energies) <= rc.targets.specific_energy_max_kWh_t

    def test_cooled_temperature_is_under_limit_at_all_four_bath_corners(self, result, rc):
        corners = [row for row in result["results"] if row["corner_thermal_checked"]]
        assert {(row["T_C"], row["fe_M"]) for row in corners} == {
            (40.0, 0.75), (40.0, 1.25), (60.0, 0.75), (60.0, 1.25)
        }
        assert len(corners) == 4
        assert {row["j_mA_cm2"] for row in corners} == {result["thermal_j_mA_cm2"]}
        for row in corners:
            assert row["thermal"]["steady_state_T_C"] <= rc.targets.thermal_limit_C

    def test_margins_are_numeric_headroom_against_targets(self, result):
        summary = margins(result)
        for target, values in summary.items():
            assert target in result["results"][0]["verdicts"] or target == "thermal"
            assert isinstance(values["median"], float)
            assert isinstance(values["min"], float)
            assert not math.isnan(values["median"])
            assert not math.isnan(values["min"])
            assert values["min"] >= 0.0

    def test_off_design_point_fails_a_numerical_target(self, rc):
        off_design = sweep_window(
            rc, t_grid=(50.0,), fe_grid=(0.25,), j_grid=(300.0,)
        )["results"][0]
        assert off_design["all_pass"] is False
        assert any(not verdict["pass"] for verdict in off_design["verdicts"].values())

    def test_boundary_reports_a_sampled_first_trip(self, result):
        boundary = window_boundary(result)
        assert boundary["specific_energy"] is not None
        axis, value = boundary["specific_energy"]
        assert axis in {"T_C", "fe_M", "j_mA_cm2"}
        assert isinstance(value, float)
