"""Assertions for Level-0, physics-derived economics (not gate evidence)."""
from math import isfinite

import pytest

from models.economics_from_physics import (
    REFERENCE_J_MA_CM2,
    derived_operating_point,
    physics_lcofe,
    reference_cell,
    sweep_economics,
    uncertainty_propagation,
)
from models.electrochemistry import specific_energy_kWh_per_t


@pytest.fixture(scope="module")
def cell():
    return reference_cell()


def test_reference_point_meets_published_screening_targets(cell):
    result = derived_operating_point(cell, REFERENCE_J_MA_CM2)
    # This is deliberately a real model finding, not a tuned green target.
    assert result["verdicts"]["FE"]["pass"]
    assert not result["verdicts"]["specific_energy"]["pass"]
    assert result["specific_energy_kWh_t"] > cell.targets.specific_energy_max_kWh_t
    assert result["verdicts"]["transport_limit"]["pass"]
    assert not result["all_targets_pass"]
    assert result["specific_energy_kWh_t"] == pytest.approx(
        specific_energy_kWh_per_t(result["V_cell"], result["current_efficiency"])
    )
    assert result["transport_limit_mA_cm2"] > REFERENCE_J_MA_CM2


def test_physics_lcofe_is_positive_and_reports_assumption_gap(cell):
    result = physics_lcofe(cell)
    assert isfinite(result["LCOFe_usd_per_t"]) and result["LCOFe_usd_per_t"] > 0
    assert result["annual_capacity_t_yr"] > 0
    assert isfinite(result["LCOFe_gap_usd_per_t"])
    assert result["LCOFe_gap_sign"] in {"higher", "lower", "neutral"}


def test_sweep_covers_range_and_energy_is_ordered(cell):
    rows = sweep_economics(cell)
    assert rows[0]["j_mA_cm2"] == pytest.approx(50.0)
    assert rows[-1]["j_mA_cm2"] == pytest.approx(500.0)
    valid_rows = [row for row in rows if not row["invalid"]]
    invalid_rows = [row for row in rows if row["invalid"]]
    assert invalid_rows  # beyond transport is surfaced rather than priced.
    energies = [row["specific_energy_kWh_t"] for row in valid_rows]
    assert all(right >= left * 0.995 for left, right in zip(energies, energies[1:]))


def test_uncertainty_is_non_degenerate_and_identifies_measurement(cell):
    result = uncertainty_propagation(cell)
    assert result["LCOFe_min_usd_per_t"] < result["base_LCOFe_usd_per_t"] < result["LCOFe_max_usd_per_t"]
    assert result["largest_driver"] in {"FE", "V_cell"}
    assert result["measurement_recommendation"]


def test_transport_exceedance_fails_loudly(cell):
    with pytest.raises(ValueError, match="transport limit"):
        derived_operating_point(cell, 100_000.0)
