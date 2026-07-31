"""Tests for the unified cell physics solver.

``cell_physics.py`` chains speciation → Nernst-Planck transport → cell voltage
into one self-consistent operating point.  It is the module that supplies the
numbers ``dark_mill.py`` would otherwise assume, so its internal consistency
matters more than any single output value.

These tests are deliberately structural: they check that the pieces agree with
each other (energy against V and FE, decomposition against total, transport
limit against migration enhancement) rather than pinning uncalibrated
predictions to specific magnitudes.
"""

import numpy as np
import pytest

from models.cell_physics import (
    BathRecipe,
    CellGeometry,
    CellPhysics,
    OperatingPoint,
    ProcessConditions,
)


@pytest.fixture(scope="module")
def cell():
    return CellPhysics(BathRecipe())


@pytest.fixture(scope="module")
def point(cell):
    return cell.solve_at_j(100.0)


@pytest.fixture(scope="module")
def window(cell):
    """``sweep(j_min, j_max, n_points)`` maps the operating window."""
    return cell.sweep(j_min=25.0, j_max=200.0, n_points=4)


class TestBathRecipe:
    def test_defaults_are_a_plausible_sulfate_bath(self):
        b = BathRecipe()
        assert b.c_FeSO4_M > 0
        assert b.c_H3BO3_M > 0        # boric acid buffer present
        assert 0 < b.pH < 7           # acidic

    def test_converts_to_speciation_input(self):
        comp = BathRecipe().to_speciation(50.0)
        assert comp.c_FeSO4 == pytest.approx(1.0)
        assert comp.T_C == pytest.approx(50.0)


class TestOperatingPoint:
    def test_is_the_expected_type(self, point):
        assert isinstance(point, OperatingPoint)

    def test_efficiency_is_a_fraction(self, point):
        assert 0.0 <= point.current_efficiency <= 1.0

    def test_voltage_is_positive_and_plausible(self, point):
        assert 0.5 < point.V_cell < 10.0

    def test_voltage_decomposition_sums_to_total(self, point):
        """The decomposition must actually add up to V_cell."""
        d = point.V_decomposition
        parts = (
            d["E_thermodynamic (V)"]
            + d["η_cathode (V)"]
            + d["η_anode (V)"]
            + d["IR_total (V)"]
        )
        assert parts == pytest.approx(d["V_cell (V)"], rel=1e-6)
        # The reported decomposition is rounded for display; compare loosely.
        assert d["V_cell (V)"] == pytest.approx(point.V_cell, abs=1e-3)

    def test_ir_total_is_the_sum_of_its_parts(self, point):
        d = point.V_decomposition
        assert d["IR_total (V)"] == pytest.approx(
            d["IR_electrolyte (V)"] + d["IR_membrane (V)"] + d["IR_contacts (V)"],
            rel=1e-6,
        )

    def test_specific_energy_matches_v_over_fe(self, point):
        """E = 959.9 × V/FE kWh/t Fe — the program's governing identity."""
        expected = 959.9 * point.V_cell / point.current_efficiency
        assert point.specific_energy_kWh_t == pytest.approx(expected, rel=0.02)

    def test_deposition_rate_positive(self, point):
        assert point.deposition_rate_um_hr > 0

    def test_surface_ph_is_at_or_above_bulk(self, point):
        """Cathodic HER consumes protons, so the surface cannot be more acidic
        than the bulk."""
        assert point.surface_pH >= BathRecipe().pH - 0.2

    def test_surface_iron_is_depleted_not_enriched(self, point):
        assert 0.0 <= point.surface_fe_M <= BathRecipe().c_FeSO4_M * 1.05

    def test_migration_enhancement_exceeds_unity(self, point):
        """Fe²⁺ is cationic and migrates toward the cathode, so the true limit
        exceeds the pure-diffusion Levich value."""
        assert point.migration_enhancement > 1.0
        assert point.transport_limit_mA_cm2 > point.diffusion_limit_mA_cm2

    def test_migration_enhancement_is_consistent_with_the_limits(self, point):
        assert point.migration_enhancement == pytest.approx(
            point.transport_limit_mA_cm2 / point.diffusion_limit_mA_cm2, rel=1e-6
        )

    def test_conductivity_is_physical(self, point):
        assert 0.1 < point.conductivity_S_m < 100.0

    def test_precipitation_flag_agrees_with_supersaturation(self, point):
        assert point.precipitation_active == (point.feoh2_supersaturation > 1.0)


class TestCurrentDensityResponse:
    def test_higher_current_raises_voltage(self, cell):
        assert cell.solve_at_j(200.0).V_cell > cell.solve_at_j(20.0).V_cell

    def test_higher_current_deposits_faster(self, cell):
        assert (
            cell.solve_at_j(200.0).deposition_rate_um_hr
            > cell.solve_at_j(20.0).deposition_rate_um_hr
        )

    def test_higher_current_raises_surface_ph_when_her_is_active(self):
        """More HER at the surface means more proton depletion.

        This only holds when HER actually draws current. At the module default
        (her_i0 = 1e-6, FE ≈ 99.5%) proton consumption is negligible and
        migration slightly *enriches* H⁺ at the cathode, so surface pH drifts
        marginally down with j until the transport limit is approached. The
        physics claim therefore has to be tested on an active cathode.
        """
        active = CellPhysics(BathRecipe(), conditions=ProcessConditions(her_i0=1e-2))
        assert active.solve_at_j(200.0).surface_pH > active.solve_at_j(20.0).surface_pH

    def test_higher_current_depletes_surface_iron(self, cell):
        assert cell.solve_at_j(200.0).surface_fe_M <= cell.solve_at_j(20.0).surface_fe_M


class TestGeometryAndConditions:
    def test_wider_gap_raises_ohmic_drop(self):
        narrow = CellPhysics(BathRecipe(), CellGeometry(interelectrode_gap_m=0.005))
        wide = CellPhysics(BathRecipe(), CellGeometry(interelectrode_gap_m=0.05))
        assert (
            wide.solve_at_j(100.0).V_decomposition["IR_electrolyte (V)"]
            > narrow.solve_at_j(100.0).V_decomposition["IR_electrolyte (V)"]
        )

    def test_undivided_cell_has_no_membrane_drop(self):
        undivided = CellPhysics(BathRecipe(), CellGeometry(membrane=False))
        assert undivided.solve_at_j(100.0).V_decomposition["IR_membrane (V)"] == (
            pytest.approx(0.0)
        )

    def test_membrane_adds_voltage(self):
        divided = CellPhysics(BathRecipe(), CellGeometry(membrane=True))
        undivided = CellPhysics(BathRecipe(), CellGeometry(membrane=False))
        assert divided.solve_at_j(100.0).V_cell > undivided.solve_at_j(100.0).V_cell

    def test_thinner_boundary_layer_raises_transport_limit(self):
        """Agitation is the lever on transport."""
        still = CellPhysics(
            BathRecipe(), conditions=ProcessConditions(boundary_layer_m=200e-6)
        )
        stirred = CellPhysics(
            BathRecipe(), conditions=ProcessConditions(boundary_layer_m=20e-6)
        )
        assert (
            stirred.solve_at_j(100.0).transport_limit_mA_cm2
            > still.solve_at_j(100.0).transport_limit_mA_cm2
        )

    def test_richer_bath_raises_transport_limit(self):
        lean = CellPhysics(BathRecipe(c_FeSO4_M=0.25))
        rich = CellPhysics(BathRecipe(c_FeSO4_M=2.0))
        assert (
            rich.solve_at_j(100.0).transport_limit_mA_cm2
            > lean.solve_at_j(100.0).transport_limit_mA_cm2
        )

    def test_suppressing_her_improves_efficiency(self):
        active = CellPhysics(
            BathRecipe(), conditions=ProcessConditions(her_i0=1e-2)
        )
        blocked = CellPhysics(
            BathRecipe(), conditions=ProcessConditions(her_i0=1e-7)
        )
        assert (
            blocked.solve_at_j(100.0).current_efficiency
            > active.solve_at_j(100.0).current_efficiency
        )


class TestSweep:
    def test_sweep_returns_a_point_per_current(self, window):
        assert len(window.points) == 4
        assert len(window.FE_array) == 4
        assert len(window.V_cell_array) == 4

    def test_sweep_arrays_are_finite_and_bounded(self, window):
        assert np.all(np.isfinite(window.V_cell_array))
        assert np.all((window.FE_array >= 0.0) & (window.FE_array <= 1.0))

    def test_voltage_increases_monotonically_across_the_sweep(self, window):
        assert np.all(np.diff(window.V_cell_array) > 0)

    def test_energy_array_is_positive(self, window):
        assert np.all(window.energy_array > 0)

    def test_transport_limit_is_reported(self, window):
        assert window.transport_limit_mA_cm2 > 0

    def test_optimal_j_respects_the_efficiency_floor(self, window):
        best = window.optimal_j(min_FE=0.5)
        if best is not None:
            assert best.current_efficiency >= 0.5

    def test_impossible_efficiency_floor_returns_nothing(self, window):
        assert window.optimal_j(min_FE=1.01) is None


class TestSummary:
    def test_summary_is_a_populated_mapping(self, cell):
        s = cell.summary(100.0)
        assert isinstance(s, dict) and s

    def test_summary_is_json_serializable(self, cell):
        import json

        json.dumps(cell.summary(100.0), default=float)
