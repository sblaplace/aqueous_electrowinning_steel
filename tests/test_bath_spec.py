"""Tests for the unified BathSpec chemistry state.

These are structural checks: the goal is to ensure sulfate/chloride/ammonium/O₂
inventories are represented once and that CellPhysics can consume that richer
state without losing current-balance accounting.
"""

import pytest

from models.bath_spec import BathSpec
from models.cell_physics import BathRecipe, CellPhysics, ProcessConditions


class TestBathSpec:
    def test_reference_sulfate_maps_to_legacy_recipe_kwargs(self):
        spec = BathSpec.reference_sulfate(fe2_M=1.25, na2so4_M=0.2, pH=2.1)
        kwargs = spec.to_legacy_bath_recipe_kwargs()

        assert kwargs["c_FeSO4_M"] == pytest.approx(1.25)
        assert kwargs["c_Na2SO4_M"] == pytest.approx(0.2)
        assert kwargs["pH"] == pytest.approx(2.1)
        assert spec.surface_state_bath_type() == "sulfate"

    def test_legacy_recipe_round_trips_into_bath_spec(self):
        recipe = BathRecipe(c_FeSO4_M=0.8, c_Na2SO4_M=0.1, pH=2.4)
        spec = BathSpec.from_legacy_recipe(recipe, temperature_C=55.0)

        assert spec.family == "sulfate"
        assert spec.fe2_total_M == pytest.approx(0.8)
        assert spec.na2so4_M == pytest.approx(0.1)
        assert spec.temperature_C == pytest.approx(55.0)

    def test_chloride_spec_reports_chloride_diagnostics(self):
        spec = BathSpec.aware_chloride(fecl2_M=1.0, licl_M=10.0, temperature_C=60.0)
        diag = spec.solve_bulk_speciation()

        assert diag["bath_family"] == "chloride"
        assert diag["c_chloride_total_M"] > 10.0
        assert diag["conductivity_S_m"] > 0.0
        assert diag["activity_model"] == "pitzer_fecl2"
        assert "chloride" in diag["transport_solver_basis"]
        assert spec.surface_state_bath_type() == "aware"
        assert spec.feature_inventory()["chloride_present"] is True
        assert spec.feature_inventory()["sulfate_present"] is False

    def test_ammonium_and_oxygen_are_in_one_bulk_diagnostic(self):
        spec = BathSpec.reference_sulfate(
            ammonium_total_M=0.5,
            dissolved_o2_fraction_sat=0.25,
            temperature_C=50.0,
        )
        diag = spec.solve_bulk_speciation()

        assert diag["c_NH4_total_M"] == pytest.approx(0.5)
        assert diag["dissolved_o2_fraction_sat"] == pytest.approx(0.25)
        assert "ammonium" in diag
        assert "dissolved_oxygen" in diag
        assert diag["dissolved_oxygen"]["bulk_M"] > 0.0
        assert diag["ammonium"]["free_fe2_M"] > 0.0


class TestCellPhysicsRichChemistry:
    def test_cell_physics_accepts_bath_spec(self):
        spec = BathSpec.reference_sulfate()
        cell = CellPhysics(spec)

        assert cell.bath_spec.name == "reference_sulfate"
        assert cell.bath.c_FeSO4_M == pytest.approx(spec.fe2_total_M)

    def test_rich_path_exposes_current_breakdown_and_parasitic_loss(self):
        spec = BathSpec.reference_sulfate(
            dissolved_o2_fraction_sat=1.0,
            fe3_M=1e-6,
            metadata={"cathode_area_m2": 1.0e-3, "catholyte_volume_L": 0.5},
        )
        cell = CellPhysics(
            spec,
            conditions=ProcessConditions.rich(boundary_layer_m=100e-6),
        )
        point = cell.solve_at_j(100.0)

        b = point.current_breakdown_A_m2
        assigned = (
            b["Fe_deposition_A_m2"]
            + b["HER_A_m2"]
            + b["ORR_A_m2"]
            + b["Fe3_shuttle_A_m2"]
            + b["unassigned_A_m2"]
        )
        assert assigned == pytest.approx(b["applied_A_m2"], rel=1e-3, abs=1e-3)
        assert point.parasitic_current_A_m2 > 0.0
        assert point.orr_current_A_m2 > 0.0
        assert point.fe3_shuttle_current_A_m2 > 0.0
        assert point.current_efficiency <= point.transport_current_efficiency
        assert point.chemistry_diagnostics["enabled_features"]["surface_state_her"] is True
        assert point.chemistry_diagnostics["reactive_film"]["surface_state_bath_type"] == "sulfate"

    def test_one_feature_can_be_enabled_without_rich_mode(self):
        spec = BathSpec.reference_sulfate(dissolved_o2_fraction_sat=1.0)
        cell = CellPhysics(spec, conditions=ProcessConditions(dissolved_oxygen=True))
        point = cell.solve_at_j(100.0)

        assert point.orr_current_A_m2 > 0.0
        assert point.chemistry_diagnostics["enabled_features"]["dissolved_oxygen"] is True
        assert point.chemistry_diagnostics["enabled_features"]["surface_state_her"] is False
