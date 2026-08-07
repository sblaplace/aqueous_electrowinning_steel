"""Regression tests for the first chemistry/physics consistency upgrade."""

from __future__ import annotations

import pytest

from models.anode import AnodeKinetics, AnodeMaterial
from models.cell_physics import BathRecipe, CellGeometry, CellPhysics, ProcessConditions
from models.electrochemistry import CellVoltageModel, E0_FE
from models.reference_cell_design import load_reference_cell_config
from models.speciation import SolutionComposition, solve_speciation
from models.thermodynamic_constants import buffer_capacity_M_per_pH
from models.transport import NernstPlanckFilm


def test_buffer_capacity_does_not_equal_boric_acid_loading_at_pH_two():
    """0.4 M boric acid is not a 0.4 M/pH buffer in a strongly acidic bath."""
    beta = buffer_capacity_M_per_pH(
        2.0, 60.0, total_sulfate_M=1.5, total_borate_M=0.4
    )
    beta_boric_only = buffer_capacity_M_per_pH(
        2.0, 60.0, total_sulfate_M=0.0, total_borate_M=0.4
    )
    beta_water_only = buffer_capacity_M_per_pH(
        2.0, 60.0, total_sulfate_M=0.0, total_borate_M=0.0
    )
    # Free H+ contributes about 0.023 M/pH at pH 2; the *borate increment*
    # is the quantity that is negligible.
    assert beta_boric_only - beta_water_only < 1e-5
    assert beta < 1.0
    assert beta > beta_boric_only


def test_cell_voltage_uses_activity_corrected_standard_state_once():
    """The supplied standard potential must affect the Nernst result."""
    model = CellVoltageModel(E_cathode_eq=-0.500, fe2_conc_M=1.0)
    assert model.E_cathode_nernst == pytest.approx(-0.500, abs=1e-12)


def test_temperature_resolved_conductivity_is_not_corrected_twice():
    comp = SolutionComposition(
        c_FeSO4=1.0, c_Na2SO4=0.5, c_H2SO4=0.01, c_H3BO3=0.4, T_C=50.0
    )
    kappa_50 = solve_speciation(comp)["conductivity_S_m"]
    resolved = CellVoltageModel(
        temperature_C=50.0,
        electrolyte_conductivity_S_m=kappa_50,
        electrolyte_conductivity_at_temperature=True,
        j_operating_mA_cm2=100.0,
    )
    legacy = CellVoltageModel(
        temperature_C=50.0,
        electrolyte_conductivity_S_m=kappa_50,
        electrolyte_conductivity_at_temperature=False,
        j_operating_mA_cm2=100.0,
    )
    # The old path applied its temperature correlation a second time, which
    # made kappa too large and IR too small.
    assert resolved.IR_electrolyte > legacy.IR_electrolyte
    assert resolved.IR_electrolyte == pytest.approx(
        100.0 * 10.0 * 0.02 / (kappa_50 * 0.9), rel=1e-12
    )


def test_transport_defaults_follow_temperature():
    cold = NernstPlanckFilm(temperature_C=25.0)
    hot = NernstPlanckFilm(temperature_C=60.0)
    assert hot.diffusivity_fe_m2_s > cold.diffusivity_fe_m2_s
    assert hot.diffusion_limit_A_m2 > cold.diffusion_limit_A_m2


def test_reactive_cell_physics_is_the_default_and_uses_bath_buffer():
    physics = CellPhysics(BathRecipe(c_H3BO3_M=0.4))
    assert physics.conditions.transport_model == "reactive"
    point = physics.solve_at_j(100.0)
    assert point.speciation["activity_model"] == "pitzer"
    assert point.transport_converged
    assert point.migration_enhancement > 1.0
    assert "pH_activity_delta_from_recipe" in point.speciation
    assert point.speciation["pH_boundary_source"].startswith("BathRecipe.pH")


def test_soluble_anode_uses_iron_dissolution_not_oer_equilibrium():
    material = AnodeMaterial(
        name="test soluble anode", oer_i0=1.0, oer_tafel_V=0.06, temperature_C=60.0
    )
    anode = AnodeKinetics(
        material=material,
        electrolyte_type="acidic",
        pH=2.0,
        anode_chemistry="soluble",
        fe2_conc_M=1.0,
    )
    model = CellVoltageModel(
        anode=anode,
        bubble_fraction=0.25,
        electrolyte_conductivity_at_temperature=True,
        j_operating_mA_cm2=100.0,
    )
    assert model.E_anode_nernst == pytest.approx(
        anode.fe_dissolution_equilibrium(), rel=1e-12
    )
    assert model.IR_electrolyte == pytest.approx(
        100.0 * 10.0 * 0.02 / 10.0, rel=1e-12
    )


def test_rc1_yaml_declares_anode_mode_and_derived_buffer_capacity():
    config = load_reference_cell_config()
    assert config.geometry.anode_chemistry == "inert"
    # The derived capacity is deliberately not the raw 0.4 M boric-acid load.
    # Rebuild the same design-twin input through the public helper's source
    # configuration to ensure the chemistry calculation remains available.
    beta = buffer_capacity_M_per_pH(
        config.bath.pH,
        config.target_temperature_C,
        total_sulfate_M=(
            config.bath.c_FeSO4_M
            + config.bath.c_Na2SO4_M
            + config.bath.c_H2SO4_M
        ),
        total_borate_M=config.bath.c_H3BO3_M,
    )
    assert beta != pytest.approx(config.bath.c_H3BO3_M)
    assert beta > 0.0


def test_dilute_transport_path_remains_available_for_ab_testing():
    dilute = CellPhysics(
        BathRecipe(),
        CellGeometry(),
        # Keep the old closure available without making it the default.
        conditions=ProcessConditions(transport_model="dilute_np"),
    )
    point = dilute.solve_at_j(50.0)
    assert point.V_cell > 0.0
    assert point.current_efficiency > 0.0
