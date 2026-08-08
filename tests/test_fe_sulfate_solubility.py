"""
Unit tests for FeSO4 temperature-dependent solubility and retrograde scaling model.
"""

from models.fe_sulfate_solubility import (
    stable_solid_phase,
    feso4_binary_solubility_mol_L,
    feso4_solubility_with_common_ion,
    assess_heat_exchanger_scaling,
)


def test_stable_solid_phase_transition():
    """Verify phase change from melanterite to szomolnokite across 56.7 °C."""
    assert stable_solid_phase(25.0).name == "melanterite"
    assert stable_solid_phase(50.0).name == "melanterite"
    assert stable_solid_phase(60.0).name == "szomolnokite"
    assert stable_solid_phase(85.0).name == "szomolnokite"


def test_prograde_and_retrograde_solubility_branches():
    """Verify prograde solubility below 56.7 °C and retrograde solubility above 56.7 °C."""
    c_20 = feso4_binary_solubility_mol_L(20.0)
    c_50 = feso4_binary_solubility_mol_L(50.0)
    c_56 = feso4_binary_solubility_mol_L(56.7)
    c_75 = feso4_binary_solubility_mol_L(75.0)
    c_90 = feso4_binary_solubility_mol_L(90.0)

    # Prograde branch: solubility increases with temperature
    assert c_50 > c_20
    assert c_56 > c_50

    # Retrograde branch: solubility decreases with temperature
    assert c_75 < c_56
    assert c_90 < c_75

    # Known literature bounds (Linke & Seidell)
    assert 1.3 <= c_20 <= 1.7
    assert 1.9 <= c_56 <= 2.3
    assert 1.0 <= c_90 <= 1.5


def test_common_ion_sulfate_depression():
    """Verify that background sulfate depresses Fe2+ saturation limit."""
    c_fe_pure = feso4_solubility_with_common_ion(60.0, background_sulfate_mol_L=0.0)
    c_fe_with_na2so4 = feso4_solubility_with_common_ion(60.0, background_sulfate_mol_L=0.5)
    c_fe_high_na2so4 = feso4_solubility_with_common_ion(60.0, background_sulfate_mol_L=1.0)

    assert c_fe_with_na2so4 < c_fe_pure
    assert c_fe_high_na2so4 < c_fe_with_na2so4


def test_heat_exchanger_scaling_assessment():
    """Verify scaling risk assessment on hot surfaces in retrograde regime."""
    # Case 1: Moderate bulk and wall temp -> safe
    res_safe = assess_heat_exchanger_scaling(
        bulk_temp_C=50.0,
        wall_temp_C=55.0,
        bulk_fe2_mol_L=1.5,
        background_sulfate_mol_L=0.2,
    )
    assert not res_safe.is_scaling_risk
    assert res_safe.supersaturation_ratio_wall < 1.0

    # Case 2: High bulk Fe2+ and very hot wall in retrograde regime -> severe scaling
    res_fouling = assess_heat_exchanger_scaling(
        bulk_temp_C=65.0,
        wall_temp_C=95.0,
        bulk_fe2_mol_L=1.8,
        background_sulfate_mol_L=0.5,
    )
    assert res_fouling.is_scaling_risk
    assert res_fouling.supersaturation_ratio_wall > 1.0
    assert res_fouling.stable_phase_wall == "szomolnokite"
    assert res_fouling.max_safe_wall_temp_C < 95.0
