"""Tests for chemical batch recipes and factorial DOE matrix generation."""

import pytest
import pandas as pd
from models.experimental_matrix import ChemicalRecipe, calculate_batch_recipe, predict_plating_run, generate_factorial_doe


def test_calculate_batch_recipe():
    """Recipe calculation for 1.0 L solution."""
    rec = ChemicalRecipe(c_FeSO4_M=1.0, c_Na2SO4_M=0.5, volume_L=1.0)
    b = calculate_batch_recipe(rec)
    
    assert abs(b["FeSO4_7H2O_g"] - 278.01) < 1.0
    assert abs(b["Na2SO4_g"] - 71.02) < 1.0
    assert b["ascorbic_acid_g"] == 2.0


def test_predict_plating_run():
    """Predict plating yield and thickness for 10 cm2 cathode at 200 mA/cm2 for 2 hours."""
    p = predict_plating_run(j_mA_cm2=200.0, area_cm2=10.0, t_run_hr=2.0, pH_bulk=2.5, T_C=50.0)
    
    assert p["current_A"] == 2.0
    assert p["charge_Coulombs"] == 2.0 * 2.0 * 3600.0
    assert p["m_fe_expected_g"] > 1.0
    assert p["deposit_thickness_um"] > 10.0


def test_generate_factorial_doe():
    """Full-factorial DOE matrix generation."""
    df = generate_factorial_doe(j_levels=[100.0, 200.0], pH_levels=[2.5], T_levels=[50.0])
    
    assert len(df) == 2
    assert "predicted_FE_pct" in df.columns
    assert "m_fe_expected_g" in df.columns
