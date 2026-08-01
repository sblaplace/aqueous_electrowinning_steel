"""Experimental recipe and Factorial DOE run matrix generator for lab trials.

Translates desired electrochemical test conditions (j, T, pH, bath concentrations)
into explicit lab batch chemical masses (FeSO4*7H2O, Na2SO4, H3BO3, ascorbic acid),
predicted deposit yields/thicknesses, and full-factorial DOE run sheets for
Phase I, Phase II, and Phase III experimental campaigns.

References:
- Montgomery, D. C. (2017). Design and Analysis of Experiments. Wiley.
- ASTM B568 / B567 Standard Test Methods for Electrodeposited Coating Thickness.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd

from .operating_window import evaluate_operating_point

# Molecular weights (g/mol)
MW_FE_SO4_7H2O = 278.01
MW_NA2SO4 = 142.04
MW_H3BO3 = 61.83
MW_ASCORBIC = 176.12
MW_FE = 55.845
RHO_FE = 7.874 # g/cm^3
F = 96485.33212 # C/mol


@dataclass
class ChemicalRecipe:
    """Batch recipe for preparing 1.0 L of electrowinning bath."""
    c_FeSO4_M: float = 1.0
    c_Na2SO4_M: float = 0.5
    c_H3BO3_M: float = 0.4
    c_ascorbic_g_L: float = 2.0
    target_pH: float = 2.5
    volume_L: float = 1.0


def calculate_batch_recipe(recipe: ChemicalRecipe) -> Dict[str, Any]:
    """Calculate exact chemical masses required for a lab batch volume."""
    vol = recipe.volume_L

    mass_FeSO4_7H2O_g = recipe.c_FeSO4_M * vol * MW_FE_SO4_7H2O
    mass_Na2SO4_g = recipe.c_Na2SO4_M * vol * MW_NA2SO4
    mass_H3BO3_g = recipe.c_H3BO3_M * vol * MW_H3BO3
    mass_ascorbic_g = recipe.c_ascorbic_g_L * vol

    # Estimate concentrated H2SO4 (98% w/w, d=1.84 g/mL, ~18.4 M) for pH adjustment
    # Assuming initial unadjusted pH ~ 3.5 for 1 M FeSO4
    target_H_M = 10.0 ** (-recipe.target_pH)
    vol_H2SO4_98_mL = max(0.0, (target_H_M * vol / (2.0 * 18.4)) * 1000.0)

    return {
        "volume_L": vol,
        "c_FeSO4_M": recipe.c_FeSO4_M,
        "c_Na2SO4_M": recipe.c_Na2SO4_M,
        "c_H3BO3_M": recipe.c_H3BO3_M,
        "c_ascorbic_g_L": recipe.c_ascorbic_g_L,
        "target_pH": recipe.target_pH,
        "FeSO4_7H2O_g": float(mass_FeSO4_7H2O_g),
        "Na2SO4_g": float(mass_Na2SO4_g),
        "H3BO3_g": float(mass_H3BO3_g),
        "ascorbic_acid_g": float(mass_ascorbic_g),
        "est_H2SO4_98pct_mL": float(vol_H2SO4_98_mL),
    }


def predict_plating_run(
    j_mA_cm2: float,
    area_cm2: float,
    t_run_hr: float,
    pH_bulk: float,
    T_C: float,
    c_Fe2_M: float = 1.0
) -> Dict[str, Any]:
    """Predict yield, deposit thickness, charge, and energy for a lab plating run."""
    op_res = evaluate_operating_point(
        j_mA_cm2=j_mA_cm2,
        pH_bulk=pH_bulk,
        T_C=T_C,
        c_Fe2_M=c_Fe2_M
    )

    FE = op_res["FE"]
    V_cell = op_res["V_cell"]

    I_A = (j_mA_cm2 / 1000.0) * area_cm2
    t_sec = t_run_hr * 3600.0
    Q_coulombs = I_A * t_sec

    # Mass yield via Faraday's law
    m_fe_theoretical_g = (Q_coulombs * MW_FE) / (2.0 * F)
    m_fe_expected_g = m_fe_theoretical_g * FE

    # Expected thickness in um
    thickness_um = (m_fe_expected_g / (RHO_FE * area_cm2)) * 10000.0

    # Energy consumed in Wh
    energy_Wh = (I_A * V_cell * t_sec) / 3600.0

    return {
        "j_mA_cm2": j_mA_cm2,
        "area_cm2": area_cm2,
        "current_A": float(I_A),
        "t_run_hr": t_run_hr,
        "charge_Coulombs": float(Q_coulombs),
        "predicted_FE": float(FE),
        "V_cell": float(V_cell),
        "m_fe_expected_g": float(m_fe_expected_g),
        "deposit_thickness_um": float(thickness_um),
        "energy_Wh": float(energy_Wh),
        "pass_status": op_res["is_pass"],
        "reasons": op_res["reasons"],
    }


def generate_factorial_doe(
    j_levels: List[float] = [100.0, 250.0, 400.0],
    pH_levels: List[float] = [2.0, 3.0],
    T_levels: List[float] = [35.0, 50.0, 65.0],
    area_cm2: float = 10.0,
    t_run_hr: float = 2.0
) -> pd.DataFrame:
    """Generate full-factorial DOE run matrix for Phase I/II lab trials."""
    rows = []
    run_id = 1

    for j in j_levels:
        for pH in pH_levels:
            for T in T_levels:
                pred = predict_plating_run(
                    j_mA_cm2=j,
                    area_cm2=area_cm2,
                    t_run_hr=t_run_hr,
                    pH_bulk=pH,
                    T_C=T
                )

                rows.append({
                    "run_id": f"RUN-{run_id:03d}",
                    "j_mA_cm2": j,
                    "pH_bulk": pH,
                    "T_C": T,
                    "current_A": round(pred["current_A"], 3),
                    "t_run_hr": t_run_hr,
                    "predicted_FE_pct": round(pred["predicted_FE"] * 100.0, 1),
                    "V_cell_V": round(pred["V_cell"], 2),
                    "m_fe_expected_g": round(pred["m_fe_expected_g"], 3),
                    "thickness_um": round(pred["deposit_thickness_um"], 1),
                    "energy_Wh": round(pred["energy_Wh"], 2),
                    "pass_status": pred["pass_status"],
                    "reasons": "; ".join(pred["reasons"]) if pred["reasons"] else "PASS"
                })
                run_id += 1

    df = pd.DataFrame(rows)
    return df
