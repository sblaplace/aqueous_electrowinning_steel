"""
Central parameter registry for the aqueous electrowinning model chain.

Every screening coefficient, empirical constant, and literature-sourced
correlation lives here as a :class:`Parameter` entry carrying its nominal
value, uncertainty, bounds, distribution type, and literature source.

The registry is the single source of truth for uncertainty propagation,
sensitivity analysis, and calibration workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple


@dataclass(frozen=True)
class Parameter:
    """A named model coefficient with its uncertainty distribution."""

    name: str
    mean: float
    std: float
    bounds: Tuple[float, float]
    distribution: Literal["normal", "uniform", "lognormal", "triangular"]
    source: str
    calibrated: bool = False
    module: str = ""

    def __repr__(self) -> str:
        tag = "cal" if self.calibrated else "scr"
        return f"Parameter({self.name}={self.mean:.4g} +/- {self.std:.4g} [{tag}])"


# ---------------------------------------------------------------------------
# Helper for compact registry entries
# ---------------------------------------------------------------------------

def _p(
    name: str,
    mean: float,
    std: float,
    bounds: Tuple[float, float],
    distribution: Literal["normal", "uniform", "lognormal", "triangular"],
    source: str,
    module: str = "",
    calibrated: bool = False,
) -> Parameter:
    return Parameter(
        name=name, mean=mean, std=std, bounds=bounds,
        distribution=distribution, source=source, module=module,
        calibrated=calibrated,
    )


# ===========================================================================
# Registry: >=40 named parameters across all model modules
# ===========================================================================

REGISTRY: Dict[str, Parameter] = {}

# ---------------------------------------------------------------------------
# mechanical_properties.py
# ---------------------------------------------------------------------------
_mech_src = "Takaki 2002; Rajagopalan & Vaidya; Leslie 1981; screening composites"

REGISTRY["sigma0_fe_MPa"] = _p(
    "sigma0_fe_MPa", 100.0, 30.0, (20.0, 200.0), "triangular",
    "Hall-Petch friction stress bcc Fe (Takaki 2002)", module="mechanical_properties")

REGISTRY["k_hp_MPa_sqrt_m"] = _p(
    "k_hp_MPa_sqrt_m", 0.50, 0.10, (0.20, 0.80), "triangular",
    "Hall-Petch slope bcc Fe (Rajagopalan & Vaidya)", module="mechanical_properties")

REGISTRY["k_ss_ni_MPa_per_wt"] = _p(
    "k_ss_ni_MPa_per_wt", 38.0, 8.0, (10.0, 70.0), "triangular",
    "Ni solid-solution in ferrite (Leslie 1981)", module="mechanical_properties")

REGISTRY["ss_ni_exp"] = _p(
    "ss_ni_exp", 0.75, 0.10, (0.40, 1.0), "triangular",
    "Ni SS saturation exponent (screening)", module="mechanical_properties")

REGISTRY["ss_ni_sat_wt"] = _p(
    "ss_ni_sat_wt", 20.0, 5.0, (8.0, 35.0), "triangular",
    "Ni SS saturation threshold wt% (screening)", module="mechanical_properties")

REGISTRY["k_carbon_MPa_per_wt"] = _p(
    "k_carbon_MPa_per_wt", 180.0, 40.0, (50.0, 350.0), "triangular",
    "Carbon dispersion strengthening (composite literature)", module="mechanical_properties")

REGISTRY["carbon_nl_exp"] = _p(
    "carbon_nl_exp", 0.60, 0.15, (0.20, 1.0), "triangular",
    "Sub-linear carbon loading exponent (screening)", module="mechanical_properties")

REGISTRY["carbon_size_ref_um"] = _p(
    "carbon_size_ref_um", 1.5, 0.5, (0.3, 5.0), "triangular",
    "Reference particle size um (screening)", module="mechanical_properties")

REGISTRY["carbon_size_exp"] = _p(
    "carbon_size_exp", -0.25, 0.10, (-0.60, 0.0), "triangular",
    "Orowan particle-size exponent (screening)", module="mechanical_properties")

REGISTRY["load_transfer_frac"] = _p(
    "load_transfer_frac", 0.15, 0.05, (0.02, 0.40), "triangular",
    "Shear-lag load-transfer fraction (screening)", module="mechanical_properties")

REGISTRY["porosity_penalty_exp"] = _p(
    "porosity_penalty_exp", 1.8, 0.3, (0.8, 3.0), "triangular",
    "Gibson-Ashby porosity knockdown exponent (screening)", module="mechanical_properties")

REGISTRY["porosity_max"] = _p(
    "porosity_max", 0.30, 0.05, (0.10, 0.50), "triangular",
    "Maximum equivalent porosity (screening)", module="mechanical_properties")

REGISTRY["tabor_factor"] = _p(
    "tabor_factor", 3.2, 0.3, (2.5, 4.0), "triangular",
    "Tabor HV/sigma_y factor (screening)", module="mechanical_properties")

REGISTRY["uts_over_ys_base"] = _p(
    "uts_over_ys_base", 1.25, 0.10, (1.05, 1.60), "triangular",
    "Base UTS/YS ratio (screening)", module="mechanical_properties")

REGISTRY["elongation_base_pct"] = _p(
    "elongation_base_pct", 22.0, 5.0, (5.0, 40.0), "triangular",
    "Base elongation pure Fe pct (screening)", module="mechanical_properties")

REGISTRY["G_fe_GPa"] = _p(
    "G_fe_GPa", 80.0, 5.0, (60.0, 100.0), "normal",
    "Shear modulus bcc Fe (CRC Handbook)", module="mechanical_properties")

REGISTRY["burgers_b_m"] = _p(
    "burgers_b_m", 0.25e-9, 0.01e-9, (0.20e-9, 0.30e-9), "normal",
    "Burgers vector bcc Fe (CRC Handbook)", module="mechanical_properties")

# Grain-size estimation parameters
REGISTRY["grain_d0_dc_ref_um"] = _p(
    "grain_d0_dc_ref_um", 3.5, 1.0, (0.5, 8.0), "triangular",
    "DC reference grain size at 100 mA/cm2 (screening)", module="mechanical_properties")

REGISTRY["grain_j_ref_mA_cm2"] = _p(
    "grain_j_ref_mA_cm2", 100.0, 20.0, (30.0, 300.0), "triangular",
    "Reference current density for grain-size model (screening)", module="mechanical_properties")

REGISTRY["grain_j_exponent"] = _p(
    "grain_j_exponent", 0.30, 0.05, (0.10, 0.50), "triangular",
    "Grain-size current-density exponent (screening)", module="mechanical_properties")

REGISTRY["grain_pe_factor_base"] = _p(
    "grain_pe_factor_base", 0.65, 0.15, (0.20, 1.0), "triangular",
    "PE grain-size reduction factor at 50pct duty (screening)", module="mechanical_properties")

REGISTRY["grain_pre_factor_base"] = _p(
    "grain_pre_factor_base", 0.35, 0.10, (0.10, 0.70), "triangular",
    "PRE grain-size reduction factor (screening)", module="mechanical_properties")

# ---------------------------------------------------------------------------
# carburization.py
# ---------------------------------------------------------------------------
_carb_src = "Wert & Zener; Maynier et al.; screening diffusion literature"

REGISTRY["D0_ferrite_m2_s"] = _p(
    "D0_ferrite_m2_s", 6.2e-7, 2.0e-7, (1.0e-7, 3.0e-6), "lognormal",
    "C diffusivity pre-exponential bcc a-Fe (Wert & Zener)", module="carburization")

REGISTRY["Q_ferrite_kJ_mol"] = _p(
    "Q_ferrite_kJ_mol", 80.0, 10.0, (50.0, 120.0), "normal",
    "C diffusion activation energy bcc a-Fe (literature)", module="carburization")

REGISTRY["D0_austenite_m2_s"] = _p(
    "D0_austenite_m2_s", 2.3e-5, 5.0e-6, (1.0e-5, 5.0e-5), "lognormal",
    "C diffusivity pre-exponential fcc g-Fe (literature)", module="carburization")

REGISTRY["Q_austenite_kJ_mol"] = _p(
    "Q_austenite_kJ_mol", 148.0, 15.0, (100.0, 200.0), "normal",
    "C diffusion activation energy fcc g-Fe (literature)", module="carburization")

REGISTRY["a3_temp_C"] = _p(
    "a3_temp_C", 912.0, 5.0, (890.0, 930.0), "normal",
    "a->g transition temperature pure Fe (ASM)", module="carburization")

REGISTRY["HV_base_Maynier"] = _p(
    "HV_base_Maynier", 127.0, 20.0, (60.0, 200.0), "normal",
    "Maynier hardness intercept (Maynier et al.)", module="carburization")

REGISTRY["HV_per_C_wt_Maynier"] = _p(
    "HV_per_C_wt_Maynier", 949.0, 100.0, (600.0, 1300.0), "normal",
    "Maynier HV per wt pct C coefficient (Maynier et al.)", module="carburization")

REGISTRY["HV_sat"] = _p(
    "HV_sat", 900.0, 50.0, (700.0, 1100.0), "normal",
    "As-quenched martensite hardness cap (screening)", module="carburization")

# ---------------------------------------------------------------------------
# carbon_potential.py — deltaG correlations (J/mol)
# ---------------------------------------------------------------------------
REGISTRY["dG_boudouard_intercept"] = _p(
    "dG_boudouard_intercept", 170700.0, 5000.0, (150000.0, 195000.0), "normal",
    "Boudouard dG intercept (Richardson-Ellingham)", module="carbon_potential")

REGISTRY["dG_boudouard_slope"] = _p(
    "dG_boudouard_slope", -174.5, 5.0, (-200.0, -150.0), "normal",
    "Boudouard dG slope (Richardson-Ellingham)", module="carbon_potential")

REGISTRY["dG_ch4_intercept"] = _p(
    "dG_ch4_intercept", 90000.0, 5000.0, (70000.0, 110000.0), "normal",
    "CH4 cracking dG intercept (screening)", module="carbon_potential")

REGISTRY["dG_ch4_slope"] = _p(
    "dG_ch4_slope", -109.0, 5.0, (-130.0, -85.0), "normal",
    "CH4 cracking dG slope (screening)", module="carbon_potential")

REGISTRY["dG_wgs_intercept"] = _p(
    "dG_wgs_intercept", -41000.0, 3000.0, (-55000.0, -28000.0), "normal",
    "Water-gas shift dG intercept (screening)", module="carbon_potential")

REGISTRY["dG_wgs_slope"] = _p(
    "dG_wgs_slope", 41.5, 3.0, (30.0, 55.0), "normal",
    "Water-gas shift dG slope (screening)", module="carbon_potential")

# ---------------------------------------------------------------------------
# tempering.py
# ---------------------------------------------------------------------------
REGISTRY["Ms_intercept"] = _p(
    "Ms_intercept", 539.0, 15.0, (490.0, 590.0), "normal",
    "Andrews Ms intercept (Andrews 1965)", module="tempering")

REGISTRY["Ms_C_coeff"] = _p(
    "Ms_C_coeff", -423.0, 30.0, (-520.0, -320.0), "normal",
    "Andrews Ms C coefficient", module="tempering")

REGISTRY["Ms_Mn_coeff"] = _p(
    "Ms_Mn_coeff", -30.4, 5.0, (-45.0, -15.0), "normal",
    "Andrews Ms Mn coefficient", module="tempering")

REGISTRY["Ms_Ni_coeff"] = _p(
    "Ms_Ni_coeff", -17.7, 3.0, (-28.0, -8.0), "normal",
    "Andrews Ms Ni coefficient", module="tempering")

REGISTRY["Ms_Cr_coeff"] = _p(
    "Ms_Cr_coeff", -12.1, 2.0, (-18.0, -5.0), "normal",
    "Andrews Ms Cr coefficient", module="tempering")

REGISTRY["Ms_Mo_coeff"] = _p(
    "Ms_Mo_coeff", -7.5, 1.5, (-12.0, -3.0), "normal",
    "Andrews Ms Mo coefficient", module="tempering")

REGISTRY["KM_alpha_K_inv"] = _p(
    "KM_alpha_K_inv", 0.011, 0.002, (0.005, 0.020), "normal",
    "Koistinen-Marburger alpha (Koistinen & Marburger 1959)", module="tempering")

REGISTRY["C_HJ"] = _p(
    "C_HJ", 19.5, 2.0, (15.0, 25.0), "normal",
    "Hollomon-Jaffe C constant (Hollomon & Jaffe 1945)", module="tempering")

REGISTRY["k_softening"] = _p(
    "k_softening", 0.00018, 0.00005, (0.00005, 0.0005), "triangular",
    "Tempering softening rate constant (screening)", module="tempering")

REGISTRY["softening_floor"] = _p(
    "softening_floor", 0.35, 0.05, (0.15, 0.55), "triangular",
    "Tempering softening floor fraction (screening)", module="tempering")

# ---------------------------------------------------------------------------
# transport.py — ionic diffusivities (m2/s) and water equilibrium
# ---------------------------------------------------------------------------
REGISTRY["D_Fe2"] = _p(
    "D_Fe2", 7.2e-10, 1.0e-10, (4.0e-10, 12.0e-10), "lognormal",
    "Fe2+ diffusivity at inf dilution 25C (CRC)", module="transport")

REGISTRY["D_H_plus"] = _p(
    "D_H_plus", 9.31e-9, 1.0e-9, (5.0e-9, 13.0e-9), "lognormal",
    "H+ diffusivity at inf dilution 25C (CRC)", module="transport")

REGISTRY["D_OH_minus"] = _p(
    "D_OH_minus", 5.27e-9, 1.0e-9, (3.0e-9, 9.0e-9), "lognormal",
    "OH- diffusivity at inf dilution 25C (CRC)", module="transport")

REGISTRY["D_Na_plus"] = _p(
    "D_Na_plus", 1.33e-9, 0.3e-9, (0.5e-9, 2.5e-9), "lognormal",
    "Na+ diffusivity at inf dilution 25C (CRC)", module="transport")

REGISTRY["D_SO4_2minus"] = _p(
    "D_SO4_2minus", 1.07e-9, 0.3e-9, (0.3e-9, 2.5e-9), "lognormal",
    "SO4 2- diffusivity at inf dilution 25C (CRC)", module="transport")

# ---------------------------------------------------------------------------
# kinetics.py — exchange current densities and Tafel slopes
# ---------------------------------------------------------------------------
REGISTRY["fe_i0"] = _p(
    "fe_i0", 1.0e-2, 5.0e-3, (1.0e-4, 1.0e-1), "lognormal",
    "Fe2+/Fe exchange current density (screening)", module="kinetics")

REGISTRY["her_i0"] = _p(
    "her_i0", 1.0e-3, 5.0e-4, (1.0e-7, 1.0e-1), "lognormal",
    "HER exchange current density (screening)", module="kinetics")

REGISTRY["fe_tafel_V"] = _p(
    "fe_tafel_V", 0.120, 0.020, (0.060, 0.200), "normal",
    "Fe cathodic Tafel slope (screening)", module="kinetics")

REGISTRY["her_tafel_V"] = _p(
    "her_tafel_V", 0.140, 0.020, (0.080, 0.220), "normal",
    "HER cathodic Tafel slope (screening)", module="kinetics")

REGISTRY["fe_E_eq"] = _p(
    "fe_E_eq", -0.440, 0.010, (-0.500, -0.380), "normal",
    "Fe2+/Fe standard potential V vs SHE (CRC)", module="kinetics")

# ---------------------------------------------------------------------------
# co_deposition.py — Guglielmi and Ni kinetics
# ---------------------------------------------------------------------------
REGISTRY["guglielmi_k_ref"] = _p(
    "guglielmi_k_ref", 0.015, 0.005, (0.001, 0.050), "lognormal",
    "Guglielmi Langmuir adsorption coefficient L/g (Guglielmi 1972)", module="co_deposition")

REGISTRY["rho_carbon"] = _p(
    "rho_carbon", 2200.0, 200.0, (1500.0, 3000.0), "normal",
    "Carbon particle density kg/m3 (screening)", module="co_deposition")

REGISTRY["d_p_default_um"] = _p(
    "d_p_default_um", 1.5, 0.5, (0.1, 10.0), "lognormal",
    "Default carbon particle diameter um (screening)", module="co_deposition")

REGISTRY["ni_i0"] = _p(
    "ni_i0", 5.0e-3, 2.0e-3, (1.0e-4, 5.0e-2), "lognormal",
    "Ni2+/Ni exchange current density (screening)", module="co_deposition")

REGISTRY["ni_tafel_V"] = _p(
    "ni_tafel_V", 0.100, 0.015, (0.050, 0.180), "normal",
    "Ni cathodic Tafel slope (screening)", module="co_deposition")

REGISTRY["D_Ni2"] = _p(
    "D_Ni2", 6.6e-10, 1.0e-10, (3.0e-10, 12.0e-10), "lognormal",
    "Ni2+ diffusivity at inf dilution (CRC)", module="co_deposition")

# ---------------------------------------------------------------------------
# pulse.py
# ---------------------------------------------------------------------------
REGISTRY["D_Fe2_pulse"] = _p(
    "D_Fe2_pulse", 6.0e-10, 1.0e-10, (3.0e-10, 10.0e-10), "lognormal",
    "Fe2+ diffusivity for pulse model (screening)", module="pulse")

REGISTRY["D_H_pulse"] = _p(
    "D_H_pulse", 9.31e-9, 1.0e-9, (5.0e-9, 13.0e-9), "lognormal",
    "H+ diffusivity for pulse model (CRC)", module="pulse")

# ---------------------------------------------------------------------------
# anode.py — OER/CER and bubble parameters
# ---------------------------------------------------------------------------
REGISTRY["E0_OER_acidic"] = _p(
    "E0_OER_acidic", 1.229, 0.005, (1.210, 1.250), "normal",
    "OER standard potential V vs SHE (CRC)", module="anode")

REGISTRY["E0_CER"] = _p(
    "E0_CER", 1.360, 0.005, (1.340, 1.380), "normal",
    "CER standard potential V vs SHE (CRC)", module="anode")

REGISTRY["bubble_j_char_mA_cm2"] = _p(
    "bubble_j_char_mA_cm2", 150.0, 50.0, (30.0, 400.0), "triangular",
    "Characteristic current density for bubble saturation (screening)", module="anode")

REGISTRY["bubble_temp_coeff"] = _p(
    "bubble_temp_coeff", 0.007, 0.003, (0.001, 0.020), "triangular",
    "Bubble fraction temperature correction K^-1 (screening)", module="anode")

REGISTRY["oer_ea_IrO2_kJ_mol"] = _p(
    "oer_ea_IrO2_kJ_mol", 40.0, 10.0, (15.0, 80.0), "normal",
    "OER activation energy IrO2 kJ/mol (Trasatti 2000)", module="anode")

# ---------------------------------------------------------------------------
# closed_loop.py — anode durability and CSTR dynamics
# ---------------------------------------------------------------------------
REGISTRY["coating_loading_g_m2"] = _p(
    "coating_loading_g_m2", 12.0, 3.0, (3.0, 30.0), "triangular",
    "Anode coating loading g/m2 (screening)", module="closed_loop")

REGISTRY["base_wear_mg_per_kAh"] = _p(
    "base_wear_mg_per_kAh", 0.35, 0.15, (0.05, 1.0), "triangular",
    "Base coating wear mg/kAh (screening)", module="closed_loop")

REGISTRY["temp_accel_per_C"] = _p(
    "temp_accel_per_C", 0.025, 0.010, (0.005, 0.080), "triangular",
    "Temperature acceleration factor per degC (screening)", module="closed_loop")

REGISTRY["activity_exponent"] = _p(
    "activity_exponent", 1.5, 0.3, (0.5, 3.0), "triangular",
    "Coating activity degradation exponent (screening)", module="closed_loop")

REGISTRY["resistance_growth_ohm_m2"] = _p(
    "resistance_growth_ohm_m2", 4.0e-4, 2.0e-4, (1.0e-5, 2.0e-3), "lognormal",
    "Anode resistance growth ohm.m2 (screening)", module="closed_loop")

REGISTRY["precipitation_rate_per_hr"] = _p(
    "precipitation_rate_per_hr", 0.5, 0.2, (0.05, 2.0), "triangular",
    "Fe precipitation rate constant 1/hr (screening)", module="closed_loop")

REGISTRY["ligand_decay_per_hr"] = _p(
    "ligand_decay_per_hr", 1.0e-4, 5.0e-5, (1.0e-6, 5.0e-4), "lognormal",
    "Ligand decomposition rate constant 1/hr (screening)", module="closed_loop")

# ---------------------------------------------------------------------------
# Total count check: len(REGISTRY) >= 40
# ---------------------------------------------------------------------------
assert len(REGISTRY) >= 40, f"Registry has {len(REGISTRY)} params, need >= 40"


def registry_summary() -> dict:
    """Return a quick summary of the registry contents."""
    by_module: Dict[str, int] = {}
    for p in REGISTRY.values():
        by_module[p.module] = by_module.get(p.module, 0) + 1
    return {
        "total_parameters": len(REGISTRY),
        "by_module": by_module,
        "calibrated": sum(1 for p in REGISTRY.values() if p.calibrated),
        "screening": sum(1 for p in REGISTRY.values() if not p.calibrated),
    }
