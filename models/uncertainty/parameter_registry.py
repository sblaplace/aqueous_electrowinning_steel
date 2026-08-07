"""
Central parameter registry for the aqueous electrowinning model chain.

Every screening coefficient, empirical constant, and literature-sourced
correlation lives here as a :class:`Parameter` entry carrying its nominal
value, uncertainty, bounds, distribution type, and literature source.

The registry is the single source of truth for uncertainty propagation,
sensitivity analysis, and calibration workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple


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
    "her_i0", 1.0e-6, 5.0e-4, (1.0e-7, 1.0e-1), "lognormal",
    "HER exchange current density (screening) — nominal 1e-6 per ProcessConditions.her_i0 / SCREENING_SENSITIVITY_BUDGET; 1e-3 in the registry was a 1000x regression that collapsed nominal FE to ~11%", module="kinetics")

REGISTRY["fe_tafel_V"] = _p(
    "fe_tafel_V", 0.120, 0.020, (0.060, 0.200), "normal",
    "Fe cathodic Tafel slope (screening)", module="kinetics")

REGISTRY["fe_i0_Ea_J_mol"] = _p(
    "fe_i0_Ea_J_mol", 50.0e3, 12.0e3, (25.0e3, 85.0e3), "uniform",
    "Apparent activation energy, Fe2+ deposition (screening; metal-deposition "
    "family 40-60 kJ/mol)", module="kinetics")

REGISTRY["her_i0_Ea_J_mol"] = _p(
    "her_i0_Ea_J_mol", 60.0e3, 15.0e3, (30.0e3, 95.0e3), "uniform",
    "Apparent activation energy, HER on Fe (screening; iron-group HER family "
    "50-90 kJ/mol)", module="kinetics")

REGISTRY["her_tafel_V"] = _p(
    "her_tafel_V", 0.140, 0.020, (0.080, 0.220), "normal",
    "HER cathodic Tafel slope (screening)", module="kinetics")

REGISTRY["fe_E_eq"] = _p(
    "fe_E_eq", -0.440, 0.010, (-0.500, -0.380), "normal",
    "Fe2+/Fe standard potential V vs SHE (CRC)", module="kinetics")

# ---------------------------------------------------------------------------
# surface_state.py — DFT-anchored hydrogen adsorption free energy
# ---------------------------------------------------------------------------
# The single largest *untracked* economics uncertainty (CHEM_PHYS_IMPROVEMENTS
# §3.1). dG_H* anchors the Volmer/Heyrovsky hydrogen coverage theta_H(eta)
# (models/surface_state.py volmer_coverage) which sets the effective HER
# exchange current i0,H_eff; that propagates to FE at the gate and thence to
# V_cell/kWh and LCOFe. The +/-0.15 eV DFT band (Norskov-family CHE volcano
# scatter across Fe(110)/(100)/(211) low-index facets) swings i0,H by ~2-3x
# and FE by ~10-15% — the dominant economics lever.
#
# Units: eV (registry mean/std/bounds). Converted to J/mol (x F) where it
# enters volmer_coverage (surface_state.py DG_HSTAR_FE110_J = -0.40*F anchors
# the nominal). Bounds span the full +/-0.15 eV declared DFT band so the
# MC/Sobol sweep exercises it end-to-end.
REGISTRY["dG_Hstar_eV"] = _p(
    "dG_Hstar_eV", -0.40, 0.15, (-0.55, -0.25), "normal",
    "DFT hydrogen-adsorption free energy DeltaG_H* on Fe (Norskov CHE volcano, "
    "Fe(110)/(100)/(211) low-index facets; surface_state.py)",
    module="surface_state")

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
# electrochemistry.py — temperature, membrane, anode shuttle
# ---------------------------------------------------------------------------
_echem_src = "CRC Handbook; Nafion 117 datasheet; screening electrolysis literature"

REGISTRY["T_operating_C"] = _p(
    "T_operating_C", 60.0, 10.0, (25.0, 90.0), "triangular",
    "Operating temperature degC (screening)", module="electrochemistry")

REGISTRY["kappa_ref_S_m"] = _p(
    "kappa_ref_S_m", 10.0, 3.0, (2.0, 25.0), "triangular",
    "Electrolyte conductivity at 25C S/m (CRC)", module="electrochemistry")

REGISTRY["kappa_Ea_kJ_mol"] = _p(
    "kappa_Ea_kJ_mol", 15.0, 3.0, (8.0, 25.0), "triangular",
    "Ionic conductivity activation energy kJ/mol (screening)", module="electrochemistry")

# 2026-08 additions: the anode-side conductivity of the divided cell is
# carried as an explicitly assumed/unmeasured screening parameter (no
# instrument reading exists), and the Fe2+ autoxidation rate mirrors
# bath_startup.k_ox_ref (Sung & Morgan screening value).  Both have
# negligible prior variance next to the Ea block (std^2 = 400 and 2.5e-9
# vs the 4.28e8 registry total), so planner rankings do not move.
REGISTRY["anolyte_conductivity_S_m"] = _p(
    "anolyte_conductivity_S_m", 60.0, 20.0, (20.0, 100.0), "uniform",
    "Divided-cell anolyte conductivity S/m (assumed, unmeasured; screening)",
    module="cell")

REGISTRY["fe2_autoxidation_k_ref"] = _p(
    "fe2_autoxidation_k_ref", 1.0e-4, 5.0e-5, (1.0e-5, 1.0e-3), "lognormal",
    "Fe2+ autoxidation rate constant M^-1 s^-1 at 25 C, pH 2 "
    "(Sung & Morgan 1980 screening; models/bath_startup.py)", module="bath_startup")

REGISTRY["interelectrode_gap_m"] = _p(
    "interelectrode_gap_m", 0.02, 0.005, (0.005, 0.05), "triangular",
    "Interelectrode gap m (screening)", module="electrochemistry")

REGISTRY["contact_resistance_ohm_m2"] = _p(
    "contact_resistance_ohm_m2", 5.0e-4, 2.0e-4, (1.0e-4, 2.0e-3), "lognormal",
    "Contact/busbar area-specific resistance ohm.m2 (screening)", module="electrochemistry")

REGISTRY["membrane_R_ohm_m2"] = _p(
    "membrane_R_ohm_m2", 0.002, 0.001, (0.0005, 0.01), "lognormal",
    "Nafion 117 membrane resistance ohm.m2 (Nafion datasheet)", module="electrochemistry")

REGISTRY["membrane_fe3_crossover_1_hr"] = _p(
    "membrane_fe3_crossover_1_hr", 0.05, 0.03, (0.001, 0.20), "lognormal",
    "Fe3+ crossover rate through membrane 1/hr (screening)", module="electrochemistry")

REGISTRY["membrane_cost_per_m2"] = _p(
    "membrane_cost_per_m2", 500.0, 100.0, (100.0, 1500.0), "triangular",
    "Membrane cost $/m2 (screening)", module="electrochemistry")

REGISTRY["E0_Fe3_Fe2_V"] = _p(
    "E0_Fe3_Fe2_V", 0.771, 0.005, (0.750, 0.800), "normal",
    "Fe3+/Fe2+ standard potential V vs SHE (CRC)", module="electrochemistry")

REGISTRY["fe_shuttle_i0"] = _p(
    "fe_shuttle_i0", 0.10, 0.05, (0.001, 1.0), "lognormal",
    "Fe2+/Fe3+ exchange current density A/m2 (screening)", module="electrochemistry")

REGISTRY["fe_shuttle_tafel_V"] = _p(
    "fe_shuttle_tafel_V", 0.120, 0.020, (0.060, 0.200), "normal",
    "Fe2+/Fe3+ anodic Tafel slope V/decade (screening)", module="electrochemistry")

REGISTRY["oer_tafel_V"] = _p(
    "oer_tafel_V", 0.060, 0.015, (0.030, 0.120), "normal",
    "OER anodic Tafel slope V/decade (Trasatti 2000)", module="electrochemistry")

# ---------------------------------------------------------------------------
# thermal_balance.py — cell heat-transfer / thermal-management properties
# These are the thermal/transport props that must be shared *with* the
# electrochem and envelope layers so a single parameterization closes the
# thermal transient (Q_gen from V_cell/I vs ambient + jacket losses).
# ---------------------------------------------------------------------------
_therm_src = "Incropera et al.; Danly 1981; screening electrolysis cell design"

REGISTRY["thermoneutral_V"] = _p(
    "thermoneutral_V", 1.28, 0.05, (1.20, 1.40), "normal",
    "Thermoneutral potential Fe deposit + OER at 25C, V (Danly 1981)", module="thermal")

REGISTRY["volume_L"] = _p(
    "volume_L", 40.0, 10.0, (2.0, 500.0), "triangular",
    "Single-cell electrolyte volume L (screening)", module="thermal")

REGISTRY["hardware_C_J_K"] = _p(
    "hardware_C_J_K", 500.0, 100.0, (100.0, 2000.0), "triangular",
    "Cell body/electrode thermal mass J/K (screening)", module="thermal")

REGISTRY["UA_amb_W_K"] = _p(
    "UA_amb_W_K", 3.0, 1.0, (0.5, 50.0), "triangular",
    "Ambient overall heat-transfer coeff x area W/K (Incropera screening)", module="thermal")

REGISTRY["A_surface_m2"] = _p(
    "A_surface_m2", 0.04, 0.02, (0.005, 2.0), "triangular",
    "Open-top electrolyte surface area for evaporation m2 (cell design)", module="thermal")

REGISTRY["relative_humidity"] = _p(
    "relative_humidity", 0.50, 0.20, (0.0, 1.0), "uniform",
    "Ambient relative humidity (site climatology)", module="thermal")

REGISTRY["T_ambient_C"] = _p(
    "T_ambient_C", 25.0, 8.0, (-20.0, 45.0), "uniform",
    "Ambient air temperature C (shared site input)", module="thermal")

REGISTRY["UA_jacket_W_K"] = _p(
    "UA_jacket_W_K", 25.0, 10.0, (0.0, 1000.0), "triangular",
    "Active cooling-jacket UA W/K (heat-exchanger screening)", module="thermal")

REGISTRY["electrode_area_m2"] = _p(
    "electrode_area_m2", 0.05, 0.02, (0.001, 2.0), "triangular",
    "Geometric cathode area m2 (cell architecture)", module="thermal")

# ---------------------------------------------------------------------------
# crate.py — envelope / site structural-environmental properties
# Shared with env_coupling.py: the same wind gust, ambient T and rain drive
# both the crate stability verdict and the thermal ambient-loss disturbance.
# ---------------------------------------------------------------------------
_crate_src = "screening container/civil design; ASCE 7; site climatology"

REGISTRY["crate_mass_kg"] = _p(
    "crate_mass_kg", 4500.0, 1000.0, (500.0, 20000.0), "triangular",
    "Crate self-mass excluding ballast kg (container spec)", module="crate")

REGISTRY["crate_length_m"] = _p(
    "crate_length_m", 12.19, 0.0, (6.0, 15.0), "uniform",
    "Envelope length m (40-ft container nominal)", module="crate")

REGISTRY["crate_width_m"] = _p(
    "crate_width_m", 2.44, 0.0, (2.0, 3.0), "uniform",
    "Envelope width m (container nominal)", module="crate")

REGISTRY["crate_height_m"] = _p(
    "crate_height_m", 2.59, 0.0, (2.0, 4.0), "uniform",
    "Envelope height m (container nominal)", module="crate")

REGISTRY["drag_coefficient"] = _p(
    "drag_coefficient", 1.2, 0.1, (0.8, 2.0), "triangular",
    "Envelope drag coefficient (box/container shape)", module="crate")

REGISTRY["design_gust_m_s"] = _p(
    "design_gust_m_s", 40.0, 10.0, (10.0, 80.0), "triangular",
    "Design 3-s wind gust m/s at site (ASCE 7 / site survey)", module="crate")

REGISTRY["soil_bearing_kPa"] = _p(
    "soil_bearing_kPa", 100.0, 30.0, (30.0, 300.0), "triangular",
    "Allowable soil bearing pressure kPa (geotech screening)", module="crate")

REGISTRY["rain_design_mm_hr"] = _p(
    "rain_design_mm_hr", 50.0, 25.0, (0.0, 200.0), "triangular",
    "Design rainfall intensity mm/hr (site climatology)", module="crate")

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
