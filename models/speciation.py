"""Multicomponent concentrated electrolyte speciation and activity coefficient model.

Solves non-ideal ion activities, bisulfate dissociation, ferrous sulfate
ion-pairing, ionic strength, electrolyte conductivity, and thermodynamic
Nernst reversible potentials for aqueous iron electrowinning baths.

References:
- Davies, C. W. (1962). Ion Association. Butterworths, London.
- Pitzer, K. S. (1991). Activity Coefficients in Electrolyte Solutions. CRC Press.
- Beverskog, B., & Puigdomenech, I. (1996). Revised Pourbaix diagrams for iron at 25-300 °C.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

# Physical constants
R = 8.314462618  # J/(mol*K)
F = 96485.33212  # C/mol
E0_FE2_FE = -0.447  # V vs SHE for Fe2+ + 2e- -> Fe at 25 °C
KW_25 = 1.0e-14
KSP_FE_OH2_25 = 4.87e-17  # Ksp for Fe(OH)2 at 25 °C
K_HSO4_25 = 1.05e-2  # Ka2 for HSO4- -> H+ + SO4(2-) at 25 °C
K_FESO4_PAIR_25 = 200.0  # Formation constant for FeSO4(aq) pair at 25 °C (L/mol)


@dataclass
class SolutionComposition:
    """Nominal bath recipe concentrations (mol/L or M)."""
    c_FeSO4: float = 1.0      # M FeSO4
    c_Na2SO4: float = 0.5     # M Na2SO4 (supporting electrolyte)
    c_H2SO4: float = 0.01     # M H2SO4 (pH adjustment)
    c_H3BO3: float = 0.4      # M H3BO3 (boric acid buffer)
    T_C: float = 50.0         # °C


def davies_A(T_C: float) -> float:
    """Temperature-dependent Davies / Debye-Hückel A parameter (kg^0.5 / mol^0.5)."""
    T_K = T_C + 273.15
    # Empirical fit for dielectric constant of water vs temperature
    epsilon = 87.740 - 0.4008 * T_C + 9.398e-4 * (T_C ** 2) - 1.410e-6 * (T_C ** 3)
    # A = 1.8246e6 / (epsilon * T_K)^1.5
    A = 1.8246e6 / ((epsilon * T_K) ** 1.5)
    return float(A)


def davies_gamma(z: int, I: float, A: float) -> float:
    """Davies equation for single ion activity coefficient gamma_i.
    
    log10(gamma_i) = -A * z^2 * (sqrt(I)/(1 + sqrt(I)) - 0.3 * I)
    """
    if I <= 0.0:
        return 1.0
    sqrt_I = math.sqrt(I)
    log_gamma = -A * (z ** 2) * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I)
    return float(10.0 ** log_gamma)


def solve_speciation(comp: SolutionComposition, max_iter: int = 200, tol: float = 1e-6) -> Dict[str, Any]:
    """Iteratively solve ionic equilibria and activity coefficients.
    
    Equilibria:
    1) HSO4- <-> H+ + SO4(2-)   (Ka2)
    2) Fe(2+) + SO4(2-) <-> FeSO4(aq)   (K_pair)
    
    Mass balances:
    - Total Fe = c_FeSO4 = [Fe2+] + [FeSO4(aq)]
    - Total Na = 2 * c_Na2SO4
    - Total SO4 = c_FeSO4 + c_Na2SO4 + c_H2SO4 = [SO4(2-)] + [HSO4-] + [FeSO4(aq)]
    - Total H = 2 * c_H2SO4 = [H+] + [HSO4-]
    
    Returns dictionary with free concentrations, activity coefficients, activities,
    ionic strength, conductivity, Nernst potential, and Fe(OH)2 precipitation pH.
    """
    T_K = comp.T_C + 273.15
    A_dh = davies_A(comp.T_C)

    # Temperature adjustment for Ka2(HSO4-): dH0 ~ -22.4 kJ/mol
    Ka2_T = K_HSO4_25 * math.exp((-22400.0 / R) * (1.0 / T_K - 1.0 / 298.15))
    # Temperature adjustment for K_pair(FeSO4): dH0 ~ +8.0 kJ/mol (endothermic pair formation)
    K_pair_T = K_FESO4_PAIR_25 * math.exp((8000.0 / R) * (1.0 / T_K - 1.0 / 298.15))

    # Initial guesses assuming no pairing/association
    c_Fe2 = comp.c_FeSO4
    c_Na = 2.0 * comp.c_Na2SO4
    c_H = 2.0 * comp.c_H2SO4
    c_HSO4 = 0.0
    c_FeSO4_pair = 0.0
    c_SO4 = comp.c_FeSO4 + comp.c_Na2SO4 + comp.c_H2SO4

    # Initial ionic strength
    I = 0.5 * (4.0 * c_Fe2 + 1.0 * c_Na + 1.0 * c_H + 4.0 * c_SO4)

    for _ in range(max_iter):
        I_prev = I

        # Calculate activity coefficients
        gamma1 = davies_gamma(1, I, A_dh)  # H+, Na+, HSO4-
        gamma2 = davies_gamma(2, I, A_dh)  # Fe2+, SO4(2-)
        gamma0 = 1.0                       # Neutral FeSO4(aq)

        # Conditional equilibrium constants (concentration quotient Kc = K_thermo / gamma_ratio)
        Kc_HSO4 = Ka2_T * gamma1 / (gamma1 * gamma2)  # Ka2 = (a_H * a_SO4) / a_HSO4
        Kc_pair = K_pair_T * (gamma2 * gamma2) / gamma0  # K_pair = a_FeSO4 / (a_Fe2 * a_SO4)

        # Inner loop to solve nonlinear system for given gamma values
        for _inner in range(20):
            # c_FeSO4_pair = Kc_pair * c_Fe2 * c_SO4
            # c_Fe2 = comp.c_FeSO4 - c_FeSO4_pair  => c_Fe2 = comp.c_FeSO4 / (1 + Kc_pair * c_SO4)
            c_Fe2 = comp.c_FeSO4 / (1.0 + Kc_pair * c_SO4)
            c_FeSO4_pair = comp.c_FeSO4 - c_Fe2

            # c_HSO4 = (c_H * c_SO4) / Kc_HSO4
            # c_H = 2*c_H2SO4 - c_HSO4  => c_H = 2*c_H2SO4 / (1 + c_SO4 / Kc_HSO4)
            c_H = (2.0 * comp.c_H2SO4) / (1.0 + c_SO4 / Kc_HSO4)
            c_HSO4 = 2.0 * comp.c_H2SO4 - c_H

            # Update c_SO4 from SO4 balance:
            # c_SO4 = Total_SO4 - c_HSO4 - c_FeSO4_pair
            c_SO4_new = (comp.c_FeSO4 + comp.c_Na2SO4 + comp.c_H2SO4) - c_HSO4 - c_FeSO4_pair
            c_SO4_new = max(1e-8, c_SO4_new)

            c_SO4 = 0.5 * (c_SO4 + c_SO4_new)

        # Recalculate ionic strength
        I_new = 0.5 * (4.0 * c_Fe2 + 1.0 * c_Na + 1.0 * c_H + 1.0 * c_HSO4 + 4.0 * c_SO4)
        I = max(1e-6, 0.5 * (I + I_new))

        if abs(I - I_prev) / I < tol:
            break

    # Thermodynamic activities
    gamma_Fe2 = davies_gamma(2, I, A_dh)
    gamma_H = davies_gamma(1, I, A_dh)
    gamma_SO4 = davies_gamma(2, I, A_dh)

    a_Fe2 = gamma_Fe2 * c_Fe2
    a_H = gamma_H * c_H
    pH_act = -math.log10(max(1e-14, a_H))
    pH_conc = -math.log10(max(1e-14, c_H))

    # Nernst potential vs SHE for Fe2+/Fe
    E_rev_Fe = E0_FE2_FE + (R * T_K / (2.0 * F)) * math.log(max(1e-12, a_Fe2))

    # Estimate Kw and Ksp(Fe(OH)2) at temperature T
    Kw_T = KW_25 * math.exp((-55800.0 / R) * (1.0 / T_K - 1.0 / 298.15))
    Ksp_T = KSP_FE_OH2_25 * math.exp((-25000.0 / R) * (1.0 / T_K - 1.0 / 298.15))

    # Hydroxide activity for precipitation: a_OH = sqrt(Ksp_T / a_Fe2)
    a_OH_precip = math.sqrt(max(1e-30, Ksp_T / max(1e-12, a_Fe2)))
    a_H_precip = Kw_T / max(1e-30, a_OH_precip)
    pH_precip = -math.log10(max(1e-14, a_H_precip))

    # Approximate electrolyte conductivity kappa (S/m) via ionic mobility
    # Limiting molar conductivities at 25 °C (S*cm^2/mol -> S*m^2/mol by / 10000)
    # Fe2+: 108.0, Na+: 50.1, H+: 349.6, SO4(2-): 160.0, HSO4-: 50.0
    lambda_25 = {'Fe2': 0.0108, 'Na': 0.00501, 'H': 0.03496, 'SO4': 0.0160, 'HSO4': 0.0050}
    # Viscosity temperature factor ~ 1 + 0.02 * (T_C - 25)
    T_factor = 1.0 + 0.022 * (comp.T_C - 25.0)
    # Ionic strength screening factor (Walden/Debye-Hückel-Onsager effect ~ 1 / (1 + 0.5*sqrt(I)))
    I_factor = 1.0 / (1.0 + 0.45 * math.sqrt(I))

    kappa = 1000.0 * (
        c_Fe2 * lambda_25['Fe2'] +
        c_Na * lambda_25['Na'] +
        c_H * lambda_25['H'] +
        c_SO4 * lambda_25['SO4'] +
        c_HSO4 * lambda_25['HSO4']
    ) * T_factor * I_factor  # S/m

    return {
        "temperature_C": comp.T_C,
        "ionic_strength_M": float(I),
        "davies_A": float(A_dh),
        "gamma_Fe2": float(gamma_Fe2),
        "gamma_H": float(gamma_H),
        "gamma_SO4": float(gamma_SO4),
        "c_Fe2_free_M": float(c_Fe2),
        "c_SO4_free_M": float(c_SO4),
        "c_H_free_M": float(c_H),
        "c_HSO4_free_M": float(c_HSO4),
        "c_FeSO4_pair_M": float(c_FeSO4_pair),
        "a_Fe2": float(a_Fe2),
        "a_H": float(a_H),
        "pH_activity": float(pH_act),
        "pH_concentration": float(pH_conc),
        "E_rev_Fe_V_SHE": float(E_rev_Fe),
        "pH_precip_Fe_OH2": float(pH_precip),
        "conductivity_S_m": float(kappa),
        "fe2_pair_percentage": float(100.0 * c_FeSO4_pair / comp.c_FeSO4),
    }


def speciation_temperature_sweep(comp: SolutionComposition, T_min: float = 20.0, T_max: float = 80.0, num: int = 13) -> Dict[str, np.ndarray]:
    """Perform speciation sweep across temperature range."""
    temps = np.linspace(T_min, T_max, num)
    res_list = []
    for T in temps:
        c = SolutionComposition(c_FeSO4=comp.c_FeSO4, c_Na2SO4=comp.c_Na2SO4,
                                c_H2SO4=comp.c_H2SO4, c_H3BO3=comp.c_H3BO3, T_C=float(T))
        res_list.append(solve_speciation(c))

    out = {"temperature_C": temps}
    keys = res_list[0].keys()
    for k in keys:
        if k != "temperature_C":
            out[k] = np.array([r[k] for r in res_list], dtype=float)
    return out
