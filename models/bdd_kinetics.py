"""
Bockris–Dražić–Despić (BDD) and Epelboin–Wiart multi-step iron deposition microkinetics.

Physics and Chemistry
---------------------
Standard Butler–Volmer models (e.g. :mod:`models.kinetics`) treat iron electrodeposition
as an elementary 2-electron transfer:
    Fe²⁺ + 2e⁻ → Fe(s)    (Tafel slope ≈ 118 mV/dec, reaction order p_OH⁻ = 0)

In reality, the high solvation energy of hexaaqua iron [Fe(H₂O)₆]²⁺ prevents simultaneous
2-electron transfer.  Cathodic iron electrocrystallization proceeds through a
**sequential catalytic mechanism** involving adsorbed hydroxo-intermediates:

1. **Pre-equilibrium hydrolysis** (homogeneous or surface-adjacent):
     Fe²⁺ + H₂O ⇌ FeOH⁺ + H⁺         (log Khyd ≈ −9.5 at 25 °C)

2. **First single-electron transfer** (formation of adsorbed Fe(I) intermediate):
     FeOH⁺ + e⁻ ⇌ (FeOH)ads          (rate constant k_1, transfer coefficient β_1 ≈ 0.5)

3. **Second single-electron transfer / crystallization**:
     (FeOH)ads + e⁻ → Fe(s) + OH⁻    (rate constant k_2, transfer coefficient β_2 ≈ 0.5)
   *or the autocatalytic Epelboin–Wiart step*:
     (FeOH)ads + Fe²⁺ + 2e⁻ → 2 Fe(s) + OH⁻

Kinetic Predictions of the BDD Mechanism
----------------------------------------
* **Dual Tafel slopes**:
  - At low overpotentials (|η| < 80 mV): second step is rate-determining, yielding a
    Tafel slope of **b ≈ 2.303 RT / ((1+β)F) ≈ 40 mV/dec** (at 25 °C).
  - At high overpotentials (|η| > 150 mV): first step becomes rate-determining,
    intermediate coverage θ saturates, and the Tafel slope shifts to **b ≈ 120 mV/dec**.

* **Positive reaction order in hydroxide**:
  - The apparent reaction order is **p_OH⁻ = +1.0** (or p_H⁺ = −1.0).
  - As local HER raises boundary-layer pH from 2.0 to 3.5, [FeOH⁺] increases by 30×,
    **catalyzing the Fe deposition rate** without modifying intrinsic exchange currents.

* **Faradaic admittance and inductive loops in EIS**:
  - Relaxation of the adsorbed intermediate coverage θ(t) creates a phase lag between
    potential and current, manifesting as a **low-frequency inductive loop** (negative
    imaginary impedance) in Nyquist plots.

References
----------
* Bockris, J. O'M., Dražić, D., & Despić, A. R. (1961). "The electrode kinetics
  of the deposition and dissolution of iron." Electrochim. Acta, 4(2-4), 325–361.
* Epelboin, I., & Wiart, R. (1971). "Mechanism of the electrocrystallization of
  nickel and cobalt in acidic solutions." J. Electrochem. Soc., 118(10), 1577.
* Matlosz, M. (1993). "Competitive adsorption in the kinetics of anomalous
  nickel-iron electrodeposition." J. Electrochem. Soc., 140(8), 2272.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Physical constants
FARADAY = 96485.33212      # C/mol
R_GAS = 8.314462618        # J/(mol·K)
T_REF = 298.15             # K

# Thermodynamic constants for ferrous hydrolysis
# Fe²⁺ + H₂O ⇌ FeOH⁺ + H⁺, log K_hyd ≈ -9.5 at 25 °C (Baes & Mesmer 1976)
LOG_K_HYD_25 = -9.50
DH_HYD_KJ_MOL = 55.0       # Endothermic hydrolysis (kJ/mol)


@dataclass(frozen=True)
class BDDKineticParams:
    """Kinetic rate parameters for the Bockris-Dražic-Despic mechanism."""

    k1_forward: float = 5.0e6        # Rate constant for FeOH⁺ + e⁻ → (FeOH)ads (L/(mol·s))
    k1_reverse: float = 20.0         # Rate constant for (FeOH)ads → FeOH⁺ + e⁻ (s⁻¹)
    k2_forward: float = 0.50         # Rate constant for (FeOH)ads + e⁻ → Fe + OH⁻ (s⁻¹)
    beta_1: float = 0.55             # Symmetry factor for step 1
    beta_2: float = 0.45             # Symmetry factor for step 2
    site_density_mol_m2: float = 1.7e-5  # Fe surface site capacity (mol/m²)
    temperature_K: float = 298.15


def hydrolysis_equilibrium_constant(temperature_K: float = 298.15) -> float:
    """van 't Hoff corrected hydrolysis constant K_hyd for Fe²⁺ + H₂O ⇌ FeOH⁺ + H⁺."""
    t_k = max(float(temperature_K), 273.15)
    d_inv_t = (1.0 / T_REF) - (1.0 / t_k)
    ln_k = (LOG_K_HYD_25 * math.log(10.0)) + ((DH_HYD_KJ_MOL * 1e3 / R_GAS) * d_inv_t)
    return math.exp(ln_k)


def feoh_plus_concentration_mol_L(
    fe2_mol_L: float,
    ph: float,
    temperature_K: float = 298.15,
) -> float:
    """Compute equilibrium concentration of electroactive FeOH⁺ (mol/L)."""
    k_hyd = hydrolysis_equilibrium_constant(temperature_K)
    c_h = 10.0 ** (-ph)
    # [FeOH⁺] = K_hyd * [Fe²⁺] / [H⁺]
    return (k_hyd * max(float(fe2_mol_L), 0.0)) / max(c_h, 1e-14)


@dataclass
class BDDStateResult:
    """State of the BDD iron deposition interface at a given potential."""

    overpotential_V: float
    surface_ph: float
    fe2_bulk_mol_L: float
    feoh_plus_mol_L: float
    intermediate_coverage_theta: float  # Fraction of sites occupied by (FeOH)ads
    cathodic_current_density_A_m2: float
    apparent_tafel_slope_mV_dec: float
    reaction_order_oh_minus: float
    inductive_relaxation_time_s: float
    rate_determining_step: str


def solve_bdd_kinetics(
    overpotential_V: float,
    ph: float,
    fe2_mol_L: float = 1.5,
    params: Optional[BDDKineticParams] = None,
    temperature_K: float = 298.15,
) -> BDDStateResult:
    """
    Solve the steady-state BDD multi-step iron deposition kinetics.

    Parameters
    ----------
    overpotential_V : float
        Cathodic overpotential magnitude (positive in cathodic direction, V).
    ph : float
        Electrode surface pH.
    fe2_mol_L : float
        Ferrous iron concentration (mol/L).
    params : BDDKineticParams, optional
        Microkinetic rate parameters.
    temperature_K : float
        Operating temperature (K).

    Returns
    -------
    BDDStateResult
        Comprehensive kinetic state including coverage, partial current, and Tafel slope.
    """
    if params is None:
        params = BDDKineticParams(temperature_K=temperature_K)

    eta = max(float(overpotential_V), 0.0)
    t_k = max(float(temperature_K), 273.15)
    f_rt = FARADAY / (R_GAS * t_k)

    c_feoh = feoh_plus_concentration_mol_L(fe2_mol_L, ph, t_k)  # mol/L
    c_feoh_si = c_feoh * 1e3                                     # mol/m³

    # Potential-dependent rate coefficients (cathodic eta is positive driving force)
    # v1 = k1_f * [FeOH+] * (1 - theta) * exp(beta1 * F * eta / RT)
    # v-1 = k1_r * theta * exp(-(1 - beta1) * F * eta / RT)
    # v2 = k2_f * theta * exp(beta2 * F * eta / RT)

    k1_f_e = params.k1_forward * math.exp(params.beta_1 * f_rt * eta)
    k1_r_e = params.k1_reverse * math.exp(-(1.0 - params.beta_1) * f_rt * eta)
    k2_f_e = params.k2_forward * math.exp(params.beta_2 * f_rt * eta)

    # Steady-state coverage dtheta/dt = 0:
    # dtheta/dt = k1_f_e * c_feoh * (1 - theta) - k1_r_e * theta - k2_f_e * theta = 0
    # theta * (k1_f_e * c_feoh + k1_r_e + k2_f_e) = k1_f_e * c_feoh

    denom = (k1_f_e * c_feoh) + k1_r_e + k2_f_e
    if denom > 1e-15:
        theta = (k1_f_e * c_feoh) / denom
    else:
        theta = 0.0
    theta = min(max(theta, 0.0), 1.0)

    # Total cathodic current density: j = 2 * F * v2 = 2 * F * (N_s * k2_f_e * theta)
    v2 = params.site_density_mol_m2 * k2_f_e * theta
    j_fe = 2.0 * FARADAY * v2  # A/m²

    # Inductive relaxation time tau = 1 / (k1_f_e * c_feoh + k1_r_e + k2_f_e)
    tau_ind = 1.0 / max(denom, 1e-12)

    # Tafel slope calculation via differential step
    d_eta = 0.005  # 5 mV step
    k1_f_e_plus = params.k1_forward * math.exp(params.beta_1 * f_rt * (eta + d_eta))
    k1_r_e_plus = params.k1_reverse * math.exp(-(1.0 - params.beta_1) * f_rt * (eta + d_eta))
    k2_f_e_plus = params.k2_forward * math.exp(params.beta_2 * f_rt * (eta + d_eta))
    denom_plus = (k1_f_e_plus * c_feoh) + k1_r_e_plus + k2_f_e_plus
    theta_plus = (k1_f_e_plus * c_feoh) / max(denom_plus, 1e-15)
    v2_plus = params.site_density_mol_m2 * k2_f_e_plus * theta_plus
    j_fe_plus = 2.0 * FARADAY * v2_plus

    if j_fe > 1e-9 and j_fe_plus > 1e-9:
        d_log_j = math.log10(j_fe_plus) - math.log10(j_fe)
        tafel_slope = (d_eta / max(d_log_j, 1e-6)) * 1e3  # mV/decade
    else:
        tafel_slope = 40.0

    # Numerical reaction order in OH- (or -d log j / d pH)
    d_ph = 0.10
    c_feoh_dph = feoh_plus_concentration_mol_L(fe2_mol_L, ph + d_ph, t_k)
    denom_dph = (k1_f_e * c_feoh_dph) + k1_r_e + k2_f_e
    theta_dph = (k1_f_e * c_feoh_dph) / max(denom_dph, 1e-15)
    j_fe_dph = 2.0 * FARADAY * (params.site_density_mol_m2 * k2_f_e * theta_dph)

    if j_fe > 1e-9 and j_fe_dph > 1e-9:
        p_oh = (math.log10(j_fe_dph) - math.log10(j_fe)) / d_ph
    else:
        p_oh = 1.0

    if theta < 0.20:
        rds = "Step 2 rate-determining (low overpotential, ~40 mV/dec)"
    elif theta > 0.80:
        rds = "Step 1 rate-determining (high overpotential, ~120 mV/dec)"
    else:
        rds = "Mixed intermediate coverage regime"

    return BDDStateResult(
        overpotential_V=eta,
        surface_ph=ph,
        fe2_bulk_mol_L=fe2_mol_L,
        feoh_plus_mol_L=c_feoh,
        intermediate_coverage_theta=theta,
        cathodic_current_density_A_m2=j_fe,
        apparent_tafel_slope_mV_dec=tafel_slope,
        reaction_order_oh_minus=p_oh,
        inductive_relaxation_time_s=tau_ind,
        rate_determining_step=rds,
    )


def main() -> None:
    """CLI entrypoint for BDD multi-step iron deposition kinetics."""
    print("=================================================================")
    print(" Bockris–Dražic–Despic (BDD) Multi-Step Iron Deposition Kinetics")
    print("=================================================================")
    print("Overpotential sweep at pH 2.5, 1.5 M Fe2+:")
    for eta in [0.020, 0.040, 0.080, 0.150, 0.250, 0.350]:
        res = solve_bdd_kinetics(eta, ph=2.5, fe2_mol_L=1.5)
        print(f"  η = {eta*1e3:4.0f} mV | θ_FeOH = {res.intermediate_coverage_theta:4.2f} | j_Fe = {res.cathodic_current_density_A_m2/10.0:6.2f} mA/cm² | Tafel = {res.apparent_tafel_slope_mV_dec:5.1f} mV/dec | {res.rate_determining_step}")


if __name__ == "__main__":
    main()
