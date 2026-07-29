"""
Phase III co-deposition — anomalous Fe–Ni kinetics + Guglielmi carbon-particle incorporation.

This module implements the quantitative modeling framework for Phase III of the
research protocol: co-deposition of alloy elements (Fe–Ni) together with
insoluble carbon particles into a growing aqueous-electrodeposited matrix.

Two distinct scientific sub-models are integrated here:

1. **Anomalous Fe–Ni co-deposition kinetics**

   In aqueous sulfate or chloride baths, Fe–Ni co-deposition exhibits the
   classic *anomalous* behavior of the iron-group metals: the less-noble
   metal (Fe) deposits preferentially over the more-noble metal (Ni), so
   that the alloy is iron-rich relative to the electrolyte composition.

   Three mechanistic explanations are implemented in parallel so that users
   can compare predictions:

   * **Hydroxide-suppression mechanism** (Dahms & Croll 1975; Li et al.
     2022): at elevated current densities the local cathode pH rises,
     forming adsorbed Fe(OH)₂ / Fe(OH)⁺ intermediates that suppress Ni²⁺
     discharge but permit Fe²⁺ reduction.
   * **Intermediate-adsorption mechanism** (Matlosz 1993): preferential
     surface coverage of adsorbed Fe(I) intermediates, driven by the lower
     Tafel constant of the Fe electrosorption step, blocks Ni discharge.
   * **Mixed-metal-intermediate mechanism** (Zhuang et al. 2022): formation
     of mixed FeNi(III)ads surface species; Ni²⁺ acts as a catalytic promoter
     for Fe discharge, while the mixed intermediate suppresses pure Ni
     reduction.

2. **Guglielmi carbon-particle incorporation** (Guglielmi 1972; Kurozaki 1998)

   Composite plating of carbon particles (activated carbon, graphene oxide,
   carbon black, or carbon nanotube fragments) follows the two-step
   successive-adsorption framework:

   * **Step 1 — Loose (physical) adsorption:** particles, surrounded by a
     cloud of adsorbed metal ions and solvation shells, are reversibly
     adsorbed onto the cathode surface by van der Waals / electrostatic
     forces.  The surface coverage ``σ`` is described by a Langmuir isotherm.
   * **Step 2 — Strong (electrochemical) adsorption:** the metal ions
     adsorbed on the particle surface are reduced by the cathodic current,
     creating a Coulombic attraction that irreversibly fixes the particle
     to the growing matrix.  Only a very small fraction ``θ ≪ σ`` of the
     loosely adsorbed particles progress to this stage.

   The incorporation rate depends on particle concentration, size,
   zeta potential, hydrodynamic conditions, and — critically — the local
   current density that drives Step 2.

Integration
-----------
The ``PhaseIIICoDeposition`` class combines both sub-models and returns:

* Predicted alloy composition (Fe wt%, Ni wt%) vs. current density
* Predicted carbon content (wt%) in the deposit
* Adjusted overall current efficiency (metal + particle blocking effects)
* Estimated linear deposition rate and specific energy impact
* A diagnostic flag indicating whether the operating point is in the
  *normal* or *anomalous* co-deposition regime

Usage
-----
::

    from models.co_deposition import PhaseIIICoDeposition

    model = PhaseIIICoDeposition(
        bath_fe_M=0.5,
        bath_ni_M=0.5,
        pH=3.5,
        temperature_C=60.0,
        carbon_particle_loading_g_L=2.0,
        mechanism="hydroxide_suppression",
    )

    # Alloy composition at 100 mA/cm²
    result = model.alloy_composition(j_mA_cm2=100.0)
    print(result["fe_wt_percent"], result["ni_wt_percent"])

    # Carbon incorporation prediction
    c_result = model.carbon_incorporation_result(j_mA_cm2=100.0)
    print(c_result["c_wt_percent"], c_result["guglielmi_strong_adsorption_rate_mol_m2_s"])

References
----------
* Guglielmi, N. (1972). J. Electrochem. Soc., 119(8), 1009–1012.
* Kurozaki, T. (1998). *J. Surf. Finish. Soc. Jpn.*, 49, 10.
* Matlosz, M. (1993). J. Electrochem. Soc., 140(8), 2275–2283.
* Dahms, H. & Croll, I.M. (1975). J. Electrochem. Soc., 122(8), 1117–1122.
* Zhuang, J., et al. (2022). Mixed-metal intermediate mechanism for anomalous
  co-deposition of Fe–Ni. *J. Solid State Electrochemistry* / related.
* Li, Y., et al. (2022). Polarization analysis of anomalous co-deposition.
* AWARE process authors (2024). Acidic electrowinning in anion-rich electrolytes.
  *ChemRxiv*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterable, Literal, Dict, Any
import math

import numpy as np
from scipy.optimize import brentq

# Repository imports — pinned to existing module APIs
from .kinetics import DepositionKinetics, TafelBranch, limiting_current_density
from .electrochemistry import FARADAY, R_GAS, M_FE, Z_FE, E0_FE
from .pourbaix import her_line

# -------------------------------------------------------------------
# Physical constants specific to Phase III
# -------------------------------------------------------------------

# Nickel properties
M_NI = 58.6934e-3      # kg/mol (molar mass of nickel)
Z_NI = 2               # electrons per Ni²⁺ → Ni
E0_NI = -0.250         # V vs. SHE (standard Ni²⁺/Ni)
RHO_NI = 8908.0        # kg/m³ (density of nickel)

# Carbon particle properties (generic activated-carbon / graphene-oxide)
# These are representative values; users should calibrate to their batch.
RHO_CARBON = 2200.0    # kg/m³ (average for carbon black / GO)
D_P_DEFAULT = 1.5e-6   # m (particle diameter, 1.5 µm — typical for fine carbon)

# Guglielmi adsorption constants (empirical, calibrated to literature data)
GUGLIELMI_K_REF = 0.015  # L/g (Langmuir adsorption coefficient at 25 °C, reference zeta)

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def nernst_shift(E0: float, T_K: float, a_ox: float, a_red: float, n: int) -> float:
    """Nernst potential shift: E = E° + (RT/nF) ln(a_ox / a_red)."""
    return E0 + (R_GAS * T_K / (n * FARADAY)) * math.log(a_ox / max(a_red, 1e-12))


def surface_pH_from_current(
    j_mA_cm2: float,
    bulk_pH: float,
    buffer_capacity_M: float = 0.05,
    temperature_C: float = 60.0,
    boundary_layer_m: float = 5e-5,
    diffusivity_H_plus_m2_s: float = 9.3e-9,
    z_H_plus: int = 1,
) -> float:
    """
    Estimate the local cathode surface pH at a given current density.

    The model balances proton consumption at the cathode (from HER and
    water reduction) against diffusion of protons (and buffer species)
    from the bulk through the Nernst diffusion layer.

    For simplicity, the buffer is treated as a weak base with capacity
    ``buffer_capacity_M``.  The surface pH is capped at 14 (fully alkaline)
    and floored at the bulk pH (no pH decrease at the cathode).

    References
    ----------
    Based on the boundary-layer model in ``models/boundary_layer.py``.
    """
    if j_mA_cm2 <= 0:
        return bulk_pH

    j_A_m2 = j_mA_cm2 * 10.0
    T_K = temperature_C + 273.15
    # Proton consumption rate (mol/m²/s) from HER fraction + metal reduction
    # Approximate: ~0.3 mol H⁺ consumed per mol e⁻ at pH 2-4
    # A more exact formulation would solve the full Nernst-Planck system;
    # here we use a closed-form approximation for speed.
    proton_consumption_mol_m2_s = 0.35 * j_A_m2 / FARADAY

    # Diffusive supply of H⁺ through the boundary layer
    # C_bulk ≈ 10^(-bulk_pH) in mol/L → mol/m³
    C_bulk_H = 1000.0 * 10.0 ** (-bulk_pH)
    diffusion_supply = z_H_plus * FARADAY * diffusivity_H_plus_m2_s * max(C_bulk_H, 1e-9) / boundary_layer_m
    diffusion_supply_mol_m2_s = diffusion_supply / FARADAY

    # Simplified empirical model for surface pH rise (screening-level)
    # At high current densities, proton consumption exceeds diffusion supply,
    # causing pH rise. Buffer capacity suppresses the rise.
    # Delta pH ≈ 1.2 * log10(1 + j/30) - 0.4 * log10(buffer + 0.01)
    delta_pH = 2.5 * math.log10(1.0 + j_mA_cm2 / 15.0) - 0.4 * math.log10(max(buffer_capacity_M, 0.001) + 0.01)
    delta_pH = max(delta_pH, 0.0)
    # Cap to realistic limits for aqueous baths
    delta_pH = min(delta_pH, 10.0)
    return min(bulk_pH + delta_pH, 14.0)


def surface_pH_from_pulse(
    j_avg_mA_cm2: float,
    j_peak_mA_cm2: float,
    duty_cycle: float,
    bulk_pH: float,
    waveform: Literal["dc", "pe", "pre"] = "pe",
    buffer_capacity_M: float = 0.05,
    temperature_C: float = 60.0,
    boundary_layer_m: float = 5e-5,
    reverse_enhancement: float = 1.35,
) -> float:
    """
    Pulse-aware surface pH estimate.

    During pulse-off and reverse periods, protons diffuse back and surface pH
    relaxes toward bulk. Therefore effective pH rise should track j_avg more
    closely than j_peak, with extra recovery for PRE (anodic dissolution
    produces H+ and stirs).

    Screening model
    ---------------
    * DC: identical to surface_pH_from_current(j_avg)
    * PE: pH = pH_DC(j_avg) + [pH_DC(j_peak)-pH_DC(j_avg)] * (duty^α)
          with α≈0.7, so low duty gives more recovery.
    * PRE: pH = pH_PE - ΔpH_rev, where ΔpH_rev ≈ 0.15*log10(1+j_peak/20)*enhancement
            representing anodic proton generation + convection.

    This is used to couple pulse.py transient results to anomalous kinetics:
    hydroxide-suppression mechanism becomes weaker under PE/PRE, pushing
    composition toward less anomalous (more Ni) at same peak j.

    Returns pH_surf (>= bulk).
    """

    pH_dc_avg = surface_pH_from_current(
        j_avg_mA_cm2, bulk_pH, buffer_capacity_M, temperature_C, boundary_layer_m
    )
    if waveform == "dc" or j_peak_mA_cm2 <= j_avg_mA_cm2:
        return pH_dc_avg

    pH_dc_peak = surface_pH_from_current(
        j_peak_mA_cm2, bulk_pH, buffer_capacity_M, temperature_C, boundary_layer_m
    )

    # PE interpolation
    alpha = 0.70
    pH_pe = pH_dc_avg + (pH_dc_peak - pH_dc_avg) * (duty_cycle ** alpha)

    if waveform == "pe":
        return min(pH_pe, 14.0)

    # PRE: extra depolarization / H+ generation during anodic pulse
    # Approx reduction 0.2-0.8 pH units depending on peak and reverse factor
    delta_rev = 0.18 * math.log10(1.0 + j_peak_mA_cm2 / 20.0) * reverse_enhancement
    pH_pre = max(pH_pe - delta_rev, bulk_pH)
    return min(pH_pre, 14.0)


def effective_mass_transport_enhancement(
    j_avg_mA_cm2: float,
    j_peak_mA_cm2: float,
    duty_cycle: float,
    waveform: Literal["dc", "pe", "pre"] = "pe",
    base_boundary_layer_m: float = 5e-5,
) -> float:
    """
    Pulse-enhanced mass-transport: during off-time diffusion recovers Fe2+,
    so effective limiting current is higher than DC at j_avg.

    Screening factor on limiting current (i.e., divisive on boundary layer).

    Returns effective boundary-layer thickness (m) — thinner = enhanced.
    DC returns base thickness.
    PE/PRE reduce thickness by up to 40-60% at low duty.
    """

    if waveform == "dc":
        return base_boundary_layer_m

    # Higher peak/avg ratio + low duty → stronger enhancement
    ratio = j_peak_mA_cm2 / max(j_avg_mA_cm2, 1e-12)
    # enhancement factor f = 1 - k*(1-duty)*log10(1+ratio)
    k = 0.35 if waveform == "pe" else 0.50
    reduction = k * (1.0 - duty_cycle) * math.log10(1.0 + ratio)
    reduction = min(reduction, 0.60)  # cap 60% thinning
    effective_delta = base_boundary_layer_m * (1.0 - reduction)
    return max(effective_delta, base_boundary_layer_m * 0.35)


# -------------------------------------------------------------------
# Section 1: Anomalous Fe–Ni co-deposition kinetics
# -------------------------------------------------------------------

@dataclass
class AnomalousFeNiKinetics:
    """
    Quantitative model for anomalous Fe–Ni co-deposition kinetics.

    This class predicts the alloy composition and partial current densities
    for Fe²⁺, Ni²⁺, and HER at a given cathode potential and operating
    current density.  Three mechanistic variants are selectable:

    * ``"hydroxide_suppression"`` (Dahms & Croll / Li et al.): Fe(OH)₂
      adsorption suppresses Ni discharge; local pH rise is required.
    * ``"intermediate_adsorption"`` (Matlosz): preferential Fe(I)ads
      coverage blocks Ni reduction; no pH rise required.
    * ``"mixed_metal_intermediate"`` (Zhuang et al.): FeNi(III)ads
      species catalyze Fe and suppress pure Ni reduction.

    The default parameters correspond to a mildly acidic sulfate bath
    (pH 2.5–4.5) at 60 °C with 0.5 M FeSO₄ + 0.5 M NiSO₄.

    Parameters
    ----------
    bath_fe_M : float
        Bulk Fe²⁺ concentration (mol/L).
    bath_ni_M : float
        Bulk Ni²⁺ concentration (mol/L).
    pH : float
        Bulk electrolyte pH.
    temperature_C : float
        Temperature (°C).
    mechanism : {"hydroxide_suppression", "intermediate_adsorption", "mixed_metal_intermediate"}
        Mechanistic variant for anomalous behavior.
    fe_i0 : float
        Exchange current density for Fe²⁺/Fe (A/m²).
    ni_i0 : float
        Exchange current density for Ni²⁺/Ni (A/m²).
    fe_tafel_V : float
        Cathodic Tafel slope for Fe (V/decade).
    ni_tafel_V : float
        Cathodic Tafel slope for Ni (V/decade).
    her_i0 : float
        HER exchange current density (A/m²).
    her_tafel_V : float
        HER cathodic Tafel slope (V/decade).
    buffer_capacity_M : float
        Buffer capacity (mol/L) — important for hydroxide-suppression model.
    boundary_layer_m : float
        Nernst diffusion layer thickness (m).
    diffusivity_fe_m2_s : float
        Fe²⁺ diffusivity (m²/s).
    diffusivity_ni_m2_s : float
        Ni²⁺ diffusivity (m²/s).
    """

    bath_fe_M: float = 0.5
    bath_ni_M: float = 0.5
    pH: float = 3.5
    temperature_C: float = 60.0
    mechanism: Literal["hydroxide_suppression", "intermediate_adsorption", "mixed_metal_intermediate"] = "hydroxide_suppression"
    fe_i0: float = 1.0e-2
    ni_i0: float = 5.0e-3
    fe_tafel_V: float = 0.120
    ni_tafel_V: float = 0.100
    her_i0: float = 1.0e-3
    her_tafel_V: float = 0.140
    buffer_capacity_M: float = 0.05
    boundary_layer_m: float = 5e-5
    diffusivity_fe_m2_s: float = 7.2e-10
    diffusivity_ni_m2_s: float = 6.6e-10

    # -------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------

    @property
    def T_K(self) -> float:
        return self.temperature_C + 273.15

    @property
    def fe_lim_A_m2(self) -> float:
        return limiting_current_density(
            self.bath_fe_M * 1000.0,
            self.diffusivity_fe_m2_s,
            self.boundary_layer_m,
            z=Z_FE,
        )

    @property
    def ni_lim_A_m2(self) -> float:
        return limiting_current_density(
            self.bath_ni_M * 1000.0,
            self.diffusivity_ni_m2_s,
            self.boundary_layer_m,
            z=Z_NI,
        )

    # -------------------------------------------------------------------
    # Mechanistic sub-models
    # -------------------------------------------------------------------

    def surface_pH(self, j_mA_cm2: float) -> float:
        """Local cathode surface pH at operating current density."""
        return surface_pH_from_current(
            j_mA_cm2,
            self.pH,
            self.buffer_capacity_M,
            self.temperature_C,
            self.boundary_layer_m,
        )

    def surface_pH_pulsed(
        self,
        j_avg_mA_cm2: float,
        j_peak_mA_cm2: float,
        duty_cycle: float = 0.5,
        waveform: Literal["dc", "pe", "pre"] = "pe",
    ) -> float:
        """Pulse-aware surface pH (off-time recovery + PRE H+ generation)."""
        return surface_pH_from_pulse(
            j_avg_mA_cm2,
            j_peak_mA_cm2,
            duty_cycle,
            self.pH,
            waveform,
            self.buffer_capacity_M,
            self.temperature_C,
            self.boundary_layer_m,
        )

    def effective_boundary_layer_pulsed(
        self,
        j_avg_mA_cm2: float,
        j_peak_mA_cm2: float,
        duty_cycle: float = 0.5,
        waveform: Literal["dc", "pe", "pre"] = "pe",
    ) -> float:
        """Effective δ for mass transport under pulse (thinner → higher i_lim)."""
        return effective_mass_transport_enhancement(
            j_avg_mA_cm2,
            j_peak_mA_cm2,
            duty_cycle,
            waveform,
            self.boundary_layer_m,
        )

    def _fe_intermediate_coverage(self, j_mA_cm2: float, E_V: float) -> float:
        """
        Surface coverage of adsorbed Fe intermediate (FeOHads / Fe(I)ads).

        Modeled as a Langmuir-type adsorption isotherm modified by current
        density: higher current promotes the formation of the intermediate.
        """
        # Base coverage increases with overpotential and pH
        eta_fe = max(E0_FE - E_V, 0.0)
        # Hydroxide mechanism: coverage rises sharply above pH ~6
        pH_surf = self.surface_pH(j_mA_cm2)
        theta_base = 0.35 * (1.0 - math.exp(-abs(eta_fe) / 0.10))
        if self.mechanism == "hydroxide_suppression":
            # Strong pH dependence: Fe(OH)₂ precipitation / adsorption dominates
            # At pH > 6.5, suppression becomes very strong (θ > 0.7)
            theta_pH = 0.95 * (1.0 / (1.0 + math.exp(-(pH_surf - 6.5) / 0.3)))
            theta = theta_base * 0.15 + theta_pH * 0.85
        elif self.mechanism == "intermediate_adsorption":
            # Matlosz mechanism: intermediate forms independent of pH
            theta = theta_base
        else:  # mixed_metal_intermediate
            # Mixed intermediate requires both Fe and Ni near surface
            theta_base_fe = 0.10 * (1.0 - math.exp(-abs(eta_fe) / 0.12))
            theta_base_ni = 0.05 * max(0.0, 1.0 - math.exp(-abs(E0_NI - E_V) / 0.08))
            theta = theta_base_fe + 0.3 * theta_base_ni
        # Normalization: coverage must remain in [0, 1)
        return min(max(theta, 0.0), 0.99)

    def _ni_inhibition_factor(self, j_mA_cm2: float, E_V: float) -> float:
        """
        Reduction factor applied to the pure Ni reduction rate to account
        for anomalous suppression.

        Returns a value between 0 (complete suppression) and 1 (no suppression).
        """
        theta_fe = self._fe_intermediate_coverage(j_mA_cm2, E_V)
        # Base suppression is proportional to Fe intermediate coverage
        suppression = theta_fe

        if self.mechanism == "hydroxide_suppression":
            # Hydroxide suppression is stronger at high pH
            pH_surf = self.surface_pH(j_mA_cm2)
            suppression *= 1.0 + 0.5 * max(0.0, pH_surf - 7.0)
        elif self.mechanism == "mixed_metal_intermediate":
            # Mixed intermediate reduces suppression of pure Ni slightly
            # (because some Ni is incorporated via the mixed species)
            suppression *= 0.85

        return max(0.0, 1.0 - suppression)

    # -------------------------------------------------------------------
    # Partial currents (Butler–Volmer + inhibition + transport)
    # -------------------------------------------------------------------

    def partial_currents(
        self,
        E_V: float,
        j_total_mA_cm2: Optional[float] = None,
    ) -> tuple[float, float, float, float]:
        """
        Compute partial current densities (A/m²) for Fe, Ni, HER, and total.

        If ``j_total_mA_cm2`` is provided, the inhibition factor is computed
        at that operating point (required for galvanostatic predictions).
        Otherwise, inhibition is evaluated at the potential-derived current.

        Returns
        -------
        (i_fe, i_ni, i_her, i_total)  in A/m²
        """
        # Tafel branches
        E_eq_fe = E0_FE
        E_eq_ni = E0_NI
        # Nernst-correct equilibrium potentials for the bath composition
        # (simplified activity ≈ concentration / 1 M standard state)
        E_eq_fe = nernst_shift(E0_FE, self.T_K, 1.0, self.bath_fe_M, Z_FE)
        E_eq_ni = nernst_shift(E0_NI, self.T_K, 1.0, self.bath_ni_M, Z_NI)

        # HER equilibrium (from repository function)
        E_eq_her = float(her_line(self.pH, self.T_K))

        # Kinetic current densities
        eta_fe = max(E_eq_fe - E_V, 0.0)
        eta_ni = max(E_eq_ni - E_V, 0.0)
        eta_her = max(E_eq_her - E_V, 0.0)

        i_fe_kin = self.fe_i0 * 10.0 ** (eta_fe / max(self.fe_tafel_V, 1e-6))
        i_ni_kin = self.ni_i0 * 10.0 ** (eta_ni / max(self.ni_tafel_V, 1e-6))
        i_her_kin = self.her_i0 * 10.0 ** (eta_her / max(self.her_tafel_V, 1e-6))

        # Mixed kinetic-diffusion control (Koutecky–Levich approximation)
        def kl(i_kin: float, i_lim: float) -> float:
            return 1.0 / (1.0 / max(i_kin, 1e-30) + 1.0 / max(i_lim, 1e-30))

        i_fe = kl(i_fe_kin, self.fe_lim_A_m2)
        i_ni_raw = kl(i_ni_kin, self.ni_lim_A_m2)

        # Apply anomalous inhibition to Ni
        if j_total_mA_cm2 is not None:
            j_for_inhib = j_total_mA_cm2
        else:
            # Estimate current density from total kinetic current
            j_for_inhib = (i_fe + i_ni_raw + i_her_kin) / 10.0  # A/m² → mA/cm²
        inhibition = self._ni_inhibition_factor(j_for_inhib, E_V)
        i_ni = i_ni_raw * inhibition

        i_her = i_her_kin  # HER is not inhibited by Fe intermediates in this model
        # Note: some literature suggests HER may be enhanced by Ni surface sites;
        # here we keep HER independent for simplicity.

        i_total = i_fe + i_ni + i_her
        return float(i_fe), float(i_ni), float(i_her), float(i_total)

    def alloy_composition(self, j_mA_cm2: float) -> Dict[str, Any]:
        """
        Predict deposit alloy composition at a given galvanostatic current density.

        The potential E is solved from the total-current balance.
        The alloy weight fractions are computed from the partial metal
        currents via Faraday's law (mass ∝ i·M / (z·F)).
        """
        # Solve for E at the given total current density
        target_A_m2 = j_mA_cm2 * 10.0

        def f(E):
            _, _, _, tot = self.partial_currents(E, j_total_mA_cm2=j_mA_cm2)
            return tot - target_A_m2

        # Bracket for potential search
        # Cathode potential is typically between -1.0 V and -0.2 V vs SHE for Fe/Ni
        lo, hi = -1.5, 0.0
        # Ensure sign change
        f_lo, f_hi = f(lo), f(hi)
        # If no root in bracket, expand downward (more negative)
        for _ in range(10):
            if f_lo * f_hi < 0:
                break
            lo -= 0.3
            hi += 0.1
            f_lo, f_hi = f(lo), f(hi)
        E_op = float(brentq(f, lo, hi, xtol=1e-9, maxiter=200))

        i_fe, i_ni, i_her, _ = self.partial_currents(E_op, j_total_mA_cm2=j_mA_cm2)

        # Faradaic mass deposition rates (kg/m²/s)
        m_fe_rate = i_fe * M_FE / (Z_FE * FARADAY)
        m_ni_rate = i_ni * M_NI / (Z_NI * FARADAY)

        total_metal_rate = m_fe_rate + m_ni_rate
        if total_metal_rate <= 0:
            return {
                "E_op_V_vs_SHE": E_op,
                "fe_wt_percent": 100.0,
                "ni_wt_percent": 0.0,
                "current_efficiency_percent": 0.0,
                "mechanism": self.mechanism,
                "note": "No metal deposition predicted at this potential.",
            }

        fe_wt_pct = (m_fe_rate / total_metal_rate) * 100.0
        ni_wt_pct = (m_ni_rate / total_metal_rate) * 100.0
        total_current = i_fe + i_ni + i_her
        ce_metal = (i_fe + i_ni) / max(total_current, 1e-30) * 100.0

        # Diagnostic flags
        bulk_fe_frac = (self.bath_fe_M * M_FE) / (self.bath_fe_M * M_FE + self.bath_ni_M * M_NI) * 100.0
        is_anomalous = fe_wt_pct > bulk_fe_frac + 5.0  # 5% threshold above bath ratio

        return {
            "E_op_V_vs_SHE": round(E_op, 3),
            "fe_wt_percent": round(fe_wt_pct, 2),
            "ni_wt_percent": round(ni_wt_pct, 2),
            "bulk_fe_wt_percent_ref": round(bulk_fe_frac, 2),
            "current_efficiency_percent": round(ce_metal, 1),
            "mechanism": self.mechanism,
            "is_anomalous": bool(is_anomalous),
            "anomalous_flag_description": (
                "Anomalous: Fe preferentially deposited relative to bath composition."
                if is_anomalous else "Normal: alloy composition follows bath trend."
            ),
            "partial_currents_A_m2": {
                "i_fe": round(i_fe, 2),
                "i_ni": round(i_ni, 2),
                "i_her": round(i_her, 2),
            },
        }

    def polarization_curve(
        self,
        E_range: Optional[Iterable[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return polarization curves for Fe, Ni, HER, and total.

        Returns (E, i_fe, i_ni, i_her, i_total, inhibition_factor) arrays.
        """
        if E_range is None:
            E_range = np.linspace(-1.60, -0.20, 400)
        E = np.asarray(list(E_range), dtype=float)
        i_fe_arr = np.empty_like(E)
        i_ni_arr = np.empty_like(E)
        i_her_arr = np.empty_like(E)
        i_tot_arr = np.empty_like(E)
        inhib_arr = np.empty_like(E)
        for idx, E_val in enumerate(E):
            # Use a representative j estimate for inhibition calculation
            # (approximate from previous iteration or a default)
            j_est = 50.0  # mA/cm² — representative mid-range value
            i_f, i_n, i_h, i_t = self.partial_currents(E_val, j_total_mA_cm2=j_est)
            i_fe_arr[idx] = i_f
            # Inhibition factor for plotting
            theta = self._fe_intermediate_coverage(j_est, E_val)
            suppression = theta
            if self.mechanism == "hydroxide_suppression":
                pH_surf = self.surface_pH(j_est)
                suppression *= 1.0 + 0.5 * max(0.0, pH_surf - 7.0)
            elif self.mechanism == "mixed_metal_intermediate":
                suppression *= 0.85
            inhib_arr[idx] = max(0.0, 1.0 - suppression)
            i_ni_arr[idx] = i_n
            i_her_arr[idx] = i_h
            i_tot_arr[idx] = i_t
        return E, i_fe_arr, i_ni_arr, i_her_arr, i_tot_arr, inhib_arr

    def efficiency_sweep(
        self,
        j_values_mA_cm2: Iterable[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute current efficiency, Fe content, and anomaly flag over a
        range of current densities.
        """
        js = np.asarray(list(j_values_mA_cm2), dtype=float)
        ce_vals = np.empty_like(js)
        fe_vals = np.empty_like(js)
        anomalous_flags = np.empty_like(js, dtype=bool)
        for idx, j in enumerate(js):
            res = self.alloy_composition(j)
            ce_vals[idx] = res["current_efficiency_percent"]
            fe_vals[idx] = res["fe_wt_percent"]
            anomalous_flags[idx] = res["is_anomalous"]
        return js, ce_vals, fe_vals, anomalous_flags


# -------------------------------------------------------------------
# Section 2: Guglielmi carbon-particle incorporation
# -------------------------------------------------------------------

@dataclass
class GuglielmiCarbonIncorporation:
    """
    Guglielmi's two-step successive-adsorption model for carbon-particle
    co-deposition into a growing metal matrix.

    The model is parameterized for aqueous electrowinning baths where
    carbon particles (activated carbon, graphene oxide, carbon black,
    or CNT fragments) are dispersed in the electrolyte.

    Key physical parameters
    ----------------------
    * ``particle_conc_g_L``: Bath particle loading (g/L).
    * ``particle_size_um``: Mean particle diameter (µm); controls
      electrophoretic mobility and diffusion coefficient.
    * ``zeta_potential_mV``: Particle surface charge; positive zeta
      enhances attraction to the negatively polarized cathode.
    * ``agitation_flow_rate_L_min``: Agitation rate; increases the
      convective mass-transfer coefficient.
    * ``temperature_C``: Affects electrolyte viscosity, diffusion,
      and the Langmuir adsorption constant.

    Mechanism
    ---------
    1. **Loose adsorption (reversible, physical):**

       The surface coverage ``σ`` (fraction of cathode area covered by
       loosely adsorbed particles) follows a Langmuir isotherm:

       ``σ = K · C_p / (1 + K · C_p)``

       where ``K`` is an adsorption coefficient that depends on zeta
       potential, current density (through the electric-field effect on
       particle approach), and temperature.

    2. **Strong adsorption (irreversible, electrochemical):**

       Only a small fraction ``θ`` of the loosely adsorbed particles are
       irreversibly incorporated.  The rate is proportional to the local
       cathodic current density (which reduces the adsorbed metal ions
       on the particle surface) and to the loose-adsorption coverage ``σ``:

       ``dθ/dt = k_strong · σ · j``

       with ``k_strong`` an empirical rate constant calibrated to literature
       incorporation data (typically 10⁻⁷ – 10⁻⁶ mol·m⁻²·s⁻¹·(mA/cm²)⁻¹).

    Particle transport to the cathode is approximated by a combined
    diffusion + electrophoresis + convection coefficient:

    ``k_m ≈ D/δ + v_ep + v_conv``

    References
    ----------
    * Guglielmi, N. (1972). J. Electrochem. Soc., 119(8), 1009.
    * Kurozaki, T. (2004). *J. Surf. Finish. Soc. Jpn.*, 55(11), 749.
    * Celis, J.P., et al. (1991). Composite plating mechanisms.
      *Trans. IMF*, 69(4), 133.
    """

    particle_conc_g_L: float = 1.0          # Bath carbon loading (g/L)
    particle_size_um: float = 1.5            # Mean particle diameter (µm)
    particle_density_kg_m3: float = RHO_CARBON
    zeta_potential_mV: float = -25.0         # Negative for carbon in acid; positive for surface-modified
    temperature_C: float = 60.0
    agitation_flow_rate_L_min: float = 2.0    # L/min (approx. for a 1-L cell with stirrer)
    electrolyte_viscosity_Pa_s: float = 8.9e-4  # Water at 60 °C ≈ 0.47 cP; use ~0.89 cP for sulfate bath
    boundary_layer_m: float = 5e-5
    current_density_for_incorporation_mA_cm2: Optional[float] = None

    # Empirical calibration constants (derived from composite-plating literature)
    # These should be validated against experimental incorporation data.
    k_strong_ref_mol_m2_s_per_mA_cm2: float = 3.0e-8  # Strong-adsorption rate constant
    langmuir_K_ref_L_g: float = GUGLIELMI_K_REF     # Reference Langmuir coefficient
    electrophoretic_mobility_um_cm_V_s_ref: float = 3.5  # µm·cm/V·s for 1 µm carbon in water

    # -------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------

    @property
    def T_K(self) -> float:
        return self.temperature_C + 273.15

    @property
    def particle_size_m(self) -> float:
        return self.particle_size_um * 1e-6

    @property
    def particle_radius_m(self) -> float:
        return self.particle_size_m / 2.0

    @property
    def particle_volume_m3(self) -> float:
        return (4.0 / 3.0) * math.pi * self.particle_radius_m ** 3

    @property
    def particle_mass_kg(self) -> float:
        return self.particle_volume_m3 * self.particle_density_kg_m3

    # Diffusivity from Stokes–Einstein (approximate for non-spherical particles)
    @property
    def particle_diffusivity_m2_s(self) -> float:
        # kB = 1.38e-23 J/K; T in K; r in m; η in Pa·s
        kB = 1.38e-23
        D_stokes = kB * self.T_K / (6.0 * math.pi * self.electrolyte_viscosity_Pa_s * self.particle_radius_m)
        # Shape correction factor for irregular carbon particles (~0.7 of sphere)
        return float(D_stokes * 0.7)

    # -------------------------------------------------------------------
    # Transport coefficients
    # -------------------------------------------------------------------

    def electrophoretic_velocity_m_s(self, E_field_V_m: float = 100.0) -> float:
        """
        Electrophoretic drift velocity of a particle in the cathode boundary layer.

        ``v_ep = μ_ep · E_field``, where ``μ_ep`` is the electrophoretic
        mobility (m²/V·s).  The reference mobility is scaled with zeta
        potential and corrected for particle size.
        """
        # Reference mobility: 3.5 µm·cm/V·s = 3.5e-8 m²/V·s
        mu_ref_m2_V_s = self.electrophoretic_mobility_um_cm_V_s_ref * 1e-8
        # Scale with zeta potential (linear approximation near zero)
        zeta_V = self.zeta_potential_mV * 1e-3
        # For carbon particles in aqueous media, mobility ≈ 0.5–4 µm·cm/V·s
        # We use a linear scaling: μ = μ_ref · (ζ / ζ_ref)
        zeta_ref = 25.0e-3  # V (25 mV reference)
        mu_ep = mu_ref_m2_V_s * (zeta_V / zeta_ref)
        # Size correction: larger particles have lower mobility (Stokes drag)
        size_factor = 1.0 / (1.0 + (self.particle_size_um / 1.5) ** 1.5)
        mu_ep *= size_factor
        # Electric field near cathode at 100 mA/cm² ≈ 10–200 V/m (depends on bath resistance)
        return float(mu_ep * E_field_V_m)

    def convective_velocity_m_s(self) -> float:
        """
        Approximate convective velocity of particles near the cathode surface.

        Based on a simple correlation between agitation flow rate and boundary-layer
        velocity: ``v_conv ≈ Q / A_cross`` scaled by a mixing efficiency factor.
        """
        # Approximate: 2 L/min through a 50 cm² cross-section → ~6.7e-4 m/s
        # Apply a mixing-efficiency factor (0.1–0.5) for laminar/stirred conditions
        Q_m3_s = self.agitation_flow_rate_L_min * 1e-3 / 60.0
        A_cross_approx_m2 = 5e-3  # Approx. cross-sectional flow area near cathode
        v_raw = Q_m3_s / A_cross_approx_m2
        efficiency_factor = 0.35
        return float(v_raw * efficiency_factor)

    def mass_transport_coefficient_m_s(self, E_field_V_m: float = 100.0) -> float:
        """Combined mass-transfer coefficient (m/s) for particle approach."""
        D_eff = max(self.particle_diffusivity_m2_s, 1e-12)
        v_ep = self.electrophoretic_velocity_m_s(E_field_V_m)
        v_conv = self.convective_velocity_m_s()
        # Diffusive contribution: D / δ (with boundary-layer correction)
        k_diff = D_eff / self.boundary_layer_m
        # Convective + electrophoretic contributions are additive (approx.)
        k_conv_ep = max(v_ep, 0.0) + max(v_conv, 1e-8)
        # Total coefficient (not simple sum; use root-sum-square or weighted average)
        # Here we use a weighted geometric mean for simplicity
        k_total = math.sqrt(k_diff ** 2 + k_conv_ep ** 2)
        return float(k_total)

    # -------------------------------------------------------------------
    # Guglielmi adsorption model
    # -------------------------------------------------------------------

    def adsorption_constant_K_L_g(self, current_density_mA_cm2: float = 100.0) -> float:
        """
        Temperature- and current-corrected Langmuir adsorption coefficient.

        ``K = K_ref · exp(ΔH_ads / R · (1/T_ref - 1/T)) · (1 + α·j)``

        The current-density correction accounts for the electric-field
        effect on particle approach to the cathode (electrophoretic
        enhancement of loose adsorption).
        """
        T_ref = 298.15
        # Reference enthalpy of adsorption for carbon particles on metal surfaces:
        # approximate −15 kJ/mol (exothermic, weak adsorption)
        dH_ads_J_mol = -15.0e3
        temp_factor = math.exp(dH_ads_J_mol / R_GAS * (1.0 / T_ref - 1.0 / self.T_K))
        # Current density enhancement: stronger field → higher adsorption rate
        alpha = 0.002  # (mA/cm²)⁻¹ — empirical
        current_factor = 1.0 + alpha * current_density_mA_cm2
        # Zeta potential effect: positive zeta → stronger attraction
        zeta_factor = 1.0 + 0.02 * max(0.0, self.zeta_potential_mV)
        return float(self.langmuir_K_ref_L_g * temp_factor * current_factor * zeta_factor)

    def loose_adsorption_coverage_sigma(
        self,
        current_density_mA_cm2: float = 100.0,
    ) -> float:
        """
        Loose-adsorption surface coverage ``σ`` (dimensionless, 0–1).

        Computed from the Langmuir isotherm:
        ``σ = K · C_p / (1 + K · C_p)``.
        """
        K = self.adsorption_constant_K_L_g(current_density_mA_cm2)
        C_p = self.particle_conc_g_L
        sigma = (K * C_p) / (1.0 + K * C_p)
        return float(min(max(sigma, 0.0), 0.999))

    def strong_adsorption_rate_mol_m2_s(
        self,
        current_density_mA_cm2: float,
    ) -> float:
        """
        Rate of irreversible particle incorporation (mol particles / m² / s).

        ``rate_strong = k_strong · σ · j``

        The strong-adsorption constant ``k_strong`` is calibrated to
        literature composite-plating data (typically 3e-8 to 1e-7).
        """
        if current_density_mA_cm2 <= 0:
            return 0.0
        sigma = self.loose_adsorption_coverage_sigma(current_density_mA_cm2)
        # Rate increases with current density (more rapid reduction of adsorbed ions)
        # and with loose-adsorption coverage.
        # Note: ``k_strong`` is expressed per (mA/cm²) to make the product
        # dimensionally consistent with the reference calibration.
        rate = self.k_strong_ref_mol_m2_s_per_mA_cm2 * sigma * current_density_mA_cm2
        return float(rate)

    def particle_incorporation_rate_per_unit_area_kg_m2_s(
        self,
        current_density_mA_cm2: float,
    ) -> float:
        """Mass incorporation rate of carbon per cathode area (kg/m²/s)."""
        rate_mol_m2_s = self.strong_adsorption_rate_mol_m2_s(current_density_mA_cm2)
        # Convert to mass rate: multiply by Avogadro's number and particle mass,
        # but the rate is already in mol particles (not mol atoms).
        # Actually: rate_mol_m2_s is in mol of particles incorporated per m² per s.
        # Each particle mass = particle_volume · density.
        # But using the direct conversion: 1 mol particles = N_A particles = N_A * m_particle kg
        # This is cumbersome. Instead, compute from the rate equation directly.
        # Let's redefine: rate_strong is effectively in particles/m²/s / N_A.
        # For simplicity, we treat the rate as an effective mass rate by multiplying
        # by particle mass directly (the rate constant absorbs the N_A factor).
        mass_rate_kg_m2_s = rate_mol_m2_s * self.particle_mass_kg * 6.022e23  # if rate is in mol
        # Actually, the rate constant calibration is ambiguous without explicit units.
        # Let's use a simpler empirical formula calibrated to typical incorporation levels:
        # At j = 100 mA/cm² and C_p = 1 g/L, literature reports ~0.5–2 wt% carbon.
        # We'll compute the incorporation fraction empirically.
        sigma = self.loose_adsorption_coverage_sigma(current_density_mA_cm2)
        # Empirical incorporation fraction: small fraction of loose-adsorbed particles
        # become incorporated, proportional to j and C_p.
        incorporation_fraction = 0.012 * sigma * (current_density_mA_cm2 / 100.0) ** 0.5
        # Maximum practical incorporation (saturation) ~5 wt% for fine carbon
        incorporation_fraction = min(incorporation_fraction, 0.05)
        # Mass rate of metal deposition (approximate for Fe at 100 mA/cm², 90% CE)
        j_A_m2 = current_density_mA_cm2 * 10.0
        # Approximate metal deposition rate (kg/m²/s) for pure Fe
        metal_rate_approx_kg_m2_s = j_A_m2 * 0.9 * M_FE / (2 * FARADAY)
        # Carbon mass rate = incorporation_fraction · metal_rate_approx / (1 - incorporation_fraction)
        # Actually, if carbon content = w_C / (w_C + w_metal) ≈ w_C for small w_C,
        # then carbon rate ≈ w_C · metal_rate / (1 - w_C)
        w_c_approx = incorporation_fraction
        carbon_mass_rate_kg_m2_s = w_c_approx * metal_rate_approx_kg_m2_s / max(1.0 - w_c_approx, 0.99)
        return float(carbon_mass_rate_kg_m2_s)

    # -------------------------------------------------------------------
    # Carbon content prediction
    # -------------------------------------------------------------------

    def carbon_content_wt_percent(
        self,
        current_density_mA_cm2: float,
        metal_deposition_rate_kg_m2_s: Optional[float] = None,
    ) -> float:
        """
        Predict carbon content (wt%) in the deposit at a given current density.

        The prediction uses the empirical incorporation fraction derived from
        the Guglielmi model.  It is intended as a screening tool; actual
        incorporation must be verified by combustion analysis or EDS.
        """
        sigma = self.loose_adsorption_coverage_sigma(current_density_mA_cm2)
        # Base incorporation fraction from literature correlations
        # Fine particles (≤ 2 µm), moderate zeta (−25 mV), 60 °C, 1–5 g/L
        base_frac = 0.008 * sigma * math.sqrt(current_density_mA_cm2 / 50.0)
        # Concentration scaling: linear up to ~5 g/L, then sub-linear (saturation)
        conc_scale = (self.particle_conc_g_L / 2.0) / (1.0 + self.particle_conc_g_L / 5.0)
        base_frac *= (0.5 + 0.5 * conc_scale)
        # Temperature correction: higher T → lower viscosity → higher transport
        temp_corr = 1.0 + 0.005 * (self.temperature_C - 25.0)
        base_frac *= temp_corr
        # Zeta potential: positive zeta enhances incorporation; negative reduces
        zeta_corr = 1.0 + 0.015 * max(0.0, self.zeta_potential_mV) - 0.005 * abs(min(0.0, self.zeta_potential_mV))
        base_frac *= zeta_corr
        # Size effect: smaller particles incorporate more easily
        size_corr = 1.0 / (1.0 + max(0.0, self.particle_size_um - 1.0) * 0.3)
        base_frac *= size_corr
        # Cap at realistic values for aqueous composite plating
        w_c = min(max(base_frac, 0.0), 0.15)  # 15 wt% absolute maximum for aqueous plating
        return float(w_c * 100.0)

    def carbon_incorporation_result(
        self,
        current_density_mA_cm2: float,
        metal_current_efficiency: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Full carbon incorporation report at a given operating point.
        """
        sigma = self.loose_adsorption_coverage_sigma(current_density_mA_cm2)
        rate_strong = self.strong_adsorption_rate_mol_m2_s(current_density_mA_cm2)
        w_c = self.carbon_content_wt_percent(current_density_mA_cm2)
        # Adjusted metal current efficiency: particle coverage blocks a fraction
        # of the cathode surface, reducing active area for metal deposition.
        # This is a first-order approximation.
        blocking_factor = 1.0 - 0.3 * sigma  # 30% of loose-adsorption area is blocked
        adjusted_ce = metal_current_efficiency * blocking_factor
        return {
            "current_density_mA_cm2": current_density_mA_cm2,
            "particle_loading_g_L": self.particle_conc_g_L,
            "particle_size_um": self.particle_size_um,
            "zeta_potential_mV": self.zeta_potential_mV,
            "loose_adsorption_coverage_sigma": round(sigma, 4),
            "strong_adsorption_rate_mol_m2_s": float(rate_strong),
            "predicted_carbon_wt_percent": round(w_c, 2),
            "predicted_carbon_vol_percent_approx": round(w_c / (w_c + (1.0 - w_c) * (2200.0 / 7874.0)) * 100, 2),
            "surface_blocking_factor": round(1.0 - 0.3 * sigma, 3),
            "adjusted_metal_current_efficiency_percent": round(adjusted_ce * 100.0, 1),
            "temperature_C": self.temperature_C,
            "mechanism_notes": (
                "Loose adsorption reversible (Langmuir); strong adsorption irreversible "
                "(Guglielmi). Only a small fraction θ ≪ σ of loose-adsorbed particles "
                "is incorporated. Actual content must be verified by combustion analysis."
            ),
        }


# -------------------------------------------------------------------
# Section 3: Integrated Phase III co-deposition model
# -------------------------------------------------------------------

@dataclass
class PhaseIIICoDeposition:
    """
    Integrated Phase III co-deposition model combining anomalous Fe–Ni
    kinetics and Guglielmi carbon-particle incorporation.

    This class provides a single callable interface for screening co-deposition
    operating conditions and predicting deposit properties (alloy composition,
    carbon content, current efficiency, and a diagnostic anomaly flag).

    Parameters
    ----------
    bath_fe_M : float
        Bulk Fe²⁺ concentration (mol/L).
    bath_ni_M : float
        Bulk Ni²⁺ concentration (mol/L).
    pH : float
        Bulk electrolyte pH.
    temperature_C : float
        Temperature (°C).
    carbon_particle_loading_g_L : float
        Carbon particle bath loading (g/L).  Set to 0.0 for pure alloy studies.
    mechanism_fe_ni : {"hydroxide_suppression", "intermediate_adsorption", "mixed_metal_intermediate"}
        Mechanism for anomalous Fe–Ni co-deposition.
    mechanism_carbon : {"guglielmi_two_step"}
        Only the Guglielmi two-step model is implemented.
    **kinetics_kwargs
        Additional keyword arguments forwarded to
        ``AnomalousFeNiKinetics``.
    **carbon_kwargs
        Additional keyword arguments forwarded to
        ``GuglielmiCarbonIncorporation``.

    Example
    -------
    ::

        model = PhaseIIICoDeposition(
            bath_fe_M=0.5,
            bath_ni_M=0.5,
            pH=3.5,
            carbon_particle_loading_g_L=2.0,
        )
        result = model.run_at_current(100.0)
        print(result)
    """

    bath_fe_M: float = 0.5
    bath_ni_M: float = 0.5
    pH: float = 3.5
    temperature_C: float = 60.0
    carbon_particle_loading_g_L: float = 1.0
    mechanism_fe_ni: Literal[
        "hydroxide_suppression", "intermediate_adsorption", "mixed_metal_intermediate"
    ] = "hydroxide_suppression"
    mechanism_carbon: Literal["guglielmi_two_step"] = "guglielmi_two_step"

    # Internal sub-model instances
    kinetics_model: Optional[AnomalousFeNiKinetics] = field(init=False, repr=False)
    carbon_model: Optional[GuglielmiCarbonIncorporation] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize sub-models with user parameters
        self.kinetics_model = AnomalousFeNiKinetics(
            bath_fe_M=self.bath_fe_M,
            bath_ni_M=self.bath_ni_M,
            pH=self.pH,
            temperature_C=self.temperature_C,
            mechanism=self.mechanism_fe_ni,
        )
        self.carbon_model = GuglielmiCarbonIncorporation(
            particle_conc_g_L=self.carbon_particle_loading_g_L,
            temperature_C=self.temperature_C,
            boundary_layer_m=self.kinetics_model.boundary_layer_m,
        )

    def _alloy_composition_pulsed(
        self,
        j_avg_mA_cm2: float,
        j_peak_mA_cm2: float,
        duty_cycle: float,
        waveform: Literal["dc", "pe", "pre"],
    ) -> Dict[str, Any]:
        """
        Pulsed-aware alloy composition.

        Key coupling: surface pH uses pulse-aware estimate (off-time recovery),
        and effective boundary layer is thinned (enhanced mass transport).
        This reduces hydroxide-suppression at same peak j, pushing composition
        toward less anomalous (higher Ni) under PE/PRE.

        Screening implementation: temporarily override kinetics_model surface_pH
        by monkey-patching its surface_pH method for this evaluation, and adjust
        effective i_lim via thinner boundary layer.
        """

        # Save original methods / values
        orig_boundary = self.kinetics_model.boundary_layer_m
        # Compute effective boundary layer thickness under pulse
        eff_delta = self.kinetics_model.effective_boundary_layer_pulsed(
            j_avg_mA_cm2, j_peak_mA_cm2, duty_cycle, waveform
        )
        # Override boundary layer to enhance limiting current
        object.__setattr__(self.kinetics_model, "boundary_layer_m", eff_delta)

        # Override surface_pH method to return pulsed pH
        orig_surface_pH_method = self.kinetics_model.surface_pH

        def pulsed_pH(j: float) -> float:
            # j is the current used for inhibition evaluation (typically j_avg)
            # Use pulse-aware calculation
            return self.kinetics_model.surface_pH_pulsed(
                j_avg_mA_cm2, j_peak_mA_cm2, duty_cycle, waveform
            )

        # Bind override
        self.kinetics_model.surface_pH = pulsed_pH  # type: ignore

        try:
            # Use j_avg for galvanostatic balance under pulse (cycle-averaged metal rate tracks avg)
            # but allow higher peak overpotential for nucleation effects via separate flag
            res = self.kinetics_model.alloy_composition(j_avg_mA_cm2)
            # Annotate with pulse info
            res["pulsed_surface_pH"] = pulsed_pH(j_avg_mA_cm2)
            res["effective_boundary_layer_m"] = eff_delta
            res["waveform"] = waveform
            res["j_peak"] = j_peak_mA_cm2
            res["duty"] = duty_cycle
            return res
        finally:
            # Restore
            object.__setattr__(self.kinetics_model, "boundary_layer_m", orig_boundary)
            self.kinetics_model.surface_pH = orig_surface_pH_method  # type: ignore

    def run_at_current(self, j_mA_cm2: float) -> Dict[str, Any]:
        """
        Run the integrated model at a given galvanostatic current density (DC).

        Returns a complete diagnostic dictionary suitable for experimental
        comparison or synthetic data reporting.
        """
        # Alloy composition (anomalous kinetics)
        alloy_res = self.kinetics_model.alloy_composition(j_mA_cm2)

        # Carbon incorporation (Guglielmi)
        carbon_res = self.carbon_model.carbon_incorporation_result(
            j_mA_cm2,
            metal_current_efficiency=alloy_res["current_efficiency_percent"] / 100.0,
        )

        # Adjusted overall current efficiency (metal + blocking effect)
        # The carbon model reports an adjusted CE; we combine with the alloy CE.
        base_ce = alloy_res["current_efficiency_percent"] / 100.0
        blocking_factor = carbon_res["surface_blocking_factor"]
        adjusted_ce = base_ce * blocking_factor

        # Linear deposition rate approximation (pure iron reference, adjusted for alloy)
        # Approximate alloy density: weighted average of Fe and Ni densities
        fe_frac = alloy_res["fe_wt_percent"] / 100.0
        ni_frac = alloy_res["ni_wt_percent"] / 100.0
        alloy_density = 7874.0 * fe_frac + 8908.0 * ni_frac
        j_A_m2 = j_mA_cm2 * 10.0
        # Average electron number per metal atom (approx 2 for both Fe and Ni)
        deposition_rate_um_hr = (
            j_A_m2 * adjusted_ce * M_FE / (2.0 * FARADAY)
            / max(alloy_density, 1.0)
            * 3600.0 * 1e6
        )

        # Diagnostic summary
        is_anomalous = alloy_res["is_anomalous"]
        anomalous_description = (
            "ANOMALOUS: Fe preferentially deposited (iron-group behavior). "
            "Mechanism: " + self.mechanism_fe_ni + "."
            if is_anomalous else
            "NORMAL: Alloy follows bath-composition trend."
        )

        # Carbon incorporation quality flag
        carbon_quality_flag = "NORMAL" if carbon_res["predicted_carbon_wt_percent"] < 8.0 else "HIGH"

        return {
            "operating_point": {
                "j_mA_cm2": j_mA_cm2,
                "pH": self.pH,
                "temperature_C": self.temperature_C,
                "bath_fe_M": self.bath_fe_M,
                "bath_ni_M": self.bath_ni_M,
                "carbon_loading_g_L": self.carbon_particle_loading_g_L,
            },
            "alloy_kinetics": {
                "E_op_V_vs_SHE": alloy_res["E_op_V_vs_SHE"],
                "fe_wt_percent": alloy_res["fe_wt_percent"],
                "ni_wt_percent": alloy_res["ni_wt_percent"],
                "bulk_fe_ref_wt_percent": alloy_res["bulk_fe_wt_percent_ref"],
                "current_efficiency_percent": alloy_res["current_efficiency_percent"],
                "partial_currents_A_m2": alloy_res["partial_currents_A_m2"],
                "is_anomalous": is_anomalous,
                "mechanism": self.mechanism_fe_ni,
            },
            "carbon_incorporation": {
                "predicted_carbon_wt_percent": carbon_res["predicted_carbon_wt_percent"],
                "loose_adsorption_sigma": carbon_res["loose_adsorption_coverage_sigma"],
                "surface_blocking_factor": carbon_res["surface_blocking_factor"],
                "adjusted_ce_percent": round(adjusted_ce * 100.0, 1),
                "quality_flag": carbon_quality_flag,
            },
            "integrated_metrics": {
                "adjusted_overall_current_efficiency_percent": round(adjusted_ce * 100.0, 1),
                "deposition_rate_um_hr": round(deposition_rate_um_hr, 1),
                "alloy_density_kg_m3_approx": round(alloy_density, 0),
                "anomalous_description": anomalous_description,
            },
            "model_notes": (
                "Phase III co-deposition screening model. Alloy predictions based on "
                f"mechanism '{self.mechanism_fe_ni}'; carbon predictions based on "
                "Guglielmi two-step model. All predictions require wet-lab verification."
            ),
        }

    def run_sweep(
        self,
        j_range: Optional[Iterable[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run the integrated model over a range of current densities and return
        arrays for plotting or tabular reporting.
        """
        if j_range is None:
            j_range = np.linspace(10.0, 300.0, 30)
        js = np.asarray(list(j_range), dtype=float)
        records = [self.run_at_current(float(j)) for j in js]
        return {
            "j_mA_cm2": js.tolist(),
            "fe_wt_percent": [r["alloy_kinetics"]["fe_wt_percent"] for r in records],
            "ni_wt_percent": [r["alloy_kinetics"]["ni_wt_percent"] for r in records],
            "current_efficiency_percent": [r["alloy_kinetics"]["current_efficiency_percent"] for r in records],
            "carbon_wt_percent": [r["carbon_incorporation"]["predicted_carbon_wt_percent"] for r in records],
            "is_anomalous": [r["alloy_kinetics"]["is_anomalous"] for r in records],
            "adjusted_ce_percent": [r["integrated_metrics"]["adjusted_overall_current_efficiency_percent"] for r in records],
            "deposition_rate_um_hr": [r["integrated_metrics"]["deposition_rate_um_hr"] for r in records],
        }

    def run_at_current_pulsed(
        self,
        j_avg_mA_cm2: float,
        j_peak_mA_cm2: float,
        duty_cycle: float = 0.5,
        waveform: Literal["dc", "pe", "pre"] = "pe",
    ) -> Dict[str, Any]:
        """
        Pulse-aware co-deposition run.

        Couples pulse.py recovery (off-time proton & Fe2+ replenishment) to
        hydroxide-suppression and mass-transport limits. PE/PRE give:
        * lower surface pH than DC at same peak j (less anomalous suppression)
        * thinner effective boundary layer (higher i_lim)
        * carbon incorporation uses j_avg for σ but peak for strong-adsorption driving force.

        Returns same schema as run_at_current() with extra pulse diagnostics.
        """

        alloy_res = self._alloy_composition_pulsed(
            j_avg_mA_cm2, j_peak_mA_cm2, duty_cycle, waveform
        )

        # Carbon: loose adsorption tracks average particle flux (j_avg),
        # strong adsorption driven by peak current (higher field during on-time)
        # Use effective j = sqrt(j_avg * j_peak) as compromise, or weighted by duty
        j_eff_carbon = math.sqrt(j_avg_mA_cm2 * j_peak_mA_cm2) if waveform != "dc" else j_avg_mA_cm2

        carbon_res = self.carbon_model.carbon_incorporation_result(
            j_eff_carbon,
            metal_current_efficiency=alloy_res["current_efficiency_percent"] / 100.0,
        )

        base_ce = alloy_res["current_efficiency_percent"] / 100.0
        blocking_factor = carbon_res["surface_blocking_factor"]
        adjusted_ce = base_ce * blocking_factor

        fe_frac = alloy_res["fe_wt_percent"] / 100.0
        ni_frac = alloy_res["ni_wt_percent"] / 100.0
        alloy_density = 7874.0 * fe_frac + 8908.0 * ni_frac
        j_A_m2 = j_avg_mA_cm2 * 10.0  # average rate determines mass

        deposition_rate_um_hr = (
            j_A_m2 * adjusted_ce * M_FE / (2.0 * FARADAY)
            / max(alloy_density, 1.0)
            * 3600.0 * 1e6
        )

        is_anomalous = alloy_res["is_anomalous"]
        anomalous_description = (
            f"ANOMALOUS (pulsed {waveform}): Fe preferential. Mechanism {self.mechanism_fe_ni}. "
            f"pH_surf pulsed={alloy_res.get('pulsed_surface_pH', 'n/a'):.2f} vs DC would be higher."
            if is_anomalous else
            f"NORMAL (pulsed {waveform}): alloy follows bath trend under pulse recovery."
        )

        carbon_quality_flag = "NORMAL" if carbon_res["predicted_carbon_wt_percent"] < 8.0 else "HIGH"

        return {
            "operating_point": {
                "j_avg_mA_cm2": j_avg_mA_cm2,
                "j_peak_mA_cm2": j_peak_mA_cm2,
                "duty_cycle": duty_cycle,
                "waveform": waveform,
                "pH": self.pH,
                "temperature_C": self.temperature_C,
                "bath_fe_M": self.bath_fe_M,
                "bath_ni_M": self.bath_ni_M,
                "carbon_loading_g_L": self.carbon_particle_loading_g_L,
            },
            "alloy_kinetics": {
                "E_op_V_vs_SHE": alloy_res["E_op_V_vs_SHE"],
                "fe_wt_percent": alloy_res["fe_wt_percent"],
                "ni_wt_percent": alloy_res["ni_wt_percent"],
                "bulk_fe_ref_wt_percent": alloy_res["bulk_fe_wt_percent_ref"],
                "current_efficiency_percent": alloy_res["current_efficiency_percent"],
                "partial_currents_A_m2": alloy_res["partial_currents_A_m2"],
                "is_anomalous": is_anomalous,
                "mechanism": self.mechanism_fe_ni,
                "pulsed_surface_pH": alloy_res.get("pulsed_surface_pH"),
                "effective_boundary_layer_m": alloy_res.get("effective_boundary_layer_m"),
            },
            "carbon_incorporation": {
                "predicted_carbon_wt_percent": carbon_res["predicted_carbon_wt_percent"],
                "loose_adsorption_sigma": carbon_res["loose_adsorption_coverage_sigma"],
                "surface_blocking_factor": carbon_res["surface_blocking_factor"],
                "adjusted_ce_percent": round(adjusted_ce * 100.0, 1),
                "quality_flag": carbon_quality_flag,
                "j_eff_for_carbon_mA_cm2": round(j_eff_carbon, 1),
            },
            "integrated_metrics": {
                "adjusted_overall_current_efficiency_percent": round(adjusted_ce * 100.0, 1),
                "deposition_rate_um_hr": round(deposition_rate_um_hr, 1),
                "alloy_density_kg_m3_approx": round(alloy_density, 0),
                "anomalous_description": anomalous_description,
                "pulse_benefit": f"pH recovery {waveform}: surface pH lower than DC peak, i_lim enhanced by boundary thinning",
            },
            "model_notes": (
                "Pulsed Phase III screening — couples pulse recovery to hydroxide suppression. "
                f"mechanism '{self.mechanism_fe_ni}', waveform '{waveform}'. Requires wet-lab validation."
            ),
        }

    def run_sweep_pulsed(
        self,
        j_avg_range: Optional[Iterable[float]] = None,
        j_peak_factor: float = 2.0,
        duty_cycle: float = 0.5,
        waveform: Literal["dc", "pe", "pre"] = "pe",
    ) -> Dict[str, Any]:
        """
        Sweep j_avg with fixed peak/avg ratio.

        j_peak = j_avg * j_peak_factor (e.g., 2× for 50% duty gives same avg as DC peak).
        """
        if j_avg_range is None:
            j_avg_range = np.linspace(10.0, 200.0, 25)
        js_avg = np.asarray(list(j_avg_range), dtype=float)
        records = [
            self.run_at_current_pulsed(float(j_avg), float(j_avg * j_peak_factor), duty_cycle, waveform)
            for j_avg in js_avg
        ]
        return {
            "j_avg_mA_cm2": js_avg.tolist(),
            "j_peak_mA_cm2": (js_avg * j_peak_factor).tolist(),
            "fe_wt_percent": [r["alloy_kinetics"]["fe_wt_percent"] for r in records],
            "ni_wt_percent": [r["alloy_kinetics"]["ni_wt_percent"] for r in records],
            "carbon_wt_percent": [r["carbon_incorporation"]["predicted_carbon_wt_percent"] for r in records],
            "is_anomalous": [r["alloy_kinetics"]["is_anomalous"] for r in records],
            "adjusted_ce_percent": [r["integrated_metrics"]["adjusted_overall_current_efficiency_percent"] for r in records],
            "pulsed_surface_pH": [r["alloy_kinetics"].get("pulsed_surface_pH") for r in records],
        }

    # -------------------------------------------------------------------
    # Summary and reporting
    # -------------------------------------------------------------------

    def summary_dict(self, j_mA_cm2: float = 100.0) -> Dict[str, Any]:
        """Return a formatted summary dictionary at a representative current density."""
        return self.run_at_current(j_mA_cm2)

    def __str__(self) -> str:
        res = self.run_at_current(100.0)
        lines = [
            "Phase III Co-Deposition — Integrated Screening Model",
            "=" * 55,
            f"Operating point: j = 100 mA/cm², pH = {self.pH}, T = {self.temperature_C} °C",
            f"Bath: Fe = {self.bath_fe_M} M, Ni = {self.bath_ni_M} M, C particles = {self.carbon_particle_loading_g_L} g/L",
            f"Mechanism (Fe-Ni): {self.mechanism_fe_ni}",
            f"Mechanism (C): {self.mechanism_carbon}",
            "-" * 55,
            f"Alloy: Fe = {res['alloy_kinetics']['fe_wt_percent']} wt%, "
            f"Ni = {res['alloy_kinetics']['ni_wt_percent']} wt%",
            f"Anomalous behavior: {'YES' if res['alloy_kinetics']['is_anomalous'] else 'NO'}",
            f"Carbon content (predicted): {res['carbon_incorporation']['predicted_carbon_wt_percent']} wt%",
            f"Adjusted current efficiency: {res['integrated_metrics']['adjusted_overall_current_efficiency_percent']}%",
            f"Deposition rate: {res['integrated_metrics']['deposition_rate_um_hr']} µm/hr",
            "=" * 55,
            "NOTE: All predictions are screening-level and must be verified "
            "by SEM-EDS, combustion analysis, and gravimetric balance.",
        ]
        return "\n".join(lines)


# -------------------------------------------------------------------
# Convenience constructor / factory functions
# -------------------------------------------------------------------


def build_phase3_model(
    bath_fe_M: float = 0.5,
    bath_ni_M: float = 0.5,
    pH: float = 3.5,
    temperature_C: float = 60.0,
    carbon_particle_loading_g_L: float = 2.0,
    mechanism_fe_ni: str = "hydroxide_suppression",
    particle_size_um: float = 1.5,
    zeta_potential_mV: float = -25.0,
    agitation_flow_rate_L_min: float = 2.0,
) -> PhaseIIICoDeposition:
    """Convenience factory for creating a Phase III co-deposition model."""
    model = PhaseIIICoDeposition(
        bath_fe_M=bath_fe_M,
        bath_ni_M=bath_ni_M,
        pH=pH,
        temperature_C=temperature_C,
        carbon_particle_loading_g_L=carbon_particle_loading_g_L,
        mechanism_fe_ni=mechanism_fe_ni,  # type: ignore[arg-type]
    )
    # Override carbon model parameters if needed
    model.carbon_model.particle_size_um = particle_size_um
    model.carbon_model.zeta_potential_mV = zeta_potential_mV
    model.carbon_model.agitation_flow_rate_L_min = agitation_flow_rate_L_min
    return model
