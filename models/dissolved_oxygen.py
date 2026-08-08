"""
Dissolved oxygen solubility, cathodic reduction (ORR), and homogeneous ferrous oxidation kinetics.

Physics and Chemistry
---------------------
In aqueous iron electrowinning, the electrolyte is in contact with the atmosphere
or carry-over air, allowing oxygen (O₂) to dissolve into the bath. This module
models the two major parasitic pathways driven by dissolved oxygen:

1. **Cathodic Oxygen Reduction Reaction (ORR)**:
   Dissolved O₂ is highly electroactive. Since the equilibrium potential of ORR
   (E° = 1.23 V vs SHE) is far more positive than iron deposition (E° = -0.44 V vs SHE),
   ORR operates at its transport-limited current density across the entire operating window:
     j_lim,ORR = 4 * F * D_O2 * C_O2,bulk / δ
   This represents a constant, parasitic current efficiency penalty on Faradaic Efficiency (FE)
   and induces additional interfacial alkalization (proton consumption).

2. **Bulk Homogeneous Chemical Oxidation of Fe²⁺ to Fe³⁺**:
   Dissolved O₂ reacts directly with ferrous iron in the bulk electrolyte:
     4 Fe²⁺ + O₂ + 4 H⁺ ──► 4 Fe³⁺ + 2 H₂O
   The rate of this reaction is highly accelerated by temperature, pH (since basic/neutral
   conditions accelerate Fe(II) oxidation), and ferrous concentration:
     -d[Fe²⁺]/dt = 4 * k_hom * [Fe²⁺]² * [O₂] * [OH⁻]²
   This is the primary non-electrochemical source of Fe³⁺ in the bulk bath, which subsequently
   diffuses to the cathode, feeding the parasitic "ferric shuttle" loop.

3. **Solubility & Salinity Correction (Sechenov Equation)**:
   The solubility of O₂ in water decreases with temperature and salt concentration
   (salting-out effect), which is modeled using the Sechenov equation:
     log10(C_sat / C_sat,pure) = - K_s * I
   where K_s is the Sechenov salt parameter and I is the ionic strength of the electrolyte.

References
----------
* Millero, F. J., et al. (1987). "The oxidation of Fe(II) with O₂ in natural waters."
  Geochim. Cosmochim. Acta, 51(4), 793–803.
* Tromans, D. (1998). "Temperature and salinity effects on oxygen solubility in
  water: aqueous solutions of electrolytes." Hydrometallurgy, 50(3), 279-296.
* King, D. W. (1998). "Role of carbonate speciation in the reaction of Fe(II) with O₂."
  Environ. Sci. Technol., 32(19), 2997-3003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass



@dataclass(frozen=True)
class DissolvedOxygenParams:
    """Parameters for dissolved oxygen solubility and reaction kinetics in the bath."""

    temperature_C: float = 60.0          # Operating temperature (°C)
    ionic_strength_M: float = 2.5       # Electrolyte ionic strength (M), typical of 1.5 M FeSO4 + 0.5 M Na2SO4
    pH: float = 2.5                      # Bulk electrolyte pH
    fe2_M: float = 1.5                   # Ferrous concentration (M)
    delta_um: float = 100.0              # Cathode boundary layer thickness (µm)
    air_leak_rate_L_hr: float = 5.0      # Air ingress leak rate (L/hr) into cell headspace
    gas_volume_L: float = 10.0           # Headspace gas volume (L)
    liquid_volume_L: float = 20.0        # Active electrolyte liquid volume (L)

    @property
    def temperature_K(self) -> float:
        """Temperature in Kelvin (K)."""
        return self.temperature_C + 273.15

    @property
    def d_o2_m2_s(self) -> float:
        """
        Temperature-dependent diffusion coefficient of dissolved O₂ in water (m²/s).
        Obeying Arrhenius kinetics: D = D0 * exp(-Ea / RT).
        """
        d0 = 2.9e-6  # m²/s
        ea = 19.3e3  # J/mol
        r = 8.314
        return d0 * math.exp(-ea / (r * self.temperature_K))

    @property
    def delta_m(self) -> float:
        """Boundary layer thickness in meters (m)."""
        return self.delta_um * 1e-6


def pure_water_o2_saturation_M(temperature_C: float) -> float:
    """
    Compute dissolved O₂ saturation concentration in pure water (M) at 1 atm air.
    Uses a highly accurate and stable rational temperature fit calibrated against
    standard solubility tables (0 to 100 °C).
    """
    t = max(float(temperature_C), 0.0)
    # Rational fit: ratio of O2 density to molecular weight, matching:
    # 0 °C  -> 14.16 mg/L (4.425e-4 M)
    # 25 °C -> 8.26 mg/L  (2.58e-4 M)
    # 60 °C -> 4.72 mg/L  (1.47e-4 M)
    denom = 1.0 + 0.02516 * t + 0.000136 * (t ** 2)
    c_sat_pure = (1.416e-2 / 32.0) / denom
    return c_sat_pure


def dissolved_oxygen_solubility_M(
    temperature_C: float,
    ionic_strength_M: float,
    po2_fraction: float = 0.2095,
) -> float:
    """
    Calculate dissolved oxygen saturation solubility in the electrolyte (M).
    Adjusts pure-water saturation for temperature, ionic strength (Sechenov effect),
    and local oxygen partial pressure fraction.

    Parameters
    ----------
    temperature_C : float
        Temperature of the electrolyte (°C).
    ionic_strength_M : float
        Ionic strength of the electrolyte (mol/L).
    po2_fraction : float, default 0.2095
        Volume fraction/partial pressure of O₂ in headspace (0.2095 for air).

    Returns
    -------
    float
        O₂ solubility in mol/L (M).
    """
    c_pure_air = pure_water_o2_saturation_M(temperature_C)
    # Adjust for custom pO2 (pure_water_o2_saturation is fit for standard air, po2=0.2095)
    c_pure = c_pure_air * (po2_fraction / 0.2095)

    # Sechenov salt correction: log10(C_sat/C_sat,pure) = - K_s * I
    # K_s decreases slightly with temperature
    ks = 0.132 - 0.00035 * (temperature_C - 25.0)
    ks = max(ks, 0.05)

    c_sat = c_pure * (10.0 ** (-ks * ionic_strength_M))
    return max(c_sat, 1e-12)


def cathodic_orr_limiting_current_A_m2(
    params: DissolvedOxygenParams,
    do_fraction_sat: float = 1.0,
) -> float:
    """
    Calculate the mass-transfer transport-limited cathodic ORR current density.
    This reaction is 4-electron transfer: O₂ + 4 H⁺ + 4 e⁻ ──► 2 H₂O.

    Parameters
    ----------
    params : DissolvedOxygenParams
        Cell and boundary-layer physical parameters.
    do_fraction_sat : float, default 1.0
        The saturation level of O₂ in the bulk (1.0 = fully air-saturated).

    Returns
    -------
    float
        Limiting current density of ORR in A/m² (positive = cathodic).
    """
    c_sat = dissolved_oxygen_solubility_M(params.temperature_C, params.ionic_strength_M)
    c_bulk = c_sat * max(float(do_fraction_sat), 0.0)

    # Faraday's constant: 96485.3 C/mol
    f = 96485.3
    # j_lim = 4 * F * D * C_bulk / delta
    j_lim = (4.0 * f * params.d_o2_m2_s * c_bulk * 1000.0) / params.delta_m  # C_bulk in M, so *1000 for mol/m³
    return j_lim


def homogeneous_fe2_oxidation_rate_M_s(
    params: DissolvedOxygenParams,
    do_fraction_sat: float = 1.0,
) -> float:
    """
    Compute the homogeneous chemical oxidation rate of Fe²⁺ to Fe³⁺ by dissolved oxygen.
    Equation: 4 Fe²⁺ + O₂ + 4 H⁺ ──► 4 Fe³⁺ + 2 H₂O
    The rate: r = -d[Fe²⁺]/dt = 4 * k_hom * [Fe²⁺]² * [O₂] * [OH⁻]² (mol/L/s)

    Parameters
    ----------
    params : DissolvedOxygenParams
        Solution temperature, pH, ionic strength, and Fe²⁺ concentration.
    do_fraction_sat : float, default 1.0
        Dissolved O₂ saturation ratio.

    Returns
    -------
    float
        The rate of Fe³⁺ production in M/s (which is equal to 4x the O₂ consumption rate).
    """
    c_sat = dissolved_oxygen_solubility_M(params.temperature_C, params.ionic_strength_M)
    c_o2 = c_sat * max(float(do_fraction_sat), 0.0)

    # Autoprotolysis constant of water Kw vs T (approximate)
    # Kw = [H+][OH-]
    t_k = params.temperature_K
    p_kw = 4471.33 / t_k - 6.0846 + 0.017053 * t_k
    kw = 10.0 ** (-p_kw)

    h_activity = 10.0 ** (-params.pH)
    oh_activity = kw / h_activity

    # k_hom is highly temperature-dependent, fit from Millero (1987)
    # log10(k_hom) at 25°C and low ionic strength is ~11.0 to 12.0 M⁻³ s⁻¹
    # Temperature activation energy Ea ≈ 96 kJ/mol
    ea_j_mol = 96e3
    r_const = 8.314
    # Reference value at 25°C (298.15 K)
    k_ref = 3.5e11  # M⁻³ s⁻¹ (fitted for bulk solutions)
    k_hom = k_ref * math.exp(-(ea_j_mol / r_const) * (1.0 / t_k - 1.0 / 298.15))

    # Rate of O₂ chemical consumption: -d[O2]/dt = k_hom * [Fe2+]² * [O₂] * [OH⁻]²
    # We use concentration of Fe2+ and activities of O2 and OH- as a screening approximation
    rate_o2 = k_hom * (params.fe2_M ** 2) * c_o2 * (oh_activity ** 2)

    # 4 Fe²⁺ oxidized per 1 O₂ consumed
    fe3_gen_rate = 4.0 * rate_o2
    return max(fe3_gen_rate, 0.0)


@dataclass
class O2ControlAnalysis:
    """Analysis of headspace nitrogen sweep and sealing effectiveness on dissolved oxygen."""

    leak_rate_L_hr: float
    nitrogen_flow_L_min: float
    equilibrium_pO2_pct: float
    steady_state_DO_mM: float
    percent_saturation: float
    orr_penalty_A_m2: float
    fe3_generation_ppm_hr: float       # ppm weight Fe³⁺ generated per hour in bulk


def analyze_oxygen_ingress_control(
    params: DissolvedOxygenParams,
    nitrogen_flow_L_min: float = 2.0,
) -> O2ControlAnalysis:
    """
    Perform a steady-state mass balance in the headspace and liquid volume to evaluate
    the effectiveness of a nitrogen purge in suppressing dissolved O₂.

    Parameters
    ----------
    params : DissolvedOxygenParams
        Cell volumes, leak rates, and thermodynamic constants.
    nitrogen_flow_L_min : float, default 2.0
        Sweep rate of pure nitrogen gas sweep in L/min.

    Returns
    -------
    O2ControlAnalysis
         हेडस्पेस O₂ control and electrolytic penalties.
    """
    # Headspace gas balance:
    # Inflow: air leak (20.95% O2) + nitrogen sweep (0% O2)
    # Outflow: mixed gas out.
    # q_air = air_leak_rate_L_hr (L/hr)
    # q_n2 = nitrogen_flow_L_min * 60 (L/hr)
    q_air = max(params.air_leak_rate_L_hr, 0.0)
    q_n2 = max(nitrogen_flow_L_min * 60.0, 0.0)

    total_outflow = q_air + q_n2
    if total_outflow > 1e-6:
        pO2_fraction = (q_air * 0.2095) / total_outflow
    else:
        pO2_fraction = 0.2095

    # Steady state dissolved oxygen balance:
    # Oxygen enters liquid from headspace gas interface and is consumed by:
    # 1. Homogeneous ferrous oxidation: r_hom = k_hom * [Fe2+]² * [O2] * [OH-]²
    # At steady state (assuming gas-liquid equilibrium dominates or is fast),
    # the DO concentration is driven by the headspace partial pressure pO2.
    # In a fast-equilibrium limit:
    c_sat_active = dissolved_oxygen_solubility_M(params.temperature_C, params.ionic_strength_M, pO2_fraction)
    c_sat_air = dissolved_oxygen_solubility_M(params.temperature_C, params.ionic_strength_M, 0.2095)

    pct_sat = (c_sat_active / c_sat_air) * 100.0 if c_sat_air > 0 else 0.0

    # Compute cathodic and homogeneous penalties at this steady state
    j_orr = cathodic_orr_limiting_current_A_m2(params, do_fraction_sat=(c_sat_active / c_sat_air))
    r_fe3_M_s = homogeneous_fe2_oxidation_rate_M_s(params, do_fraction_sat=(c_sat_active / c_sat_air))

    # Convert M/s of Fe3+ to ppm weight Fe per hour:
    # 1 M Fe = 55.85 g/L. In 1 L of electrolyte (density ~ 1.2 kg/L), 1 M Fe = 55.85 g / 1200 g = 46.54 ppk = 46540 ppm.
    # ppm_weight/hr = r_fe3 (mol/L/s) * 55.85 (g/mol) * (1 / 1200 g/L) * 1e6 (ppm) * 3600 (s/hr)
    ppm_hr = r_fe3_M_s * 55.85 * (1e6 / 1200.0) * 3600.0

    return O2ControlAnalysis(
        leak_rate_L_hr=q_air,
        nitrogen_flow_L_min=nitrogen_flow_L_min,
        equilibrium_pO2_pct=pO2_fraction * 100.0,
        steady_state_DO_mM=c_sat_active * 1000.0,
        percent_saturation=pct_sat,
        orr_penalty_A_m2=j_orr,
        fe3_generation_ppm_hr=ppm_hr,
    )


def main() -> None:
    """CLI entrypoint for dissolved oxygen impact analysis."""
    print("=================================================================")
    print(" Dissolved Oxygen solubility, ORR, and Homogeneous Fe(II) Kinetics")
    print("=================================================================")
    
    for t in [25.0, 60.0, 80.0]:
        c_pure = pure_water_o2_saturation_M(t)
        c_salt = dissolved_oxygen_solubility_M(t, 2.5, 0.2095)
        print(f"At T = {t:4.1f} °C:")
        print(f"  O₂ sat (pure water)  : {c_pure * 1000.0:6.3f} mM ({c_pure*32.0*1000.0:4.1f} mg/L)")
        print(f"  O₂ sat (2.5M sulfate): {c_salt * 1000.0:6.3f} mM ({c_salt*32.0*1000.0:4.1f} mg/L) [Salting-Out]")

    print("\n-----------------------------------------------------------------")
    print(" Headspace Nitrogen Sweep sweep vs Air Ingress Leak (T=60°C, pH=2.5)")
    print("-----------------------------------------------------------------")
    params = DissolvedOxygenParams(temperature_C=60.0, pH=2.5)
    
    for n2_flow in [0.0, 0.5, 2.0, 5.0]:
        analysis = analyze_oxygen_ingress_control(params, n2_flow)
        print(f"N₂ sweep: {n2_flow:4.1f} L/min | Equilibrium headspace pO₂: {analysis.equilibrium_pO2_pct:5.2f}%")
        print(f"  Steady-state DO      : {analysis.steady_state_DO_mM:6.4f} mM ({analysis.percent_saturation:4.1f}% of saturated air)")
        print(f"  Parasitic ORR current: {analysis.orr_penalty_A_m2:6.3f} A/m²")
        print(f"  Bulk Fe³⁺ generation : {analysis.fe3_generation_ppm_hr:6.2f} ppm/hr\n")


if __name__ == "__main__":
    main()
