"""
Thermogalvanic Seebeck potentials and non-isothermal current maldistribution in industrial stacks.

Physics and Chemistry
---------------------
In industrial-scale electrowinning reactors (e.g., 1-meter tall vertical plates),
the temperature of the electrolyte is not spatially uniform. Ohmic dissipation (I²R)
and reaction enthalpy generate substantial heat, while convective flow and wall heat loss
establish a spatial temperature gradient along the electrode height (y):
     T(y) = T_inlet + ΔT_thermal * (y / H)

In electrolytes, a temperature gradient acts as a thermogalvanic battery, shifting the local
open-circuit / equilibrium potential of the redox reactions (the Seebeck or thermogalvanic effect):
     E_eq(T) = E_eq(T_ref) + S_thermo * (T - T_ref)
where S_thermo = dE_eq/dT (V/K) is the thermogalvanic coefficient of the reaction:
     - Iron deposition (Fe²⁺ + 2e⁻ ⇌ Fe): S_Fe ≈ +1.20 mV/K
     - Hydrogen evolution (2H⁺ + 2e⁻ ⇌ H₂): S_HER ≈ +0.87 mV/K
     - Oxygen evolution (OER, at the anode): S_OER ≈ -1.36 mV/K

Because the metal phase of the cathode remains a near-perfect equipotential, this vertical
variation in E_eq(y) directly shifts the local overpotential:
     η_local(y) = V_metal - V_solution - E_eq(T(y))
Since charge transfer kinetics (Butler–Volmer) depend exponentially on overpotential,
a 10–15 mV Seebeck potential difference along the cell height drives non-uniform
current distribution (maldistribution), triggering localized hot spots and dendritic
growth at the hotter sections.

References
----------
* Agar, J. N., & Breck, W. G. (1957). "Thermogalvanic Cells." Trans. Faraday Soc., 53, 167.
* Newman, J., & Thomas-Alyea, K. E. (2004). "Electrochemical Systems." John Wiley & Sons.
* Gunther, A., et al. (2012). "Thermogalvanic cells for waste heat harvesting."
  Electrochimica Acta, 80, 245–251.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class ThermogalvanicParams:
    """Thermogalvanic coefficients and fluid properties for non-isothermal cell scaling."""

    s_fe_V_K: float = 1.2e-3             # Seebeck coefficient for Fe2+/Fe (V/K)
    s_her_V_K: float = 0.87e-3           # Seebeck coefficient for HER (V/K)
    s_oer_V_K: float = -1.36e-3          # Seebeck coefficient for OER (V/K)
    e_fe_ref_V: float = -0.44            # E_eq of Fe2+/Fe at 25 °C (V vs SHE)
    e_her_ref_V: float = 0.0             # E_eq of HER at 25 °C (V vs SHE)
    electrolyte_density_kg_m3: float = 1200.0  # Density of sulfate bath (kg/m³)
    specific_heat_J_kgK: float = 3800.0  # Specific heat capacity (J/kg·K)
    channel_gap_m: float = 0.02          # Electrode spacing / gap (m)
    electrode_height_m: float = 1.0      # Vertical height of the industrial electrode (m)


def get_thermogalvanic_equilibrium_potential(
    temperature_C: float,
    reaction: str = "fe",
    t_ref_C: float = 25.0,
    params: Optional[ThermogalvanicParams] = None,
) -> float:
    """
    Calculate the non-isothermal equilibrium potential (V vs SHE) at a given temperature.
    """
    if params is None:
        params = ThermogalvanicParams()

    t_k = temperature_C + 273.15
    t_ref_k = t_ref_C + 273.15
    delta_t = t_k - t_ref_k

    if reaction.lower() == "fe":
        e_ref = params.e_fe_ref_V
        s_coeff = params.s_fe_V_K
    elif reaction.lower() == "her":
        e_ref = params.e_her_ref_V
        s_coeff = params.s_her_V_K
    elif reaction.lower() == "oer":
        # OER E_eq at 25°C is ~1.23 V vs SHE
        e_ref = 1.23
        s_coeff = params.s_oer_V_K
    else:
        raise ValueError(f"Unknown reaction type: {reaction}")

    return e_ref + s_coeff * delta_t


@dataclass
class MaldistributionResult:
    """Current density maldistribution along a vertical non-isothermal electrode."""

    height_steps_m: List[float]
    temperatures_C: List[float]
    local_fe_currents_mA_cm2: List[float]
    local_her_currents_mA_cm2: List[float]
    maldistribution_index: float        # (j_max - j_min) / j_avg (0 = perfectly uniform)
    peak_current_mA_cm2: float
    trough_current_mA_cm2: float
    total_thermal_rise_C: float


def solve_vertical_current_maldistribution(
    j_avg_mA_cm2: float,
    v_cell_V: float,
    t_inlet_C: float,
    flow_velocity_m_s: float = 0.1,
    params: Optional[ThermogalvanicParams] = None,
) -> MaldistributionResult:
    """
    Solve the coupled thermal-electrochemical 1D equations along the electrode height
    to calculate the vertical temperature profile, local equilibrium potentials,
    and resulting current density maldistribution.

    Parameters
    ----------
    j_avg_mA_cm2 : float
        Nominal average cell current density (mA/cm²).
    v_cell_V : float
        Operating cell voltage (V).
    t_inlet_C : float
        Inlet electrolyte temperature (°C).
    flow_velocity_m_s : float, default 0.1
        Electrolyte velocity in the vertical channel (m/s).
    params : ThermogalvanicParams, optional
        Scaling parameters.

    Returns
    -------
    MaldistributionResult
        Vertical profiles and maldistribution metrics.
    """
    if params is None:
        params = ThermogalvanicParams()

    h = params.electrode_height_m
    u = max(float(flow_velocity_m_s), 1e-4)
    j_avg_SI = max(float(j_avg_mA_cm2), 0.0) * 10.0  # mA/cm² -> A/m²

    # Discrete segments along the height (y = 0 to H)
    n_nodes = 21
    y_eval = np.linspace(0.0, h, n_nodes)

    # Ohmic power dissipation in the channel: Q_ohmic = j_avg * V_cell (W/m² of electrode)
    # The volumetric temperature rise rate in the flowing fluid channel:
    # dT/dy = (j_avg * V_cell) / (rho * Cp * u * gap)
    denom = params.electrolyte_density_kg_m3 * params.specific_heat_J_kgK * u * params.channel_gap_m
    dt_dy = (j_avg_SI * float(v_cell_V)) / denom

    # Calculate temperature profile along height
    t_profile = t_inlet_C + dt_dy * y_eval
    t_rise = t_profile[-1] - t_profile[0]

    # Solve for local overpotentials and Butler-Volmer currents.
    # To maintain j_avg on average, we solve for an average overpotential shift V_shift
    # such that 1/H * Integral_{0}^{H} j_Fe(y) dy = j_avg.
    # Tafel slope for Fe deposition is RT/alpha*F ≈ 120 mV/dec at 25°C.
    # We use a simplified Tafel approximation to find the vertical distribution:
    # j_local(y) = j_avg_SI * exp( alpha * F * (V_shift - s_fe * ΔT(y)) / (R * T(y)) )
    r_const = 8.314
    f_const = 96485.3
    alpha_fe = 0.5  # Typical cathodic transfer coefficient for Fe

    # We solve for the shift factor to satisfy average current density.
    # Let's compute local currents relative to overpotential shifts.
    # For numerical stability and simplicity, we express:
    # j_local(y) = C * exp(-alpha * F * s_fe * (T(y) - T_inlet) / (R * T(y)))
    # and then normalize such that the average equals j_avg.
    t_k = t_profile + 273.15
    delta_t = t_profile - t_inlet_C

    # Local thermodynamic Seebeck shift in equilibrium potential: s_fe * delta_t
    potential_shift_V = params.s_fe_V_K * delta_t
    
    # Kinetic sensitivity: exp(-alpha * F * E_eq_shift / (R * T))
    # Note that a positive S_Fe means equilibrium potential shifts positive (easier to plate).
    # Thus overpotential η = V_m - V_s - E_eq shifts negative, reducing driving force.
    # Therefore, the hotter regions see lower overpotential and plate *less* if Butler-Volmer
    # is the only term. (However, exchange current density j0 increases with T, which we also model).
    
    # j0(T) = j0_ref * exp(-Ea / R * (1/T - 1/T_ref))
    # Ea for Fe deposition is ~40 kJ/mol
    ea_fe = 40.0e3
    j0_factors = np.exp(-(ea_fe / r_const) * (1.0 / t_k - 1.0 / (t_inlet_C + 273.15 + 1e-6)))

    # Local potential driving term
    kinetic_driving = np.exp((alpha_fe * f_const * potential_shift_V) / (r_const * t_k))
    relative_j = j0_factors * kinetic_driving
    
    # Normalize local current profile to match average j_avg_mA_cm2
    mean_rel = np.mean(relative_j)
    j_local_fe = (relative_j / mean_rel) * j_avg_mA_cm2 if mean_rel > 0 else relative_j * 0.0

    # Solve similarly for HER
    # Ea for HER on iron is ~35 kJ/mol
    ea_her = 35.0e3
    j0_her_factors = np.exp(-(ea_her / r_const) * (1.0 / t_k - 1.0 / (t_inlet_C + 273.15 + 1e-6)))
    her_potential_shift_V = params.s_her_V_K * delta_t
    her_kinetic_driving = np.exp((0.5 * f_const * her_potential_shift_V) / (r_const * t_k))
    relative_her = j0_her_factors * her_kinetic_driving
    
    # Assume average HER is ~20% of Fe current for scaling comparison
    j_local_her = (relative_her / np.mean(relative_her)) * (0.20 * j_avg_mA_cm2) if np.mean(relative_her) > 0 else relative_her * 0.0

    j_max = float(np.max(j_local_fe))
    j_min = float(np.min(j_local_fe))
    mdi = (j_max - j_min) / j_avg_mA_cm2 if j_avg_mA_cm2 > 0 else 0.0

    return MaldistributionResult(
        height_steps_m=list(y_eval),
        temperatures_C=list(t_profile),
        local_fe_currents_mA_cm2=list(j_local_fe),
        local_her_currents_mA_cm2=list(j_local_her),
        maldistribution_index=mdi,
        peak_current_mA_cm2=j_max,
        trough_current_mA_cm2=j_min,
        total_thermal_rise_C=t_rise,
    )


def main() -> None:
    """CLI entrypoint for thermogalvanic and maldistribution analysis."""
    print("=================================================================")
    print(" Thermogalvanic Seebeck Potentials & Vertical Current Distribution")
    print("=================================================================")
    params = ThermogalvanicParams()
    print(f"Industrial Cathode Height: {params.electrode_height_m:.1f} m")
    print(f"Channel Gap Size         : {params.channel_gap_m*1000:.1f} mm")
    print(f"Fe Seebeck Coefficient   : {params.s_fe_V_K*1000:.2f} mV/K\n")

    print("Electrolyte Flow Velocity vs Vertical Maldistribution (j_avg = 150 mA/cm², V_cell = 3.2 V):")
    for u in [0.01, 0.05, 0.1, 0.5]:
        res = solve_vertical_current_maldistribution(150.0, 3.2, 60.0, u, params)
        print(f"  Velocity u = {u:5.3f} m/s | Temp Rise: {res.total_thermal_rise_C:5.1f} °C")
        print(f"    Peak Current: {res.peak_current_mA_cm2:6.1f} mA/cm² | Trough: {res.trough_current_mA_cm2:6.1f} mA/cm²")
        print(f"    Maldistribution Index (MDI): {res.maldistribution_index*100:5.2f}%\n")


if __name__ == "__main__":
    main()
