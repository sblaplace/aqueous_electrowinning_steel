"""Dynamic cell heat balance and thermal management model for electrowinning cells.

Models Joule heating, overpotential dissipation, ambient convection/radiation,
evaporative cooling, and active jacketed cooling requirements for benchtop
and pilot-scale divided electrowinning cells.

References:
- Incropera, F. P., et al. (2007). Fundamentals of Heat and Mass Transfer. Wiley.
- Danly, D. E. (1981). Scale-up of organic electrosynthesis. Journal of Electrochemical Society.
"""

from dataclasses import dataclass
from typing import Dict, Any
import math
import numpy as np

# Physical constants
E_THERM_FE = 1.28  # V, thermoneutral potential for Fe2+ + H2O -> Fe + 0.5 O2 + 2H+ at 25 °C
CP_WATER = 4.184   # J/(g*K)
RHO_WATER = 1.0    # g/cm^3 = kg/L
H_FG_WATER = 2260.0 # J/g latent heat of vaporization


@dataclass
class CellThermalParams:
    """Parameters for cell thermal balance simulation."""
    V_cell: float = 2.5        # V, total cell voltage
    current_A: float = 10.0    # A, total operating current
    volume_L: float = 2.0      # L, total electrolyte volume
    hardware_C_J_K: float = 500.0  # J/K, thermal mass of cell body/electrodes

    T_init_C: float = 25.0     # °C, initial electrolyte temperature
    T_amb_C: float = 22.0      # °C, ambient air temperature
    UA_amb_W_K: float = 1.5    # W/K, ambient overall heat transfer coefficient * area

    A_surface_m2: float = 0.04 # m^2, open top electrolyte surface area for evaporation
    relative_humidity: float = 0.50 # Ambient relative humidity (0-1)

    # Active cooling jacket parameters
    cooling_active: bool = False
    T_cool_in_C: float = 15.0  # °C, coolant inlet temperature
    UA_jacket_W_K: float = 10.0 # W/K, heat exchanger UA value
    T_target_C: float = 50.0    # °C, target used for the cooling-duty calculation


def evaporative_heat_loss_W(T_C: float, T_amb_C: float, A_surf_m2: float, RH: float) -> float:
    """Estimate evaporative cooling loss (W) from open cell top."""
    if T_C <= T_amb_C:
        return 0.0
    # Antoine equation for saturation vapor pressure of water (kPa)
    p_sat_T = 0.133322 * math.exp(18.3036 - 3816.44 / (T_C + 227.02))
    p_sat_amb = 0.133322 * math.exp(18.3036 - 3816.44 / (T_amb_C + 227.02))
    p_v_amb = RH * p_sat_amb

    # Mass transfer coefficient km ~ 0.015 m/s for free convection over liquid
    # Evaporation rate m_dot (g/s) = km * A * (rho_v_sat - rho_v_amb)
    # Vapor density rho_v ~ p_v / (R_spec * T_K) with R_spec = 0.4615 kJ/(kg*K)
    T_K = T_C + 273.15
    d_pv = max(0.0, p_sat_T - p_v_amb) # kPa
    m_dot_g_s = 0.015 * A_surf_m2 * (d_pv / (0.4615 * T_K)) * 1000.0 # g/s

    Q_evap = m_dot_g_s * H_FG_WATER # W
    return max(0.0, float(Q_evap))


def simulate_thermal_transient(p: CellThermalParams, t_end_hr: float = 4.0, dt_s: float = 5.0) -> Dict[str, Any]:
    """Simulate transient electrolyte temperature T(t) over time.
    
    Heat balance:
    C_total * dT/dt = Q_gen - Q_amb - Q_evap - Q_jacket
    
    Q_gen = I * (V_cell - E_therm)
    """
    total_steps = int(t_end_hr * 3600.0 / dt_s)
    t_array = np.linspace(0.0, t_end_hr, total_steps)
    T_array = np.zeros(total_steps)
    Q_gen_array = np.zeros(total_steps)
    Q_amb_array = np.zeros(total_steps)
    Q_evap_array = np.zeros(total_steps)
    Q_jacket_array = np.zeros(total_steps)

    # Total cell thermal mass (J/K)
    C_elec = p.volume_L * 1000.0 * RHO_WATER * CP_WATER
    C_total = C_elec + p.hardware_C_J_K

    # Heat generation (W)
    Q_gen = max(0.0, p.current_A * (p.V_cell - E_THERM_FE))

    T_curr = p.T_init_C

    for i in range(total_steps):
        T_array[i] = T_curr
        Q_gen_array[i] = Q_gen

        Q_amb = p.UA_amb_W_K * (T_curr - p.T_amb_C)
        Q_evap = evaporative_heat_loss_W(T_curr, p.T_amb_C, p.A_surface_m2, p.relative_humidity)

        if p.cooling_active:
            Q_jacket = p.UA_jacket_W_K * (T_curr - p.T_cool_in_C)
        else:
            Q_jacket = 0.0

        Q_amb_array[i] = Q_amb
        Q_evap_array[i] = Q_evap
        Q_jacket_array[i] = Q_jacket

        dT_dt = (Q_gen - Q_amb - Q_evap - Q_jacket) / C_total
        T_curr += dT_dt * dt_s

    # Steady state equilibrium temperature
    T_ss = T_array[-1]

    # Cooling duty required to maintain the configured target temperature.
    # Keep the historical 50 °C value as a compatibility/reporting field;
    # integrated reference-cell runs use T_target_C from their design basis.
    T_target = float(p.T_target_C)
    Q_amb_target = p.UA_amb_W_K * (T_target - p.T_amb_C)
    Q_evap_target = evaporative_heat_loss_W(T_target, p.T_amb_C, p.A_surface_m2, p.relative_humidity)
    Q_cool_req_W = max(0.0, Q_gen - Q_amb_target - Q_evap_target)
    T_legacy = 50.0
    Q_amb_legacy = p.UA_amb_W_K * (T_legacy - p.T_amb_C)
    Q_evap_legacy = evaporative_heat_loss_W(
        T_legacy, p.T_amb_C, p.A_surface_m2, p.relative_humidity
    )
    Q_cool_legacy_W = max(0.0, Q_gen - Q_amb_legacy - Q_evap_legacy)

    return {
        "time_hr": t_array,
        "temperature_C": T_array,
        "Q_gen_W": Q_gen_array,
        "Q_amb_W": Q_amb_array,
        "Q_evap_W": Q_evap_array,
        "Q_jacket_W": Q_jacket_array,
        "T_ss_C": float(T_ss),
        "T_max_C": float(np.max(T_array)),
        "T_target_C": T_target,
        "cooling_duty_target_W": float(Q_cool_req_W),
        "cooling_duty_50C_W": float(Q_cool_legacy_W),
        "thermal_mass_kJ_K": float(C_total / 1000.0),
        "heat_gen_power_W": float(Q_gen),
    }
