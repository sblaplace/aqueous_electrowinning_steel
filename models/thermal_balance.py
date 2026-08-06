"""
Cell heat balance and thermal management for aqueous electrowinning.

The previous version used a constant thermoneutral voltage
(``E_THERM_FE = 1.28 V``) independent of pH, composition, temperature,
and — most importantly — Faradaic efficiency, so a cell running 100% HER
generated the same heat as one plating iron.  The 2026-08 revision makes
heat generation respect the actual electrochemistry:

    Q_gen = I·V_cell − FE·I·E_therm,Fe − (1−FE)·I·E_therm,HER

i.e. each ampere-second is split between Fe deposition (its reversible
heat, set by ΔH for Fe²⁺+2e⁻→Fe) and the HER side reaction (ΔH for
2H⁺+2e⁻→H₂, which is essentially zero over liquid water), with all
overpotentials and ohmic drops always dissipated as heat.  ``E_therm``
also carries a small temperature coefficient (dE/dT from the reaction
entropy), so cooling demand changes with bath temperature.

All legacy fields (``hardware_C_J_K``, ``UA_amb_W_K``,
``A_surface_m2``, ``relative_humidity``) and the constant-V construction
path are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

# ─── Physical constants ──────────────────────────────────────────────
E_THERM_FE = 1.28       # V, thermoneutral for Fe²⁺+2e⁻→Fe (legacy default)
E_THERM_HER = 0.0       # V, 2H⁺+2e⁻→H₂ over liquid water (ΔH≈0)
CP_WATER = 4.184        # J/(g·K)
RHO_WATER = 1.0         # kg/L
H_FG_WATER = 2260.0     # J/g latent heat of vaporization

# Temperature coefficient of E_therm,Fe (V/K).  From the reaction entropy
# (ΔS ≈ −20 to −40 J/(mol·K) for Fe deposition); small but non-zero.
DETHERM_DT_FE = -2.0e-4
T_REF = 298.15


def thermoneutral_voltage_Fe(T_C: float, E_ref: float = E_THERM_FE) -> float:
    """E_therm for Fe deposition at T_C, with dE/dT correction."""
    return E_ref + DETHERM_DT_FE * (T_C + 273.15 - T_REF)


def effective_thermoneutral_voltage(
    current_efficiency: float,
    T_C: float = 60.0,
    E_therm_fe: float = E_THERM_FE,
) -> float:
    """FE-weighted thermoneutral voltage (V)."""
    fe = min(max(current_efficiency, 0.0), 1.0)
    e_fe = thermoneutral_voltage_Fe(T_C, E_therm_fe)
    return fe * e_fe + (1.0 - fe) * E_THERM_HER


@dataclass
class CellThermalParams:
    """Parameters for one cell's lumped-capacitance heat balance.

    All legacy field names are accepted.  New in 2026-08:
    ``current_efficiency`` / ``E_therm_fe`` / ``auxiliary_heat_W`` for
    chemistry-aware heat generation.
    """

    V_cell: float = 2.5
    current_A: float = 10.0
    volume_L: float = 2.0
    hardware_C_J_K: float = 500.0

    T_init_C: float = 25.0
    T_amb_C: float = 22.0
    UA_amb_W_K: float = 1.5

    A_surface_m2: float = 0.04
    relative_humidity: float = 0.50

    cooling_active: bool = False
    T_cool_in_C: float = 15.0
    UA_jacket_W_K: float = 10.0
    T_target_C: float = 50.0

    # 2026-08 chemistry-aware generation (None → legacy constant E_THERM_FE)
    current_efficiency: Optional[float] = None
    E_therm_fe: float = E_THERM_FE
    auxiliary_heat_W: float = 0.0


def evaporative_heat_loss_W(
    T_C: float, T_amb_C: float, A_surface_m2: float, RH: float
) -> float:
    """Evaporative cooling from the open electrolyte surface (W).

    Antoine vapor pressure + a free-convection mass-transfer coefficient,
    matching the pre-2026 screening correlation.
    """
    if T_C <= T_amb_C:
        return 0.0
    p_sat_T = 0.133322 * np.exp(18.3036 - 3816.44 / (T_C + 227.02))
    p_sat_amb = 0.133322 * np.exp(18.3036 - 3816.44 / (T_amb_C + 227.02))
    p_v_amb = RH * p_sat_amb
    T_K = T_C + 273.15
    d_pv = max(0.0, p_sat_T - p_v_amb)  # kPa
    m_dot_g_s = 0.015 * A_surface_m2 * (d_pv / (0.4615 * T_K)) * 1000.0
    return max(0.0, float(m_dot_g_s * H_FG_WATER))


def heat_generation_W(
    V_cell: float,
    current_A: float,
    current_efficiency: Optional[float] = None,
    T_C: float = 60.0,
    E_therm_fe: float = E_THERM_FE,
    auxiliary_heat_W: float = 0.0,
) -> float:
    """Net irreversible heat released into the bath (W).

    With ``current_efficiency=None`` this reproduces the legacy
    ``I·(V − E_THERM_FE)``.  Otherwise each reaction carries its own
    thermoneutral voltage, so HER-dominated operation correctly produces
    *more* heat (its reversible heat is ~0) than Fe-plating operation.
    """
    elec_in = V_cell * current_A
    if current_efficiency is None:
        e_therm = E_THERM_FE
    else:
        e_therm = effective_thermoneutral_voltage(
            current_efficiency, T_C, E_therm_fe
        )
    reversible_power = e_therm * current_A
    return float(max(elec_in - reversible_power + auxiliary_heat_W, 0.0))


def _total_heat_capacity(p: CellThermalParams) -> float:
    C_elec = p.volume_L * 1000.0 * RHO_WATER * CP_WATER
    return C_elec + p.hardware_C_J_K


def simulate_thermal_transient(
    p: CellThermalParams, t_end_hr: float = 4.0, dt_s: float = 5.0
) -> Dict[str, Any]:
    """Integrate the lumped heat balance forward in time.

    Heat generation now uses the FE-weighted thermoneutral voltage when
    ``p.current_efficiency`` is set; all legacy fields and return keys are
    preserved.
    """
    total_steps = max(1, int(round(t_end_hr * 3600.0 / dt_s)))
    t_array = np.linspace(0.0, t_end_hr, total_steps)
    T_array = np.zeros(total_steps)
    Q_gen_array = np.zeros(total_steps)
    Q_amb_array = np.zeros(total_steps)
    Q_evap_array = np.zeros(total_steps)
    Q_jacket_array = np.zeros(total_steps)

    C_total = _total_heat_capacity(p)
    T_curr = p.T_init_C

    for i in range(total_steps):
        T_array[i] = T_curr
        Q_gen = heat_generation_W(
            p.V_cell, p.current_A, p.current_efficiency,
            T_curr, p.E_therm_fe, p.auxiliary_heat_W,
        )
        Q_gen_array[i] = Q_gen
        Q_amb = p.UA_amb_W_K * (T_curr - p.T_amb_C)
        Q_evap = evaporative_heat_loss_W(
            T_curr, p.T_amb_C, p.A_surface_m2, p.relative_humidity
        )
        Q_amb_array[i] = Q_amb
        Q_evap_array[i] = Q_evap

        if p.cooling_active:
            Q_jacket = p.UA_jacket_W_K * max(T_curr - p.T_cool_in_C, 0.0)
        else:
            Q_jacket = 0.0
        Q_jacket_array[i] = Q_jacket

        dT_dt = (Q_gen - Q_amb - Q_evap - Q_jacket) / C_total
        T_curr += dT_dt * dt_s

    T_ss = float(T_array[-1])

    # Cooling duties at the target and at the legacy 50 °C point.
    def _cool_duty(T_target: float) -> float:
        q_gen = heat_generation_W(
            p.V_cell, p.current_A, p.current_efficiency,
            T_target, p.E_therm_fe, p.auxiliary_heat_W,
        )
        q_amb = p.UA_amb_W_K * (T_target - p.T_amb_C)
        q_evap = evaporative_heat_loss_W(
            T_target, p.T_amb_C, p.A_surface_m2, p.relative_humidity
        )
        return float(max(q_gen - q_amb - q_evap, 0.0))

    return {
        "time_hr": t_array,
        "temperature_C": T_array,
        "Q_gen_W": Q_gen_array,
        "Q_amb_W": Q_amb_array,
        "Q_evap_W": Q_evap_array,
        "Q_jacket_W": Q_jacket_array,
        "T_ss_C": T_ss,
        "T_max_C": float(np.max(T_array)),
        "T_target_C": float(p.T_target_C),
        "cooling_duty_target_W": _cool_duty(float(p.T_target_C)),
        "cooling_duty_50C_W": _cool_duty(50.0),
        "thermal_mass_kJ_K": float(C_total / 1000.0),
        "heat_gen_power_W": float(Q_gen_array[0]),
    }
