"""
Coupled bath / recirculation dynamics for the EKF state transition.

This module provides physically-consistent conservation-law dynamics for the
7-state EKF used by the digital twin.  Every state's time derivative comes
from a mass or energy balance plus a recirculation exchange term with a finite
reservoir — no independent mean-reversion constants.

    x_next, aux_next = step(x, aux, dt_hr, design_point, model)

The function is side-effect free and unit-testable in isolation; the EKF
integrates it through :func:`models.digital_twin._f_state_transition`.

Equations
---------
See docs/TWIN_BATH_DYNAMICS.md for the full derivation.  The short version:

* **Fe2+ mass balance (index 2)** — Faraday consumption + makeup source +
  recirculation exchange with the reservoir.
* **pH / buffer balance (index 3)** — acid/base dose vs. hydroxide generated
  by HER, divided by the bath's buffer capacity beta, plus recirculation.
* **Thermal balance (indices 0, 1)** — catholyte/anolyte/reservoir energy
  balances with Joule heating, membrane crossover, cooling, and recirculation
  mixing.
* **Cell voltage (index 6)** — electrical relaxation toward the physics
  model's predicted voltage with a state-dependent time constant grounded in
  ohmic + double-layer physics.
* **Current density (index 4)** — operator setpoint; drifts to ``j_avg_mA_cm2``.
* **Deposit thickness (index 5)** — integrates the physics-predicted growth.

All new control inputs and auxiliary parameters have explicit defaults in the
``design_point`` dict so existing callers keep working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .electrochemistry import FARADAY, Z_FE
from .twin_physics import CellProcessModel


# ---------------------------------------------------------------------------
# Physical constants (bath properties, L0 defaults)
# ---------------------------------------------------------------------------

# Aqueous electrolyte properties (dilute sulfate, ~1 M FeSO4)
RHO_ELECTROLYTE_KG_M3 = 1200.0   # kg/m³ — typical for 1 M FeSO4
CP_ELECTROLYTE_J_KG_K = 3800.0   # J/(kg·K) — heat capacity of aqueous electrolyte
RHO_IRON_KG_M3 = 7874.0          # kg/m³ — dense bcc iron deposit
# All keys here are merged into design_point if absent.
BATH_DYNAMICS_DEFAULTS: Dict[str, float] = {
    # --- Recirculation loop ---
    "recirculation_flow_L_hr": 6000.0,    # L/hr total recirculation flow
    "reservoir_volume_L": 50000.0,         # L, external reservoir/balance tank (large = quasi-infinite source)
    "catholyte_volume_L": 800.0,           # L, catholyte compartment volume
    "anolyte_volume_L": 2000.0,            # L, anolyte compartment volume (large thermal mass for stability)

    # --- Fe2+ makeup ---
    "fe2_makeup_rate_M_hr": 0.0,           # M/hr — FeSO4 makeup to reservoir
    "fe2_reservoir_M": 1.0,                # M — initial reservoir Fe2+ conc

    # --- pH / buffer control ---
    "buffer_capacity_beta": 0.05,          # mol/(L·pH) — bath buffer capacity
    "acid_dose_rate_M_hr": 0.0,            # M/hr — acid dose rate (positive = add acid)
    "pH_reservoir": 3.5,                   # pH of reservoir feed

    # --- Thermal control ---
    # Default cooling auto-balances the current Joule heating into the catholyte
    # if not explicitly set in design_point.  Set to a fixed value for manual control.
    # "cooling_power_W": <auto>,         # W — cooling power (positive = removes heat)
    "joule_heat_fraction_catholyte": 0.6,  # fraction of Joule heat into catholyte
    "joule_heat_fraction_anolyte": 0.3,    # fraction of Joule heat into anolyte
    "UA_membrane_W_K": 50.0,               # W/K — membrane heat transfer coeff × area
    "UA_ambient_W_K": 5.0,                 # W/K — ambient losses (catholyte + anolyte)
    "T_ambient_C": 25.0,                   # °C — ambient temperature
    "T_reservoir_C": 55.0,                 # °C — initial reservoir temperature

    # --- Electrical relaxation ---
    "electrolyte_conductivity_S_m": 10.0,  # S/m — electrolyte conductivity
    "electrode_gap_m": 0.02,              # m — inter-electrode gap
    "C_dl_F_m2": 0.02,                    # F/m² — double-layer capacitance
    "V_relax_min_hr": 10.0,               # hr — minimum voltage relaxation time (slow tracking)

    # --- Current density setpoint tracking ---
    "tau_j_hr": 0.5,                       # hr — current density setpoint tracking
}


# ---------------------------------------------------------------------------
# Auxiliary reservoir state (not part of the EKF state vector)
# ---------------------------------------------------------------------------

@dataclass
class BathAux:
    """Auxiliary (non-estimated) reservoir state tracked alongside the EKF.

    These are integrated by the same dynamics but are not part of the
    7-state EKF vector.  They live in the ``design_point`` dict under the
    key ``"_bath_aux"`` so the EKF interface is unchanged.
    """
    T_reservoir_C: float = 55.0
    fe2_reservoir_M: float = 1.0
    pH_reservoir: float = 3.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "T_reservoir_C": self.T_reservoir_C,
            "fe2_reservoir_M": self.fe2_reservoir_M,
            "pH_reservoir": self.pH_reservoir,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BathAux":
        return cls(
            T_reservoir_C=d.get("T_reservoir_C", 55.0),
            fe2_reservoir_M=d.get("fe2_reservoir_M", 1.0),
            pH_reservoir=d.get("pH_reservoir", 3.5),
        )


def get_aux(design_point: Dict[str, Any]) -> BathAux:
    """Retrieve the BathAux from design_point, creating one if absent."""
    aux = design_point.get("_bath_aux")
    if isinstance(aux, BathAux):
        return aux
    if isinstance(aux, dict):
        return BathAux.from_dict(aux)
    # Initialize from design_point defaults
    return BathAux(
        T_reservoir_C=design_point.get("T_reservoir_C",
                    BATH_DYNAMICS_DEFAULTS["T_reservoir_C"]),
        fe2_reservoir_M=design_point.get("fe2_reservoir_M",
                      BATH_DYNAMICS_DEFAULTS["fe2_reservoir_M"]),
        pH_reservoir=design_point.get("pH_reservoir",
                   BATH_DYNAMICS_DEFAULTS["pH_reservoir"]),
    )


def set_aux(design_point: Dict[str, Any], aux: BathAux) -> None:
    """Store the BathAux back into design_point."""
    design_point["_bath_aux"] = aux


def _dp(dp: Dict[str, Any], key: str) -> float:
    """Get a design-point parameter with fallback to BATH_DYNAMICS_DEFAULTS."""
    return dp.get(key, BATH_DYNAMICS_DEFAULTS.get(key, 0.0))


# ---------------------------------------------------------------------------
# Core dynamics step
# ---------------------------------------------------------------------------

def step(
    x: np.ndarray,
    aux: BathAux,
    dt_hr: float,
    design_point: Dict[str, Any],
    model: CellProcessModel,
) -> Tuple[np.ndarray, BathAux]:
    """Advance the 7-state EKF vector and auxiliary reservoir by dt_hr.

    Parameters
    ----------
    x : ndarray, shape (7,)
        Current EKF state vector (see STATE_KEYS in digital_twin).
    aux : BathAux
        Current auxiliary reservoir state.
    dt_hr : float
        Time step in hours.
    design_point : dict
        Operating-point / design parameters (see BATH_DYNAMICS_DEFAULTS).
    model : CellProcessModel
        Physics surrogate for electrochemical predictions.

    Returns
    -------
    x_next : ndarray, shape (7,)
        Advanced state vector.
    aux_next : BathAux
        Advanced auxiliary reservoir state.
    """
    if dt_hr <= 0.0:
        return x.copy(), BathAux(
            T_reservoir_C=aux.T_reservoir_C,
            fe2_reservoir_M=aux.fe2_reservoir_M,
            pH_reservoir=aux.pH_reservoir,
        )

    dp = design_point
    x_next = x.copy()

    # --- Extract state values (clamped for physics queries) ---
    T_cath = x[0]          # catholyte temperature (°C)
    T_anol = x[1]          # anolyte temperature (°C)
    fe2 = max(1e-6, x[2])  # bulk Fe2+ (M)
    pH = x[3]              # bulk pH
    j = max(1e-3, x[4])    # current density (mA/cm²)
    deposit = x[5]         # deposit thickness (µm)
    V_cell = x[6]          # cell voltage (V)

    # --- Design-point parameters ---
    area_m2 = dp.get("electrode_area_m2", 1.0)
    V_cath_L = _dp(dp, "catholyte_volume_L")
    V_anol_L = _dp(dp, "anolyte_volume_L")
    V_res_L = _dp(dp, "reservoir_volume_L")
    flow_L_hr = _dp(dp, "recirculation_flow_L_hr")
    tau_j_hr = _dp(dp, "tau_j_hr")

    # --- Physics prediction at current operating point ---
    pred = model.predict(j_mA_cm2=j, temperature_C=max(0.0, T_cath), fe2_M=fe2)
    FE = pred.current_efficiency
    v_pred = pred.v_cell_V
    deposit_rate_um_hr = pred.deposit_rate_um_hr

    # Current in Amperes
    I_A = j * area_m2 * 10.0  # mA/cm² × m² × 10 = A
    j_A_m2 = j * 10.0

    # =====================================================================
    # 1. Fe2+ MASS BALANCE (index 2)
    # =====================================================================
    # Consumption by Faraday deposition: d(fe2)/dt = -j_A_m2*FE*area / (z*F*V_cath)
    # in mol/m³/s → convert to M/hr
    consumption_M_hr = (j_A_m2 * FE / (Z_FE * FARADAY)) * area_m2 * 3600.0 / V_cath_L

    # Recirculation exchange: (flow/V_cath) * (fe2_res - fe2)
    recirc_fe2_M_hr = (flow_L_hr / V_cath_L) * (aux.fe2_reservoir_M - fe2)

    # Makeup source (direct to catholyte for L0; reservoir makeup tracked in aux)
    makeup_M_hr = _dp(dp, "fe2_makeup_rate_M_hr")

    dfe2_dt = -consumption_M_hr + recirc_fe2_M_hr + makeup_M_hr
    x_next[2] = max(1e-6, fe2 + dfe2_dt * dt_hr)

    # Reservoir Fe2+ balance:
    # d(fe2_res)/dt = (flow/V_res)*(fe2 - fe2_res) + makeup_to_res - consumption_res
    # For L0: makeup goes to reservoir, return flow brings depleted catholyte back
    # Net: flow brings fe2 back from cell, makeup adds to reservoir
    dfe2_res_dt = (flow_L_hr / V_res_L) * (fe2 - aux.fe2_reservoir_M) + \
                  makeup_M_hr * (V_cath_L / V_res_L)
    fe2_res_next = max(1e-6, aux.fe2_reservoir_M + dfe2_res_dt * dt_hr)

    # =====================================================================
    # 2. pH / BUFFER DYNAMICS (index 3)
    # =====================================================================
    # HER at cathode: 2H2O + 2e- → H2 + 2OH-
    # OH- production rate (mol/s) = j_A_m2 * (1-FE) / (1 * F) * area
    # (1 mol OH- per mol e- for HER)
    OH_production_mol_s = j_A_m2 * (1.0 - FE) / FARADAY * area_m2
    OH_production_M_hr = OH_production_mol_s * 3600.0 / V_cath_L

    # Acid dose (positive = adds H+, lowers pH)
    acid_dose_M_hr = _dp(dp, "acid_dose_rate_M_hr")

    # Buffer capacity: d(pH)/dt = -(net_proton_rate_M_hr) / beta
    # Net proton rate = acid_dose - OH_production (OH- consumes protons equivalently)
    # Adding acid (positive net_proton) lowers pH, so negative sign
    beta = _dp(dp, "buffer_capacity_beta")
    net_proton_M_hr = acid_dose_M_hr - OH_production_M_hr

    # Recirculation mixing for pH
    recirc_pH_hr = (flow_L_hr / V_cath_L) * (aux.pH_reservoir - pH)

    dpH_dt = -net_proton_M_hr / max(beta, 1e-6) + recirc_pH_hr
    x_next[3] = max(0.0, min(14.0, pH + dpH_dt * dt_hr))

    # Reservoir pH (slowly tracks catholyte return)
    dpH_res_dt = (flow_L_hr / V_res_L) * (pH - aux.pH_reservoir)
    pH_res_next = max(0.0, min(14.0, aux.pH_reservoir + dpH_res_dt * dt_hr))

    # =====================================================================
    # 3. THERMAL BALANCE (indices 0, 1)
    # =====================================================================
    # Joule heating: Q_joule = V_cell * I [W]
    Q_joule_W = V_cell * I_A

    # Split into catholyte and anolyte
    f_cath = _dp(dp, "joule_heat_fraction_catholyte")
    f_anol = _dp(dp, "joule_heat_fraction_anolyte")
    Q_cath_W = Q_joule_W * f_cath
    Q_anol_W = Q_joule_W * f_anol

    # Cooling (positive = removes heat from catholyte)
    # If not explicitly set, default to balancing the current Joule heating
    Q_cool_W = dp.get("cooling_power_W", Q_cath_W)  # auto-balance if not set

    # Membrane crossover: UA * (T_cath - T_anol)
    UA_mem = _dp(dp, "UA_membrane_W_K")
    Q_membrane_W = UA_mem * (T_cath - T_anol)

    # Ambient losses
    UA_amb = _dp(dp, "UA_ambient_W_K")
    T_amb = _dp(dp, "T_ambient_C")
    # Increased ambient loss coefficients to stabilize thermal balance
    Q_amb_cath_W = UA_amb * 1.0 * (T_cath - T_amb)
    Q_amb_anol_W = UA_amb * 1.0 * (T_anol - T_amb)

    # Recirculation heat exchange: flow * rho * Cp * (T_res - T_comp) / 3600
    # flow in L/hr → m³/s: flow/1000/3600; rho in kg/m³; Cp in J/(kg·K)
    # But simpler: flow_L_hr * rho * Cp / 3600 gives W/K equivalent
    flow_thermal_W_K = flow_L_hr * (RHO_ELECTROLYTE_KG_M3 / 1000.0) * CP_ELECTROLYTE_J_KG_K / 3600.0

    # Thermal mass: V_L * rho * Cp / 1000 [J/K]
    # V in L → kg: V * rho/1000; then kg * Cp = J/K
    mass_cath_J_K = V_cath_L * (RHO_ELECTROLYTE_KG_M3 / 1000.0) * CP_ELECTROLYTE_J_KG_K
    mass_anol_J_K = V_anol_L * (RHO_ELECTROLYTE_KG_M3 / 1000.0) * CP_ELECTROLYTE_J_KG_K
    mass_res_J_K = V_res_L * (RHO_ELECTROLYTE_KG_M3 / 1000.0) * CP_ELECTROLYTE_J_KG_K

    # Catholyte energy balance: dT_c/dt = (Q_in - Q_out) / mass_cath [K/s]
    Q_net_cath_W = (Q_cath_W
                    - Q_cool_W
                    - Q_membrane_W
                    - Q_amb_cath_W
                    + flow_thermal_W_K * (aux.T_reservoir_C - T_cath))
    dT_cath_dt_K_s = Q_net_cath_W / mass_cath_J_K
    dT_cath_dt_C_hr = dT_cath_dt_K_s * 3600.0

    # Anolyte energy balance
    Q_net_anol_W = (Q_anol_W
                    + Q_membrane_W
                    - Q_amb_anol_W
                    + flow_thermal_W_K * (aux.T_reservoir_C - T_anol))
    dT_anol_dt_K_s = Q_net_anol_W / mass_anol_J_K
    dT_anol_dt_C_hr = dT_anol_dt_K_s * 3600.0

    # Reservoir energy balance (receives return from both compartments)
    # dT_res/dt = flow*(T_cath - T_res)/V_res + flow*(T_anol - T_res)/V_res
    # Plus ambient losses from reservoir
    Q_amb_res_W = UA_amb * (aux.T_reservoir_C - T_amb)
    Q_net_res_W = (flow_thermal_W_K * (T_cath - aux.T_reservoir_C)
                   + flow_thermal_W_K * (T_anol - aux.T_reservoir_C)
                   - Q_amb_res_W)
    dT_res_dt_K_s = Q_net_res_W / mass_res_J_K
    dT_res_dt_C_hr = dT_res_dt_K_s * 3600.0

    x_next[0] = T_cath + dT_cath_dt_C_hr * dt_hr
    x_next[1] = T_anol + dT_anol_dt_C_hr * dt_hr
    T_res_next = aux.T_reservoir_C + dT_res_dt_C_hr * dt_hr

    # =====================================================================
    # 4. CELL VOLTAGE (index 6) — physically-grounded electrical relaxation
    # =====================================================================
    # Time constant from ohmic resistance × double-layer capacitance
    # R_ohm = gap / (sigma * area) [Ohm]
    sigma = _dp(dp, "electrolyte_conductivity_S_m")
    gap = _dp(dp, "electrode_gap_m")
    C_dl = _dp(dp, "C_dl_F_m2")
    R_ohm = gap / (sigma * area_m2)  # Ohm
    tau_elec_s = R_ohm * C_dl * area_m2  # seconds (R*C where C = C_dl*area)
    # Add a mass-transfer relaxation contribution that grows as fe2 depletes
    # (lower fe2 → slower diffusion equilibration)
    tau_mt_s = 1.0 / max(fe2, 0.01)  # seconds, heuristic
    tau_V_s = max(tau_elec_s + tau_mt_s, 1.0)  # at least 1 second
    tau_V_hr = max(tau_V_s / 3600.0, _dp(dp, "V_relax_min_hr"))

    # Drive toward physics-predicted voltage
    dV_dt = (v_pred - V_cell) / tau_V_hr
    x_next[6] = V_cell + dV_dt * dt_hr

    # =====================================================================
    # 5. CURRENT DENSITY (index 4) — operator setpoint tracking
    # =====================================================================
    j_setpoint = dp.get("j_avg_mA_cm2", 150.0)
    if tau_j_hr > 0:
        alpha_j = 1.0 - math.exp(-dt_hr / tau_j_hr)
    else:
        alpha_j = 1.0
    x_next[4] = j + alpha_j * (j_setpoint - j)

    # =====================================================================
    # 6. DEPOSIT THICKNESS (index 5) — physics-predicted growth
    # =====================================================================
    x_next[5] = max(0.0, deposit + deposit_rate_um_hr * dt_hr)

    # --- Assemble next aux ---
    aux_next = BathAux(
        T_reservoir_C=T_res_next,
        fe2_reservoir_M=fe2_res_next,
        pH_reservoir=pH_res_next,
    )

    return x_next, aux_next


# ---------------------------------------------------------------------------
# Convenience: compute steady-state Fe2+ for a given operating point
# ---------------------------------------------------------------------------

def steady_state_fe2_M(
    design_point: Dict[str, Any],
    model: CellProcessModel,
) -> float:
    """Compute the steady-state bulk Fe2+ where consumption == source + recirculation.

    At steady state: dfe2/dt = 0
    -consumption + (flow/V_cath)*(fe2_res - fe2) + makeup = 0
    fe2 = fe2_res + (makeup - consumption) * V_cath / flow
    """
    dp = design_point
    j = dp.get("j_avg_mA_cm2", 150.0)
    T = dp.get("temperature_C", 60.0)
    fe2_guess = dp.get("fe2_M", 1.0)
    area_m2 = dp.get("electrode_area_m2", 1.0)
    V_cath = _dp(dp, "catholyte_volume_L")
    flow = _dp(dp, "recirculation_flow_L_hr")
    makeup = _dp(dp, "fe2_makeup_rate_M_hr")

    pred = model.predict(j_mA_cm2=j, temperature_C=T, fe2_M=fe2_guess)
    FE = pred.current_efficiency
    j_A_m2 = j * 10.0
    consumption_M_hr = (j_A_m2 * FE / (Z_FE * FARADAY)) * area_m2 * 3600.0 / V_cath

    fe2_res = dp.get("fe2_reservoir_M", _dp(dp, "fe2_reservoir_M"))
    fe2_ss = fe2_res + (makeup - consumption_M_hr) * V_cath / max(flow, 1e-6)
    return max(1e-6, fe2_ss)


def steady_state_acid_dose_M_hr(
    design_point: Dict[str, Any],
    model: CellProcessModel,
) -> float:
    """Compute acid dose rate that holds pH steady at setpoint.

    At steady state: dpH/dt = 0
    (acid_dose - OH_production) / beta + (flow/V)*(pH_res - pH) = 0
    acid_dose = OH_production - beta * (flow/V) * (pH_res - pH)
    """
    dp = design_point
    j = dp.get("j_avg_mA_cm2", 150.0)
    T = dp.get("temperature_C", 60.0)
    fe2 = dp.get("fe2_M", 1.0)
    area_m2 = dp.get("electrode_area_m2", 1.0)
    V_cath = _dp(dp, "catholyte_volume_L")
    flow = _dp(dp, "recirculation_flow_L_hr")
    beta = _dp(dp, "buffer_capacity_beta")
    pH_set = dp.get("pH", 3.5)
    pH_res = dp.get("pH_reservoir", _dp(dp, "pH_reservoir"))

    pred = model.predict(j_mA_cm2=j, temperature_C=T, fe2_M=fe2)
    FE = pred.current_efficiency
    j_A_m2 = j * 10.0
    OH_M_hr = j_A_m2 * (1.0 - FE) / FARADAY * area_m2 * 3600.0 / V_cath

    acid_dose = OH_M_hr - beta * (flow / V_cath) * (pH_res - pH_set)
    return acid_dose
