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
* **Fe³⁺ redox shuttle (optional CSTR extension, 2026-08)** — with
  ``fe3_shuttle_enabled`` on, three extra auxiliary states carry the
  production → shuttle | sludge triangle of ``models/fe3_shuttle.py`` in
  time-integrated form: dissolved Fe³⁺ in catholyte and reservoir, plus a
  cumulative Fe(OH)₃ sludge ledger.  Back-couplings: autoxidation drains the
  Fe²⁺ balance, the cathodic shuttle returns it (and steals
  ``i_sh = F·k_m·[Fe³⁺]`` of applied current from deposit growth and the
  HER/OH⁻ split), and the net +2 H⁺ per mol of sludge loads the pH balance.
  Off by default; when off every added term is exactly 0.0/identity and all
  pre-existing results are byte-identical.

All new control inputs and auxiliary parameters have explicit defaults in the
``design_point`` dict so existing callers keep working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .bath_startup import dissolved_o2_saturation_mol_L, fe2_oxidation_rate
from .electrochemistry import FARADAY, Z_FE
from .fe3_shuttle import D_FE3_REF_M2_S, fe3_solubility_cap_M
from .ferric_hydroxide_phases import (
    age_inventory,
    blended_cap_M,
    initial_phase_for_bath,
)
from .twin_physics import CellProcessModel
from .env_coupling import DisturbanceInputs
from .water_drag import water_volume_flux_L_m2_hr


# ---------------------------------------------------------------------------
# Physical constants (bath properties, L0 defaults)
# ---------------------------------------------------------------------------

# Aqueous electrolyte properties (dilute sulfate, ~1 M FeSO4)
RHO_ELECTROLYTE_KG_M3 = 1200.0   # kg/m³ — typical for 1 M FeSO4
CP_ELECTROLYTE_J_KG_K = 3800.0   # J/(kg·K) — heat capacity of aqueous electrolyte
RHO_IRON_KG_M3 = 7874.0          # kg/m³ — dense bcc iron deposit
# All keys here are merged into design_point if absent.
BATH_DYNAMICS_DEFAULTS: Dict[str, Any] = {
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
    "temperature_control_gain_W_K": 3000.0, # W/K — heater/chiller holding the operating setpoint
    "pH_control_gain_M_hr_ph": 0.05,       # (M/hr) gentle acid-dose feedback (≈1 h pH loop)
    "heat_exchange_area_m2": 10.0,         # m² — exposed surface for convective/rain cooling
    "T_reservoir_C": 55.0,                 # °C — initial reservoir temperature

    # --- Fe3+ redox shuttle (CSTR extension; OFF by default → byte-identical) ---
    # Chemistry, steady-state closed form and scenario helpers live in
    # models/fe3_shuttle.py; these knobs wire the same terms into the
    # time-integrated bath dynamics.  Enable via ``apply_fe3_scenario``.
    "fe3_shuttle_enabled": False,          # master switch (byte-identical when off)
    "fe3_o2_fraction_of_sat": 0.005,       # dissolved O2 as fraction of air saturation (sealed cell)
    "fe3_crossover_o2_flux_mol_m2_s": 0.0, # anolyte O2 crossover fault flux (mol/m²/s)
    "fe3_d_m2_s": D_FE3_REF_M2_S,          # Fe3+ diffusivity (screening family)
    "fe3_boundary_layer_m": 50e-6,         # cathode diffusion-layer thickness for Fe3+
    "fe3_k_ox_ref": 1.0e-4,                # autoxidation k_ref, M⁻¹ s⁻¹ (bath_startup screening value)
    "fe3_Ea_ox_J_mol": 50_000.0,           # autoxidation apparent activation energy

    # --- Fe(III) phase-specific sludge (Ostwald aging; OFF by default) ---
    # With ``ferric_phase_aging_enabled`` on, the Fe(III) sludge inventory is
    # aged along the ferrihydrite -> goethite -> hematite (sulfate) or
    # akaganeite -> goethite -> hematite (chloride) ladder instead of being
    # lumped as Fe(OH)3, so the [Fe3+] cap (and hence the H+ release schedule
    # and the sludge bleed) follows the phase actually present.  When off
    # everything above is byte-identical to the legacy Fe(OH)3 model.
    "ferric_phase_aging_enabled": False,
    "bath_anion": "sulfate",               # "sulfate" -> ferrihydrite, "chloride" -> akaganeite
    "ferric_aging_t12_fh_gh_hr": 720.0,    # ferrihydrite/akaganeite -> goethite t1/2 @25C
    "ferric_aging_t12_gh_hem_hr": 8760.0,  # goethite -> hematite t1/2 @25C

    # --- Electrical relaxation ---
    "electrolyte_conductivity_S_m": 10.0,  # S/m — electrolyte conductivity
    "electrode_gap_m": 0.02,              # m — inter-electrode gap
    "C_dl_F_m2": 0.02,                    # F/m² — double-layer capacitance
    "V_relax_min_hr": 0.1,                # hr — voltage relaxation floor (fast; ms-s physically)

    # --- Current density setpoint tracking ---
    "tau_j_hr": 0.5,                       # hr — current density setpoint tracking

    # --- Electro-osmotic water drag (CSTR extension; OFF by default → byte-identical) ---
    # Water crosses the membrane with the H⁺ current (models/water_drag.py),
    # leaving the catholyte and concentrating its non-volatile solutes.  When
    # off every added term is exactly 0.0/identity, so default runs are
    # byte-identical.  ``membrane_area_m2`` defaults to the electrode area.
    "water_drag_enabled": False,
    "membrane_area_m2": None,
}


# ---------------------------------------------------------------------------
# Auxiliary reservoir state (not part of the EKF state vector)
# ---------------------------------------------------------------------------

@dataclass
class BathAux:
    """Auxiliary (non-estimated) bath state tracked alongside the EKF.

    These are integrated by the same dynamics but are not part of the
    7-state EKF vector.  They live in the ``design_point`` dict under the
    key ``"_bath_aux"`` so the EKF interface is unchanged.

    Reservoir fields hold the external balance-tank state; the Fe³⁺ fields
    hold the redox-shuttle CSTR extension (2026-08; zero and untouched
    while ``fe3_shuttle_enabled`` is off):

    * ``fe3_catholyte_M`` — dissolved Fe³⁺ in the catholyte compartment (M);
    * ``fe3_reservoir_M`` — dissolved Fe³⁺ in the reservoir (M);
    * ``fe3_sludge_cumulative_mol`` — total Fe lost to Fe(OH)₃ sludge since
      start (mol), so the iron ledger closes when precipitation is active.
    """
    T_reservoir_C: float = 55.0
    fe2_reservoir_M: float = 1.0
    pH_reservoir: float = 3.5
    fe3_catholyte_M: float = 0.0
    fe3_reservoir_M: float = 0.0
    fe3_sludge_cumulative_mol: float = 0.0
    # Fe(III) phase-specific sludge ledger (Ostwald aging; only touched while
    # ``ferric_phase_aging_enabled`` is on).  ``ferric_phase_inventory`` maps
    # phase name -> mol of Fe held as that solid; ``ferric_phase_initial`` is
    # the phase fresh precipitation lands in for this bath's anion.
    ferric_phase_inventory: Dict[str, float] = None  # type: ignore[assignment]
    ferric_phase_initial: str = "ferrihydrite_2line"
    # Electro-osmotic water drag (CSTR extension; only touched while
    # ``water_drag_enabled`` is on).  ``catholyte_volume_L`` is the running
    # catholyte volume (L) after trans-membrane water loss — None means "use
    # the design-point volume" (the default, drag off).  ``membrane_age_hr``
    # accumulates operating hours for the drag-coefficient aging term.
    catholyte_volume_L: Optional[float] = None
    membrane_age_hr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "T_reservoir_C": self.T_reservoir_C,
            "fe2_reservoir_M": self.fe2_reservoir_M,
            "pH_reservoir": self.pH_reservoir,
            "fe3_catholyte_M": self.fe3_catholyte_M,
            "fe3_reservoir_M": self.fe3_reservoir_M,
            "fe3_sludge_cumulative_mol": self.fe3_sludge_cumulative_mol,
            "ferric_phase_inventory": dict(self.ferric_phase_inventory or {}),
            "ferric_phase_initial": self.ferric_phase_initial,
            "catholyte_volume_L": self.catholyte_volume_L,
            "membrane_age_hr": self.membrane_age_hr,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BathAux":
        inv = d.get("ferric_phase_inventory")
        inv_dict = dict(inv) if isinstance(inv, dict) else {}
        return cls(
            T_reservoir_C=d.get("T_reservoir_C", 55.0),
            fe2_reservoir_M=d.get("fe2_reservoir_M", 1.0),
            pH_reservoir=d.get("pH_reservoir", 3.5),
            fe3_catholyte_M=d.get("fe3_catholyte_M", 0.0),
            fe3_reservoir_M=d.get("fe3_reservoir_M", 0.0),
            fe3_sludge_cumulative_mol=d.get("fe3_sludge_cumulative_mol", 0.0),
            ferric_phase_inventory=inv_dict,
            ferric_phase_initial=str(
                d.get("ferric_phase_initial", "ferrihydrite_2line")),
            catholyte_volume_L=d.get("catholyte_volume_L"),
            membrane_age_hr=d.get("membrane_age_hr", 0.0),
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
        fe3_catholyte_M=design_point.get("fe3_catholyte_M", 0.0),
        fe3_reservoir_M=design_point.get("fe3_reservoir_M", 0.0),
        fe3_sludge_cumulative_mol=design_point.get("fe3_sludge_cumulative_mol", 0.0),
        ferric_phase_inventory=dict(design_point.get("ferric_phase_inventory") or {}),
        ferric_phase_initial=str(design_point.get("ferric_phase_initial",
            initial_phase_for_bath(design_point.get("bath_anion", "sulfate")))),
        catholyte_volume_L=design_point.get("catholyte_volume_L"),
        membrane_age_hr=design_point.get("membrane_age_hr", 0.0),
    )


def set_aux(design_point: Dict[str, Any], aux: BathAux) -> None:
    """Store the BathAux back into design_point."""
    design_point["_bath_aux"] = aux


def _dp(dp: Dict[str, Any], key: str) -> float:
    """Get a design-point parameter with fallback to BATH_DYNAMICS_DEFAULTS."""
    return dp.get(key, BATH_DYNAMICS_DEFAULTS.get(key, 0.0))


def _enabled_env(dp: Dict[str, Any]) -> Optional[DisturbanceInputs]:
    """Return the enabled :class:`DisturbanceInputs` in the design point.

    Returns ``None`` when coupling is absent or disabled, so the calling
    dynamics apply zero disturbance (the brief's default / byte-identical
    coupling-off behaviour).
    """
    env = dp.get("_env_dist")
    if isinstance(env, DisturbanceInputs):
        return env if env.enabled else None
    if isinstance(env, dict):
        d = DisturbanceInputs.from_dict(env)
        return d if d.enabled else None
    return None


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
            fe3_catholyte_M=aux.fe3_catholyte_M,
            fe3_reservoir_M=aux.fe3_reservoir_M,
            fe3_sludge_cumulative_mol=aux.fe3_sludge_cumulative_mol,
            ferric_phase_inventory=dict(aux.ferric_phase_inventory or {}),
            ferric_phase_initial=aux.ferric_phase_initial,
            catholyte_volume_L=aux.catholyte_volume_L,
            membrane_age_hr=aux.membrane_age_hr,
        )

    dp = design_point
    x_next = x.copy()

    # Environmental disturbance (coupling-on only; zero by default)
    env = _enabled_env(dp)

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
    # Electro-osmotic water drag (opt-in).  Running catholyte volume used in
    # the concentration-carrying mass-balance denominators: equal to the design
    # volume when drag is off (byte-identical default), shrinks each step when
    # on as water leaves the catholyte through the membrane.
    water_drag_enabled = bool(_dp(dp, "water_drag_enabled"))
    V_cath_dyn = V_cath_L
    if water_drag_enabled and aux.catholyte_volume_L is not None:
        V_cath_dyn = max(aux.catholyte_volume_L, 0.01 * V_cath_L)
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
    # 0. Fe3+ REDOX SHUTTLE TERMS (optional CSTR extension — OFF by default;
    #    every term below is exactly 0.0/identity when disabled, so default
    #    runs stay byte-identical).  Static steady-state counterpart and
    #    chemistry notes: models/fe3_shuttle.py (production → shuttle | sludge).
    # =====================================================================
    fe3 = max(0.0, aux.fe3_catholyte_M)
    fe3_next = fe3
    fe3_res_next = max(0.0, aux.fe3_reservoir_M)
    sludge_cum_next = max(0.0, aux.fe3_sludge_cumulative_mol)
    i_shuttle_A_m2 = 0.0        # parasitic shuttle current (A/m²), start-of-step
    r_prod_M_hr = 0.0           # Fe3+ production rate (M/hr)
    shuttle_return_M_hr = 0.0   # shuttle flux Fe3+→Fe2+ returned to fe2 (M/hr)
    precip_rate_M_hr = 0.0      # Fe(OH)3 precipitation in catholyte (M/hr)

    # Fe(III) phase-specific sludge (Ostwald aging; untouched while
    # ``ferric_phase_aging_enabled`` is off).  ``phase_inv`` maps phase name ->
    # mol of Fe held as that solid; ``ferric_initial`` is where fresh
    # precipitation lands for this bath's anion.
    ferric_aging_enabled = bool(_dp(dp, "ferric_phase_aging_enabled"))
    bath_anion = str(_dp(dp, "bath_anion"))
    ferric_initial = str(
        dp.get("ferric_phase_initial") or initial_phase_for_bath(bath_anion))
    phase_inv = dict(aux.ferric_phase_inventory or {})
    if ferric_initial not in phase_inv:
        phase_inv[ferric_initial] = 0.0

    fe3_enabled = bool(_dp(dp, "fe3_shuttle_enabled"))
    if fe3_enabled:
        # Shuttle sink mass-transfer coefficient (m/s) and area/volume (1/m),
        # same screened quantities as fe3_shuttle.ShuttleParams.
        km_fe3_m_s = _dp(dp, "fe3_d_m2_s") / max(_dp(dp, "fe3_boundary_layer_m"), 1e-12)
        area_per_vol_1_m = area_m2 / (V_cath_L / 1000.0)
        k_shuttle_1_hr = km_fe3_m_s * area_per_vol_1_m * 3600.0

        # Production: homogeneous autoxidation at the pinned O2 level plus
        # the optional anolyte-crossover flux (4 Fe3+ per O2; flux·A/V_L is
        # already mol/L/s — fe3_shuttle 2026-08-06 erratum).
        T_for_o2 = max(T_cath, 0.0)
        o2_M = (_dp(dp, "fe3_o2_fraction_of_sat")
                * dissolved_o2_saturation_mol_L(T_for_o2))
        r_prod_M_s = fe2_oxidation_rate(
            fe2, o2_M, pH, T_for_o2,
            _dp(dp, "fe3_k_ox_ref"), _dp(dp, "fe3_Ea_ox_J_mol"),
        )
        r_prod_M_s += (4.0 * _dp(dp, "fe3_crossover_o2_flux_mol_m2_s")
                       * area_m2 / V_cath_L)
        r_prod_M_hr = r_prod_M_s * 3600.0

        # Parasitic shuttle current at the start-of-step Fe3+ level
        # (fe3 is M = mol/L; ×1000 → mol/m³ for the A/m² flux).
        i_shuttle_A_m2 = FARADAY * km_fe3_m_s * fe3 * 1000.0

        # Catholyte Fe3+ CSTR step.  Sources: production + recirculation
        # inflow; sinks: cathodic shuttle return + recirculation outflow.
        # Relaxed EXACTLY by exponential toward the step's quasi-steady
        # (frozen-rate) point — unconditionally stable even when fast
        # recirculation or a large A/V makes the compartment stiff (same
        # treatment as the cell-voltage relaxation below).
        fe3_res = fe3_res_next
        recirc_in_1_hr = flow_L_hr / V_cath_L
        k_tot_1_hr = k_shuttle_1_hr + recirc_in_1_hr
        fe3_star = (r_prod_M_hr + recirc_in_1_hr * fe3_res) / k_tot_1_hr
        decay = math.exp(-k_tot_1_hr * dt_hr)
        fe3_tent = fe3_star + (fe3 - fe3_star) * decay
        # Exact step-integral of the shuttle return for the Fe2+ ledger:
        #   ∫ k_shuttle · fe3(t) dt  over the step.
        integral_fe3_M_hr = (fe3_star * dt_hr
                             + (fe3 - fe3_star) * (1.0 - decay) / k_tot_1_hr)
        shuttle_return_M_hr = k_shuttle_1_hr * integral_fe3_M_hr / dt_hr

        # Hydrolysis cap, operator-split (instant precipitation of the
        # above-cap excess — the dynamic analogue of fe3_shuttle's min(cap, ·)).
        # With Fe(III) phase aging on, the cap is the inventory-weighted,
        # Ostwald-aged phase cap (ferrihydrite -> goethite -> hematite /
        # akaganeite on the chloride path) instead of the single Fe(OH)3 solid,
        # so the H+ release follows the phase's aging schedule.
        if ferric_aging_enabled:
            cap_cath = blended_cap_M(pH, phase_inv, ferric_initial)
        else:
            cap_cath = fe3_solubility_cap_M(pH)
        precip_M = max(0.0, fe3_tent - cap_cath)
        fe3_next = fe3_tent - precip_M
        sludge_cum_next += precip_M * V_cath_L
        precip_rate_M_hr = precip_M / dt_hr
        if ferric_aging_enabled:
            # Fresh precipitate lands in the initial phase, then Ostwald-ages
            # toward the less-soluble tail; the blended cap for the NEXT step
            # drops as the inventory matures, pulling further Fe3+ out (and
            # releasing 3 H+/Fe) on the aging schedule.
            phase_inv[ferric_initial] += precip_M * V_cath_L
            phase_inv = age_inventory(
                phase_inv,
                bath_anion=bath_anion,
                dt_hr=dt_hr,
                temperature_C=max(T_cath, 0.0),
                t12_fh_gh_hr=_dp(dp, "ferric_aging_t12_fh_gh_hr"),
                t12_gh_hem_hr=_dp(dp, "ferric_aging_t12_gh_hem_hr"),
            )

        # Reservoir Fe3+: passive mixer fed by the catholyte return; the same
        # hydrolysis cap applies at the reservoir pH (a pH ~3.5 balance tank
        # holds almost no Fe3+) and its precipitation joins the same sludge
        # ledger.  The reservoir's own autoxidation is NOT modelled (documented
        # limitation — the scenario O2 pinning describes the catholyte).
        fe3_res_tent = fe3_res + (flow_L_hr / V_res_L) * (fe3 - fe3_res) * dt_hr
        precip_res_M = max(0.0, fe3_res_tent - fe3_solubility_cap_M(aux.pH_reservoir))
        fe3_res_next = max(0.0, fe3_res_tent - precip_res_M)
        sludge_cum_next += precip_res_M * V_res_L

    # Galvanostatic split: the shuttle rides on top of the intentional
    # current, so the Fe/HER pair shares j − i_sh (fe3_shuttle.ce_penalty_at_j).
    j_fe_her_A_m2 = max(j_A_m2 - i_shuttle_A_m2, 0.0)

    # =====================================================================
    # 1. Fe2+ MASS BALANCE (index 2)
    # =====================================================================
    # Consumption by Faraday deposition: d(fe2)/dt = -j_A_m2*FE*area / (z*F*V_cath)
    # in mol/m³/s → convert to M/hr
    consumption_M_hr = (j_fe_her_A_m2 * FE / (Z_FE * FARADAY)) * area_m2 * 3600.0 / V_cath_dyn

    # Recirculation exchange: (flow/V_cath) * (fe2_res - fe2)
    recirc_fe2_M_hr = (flow_L_hr / V_cath_dyn) * (aux.fe2_reservoir_M - fe2)

    # Makeup source (direct to catholyte for L0; reservoir makeup tracked in aux)
    makeup_M_hr = _dp(dp, "fe2_makeup_rate_M_hr")

    dfe2_dt = -consumption_M_hr + recirc_fe2_M_hr + makeup_M_hr
    # Redox transfer: autoxidation removes Fe2+ (r_prod); the cathodic shuttle
    # returns it as Fe2+.  Net inventory leaves only via Fe(OH)3 precipitation,
    # which is charged to the sludge ledger instead (both zero when disabled).
    dfe2_dt += shuttle_return_M_hr - r_prod_M_hr
    # Ingress dilution (coupling-on): dilute with Fe2+-free water toward 0.
    if env is not None:
        dfe2_dt -= env.ingress_dilution_rate_1_hr * fe2
    x_next[2] = max(1e-6, fe2 + dfe2_dt * dt_hr)

    # Reservoir Fe2+ balance:
    # d(fe2_res)/dt = (flow/V_res)*(fe2 - fe2_res) + makeup_to_res - consumption_res
    # For L0: makeup goes to reservoir, return flow brings depleted catholyte back
    # Net: flow brings fe2 back from cell, makeup adds to reservoir
    dfe2_res_dt = (flow_L_hr / V_res_L) * (fe2 - aux.fe2_reservoir_M) + \
                  makeup_M_hr * (V_cath_dyn / V_res_L)
    fe2_res_next = max(1e-6, aux.fe2_reservoir_M + dfe2_res_dt * dt_hr)

    # =====================================================================
    # 2. pH / BUFFER DYNAMICS (index 3)
    # =====================================================================
    # HER at cathode: 2H2O + 2e- → H2 + 2OH-
    # OH- production rate (mol/s) = j_A_m2 * (1-FE) / (1 * F) * area
    # (1 mol OH- per mol e- for HER; the Fe3+ shuttle makes no OH-)
    OH_production_mol_s = j_fe_her_A_m2 * (1.0 - FE) / FARADAY * area_m2
    OH_production_M_hr = OH_production_mol_s * 3600.0 / V_cath_dyn

    # Acid dose: explicit rate + pH feedback holding the pH setpoint (a real
    # cell doses acid to hold pH against HER hydroxide production).  The steady-
    # state dose that exactly cancels HER OH- is bath_dynamics.steady_state_acid_dose_M_hr.
    base_acid = _dp(dp, "acid_dose_rate_M_hr")
    K_pH = _dp(dp, "pH_control_gain_M_hr_ph")
    pH_set = dp.get("pH", 3.5)
    acid_dose_M_hr = base_acid + K_pH * (pH_set - pH)

    # Buffer capacity: d(pH)/dt = -(net_proton_rate_M_hr) / beta
    # Net proton rate = acid_dose - OH_production (OH- consumes protons equivalently)
    # Adding acid (positive net_proton) lowers pH, so negative sign.
    # Redox proton terms (zero when the shuttle is disabled):
    #   autoxidation  4 Fe²⁺ + O₂ + 4H⁺ → 4 Fe³⁺ + 2H₂O   consumes 1 H⁺/Fe
    #   precipitation Fe³⁺ + 3H₂O → Fe(OH)₃(s) + 3H⁺        releases 3 H⁺/Fe
    # together a net +2 H⁺ per mol of sludge formed (the shuttle itself is
    # proton-neutral); reservoir precipitation protons are not tracked.
    beta = _dp(dp, "buffer_capacity_beta")
    net_proton_M_hr = (acid_dose_M_hr - OH_production_M_hr
                       - r_prod_M_hr + 3.0 * precip_rate_M_hr)

    # Recirculation mixing for pH
    recirc_pH_hr = (flow_L_hr / V_cath_dyn) * (aux.pH_reservoir - pH)

    dpH_dt = -net_proton_M_hr / max(beta, 1e-6) + recirc_pH_hr
    # Ingress dilution drags pH toward neutral rainwater (coupling-on).
    if env is not None:
        dpH_dt += env.ingress_dilution_rate_1_hr * (7.0 - pH)
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
    T_amb = env.T_ambient_C if env is not None else _dp(dp, "T_ambient_C")
    Q_amb_cath_W = UA_amb * (T_cath - T_amb)
    Q_amb_anol_W = UA_amb * (T_anol - T_amb)

    # Environmental convective + rain cooling (coupling-on; zero by default)
    Q_conv_cath_W = 0.0
    Q_conv_anol_W = 0.0
    Q_rain_cath_W = 0.0
    if env is not None:
        A_heat = _dp(dp, "heat_exchange_area_m2")
        Q_conv_cath_W = env.h_conv_W_m2_K * A_heat * (T_cath - T_amb)
        Q_conv_anol_W = env.h_conv_W_m2_K * A_heat * (T_anol - T_amb)
        Q_rain_cath_W = env.rain_cooling_W_m2 * A_heat

    # Recirculation heat exchange: flow * rho * Cp * (T_res - T_comp) / 3600
    # flow in L/hr → m³/s: flow/1000/3600; rho in kg/m³; Cp in J/(kg·K)
    # But simpler: flow_L_hr * rho * Cp / 3600 gives W/K equivalent
    flow_thermal_W_K = flow_L_hr * (RHO_ELECTROLYTE_KG_M3 / 1000.0) * CP_ELECTROLYTE_J_KG_K / 3600.0

    # Thermal mass: V_L * rho * Cp / 1000 [J/K]
    # V in L → kg: V * rho/1000; then kg * Cp = J/K
    mass_cath_J_K = V_cath_L * (RHO_ELECTROLYTE_KG_M3 / 1000.0) * CP_ELECTROLYTE_J_KG_K
    mass_anol_J_K = V_anol_L * (RHO_ELECTROLYTE_KG_M3 / 1000.0) * CP_ELECTROLYTE_J_KG_K
    mass_res_J_K = V_res_L * (RHO_ELECTROLYTE_KG_M3 / 1000.0) * CP_ELECTROLYTE_J_KG_K

    # Active temperature control: hold the cell (and reservoir) near the
    # design-point operating setpoint (heater/chiller).  A real electrowinning
    # cell regulates its process temperature this way.  The reservoir is
    # conditioned to the same setpoint so the large recirculation conductance
    # reinforces (rather than drags) the compartment temperature toward the tank.
    K_T_ctrl = _dp(dp, "temperature_control_gain_W_K")
    T_setpoint = _dp(dp, "temperature_C")
    Q_ctrl_cath_W = K_T_ctrl * (T_setpoint - T_cath)
    Q_ctrl_anol_W = K_T_ctrl * (T_setpoint - T_anol)
    Q_ctrl_res_W = K_T_ctrl * (T_setpoint - aux.T_reservoir_C)

    # Catholyte energy balance: dT_c/dt = (Q_in - Q_out) / mass_cath [K/s]
    Q_net_cath_W = (Q_cath_W
                    + Q_ctrl_cath_W
                    - Q_cool_W
                    - Q_membrane_W
                    - Q_amb_cath_W
                    - Q_conv_cath_W
                    - Q_rain_cath_W
                    + flow_thermal_W_K * (aux.T_reservoir_C - T_cath))
    dT_cath_dt_K_s = Q_net_cath_W / mass_cath_J_K
    dT_cath_dt_C_hr = dT_cath_dt_K_s * 3600.0

    # Anolyte energy balance
    Q_net_anol_W = (Q_anol_W
                    + Q_ctrl_anol_W
                    + Q_membrane_W
                    - Q_amb_anol_W
                    - Q_conv_anol_W
                    + flow_thermal_W_K * (aux.T_reservoir_C - T_anol))
    dT_anol_dt_K_s = Q_net_anol_W / mass_anol_J_K
    dT_anol_dt_C_hr = dT_anol_dt_K_s * 3600.0

    # Reservoir energy balance (receives return from both compartments)
    # dT_res/dt = flow*(T_cath - T_res)/V_res + flow*(T_anol - T_res)/V_res
    # Plus ambient losses from reservoir
    Q_amb_res_W = UA_amb * (aux.T_reservoir_C - T_amb)
    Q_net_res_W = (flow_thermal_W_K * (T_cath - aux.T_reservoir_C)
                   + flow_thermal_W_K * (T_anol - aux.T_reservoir_C)
                   + Q_ctrl_res_W
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
    tau_V_s = max(tau_elec_s + tau_mt_s, 1.0)  # seconds
    tau_V_hr = max(tau_V_s / 3600.0, _dp(dp, "V_relax_min_hr"))

    # Exact exponential tracking of the physics-predicted cell voltage.  This is
    # stable for any tau (explicit Euler goes unstable when tau < dt), so the
    # state can relax with a physically short electrical time constant instead of
    # an artificial 10 h floor that let the weakly-observed voltage drift.
    x_next[6] = v_pred + (V_cell - v_pred) * math.exp(-dt_hr / tau_V_hr)

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
    # The physics model's rate assumes the full applied j reaches the Fe/HER
    # pair; scale down by the shuttle slip (identity factor 1.0 when disabled).
    deposit_rate_shuttle = deposit_rate_um_hr * (j_fe_her_A_m2 / j_A_m2)
    x_next[5] = max(0.0, deposit + deposit_rate_shuttle * dt_hr)

    # =====================================================================
    # 7. ELECTRO-OSMOTIC WATER DRAG (CSTR extension — OFF by default)
    # =====================================================================
    # Water leaves the catholyte through the membrane with the H⁺ current
    # (models/water_drag.py), shrinking the catholyte volume and concentrating
    # its non-volatile solutes.  Two effects, both only when ``water_drag_enabled``:
    #   (a) the running volume V_cath_dyn (already threaded into the fe2/pH
    #       mass-balance denominators above) shrinks, so source/sink fluxes
    #       distribute over a smaller volume;
    #   (b) the solutes already present concentrate by the volume ratio.
    # The reservoir volume is treated as the balance tank that absorbs the
    # exiting water (anolyte composition is not a tracked state here), so the
    # closed electrolyte balance is represented in the catholyte ledger.
    cath_vol_next = aux.catholyte_volume_L
    mem_age_next = aux.membrane_age_hr
    if water_drag_enabled:
        # Starting volume: the running volume, or the design volume on first step.
        vol_old = V_cath_dyn
        if aux.catholyte_volume_L is None:
            vol_old = V_cath_L
        mem_area_m2 = dp.get("membrane_area_m2") or area_m2
        drag_L_m2_hr = water_volume_flux_L_m2_hr(
            j_A_m2=j_A_m2,
            temperature_C=max(0.0, T_cath),
            membrane_age_hr=mem_age_next,
        )
        dV_L = drag_L_m2_hr * mem_area_m2 * dt_hr
        vol_new = max(vol_old - dV_L, 0.25 * vol_old)  # safety floor
        if vol_new != vol_old:
            ratio = vol_old / vol_new
            # Non-volatile solutes concentrate: [Fe²⁺] ∝ V⁻¹, and [H⁺] ∝ V⁻¹
            # so pH falls by log10(ratio).
            x_next[2] = max(1e-6, x_next[2] * ratio)
            x_next[3] = max(0.0, min(14.0, x_next[3] - math.log10(ratio)))
        cath_vol_next = vol_new
        mem_age_next = mem_age_next + dt_hr

    # --- Assemble next aux ---
    aux_next = BathAux(
        T_reservoir_C=T_res_next,
        fe2_reservoir_M=fe2_res_next,
        pH_reservoir=pH_res_next,
        fe3_catholyte_M=fe3_next,
        fe3_reservoir_M=fe3_res_next,
        fe3_sludge_cumulative_mol=sludge_cum_next,
        ferric_phase_inventory=(
            dict(phase_inv) if (fe3_enabled and ferric_aging_enabled)
            else dict(aux.ferric_phase_inventory or {})),
        ferric_phase_initial=(ferric_initial if (fe3_enabled and ferric_aging_enabled)
                              else aux.ferric_phase_initial),
        catholyte_volume_L=cath_vol_next,
        membrane_age_hr=mem_age_next,
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


# ---------------------------------------------------------------------------
# Fe3+ shuttle helpers (static counterpart: models/fe3_shuttle.py)
# ---------------------------------------------------------------------------

def apply_fe3_scenario(design_point: Dict[str, Any], scenario: "Any") -> Dict[str, Any]:
    """Enable the Fe³⁺ CSTR terms and map an fe3_shuttle scenario onto the
    design point (mutates and returns the dict).

    ``scenario`` is an ``fe3_shuttle.ShuttleScenario`` (e.g.
    ``sealed_divided_cell()``); its O₂ pinning and crossover flux become the
    ``fe3_*`` knobs.  Remaining knobs (boundary layer, D, k_ox, Ea) take the
    ``ShuttleParams`` screening defaults unless already present.
    """
    design_point["fe3_shuttle_enabled"] = True
    design_point["fe3_o2_fraction_of_sat"] = scenario.o2_fraction_of_sat
    design_point["fe3_crossover_o2_flux_mol_m2_s"] = scenario.crossover_o2_flux_mol_m2_s
    return design_point


def enable_water_drag(
    design_point: Dict[str, Any],
    membrane_area_m2: Optional[float] = None,
) -> Dict[str, Any]:
    """Enable the electro-osmotic water-drag CSTR term (mutates and returns
    the design point).

    Sets ``water_drag_enabled`` and optionally ``membrane_area_m2`` (defaults
    to the electrode area).  The drag coefficient knobs (n_w_ref, etc.) tune
    via models/water_drag defaults; no scenario object is needed.
    """
    design_point["water_drag_enabled"] = True
    if membrane_area_m2 is not None:
        design_point["membrane_area_m2"] = membrane_area_m2
    return design_point


def fe3_shuttle_terms(
    x: "np.ndarray",
    aux: BathAux,
    design_point: Dict[str, Any],
) -> Dict[str, float]:
    """Instantaneous Fe³⁺ CSTR terms at the current state (diagnostics/logging).

    Returns production rate, shuttle current, sink rate, hydrolysis cap and
    the CE penalty at the state's current density — the same quantities the
    static module reports, evaluated on the *dynamic* state.  All zeros when
    the extension is disabled.
    """
    out = {
        "fe3_M": max(0.0, aux.fe3_catholyte_M),
        "fe3_reservoir_M": max(0.0, aux.fe3_reservoir_M),
        "fe3_sludge_cumulative_mol": max(0.0, aux.fe3_sludge_cumulative_mol),
        "r_prod_M_s": 0.0,
        "k_shuttle_1_s": 0.0,
        "shuttle_sink_M_s": 0.0,
        "i_shuttle_A_m2": 0.0,
        "fe3_solubility_cap_M": math.inf,
        "ce_loss_fraction": 0.0,
        "enabled": False,
    }
    if not bool(_dp(design_point, "fe3_shuttle_enabled")):
        return out
    dp = design_point
    area_m2 = dp.get("electrode_area_m2", 1.0)
    V_cath_L = _dp(dp, "catholyte_volume_L")
    T_cath = max(0.0, float(x[0]))
    fe2 = max(1e-6, float(x[2]))
    pH = float(x[3])
    j_A_m2 = max(1e-3, float(x[4])) * 10.0

    km = _dp(dp, "fe3_d_m2_s") / max(_dp(dp, "fe3_boundary_layer_m"), 1e-12)
    k_shuttle = km * (area_m2 / (V_cath_L / 1000.0))
    o2_M = _dp(dp, "fe3_o2_fraction_of_sat") * dissolved_o2_saturation_mol_L(T_cath)
    r_prod = fe2_oxidation_rate(fe2, o2_M, pH, T_cath,
                                _dp(dp, "fe3_k_ox_ref"), _dp(dp, "fe3_Ea_ox_J_mol"))
    r_prod += 4.0 * _dp(dp, "fe3_crossover_o2_flux_mol_m2_s") * area_m2 / V_cath_L
    i_sh = FARADAY * km * out["fe3_M"] * 1000.0
    out.update({
        "r_prod_M_s": r_prod,
        "k_shuttle_1_s": k_shuttle,
        "shuttle_sink_M_s": k_shuttle * out["fe3_M"],
        "i_shuttle_A_m2": i_sh,
        "fe3_solubility_cap_M": fe3_solubility_cap_M(pH),
        "ce_loss_fraction": min(i_sh / j_A_m2, 1.0),
        "enabled": True,
    })
    return out


def steady_state_fe3_M(design_point: Dict[str, Any]) -> float:
    """Static steady-state [Fe³⁺] prediction for cross-checking the CSTR.

    Maps the design point onto ``fe3_shuttle.ShuttleParams``/scenario and
    returns that module's closed-form ``fe3_ss_M``.  The dynamic bath, held at
    fixed (T, pH, fe2) with precipitation inactive everywhere, must relax to
    the same value: the recirculation terms cancel identically at mutual
    steady state, leaving ``[Fe³⁺]_ss = r_prod / (k_m·A/V)``.

    Note the static module's ``ShuttleParams.cathode_area_m2`` scales the
    crossover term by the *cathode* area, so it is mapped from
    ``electrode_area_m2`` here.
    """
    from .fe3_shuttle import ShuttleParams, ShuttleScenario, steady_state

    dp = design_point
    p = ShuttleParams(
        temperature_C=dp.get("temperature_C", 60.0),
        pH=dp.get("pH", 3.5),
        fe2_M=dp.get("fe2_M", 1.0),
        cathode_area_m2=dp.get("electrode_area_m2", 1.0),
        catholyte_volume_L=_dp(dp, "catholyte_volume_L"),
        boundary_layer_m=_dp(dp, "fe3_boundary_layer_m"),
        d_fe3_m2_s=_dp(dp, "fe3_d_m2_s"),
        k_ox_ref=_dp(dp, "fe3_k_ox_ref"),
        Ea_ox_J_mol=_dp(dp, "fe3_Ea_ox_J_mol"),
    )
    s = ShuttleScenario(
        "design_point",
        o2_fraction_of_sat=_dp(dp, "fe3_o2_fraction_of_sat"),
        crossover_o2_flux_mol_m2_s=_dp(dp, "fe3_crossover_o2_flux_mol_m2_s"),
    )
    return float(steady_state(p, s)["fe3_ss_M"])
