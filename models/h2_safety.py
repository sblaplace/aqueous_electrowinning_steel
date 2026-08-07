"""
Hydrogen safety envelope for aqueous iron electrowinning cells.

Why this module exists
----------------------
The program generates H₂ as a parasitic product of the HER that competes
with Fe deposition.  At FE = 80 %, one-fifth of the applied current
makes H₂; at FE = 50 %, half does.  In a ventilated industrial cell this
is routine; in a re-deployable bench unit in a small enclosure it is the
first thing a site HSE review will ask about.

The gas_holdup module tracks void fraction inside the cathode channel
and the current redistribution that results; the reference-cell
deployment package lists H₂ monitors and LEL alarms.  Neither answers
the *enclosure* question: given this cell's H₂ generation rate,
enclosure volume, and ventilation, how long until the enclosure reaches
25 % LEL (1 % v/v H₂ in air)?

This module closes that gap with a first-principles mass balance:

    dC/dt = G / V_enc − Q_vent · C / V_enc

where G is the H₂ molar generation rate (mol/s), V_enc is the enclosure
volume (m³), and Q_vent is the volumetric ventilation rate (m³/s).

References
----------
* NFPA 2 (Hydrogen Technologies Code) — LEL = 4 % v/v in air; 25 % LEL
  = 1 % v/v is the standard alarm setpoint.
* OSHA / ATEX — ventilation requirement to keep below 25 % LEL under
  steady-state worst-case generation.
* Ideal gas law at 25 °C, 1 atm: 1 mol H₂ = 24.47 L.

Units
-----
Current densities: A/m² (internally), mA/cm² (public API).
Concentrations: % v/v in air (public API), mol/m³ (internally).
Times: seconds (internally), minutes (public API).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .electrochemistry import FARADAY

# ─── Constants ──────────────────────────────────────────────────────

# LEL (Lower Explosive Limit) for H₂ in air = 4 % v/v (NFPA 2).
LEL_H2_VOLUME_PERCENT = 4.0

# Standard alarm setpoint = 25 % of LEL = 1 % v/v.
ALARM_SETPOINT_FRACTION = 0.25
ALARM_SETPOINT_PERCENT = LEL_H2_VOLUME_PERCENT * ALARM_SETPOINT_FRACTION  # 1.0 %

# Molar volume of ideal gas at 25 °C, 1 atm (L/mol).
VMOL_25C_1ATM_L = 24.47

# Safety factor: ventilation is sized for 25 % of LEL steady-state,
# with a 2× engineering margin (standard HSE practice for intermittent
# generation).
VENTILATION_SAFETY_FACTOR = 2.0


@dataclass(frozen=True)
class EnclosureSpec:
    """Physical enclosure for the electrowinning cell.

    Parameters
    ----------
    volume_m3 : float
        Internal volume of the enclosure (m³).  A small bench cell
        cabinet is ~0.1–0.5 m³; a walk-in enclosure is 2–10 m³.
    ventilation_m3_s : float
        Volumetric ventilation rate (m³/s).  Zero means sealed
        (worst-case accumulation).  Natural ventilation in a small
        cabinet is ~0.001–0.01 m³/s; forced ventilation with a fan
        is 0.05–0.5 m³/s.
    temperature_C : float
        Enclosure air temperature (°C), for ideal-gas density.
    """
    volume_m3: float
    ventilation_m3_s: float = 0.01   # ~0.6 m³/hr, small natural-draft cabinet
    temperature_C: float = 25.0

    def __post_init__(self) -> None:
        if self.volume_m3 <= 0:
            raise ValueError("volume_m3 must be positive")
        if self.ventilation_m3_s < 0:
            raise ValueError("ventilation_m3_s must be non-negative")

    @property
    def vmol_m3(self) -> float:
        """Molar volume of ideal gas at enclosure temperature (m³/mol)."""
        T_K = self.temperature_C + 273.15
        return VMOL_25C_1ATM_L * 1e-3 * (T_K / 298.15)


@dataclass(frozen=True)
class H2GenerationRate:
    """H₂ generation from an electrowinning cell.

    Parameters
    ----------
    j_total_A_m2 : float
        Total applied cathodic current density (A/m²).
    fe_fraction : float
        Faradaic efficiency for Fe deposition (0–1).  The remainder
        (1 - fe_fraction) goes to HER.
    cathode_area_m2 : float
        Cathode geometric area (m²).
    n_cells : int
        Number of cells in the enclosure.
    """
    j_total_A_m2: float
    fe_fraction: float
    cathode_area_m2: float
    n_cells: int = 1

    def __post_init__(self) -> None:
        if self.j_total_A_m2 <= 0:
            raise ValueError("j_total_A_m2 must be positive")
        if not (0 < self.fe_fraction <= 1.0):
            raise ValueError("fe_fraction must be in (0, 1]")
        if self.cathode_area_m2 <= 0:
            raise ValueError("cathode_area_m2 must be positive")
        if self.n_cells < 1:
            raise ValueError("n_cells must be >= 1")

    @property
    def her_fraction(self) -> float:
        """Fraction of current going to HER."""
        return 1.0 - self.fe_fraction

    @property
    def her_current_A(self) -> float:
        """Total HER current across all cells (A)."""
        return self.j_total_A_m2 * self.cathode_area_m2 * self.n_cells * self.her_fraction

    @property
    def h2_mol_per_s(self) -> float:
        """H₂ generation rate (mol/s) from 2 H⁺ + 2 e⁻ → H₂.

        Each mole of H₂ requires 2 moles of electrons (2F coulombs).
        """
        return self.her_current_A / (2.0 * FARADAY)

    @property
    def h2_L_per_hour(self) -> float:
        """H₂ generation rate in L/hr at 25 °C, 1 atm."""
        return self.h2_mol_per_s * VMOL_25C_1ATM_L * 3600.0

    @property
    def h2_g_per_hour(self) -> float:
        """H₂ generation rate in g/hr (M_H2 = 2.016 g/mol)."""
        return self.h2_mol_per_s * 2.016 * 3600.0


@dataclass(frozen=True)
class H2SafetyResult:
    """Enclosure H₂ safety assessment.

    Fields
    ------
    h2_percent_at_1hr : float
        H₂ concentration (% v/v) after 1 hour with zero ventilation.
    time_to_25pct_lel_s : float
        Time (s) to reach 25 % LEL (1 % v/v) with zero ventilation.
        ``float('inf')`` if ventilation keeps steady-state below alarm.
    time_to_lel_s : float
        Time (s) to reach 100 % LEL (4 % v/v) with zero ventilation.
    steady_state_percent : float
        Steady-state H₂ concentration (% v/v) with ventilation on.
    min_ventilation_m3_s : float
        Minimum ventilation rate to hold steady-state below 25 % LEL,
        with the 2× safety factor.
    min_ventilation_ach : float
        Same as above, in air changes per hour.
    alarm_triggers : bool
        True if steady-state exceeds 25 % LEL alarm setpoint.
   危险_危险 : bool
        True if steady-state exceeds 100 % LEL (immediate danger).
    """
    generation: H2GenerationRate
    enclosure: EnclosureSpec
    h2_percent_at_1hr: float
    time_to_25pct_lel_s: float
    time_to_lel_s: float
    steady_state_percent: float
    min_ventilation_m3_s: float
    min_ventilation_ach: float
    alarm_triggers: bool
    immediate_danger: bool

    def summary(self) -> str:
        """Human-readable safety summary."""
        lines = [
            "═══ H₂ Safety Assessment ═══",
            f"  Generation:  {self.generation.h2_L_per_hour:.1f} L/hr "
            f"({self.generation.h2_g_per_hour:.2f} g/hr)",
            f"  HER fraction: {self.generation.her_fraction * 100:.1f} % "
            f"of total current",
            f"  Enclosure:    {self.enclosure.volume_m3:.2f} m³, "
            f"ventilation {self.enclosure.ventilation_m3_s:.4f} m³/s "
            f"({self.enclosure.ventilation_m3_s * 3600:.1f} m³/hr)",
            "",
        ]
        if self.enclosure.ventilation_m3_s > 0:
            lines.append(
                f"  Steady-state: {self.steady_state_percent:.3f} % v/v H₂ "
                f"(alarm at {ALARM_SETPOINT_PERCENT:.1f} %)"
            )
            if self.alarm_triggers:
                lines.append("  ⚠ ALARM: Steady-state exceeds 25 % LEL")
            if self.immediate_danger:
                lines.append("  ✗ DANGER: Steady-state exceeds 100 % LEL")
        else:
            lines.append("  ⚠ SEALED enclosure (no ventilation)")
        if math.isfinite(self.time_to_25pct_lel_s):
            lines.append(
                f"  Time to 25 % LEL (sealed): {self.time_to_25pct_lel_s / 60:.1f} min"
            )
        else:
            lines.append("  Time to 25 % LEL (sealed): never (ventilation holds)")
        if math.isfinite(self.time_to_lel_s):
            lines.append(
                f"  Time to LEL (sealed):       {self.time_to_lel_s / 60:.1f} min"
            )
        lines.append("")
        lines.append(
            f"  Min ventilation for < 25 % LEL: "
            f"{self.min_ventilation_m3_s:.4f} m³/s "
            f"({self.min_ventilation_ach:.1f} ACH)"
        )
        return "\n".join(lines)


def assess_h2_safety(
    generation: H2GenerationRate,
    enclosure: EnclosureSpec,
) -> H2SafetyResult:
    """Compute the H₂ safety envelope for an enclosure.

    Parameters
    ----------
    generation : H2GenerationRate
        H₂ generation parameters (current, FE, area, cells).
    enclosure : EnclosureSpec
        Enclosure geometry and ventilation.

    Returns
    -------
    H2SafetyResult
        Full safety assessment with times to alarm, steady-state
        concentration, and minimum ventilation requirement.
    """
    G = generation.h2_mol_per_s        # mol/s
    V = enclosure.volume_m3             # m³
    Q = enclosure.ventilation_m3_s      # m³/s
    vmol = enclosure.vmol_m3            # m³/mol

    # Conversion: mol/m³ → % v/v
    # C_% = (n/V) × vmol × 100
    # where n/V is mol/m³ and vmol is m³/mol
    def mol_m3_to_percent(c_mol_m3: float) -> float:
        return c_mol_m3 * vmol * 100.0

    def percent_to_mol_m3(c_pct: float) -> float:
        return c_pct / (vmol * 100.0)

    # ── Steady-state concentration ────────────────────────────────
    # C_ss = G / Q  (mol/m³), valid for Q > 0
    if Q > 0:
        c_ss_mol_m3 = G / Q
        c_ss_pct = mol_m3_to_percent(c_ss_mol_m3)
    else:
        c_ss_pct = float('inf')

    # ── Time to reach alarm setpoint (25 % LEL = 1 % v/v) ────────
    # With ventilation: dC/dt = G/V - Q·C/V
    # C(t) = C_ss · (1 - exp(-Q·t/V))
    # Solve for t when C(t) = C_alarm:
    #   t = -(V/Q) · ln(1 - C_alarm / C_ss)
    # Without ventilation (Q = 0): C(t) = G·t / V
    #   t = C_alarm · V / G
    alarm_pct = ALARM_SETPOINT_PERCENT   # 1.0 % v/v
    lel_pct = LEL_H2_VOLUME_PERCENT      # 4.0 % v/v

    if Q > 0:
        if c_ss_pct <= alarm_pct:
            t_25_lel = float('inf')  # ventilation holds below alarm
        elif c_ss_pct <= 0:
            t_25_lel = float('inf')
        else:
            ratio = alarm_pct / c_ss_pct
            if ratio >= 1.0:
                t_25_lel = float('inf')
            else:
                t_25_lel = -(V / Q) * math.log(1.0 - ratio)

        if c_ss_pct <= lel_pct:
            t_lel = float('inf')
        elif c_ss_pct <= 0:
            t_lel = float('inf')
        else:
            ratio_lel = lel_pct / c_ss_pct
            if ratio_lel >= 1.0:
                t_lel = float('inf')
            else:
                t_lel = -(V / Q) * math.log(1.0 - ratio_lel)
    else:
        # Sealed: linear accumulation
        c_alarm_mol_m3 = percent_to_mol_m3(alarm_pct)
        c_lel_mol_m3 = percent_to_mol_m3(lel_pct)
        if G > 0:
            t_25_lel = c_alarm_mol_m3 * V / G
            t_lel = c_lel_mol_m3 * V / G
        else:
            t_25_lel = float('inf')
            t_lel = float('inf')

    # ── Concentration at 1 hour (sealed worst case) ──────────────
    if G > 0:
        c_1hr_mol_m3 = G * 3600.0 / V  # sealed, no ventilation
        c_1hr_pct = mol_m3_to_percent(c_1hr_mol_m3)
    else:
        c_1hr_pct = 0.0

    # ── Minimum ventilation for < 25 % LEL steady-state ──────────
    # C_ss = G / Q_min  →  Q_min = G / C_alarm (with safety factor)
    c_alarm_mol_m3 = percent_to_mol_m3(alarm_pct)
    if G > 0 and c_alarm_mol_m3 > 0:
        q_min = G / c_alarm_mol_m3 * VENTILATION_SAFETY_FACTOR
    else:
        q_min = 0.0

    # Air changes per hour
    ach = q_min * 3600.0 / V if V > 0 else 0.0

    return H2SafetyResult(
        generation=generation,
        enclosure=enclosure,
        h2_percent_at_1hr=c_1hr_pct,
        time_to_25pct_lel_s=t_25_lel,
        time_to_lel_s=t_lel,
        steady_state_percent=c_ss_pct,
        min_ventilation_m3_s=q_min,
        min_ventilation_ach=ach,
        alarm_triggers=c_ss_pct > alarm_pct if math.isfinite(c_ss_pct) else True,
        immediate_danger=c_ss_pct > lel_pct if math.isfinite(c_ss_pct) else True,
    )


# ─── Convenience functions ──────────────────────────────────────────

def min_ventilation_for_fe_rate(
    fe_production_kg_hr: float,
    fe_percent: float = 80.0,
    enclosure_m3: float = 1.0,
) -> float:
    """Quick estimate: minimum ventilation (m³/s) for a given Fe rate.

    Assumes pure Fe²⁺ + 2e⁻ → Fe, with HER taking the remaining
    (100 - fe_percent) % of current.  Returns the ventilation rate
    needed to hold steady-state below 25 % LEL (1 % v/v H₂).
    """
    # Fe production → total current
    # M_Fe = 55.845 g/mol; 2 e⁻ per Fe
    fe_mol_s = fe_production_kg_hr * 1000.0 / 55.845 / 3600.0
    fe_fraction = fe_percent / 100.0
    # Total current: I = 2F · n_Fe / fe_fraction
    i_total_A = 2.0 * FARADAY * fe_mol_s / fe_fraction
    # HER current
    i_her_A = i_total_A * (1.0 - fe_fraction)
    # H₂ generation
    h2_mol_s = i_her_A / (2.0 * FARADAY)
    # Min ventilation
    vmol = VMOL_25C_1ATM_L * 1e-3  # m³/mol at 25 °C
    c_alarm_mol_m3 = ALARM_SETPOINT_PERCENT / (vmol * 100.0)
    q_min = h2_mol_s / c_alarm_mol_m3 * VENTILATION_SAFETY_FACTOR
    return q_min


def bench_cell_worst_case(
    j_mA_cm2: float = 300.0,
    fe_percent: float = 80.0,
    area_cm2: float = 25.0,
) -> H2SafetyResult:
    """Quick reference: bench cell (25 cm², 1 L enclosure) at given j.

    Returns the full safety assessment.  Useful for the "is this safe
    on my bench?" question during Phase I planning.
    """
    gen = H2GenerationRate(
        j_total_A_m2=j_mA_cm2 * 10.0,  # mA/cm² → A/m²
        fe_fraction=fe_percent / 100.0,
        cathode_area_m2=area_cm2 * 1e-4,
        n_cells=1,
    )
    enc = EnclosureSpec(
        volume_m3=0.001,  # 1 L sealed container
        ventilation_m3_s=0.0,  # sealed worst case
    )
    return assess_h2_safety(gen, enc)


def ventilation_sizing(
    j_mA_cm2: float = 300.0,
    fe_percent: float = 80.0,
    area_cm2: float = 25.0,
    n_cells: int = 1,
    enclosure_m3: float = 0.5,
) -> Tuple[float, float]:
    """Ventilation sizing for a given cell configuration.

    Returns
    -------
    (q_min_m3_s, ach)
        Minimum ventilation rate (m³/s) and air changes per hour
        to hold steady-state below 25 % LEL.
    """
    gen = H2GenerationRate(
        j_total_A_m2=j_mA_cm2 * 10.0,
        fe_fraction=fe_percent / 100.0,
        cathode_area_m2=area_cm2 * 1e-4,
        n_cells=n_cells,
    )
    vmol = VMOL_25C_1ATM_L * 1e-3
    c_alarm = ALARM_SETPOINT_PERCENT / (vmol * 100.0)
    q_min = gen.h2_mol_per_s / c_alarm * VENTILATION_SAFETY_FACTOR
    ach = q_min * 3600.0 / enclosure_m3
    return q_min, ach


__all__ = [
    "LEL_H2_VOLUME_PERCENT",
    "ALARM_SETPOINT_FRACTION",
    "ALARM_SETPOINT_PERCENT",
    "VENTILATION_SAFETY_FACTOR",
    "EnclosureSpec",
    "H2GenerationRate",
    "H2SafetyResult",
    "assess_h2_safety",
    "min_ventilation_for_fe_rate",
    "bench_cell_worst_case",
    "ventilation_sizing",
]
