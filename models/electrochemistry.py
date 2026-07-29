"""
Electrochemistry utilities for aqueous electrowinning modeling.

Provides fundamental constants, Faraday's law calculations, and
cell voltage decomposition for iron electrowinning.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Physical Constants ────────────────────────────────────────────────
FARADAY = 96485.3321  # C/mol (Faraday constant)
R_GAS = 8.314462      # J/(mol·K) (universal gas constant)
AVOGADRO = 6.022e23   # 1/mol

# ─── Iron-specific Constants ──────────────────────────────────────────
M_FE = 55.845e-3      # kg/mol (molar mass of iron)
Z_FE = 2              # electrons per Fe²⁺ → Fe
E0_FE = -0.440        # V vs. SHE (standard reduction potential)
RHO_FE = 7874.0       # kg/m³ (density of iron)

# ─── Oxygen Evolution (Anode) ─────────────────────────────────────────
E0_OER = 1.229        # V vs. SHE (standard OER potential)
Z_OER = 4             # electrons per O₂


@dataclass
class CellVoltageModel:
    """
    Decomposes total cell voltage into thermodynamic and kinetic components.

    V_cell = |E_anode - E_cathode| + η_anode + η_cathode + iR_drop

    Parameters
    ----------
    E_cathode_eq : float
        Equilibrium cathode potential (V vs. SHE). Default is E°(Fe²⁺/Fe).
    E_anode_eq : float
        Equilibrium anode potential (V vs. SHE). Default is E°(OER).
    eta_cathode : float
        Cathode overpotential at operating current density (V). Positive value.
    eta_anode : float
        Anode overpotential at operating current density (V). Positive value.
    ir_drop : float
        Ohmic drop across electrolyte, membrane, contacts (V).
    """
    E_cathode_eq: float = E0_FE
    E_anode_eq: float = E0_OER
    eta_cathode: float = 0.30
    eta_anode: float = 0.40
    ir_drop: float = 0.20

    @property
    def E_thermodynamic(self) -> float:
        """Minimum thermodynamic cell voltage (V)."""
        return abs(self.E_anode_eq - self.E_cathode_eq)

    @property
    def V_cell(self) -> float:
        """Total cell voltage (V) including all overpotentials."""
        return self.E_thermodynamic + self.eta_cathode + self.eta_anode + self.ir_drop

    def summary(self) -> dict:
        return {
            "E_thermodynamic (V)": round(self.E_thermodynamic, 3),
            "η_cathode (V)": round(self.eta_cathode, 3),
            "η_anode (V)": round(self.eta_anode, 3),
            "iR drop (V)": round(self.ir_drop, 3),
            "V_cell (V)": round(self.V_cell, 3),
        }


def nernst_shift(E0: float, T: float, activity_ratio: float, n: int) -> float:
    """
    Calculate Nernst-adjusted potential.

    Parameters
    ----------
    E0 : float
        Standard potential (V vs. SHE).
    T : float
        Temperature (K).
    activity_ratio : float
        Ratio of activities (oxidized/reduced).
    n : int
        Number of electrons transferred.

    Returns
    -------
    float
        Adjusted potential (V vs. SHE).
    """
    return E0 + (R_GAS * T / (n * FARADAY)) * np.log(activity_ratio)


def specific_energy_kWh_per_kg(
    V_cell: float,
    current_efficiency: float,
    z: int = Z_FE,
    M: float = M_FE,
) -> float:
    """
    Calculate specific energy consumption in kWh per kg of metal.

    E_specific = (V_cell × z × F) / (CE × M × 3.6e6)

    Parameters
    ----------
    V_cell : float
        Total cell voltage (V).
    current_efficiency : float
        Fractional current efficiency (0–1).
    z : int
        Number of electrons per metal atom.
    M : float
        Molar mass of metal (kg/mol).

    Returns
    -------
    float
        Specific energy consumption (kWh/kg).
    """
    return (V_cell * z * FARADAY) / (current_efficiency * M * 3.6e6)


def specific_energy_kWh_per_t(V_cell: float, current_efficiency: float,
                               z: int = Z_FE, M: float = M_FE) -> float:
    """Specific energy consumption in kWh per metric tonne."""
    return specific_energy_kWh_per_kg(V_cell, current_efficiency, z, M) * 1000.0


def production_rate_kg_per_hr(
    current_A: float,
    current_efficiency: float,
    z: int = Z_FE,
    M: float = M_FE,
) -> float:
    """
    Faraday's law: mass production rate (kg/hr) at given total current.

    m_dot = (I × CE × M × 3600) / (z × F)

    Parameters
    ----------
    current_A : float
        Total current (Amperes).
    current_efficiency : float
        Fractional current efficiency (0–1).

    Returns
    -------
    float
        Production rate (kg/hr).
    """
    return (current_A * current_efficiency * M * 3600.0) / (z * FARADAY)


def current_density_to_production(
    j_mA_cm2: float,
    electrode_area_m2: float,
    current_efficiency: float,
    z: int = Z_FE,
    M: float = M_FE,
) -> float:
    """
    Production rate (kg/hr) from current density and electrode area.

    Parameters
    ----------
    j_mA_cm2 : float
        Current density (mA/cm²).
    electrode_area_m2 : float
        Single electrode area (m²).
    current_efficiency : float
        Fractional current efficiency (0–1).

    Returns
    -------
    float
        Production rate per cell (kg/hr).
    """
    j_A_m2 = j_mA_cm2 * 10.0  # mA/cm² → A/m²
    total_current = j_A_m2 * electrode_area_m2
    return production_rate_kg_per_hr(total_current, current_efficiency, z, M)


def hydrogen_evolution_loss(
    current_efficiency: float,
    V_cell: float,
) -> float:
    """
    Power wasted on HER per kg of iron produced (kWh/kg Fe).

    The HER fraction of current is (1 - CE), and it still draws full cell voltage.
    """
    fe_energy = specific_energy_kWh_per_kg(V_cell, current_efficiency)
    return fe_energy * (1.0 - current_efficiency)
