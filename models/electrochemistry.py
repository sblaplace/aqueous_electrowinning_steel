"""
Electrochemistry utilities for aqueous electrowinning modeling.

Provides fundamental constants, Faraday's law calculations, and
cell voltage decomposition for iron electrowinning.

This module now includes:
- Full V_cell decomposition (E_cathode, E_anode, η, IR drops)
- Temperature-dependent conductivity, diffusivity, viscosity
- Divided cell / membrane mode
- Fe²⁺/Fe³⁺ anode shuttle as competing anode reaction
- Energy = f(V_cell, FE) with correct Faraday constant
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from .thermodynamic_constants import (
    E0_FE_REDUCTION_V,
    E0_OER_V,
    E0_FE3_FE2_V,
    FARADAY as FARADAY_CANONICAL,
    R_GAS as R_GAS_CANONICAL,
)

if TYPE_CHECKING:
    from .anode import AnodeKinetics

# ─── Physical Constants ────────────────────────────────────────────────
FARADAY = FARADAY_CANONICAL  # C/mol (Faraday constant)
R_GAS = R_GAS_CANONICAL      # J/(mol·K) (universal gas constant)
AVOGADRO = 6.022e23   # 1/mol

# ─── Iron-specific Constants ──────────────────────────────────────────
M_FE = 55.845e-3      # kg/mol (molar mass of iron)
M_FE_G = 55.845        # g/mol  (molar mass of iron, for gravimetric calculations)
Z_FE = 2              # electrons per Fe²⁺ → Fe
E0_FE = E0_FE_REDUCTION_V  # V vs. SHE, shared Fe²⁺/Fe standard state
RHO_FE = 7874.0       # kg/m³ (density of iron)

# ─── Oxygen Evolution (Anode) ─────────────────────────────────────────
E0_OER = E0_OER_V     # V vs. SHE, shared OER standard state
Z_OER = 4             # electrons per O₂

# ─── Fe²⁺/Fe³⁺ Shuttle (Anode alternative) ────────────────────────────
E0_FE3_FE2 = E0_FE3_FE2_V  # V vs. SHE (Fe³⁺ + e⁻ → Fe²⁺)
Z_FE3_FE2 = 1         # electrons for the redox shuttle

T_REF = 298.15         # K — reference temperature


# ─── Temperature-dependent property models ──────────────────────────────

def conductivity_S_m(
    T: float,
    kappa_ref: float = 10.0,
    T_ref: float = T_REF,
    Ea_kJ_mol: float = 15.0,
) -> float:
    """
    Ionic conductivity of the electrolyte (S/m) at temperature T (K).

    Arrhenius model: κ(T) = κ_ref · exp[Ea/R · (1/T_ref − 1/T)]

    Parameters
    ----------
    T : float
        Temperature (K).
    kappa_ref : float
        Conductivity at T_ref (S/m). Default 10 S/m (typical 1 M FeSO₄).
    T_ref : float
        Reference temperature (K).
    Ea_kJ_mol : float
        Activation energy for ionic conduction (kJ/mol).

    Returns
    -------
    float
        Conductivity (S/m).
    """
    Ea = Ea_kJ_mol * 1000.0  # J/mol
    return kappa_ref * np.exp(Ea / R_GAS * (1.0 / T_ref - 1.0 / T))


def diffusivity_m2_s(
    T: float,
    D_ref: float = 7.2e-10,
    T_ref: float = T_REF,
) -> float:
    """
    Ion diffusivity (m²/s) at temperature T via Stokes-Einstein scaling.

    D(T) = D_ref · (T / T_ref) · (μ_ref / μ(T))

    Approximated as linear in T for the temperature range of interest
    (20–90 °C): D(T) ≈ D_ref · (T / T_ref) · exp[Ea_D/R · (1/T_ref − 1/T)]

    Parameters
    ----------
    T : float
        Temperature (K).
    D_ref : float
        Diffusivity at T_ref (m²/s).
    T_ref : float
        Reference temperature (K).

    Returns
    -------
    float
        Diffusivity (m²/s).
    """
    # Activation energy for diffusion ≈ 15–20 kJ/mol in aqueous electrolytes
    Ea_D = 15000.0  # J/mol
    return D_ref * (T / T_ref) * np.exp(Ea_D / R_GAS * (1.0 / T_ref - 1.0 / T))


def viscosity_Pa_s(
    T: float,
    A_andrade: float = 2.414e-5,
    B_andrade: float = 247.8,
) -> float:
    """
    Dynamic viscosity of water (Pa·s) at temperature T (K).

    Andrade equation: μ = A · exp(B / T)

    Parameters
    ----------
    T : float
        Temperature (K).
    A_andrade : float
        Pre-exponential factor (Pa·s).
    B_andrade : float
        Andrade temperature coefficient (K).

    Returns
    -------
    float
        Viscosity (Pa·s).
    """
    return A_andrade * np.exp(B_andrade / T)


# ─── Fe²⁺/Fe³⁺ Anode Shuttle Model ───────────────────────────────────

@dataclass
class FeShuttleAnode:
    """
    Fe²⁺ → Fe³⁺ + e⁻ as a competing anode reaction (E° = +0.771 V vs. SHE).

    In an undivided cell, Fe²⁺ oxidised at the anode is a parasitic reaction:
    it wastes current without producing useful product and generates Fe³⁺
    that can cross back to the cathode and re-reduce, lowering CE.

    Parameters
    ----------
    E0 : float
        Standard potential Fe³⁺/Fe²⁺ (V vs. SHE).
    i0 : float
        Exchange current density (A/m²).
    tafel_V : float
        Anodic Tafel slope (V/decade).
    fe2_conc_M : float
        Fe²⁺ concentration in the anolyte (mol/L).
    fe3_conc_M : float
        Fe³⁺ concentration in the anolyte (mol/L).
    """

    E0: float = E0_FE3_FE2
    i0: float = 1.0e-1        # A/m² — moderate kinetics on inert anodes
    tafel_V: float = 0.120    # V/decade
    fe2_conc_M: float = 1.0   # mol/L
    fe3_conc_M: float = 0.01  # mol/L (starts near zero)

    def equilibrium(self, T: float = T_REF) -> float:
        """Nernst equilibrium potential (V vs. SHE)."""
        activity_ratio = max(self.fe3_conc_M, 1e-10) / max(self.fe2_conc_M, 1e-10)
        return self.E0 + (R_GAS * T / (Z_FE3_FE2 * FARADAY)) * np.log(activity_ratio)

    def overpotential(self, j_mA_cm2: float, T: float = T_REF) -> float:
        """
        Anodic overpotential for Fe²⁺/Fe³⁺ (V) at given current density.

        Uses Tafel: η = b · log10(j / j₀)
        """
        if j_mA_cm2 <= 0.0:
            return 0.0
        j_A_m2 = j_mA_cm2 * 10.0
        # Temperature-corrected exchange current (Arrhenius, Ea ~ 40 kJ/mol)
        Ea = 40000.0  # J/mol
        i0_T = self.i0 * np.exp(-Ea / R_GAS * (1.0 / T - 1.0 / T_REF))
        return float(self.tafel_V * np.log10(max(j_A_m2, 1e-30) / i0_T))


# ─── Divided Cell / Membrane Model ────────────────────────────────────

@dataclass
class MembraneModel:
    """
    Ion-exchange membrane for divided-cell operation.

    A cation-exchange membrane (e.g., Nafion) separates anolyte from
    catholyte, suppressing Fe³⁺ crossover and improving current efficiency
    at the cost of additional ohmic resistance.

    Parameters
    ----------
    R_membrane_ohm_m2 : float
        Area-specific membrane resistance (Ω·m²).
        Nafion 117 ≈ 0.001–0.003 Ω·m² in 1 M H₂SO₄ at 60 °C.
    fe3_crossover_rate : float
        Fraction of anolyte Fe³⁺ that crosses per hour (1/hr).
        0 = perfect rejection, 1 = fully permeable.
    cost_per_m2 : float
        Membrane cost ($/m²) for economic analysis.
    """

    R_membrane_ohm_m2: float = 0.002    # Ω·m²
    fe3_crossover_rate: float = 0.05    # 1/hr
    cost_per_m2: float = 500.0          # $/m²

    def IR_drop(self, j_mA_cm2: float) -> float:
        """Ohmic voltage drop across the membrane (V)."""
        j_A_m2 = j_mA_cm2 * 10.0
        return j_A_m2 * self.R_membrane_ohm_m2


# ─── Cell Voltage Decomposition ───────────────────────────────────────

@dataclass
class CellVoltageModel:
    """
    Decomposes total cell voltage into thermodynamic and kinetic components.

    Full decomposition:
        V_cell = E_cathode(Nernst) + E_anode(Nernst)
                 + η_cathode(BV) + η_anode
                 + IR_electrolyte + IR_membrane + IR_contacts

    Where:
        E_cathode = E°_Fe + (RT/2F)·ln(a_Fe2+)
        E_anode   = E°_OER + ...  or  E°_Fe3+/Fe2+ + ...
        η_cathode = Butler-Volmer / Tafel approximation for Fe + HER
        η_anode   = from AnodeKinetics model or fixed
        IR_electrolyte = j · L / (κ(T) · (1 − θ_bubble))
        IR_membrane    = j · R_membrane  (divided cell only)
        IR_contacts    = j · R_contacts  (fixed)

    When an ``anode`` model is supplied, the anode overpotential and
    equilibrium potential are computed from first principles
    (:class:`.anode.AnodeKinetics`) rather than using the fixed
    ``E_anode_eq`` / ``eta_anode`` defaults.

    Parameters
    ----------
    E_cathode_eq : float
        Standard-state cathode potential (V vs. SHE). The Nernst activity
        correction is applied separately. Default is E°(Fe²⁺/Fe).
    E_anode_eq : float
        Equilibrium anode potential (V vs. SHE). Default is E°(OER).
        Ignored when ``anode`` is supplied (overridden by AnodeKinetics).
    eta_cathode : float
        Cathode overpotential at operating current density (V). Positive value.
    eta_anode : float
        Anode overpotential at operating current density (V). Positive value.
        Ignored when ``anode`` is supplied.
    ir_drop : float
        Ohmic drop across electrolyte, membrane, contacts (V).
        Used as fallback when detailed IR decomposition is not configured.
    temperature_C : float
        Operating temperature (°C). Default 60.
    fe2_conc_M : float
        Bulk Fe²⁺ concentration (mol/L). Default 1.0.
    electrolyte_conductivity_S_m : float
        Electrolyte conductivity (S/m). By default this is a 25 °C reference;
        set ``electrolyte_conductivity_at_temperature=True`` when it has
        already been evaluated at the operating temperature.
    interelectrode_gap_m : float
        Distance between electrodes (m). Default 0.02 (2 cm).
    contact_resistance_ohm_m2 : float
        Area-specific contact/busbar resistance (Ω·m²). Default 5e-4.
    bubble_fraction : float
        Gas void fraction at the anode surface (0–1). Default 0.10.
    divided_cell : bool
        If True, apply membrane model.
    membrane : MembraneModel or None
        Membrane properties. Auto-created when divided_cell=True.
    fe_shuttle : FeShuttleAnode or None
        Fe²⁺/Fe³⁺ anode shuttle model. None = use OER.
    anode : AnodeKinetics, optional
        First-principles anode model. When supplied, overrides fe_shuttle.
    j_operating_mA_cm2 : float
        Current density used to evaluate the anode model (mA/cm²).
    """

    # Thermodynamic equilibrium potentials
    E_cathode_eq: float = E0_FE
    E_anode_eq: float = E0_OER
    # Fixed overpotentials (fallback when no kinetic model)
    eta_cathode: float = 0.30
    eta_anode: float = 0.40
    ir_drop: float = 0.20
    # Temperature and composition
    temperature_C: float = 60.0
    fe2_conc_M: float = 1.0
    # Electrolyte geometry / conductivity
    electrolyte_conductivity_S_m: float = 10.0
    interelectrode_gap_m: float = 0.02
    contact_resistance_ohm_m2: float = 5.0e-4
    bubble_fraction: float = 0.10
    # Divided cell
    divided_cell: bool = False
    membrane: Optional[MembraneModel] = field(default=None, repr=False)
    # Anode reaction
    fe_shuttle: Optional[FeShuttleAnode] = field(default=None, repr=False)
    anode: Optional["AnodeKinetics"] = field(default=None, repr=False)
    j_operating_mA_cm2: float = 100.0
    # ``False`` means the value is a 25 °C reference and this class applies
    # its temperature correlation.  Coupled speciation solvers commonly
    # return conductivity already evaluated at the operating temperature;
    # they must set this flag to avoid a second temperature correction.
    electrolyte_conductivity_at_temperature: bool = False

    def __post_init__(self):
        if self.divided_cell and self.membrane is None:
            self.membrane = MembraneModel()

    @property
    def T(self) -> float:
        """Operating temperature (K)."""
        return self.temperature_C + 273.15

    # ─── Nernst equilibrium potentials ─────────────────────────────────

    @property
    def E_cathode_nernst(self) -> float:
        """
        Cathode equilibrium potential via Nernst equation (V vs. SHE).

        Fe²⁺ + 2e⁻ → Fe
        E = E° + (RT/2F) · ln(a_Fe2+)
        """
        a_fe2 = max(self.fe2_conc_M, 1e-10)
        # ``E_cathode_eq`` is the standard-state potential.  Earlier code
        # silently ignored this field and always used the module constant,
        # making alternate/reference-state thermodynamics impossible.
        return self.E_cathode_eq + (R_GAS * self.T / (Z_FE * FARADAY)) * np.log(a_fe2)

    @property
    def E_anode_nernst(self) -> float:
        """
        Anode equilibrium potential (V vs. SHE).

        Uses the anode model if supplied, fe_shuttle if configured,
        or falls back to fixed E_anode_eq (OER).
        """
        if self.anode is not None:
            if getattr(self.anode, "is_soluble", False):
                return self.anode.fe_dissolution_equilibrium()
            return self.anode.oer_equilibrium()
        if self.fe_shuttle is not None:
            return self.fe_shuttle.equilibrium(self.T)
        # OER: approximate with fixed E_anode_eq
        return self.E_anode_eq

    # ─── Kinetic overpotentials ────────────────────────────────────────

    @property
    def _effective_anode_eq(self) -> float:
        """Anode equilibrium potential (V vs. SHE)."""
        return self.E_anode_nernst

    @property
    def _effective_eta_anode(self) -> float:
        """Anode overpotential (V) from model or fixed value."""
        if self.anode is not None:
            return self.anode.eta_anode(self.j_operating_mA_cm2)
        if self.fe_shuttle is not None:
            return self.fe_shuttle.overpotential(self.j_operating_mA_cm2, self.T)
        return self.eta_anode

    # ─── Ohmic drops ───────────────────────────────────────────────────

    @property
    def IR_electrolyte(self) -> float:
        """
        Ohmic drop across the electrolyte (V).

        IR = j · L / (κ(T) · (1 − θ_bubble))
        """
        j_A_m2 = self.j_operating_mA_cm2 * 10.0  # mA/cm² → A/m²
        if self.electrolyte_conductivity_at_temperature:
            kappa_T = self.electrolyte_conductivity_S_m
        else:
            kappa_T = conductivity_S_m(
                self.T, kappa_ref=self.electrolyte_conductivity_S_m
            )
        # A soluble Fe anode does not evolve gas; do not apply an OER bubble
        # penalty when the first-principles anode object says otherwise.
        bubble_fraction = self.bubble_fraction
        if self.anode is not None and getattr(self.anode, "is_soluble", False):
            bubble_fraction = 0.0
        # Effective conductivity reduced by bubble coverage
        kappa_eff = kappa_T * max(1.0 - bubble_fraction, 0.01)
        return j_A_m2 * self.interelectrode_gap_m / kappa_eff

    @property
    def IR_membrane(self) -> float:
        """Ohmic drop across the membrane (V). Only non-zero in divided cell."""
        if self.divided_cell and self.membrane is not None:
            return self.membrane.IR_drop(self.j_operating_mA_cm2)
        return 0.0

    @property
    def IR_contacts(self) -> float:
        """Ohmic drop at contacts/busbars (V)."""
        j_A_m2 = self.j_operating_mA_cm2 * 10.0
        return j_A_m2 * self.contact_resistance_ohm_m2

    @property
    def _total_ir_drop(self) -> float:
        """Total ohmic drop: electrolyte + membrane + contacts."""
        if self.ir_drop != 0.20:
            # Legacy mode: use fixed ir_drop if user explicitly set it
            return self.ir_drop
        return self.IR_electrolyte + self.IR_membrane + self.IR_contacts

    # ─── Cell voltage components ───────────────────────────────────────

    @property
    def E_thermodynamic(self) -> float:
        """Minimum thermodynamic cell voltage (V)."""
        return abs(self._effective_anode_eq - self.E_cathode_nernst)

    @property
    def V_cell(self) -> float:
        """Total cell voltage (V) including all overpotentials."""
        return (
            self.E_thermodynamic
            + self.eta_cathode
            + self._effective_eta_anode
            + self._total_ir_drop
        )

    @property
    def V_decomposition(self) -> dict:
        """
        Full cell voltage decomposition as a dictionary.

        Returns dict with: E_cathode, E_anode, E_thermodynamic,
        eta_cathode, eta_anode, IR_electrolyte, IR_membrane, IR_contacts,
        IR_total, V_cell.
        """
        return {
            "E_cathode (V)": round(self.E_cathode_nernst, 4),
            "E_anode (V)": round(self._effective_anode_eq, 4),
            "E_thermodynamic (V)": round(self.E_thermodynamic, 4),
            "η_cathode (V)": round(self.eta_cathode, 4),
            "η_anode (V)": round(self._effective_eta_anode, 4),
            "IR_electrolyte (V)": round(self.IR_electrolyte, 4),
            "IR_membrane (V)": round(self.IR_membrane, 4),
            "IR_contacts (V)": round(self.IR_contacts, 4),
            "IR_total (V)": round(self._total_ir_drop, 4),
            "V_cell (V)": round(self.V_cell, 4),
        }

    def summary(self) -> dict:
        return {
            "E_thermodynamic (V)": round(self.E_thermodynamic, 3),
            "η_cathode (V)": round(self.eta_cathode, 3),
            "η_anode (V)": round(self._effective_eta_anode, 3),
            "iR drop (V)": round(self._total_ir_drop, 3),
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
